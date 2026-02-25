"""
distributed_graph.py — Distributed Cognitive Graph

Partitions the UCGA cognitive graph across multiple workers, enabling
multi-device scaling through message-passing between graph partitions.

Components:
    - MessageBus: handles tensor exchange between partitions
    - GraphPartition: wraps a subset of cognitive nodes
    - DistributedCognitiveGraph: orchestrates N partitions

Architecture:
    The cognitive graph G = (V, E) is split into P partitions.
    Each partition runs its nodes locally and exchanges boundary
    activations with neighboring partitions through the MessageBus.

    For production deployment, the MessageBus can be backed by
    ``torch.distributed`` — here it uses in-process simulation.

Author: Aman Singh
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Any
from collections import defaultdict

from ucga.nodes.cognitive_node import CognitiveNode


class MessageBus:
    """
    In-process message bus for inter-partition communication.

    Stores tensors keyed by (source_partition, target_partition).
    In a real distributed setup this would wrap ``torch.distributed``
    send/recv or NCCL collectives.
    """

    def __init__(self):
        self._mailbox: Dict[Tuple[int, int], torch.Tensor] = {}

    def send(self, src: int, dst: int, tensor: torch.Tensor) -> None:
        """Post a tensor from partition *src* to partition *dst*."""
        self._mailbox[(src, dst)] = tensor

    def recv(self, src: int, dst: int) -> Optional[torch.Tensor]:
        """Retrieve a tensor sent from *src* to *dst*, or ``None``."""
        return self._mailbox.get((src, dst))

    def clear(self) -> None:
        self._mailbox.clear()

    def all_messages(self) -> Dict[Tuple[int, int], torch.Tensor]:
        """Return all currently buffered messages."""
        return dict(self._mailbox)


class GraphPartition(nn.Module):
    """
    A partition of the cognitive graph, containing a subset of nodes.

    Parameters
    ----------
    partition_id : int
        Unique partition identifier.
    nodes : Dict[str, CognitiveNode]
        Named cognitive nodes belonging to this partition.
    execution_order : List[str]
        Order in which to execute nodes within this partition.
    state_dim : int
        Cognitive state dimensionality.
    """

    def __init__(
        self,
        partition_id: int,
        nodes: Dict[str, nn.Module],
        execution_order: List[str],
        state_dim: int,
    ):
        super().__init__()
        self.partition_id = partition_id
        self.execution_order = execution_order
        self.state_dim = state_dim

        # Register nodes as sub-modules
        self._nodes = nn.ModuleDict(nodes)

        # Boundary projection: maps incoming cross-partition messages
        self.boundary_proj = nn.Sequential(
            nn.Linear(state_dim, state_dim),
            nn.LayerNorm(state_dim),
            nn.GELU(),
        )

    def forward(
        self,
        inputs: Dict[str, torch.Tensor],
        boundary_inputs: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Execute all nodes in order, optionally fusing boundary messages.

        Parameters
        ----------
        inputs : dict
            Named inputs for each node (from previous partition or encoder).
        boundary_inputs : torch.Tensor, optional
            Aggregated messages from other partitions.

        Returns
        -------
        dict
            Outputs keyed by node name.
        """
        outputs = dict(inputs)

        # Incorporate boundary messages
        if boundary_inputs is not None:
            boundary_signal = self.boundary_proj(boundary_inputs)
            outputs["_boundary"] = boundary_signal

        for node_name in self.execution_order:
            node = self._nodes[node_name]
            # Gather available inputs for this node
            node_inputs = [
                outputs[k]
                for k in outputs
                if isinstance(outputs[k], torch.Tensor) and outputs[k].size(-1) == self.state_dim
            ]
            if not node_inputs:
                continue
            outputs[node_name] = node(node_inputs)

        return outputs


class DistributedCognitiveGraph(nn.Module):
    """
    Orchestrates multiple graph partitions with inter-partition
    message passing.

    Parameters
    ----------
    state_dim : int
        Cognitive state dimensionality.
    num_partitions : int
        Number of graph partitions to create.
    nodes_per_partition : int
        Number of generic cognitive nodes per partition.
    communication_rounds : int
        Number of message-passing rounds between partitions.
    """

    def __init__(
        self,
        state_dim: int = 128,
        num_partitions: int = 3,
        nodes_per_partition: int = 2,
        communication_rounds: int = 2,
    ):
        super().__init__()
        self.state_dim = state_dim
        self.num_partitions = num_partitions
        self.communication_rounds = communication_rounds

        # Build partitions
        self.partitions = nn.ModuleList()
        for p_id in range(num_partitions):
            nodes = {}
            order = []
            for n_idx in range(nodes_per_partition):
                name = f"node_{p_id}_{n_idx}"
                nodes[name] = CognitiveNode(
                    input_dim=state_dim, state_dim=state_dim, name=name,
                )
                order.append(name)

            self.partitions.append(GraphPartition(
                partition_id=p_id,
                nodes=nodes,
                execution_order=order,
                state_dim=state_dim,
            ))

        # Cross-partition aggregation
        self.cross_partition_attn = nn.MultiheadAttention(
            embed_dim=state_dim, num_heads=4, batch_first=True,
        )
        self.output_proj = nn.Linear(state_dim * num_partitions, state_dim)

        self.message_bus = MessageBus()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Run the distributed cognitive graph.

        Parameters
        ----------
        x : torch.Tensor
            Input of shape ``(B, state_dim)``.

        Returns
        -------
        torch.Tensor
            Combined output of shape ``(B, state_dim)``.
        """
        B = x.size(0)
        self.message_bus.clear()

        # Initialize each partition with the same input
        partition_outputs: List[torch.Tensor] = [x.clone() for _ in range(self.num_partitions)]

        for round_idx in range(self.communication_rounds):
            new_outputs = []

            for p_id, partition in enumerate(self.partitions):
                # Gather boundary messages from other partitions
                boundary_tensors = []
                for other_id in range(self.num_partitions):
                    if other_id != p_id:
                        boundary_tensors.append(partition_outputs[other_id])

                boundary = torch.stack(boundary_tensors, dim=0).mean(dim=0) if boundary_tensors else None

                # Reset nodes for this round
                for node in partition._nodes.values():
                    if hasattr(node, 'reset_state'):
                        node.reset_state(B)

                # Execute partition
                result = partition(
                    inputs={"input": partition_outputs[p_id]},
                    boundary_inputs=boundary,
                )

                # Get the last node's output
                last_node = partition.execution_order[-1]
                p_out = result.get(last_node, partition_outputs[p_id])
                new_outputs.append(p_out)

                # Post messages to other partitions
                for other_id in range(self.num_partitions):
                    if other_id != p_id:
                        self.message_bus.send(p_id, other_id, p_out.detach())

            partition_outputs = new_outputs

        # Combine partition outputs
        stacked = torch.stack(partition_outputs, dim=1)  # (B, P, D)
        attended, _ = self.cross_partition_attn(stacked, stacked, stacked)  # (B, P, D)

        combined = attended.reshape(B, -1)  # (B, P*D)
        output = self.output_proj(combined)  # (B, D)

        return output

    def get_partition_states(self) -> List[Dict[str, torch.Tensor]]:
        """Return the current state of each partition's nodes."""
        states = []
        for partition in self.partitions:
            partition_state = {}
            for name, node in partition._nodes.items():
                if hasattr(node, 'get_state'):
                    partition_state[name] = node.get_state()
            states.append(partition_state)
        return states
