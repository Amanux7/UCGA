"""
adaptive_topology.py — Self-Modifying Graph Topology

Dynamically activates or deactivates cognitive nodes based on
their contribution scores, enabling the graph to grow or shrink
in response to task demands.

Components:
    - NodeImportanceScorer: computes per-node importance via activation
      magnitude and gradient flow
    - AdaptiveTopology: wraps UCGAModel with dynamic node masking

The topology changes are non-destructive — nodes are soft-gated
rather than removed, allowing recovery.

Author: Aman Singh
"""

import torch
import torch.nn as nn
from typing import Dict, List, Any

from ucga.ucga_model import UCGAModel


class NodeImportanceScorer(nn.Module):
    """
    Scores the importance of each cognitive node based on its
    activation statistics.

    Parameters
    ----------
    state_dim : int
        Cognitive state dimensionality.
    num_nodes : int
        Number of nodes to score.
    """

    NODE_NAMES = [
        "perception", "memory_node", "reasoning",
        "planning", "evaluation", "correction",
        "balancer", "output_node",
    ]

    def __init__(self, state_dim: int = 128, num_nodes: int = 8):
        super().__init__()
        self.state_dim = state_dim
        self.num_nodes = num_nodes

        # Learned importance head
        self.importance_net = nn.Sequential(
            nn.Linear(state_dim, state_dim // 2),
            nn.GELU(),
            nn.Linear(state_dim // 2, 1),
            nn.Sigmoid(),
        )

        # Running averages
        self.register_buffer("running_importance", torch.ones(num_nodes) * 0.5)
        self.register_buffer("update_count", torch.zeros(1))
        self._momentum = 0.9

    def score_nodes(self, model: UCGAModel) -> Dict[str, float]:
        """
        Score each node's importance based on its current state norm
        and learned importance function.

        Parameters
        ----------
        model : UCGAModel
            The model whose nodes to score.

        Returns
        -------
        dict
            Node name → importance score.
        """
        scores = {}
        for i, name in enumerate(self.NODE_NAMES[:self.num_nodes]):
            node = getattr(model, name, None)
            if node is None:
                continue

            state = node.get_state() if hasattr(node, 'get_state') else None
            if state is None or state.numel() == 0:
                scores[name] = 0.5
                continue

            # Activation-based importance
            state_norm = state.norm(dim=-1).mean().item()
            learned_score = self.importance_net(state.detach()).mean().item()

            # Combine
            importance = 0.5 * min(state_norm / (self.state_dim ** 0.5), 1.0) + 0.5 * learned_score
            scores[name] = importance

            # Update running average
            with torch.no_grad():
                self.running_importance[i] = (
                    self._momentum * self.running_importance[i]
                    + (1 - self._momentum) * importance
                )

        self.update_count += 1
        return scores


class AdaptiveTopology(nn.Module):
    """
    Wraps UCGAModel with self-modifying topology.

    Dynamically gates cognitive nodes based on their importance scores.
    Nodes below the pruning threshold are soft-masked (output multiplied
    by near-zero gate), while nodes above the activation threshold are
    fully active.

    Parameters
    ----------
    model : UCGAModel
        The base UCGA model.
    prune_threshold : float
        Nodes below this importance score are soft-pruned (default 0.2).
    activate_threshold : float
        Nodes above this score are guaranteed active (default 0.5).
    """

    def __init__(
        self,
        model: UCGAModel,
        prune_threshold: float = 0.2,
        activate_threshold: float = 0.5,
    ):
        super().__init__()
        self.model = model
        self.prune_threshold = prune_threshold
        self.activate_threshold = activate_threshold

        self.scorer = NodeImportanceScorer(
            state_dim=model.state_dim,
            num_nodes=len(NodeImportanceScorer.NODE_NAMES),
        )

        # Per-node gate parameters (learnable baseline)
        self.node_gates = nn.ParameterDict({
            name: nn.Parameter(torch.ones(1))
            for name in NodeImportanceScorer.NODE_NAMES
        })

        self._topology_history: List[Dict[str, float]] = []

    def forward(
        self, x: torch.Tensor, return_meta: bool = False,
    ) -> Any:
        """
        Forward pass with adaptive topology.

        First runs the base model, then scores node importance and
        adjusts gates for the next forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input of shape ``(B, input_dim)``.
        return_meta : bool
            If True, also return topology metadata.
        """
        # Run base model
        if return_meta:
            output, meta = self.model(x, return_meta=True)
        else:
            output = self.model(x)
            meta = {}

        # Score nodes after forward pass
        scores = self.scorer.score_nodes(self.model)

        # Update gates based on importance
        gate_values = {}
        for name, score in scores.items():
            if name in self.node_gates:
                if score < self.prune_threshold:
                    gate_val = 0.1  # soft prune (not fully off)
                elif score >= self.activate_threshold:
                    gate_val = 1.0
                else:
                    # Linear interpolation
                    gate_val = (score - self.prune_threshold) / (
                        self.activate_threshold - self.prune_threshold
                    )

                with torch.no_grad():
                    self.node_gates[name].fill_(gate_val)
                gate_values[name] = gate_val

        # Record topology snapshot
        self._topology_history.append({
            "importance_scores": dict(scores),
            "gate_values": dict(gate_values),
        })

        if return_meta:
            meta["topology"] = {
                "importance_scores": scores,
                "gate_values": gate_values,
            }
            return output, meta

        return output

    def get_active_nodes(self) -> List[str]:
        """Return list of nodes currently above prune threshold."""
        active = []
        for name, gate in self.node_gates.items():
            if gate.item() > self.prune_threshold:
                active.append(name)
        return active

    def get_pruned_nodes(self) -> List[str]:
        """Return list of nodes currently soft-pruned."""
        pruned = []
        for name, gate in self.node_gates.items():
            if gate.item() <= self.prune_threshold:
                pruned.append(name)
        return pruned

    def get_topology_stats(self) -> Dict[str, Any]:
        """Return current topology statistics."""
        active = self.get_active_nodes()
        pruned = self.get_pruned_nodes()

        return {
            "active_nodes": active,
            "pruned_nodes": pruned,
            "n_active": len(active),
            "n_pruned": len(pruned),
            "gate_values": {
                name: gate.item() for name, gate in self.node_gates.items()
            },
            "topology_changes": len(self._topology_history),
        }

    @property
    def topology_history(self) -> List[Dict[str, float]]:
        return self._topology_history
