"""
UCGAModel — Main orchestrator for the Unified Cognitive Graph Architecture.

Wires together all cognitive nodes into a directed graph, runs the
recursive cognitive refinement loop for *T* timesteps, and manages
persistent memory read / write operations.

The cognitive loop at each timestep:
    1. Perceive input
    2. Retrieve memory
    3. Reason (with iterative refinement)
    4. Plan
    5. Evaluate
    6. Correct (if confidence < threshold)
    7. Balance streams
    8. Produce output
    9. Update persistent memory

Architecture:  G = (V, E, W, S, M)

Author: Aman Singh
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, Any

from .nodes import (
    PerceptionNode,
    MemoryNode,
    ReasoningNode,
    PlanningNode,
    EvaluationNode,
    CorrectionNode,
    BalancerNode,
    OutputNode,
)
from .memory import PersistentMemory


class UCGAModel(nn.Module):
    """
    Unified Cognitive Graph Architecture.

    Parameters
    ----------
    input_dim : int
        Dimensionality of the raw input (after encoder).
    state_dim : int
        Internal cognitive state dimensionality.
    output_dim : int
        Task output dimensionality.
    memory_slots : int
        Number of persistent memory slots.
    cognitive_steps : int
        Number of outer cognitive-loop iterations (T).
    reasoning_steps : int
        Number of inner reasoning refinement steps (K).
    correction_threshold : float
        Confidence below which the correction node is invoked.
    """

    def __init__(
        self,
        input_dim: int = 256,
        state_dim: int = 256,
        output_dim: int = 64,
        memory_slots: int = 128,
        cognitive_steps: int = 3,
        reasoning_steps: int = 3,
        correction_threshold: float = 0.5,
    ):
        super().__init__()
        self.state_dim = state_dim
        self.cognitive_steps = cognitive_steps
        self.correction_threshold = correction_threshold

        # ---- Cognitive Nodes ----
        self.perception = PerceptionNode(raw_input_dim=input_dim, state_dim=state_dim)
        self.memory_node = MemoryNode(state_dim=state_dim)
        self.reasoning = ReasoningNode(state_dim=state_dim, reasoning_steps=reasoning_steps)
        self.planning = PlanningNode(state_dim=state_dim)
        self.evaluation = EvaluationNode(state_dim=state_dim)
        self.correction = CorrectionNode(state_dim=state_dim)
        self.balancer = BalancerNode(state_dim=state_dim, num_streams=3)
        self.output_node = OutputNode(state_dim=state_dim, output_dim=output_dim)

        # ---- Persistent Memory ----
        self.persistent_memory = PersistentMemory(
            num_slots=memory_slots, slot_dim=state_dim
        )

    # ------------------------------------------------------------------
    # Forward — run the full cognitive loop
    # ------------------------------------------------------------------
    def forward(
        self,
        x: torch.Tensor,
        return_meta: bool = False,
    ) -> torch.Tensor:
        """
        Run the UCGA cognitive loop.

        Parameters
        ----------
        x : torch.Tensor
            Encoded input of shape ``(B, input_dim)``.
        return_meta : bool
            If ``True``, return a dict with intermediate states.

        Returns
        -------
        torch.Tensor or (torch.Tensor, dict)
            Task output of shape ``(B, output_dim)`` and, optionally,
            a metadata dict with intermediate cognitive states.
        """
        B = x.size(0)
        meta: Dict[str, Any] = {"confidences": [], "corrections": 0}

        # Reset node states
        self._reset_all(B)

        # Retrieve memory bank
        memory_bank = self.persistent_memory.get_memory_bank(B)

        for t in range(self.cognitive_steps):
            # 1. Perceive
            percept = self.perception([x])

            # 2. Memory retrieval
            mem_state = self.memory_node([percept], memory_bank=memory_bank)

            # 3. Reasoning
            reason_state = self.reasoning([percept, mem_state])

            # 4. Planning
            plan_state = self.planning([reason_state])

            # 5. Evaluation
            eval_state = self.evaluation([plan_state, reason_state])
            confidence = self.evaluation.get_confidence()
            meta["confidences"].append(confidence.mean().item())

            # 6. Correction (conditional)
            if confidence.mean().item() < self.correction_threshold:
                corrected = self.correction([plan_state, eval_state])
                meta["corrections"] += 1
            else:
                corrected = plan_state

            # 7. Balance
            balanced = self.balancer([reason_state, corrected, mem_state])

            # Update x for next cognitive step (recurrent refinement)
            x = balanced

        # 8. Output
        output = self.output_node([balanced])

        # 9. Write to persistent memory
        self.persistent_memory.write(balanced.detach())

        if return_meta:
            return output, meta
        return output

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _reset_all(self, batch_size: int) -> None:
        """Reset the internal state of every cognitive node."""
        for module in [
            self.perception,
            self.memory_node,
            self.reasoning,
            self.planning,
            self.evaluation,
            self.correction,
            self.balancer,
            self.output_node,
        ]:
            module.reset_state(batch_size)

    def reset_memory(self) -> None:
        """Clear the persistent memory bank."""
        self.persistent_memory.reset()

    def count_parameters(self) -> int:
        """Return total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def __repr__(self) -> str:
        return (
            f"UCGAModel(\n"
            f"  state_dim={self.state_dim},\n"
            f"  cognitive_steps={self.cognitive_steps},\n"
            f"  params={self.count_parameters():,}\n"
            f")"
        )
