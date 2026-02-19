"""
CognitiveAgent — End-to-end UCGA agent that:

    1. Perceives input
    2. Retrieves memory
    3. Reasons
    4. Plans
    5. Produces output
    6. Updates persistent memory

Can be used for interactive inference or batch evaluation.

Usage:
    python agents/cognitive_agent.py

Author: Aman Singh
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import torch.nn as nn
from typing import List, Dict, Any, Optional

from ucga.ucga_model import UCGAModel
from ucga.encoders import VectorEncoder, TextEncoder, ImageEncoder


class CognitiveAgent:
    """
    High-level cognitive agent wrapping the UCGA model.

    Parameters
    ----------
    modality : str
        Input modality: ``"vector"``, ``"text"``, or ``"image"``.
    input_dim : int
        Raw input dimensionality (vector dim, or vocab size for text).
    state_dim : int
        Cognitive state dimensionality.
    output_dim : int
        Task output dimensionality.
    device : str
        Compute device.
    """

    def __init__(
        self,
        modality: str = "vector",
        input_dim: int = 64,
        state_dim: int = 128,
        output_dim: int = 32,
        device: str = "cpu",
        **model_kwargs,
    ):
        self.device = device
        self.modality = modality
        self.history: List[Dict[str, Any]] = []

        # ---- Encoder ----
        if modality == "vector":
            self.encoder = VectorEncoder(input_dim=input_dim, output_dim=state_dim).to(device)
        elif modality == "text":
            self.encoder = TextEncoder(
                vocab_size=input_dim, embed_dim=64, output_dim=state_dim
            ).to(device)
        elif modality == "image":
            self.encoder = ImageEncoder(in_channels=input_dim, output_dim=state_dim).to(device)
        else:
            raise ValueError(f"Unknown modality: {modality}")

        # ---- UCGA Model ----
        self.model = UCGAModel(
            input_dim=state_dim,
            state_dim=state_dim,
            output_dim=output_dim,
            **model_kwargs,
        ).to(device)

    @torch.no_grad()
    def perceive_and_act(self, raw_input: torch.Tensor) -> Dict[str, Any]:
        """
        Full cognitive cycle: perceive → reason → act → remember.

        Parameters
        ----------
        raw_input : torch.Tensor
            Raw task input appropriate for the chosen modality.

        Returns
        -------
        dict
            ``output``: the agent's action / prediction tensor.
            ``meta``: cognitive loop metadata (confidences, corrections).
        """
        self.model.eval()
        self.encoder.eval()

        raw_input = raw_input.to(self.device)
        encoded = self.encoder(raw_input)
        output, meta = self.model(encoded, return_meta=True)

        # Record episode in history
        self.history.append({
            "input_norm": raw_input.norm().item(),
            "output_norm": output.norm().item(),
            "confidence": meta["confidences"][-1] if meta["confidences"] else None,
            "corrections": meta["corrections"],
        })

        return {"output": output, "meta": meta}

    def reset(self) -> None:
        """Reset memory and episode history."""
        self.model.reset_memory()
        self.history.clear()

    def get_history(self) -> List[Dict[str, Any]]:
        """Return the agent's episode history."""
        return self.history

    def summary(self) -> str:
        """Human-readable agent summary."""
        lines = [
            f"CognitiveAgent(modality={self.modality})",
            f"  Model params : {self.model.count_parameters():,}",
            f"  Episodes     : {len(self.history)}",
            f"  Device       : {self.device}",
        ]
        if self.history:
            last = self.history[-1]
            lines.append(f"  Last confidence: {last['confidence']:.3f}")
            lines.append(f"  Last corrections: {last['corrections']}")
        return "\n".join(lines)


# ======================================================================
# Demo
# ======================================================================
def main():
    print("=" * 60)
    print("  UCGA Cognitive Agent — Demo")
    print("=" * 60)

    agent = CognitiveAgent(
        modality="vector",
        input_dim=64,
        state_dim=128,
        output_dim=32,
        cognitive_steps=3,
        reasoning_steps=3,
        memory_slots=64,
    )

    print(f"\n{agent.summary()}\n")

    # Simulate 5 episodes
    for episode in range(1, 6):
        x = torch.randn(1, 64)
        result = agent.perceive_and_act(x)
        out = result["output"]
        meta = result["meta"]
        print(
            f"Episode {episode}  |  "
            f"Output norm: {out.norm().item():.4f}  |  "
            f"Confidence: {meta['confidences'][-1]:.3f}  |  "
            f"Corrections: {meta['corrections']}"
        )

    print(f"\n{agent.summary()}")
    print("\n✓ Agent demo complete.")


if __name__ == "__main__":
    main()
