"""
ablation_models.py — UCGA ablation variants for systematic component analysis.

Creates model variants with specific components disabled / modified:
  - No correction loop
  - No persistent memory
  - Random-write memory
  - Variable cognitive steps (T=1, T=3, T=5)
  - Tanh baseline (original activation, no LayerNorm)

Each variant is produced by a factory function returning (UCGAModel, encoder).

Author: Dr. Elena Voss / Aman Singh
"""

import sys
import os
import copy
import torch
import torch.nn as nn
from typing import Tuple, Dict, Callable

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from ucga.ucga_model import UCGAModel
from ucga.encoders import ImageEncoder, VectorEncoder
from ucga.nodes.cognitive_node import CognitiveNode


# ======================================================================
# Variant: No Correction Loop
# ======================================================================
class UCGANoCorrection(UCGAModel):
    """UCGA with the correction node always bypassed."""

    def forward(self, x, return_meta=False):
        B = x.size(0)
        meta = {"confidences": [], "corrections": 0}
        self._reset_all(B)
        x_original = x
        memory_bank = self.persistent_memory.get_memory_bank(B)

        for t in range(self.cognitive_steps):
            percept = self.perception([x])
            mem_state = self.memory_node([percept], memory_bank=memory_bank)
            reason_state = self.reasoning([percept, mem_state])
            plan_state = self.planning([reason_state])
            eval_state = self.evaluation([plan_state, reason_state])
            confidence = self.evaluation.get_confidence()
            meta["confidences"].append(confidence.mean().item())
            # Always skip correction
            corrected = plan_state
            balanced = self.balancer([reason_state, corrected, mem_state])
            x = balanced + x_original

        output = self.output_node([balanced])
        self.persistent_memory.write(balanced.detach())

        if return_meta:
            return output, meta
        return output


# ======================================================================
# Variant: No Persistent Memory
# ======================================================================
class UCGANoMemory(UCGAModel):
    """UCGA with persistent memory replaced by a zero-tensor stub."""

    def forward(self, x, return_meta=False):
        B = x.size(0)
        meta = {"confidences": [], "corrections": 0}
        self._reset_all(B)
        x_original = x

        # Zero memory bank — no retrieval
        memory_bank = torch.zeros(
            B, self.persistent_memory.num_slots, self.state_dim,
            device=x.device,
        )

        for t in range(self.cognitive_steps):
            percept = self.perception([x])
            mem_state = self.memory_node([percept], memory_bank=memory_bank)
            reason_state = self.reasoning([percept, mem_state])
            plan_state = self.planning([reason_state])
            eval_state = self.evaluation([plan_state, reason_state])
            confidence = self.evaluation.get_confidence()
            meta["confidences"].append(confidence.mean().item())

            if confidence.mean().item() < self.correction_threshold:
                corrected = self.correction([plan_state, eval_state])
                meta["corrections"] += 1
            else:
                corrected = plan_state

            balanced = self.balancer([reason_state, corrected, mem_state])
            x = balanced + x_original

        output = self.output_node([balanced])
        # No memory write

        if return_meta:
            return output, meta
        return output


# ======================================================================
# Variant: Random-Write Memory
# ======================================================================
class UCGARandomMemory(UCGAModel):
    """UCGA that writes random vectors to memory instead of balanced state."""

    def forward(self, x, return_meta=False):
        B = x.size(0)
        meta = {"confidences": [], "corrections": 0}
        self._reset_all(B)
        x_original = x
        memory_bank = self.persistent_memory.get_memory_bank(B)

        for t in range(self.cognitive_steps):
            percept = self.perception([x])
            mem_state = self.memory_node([percept], memory_bank=memory_bank)
            reason_state = self.reasoning([percept, mem_state])
            plan_state = self.planning([reason_state])
            eval_state = self.evaluation([plan_state, reason_state])
            confidence = self.evaluation.get_confidence()
            meta["confidences"].append(confidence.mean().item())

            if confidence.mean().item() < self.correction_threshold:
                corrected = self.correction([plan_state, eval_state])
                meta["corrections"] += 1
            else:
                corrected = plan_state

            balanced = self.balancer([reason_state, corrected, mem_state])
            x = balanced + x_original

        output = self.output_node([balanced])
        # Write random vectors
        self.persistent_memory.write(torch.randn_like(balanced))

        if return_meta:
            return output, meta
        return output


# ======================================================================
# Variant: Tanh activation (no LayerNorm) — original UCGA v0
# ======================================================================
def _apply_tanh_activation(model: nn.Module):
    """Replace all GELU+LayerNorm with tanh in CognitiveNode subclasses."""
    for module in model.modules():
        if isinstance(module, CognitiveNode):
            module.activation = nn.Tanh()
            module.norm = nn.Identity()
    return model


# ======================================================================
# Factory
# ======================================================================
def create_ablation_variants(
    mode: str = "cifar10",
    state_dim: int = 128,
    memory_slots: int = 64,
) -> Dict[str, Callable]:
    """
    Return a dict of variant_name → factory_fn.

    Each factory_fn() returns a nn.Module (unified model) ready for training.

    Parameters
    ----------
    mode : str
        'cifar10' for image classification, 'agnews' for text classification.
    state_dim : int
        Cognitive state dimensionality.
    memory_slots : int
        Number of persistent memory slots.
    """
    num_classes = 10 if mode == "cifar10" else 4

    def _make_encoder():
        if mode == "cifar10":
            return ImageEncoder(in_channels=3, output_dim=state_dim)
        else:
            return VectorEncoder(input_dim=8000, output_dim=state_dim)

    class _Wrapper(nn.Module):
        """Bundles encoder + UCGA variant into a single model."""
        def __init__(self, encoder, ucga):
            super().__init__()
            self.encoder = encoder
            self.ucga = ucga

        def forward(self, x, return_meta=False):
            encoded = self.encoder(x)
            return self.ucga(encoded, return_meta=return_meta)

        def count_parameters(self):
            return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def _full_ucga():
        enc = _make_encoder()
        m = UCGAModel(
            input_dim=state_dim, state_dim=state_dim, output_dim=num_classes,
            memory_slots=memory_slots, cognitive_steps=3, reasoning_steps=2,
        )
        return _Wrapper(enc, m)

    def _no_correction():
        enc = _make_encoder()
        m = UCGANoCorrection(
            input_dim=state_dim, state_dim=state_dim, output_dim=num_classes,
            memory_slots=memory_slots, cognitive_steps=3, reasoning_steps=2,
        )
        return _Wrapper(enc, m)

    def _no_memory():
        enc = _make_encoder()
        m = UCGANoMemory(
            input_dim=state_dim, state_dim=state_dim, output_dim=num_classes,
            memory_slots=memory_slots, cognitive_steps=3, reasoning_steps=2,
        )
        return _Wrapper(enc, m)

    def _random_memory():
        enc = _make_encoder()
        m = UCGARandomMemory(
            input_dim=state_dim, state_dim=state_dim, output_dim=num_classes,
            memory_slots=memory_slots, cognitive_steps=3, reasoning_steps=2,
        )
        return _Wrapper(enc, m)

    def _make_T(T):
        def _fn():
            enc = _make_encoder()
            m = UCGAModel(
                input_dim=state_dim, state_dim=state_dim, output_dim=num_classes,
                memory_slots=memory_slots, cognitive_steps=T, reasoning_steps=2,
            )
            return _Wrapper(enc, m)
        return _fn

    def _tanh_baseline():
        enc = _make_encoder()
        m = UCGAModel(
            input_dim=state_dim, state_dim=state_dim, output_dim=num_classes,
            memory_slots=memory_slots, cognitive_steps=3, reasoning_steps=2,
        )
        _apply_tanh_activation(m)
        return _Wrapper(enc, m)

    return {
        "Full UCGA": _full_ucga,
        "No Correction": _no_correction,
        "No Memory": _no_memory,
        "Random Memory": _random_memory,
        "T=1": _make_T(1),
        "T=3": _make_T(3),
        "T=5": _make_T(5),
        "Tanh (no LN)": _tanh_baseline,
    }
