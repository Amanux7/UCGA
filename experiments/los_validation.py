"""
los_validation.py — Learning Optimization System (LOS) Validation

Validates that UCGA's persistent memory improves system performance
over time.  Compares performance with and without memory, tracks
memory retrieval accuracy, and measures the memory contribution factor.

Metrics:
    - memory_retrieval_accuracy
    - memory_usage_frequency
    - performance_improvement_over_time
    - LOS_score = performance_with_memory - performance_without_memory

Usage:
    python experiments/los_validation.py

Author: Aman Singh
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from typing import Dict, List, Any

from ucga.ucga_model import UCGAModel
from ucga.encoders import VectorEncoder
from utils.logger import get_logger

logger = get_logger("LOS-Validation")


class LOSValidator:
    """
    Learning Optimization System validator.

    Measures the contribution of persistent memory to UCGA performance
    by comparing outputs with and without memory, and tracking how
    memory usage evolves over time.

    Parameters
    ----------
    input_dim : int
        Input dimensionality.
    state_dim : int
        Cognitive state dimensionality.
    output_dim : int
        Output dimensionality.
    memory_slots : int
        Number of persistent memory slots.
    """

    def __init__(
        self,
        input_dim: int = 64,
        state_dim: int = 128,
        output_dim: int = 32,
        memory_slots: int = 64,
    ):
        self.input_dim = input_dim
        self.state_dim = state_dim

        self.encoder = VectorEncoder(input_dim=input_dim, output_dim=state_dim)
        self.model = UCGAModel(
            input_dim=state_dim,
            state_dim=state_dim,
            output_dim=output_dim,
            memory_slots=memory_slots,
            cognitive_steps=3,
        )

    @torch.no_grad()
    def measure_memory_contribution(
        self, inputs: torch.Tensor,
    ) -> Dict[str, Any]:
        """
        Compare model output with and without memory.

        Returns
        -------
        dict
            - ``output_with_memory``: output tensor
            - ``output_without_memory``: output tensor
            - ``memory_contribution``: L2 difference
            - ``cosine_similarity``: alignment between outputs
        """
        self.model.eval()
        self.encoder.eval()
        encoded = self.encoder(inputs)

        # --- WITH memory ---
        output_with, meta_with = self.model(encoded, return_meta=True)

        # --- WITHOUT memory: zero out memory bank ---
        saved_memory = self.model.persistent_memory.memory.clone()
        saved_usage = self.model.persistent_memory.usage.clone()
        self.model.persistent_memory.memory.zero_()
        self.model.persistent_memory.usage.zero_()

        output_without, meta_without = self.model(encoded, return_meta=True)

        # Restore memory
        self.model.persistent_memory.memory.copy_(saved_memory)
        self.model.persistent_memory.usage.copy_(saved_usage)

        # Metrics
        l2_diff = (output_with - output_without).norm(dim=-1).mean().item()
        cos_sim = F.cosine_similarity(output_with, output_without, dim=-1).mean().item()

        return {
            "output_with_memory": output_with,
            "output_without_memory": output_without,
            "memory_contribution": l2_diff,
            "cosine_similarity": cos_sim,
            "confidence_with": meta_with["confidences"],
            "confidence_without": meta_without["confidences"],
        }

    @torch.no_grad()
    def track_memory_over_episodes(
        self, n_episodes: int = 50, batch_size: int = 16,
    ) -> Dict[str, Any]:
        """
        Feed sequential episodes and track how memory usage and
        performance evolve over time.

        Returns
        -------
        dict
            Per-episode metrics including memory usage, output norms,
            confidences, and retrieval accuracies.
        """
        self.model.eval()
        self.encoder.eval()
        self.model.reset_memory()

        episode_logs = []

        for ep in range(n_episodes):
            x = torch.randn(batch_size, self.input_dim)
            encoded = self.encoder(x)

            # Measure memory state before forward
            usage_before = self.model.persistent_memory.usage.clone()

            output, meta = self.model(encoded, return_meta=True)

            # Measure memory state after forward
            usage_after = self.model.persistent_memory.usage.clone()

            # Memory retrieval accuracy: how well does read match the write?
            # Proxy: cosine similarity between query and retrieved content
            mem_bank = self.model.persistent_memory.get_memory_bank(batch_size)
            read_query = self.model.perception([encoded])
            retrieved = self.model.persistent_memory.read(read_query)
            retrieval_sim = F.cosine_similarity(read_query, retrieved, dim=-1).mean().item()

            # Memory usage frequency (slots used / total slots)
            slots_used = (usage_after > 0).float().mean().item()

            step_log = {
                "episode": ep,
                "output_norm": output.norm(dim=-1).mean().item(),
                "confidence": meta["confidences"][-1] if meta["confidences"] else 0.0,
                "corrections": meta["corrections"],
                "memory_slots_used_ratio": slots_used,
                "memory_usage_total": usage_after.sum().item(),
                "retrieval_similarity": retrieval_sim,
            }
            episode_logs.append(step_log)

            if ep % 10 == 0:
                logger.info(
                    f"  Episode {ep:3d}  |  "
                    f"Retrieval sim: {retrieval_sim:.4f}  |  "
                    f"Slots used: {slots_used:.2%}  |  "
                    f"Confidence: {step_log['confidence']:.4f}"
                )

        return {"episode_logs": episode_logs}

    def compute_los_score(
        self, n_samples: int = 100,
    ) -> Dict[str, Any]:
        """
        Compute the overall LOS score.

        LOS_score = mean(performance_with_memory - performance_without_memory)

        Performance is measured as output stability (lower variance = better).
        """
        logger.info("=" * 60)
        logger.info("  LOS Score Computation")
        logger.info("=" * 60)

        # Prime memory with some episodes
        logger.info("Priming memory with 20 episodes...")
        for _ in range(20):
            x = torch.randn(16, self.input_dim)
            encoded = self.encoder(x)
            self.model(encoded)

        # Now measure contribution
        x_test = torch.randn(n_samples, self.input_dim)
        result = self.measure_memory_contribution(x_test)

        # Performance metric: confidence differential
        conf_with = np.mean(result["confidence_with"])
        conf_without = np.mean(result["confidence_without"])
        los_score = conf_with - conf_without

        logger.info(f"  Confidence WITH memory:    {conf_with:.4f}")
        logger.info(f"  Confidence WITHOUT memory: {conf_without:.4f}")
        logger.info(f"  Memory contribution (L2):  {result['memory_contribution']:.4f}")
        logger.info(f"  Cosine similarity:         {result['cosine_similarity']:.4f}")
        logger.info(f"  LOS Score:                 {los_score:.4f}")

        return {
            "los_score": los_score,
            "confidence_with_memory": conf_with,
            "confidence_without_memory": conf_without,
            "memory_contribution_l2": result["memory_contribution"],
            "cosine_similarity": result["cosine_similarity"],
        }

    def plot_memory_evolution(
        self, episode_logs: List[Dict], save_path: str = "outputs/los_memory_evolution.png",
    ) -> str:
        """Plot memory contribution evolution over episodes."""
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        eps = [d["episode"] for d in episode_logs]
        ret_sim = [d["retrieval_similarity"] for d in episode_logs]
        usage = [d["memory_slots_used_ratio"] for d in episode_logs]
        confs = [d["confidence"] for d in episode_logs]

        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        # Retrieval similarity
        axes[0].plot(eps, ret_sim, "-", color="#2a9d8f", linewidth=2)
        axes[0].set_xlabel("Episode")
        axes[0].set_ylabel("Retrieval Similarity")
        axes[0].set_title("Memory Retrieval Quality", fontweight="bold")
        axes[0].grid(True, alpha=0.3)
        axes[0].fill_between(eps, ret_sim, alpha=0.15, color="#2a9d8f")

        # Memory usage
        axes[1].plot(eps, usage, "-", color="#e76f51", linewidth=2)
        axes[1].set_xlabel("Episode")
        axes[1].set_ylabel("Slots Used Ratio")
        axes[1].set_title("Memory Usage Growth", fontweight="bold")
        axes[1].grid(True, alpha=0.3)
        axes[1].set_ylim(0, 1)
        axes[1].fill_between(eps, usage, alpha=0.15, color="#e76f51")

        # Confidence over time
        axes[2].plot(eps, confs, "-", color="#264653", linewidth=2)
        axes[2].set_xlabel("Episode")
        axes[2].set_ylabel("Confidence")
        axes[2].set_title("Performance Over Time", fontweight="bold")
        axes[2].grid(True, alpha=0.3)
        axes[2].fill_between(eps, confs, alpha=0.15, color="#264653")

        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
        logger.info(f"Saved memory evolution plot: {save_path}")
        return save_path


# ======================================================================
# Main
# ======================================================================
def main():
    logger.info("=" * 60)
    logger.info("  UCGA Learning Optimization System (LOS) Validation")
    logger.info("=" * 60)

    validator = LOSValidator(
        input_dim=64,
        state_dim=128,
        output_dim=32,
        memory_slots=64,
    )

    # Track memory evolution
    logger.info("\n--- Memory Evolution (50 episodes) ---")
    result = validator.track_memory_over_episodes(n_episodes=50, batch_size=16)
    validator.plot_memory_evolution(result["episode_logs"])

    # Compute LOS score
    logger.info("\n--- LOS Score ---")
    los_result = validator.compute_los_score(n_samples=100)

    logger.info(f"\nFinal LOS Score: {los_result['los_score']:.4f}")
    logger.info("\n✓ LOS validation complete.")
    return los_result


if __name__ == "__main__":
    main()
