"""
gib_validation.py — General Intelligence Balancer (GIB) Validation

Validates that UCGA's BalancerNode dynamically adapts its weighting of
reasoning, memory, and planning streams based on input characteristics.

Metrics:
    - balance_factor per stream over time
    - reasoning_influence_weight
    - memory_influence_weight
    - planning_influence_weight
    - GIB_adaptivity_score = variance(balance_factor)

Usage:
    python experiments/gib_validation.py

Author: Aman Singh
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import torch.nn as nn
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from typing import Dict, List, Any

from ucga.ucga_model import UCGAModel
from ucga.encoders import VectorEncoder
from utils.logger import get_logger

logger = get_logger("GIB-Validation")


class GIBValidator:
    """
    General Intelligence Balancer validator.

    Hooks into the BalancerNode to capture per-stream weights and
    verify that they adapt to different input conditions.

    Parameters
    ----------
    input_dim : int
        Input dimensionality.
    state_dim : int
        Cognitive state dimensionality.
    output_dim : int
        Output dimensionality.
    """

    def __init__(
        self,
        input_dim: int = 64,
        state_dim: int = 128,
        output_dim: int = 32,
    ):
        self.input_dim = input_dim
        self.state_dim = state_dim
        self.num_streams = 3  # reasoning, corrected/plan, memory

        self.encoder = VectorEncoder(input_dim=input_dim, output_dim=state_dim)
        self.model = UCGAModel(
            input_dim=state_dim,
            state_dim=state_dim,
            output_dim=output_dim,
            cognitive_steps=3,
        )

        self.stream_labels = ["reasoning", "planning", "memory"]

    @torch.no_grad()
    def capture_balance_factors(
        self, x: torch.Tensor,
    ) -> Dict[str, Any]:
        """
        Run a forward pass and capture the BalancerNode's stream weights.

        Hooks into the balancer's stream_gate to extract the softmax
        weights that control reasoning vs. planning vs. memory influence.

        Returns
        -------
        dict
            Per-step balance factors and stream weights.
        """
        self.model.eval()
        self.encoder.eval()

        B = x.size(0)
        encoded = self.encoder(x)

        self.model._reset_all(B)
        x_loop = encoded
        x_original = encoded
        memory_bank = self.model.persistent_memory.get_memory_bank(B)

        step_data = []

        for t in range(self.model.cognitive_steps):
            percept = self.model.perception([x_loop])
            mem_state = self.model.memory_node([percept], memory_bank=memory_bank)
            reason_state = self.model.reasoning([percept, mem_state])
            plan_state = self.model.planning([reason_state])
            eval_state = self.model.evaluation([plan_state, reason_state])
            confidence = self.model.evaluation.get_confidence().mean().item()

            if confidence < self.model.correction_threshold:
                corrected = self.model.correction([plan_state, eval_state])
            else:
                corrected = plan_state

            # --- Capture balance factors ---
            inputs = [reason_state, corrected, mem_state]
            inputs_copy = list(inputs)
            while len(inputs_copy) < self.model.balancer.num_streams:
                inputs_copy.append(inputs_copy[-1])
            inputs_copy = inputs_copy[:self.model.balancer.num_streams]

            stacked = torch.stack(inputs_copy, dim=1)  # (B, S, D)
            B_cur, S, D = stacked.shape
            concat = stacked.reshape(B_cur, S * D)
            weights = self.model.balancer.stream_gate(concat)  # (B, S)

            # Mean weights across batch
            mean_weights = weights.mean(dim=0).tolist()

            balanced = self.model.balancer(inputs)
            x_loop = balanced + x_original

            step_info = {
                "step": t,
                "weights": mean_weights,
                "reasoning_weight": mean_weights[0],
                "planning_weight": mean_weights[1],
                "memory_weight": mean_weights[2] if len(mean_weights) > 2 else 0.0,
                "confidence": confidence,
            }
            step_data.append(step_info)

            logger.info(
                f"  Step {t}  |  "
                f"R: {mean_weights[0]:.3f}  "
                f"P: {mean_weights[1]:.3f}  "
                f"M: {mean_weights[2] if len(mean_weights) > 2 else 0:.3f}  |  "
                f"Conf: {confidence:.4f}"
            )

        return {"step_data": step_data}

    @torch.no_grad()
    def test_adaptivity(
        self, n_conditions: int = 10, batch_size: int = 32,
    ) -> Dict[str, Any]:
        """
        Test that balance factors change across different input types.

        Generates inputs with varying characteristics (magnitude, noise,
        sparsity) and checks that the balancer adapts its weights.

        Returns
        -------
        dict
            Per-condition weights and GIB adaptivity score.
        """
        logger.info("=" * 60)
        logger.info("  GIB Adaptivity Test")
        logger.info("=" * 60)

        all_weights = []
        condition_results = []

        for cond in range(n_conditions):
            # Varying input characteristics
            if cond < n_conditions // 3:
                # Low complexity: small magnitude, clean
                x = torch.randn(batch_size, self.input_dim) * 0.1
                desc = "low-complexity"
            elif cond < 2 * n_conditions // 3:
                # Medium complexity: moderate magnitude
                x = torch.randn(batch_size, self.input_dim) * 1.0
                desc = "medium-complexity"
            else:
                # High complexity: large magnitude, high noise
                x = torch.randn(batch_size, self.input_dim) * 5.0
                x += torch.randn_like(x) * 2.0  # extra noise
                desc = "high-complexity"

            result = self.capture_balance_factors(x)
            # Use final step weights
            final_weights = result["step_data"][-1]["weights"]
            all_weights.append(final_weights)

            condition_results.append({
                "condition": cond,
                "type": desc,
                "weights": final_weights,
                "step_data": result["step_data"],
            })

            logger.info(
                f"  Cond {cond} ({desc})  |  "
                f"Weights: [{', '.join(f'{w:.3f}' for w in final_weights)}]"
            )

        # Compute GIB adaptivity score = variance across conditions
        weights_array = np.array(all_weights)  # (n_conditions, num_streams)
        per_stream_var = weights_array.var(axis=0)
        gib_adaptivity_score = float(per_stream_var.mean())

        # Per-stream analysis
        stream_analysis = {}
        for i, label in enumerate(self.stream_labels[:weights_array.shape[1]]):
            stream_analysis[label] = {
                "mean": float(weights_array[:, i].mean()),
                "std": float(weights_array[:, i].std()),
                "min": float(weights_array[:, i].min()),
                "max": float(weights_array[:, i].max()),
            }

        logger.info(f"\n{'=' * 60}")
        logger.info(f"  GIB Adaptivity Score: {gib_adaptivity_score:.6f}")
        for label, stats in stream_analysis.items():
            logger.info(
                f"  {label:10s}: mean={stats['mean']:.3f}  "
                f"std={stats['std']:.3f}  "
                f"range=[{stats['min']:.3f}, {stats['max']:.3f}]"
            )
        logger.info(f"{'=' * 60}")

        return {
            "gib_adaptivity_score": gib_adaptivity_score,
            "per_stream_variance": per_stream_var.tolist(),
            "stream_analysis": stream_analysis,
            "condition_results": condition_results,
        }

    def plot_balance_factors(
        self,
        condition_results: List[Dict],
        save_path: str = "outputs/gib_balance_factors.png",
    ) -> str:
        """Plot balance factor distribution across conditions."""
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        # Extract per-condition weights
        conds = [r["condition"] for r in condition_results]
        weights_by_stream = {label: [] for label in self.stream_labels}

        for r in condition_results:
            w = r["weights"]
            for i, label in enumerate(self.stream_labels):
                if i < len(w):
                    weights_by_stream[label].append(w[i])

        colors = ["#e63946", "#457b9d", "#2a9d8f"]
        # Stacked area chart
        bottom = np.zeros(len(conds))
        for i, (label, ws) in enumerate(weights_by_stream.items()):
            if ws:
                ax1.bar(conds, ws, bottom=bottom, label=label,
                        color=colors[i], alpha=0.85)
                bottom += np.array(ws)

        ax1.set_xlabel("Input Condition")
        ax1.set_ylabel("Stream Weight")
        ax1.set_title("Balance Factor per Condition", fontweight="bold")
        ax1.legend()
        ax1.grid(True, alpha=0.3, axis="y")

        # Line chart: per-stream weights
        for i, (label, ws) in enumerate(weights_by_stream.items()):
            if ws:
                ax2.plot(conds, ws, "o-", label=label, color=colors[i],
                         linewidth=2, markersize=6)

        ax2.set_xlabel("Input Condition")
        ax2.set_ylabel("Weight")
        ax2.set_title("Per-Stream Weight Variation", fontweight="bold")
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
        logger.info(f"Saved balance factors plot: {save_path}")
        return save_path


# ======================================================================
# Main
# ======================================================================
def main():
    logger.info("=" * 60)
    logger.info("  UCGA General Intelligence Balancer (GIB) Validation")
    logger.info("=" * 60)

    validator = GIBValidator(
        input_dim=64,
        state_dim=128,
        output_dim=32,
    )

    # Run adaptivity test
    result = validator.test_adaptivity(n_conditions=12, batch_size=32)

    # Plot
    validator.plot_balance_factors(result["condition_results"])

    logger.info(f"\nFinal GIB Adaptivity Score: {result['gib_adaptivity_score']:.6f}")
    logger.info("\n✓ GIB validation complete.")
    return result


if __name__ == "__main__":
    main()
