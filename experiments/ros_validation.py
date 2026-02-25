"""
ros_validation.py — Reasoning Optimization System (ROS) Validation

Monitors and validates that UCGA's reasoning improves across recursive
cognitive loop iterations.  Tracks reasoning state convergence, error
reduction, and confidence growth at each iteration.

Metrics:
    - reasoning_state norm at each iteration
    - reasoning_error at each iteration (distance from converged state)
    - reasoning_convergence_rate
    - ROS_score = (initial_error - final_error) / initial_error

Usage:
    python experiments/ros_validation.py

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

logger = get_logger("ROS-Validation")


class ROSValidator:
    """
    Reasoning Optimization System validator.

    Instruments the UCGA cognitive loop to capture per-iteration
    reasoning states, errors, and confidence scores.

    Parameters
    ----------
    input_dim : int
        Input dimensionality.
    state_dim : int
        Cognitive state dimensionality.
    output_dim : int
        Output dimensionality.
    cognitive_steps : int
        Number of cognitive loop iterations to evaluate.
    reasoning_steps : int
        Number of inner reasoning refinement steps.
    """

    def __init__(
        self,
        input_dim: int = 64,
        state_dim: int = 128,
        output_dim: int = 32,
        cognitive_steps: int = 5,
        reasoning_steps: int = 5,
    ):
        self.input_dim = input_dim
        self.state_dim = state_dim
        self.cognitive_steps = cognitive_steps

        self.encoder = VectorEncoder(input_dim=input_dim, output_dim=state_dim)
        self.model = UCGAModel(
            input_dim=state_dim,
            state_dim=state_dim,
            output_dim=output_dim,
            cognitive_steps=cognitive_steps,
            reasoning_steps=reasoning_steps,
        )

        # Storage for per-iteration metrics
        self.iteration_logs: List[Dict[str, Any]] = []

    @torch.no_grad()
    def run_instrumented_forward(
        self, x: torch.Tensor,
    ) -> Dict[str, Any]:
        """
        Run the UCGA forward pass with per-iteration instrumentation.

        Hooks into each cognitive step to capture reasoning state,
        confidence, and error metrics.

        Returns
        -------
        dict
            - ``iteration_data``: list of per-step metrics
            - ``ros_score``: overall ROS score
            - ``convergence_rate``: speed of reasoning convergence
        """
        self.model.eval()
        self.encoder.eval()

        B = x.size(0)
        encoded = self.encoder(x)

        # Reset model
        self.model._reset_all(B)
        x_loop = encoded
        x_original = encoded
        memory_bank = self.model.persistent_memory.get_memory_bank(B)

        iteration_data = []
        prev_reasoning_state = None

        for t in range(self.cognitive_steps):
            # 1. Perceive
            percept = self.model.perception([x_loop])

            # 2. Memory retrieval
            mem_state = self.model.memory_node([percept], memory_bank=memory_bank)

            # 3. Reasoning — capture pre/post states
            reason_input_norm = percept.norm(dim=-1).mean().item()
            reason_state = self.model.reasoning([percept, mem_state])
            reason_output_norm = reason_state.norm(dim=-1).mean().item()

            # Reasoning error: distance from previous iteration's reasoning state
            if prev_reasoning_state is not None:
                reasoning_error = (reason_state - prev_reasoning_state).norm(dim=-1).mean().item()
            else:
                reasoning_error = reason_output_norm  # initial error = output magnitude

            prev_reasoning_state = reason_state.clone()

            # 4. Planning
            plan_state = self.model.planning([reason_state])

            # 5. Evaluation
            eval_state = self.model.evaluation([plan_state, reason_state])
            confidence = self.model.evaluation.get_confidence().mean().item()

            # 6. Correction
            if confidence < self.model.correction_threshold:
                corrected = self.model.correction([plan_state, eval_state])
            else:
                corrected = plan_state

            # 7. Balance
            balanced = self.model.balancer([reason_state, corrected, mem_state])

            # Update loop variable
            x_loop = balanced + x_original

            # Log iteration metrics
            step_data = {
                "iteration": t,
                "reasoning_input_norm": reason_input_norm,
                "reasoning_output_norm": reason_output_norm,
                "reasoning_error": reasoning_error,
                "confidence": confidence,
                "balanced_norm": balanced.norm(dim=-1).mean().item(),
            }
            iteration_data.append(step_data)

            logger.info(
                f"  Iter {t}  |  "
                f"Reason err: {reasoning_error:.4f}  |  "
                f"Confidence: {confidence:.4f}  |  "
                f"State norm: {reason_output_norm:.4f}"
            )

        # Compute ROS score
        initial_error = iteration_data[0]["reasoning_error"]
        final_error = iteration_data[-1]["reasoning_error"]
        if initial_error > 1e-8:
            ros_score = (initial_error - final_error) / initial_error
        else:
            ros_score = 0.0

        # Convergence rate: mean relative error reduction per step
        errors = [d["reasoning_error"] for d in iteration_data]
        if len(errors) > 1 and errors[0] > 1e-8:
            convergence_rate = 1.0 - (errors[-1] / errors[0]) ** (1.0 / (len(errors) - 1))
        else:
            convergence_rate = 0.0

        result = {
            "iteration_data": iteration_data,
            "ros_score": ros_score,
            "convergence_rate": convergence_rate,
            "initial_error": initial_error,
            "final_error": final_error,
        }

        self.iteration_logs.append(result)
        return result

    def run_batch_validation(
        self, n_samples: int = 50, difficulty_levels: int = 5,
    ) -> Dict[str, Any]:
        """
        Run ROS validation across multiple input difficulty levels.

        Parameters
        ----------
        n_samples : int
            Samples per difficulty level.
        difficulty_levels : int
            Number of difficulty tiers (controls input magnitude).

        Returns
        -------
        dict
            Aggregated ROS metrics across difficulty levels.
        """
        logger.info("=" * 60)
        logger.info("  ROS Batch Validation")
        logger.info("=" * 60)

        all_results = []
        for level in range(1, difficulty_levels + 1):
            logger.info(f"\n--- Difficulty Level {level}/{difficulty_levels} ---")
            # Higher difficulty = larger input magnitude + noise
            x = torch.randn(n_samples, self.input_dim) * level
            result = self.run_instrumented_forward(x)
            result["difficulty_level"] = level
            all_results.append(result)

        avg_ros = np.mean([r["ros_score"] for r in all_results])
        avg_convergence = np.mean([r["convergence_rate"] for r in all_results])

        summary = {
            "per_difficulty": all_results,
            "avg_ros_score": float(avg_ros),
            "avg_convergence_rate": float(avg_convergence),
        }

        logger.info(f"\n{'=' * 60}")
        logger.info(f"  ROS Score (avg):         {avg_ros:.4f}")
        logger.info(f"  Convergence Rate (avg):  {avg_convergence:.4f}")
        logger.info(f"{'=' * 60}")

        return summary

    def plot_reasoning_curve(
        self, result: Dict[str, Any], save_path: str = "outputs/ros_reasoning_curve.png",
    ) -> str:
        """Plot reasoning improvement curve and save to disk."""
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        data = result["iteration_data"]
        iters = [d["iteration"] for d in data]
        errors = [d["reasoning_error"] for d in data]
        confs = [d["confidence"] for d in data]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        # Reasoning error curve
        ax1.plot(iters, errors, "o-", color="#e63946", linewidth=2, markersize=8)
        ax1.set_xlabel("Cognitive Iteration", fontsize=12)
        ax1.set_ylabel("Reasoning Error", fontsize=12)
        ax1.set_title("Reasoning Error Convergence (ROS)", fontsize=14, fontweight="bold")
        ax1.grid(True, alpha=0.3)
        ax1.fill_between(iters, errors, alpha=0.15, color="#e63946")

        # Confidence curve
        ax2.plot(iters, confs, "s-", color="#457b9d", linewidth=2, markersize=8)
        ax2.set_xlabel("Cognitive Iteration", fontsize=12)
        ax2.set_ylabel("Confidence Score", fontsize=12)
        ax2.set_title("Confidence Growth", fontsize=14, fontweight="bold")
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim(0, 1)
        ax2.fill_between(iters, confs, alpha=0.15, color="#457b9d")

        # Add ROS score annotation
        ros = result["ros_score"]
        fig.suptitle(f"ROS Score: {ros:.4f}", fontsize=16, fontweight="bold", y=1.02)

        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
        logger.info(f"Saved reasoning curve: {save_path}")
        return save_path


# ======================================================================
# Main
# ======================================================================
def main():
    logger.info("=" * 60)
    logger.info("  UCGA Reasoning Optimization System (ROS) Validation")
    logger.info("=" * 60)

    validator = ROSValidator(
        input_dim=64,
        state_dim=128,
        output_dim=32,
        cognitive_steps=5,
        reasoning_steps=5,
    )

    # Single-run validation with visualization
    logger.info("\n--- Single Run Validation ---")
    x = torch.randn(16, 64)
    result = validator.run_instrumented_forward(x)
    validator.plot_reasoning_curve(result)

    logger.info(f"\nROS Score: {result['ros_score']:.4f}")
    logger.info(f"Convergence Rate: {result['convergence_rate']:.4f}")

    # Batch validation across difficulty levels
    summary = validator.run_batch_validation(n_samples=32, difficulty_levels=5)

    logger.info("\n✓ ROS validation complete.")
    return summary


if __name__ == "__main__":
    main()
