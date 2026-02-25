"""
intelligence_dashboard.py — UCGA Intelligence Dashboard

Runs all three validation systems (ROS, LOS, GIB), computes the
composite UCGA Intelligence Score, and produces a unified visual report.

Metrics displayed:
    - ROS Score (reasoning optimization)
    - LOS Score (learning optimization)
    - GIB Adaptivity Score (balancer adaptivity)
    - UCGA Intelligence Score (weighted composite)
    - Reasoning convergence graph
    - Memory contribution graph
    - Balance factor graph

Usage:
    python utils/intelligence_dashboard.py

Author: Aman Singh
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from datetime import datetime

from experiments.ros_validation import ROSValidator
from experiments.los_validation import LOSValidator
from experiments.gib_validation import GIBValidator
from utils.logger import get_logger

logger = get_logger("Intelligence-Dashboard")


# ======================================================================
# Intelligence Metric
# ======================================================================
def compute_ucga_intelligence_score(
    ros_score: float,
    los_score: float,
    gib_score: float,
    ros_weight: float = 0.4,
    los_weight: float = 0.35,
    gib_weight: float = 0.25,
) -> float:
    """
    Compute the composite UCGA Intelligence Score.

    UCGA_intelligence_score = weighted_sum(ROS_score, LOS_score, GIB_score)

    Parameters
    ----------
    ros_score : float
        Reasoning Optimization System score.
    los_score : float
        Learning Optimization System score.
    gib_score : float
        General Intelligence Balancer adaptivity score.
    ros_weight, los_weight, gib_weight : float
        Weights for each component (must sum to 1.0).

    Returns
    -------
    float
        Composite intelligence score.
    """
    return ros_weight * ros_score + los_weight * los_score + gib_weight * gib_score


# ======================================================================
# Dashboard
# ======================================================================
class IntelligenceDashboard:
    """
    Unified intelligence validation and reporting dashboard.

    Orchestrates ROS, LOS, and GIB validators and produces a
    comprehensive visual report.

    Parameters
    ----------
    input_dim : int
        Input dimensionality.
    state_dim : int
        Cognitive state dimensionality.
    output_dim : int
        Output dimensionality.
    output_dir : str
        Directory for saving plots and reports.
    """

    def __init__(
        self,
        input_dim: int = 64,
        state_dim: int = 128,
        output_dim: int = 32,
        output_dir: str = "outputs/dashboard",
    ):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        self.ros_validator = ROSValidator(
            input_dim=input_dim,
            state_dim=state_dim,
            output_dim=output_dim,
            cognitive_steps=5,
            reasoning_steps=5,
        )

        self.los_validator = LOSValidator(
            input_dim=input_dim,
            state_dim=state_dim,
            output_dim=output_dim,
            memory_slots=64,
        )

        self.gib_validator = GIBValidator(
            input_dim=input_dim,
            state_dim=state_dim,
            output_dim=output_dim,
        )

        self.results = {}

    def run_full_validation(self) -> dict:
        """
        Run complete intelligence validation: ROS + LOS + GIB.

        Returns
        -------
        dict
            All metrics and composite intelligence score.
        """
        logger.info("=" + "=" * 58 + "=")
        logger.info("   UCGA Intelligence Dashboard -- Full Validation           ")
        logger.info("=" + "=" * 58 + "=")

        # === ROS ===
        logger.info("\n" + "─" * 60)
        logger.info("  S 1  Reasoning Optimization System (ROS)")
        logger.info("─" * 60)

        x_ros = torch.randn(32, self.ros_validator.input_dim)
        ros_result = self.ros_validator.run_instrumented_forward(x_ros)
        ros_batch = self.ros_validator.run_batch_validation(n_samples=32, difficulty_levels=5)
        ros_plot = self.ros_validator.plot_reasoning_curve(
            ros_result, save_path=os.path.join(self.output_dir, "ros_convergence.png"),
        )
        ros_score = ros_batch["avg_ros_score"]

        # === LOS ===
        logger.info("\n" + "─" * 60)
        logger.info("  S 2  Learning Optimization System (LOS)")
        logger.info("─" * 60)

        los_episodes = self.los_validator.track_memory_over_episodes(
            n_episodes=50, batch_size=16,
        )
        los_plot = self.los_validator.plot_memory_evolution(
            los_episodes["episode_logs"],
            save_path=os.path.join(self.output_dir, "los_memory.png"),
        )
        los_metrics = self.los_validator.compute_los_score(n_samples=100)
        los_score = los_metrics["los_score"]

        # === GIB ===
        logger.info("\n" + "─" * 60)
        logger.info("  S 3  General Intelligence Balancer (GIB)")
        logger.info("─" * 60)

        gib_result = self.gib_validator.test_adaptivity(n_conditions=12, batch_size=32)
        gib_plot = self.gib_validator.plot_balance_factors(
            gib_result["condition_results"],
            save_path=os.path.join(self.output_dir, "gib_balance.png"),
        )
        gib_score = gib_result["gib_adaptivity_score"]

        # === Composite Score ===
        ucga_score = compute_ucga_intelligence_score(ros_score, los_score, gib_score)

        self.results = {
            "timestamp": datetime.now().isoformat(),
            "ros_score": ros_score,
            "los_score": los_score,
            "gib_adaptivity_score": gib_score,
            "ucga_intelligence_score": ucga_score,
            "ros_convergence_rate": ros_batch["avg_convergence_rate"],
            "los_memory_contribution": los_metrics["memory_contribution_l2"],
            "gib_stream_analysis": gib_result["stream_analysis"],
            "plots": {
                "ros": ros_plot,
                "los": los_plot,
                "gib": gib_plot,
            },
        }

        return self.results

    def generate_report(self) -> str:
        """Generate the combined dashboard plot and text report."""
        if not self.results:
            raise RuntimeError("Run run_full_validation() first.")

        r = self.results

        # ---- Combined dashboard figure ----
        fig = plt.figure(figsize=(20, 12))
        fig.suptitle(
            f"UCGA Intelligence Dashboard",
            fontsize=20, fontweight="bold", y=0.98,
        )

        # Scorecard panel (top)
        ax_score = fig.add_axes([0.05, 0.82, 0.9, 0.12])
        ax_score.axis("off")
        scorecard_text = (
            f"    ROS Score: {r['ros_score']:.4f}        "
            f"LOS Score: {r['los_score']:.4f}        "
            f"GIB Score: {r['gib_adaptivity_score']:.6f}        "
            f"UCGA Intelligence: {r['ucga_intelligence_score']:.4f}"
        )
        ax_score.text(
            0.5, 0.5, scorecard_text,
            transform=ax_score.transAxes, fontsize=14, fontweight="bold",
            ha="center", va="center",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="#264653", alpha=0.9),
            color="white",
        )

        # Gauge panel: bar chart of scores
        ax_gauge = fig.add_subplot(2, 3, 4)
        labels = ["ROS", "LOS", "GIB\n(×100)"]
        values = [r["ros_score"], r["los_score"], r["gib_adaptivity_score"] * 100]
        colors = ["#e63946", "#2a9d8f", "#457b9d"]
        bars = ax_gauge.bar(labels, values, color=colors, alpha=0.85, edgecolor="white")
        ax_gauge.set_title("Intelligence Scores", fontweight="bold")
        ax_gauge.set_ylabel("Score")
        ax_gauge.grid(True, alpha=0.3, axis="y")
        for bar, val in zip(bars, values):
            ax_gauge.text(
                bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{val:.3f}", ha="center", fontsize=10, fontweight="bold",
            )

        # Convergence rate panel
        ax_conv = fig.add_subplot(2, 3, 5)
        ax_conv.text(
            0.5, 0.6,
            f"{r['ros_convergence_rate']:.4f}",
            transform=ax_conv.transAxes, fontsize=36, fontweight="bold",
            ha="center", va="center", color="#e63946",
        )
        ax_conv.text(
            0.5, 0.25,
            "Convergence Rate",
            transform=ax_conv.transAxes, fontsize=12,
            ha="center", va="center", color="#666",
        )
        ax_conv.axis("off")

        # Memory contribution panel
        ax_mem = fig.add_subplot(2, 3, 6)
        ax_mem.text(
            0.5, 0.6,
            f"{r['los_memory_contribution']:.4f}",
            transform=ax_mem.transAxes, fontsize=36, fontweight="bold",
            ha="center", va="center", color="#2a9d8f",
        )
        ax_mem.text(
            0.5, 0.25,
            "Memory Contribution (L2)",
            transform=ax_mem.transAxes, fontsize=12,
            ha="center", va="center", color="#666",
        )
        ax_mem.axis("off")

        # GIB stream breakdown
        ax_stream = fig.add_subplot(2, 3, 1)
        sa = r["gib_stream_analysis"]
        stream_names = list(sa.keys())
        means = [sa[s]["mean"] for s in stream_names]
        stds = [sa[s]["std"] for s in stream_names]
        ax_stream.bar(stream_names, means, yerr=stds, color=colors[:len(stream_names)],
                      alpha=0.85, capsize=5, edgecolor="white")
        ax_stream.set_title("GIB Stream Weights", fontweight="bold")
        ax_stream.set_ylabel("Mean Weight")
        ax_stream.grid(True, alpha=0.3, axis="y")

        # Composite intelligence display
        ax_final = fig.add_subplot(2, 3, 2)
        ax_final.text(
            0.5, 0.6,
            f"{r['ucga_intelligence_score']:.4f}",
            transform=ax_final.transAxes, fontsize=42, fontweight="bold",
            ha="center", va="center", color="#264653",
        )
        ax_final.text(
            0.5, 0.2,
            "UCGA Intelligence Score\n(0.4·ROS + 0.35·LOS + 0.25·GIB)",
            transform=ax_final.transAxes, fontsize=10,
            ha="center", va="center", color="#666",
        )
        ax_final.axis("off")

        # Timestamp
        ax_ts = fig.add_subplot(2, 3, 3)
        ax_ts.text(
            0.5, 0.5,
            f"Generated:\n{r['timestamp'][:19]}",
            transform=ax_ts.transAxes, fontsize=11,
            ha="center", va="center", color="#333",
        )
        ax_ts.axis("off")

        plt.tight_layout(rect=[0, 0, 1, 0.80])
        dashboard_path = os.path.join(self.output_dir, "intelligence_dashboard.png")
        plt.savefig(dashboard_path, dpi=150, bbox_inches="tight")
        plt.close()

        logger.info(f"Saved dashboard: {dashboard_path}")

        # ---- Text report ----
        report_lines = [
            "=" * 60,
            "  UCGA INTELLIGENCE REPORT",
            "=" * 60,
            f"  Timestamp:          {r['timestamp'][:19]}",
            "",
            "  +----------------------------------------+",
            f"  |  UCGA Intelligence Score: {r['ucga_intelligence_score']:.4f}         |",
            "  +----------------------------------------+",
            "",
            "  Component Scores:",
            f"    ROS Score:             {r['ros_score']:.4f}",
            f"    LOS Score:             {r['los_score']:.4f}",
            f"    GIB Adaptivity Score:  {r['gib_adaptivity_score']:.6f}",
            "",
            "  Detailed Metrics:",
            f"    Convergence Rate:      {r['ros_convergence_rate']:.4f}",
            f"    Memory Contribution:   {r['los_memory_contribution']:.4f}",
            "",
            "  GIB Stream Analysis:",
        ]
        for stream, stats in r["gib_stream_analysis"].items():
            report_lines.append(
                f"    {stream:10s}: mean={stats['mean']:.3f}  std={stats['std']:.3f}"
            )
        report_lines.extend([
            "",
            "  Plots:",
            f"    ROS: {r['plots']['ros']}",
            f"    LOS: {r['plots']['los']}",
            f"    GIB: {r['plots']['gib']}",
            f"    Dashboard: {dashboard_path}",
            "=" * 60,
        ])

        report = "\n".join(report_lines)
        report_path = os.path.join(self.output_dir, "intelligence_report.txt")
        with open(report_path, "w") as f:
            f.write(report)

        logger.info(f"Saved report: {report_path}")
        print(report)

        return report


# ======================================================================
# Main
# ======================================================================
def main():
    dashboard = IntelligenceDashboard(
        input_dim=64,
        state_dim=128,
        output_dim=32,
        output_dir="outputs/dashboard",
    )

    results = dashboard.run_full_validation()
    report = dashboard.generate_report()

    logger.info("\n✓ Intelligence dashboard complete.")


if __name__ == "__main__":
    main()
