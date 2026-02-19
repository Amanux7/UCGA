"""
visualization.py — Plot training curves and cognitive graph structure.

Author: Aman Singh
"""

import matplotlib
matplotlib.use("Agg")  # non-interactive backend

import matplotlib.pyplot as plt
import numpy as np
from typing import List, Optional, Dict
import os


def plot_training_curve(
    losses: List[float],
    title: str = "Training Loss",
    save_path: Optional[str] = None,
    show: bool = False,
) -> None:
    """
    Plot a training loss curve.

    Parameters
    ----------
    losses : List[float]
        Per-epoch (or per-step) loss values.
    title : str
        Plot title.
    save_path : str, optional
        If provided, save the figure to this path.
    show : bool
        If ``True``, display the plot interactively.
    """
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(losses, linewidth=1.5, color="#4A90D9")
    ax.set_xlabel("Step")
    ax.set_ylabel("Loss")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        fig.savefig(save_path, dpi=150)
    if show:
        plt.show()
    plt.close(fig)


def plot_confidence_history(
    confidences: List[float],
    title: str = "Cognitive Confidence over Time",
    save_path: Optional[str] = None,
) -> None:
    """
    Plot the evaluation-node confidence across cognitive steps / episodes.
    """
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(confidences, linewidth=1.5, color="#E8524A", marker="o", markersize=3)
    ax.axhline(y=0.5, color="gray", linestyle="--", alpha=0.5, label="Correction threshold")
    ax.set_xlabel("Step")
    ax.set_ylabel("Confidence")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_cognitive_graph(save_path: Optional[str] = None) -> None:
    """
    Draw a schematic of the UCGA cognitive graph (static layout).
    """
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 8)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title("UCGA Cognitive Graph Architecture", fontsize=16, fontweight="bold")

    nodes = {
        "Perception":  (2, 7),
        "Memory":      (5, 7),
        "Reasoning":   (3.5, 5.5),
        "Planning":    (3.5, 4),
        "Evaluation":  (6, 4),
        "Correction":  (6, 5.5),
        "Balancer":    (5, 2.5),
        "Output":      (5, 1),
    }

    edges = [
        ("Perception", "Reasoning"),
        ("Memory", "Reasoning"),
        ("Reasoning", "Planning"),
        ("Planning", "Evaluation"),
        ("Evaluation", "Correction"),
        ("Correction", "Balancer"),
        ("Planning", "Balancer"),
        ("Reasoning", "Balancer"),
        ("Memory", "Balancer"),
        ("Balancer", "Output"),
        ("Perception", "Memory"),
    ]

    colors = {
        "Perception": "#4A90D9",
        "Memory":     "#50C878",
        "Reasoning":  "#E8524A",
        "Planning":   "#F5A623",
        "Evaluation": "#9B59B6",
        "Correction": "#E74C3C",
        "Balancer":   "#1ABC9C",
        "Output":     "#34495E",
    }

    # Draw edges
    for src, dst in edges:
        x0, y0 = nodes[src]
        x1, y1 = nodes[dst]
        ax.annotate(
            "", xy=(x1, y1), xytext=(x0, y0),
            arrowprops=dict(arrowstyle="->", color="#888", lw=1.5),
        )

    # Draw nodes
    for name, (x, y) in nodes.items():
        circle = plt.Circle((x, y), 0.5, color=colors[name], alpha=0.85, zorder=5)
        ax.add_patch(circle)
        ax.text(x, y, name, ha="center", va="center", fontsize=8,
                fontweight="bold", color="white", zorder=6)

    fig.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    # Quick demo
    plot_cognitive_graph(save_path="cognitive_graph.png")
    print("Generated cognitive_graph.png")
