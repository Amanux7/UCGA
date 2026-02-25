"""
compute_metrics.py — Parameter counting, FLOPs estimation, and timing utilities.

Provides helpers for the scaling & baseline comparison experiments.

Author: Dr. Elena Voss / Aman Singh
"""

import time
import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple


def count_parameters(model: nn.Module, trainable_only: bool = True) -> int:
    """Return the total number of parameters in a model."""
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())


def count_parameters_by_module(model: nn.Module) -> Dict[str, int]:
    """Return a dict mapping top-level submodule names to their param counts."""
    breakdown = {}
    for name, module in model.named_children():
        count = sum(p.numel() for p in module.parameters())
        breakdown[name] = count
    return breakdown


def estimate_flops(model: nn.Module, input_dim: int, batch_size: int = 1) -> int:
    """
    Rough FLOPs estimate by counting multiply-accumulate operations (MACs × 2)
    for all Linear and Conv layers.  This is a lightweight alternative to
    torchprofile / fvcore for models that are primarily linear + conv.

    Parameters
    ----------
    model : nn.Module
    input_dim : int
        Dimensionality of the input vector (for Linear-only models).
    batch_size : int
        Batch size for the estimate.

    Returns
    -------
    int
        Estimated FLOPs (2 × MACs).
    """
    total_macs = 0
    for module in model.modules():
        if isinstance(module, nn.Linear):
            # MACs = in_features × out_features
            total_macs += module.in_features * module.out_features
        elif isinstance(module, nn.Conv1d):
            # MACs = in_ch × out_ch × kernel_size × output_length (approximate)
            total_macs += (
                module.in_channels
                * module.out_channels
                * module.kernel_size[0]
                * 1  # output_length unknown, use 1 as per-position cost
            )
        elif isinstance(module, nn.Conv2d):
            # MACs = in_ch × out_ch × kH × kW (per output position)
            total_macs += (
                module.in_channels
                * module.out_channels
                * module.kernel_size[0]
                * module.kernel_size[1]
            )
    return total_macs * 2 * batch_size  # FLOPs = 2 × MACs


def estimate_flops_from_forward(
    model: nn.Module,
    sample_input: torch.Tensor,
) -> int:
    """
    Estimate FLOPs by hooking into Linear/Conv layers during a forward pass.
    More accurate than static counting because it captures actual tensor shapes.

    Parameters
    ----------
    model : nn.Module
    sample_input : torch.Tensor

    Returns
    -------
    int
        Estimated FLOPs.
    """
    total_macs = [0]
    hooks = []

    def _hook_linear(module, inp, out):
        batch = inp[0].shape[0]
        total_macs[0] += batch * module.in_features * module.out_features

    def _hook_conv2d(module, inp, out):
        batch, _, h_out, w_out = out.shape
        k = module.kernel_size[0] * module.kernel_size[1]
        total_macs[0] += (
            batch * module.in_channels * module.out_channels * k * h_out * w_out
        )

    for module in model.modules():
        if isinstance(module, nn.Linear):
            hooks.append(module.register_forward_hook(_hook_linear))
        elif isinstance(module, nn.Conv2d):
            hooks.append(module.register_forward_hook(_hook_conv2d))

    with torch.no_grad():
        model(sample_input)

    for h in hooks:
        h.remove()

    return total_macs[0] * 2  # FLOPs = 2 × MACs


def measure_wall_clock(
    model: nn.Module,
    sample_input: torch.Tensor,
    num_warmup: int = 3,
    num_runs: int = 10,
) -> Tuple[float, float]:
    """
    Measure wall-clock time for a single forward pass (ms).

    Returns
    -------
    Tuple[float, float]
        (mean_ms, std_ms) over `num_runs` iterations.
    """
    device = next(model.parameters()).device
    times = []

    with torch.no_grad():
        # Warmup
        for _ in range(num_warmup):
            model(sample_input)
            if device.type == "cuda":
                torch.cuda.synchronize()

        # Timed runs
        for _ in range(num_runs):
            if device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            model(sample_input)
            if device.type == "cuda":
                torch.cuda.synchronize()
            t1 = time.perf_counter()
            times.append((t1 - t0) * 1000)  # ms

    import numpy as np
    arr = np.array(times)
    return float(arr.mean()), float(arr.std())


def format_params(n: int) -> str:
    """Format parameter count: 1234567 → '1.23M'."""
    if n >= 1_000_000:
        return f"{n / 1_000_000:.2f}M"
    elif n >= 1_000:
        return f"{n / 1_000:.1f}K"
    return str(n)


def format_flops(n: int) -> str:
    """Format FLOPs: 1234567890 → '1.23G'."""
    if n >= 1_000_000_000:
        return f"{n / 1_000_000_000:.2f}G"
    elif n >= 1_000_000:
        return f"{n / 1_000_000:.2f}M"
    elif n >= 1_000:
        return f"{n / 1_000:.1f}K"
    return str(n)
