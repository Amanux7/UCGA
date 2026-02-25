"""
synthetic_benchmarks.py — Synthetic reasoning benchmarks for UCGA.

Generates lightweight synthetic versions of harder reasoning tasks
that demonstrate the value of multi-step cognitive refinement (T>1).

Benchmarks:
  1. ARC-AGI:   50 synthetic 10×10 grid transformation puzzles
  2. GSM8K-Hard: 200 synthetic multi-step arithmetic word problems
  3. HotpotQA:  100 synthetic multi-hop questions
  4. Blocksworld: 50 small planning instances (3-5 blocks)

Usage:
    python experiments/synthetic_benchmarks.py               # synthetic only
    python experiments/synthetic_benchmarks.py --full         # download real datasets (Colab)

Author: Dr. Elena Voss / Aman Singh
"""

import sys
import os
import random
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


# ======================================================================
# 1. ARC-AGI — Synthetic Grid Transformation Puzzles
# ======================================================================
class SyntheticARCDataset(Dataset):
    """
    Synthetic ARC-style dataset.  Each puzzle is a 10×10 grid with a
    simple transformation rule (rotate, reflect, fill, translate).
    The model must predict the output grid given the input grid.

    We encode this as classification: given a flattened input grid (100 dims),
    predict which transformation rule was applied (num_rules classes).
    """

    RULES = ["rotate_90", "rotate_180", "reflect_h", "reflect_v", "shift_right", "shift_down", "invert", "fill_border"]

    def __init__(self, num_samples: int = 50, grid_size: int = 10, seed: int = 42):
        super().__init__()
        rng = np.random.RandomState(seed)
        self.grid_size = grid_size
        self.num_rules = len(self.RULES)

        self.inputs = []
        self.labels = []

        for _ in range(num_samples):
            # Random grid with values 0-9
            grid = rng.randint(0, 10, size=(grid_size, grid_size)).astype(np.float32)
            rule_idx = rng.randint(0, self.num_rules)
            transformed = self._apply_rule(grid, rule_idx)
            # Input: concatenation of original + transformed (200 dims)
            combined = np.concatenate([grid.flatten(), transformed.flatten()])
            self.inputs.append(combined)
            self.labels.append(rule_idx)

        self.inputs = torch.tensor(np.array(self.inputs), dtype=torch.float32)
        self.labels = torch.tensor(self.labels, dtype=torch.long)

    def _apply_rule(self, grid: np.ndarray, rule_idx: int) -> np.ndarray:
        rule = self.RULES[rule_idx]
        if rule == "rotate_90":
            return np.rot90(grid, 1)
        elif rule == "rotate_180":
            return np.rot90(grid, 2)
        elif rule == "reflect_h":
            return grid[::-1, :].copy()
        elif rule == "reflect_v":
            return grid[:, ::-1].copy()
        elif rule == "shift_right":
            return np.roll(grid, 1, axis=1)
        elif rule == "shift_down":
            return np.roll(grid, 1, axis=0)
        elif rule == "invert":
            return 9 - grid
        elif rule == "fill_border":
            out = grid.copy()
            out[0, :] = out[-1, :] = out[:, 0] = out[:, -1] = 5
            return out
        return grid

    def __len__(self): return len(self.labels)
    def __getitem__(self, idx): return self.inputs[idx], self.labels[idx]

    @property
    def input_dim(self): return self.grid_size * self.grid_size * 2
    @property
    def num_classes(self): return self.num_rules


# ======================================================================
# 2. GSM8K-Hard — Synthetic Multi-Step Arithmetic
# ======================================================================
class SyntheticGSM8KDataset(Dataset):
    """
    Synthetic multi-step arithmetic problems.

    Each problem is a chain of 3-5 arithmetic operations:
        a OP1 b OP2 c OP3 d = ?

    The model receives the operands and operators as a feature vector
    and must predict the result bucket (discretised into 10 classes).

    Multiple cognitive steps are needed to resolve the chain sequentially.
    """

    OPS = ['+', '-', '*']

    def __init__(self, num_samples: int = 200, seed: int = 42):
        super().__init__()
        rng = np.random.RandomState(seed)

        self.inputs = []
        self.labels = []

        for _ in range(num_samples):
            num_ops = rng.randint(3, 6)  # 3-5 operations
            operands = rng.randint(1, 20, size=num_ops + 1).astype(np.float32)
            ops = rng.randint(0, len(self.OPS), size=num_ops)

            # Compute result
            result = float(operands[0])
            for i in range(num_ops):
                op = self.OPS[ops[i]]
                val = float(operands[i + 1])
                if op == '+':
                    result += val
                elif op == '-':
                    result -= val
                elif op == '*':
                    result *= val

            # Encode: operands + one-hot ops → feature vector
            # Max 6 operands + 5 ops × 3 one-hot = 21 dims, pad to 32
            features = np.zeros(32, dtype=np.float32)
            for i, v in enumerate(operands):
                if i < 6:
                    features[i] = v / 20.0  # normalise
            for i, op_idx in enumerate(ops):
                if i < 5:
                    features[6 + i * 3 + op_idx] = 1.0

            # Classify result into 10 buckets
            label = int(np.clip((result + 100) / 50, 0, 9))

            self.inputs.append(features)
            self.labels.append(label)

        self.inputs = torch.tensor(np.array(self.inputs), dtype=torch.float32)
        self.labels = torch.tensor(self.labels, dtype=torch.long)

    def __len__(self): return len(self.labels)
    def __getitem__(self, idx): return self.inputs[idx], self.labels[idx]

    @property
    def input_dim(self): return 32
    @property
    def num_classes(self): return 10


# ======================================================================
# 3. HotpotQA / MuSiQue — Synthetic Multi-Hop Questions
# ======================================================================
class SyntheticMultiHopDataset(Dataset):
    """
    Synthetic multi-hop reasoning dataset.

    Each sample contains:
      - 3-5 "fact" vectors (simulating paragraphs)
      - A "query" vector indicating which facts to chain
      - Label: which final entity (out of 10) is the answer

    The model must attend to multiple facts in sequence to
    arrive at the correct answer — benefiting from T>1 cognitive steps.
    """

    def __init__(self, num_samples: int = 100, fact_dim: int = 16, num_hops: int = 3, seed: int = 42):
        super().__init__()
        rng = np.random.RandomState(seed)
        self.num_classes = 10

        self.inputs = []
        self.labels = []

        # Create 10 entity embeddings (ground truth)
        entity_embeds = rng.randn(10, fact_dim).astype(np.float32)

        for _ in range(num_samples):
            target_entity = rng.randint(0, 10)
            # Build a chain of facts that leads to target
            num_facts = rng.randint(3, 6)
            facts = rng.randn(num_facts, fact_dim).astype(np.float32) * 0.5

            # Plant breadcrumbs: each hop fact has similarity to next
            chain_length = min(num_hops, num_facts)
            for hop in range(chain_length):
                if hop == chain_length - 1:
                    # Final fact points to target entity
                    facts[hop] += entity_embeds[target_entity] * (0.5 + hop * 0.2)
                else:
                    # Intermediate: mix with next fact
                    next_idx = min(hop + 1, num_facts - 1)
                    facts[hop] += facts[next_idx] * 0.3

            # Query: encoded version of "what entity?"
            query = entity_embeds[target_entity] * 0.3 + rng.randn(fact_dim).astype(np.float32) * 0.2

            # Feature: flatten [query, all facts, padding]
            feature = np.zeros(fact_dim * 6, dtype=np.float32)  # query + 5 facts max
            feature[:fact_dim] = query
            for i in range(min(num_facts, 5)):
                feature[fact_dim * (i + 1): fact_dim * (i + 2)] = facts[i]

            self.inputs.append(feature)
            self.labels.append(target_entity)

        self.inputs = torch.tensor(np.array(self.inputs), dtype=torch.float32)
        self.labels = torch.tensor(self.labels, dtype=torch.long)

    def __len__(self): return len(self.labels)
    def __getitem__(self, idx): return self.inputs[idx], self.labels[idx]

    @property
    def input_dim(self): return 16 * 6
    @property
    def num_classes_prop(self): return 10


# ======================================================================
# 4. Blocksworld — Synthetic Planning Instances
# ======================================================================
class SyntheticBlocksworldDataset(Dataset):
    """
    Simplified Blocksworld planning dataset.

    Each instance: given initial block configuration (3-5 blocks),
    predict the sequence of moves needed (as a classification label
    representing the plan template).

    Encodes: initial state (block positions) + goal state → plan class.
    """

    def __init__(self, num_samples: int = 50, max_blocks: int = 5, seed: int = 42):
        super().__init__()
        rng = np.random.RandomState(seed)
        self.num_plan_classes = 8  # 8 plan templates

        self.inputs = []
        self.labels = []

        for _ in range(num_samples):
            num_blocks = rng.randint(3, max_blocks + 1)

            # Initial state: each block has a position (stack_id, height)
            # Encode as vector: [block1_stack, block1_height, ...]
            num_stacks = 3
            initial = np.zeros(max_blocks * 2, dtype=np.float32)
            goal = np.zeros(max_blocks * 2, dtype=np.float32)

            blocks_initial = list(range(num_blocks))
            rng.shuffle(blocks_initial)
            blocks_goal = list(range(num_blocks))
            rng.shuffle(blocks_goal)

            # Place blocks in stacks
            for i, b in enumerate(blocks_initial):
                stack_id = i % num_stacks
                height = i // num_stacks
                initial[b * 2] = stack_id / num_stacks
                initial[b * 2 + 1] = height / max_blocks

            for i, b in enumerate(blocks_goal):
                stack_id = i % num_stacks
                height = i // num_stacks
                goal[b * 2] = stack_id / num_stacks
                goal[b * 2 + 1] = height / max_blocks

            feature = np.concatenate([initial, goal])

            # Plan class: based on number of moves needed (simplified)
            diff = np.sum(np.abs(initial - goal))
            label = int(np.clip(diff * 4, 0, self.num_plan_classes - 1))

            self.inputs.append(feature)
            self.labels.append(label)

        self.inputs = torch.tensor(np.array(self.inputs), dtype=torch.float32)
        self.labels = torch.tensor(self.labels, dtype=torch.long)

    def __len__(self): return len(self.labels)
    def __getitem__(self, idx): return self.inputs[idx], self.labels[idx]

    @property
    def input_dim(self): return 5 * 2 * 2  # max_blocks * 2 * (initial + goal)
    @property
    def num_classes(self): return self.num_plan_classes


# ======================================================================
# Convenience: get all benchmarks
# ======================================================================
def get_all_benchmarks(seed: int = 42):
    """Return dict of benchmark_name → (dataset, input_dim, num_classes)."""
    arc = SyntheticARCDataset(num_samples=50, seed=seed)
    gsm = SyntheticGSM8KDataset(num_samples=200, seed=seed)
    hotpot = SyntheticMultiHopDataset(num_samples=100, seed=seed)
    blocks = SyntheticBlocksworldDataset(num_samples=50, seed=seed)

    return {
        "ARC-AGI (synthetic)": (arc, arc.input_dim, arc.num_classes),
        "GSM8K-Hard (synthetic)": (gsm, gsm.input_dim, gsm.num_classes),
        "HotpotQA (synthetic)": (hotpot, hotpot.input_dim, 10),
        "Blocksworld (synthetic)": (blocks, blocks.input_dim, blocks.num_classes),
    }


# ======================================================================
# Full dataset stubs (for --full flag on Colab)
# ======================================================================
def get_real_benchmarks():
    """
    Download and prepare real datasets. Requires significant compute/memory.
    Use only on Colab with GPU.
    """
    benchmarks = {}

    try:
        from datasets import load_dataset
        # GSM8K
        logger_msg = "Downloading GSM8K..."
        ds = load_dataset("gsm8k", "main", split="test[:200]")
        print(f"  GSM8K: loaded {len(ds)} samples (stub — full integration WIP)")
        # TODO: implement full GSM8K text → reasoning pipeline
    except Exception as e:
        print(f"  GSM8K download failed: {e}")

    try:
        from datasets import load_dataset
        # HotpotQA
        ds = load_dataset("hotpot_qa", "fullwiki", split="validation[:100]")
        print(f"  HotpotQA: loaded {len(ds)} samples (stub — full integration WIP)")
    except Exception as e:
        print(f"  HotpotQA download failed: {e}")

    print("\n  Note: Full benchmark integration requires pretrained LM backbone.")
    print("  Using synthetic benchmarks for local experiments.\n")
    return benchmarks


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--full", action="store_true", help="Download real datasets (Colab only)")
    args = parser.parse_args()

    if args.full:
        get_real_benchmarks()
    else:
        benchmarks = get_all_benchmarks()
        for name, (ds, inp_dim, n_cls) in benchmarks.items():
            print(f"{name}: {len(ds)} samples, input_dim={inp_dim}, classes={n_cls}")
