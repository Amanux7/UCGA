import argparse
import contextlib
import csv
import io
import json
import os
import re
import sys
import tarfile
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ucga.ucga_model import UCGAModel


@dataclass
class TaskData:
    name: str
    train_facts: torch.Tensor
    train_query: torch.Tensor
    train_y: torch.Tensor
    test_facts: torch.Tensor
    test_query: torch.Tensor
    test_y: torch.Tensor
    test_support_paths: torch.Tensor
    vocab_size: int
    fact_width: int
    query_width: int


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def compute_grad_norm(parameters: Iterable[torch.nn.Parameter]) -> float:
    total_sq = 0.0
    for parameter in parameters:
        if parameter.grad is not None:
            total_sq += parameter.grad.detach().pow(2).sum().item()
    return total_sq ** 0.5


def silent_forward(model: nn.Module, *args, **kwargs):
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        return model(*args, **kwargs)


def generate_hard_synthetic_multihop(
    n_samples: int = 10000,
    vocab_size: int = 20,
    max_hops: int = 5,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    facts = torch.zeros(n_samples, max_hops, 3, dtype=torch.long)
    query = torch.zeros(n_samples, 3, dtype=torch.long)
    targets = torch.zeros(n_samples, dtype=torch.long)
    support_paths = torch.full((n_samples, max_hops), -1, dtype=torch.long)

    for i in range(n_samples):
        num_hops = np.random.randint(1, max_hops + 1)
        entities = np.random.choice(range(3, vocab_size), num_hops + 1, replace=False)

        ordered_facts = []
        for hop in range(num_hops):
            ordered_facts.append((1, int(entities[hop]), int(entities[hop + 1])))

        permutation = torch.randperm(num_hops)
        for shuffled_pos, original_pos in enumerate(permutation.tolist()):
            facts[i, shuffled_pos] = torch.tensor(ordered_facts[original_pos])
            support_paths[i, original_pos] = shuffled_pos

        query[i] = torch.tensor([2, int(entities[0]), 0])
        targets[i] = int(entities[-1])

    return facts, query, targets, support_paths


def make_synthetic_task(n_samples: int, test_fraction: float = 0.2) -> TaskData:
    facts, query, targets, supports = generate_hard_synthetic_multihop(n_samples=n_samples)
    split = int(n_samples * (1.0 - test_fraction))
    return TaskData(
        name="hard_synthetic_multihop",
        train_facts=facts[:split],
        train_query=query[:split],
        train_y=targets[:split],
        test_facts=facts[split:],
        test_query=query[split:],
        test_y=targets[split:],
        test_support_paths=supports[split:],
        vocab_size=20,
        fact_width=3,
        query_width=3,
    )


def tokenize(sentence: str) -> List[str]:
    return [token.strip() for token in re.split(r"(\W+)", sentence) if token.strip()]


def parse_babi_stories(lines: List[str]):
    data = []
    story = []
    for raw_line in lines:
        raw_line = raw_line.strip()
        if not raw_line:
            continue
        nid_text, line = raw_line.split(" ", 1)
        nid = int(nid_text)
        if nid == 1:
            story = []
        if "\t" in line:
            question, answer, supporting = line.split("\t")
            supports = [int(item) for item in supporting.split() if item]
            data.append(([item for item in story if item], tokenize(question), answer, supports))
            story.append(None)
        else:
            story.append((nid, tokenize(line)))
    return data


def build_vocab(data) -> Dict[str, int]:
    vocab = {"<PAD>": 0, "<UNK>": 1}
    for story, question, answer, _supports in data:
        for _nid, sentence in story:
            for word in sentence:
                if word not in vocab:
                    vocab[word] = len(vocab)
        for word in question:
            if word not in vocab:
                vocab[word] = len(vocab)
        if answer not in vocab:
            vocab[answer] = len(vocab)
    return vocab


def vectorize_babi(data, vocab, max_story_len: int, max_sentence_len: int, max_supports: int = 2):
    facts = np.zeros((len(data), max_story_len, max_sentence_len), dtype=np.int64)
    query = np.zeros((len(data), max_sentence_len), dtype=np.int64)
    targets = np.zeros((len(data),), dtype=np.int64)
    support_paths = np.full((len(data), max_supports), -1, dtype=np.int64)

    for i, (story, question, answer, supports) in enumerate(data):
        truncated_story = story[-max_story_len:]
        nid_to_position = {nid: position for position, (nid, _sentence) in enumerate(truncated_story)}

        for j, (_nid, sentence) in enumerate(truncated_story):
            for k, word in enumerate(sentence[:max_sentence_len]):
                facts[i, j, k] = vocab.get(word, 1)
        for k, word in enumerate(question[:max_sentence_len]):
            query[i, k] = vocab.get(word, 1)
        for k, support_nid in enumerate(supports[:max_supports]):
            support_paths[i, k] = nid_to_position.get(support_nid, -1)
        targets[i] = vocab.get(answer, 1)

    return (
        torch.tensor(facts),
        torch.tensor(query),
        torch.tensor(targets),
        torch.tensor(support_paths),
    )


def load_babi_task2(data_dir: str = "./data/babi") -> TaskData:
    os.makedirs(data_dir, exist_ok=True)
    tar_path = os.path.join(data_dir, "babi.tar.gz")
    if not os.path.exists(tar_path):
        url = "https://s3.amazonaws.com/text-datasets/babi_tasks_1-20_v1-2.tar.gz"
        print(f"Downloading bAbI dataset from {url}...")
        urllib.request.urlretrieve(url, tar_path)

    train_lines = []
    test_lines = []
    with tarfile.open(tar_path, "r:gz") as tar:
        for member in tar.getmembers():
            if "en/qa2_two-supporting-facts_train.txt" in member.name:
                train_lines = tar.extractfile(member).read().decode("utf-8").splitlines()
            elif "en/qa2_two-supporting-facts_test.txt" in member.name:
                test_lines = tar.extractfile(member).read().decode("utf-8").splitlines()

    if not train_lines or not test_lines:
        raise RuntimeError("Could not find bAbI Task 2 train/test files in archive.")

    train_data = parse_babi_stories(train_lines)
    test_data = parse_babi_stories(test_lines)
    vocab = build_vocab(train_data + test_data)
    max_story_len = min(max(len(story) for story, _question, _answer, _supports in train_data + test_data), 20)
    max_sentence_len = max(
        [len(sentence) for story, _question, _answer, _supports in train_data + test_data for _nid, sentence in story]
        + [len(question) for _story, question, _answer, _supports in train_data + test_data]
    )

    train_facts, train_query, train_y, _train_supports = vectorize_babi(
        train_data, vocab, max_story_len, max_sentence_len
    )
    test_facts, test_query, test_y, test_supports = vectorize_babi(
        test_data, vocab, max_story_len, max_sentence_len
    )
    return TaskData(
        name="babi_task2",
        train_facts=train_facts,
        train_query=train_query,
        train_y=train_y,
        test_facts=test_facts,
        test_query=test_query,
        test_y=test_y,
        test_support_paths=test_supports,
        vocab_size=len(vocab),
        fact_width=max_sentence_len,
        query_width=max_sentence_len,
    )


def stack_attention(meta: Dict) -> torch.Tensor:
    if "attentions" not in meta or not meta["attentions"]:
        return torch.empty(0)
    return torch.stack([item.squeeze(1) for item in meta["attentions"]], dim=1)


def compute_chain_metrics(attention: torch.Tensor, support_paths: torch.Tensor) -> Dict:
    if attention.numel() == 0:
        return {
            "attention_coverage": float("nan"),
            "chain_step_accuracy": float("nan"),
            "chain_full_accuracy": float("nan"),
            "per_step_chain_accuracy": [],
        }

    top_slots = attention.argmax(dim=-1)
    support_paths = support_paths.cpu()
    valid_support_count = (support_paths >= 0).sum(dim=1)
    coverage = torch.tensor([len(torch.unique(row)) for row in top_slots]).float().mean().item()

    per_step = []
    step_correct_values = []
    for step in range(top_slots.size(1)):
        if step >= support_paths.size(1):
            per_step.append(float("nan"))
            continue
        valid = support_paths[:, step] >= 0
        if valid.any():
            correct = (top_slots[valid, step] == support_paths[valid, step]).float()
            value = correct.mean().item()
            per_step.append(value)
            step_correct_values.append(value)
        else:
            per_step.append(float("nan"))

    full_scores = []
    for sample_idx in range(top_slots.size(0)):
        comparable_steps = min(int(valid_support_count[sample_idx].item()), top_slots.size(1))
        if comparable_steps <= 0:
            continue
        expected = support_paths[sample_idx, :comparable_steps]
        observed = top_slots[sample_idx, :comparable_steps]
        full_scores.append(float(torch.equal(observed, expected)))

    return {
        "attention_coverage": coverage,
        "chain_step_accuracy": float(np.mean(step_correct_values)) if step_correct_values else float("nan"),
        "chain_full_accuracy": float(np.mean(full_scores)) if full_scores else float("nan"),
        "per_step_chain_accuracy": per_step,
    }


def train_and_evaluate(
    task: TaskData,
    lambda_penalty: float,
    t_steps: int,
    seed: int,
    epochs: int,
    batch_size: int,
    device: torch.device,
) -> Dict:
    set_seed(seed)
    state_dim = 64
    embed_dim = 16

    train_ds = TensorDataset(task.train_facts.to(device), task.train_query.to(device), task.train_y.to(device))
    loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    model = UCGAModel(
        input_dim=state_dim,
        state_dim=state_dim,
        output_dim=task.vocab_size,
        cognitive_steps=t_steps,
        use_memory_attention=True,
        attention_suppression_lambda=lambda_penalty,
    ).to(device)
    embedding = nn.Embedding(task.vocab_size, embed_dim, padding_idx=0).to(device)
    fact_proj = nn.Linear(task.fact_width * embed_dim, state_dim).to(device)
    query_proj = nn.Linear(task.query_width * embed_dim, state_dim).to(device)

    parameters = list(model.parameters()) + list(embedding.parameters()) + list(fact_proj.parameters()) + list(query_proj.parameters())
    optimizer = torch.optim.Adam(parameters, lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    last_grad_norm = 0.0
    started_at = time.time()
    for _epoch in range(epochs):
        model.train()
        for batch_facts, batch_query, batch_y in loader:
            optimizer.zero_grad()
            fact_emb = embedding(batch_facts).reshape(batch_facts.size(0), batch_facts.size(1), -1)
            memory = fact_proj(fact_emb)
            query_emb = embedding(batch_query).reshape(batch_query.size(0), -1)
            query = query_proj(query_emb)
            output, _meta = silent_forward(model, query, M=memory, return_meta=True)
            loss = criterion(output, batch_y)
            loss.backward()
            last_grad_norm = compute_grad_norm(parameters)
            optimizer.step()
    runtime = time.time() - started_at

    model.eval()
    with torch.no_grad():
        test_facts = task.test_facts.to(device)
        test_query = task.test_query.to(device)
        test_y = task.test_y.to(device)
        fact_emb = embedding(test_facts).reshape(test_facts.size(0), test_facts.size(1), -1)
        memory = fact_proj(fact_emb)
        query_emb = embedding(test_query).reshape(test_query.size(0), -1)
        query = query_proj(query_emb)
        output, meta = silent_forward(model, query, M=memory, return_meta=True, return_attention=True)
        accuracy = (output.argmax(dim=1) == test_y).float().mean().item()

    attention = stack_attention(meta)
    chain_metrics = compute_chain_metrics(attention, task.test_support_paths)
    entropy = meta["entropy"]
    q_deltas = meta["q_deltas"]
    slot_shift = meta["attention_slot_shift"]

    return {
        "task": task.name,
        "lambda_penalty": lambda_penalty,
        "T": t_steps,
        "seed": seed,
        "accuracy": accuracy,
        "grad_norm": last_grad_norm,
        "alpha": model.alpha.item(),
        "runtime_sec": runtime,
        "entropy_by_step": entropy,
        "q_delta_by_step": q_deltas,
        "slot_shift_by_step": slot_shift,
        "attention": attention.numpy() if attention.numel() else np.empty((0, 0, 0)),
        **chain_metrics,
    }


def finite_mean(values: List[float]) -> float:
    finite = [value for value in values if not np.isnan(value)]
    return float(np.mean(finite)) if finite else float("nan")


def make_rows(result: Dict) -> Tuple[Dict, List[Dict]]:
    entropy = result["entropy_by_step"]
    q_delta = result["q_delta_by_step"]
    slot_shift = result["slot_shift_by_step"]
    per_step_chain = result["per_step_chain_accuracy"]
    run_row = {
        "task": result["task"],
        "lambda_penalty": result["lambda_penalty"],
        "T": result["T"],
        "seed": result["seed"],
        "accuracy": result["accuracy"],
        "entropy_first": entropy[0] if entropy else float("nan"),
        "entropy_last": entropy[-1] if entropy else float("nan"),
        "entropy_delta": (entropy[-1] - entropy[0]) if len(entropy) > 1 else 0.0,
        "entropy_mean": finite_mean(entropy),
        "q_delta_mean": finite_mean(q_delta),
        "slot_shift_mean": finite_mean(slot_shift[1:] if len(slot_shift) > 1 else slot_shift),
        "attention_coverage": result["attention_coverage"],
        "chain_step_accuracy": result["chain_step_accuracy"],
        "chain_full_accuracy": result["chain_full_accuracy"],
        "alpha": result["alpha"],
        "grad_norm": result["grad_norm"],
        "runtime_sec": result["runtime_sec"],
    }
    step_rows = []
    for step in range(result["T"]):
        step_rows.append(
            {
                "task": result["task"],
                "lambda_penalty": result["lambda_penalty"],
                "T": result["T"],
                "seed": result["seed"],
                "step": step + 1,
                "attention_entropy": entropy[step] if step < len(entropy) else float("nan"),
                "query_delta_norm": q_delta[step] if step < len(q_delta) else float("nan"),
                "alpha": result["alpha"],
                "grad_norm": result["grad_norm"],
                "slot_shift_l1": slot_shift[step] if step < len(slot_shift) else float("nan"),
                "chain_step_accuracy": per_step_chain[step] if step < len(per_step_chain) else float("nan"),
            }
        )
    return run_row, step_rows


def write_csv(path: Path, rows: List[Dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def summarize(run_rows: List[Dict]) -> List[Dict]:
    groups: Dict[Tuple, List[Dict]] = {}
    for row in run_rows:
        key = (row["task"], row["lambda_penalty"], row["T"])
        groups.setdefault(key, []).append(row)

    summary_rows = []
    for (task, lambda_penalty, t_steps), rows in sorted(groups.items()):
        summary = {"task": task, "lambda_penalty": lambda_penalty, "T": t_steps, "n": len(rows)}
        for field in [
            "accuracy",
            "entropy_delta",
            "entropy_mean",
            "q_delta_mean",
            "slot_shift_mean",
            "attention_coverage",
            "chain_step_accuracy",
            "chain_full_accuracy",
            "alpha",
            "grad_norm",
            "runtime_sec",
        ]:
            values = [float(row[field]) for row in rows if not np.isnan(float(row[field]))]
            summary[f"{field}_mean"] = float(np.mean(values)) if values else float("nan")
            summary[f"{field}_std"] = float(np.std(values)) if values else float("nan")
        summary_rows.append(summary)
    return summary_rows


def plot_heatmaps(heatmaps: Dict[Tuple[str, float, int], List[np.ndarray]], output_dir: Path) -> None:
    heatmap_dir = output_dir / "heatmaps"
    heatmap_dir.mkdir(parents=True, exist_ok=True)
    averaged = {key: np.mean(items, axis=0) for key, items in heatmaps.items() if items}

    for (task, lambda_penalty, t_steps), heatmap in averaged.items():
        fig, ax = plt.subplots(figsize=(7, 3.5))
        im = ax.imshow(heatmap, aspect="auto", vmin=0.0, vmax=max(0.05, float(np.nanmax(heatmap))))
        ax.set_title(f"{task} | lambda={lambda_penalty} | T={t_steps}")
        ax.set_xlabel("Memory slot")
        ax.set_ylabel("Recursive step")
        ax.set_yticks(range(t_steps))
        ax.set_yticklabels([str(i + 1) for i in range(t_steps)])
        fig.colorbar(im, ax=ax, label="Mean attention")
        fig.tight_layout()
        fig.savefig(heatmap_dir / f"{task}_lambda{lambda_penalty}_T{t_steps}.png", dpi=160)
        plt.close(fig)

    overview_keys = [key for key in sorted(averaged) if key[2] == 5]
    if not overview_keys:
        overview_keys = sorted(averaged)
    if overview_keys:
        cols = min(3, len(overview_keys))
        rows = int(np.ceil(len(overview_keys) / cols))
        fig, axes = plt.subplots(rows, cols, figsize=(5.2 * cols, 3.4 * rows), squeeze=False)
        for ax in axes.flat:
            ax.axis("off")
        for ax, key in zip(axes.flat, overview_keys):
            task, lambda_penalty, t_steps = key
            ax.axis("on")
            heatmap = averaged[key]
            ax.imshow(heatmap, aspect="auto", vmin=0.0, vmax=max(0.05, float(np.nanmax(heatmap))))
            ax.set_title(f"{task}\nlambda={lambda_penalty}, T={t_steps}")
            ax.set_xlabel("Slot")
            ax.set_ylabel("Step")
        fig.tight_layout()
        fig.savefig(output_dir / "attention_heatmap_overview.png", dpi=180)
        plt.close(fig)


def format_float(value: float) -> str:
    return "nan" if np.isnan(float(value)) else f"{float(value):.4f}"


def write_report(output_dir: Path, summary_rows: List[Dict], smoke: bool) -> None:
    by_key = {(row["task"], row["lambda_penalty"], row["T"]): row for row in summary_rows}
    lines = [
        "# UCGA Phase 6A Attention Suppression Report",
        "",
        "This report evaluates cumulative attention suppression in the explicit-memory recursive UCGA path.",
        "",
    ]
    if smoke:
        lines.extend(
            [
                "Note: this was generated with `--smoke`, so treat the numeric conclusions as pipeline validation only.",
                "",
            ]
        )

    lines.extend(
        [
            "## Metrics Table",
            "",
            "| Task | Lambda | T | Acc | Entropy Delta | Slot Shift | Coverage | Chain Step | Chain Full | Alpha | GradNorm |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in summary_rows:
        lines.append(
            "| {task} | {lam:.2f} | {T} | {acc} | {ent} | {shift} | {coverage} | {chain_step} | {chain_full} | {alpha} | {grad} |".format(
                task=row["task"],
                lam=row["lambda_penalty"],
                T=row["T"],
                acc=format_float(row["accuracy_mean"]),
                ent=format_float(row["entropy_delta_mean"]),
                shift=format_float(row["slot_shift_mean_mean"]),
                coverage=format_float(row["attention_coverage_mean"]),
                chain_step=format_float(row["chain_step_accuracy_mean"]),
                chain_full=format_float(row["chain_full_accuracy_mean"]),
                alpha=format_float(row["alpha_mean"]),
                grad=format_float(row["grad_norm_mean"]),
            )
        )

    lines.extend(["", "## Evaluation Questions", ""])
    for task in sorted({row["task"] for row in summary_rows}):
        entropy_deltas = [row["entropy_delta_mean"] for row in summary_rows if row["task"] == task and row["T"] > 1]
        shifts = [row["slot_shift_mean_mean"] for row in summary_rows if row["task"] == task and row["T"] > 1]
        entropy_answer = "yes" if entropy_deltas and np.nanmean(entropy_deltas) < -0.02 else "no clear decrease"
        shift_answer = "yes" if shifts and np.nanmean(shifts) > 0.10 else "weak/no"
        lines.append(f"- {task}: entropy decrease across steps: {entropy_answer}.")
        lines.append(f"- {task}: attention moves across memory slots: {shift_answer}.")

        for lambda_penalty in sorted({row["lambda_penalty"] for row in summary_rows if row["task"] == task}):
            t1 = by_key.get((task, lambda_penalty, 1))
            t5 = by_key.get((task, lambda_penalty, 5))
            if t1 and t5:
                delta = t5["accuracy_mean"] - t1["accuracy_mean"]
                lines.append(
                    f"- {task}, lambda={lambda_penalty:.2f}: T=5 vs T=1 accuracy delta = {delta:.4f}."
                )

        chain_scores = [
            row["chain_full_accuracy_mean"]
            for row in summary_rows
            if row["task"] == task and row["T"] > 1
        ]
        chain_answer = "yes" if chain_scores and np.nanmean(chain_scores) > 0.20 else "not demonstrated"
        lines.append(f"- {task}: cumulative suppression creates fact chaining: {chain_answer}.")
        lines.append("")

    lines.extend(
        [
            "## Outputs",
            "",
            "- `phase6a_run_metrics.csv`: one row per task/lambda/T/seed.",
            "- `phase6a_step_metrics.csv`: one row per recursive step with entropy, query delta, alpha, GradNorm, and slot shift.",
            "- `phase6a_summary.csv`: seed-aggregated metrics table.",
            "- `attention_heatmap_overview.png` and `heatmaps/*.png`: step-wise attention heatmaps.",
            "- `phase6a_raw_results.json`: raw metadata excluding full per-sample attention tensors.",
        ],
    )
    (output_dir / "phase6a_report.md").write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="UCGA Phase 6A cumulative attention suppression experiment")
    parser.add_argument("--lambdas", nargs="+", type=float, default=[0.25, 0.50, 1.00])
    parser.add_argument("--T", nargs="+", type=int, default=[1, 3, 5])
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--synthetic-n", type=int, default=10000)
    parser.add_argument("--babi-train-limit", type=int, default=0)
    parser.add_argument("--babi-test-limit", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/phase6a"))
    parser.add_argument("--smoke", action="store_true", help="Run a small end-to-end validation sweep.")
    return parser.parse_args()


def maybe_limit_task(task: TaskData, train_limit: int, test_limit: int) -> TaskData:
    train_slice = slice(None if train_limit <= 0 else train_limit)
    test_slice = slice(None if test_limit <= 0 else test_limit)
    return TaskData(
        name=task.name,
        train_facts=task.train_facts[train_slice],
        train_query=task.train_query[train_slice],
        train_y=task.train_y[train_slice],
        test_facts=task.test_facts[test_slice],
        test_query=task.test_query[test_slice],
        test_y=task.test_y[test_slice],
        test_support_paths=task.test_support_paths[test_slice],
        vocab_size=task.vocab_size,
        fact_width=task.fact_width,
        query_width=task.query_width,
    )


def main() -> None:
    args = parse_args()
    if args.smoke:
        args.epochs = 1
        args.synthetic_n = 512
        args.babi_train_limit = 512
        args.babi_test_limit = 256

    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    synthetic_task = make_synthetic_task(args.synthetic_n)
    babi_task = maybe_limit_task(load_babi_task2(), args.babi_train_limit, args.babi_test_limit)
    tasks = [synthetic_task, babi_task]

    run_rows = []
    step_rows = []
    raw_results = []
    heatmaps: Dict[Tuple[str, float, int], List[np.ndarray]] = {}

    for task in tasks:
        for lambda_penalty in args.lambdas:
            for t_steps in args.T:
                for seed in range(args.seeds):
                    print(
                        f"Phase 6A | task={task.name} | lambda={lambda_penalty:.2f} | T={t_steps} | seed={seed}"
                    )
                    result = train_and_evaluate(
                        task=task,
                        lambda_penalty=lambda_penalty,
                        t_steps=t_steps,
                        seed=seed,
                        epochs=args.epochs,
                        batch_size=args.batch_size,
                        device=device,
                    )
                    run_row, current_step_rows = make_rows(result)
                    run_rows.append(run_row)
                    step_rows.extend(current_step_rows)

                    attention = result.pop("attention")
                    if attention.size:
                        heatmaps.setdefault((task.name, lambda_penalty, t_steps), []).append(attention.mean(axis=0))
                    raw_results.append(result)

    summary_rows = summarize(run_rows)
    write_csv(args.output_dir / "phase6a_run_metrics.csv", run_rows)
    write_csv(args.output_dir / "phase6a_step_metrics.csv", step_rows)
    write_csv(args.output_dir / "phase6a_summary.csv", summary_rows)
    (args.output_dir / "phase6a_raw_results.json").write_text(json.dumps(raw_results, indent=2), encoding="utf-8")
    plot_heatmaps(heatmaps, args.output_dir)
    write_report(args.output_dir, summary_rows, smoke=args.smoke)

    print("\nPhase 6A summary")
    for row in summary_rows:
        print(
            f"{row['task']:<24} lambda={row['lambda_penalty']:.2f} T={row['T']} "
            f"acc={row['accuracy_mean']:.4f} entropy_delta={row['entropy_delta_mean']:.4f} "
            f"shift={row['slot_shift_mean_mean']:.4f} chain={row['chain_full_accuracy_mean']:.4f}"
        )
    print(f"\nWrote outputs to {args.output_dir}")


if __name__ == "__main__":
    main()
