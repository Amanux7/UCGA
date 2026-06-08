import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ucga.ucga_model import UCGAModel
from experiments.experiment_phase6a_attention_suppression import (
    TaskData,
    compute_chain_metrics,
    load_babi_task2,
    make_synthetic_task,
    maybe_limit_task,
    set_seed,
    silent_forward,
    stack_attention,
)


def compute_grad_norm(parameters: Iterable[torch.nn.Parameter]) -> float:
    total_sq = 0.0
    for parameter in parameters:
        if parameter.grad is not None:
            total_sq += parameter.grad.detach().pow(2).sum().item()
    return total_sq ** 0.5


def finite_mean(values: List[float]) -> float:
    finite = [float(value) for value in values if not np.isnan(float(value))]
    return float(np.mean(finite)) if finite else float("nan")


def make_model(variant: str, task: TaskData, t_steps: int, state_dim: int) -> UCGAModel:
    shared = {
        "input_dim": state_dim,
        "state_dim": state_dim,
        "output_dim": task.vocab_size,
        "cognitive_steps": t_steps,
        "use_memory_attention": True,
    }
    if variant == "baseline_explicit_memory":
        return UCGAModel(**shared)
    if variant == "history_mean":
        return UCGAModel(
            **shared,
            use_retrieval_history=True,
            retrieval_history_strategy="mean",
        )
    if variant == "gru_evidence":
        return UCGAModel(
            **shared,
            use_evidence_accumulator=True,
        )
    raise ValueError(f"Unknown Phase 6C variant: {variant}")


def train_and_evaluate(
    task: TaskData,
    variant: str,
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

    model = make_model(variant, task, t_steps, state_dim).to(device)
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
    if variant == "history_mean":
        evidence_norms = meta["history_norms"]
        evidence_ratios = meta["history_ratios"]
    else:
        evidence_norms = meta["evidence_norms"]
        evidence_ratios = meta["evidence_ratios"]

    return {
        "task": task.name,
        "variant": variant,
        "T": t_steps,
        "seed": seed,
        "accuracy": accuracy,
        "grad_norm": last_grad_norm,
        "alpha": model.alpha.item(),
        "runtime_sec": runtime,
        "entropy_by_step": meta["entropy"],
        "q_delta_by_step": meta["q_deltas"],
        "retrieval_norm_by_step": meta["retrieval_norms"],
        "history_norm_by_step": meta["history_norms"],
        "history_ratio_by_step": meta["history_ratios"],
        "evidence_norm_by_step": evidence_norms,
        "evidence_ratio_by_step": evidence_ratios,
        "evidence_update_norm_by_step": meta["evidence_update_norms"],
        "slot_shift_by_step": meta["attention_slot_shift"],
        "attention": attention.numpy() if attention.numel() else np.empty((0, 0, 0)),
        **chain_metrics,
    }


def make_rows(result: Dict) -> Tuple[Dict, List[Dict]]:
    entropy = result["entropy_by_step"]
    q_delta = result["q_delta_by_step"]
    retrieval_norm = result["retrieval_norm_by_step"]
    evidence_norm = result["evidence_norm_by_step"]
    evidence_ratio = result["evidence_ratio_by_step"]
    evidence_update = result["evidence_update_norm_by_step"]
    slot_shift = result["slot_shift_by_step"]

    run_row = {
        "task": result["task"],
        "variant": result["variant"],
        "T": result["T"],
        "seed": result["seed"],
        "accuracy": result["accuracy"],
        "grad_norm": result["grad_norm"],
        "alpha": result["alpha"],
        "entropy_mean": finite_mean(entropy),
        "entropy_delta": (entropy[-1] - entropy[0]) if len(entropy) > 1 else 0.0,
        "q_delta_mean": finite_mean(q_delta),
        "q_delta_delta": (q_delta[-1] - q_delta[0]) if len(q_delta) > 1 else 0.0,
        "retrieval_norm_mean": finite_mean(retrieval_norm),
        "evidence_norm_mean": finite_mean(evidence_norm),
        "evidence_norm_delta": (evidence_norm[-1] - evidence_norm[0]) if len(evidence_norm) > 1 else 0.0,
        "evidence_ratio_mean": finite_mean(evidence_ratio),
        "evidence_ratio_delta": (evidence_ratio[-1] - evidence_ratio[0]) if len(evidence_ratio) > 1 else 0.0,
        "evidence_update_mean": finite_mean(evidence_update),
        "evidence_update_delta": (evidence_update[-1] - evidence_update[0]) if len(evidence_update) > 1 else 0.0,
        "slot_shift_mean": finite_mean(slot_shift[1:] if len(slot_shift) > 1 else slot_shift),
        "attention_coverage": result["attention_coverage"],
        "chain_full_accuracy": result["chain_full_accuracy"],
        "runtime_sec": result["runtime_sec"],
    }

    step_rows = []
    for step in range(result["T"]):
        step_rows.append(
            {
                "task": result["task"],
                "variant": result["variant"],
                "T": result["T"],
                "seed": result["seed"],
                "step": step + 1,
                "attention_entropy": entropy[step] if step < len(entropy) else float("nan"),
                "query_delta_norm": q_delta[step] if step < len(q_delta) else float("nan"),
                "retrieval_norm": retrieval_norm[step] if step < len(retrieval_norm) else float("nan"),
                "evidence_state_norm": evidence_norm[step] if step < len(evidence_norm) else float("nan"),
                "evidence_ratio": evidence_ratio[step] if step < len(evidence_ratio) else float("nan"),
                "evidence_update_magnitude": evidence_update[step] if step < len(evidence_update) else float("nan"),
                "alpha": result["alpha"],
                "slot_shift_l1": slot_shift[step] if step < len(slot_shift) else float("nan"),
                "grad_norm": result["grad_norm"],
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
    groups: Dict[Tuple[str, str, int], List[Dict]] = {}
    for row in run_rows:
        groups.setdefault((row["task"], row["variant"], row["T"]), []).append(row)

    fields = [
        "accuracy",
        "grad_norm",
        "alpha",
        "entropy_mean",
        "entropy_delta",
        "q_delta_mean",
        "q_delta_delta",
        "retrieval_norm_mean",
        "evidence_norm_mean",
        "evidence_norm_delta",
        "evidence_ratio_mean",
        "evidence_ratio_delta",
        "evidence_update_mean",
        "evidence_update_delta",
        "slot_shift_mean",
        "attention_coverage",
        "chain_full_accuracy",
        "runtime_sec",
    ]
    summary_rows = []
    for (task, variant, t_steps), rows in sorted(groups.items()):
        summary = {"task": task, "variant": variant, "T": t_steps, "n": len(rows)}
        for field in fields:
            values = [float(row[field]) for row in rows if not np.isnan(float(row[field]))]
            summary[f"{field}_mean"] = float(np.mean(values)) if values else float("nan")
            summary[f"{field}_std"] = float(np.std(values)) if values else float("nan")
        summary_rows.append(summary)
    return summary_rows


def format_float(value: float) -> str:
    return "nan" if np.isnan(float(value)) else f"{float(value):.4f}"


def write_report(output_dir: Path, summary_rows: List[Dict], smoke: bool) -> None:
    lines = [
        "# UCGA Phase 6C Gated Evidence Accumulation Report",
        "",
        "This experiment asks whether active GRU evidence integration creates iterative reasoning where passive retrieval history did not.",
        "",
    ]
    if smoke:
        lines.extend(
            [
                "Note: this report was generated with `--smoke`; numeric values validate the pipeline, not the full research conclusion.",
                "",
            ]
        )

    lines.extend(
        [
            "## Final Comparison Table",
            "",
            "| Task | Variant | T | Accuracy | Std | GradNorm | Alpha | Entropy | Evidence Ratio | Runtime |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in summary_rows:
        lines.append(
            "| {task} | {variant} | {T} | {acc} | {std} | {grad} | {alpha} | {entropy} | {ratio} | {runtime}s |".format(
                task=row["task"],
                variant=row["variant"],
                T=row["T"],
                acc=format_float(row["accuracy_mean"]),
                std=format_float(row["accuracy_std"]),
                grad=format_float(row["grad_norm_mean"]),
                alpha=format_float(row["alpha_mean"]),
                entropy=format_float(row["entropy_mean_mean"]),
                ratio=format_float(row["evidence_ratio_mean_mean"]),
                runtime=format_float(row["runtime_sec_mean"]),
            )
        )

    lines.extend(["", "## Diagnostics", ""])
    for task in sorted({row["task"] for row in summary_rows}):
        task_rows = [row for row in summary_rows if row["task"] == task]
        for variant in ["baseline_explicit_memory", "history_mean", "gru_evidence"]:
            rows = {row["T"]: row for row in task_rows if row["variant"] == variant}
            if 1 in rows and 3 in rows and 5 in rows:
                monotonic = rows[5]["accuracy_mean"] > rows[3]["accuracy_mean"] > rows[1]["accuracy_mean"]
                lines.append(
                    f"- {task} / {variant}: T=5 > T=3 > T=1 is {'observed' if monotonic else 'not observed'}."
                )
                lines.append(
                    f"- {task} / {variant}: entropy delta at T=5={format_float(rows[5]['entropy_delta_mean'])}, "
                    f"query delta change={format_float(rows[5]['q_delta_delta_mean'])}, "
                    f"evidence ratio={format_float(rows[5]['evidence_ratio_mean_mean'])}, "
                    f"evidence update change={format_float(rows[5]['evidence_update_delta_mean'])}, "
                    f"fact-chain score={format_float(rows[5]['chain_full_accuracy_mean'])}."
                )

        baseline = {row["T"]: row for row in task_rows if row["variant"] == "baseline_explicit_memory"}
        history = {row["T"]: row for row in task_rows if row["variant"] == "history_mean"}
        gru = {row["T"]: row for row in task_rows if row["variant"] == "gru_evidence"}
        if 5 in baseline and 5 in gru:
            lines.append(
                f"- {task}: gru_evidence T=5 vs baseline T=5 accuracy delta={gru[5]['accuracy_mean'] - baseline[5]['accuracy_mean']:.4f}."
            )
        if 5 in history and 5 in gru:
            lines.append(
                f"- {task}: gru_evidence T=5 vs history_mean T=5 accuracy delta={gru[5]['accuracy_mean'] - history[5]['accuracy_mean']:.4f}."
            )
        lines.append("")

    lines.extend(["## Failure Modes", ""])
    for task in sorted({row["task"] for row in summary_rows}):
        gru_rows = [row for row in summary_rows if row["task"] == task and row["variant"] == "gru_evidence" and row["T"] > 1]
        if not gru_rows:
            continue
        ratio = float(np.nanmean([row["evidence_ratio_mean_mean"] for row in gru_rows]))
        evidence_growth = float(np.nanmean([row["evidence_norm_delta_mean"] for row in gru_rows]))
        update_delta = float(np.nanmean([row["evidence_update_delta_mean"] for row in gru_rows]))
        slot_shift = float(np.nanmean([row["slot_shift_mean_mean"] for row in gru_rows]))
        chain = float(np.nanmean([row["chain_full_accuracy_mean"] for row in gru_rows]))
        lines.append(f"- {task}:")
        if slot_shift < 0.10:
            lines.append("  - A. Retrieval quality likely remains a bottleneck: attention movement is weak.")
        if ratio < 0.25 or evidence_growth < -0.25:
            lines.append("  - B. Evidence state may collapse or be underused.")
        elif ratio > 2.0:
            lines.append("  - C. Evidence state may dominate current retrieval and oversmooth the query.")
        else:
            lines.append("  - Evidence state is active, but activity alone is not proof of reasoning.")
        if update_delta >= 0.0:
            lines.append("  - Evidence updates are not stabilizing across recursive steps.")
        if chain < 0.20:
            lines.append("  - D. Memory representation remains insufficient for reliable fact chaining.")
        lines.append("")

    lines.extend(
        [
            "## Mechanistic Interpretation",
            "",
            "The GRU accumulator gives UCGA a learnable write/retain mechanism over retrieved evidence. If accuracy does not improve while evidence ratio is nonzero, the failure is not simple forgetting. It points to either low-quality retrievals, weak supervision for support-fact order, or an update function that cannot transform evidence state into task-relevant chain composition.",
            "",
            "## Phase 7 Recommendation",
            "",
            "Add supervised or contrastive support-fact alignment diagnostics and test a structured relational memory encoder. A useful Phase 7 should separate three mechanisms: retrieval correctness, evidence-state retention, and chain-composition accuracy.",
            "",
            "## Outputs",
            "",
            "- `phase6c_comparison_table.csv`: seed-aggregated final comparison table.",
            "- `phase6c_run_metrics.csv`: one row per task/variant/T/seed.",
            "- `phase6c_step_metrics.csv`: per-step entropy, query delta, retrieval norm, evidence norm, evidence ratio, and evidence update magnitude.",
            "- `phase6c_raw_results.json`: raw per-run diagnostics excluding full attention tensors.",
        ]
    )
    (output_dir / "phase6c_report.md").write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="UCGA Phase 6C GRU evidence accumulator experiment")
    parser.add_argument(
        "--variants",
        nargs="+",
        default=["baseline_explicit_memory", "history_mean", "gru_evidence"],
    )
    parser.add_argument("--T", nargs="+", type=int, default=[1, 3, 5])
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--synthetic-n", type=int, default=10000)
    parser.add_argument("--babi-train-limit", type=int, default=0)
    parser.add_argument("--babi-test-limit", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/phase6c"))
    parser.add_argument("--smoke", action="store_true", help="Run a compact end-to-end validation sweep.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.smoke:
        args.epochs = 1
        args.synthetic_n = 512
        args.babi_train_limit = 512
        args.babi_test_limit = 256

    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(1234)
    tasks = [
        make_synthetic_task(args.synthetic_n),
        maybe_limit_task(load_babi_task2(), args.babi_train_limit, args.babi_test_limit),
    ]

    run_rows = []
    step_rows = []
    raw_results = []

    for task in tasks:
        for variant in args.variants:
            for t_steps in args.T:
                for seed in range(args.seeds):
                    print(f"Phase 6C | task={task.name} | variant={variant} | T={t_steps} | seed={seed}")
                    result = train_and_evaluate(
                        task=task,
                        variant=variant,
                        t_steps=t_steps,
                        seed=seed,
                        epochs=args.epochs,
                        batch_size=args.batch_size,
                        device=device,
                    )
                    run_row, current_step_rows = make_rows(result)
                    run_rows.append(run_row)
                    step_rows.extend(current_step_rows)
                    result.pop("attention")
                    raw_results.append(result)

    summary_rows = summarize(run_rows)
    write_csv(args.output_dir / "phase6c_run_metrics.csv", run_rows)
    write_csv(args.output_dir / "phase6c_step_metrics.csv", step_rows)
    write_csv(args.output_dir / "phase6c_comparison_table.csv", summary_rows)
    (args.output_dir / "phase6c_raw_results.json").write_text(json.dumps(raw_results, indent=2), encoding="utf-8")
    write_report(args.output_dir, summary_rows, smoke=args.smoke)

    print("\nPhase 6C final comparison")
    for row in summary_rows:
        print(
            f"{row['task']:<24} {row['variant']:<25} T={row['T']} "
            f"acc={row['accuracy_mean']:.4f} entropy={row['entropy_mean_mean']:.4f} "
            f"evidence_ratio={row['evidence_ratio_mean_mean']:.4f} runtime={row['runtime_sec_mean']:.2f}s"
        )
    print(f"\nWrote outputs to {args.output_dir}")


if __name__ == "__main__":
    main()
