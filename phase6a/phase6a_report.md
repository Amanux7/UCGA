# UCGA Phase 6A Attention Suppression Report

This report evaluates cumulative attention suppression in the explicit-memory recursive UCGA path.

Note: this was generated with `--smoke`, so treat the numeric conclusions as pipeline validation only.

## Metrics Table

| Task | Lambda | T | Acc | Entropy Delta | Slot Shift | Coverage | Chain Step | Chain Full | Alpha | GradNorm |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| babi_task2 | 0.25 | 1 | 0.1297 | 0.0000 | 0.0000 | 1.0000 | 0.0594 | 0.0594 | 0.0987 | 3.4711 |
| babi_task2 | 0.25 | 3 | 0.1211 | -0.0044 | 0.0275 | 1.3711 | 0.0824 | 0.0000 | 0.1009 | 4.0988 |
| babi_task2 | 0.25 | 5 | 0.1430 | -0.0173 | 0.0293 | 1.6422 | 0.0852 | 0.0016 | 0.0995 | 4.5933 |
| babi_task2 | 0.50 | 1 | 0.1297 | 0.0000 | 0.0000 | 1.0000 | 0.0594 | 0.0594 | 0.0987 | 3.4711 |
| babi_task2 | 0.50 | 3 | 0.1211 | -0.0029 | 0.0273 | 1.3758 | 0.0824 | 0.0000 | 0.1009 | 4.0979 |
| babi_task2 | 0.50 | 5 | 0.1430 | -0.0138 | 0.0288 | 1.6531 | 0.0848 | 0.0016 | 0.0995 | 4.5927 |
| babi_task2 | 1.00 | 1 | 0.1297 | 0.0000 | 0.0000 | 1.0000 | 0.0594 | 0.0594 | 0.0987 | 3.4711 |
| babi_task2 | 1.00 | 3 | 0.1203 | -0.0002 | 0.0275 | 1.3906 | 0.0816 | 0.0000 | 0.1009 | 4.0973 |
| babi_task2 | 1.00 | 5 | 0.1430 | -0.0077 | 0.0281 | 1.6992 | 0.0863 | 0.0016 | 0.0995 | 4.5909 |
| hard_synthetic_multihop | 0.25 | 1 | 0.0621 | 0.0000 | 0.0000 | 1.0000 | 0.3825 | 0.3825 | 0.1001 | 1.2384 |
| hard_synthetic_multihop | 0.25 | 3 | 0.0621 | 0.0013 | 0.0272 | 1.2214 | 0.3092 | 0.1515 | 0.1002 | 1.2096 |
| hard_synthetic_multihop | 0.25 | 5 | 0.0563 | -0.0050 | 0.0262 | 1.4524 | 0.2491 | 0.1476 | 0.0995 | 1.3565 |
| hard_synthetic_multihop | 0.50 | 1 | 0.0621 | 0.0000 | 0.0000 | 1.0000 | 0.3825 | 0.3825 | 0.1001 | 1.2384 |
| hard_synthetic_multihop | 0.50 | 3 | 0.0621 | 0.0043 | 0.0289 | 1.2447 | 0.3076 | 0.1515 | 0.1002 | 1.2089 |
| hard_synthetic_multihop | 0.50 | 5 | 0.0563 | 0.0008 | 0.0264 | 1.4835 | 0.2475 | 0.1456 | 0.0995 | 1.3573 |
| hard_synthetic_multihop | 1.00 | 1 | 0.0621 | 0.0000 | 0.0000 | 1.0000 | 0.3825 | 0.3825 | 0.1001 | 1.2384 |
| hard_synthetic_multihop | 1.00 | 3 | 0.0621 | 0.0089 | 0.0340 | 1.2990 | 0.3110 | 0.1515 | 0.1002 | 1.2082 |
| hard_synthetic_multihop | 1.00 | 5 | 0.0602 | 0.0081 | 0.0281 | 1.5631 | 0.2499 | 0.1476 | 0.0995 | 1.3544 |

## Evaluation Questions

- babi_task2: entropy decrease across steps: no clear decrease.
- babi_task2: attention moves across memory slots: weak/no.
- babi_task2, lambda=0.25: T=5 vs T=1 accuracy delta = 0.0133.
- babi_task2, lambda=0.50: T=5 vs T=1 accuracy delta = 0.0133.
- babi_task2, lambda=1.00: T=5 vs T=1 accuracy delta = 0.0133.
- babi_task2: cumulative suppression creates fact chaining: not demonstrated.

- hard_synthetic_multihop: entropy decrease across steps: no clear decrease.
- hard_synthetic_multihop: attention moves across memory slots: weak/no.
- hard_synthetic_multihop, lambda=0.25: T=5 vs T=1 accuracy delta = -0.0058.
- hard_synthetic_multihop, lambda=0.50: T=5 vs T=1 accuracy delta = -0.0058.
- hard_synthetic_multihop, lambda=1.00: T=5 vs T=1 accuracy delta = -0.0019.
- hard_synthetic_multihop: cumulative suppression creates fact chaining: not demonstrated.

## Outputs

- `phase6a_run_metrics.csv`: one row per task/lambda/T/seed.
- `phase6a_step_metrics.csv`: one row per recursive step with entropy, query delta, alpha, GradNorm, and slot shift.
- `phase6a_summary.csv`: seed-aggregated metrics table.
- `attention_heatmap_overview.png` and `heatmaps/*.png`: step-wise attention heatmaps.
- `phase6a_raw_results.json`: raw metadata excluding full per-sample attention tensors.
