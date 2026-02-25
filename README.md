# Unified Cognitive Graph Architecture (UCGA)

> **A novel graph-based approach to Artificial General Intelligence**

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.8+](https://img.shields.io/badge/Python-3.8+-green.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org)

**Author:** Aman Singh — Founder, UCGA Research Initiative

---

## Abstract

This repository introduces the **Unified Cognitive Graph Architecture (UCGA)**, a novel graph-native cognitive framework designed to address fundamental limitations of traditional neural architectures in reasoning, memory integration, and adaptive intelligence. UCGA models intelligence as recursive interaction between specialised cognitive nodes connected through adaptive weighted edges, incorporating persistent attention-based memory, self-evaluation, and conditional error correction.

Unlike traditional architectures that process information uniformly through stacked layers, UCGA explicitly separates cognitive functions into distinct, differentiable nodes—bridging the gap between narrow deep learning systems and the flexible, self-correcting cognition observed in biological intelligence.

### Key Innovations (Breakthroughs)

- **AGI Architecture Layers:** 7 core intelligence layers (Perception, Cognitive Graph, Working Memory, Reasoning, Global Workspace, Executive Controller, Action) connected as a dynamic cognitive system.
- **Recursive Cognitive Refinement:** A cognitive loop that iterates *T* times, progressively refining understanding and "thinking longer" about hard problems.
- **Persistent Differentiable Memory:** Attention-based cross-episode memory bank retaining long-term cognitive store rather than just within-window attention.
- **Self-Evaluation & Correction:** A built-in critic that triggers corrective feedback loops when confidence drops below learned thresholds.
- **Non-saturating Cognition:** Discovered that replacing `tanh` with `GELU + LayerNorm` completely unblocks gradient flow across deep recursive steps, preventing vanishing gradients in recurrent graph traversals.
- **Multimodal Learning:** Empirically validated on real-world benchmarks achieving **90.5%** accuracy on AG News (NLP) and **60.0%** on CIFAR-10 (Vision), establishing broad generalisation.

---

## Architecture & Mathematical Formulation

UCGA models intelligence as a 5-tuple: $\mathcal{G} = (\mathcal{V},\; \mathcal{E},\; \mathbf{W},\; \mathcal{S},\; \mathbf{M})$

- $\mathcal{V}$: Intelligence layers (Perception, Cognitive Graph, Working Memory, Reasoning, Global Workspace, Executive Controller, Action).
- $\mathcal{E}$: Directed edges encoding information flow.
- $\mathbf{W}$: Learnable weight matrices.
- $\mathcal{S}$: Internal cognitive state vectors.
- $\mathbf{M}$: Persistent external memory bank.

### AGI Cognitive Cycle

```text
1. Perceive environment
2. Update working memory
3. Activate cognitive graph
4. Perform reasoning
5. Generate goals
6. Plan actions
7. Execute actions
8. Learn and update knowledge graph
```

### Core Equations

**State Update:**
$$v_i^{(t+1)} = \mathrm{LayerNorm}\!\Bigl(\mathrm{GELU}\bigl(W_i \cdot \sum_{j \in \mathcal{N}(i)} v_j^{(t)} + b_i\bigr)\Bigr)$$

**Memory Retrieval (Attention):**
$$\alpha_k = \mathrm{softmax}\!\Bigl(\frac{Q \cdot K_k^\top}{\sqrt{d}}\Bigr), \quad r = \sum_k \alpha_k V_k$$

**Evaluation Confidence:**
$$c = \sigma\!\bigl(W_c \cdot h_{\text{eval}} + b_c\bigr) \in [0, 1]$$

**Conditional Gated Correction:**
$$g = \sigma(W_g \cdot [h_{\text{plan}}; h_{\text{eval}}])$$
$$h_{\text{corrected}} = g \odot f(h_{\text{plan}}, h_{\text{eval}}) + (1 - g) \odot h_{\text{plan}}$$

---

## AGI Development Stage & Active Efforts

UCGA is now transitioning from Cognitive Architecture Prototype to AGI-capable Cognitive System.

**Next milestones:**
- Autonomous goal generation
- Continuous self-learning
- Scientific discovery automation
- Real-world deployment

## Future Work and Roadmap

Future research will explore scaling UCGA toward full AGI capability:
- **Transformer-based Reasoning Node:** Replacing MLP-based reasoning with multi-head self-attention for complex compositional reasoning sequences.
- **Learned End-to-End Encoding:** Evolving encoders to dynamically learn representations jointly through the cognitive loop.
- **Multimodal Fusion at Scale:** Integrating vision, language, and audio streams simultaneously within the cognitive graph on web-scale datasets.
- **Autonomous Reasoning Agents:** Leveraging the architecture for multi-step planning and real-world long-horizon decision-making.
- **Dynamic Graph Topology:** Allowing the cognitive graph structure to natively grow new neuro-computational nodes and connections based on task demands.
- **Large-Scale Benchmarks:** Expanding training limits to ImageNet, full text corpora, and multi-task setups via GPU distributed training.

---

## Project Structure

```text
UCGA/
├── README.md                          # This file
├── paper/                             # arXiv-ready research paper
│   └── main.tex
├── ucga/                              # Core architecture implementation
│   ├── nodes/                         # Cognitive processing nodes (Perception, Reasoning, etc.)
│   ├── memory/                        # Persistent, working, and episodic memory hierarchy
│   ├── encoders/                      # Pluggable Text, Image, Audio, Vector encoders
│   ├── distributed_graph.py           # Distributed message passing graph
│   ├── lifelong_learner.py            # Elastic Weight Consolidation (EWC)
│   ├── adaptive_topology.py           # Self-modifying structure
│   └── ucga_model.py                  # Main architectural orchestrator
├── training/                          # Optimised scaling & curriculum training scripts
├── agents/                            # PPO RL agents and cognitive environments
├── experiments/                       # Validation scripts (ROS, LOS, GIB)
└── utils/                             # Intelligence dashboard, metrics, and loggers
```

## Quick Start
```bash
git clone https://github.com/your-username/UCGA.git
cd UCGA
pip install -r requirements.txt
```

### Training & Validation
To run the current optimized scaling phase:
```bash
python training/train_optimized.py
```
To validate intelligence metrics:
```bash
python utils/intelligence_dashboard.py
```

---

## Implementation Status

| Capability | Status |
|-----------|--------|
| Perception | ✅ Complete |
| Memory | ✅ Complete |
| Reasoning | ✅ Complete |
| Planning | ✅ Complete |
| Executive Control | ✅ Implemented |
| Action System | ✅ Implemented |
| Self-learning | ✅ Implemented |
| Knowledge Expansion | ✅ Implemented |
| PersistentMemory | ✅ Complete |
| TextEncoder (CNN & Transformer) | ✅ Complete |
| ImageEncoder & VectorEncoder | ✅ Complete |
| AudioEncoder (Mel-Spec) | ✅ Complete |
| CrossModalAttention | ✅ Complete |
| MultimodalEncoder | ✅ Complete |
| UCGAModel | ✅ Complete |
| RL Agent (PPO) & GridWorldEnv | ✅ Complete |
| ROS, LOS, GIB Validation Dashboards | ✅ Complete |
| Distributed Cognitive Graph | ✅ Complete |
| Hierarchical Memory | ✅ Complete |
| Lifelong Learner (EWC) | ✅ Complete |
| Adaptive Topology | ✅ Complete |
| Extended Scale Training Checkpoints | ⏳ In Progress |
| Research Paper | ✅ Draft |

---

## Citation & License

```bibtex
@article{singh2026ucga,
  title={Unified Cognitive Graph Architecture: A Novel Graph-Based Framework
         for Artificial General Intelligence},
  author={Singh, Aman},
  year={2026},
  note={Preprint}
}
```

MIT License — Copyright (c) 2026 Aman Singh
See [LICENSE](LICENSE) for details.
