# Unified Cognitive Graph Architecture (UCGA)
> **A novel graph-based, dual-process framework for Artificial General Intelligence**
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.8+](https://img.shields.io/badge/Python-3.8+-green.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org)
[![React Visualizer](https://img.shields.io/badge/UI-React%20%7C%20Tailwind-61DAFB.svg)](landing-page/)
**Author:** Aman Singh — Founder, UCGA Research Initiative
---
## 🌟 Abstract
This repository introduces the **Unified Cognitive Graph Architecture (UCGA)**, a novel graph-native cognitive framework designed to address fundamental limitations of traditional neural architectures in reasoning, memory integration, and adaptive intelligence. UCGA models intelligence as a recursive interaction between specialised cognitive nodes connected through adaptive weighted edges, incorporating persistent attention-based memory, self-evaluation, and conditional error correction.
Building upon the differentiable sub-symbolic core, we introduce the **UCGA Second Brain**, a **Dual-Process Cognitive Architecture** that implements a System 1 / System 2 processing paradigm:
1. **System 1 (Intuitive/Sub-symbolic):** Fast, feedforward neural encoders (Vision, Text, Audio) and differentiable persistent memory banks.
2. **System 2 (Deliberate/Symbolic):** A central orchestrator coordinating episodic vector stores (ChromaDB), semantic knowledge graphs (NetworkX), metacognitive self-critics, and sandboxed procedural tools (Python execution, web search).
---
## 🧠 Dual-Process Architecture (Second Brain)
The Second Brain maps the sub-symbolic representations computed by System 1 into a symbolic, multi-step execution loop overseen by System 2.
```text
                  [User Input / Sensory Stream]
                                │
                                ▼
  ┌───────────────────────────────────────────────────────────┐
  │               Modality Encoders (System 1)                │
  └─────────────────────────────┬─────────────────────────────┘
                                │ High-Dimensional Embeddings
                                ▼
  ┌───────────────────────────────────────────────────────────┐
  │       Episodic Vector DB       │     Semantic Graph       │  ◄── [Long-Term Memory]
  │       (ChromaDB / Vector)      │      (NetworkX / KG)     │
  └─────────────────────────────┬─────────────────────────────┘
                                │ Context Documents & Facts
                                ▼
  ┌───────────────────────────────────────────────────────────┐
  │            Central Orchestrator (System 2)               │  ◄── [Deliberate Reasoning]
  │           Generates Structured CoT Thinking             │
  └─────────────────────────────┬─────────────────────────────┘
                                │ Draft Solution
                                ▼
  ┌───────────────────────────────────────────────────────────┐
  │                Metacognitive Critic Agent                 │  ◄── [Self-Evaluation]
  │           Factual & Logical Consistency Check             │
  └─────────────────────────────┬─────────────────────────────┘
                  Is Status == NEEDS_REVISION?
                  ├─── YES ──▶ [Loop back to Reasoning with Critic Feedback]
                  └─── NO  ──▶ [Proceed to Tool / Final Output]
                                │
                                ▼
  ┌───────────────────────────────────────────────────────────┐
  │                Procedural Tool Executor                   │  ◄── [Action & Output]
  │      Python Interpreter, Web Search, File Writer APIs     │
  └───────────────────────────────────────────────────────────┘
```
### The 4-Step Cognitive Loop
1. **Context Ingestion (Memory retrieval):** 
   Queries the **Episodic Memory** (Vector DB using `all-MiniLM-L6-v2` embeddings) and **Semantic Memory** (NetworkX directed graph queried via BFS neighbor traversal) using the user query. The retrieved memories are formatted as XML context blocks.
2. **Chain-of-Thought Reasoning:** 
   The Orchestrator processes the user query along with retrieved context, emitting a step-by-step reasoning trace wrapped in `<thinking>` tags.
3. **Metacognitive Critique:** 
   The proposed `<draft_solution>` is routed to a separate **Critic Agent**. The Critic reviews the draft against the retrieved context for factual consistency and logical validity, returning a structured feedback block:
   ```xml
   <critic_feedback>
     <status>APPROVED or NEEDS_REVISION</status>
     <issues><issue>factual errors / code bugs</issue></issues>
     <suggestions><suggestion>specific corrections</suggestion></suggestions>
   </critic_feedback>
   ```
   If rejected, the Orchestrator initiates a self-correction loop, incorporating the feedback to rewrite the draft.
4. **Procedural Execution & Final Output:**
   If the Orchestrator requires external verification, it emits a `<tool_call>` tag (e.g. running code or web search). Once verified and approved, it returns the final answer in `<final_output>` tags, appending the interaction back to episodic memory.
---
## 🚀 Complete Development History
UCGA has been developed through structured research phases, systematically expanding its cognitive capabilities from basic recursive loops to a fully multi-modal, agentic cognitive system.
*   **Phase 0 — Foundation (Initial Setup):** Designed the `CognitiveNode` base class and the 9-node cognitive graph wired together in `UCGAModel`. Implemented `PersistentMemory` with attention-based read/write. Replaced `tanh` activations with **`GELU + LayerNorm`** to stabilize deep gradients.
*   **Phase 1 — Recursion Utility Experiments:** Tested temporal recurrence `T`. Showed that additional cognitive steps ($T=3$) drastically improve multi-hop reasoning (bAbI), regression accuracy, and sequence sorting.
*   **Phase 2 — ReasoningNode Upgrade (MLP → Transformer):** Replaced MLP-based reasoning with a `TransformerReasoningNode` utilizing multi-head self-attention over perception and memory vectors.
*   **Phase 3 — Comparative Experiment:** Validated that the Transformer Reasoning node excels in Parity (XOR classification) and Sequence Sorting, confirming superior test performance, state norms, and gradient stability.
*   **Phase 4 — Real Dataset Training & Encoders:** Added pluggable `ImageEncoder` (CNN trained on CIFAR-10) and sequential `TextEncoder` / `TransformerTextEncoder` (trained on AG News and SST-2).
*   **Phase 5 — Scaling & Ablation Studies:** Validated individual node contributions (`experiments/ablations.py`) and verified structural competitiveness against MLP, LSTM, and Transformer baselines.
*   **Phase 6 — Intelligence Score & Hardware Optimization:** Overcame CUDA OOM via AMP (mixed-precision) training and gradient accumulation. Introduced a unified **Intelligence Score** metric (scoring $0.60+$).
*   **Phase 7 — Memory Architecture Extension:** Enhanced memory layers with learned `read_query` projectors and a Least-Used-Slot write policy governed by usage counters.
*   **Phase 8 — Visual Builder & Landing Page UI:** Developed a React + Vite visualizer page with SVG animations representing node activation and cognitive flow.
*   **Phase 9 — Multi-Agent Scaffolding:** Connected UCGA core models to high-level multi-agent communication networks and RL environments.
*   **Phase 10 — Second Brain Dual-Process Integration:** Implemented the hybrid System 2 loop, integrating ChromaDB episodic memory, NetworkX semantic memory, Metacognitive Critic agents, sandboxed tool execution (Python interpreter, Web search, File writer), and comprehensive integration testing.
---
## 🔬 Core Equations & Mathematical Formulation
UCGA models intelligence as a 5-tuple: $\mathcal{G} = (\mathcal{V},\; \mathcal{E},\; \mathbf{W},\; \mathcal{S},\; \mathbf{M})$
- $\mathcal{V}$: Intelligence layers (Perception, Cognitive Graph, Working Memory, Reasoning, Global Workspace, Executive Controller, Action).
- $\mathcal{E}$: Directed edges encoding information flow.
- $\mathbf{W}$: Learnable weight matrices.
- $\mathcal{S}$: Internal cognitive state vectors.
- $\mathbf{M}$: Persistent external memory bank.
### State Update Equations
The recursive state update for a node $i$ at iteration step $t+1$ is formalized as:
$$v_i^{(t+1)} = \mathrm{LayerNorm}\!\Bigl(\mathrm{GELU}\bigl(W_i \cdot \sum_{j \in \mathcal{N}(i)} v_j^{(t)} + b_i\bigr)\Bigr)$$
### Memory Write Policy
To prevent memory overwrites, the write controller processes inputs sequentially to preserve updates across batches, storing inputs at the least-used slot $k$ as determined by the usage vector:
$$k = \arg\min (\text{usage})$$
$$\mathbf{M}_k = \sigma(W_{\text{gate}} \cdot \mathbf{x}) \odot (W_{\text{proj}} \cdot \mathbf{x})$$
$$\text{usage}_k = \text{usage}_k + 1$$
---
## 📂 Project Structure
```text
UCGA/
├── README.md                          # Research & Engineering Documentation
├── requirements.txt                   # Project Dependencies
├── demo_second_brain.py               # Standalone Second Brain workflow demo
├── demo_math.py                       # Recursive reasoning math demo
├── paper/                             # LaTeX Preprint Research Paper
│   ├── main.tex
│   └── v2_additions.tex
├── second_brain/                      # System 2 Cognitive Loop
│   ├── config.py                      # Subsystem configurations (paths, LLMs)
│   ├── memory/
│   │   ├── vector_store.py            # ChromaDB wrapper (Episodic Memory)
│   │   └── knowledge_graph.py         # NetworkX wrapper (Semantic Memory)
│   ├── agents/
│   │   ├── orchestrator.py            # Central 4-step loop orchestrator
│   │   └── critic.py                  # Metacognitive Critic Agent
│   ├── tools/
│   │   ├── python_interpreter.py      # Sandboxed Python execution
│   │   ├── file_writer.py             # Local path-sanitized file writer
│   │   └── web_search.py              # SerpAPI web search tool
│   └── main.py                        # CLI REPL entry point
├── ucga/                              # System 1 Sub-symbolic Core
│   ├── nodes/                         # Cognitive nodes (Reasoning, Perception, etc.)
│   ├── memory/                        # Differentiable persistent memory bank
│   ├── encoders/                      # Pluggable Text, Image, Audio encoders
│   └── ucga_model.py                  # Neural graph orchestrator
├── landing-page/                      # React-based animated visual dashboard
├── tests/                             # Unified testing suite
│   ├── test_second_brain.py           # Second Brain unit & integration tests
│   └── test_memory.py                 # Core memory tests
└── training/                          # Optimization & curriculum training scripts
```
---
## ⚡ Quick Start
### 1. Installation
Clone the repository and install the required packages:
```bash
git clone https://github.com/your-username/UCGA.git
cd UCGA
pip install -r requirements.txt
```
### 2. Run the Second Brain CLI
Interact with the Second Brain's cognitive loop using the command-line REPL. Supports `/memory` (inspect stats), `/graph` (view semantic links), `/learn [entity]: [fact]` (add semantic facts), and `/clear`:
```bash
# Mock mode (offline, no API key required)
python -m second_brain.main --mock
# Live mode (Ollama or OpenAI API key)
python -m second_brain.main
```
### 3. Run Standalone Demonstration
To run a pre-configured workflow demonstration showing memory ingestion, python execution, and critic approval:
```bash
python demo_second_brain.py
```
### 4. Run the Test Suite
Execute the comprehensive verification test suite (147 tests covering all System 1 and System 2 modules):
```bash
pytest
```
---
## ✅ Implementation Status
|
 Capability 
|
 Module / Layer 
|
 Status 
|
|
:---
|
:---
|
:---
|
|
**
Sensory Encoders
**
|
`ucga/encoders/`
|
 ✅ Vision, Text, Audio, Multimodal 
|
|
**
System 1 Memory
**
|
`ucga/memory/`
|
 ✅ Attention-based Persistent Memory 
|
|
**
Reasoning Core
**
|
`ucga/nodes/`
|
 ✅ MLP & Transformer Reasoning Nodes 
|
|
**
Self-Correction
**
|
`ucga/nodes/`
|
 ✅ Evaluation & Correction Nodes 
|
|
**
Episodic Storage
**
|
`second_brain/memory/`
|
 ✅ ChromaDB Vector Store RAG 
|
|
**
Semantic Storage
**
|
`second_brain/memory/`
|
 ✅ NetworkX Knowledge Graph (BFS) 
|
|
**
Cognitive Loop
**
|
`second_brain/agents/`
|
 ✅ 4-Step Orchestrator 
|
|
**
Self-Evaluation
**
|
`second_brain/agents/`
|
 ✅ Metacognitive Critic (XML parsing) 
|
|
**
Procedural Tools
**
|
`second_brain/tools/`
|
 ✅ Sandboxed Python, File Writer, Web Search 
|
|
**
CLI & Simulator
**
|
`second_brain/main.py`
|
 ✅ Interactive REPL & Web Simulator 
|
---
## 📜 Citation & License
If you use UCGA or the Second Brain in your research, please cite:
```bibtex
@article{singh2026ucga,
  title={Unified Cognitive Graph Architecture: A Dual-Process Framework
         for Artificial General Intelligence},
  author={Singh, Aman},
  year={2026},
  note={Preprint}
}
```
Licensed under the [MIT License](LICENSE).
