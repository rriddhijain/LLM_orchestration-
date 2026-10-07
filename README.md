# ⚡ Predictor-Free Self-Scheduling in LLM Serving

> **Sprint Milestone 1 (M1): 10-Day Half-Project Checkpoint**  
> **Objective:** Build an instrumented LLM serving testbed, reproduce Head-of-Line (HOL) blocking under an unbalanced workload, implement the classical non-predictive MLFQ baseline, and deploy a working proof-of-concept for training-free self-scheduling.

---

## 📌 Table of Contents
1. [Overview & Project Scope](#-overview--project-scope)
2. [Architecture & Schedulers](#-architecture--schedulers)
3. [Repository Structure](#-repository-structure)
4. [Roles & Team Allocation](#-roles--team-allocation)
5. [10-Day Step-by-Step Execution Plan](#-10-day-step-by-step-execution-plan)
6. [Literature Reference Matrix](#-literature-reference-matrix)
7. [Target Metrics & Evaluation](#-target-metrics--evaluation)
8. [Quick Start & Setup](#-quick-start--setup)

---

## 🎯 Overview & Project Scope

In LLM serving workloads, short requests often get stuck behind long-running generation tasks, leading to severe **Head-of-Line (HOL) blocking**. This project builds an instrumented serving engine to benchmark and compare traditional and novel queue scheduling techniques.

### Boundary Conditions
* **Target Model:** Single lightweight model (`meta-llama/Llama-3.2-1B` or `Qwen/Qwen2.5-Coder-1.5B-Instruct`).
* **Evaluation Workload:** Synthetic Bimodal Trace (80% short ~20 tokens, 20% long ~512 tokens with Poisson arrivals).
* **M1 Deliverable Package:** Runnable Git repository, automated benchmark CSV logs, empirical plots (Wait-Time CDF and Normalized Latency), a 4-page technical summary brief, and a 6-slide presentation deck.

---

## 🏗️ Architecture & Schedulers

We compare three primary scheduling policies:

1. **Default FCFS (First-Come, First-Served):** Ground truth baseline demonstrating standard HOL queueing delays.
2. **Skip-Join MLFQ (Multi-Level Feedback Queue):** Classical non-predictive heuristic (from FastServe) using token quantum limits ($q_1=32, q_2=128, q_3=\infty$) to demote long jobs.
3. **Predictor-Free Self-Scheduler:** Novel mechanism using model-internal signals and prompt metadata to prioritize requests without requiring an auxiliary predictor model (e.g., BERT), featuring aging counters for starvation prevention.

---

## 🗂️ Repository Structure

```bash
llm-self-scheduling/
├── README.md                   # Setup guide and replication instructions
├── requirements.txt            # vLLM/SGLang, PyTorch, httpx, matplotlib, pandas
├── run_m1_benchmarks.sh        # Execution script for full bimodal sweep across policies
├── engine/
│   ├── __init__.py
│   ├── engine_wrapper.py       # (Member 1) Exposes engine hooks & step dispatch loops
│   └── memory_monitor.py       # (Member 1) Tracks KV cache usage & preemption allocation
├── schedulers/
│   ├── __init__.py
│   ├── base.py                 # (Member 1) Abstract SchedulerPolicy base class
│   ├── fcfs.py                 # (Member 2) Ground truth FCFS implementation
│   ├── mlfq.py                 # (Member 2) FastServe-style skip-join 3-level MLFQ
│   └── predictor_free.py       # (Member 3) Self-scheduling priority ranker
├── benchmarks/
│   ├── trace_generator.py      # (Member 4) Bimodal synthetic workload generator (80/20)
│   ├── async_runner.py         # (Member 4) Asynchronous client measuring TTFT, TPOT, JCT
│   └── traces/
│       └── bimodal_trace.json  # (Member 4) Static benchmark dataset
├── eval/
│   ├── plot_results.py         # (Member 4) Generates CDFs and Normalized Latency charts
│   └── results/                # Directory storing raw CSV logs
└── docs/
    ├── M1_Technical_Report.pdf # (All Members) 4-page technical progress report
    └── M1_Presentation.pptx    # (All Members) 6-slide milestone presentation
```

---

## 👥 Roles & Team Allocation

| Role | Domain Owned | Specific Code Deliverables |
|---|---|---|
| **Member 1** | **Engine Internals & Hooking** | `engine/engine_wrapper.py`, `engine/memory_monitor.py`, `schedulers/base.py`. |
| **Member 2** | **Classical Baselines & Preemption** | `schedulers/fcfs.py`, `schedulers/mlfq.py` (3-level queue: $q_1=32, q_2=128, q_3=\infty$), preemption hooks. |
| **Member 3** | **Predictor-Free Scheduler** | `schedulers/predictor_free.py` (ranking prompt / runtime metadata extractor, aging counters). |
| **Member 4** | **Traces, Benchmarks & Metrics** | `benchmarks/trace_generator.py`, `benchmarks/async_runner.py`, `eval/plot_results.py`. |

---

## 🗓️ 10-Day Step-by-Step Execution Plan

```
[Days 1-2] Setup & Reading ──► [Days 3-5] Core Scheduling Logic ──► [Days 6-7] System Integration ──► [Days 8-10] Benchmark & Report
```

| Timeline | Member 1 (Engine Core) | Member 2 (MLFQ Baseline) | Member 3 (Self-Scheduler) | Member 4 (Workload & Eval) | Target Checkpoint |
|---|---|---|---|---|---|
| **Day 1** | Verify local GPU environment (CUDA 12.x, PyTorch); clone vLLM repo; run 1B/1.5B test inference. | Read FastServe Sec 4; design data structures for request token tracking and MLFQ states. | Read Zhen et al. Sec 3; extract ranking prompt structure and scoring logic. | Read FastServe Sec 6.1 & Splitwise; build Poisson arrival generator script. | GPU environment verified; all members finish reading. |
| **Day 2** | Trace `scheduler.py` (`_schedule()` loop); isolate queue transitions and block allocations. | Implement `FCFSScheduler` class; verify clean FIFO dispatch order. | Prototype standalone Python ranking module using model generation metadata or rank prompts. | Generate **Synthetic Bimodal Trace**: 80% short (20 tokens), 20% long (512 tokens). | Bimodal trace validated; FCFS dispatch confirmed. |
| **Day 3** | Expose modular `SchedulerPolicy` interface in engine wrapper to swap schedulers dynamically. | Implement 3-level MLFQ logic ($q_1=32, q_2=128, q_3=\infty$) without preemption. | Connect dummy sorting hook into `SchedulerPolicy` interface; verify engine step stability. | Write async benchmark runner (`httpx`/`asyncio`) measuring per-request TTFT, TPOT, and JCT. | **Checkpoint M1A:** Engine runs FCFS under bimodal load; HOL blocking confirmed. |
| **Day 4** | Build memory safety checks to prevent GPU KV-cache OOMs when preemption occurs. | Implement quantum expiration checks inside step-iteration loop to flag preempted requests. | Implement self-scheduling prompt pass / metadata scorer on queued request batch. | Instrument automated CSV telemetry logger (request ID, arrival, TTFT, TPOT, JCT, tokens). | MLFQ demotion & self-scheduler hook run in isolated loops. |
| **Day 5** | Test KV-cache recomputation vs. swapping state recovery when jobs are preempted. | Finalize `FastServeMLFQScheduler`: ensure requests exceeding $q_i$ demote cleanly. | Implement starvation mitigation: aging counter promoting jobs waiting $> T_{\text{thresh}}$. | Stress test benchmark client under arrival rates ($\lambda = 2, 4, 8 \text{ req/s}$) for stability. | MLFQ baseline running cleanly with zero memory leaks. |
| **Day 6** | Profile CPU/GPU overhead added by scheduler queue transformations. | Write unit tests for MLFQ queue demotion, quantum resets, and request completion. | Finalize `PredictorFreeScheduler`; hook live priority output to sort `waiting_queue`. | Run smoke test across full suite: FCFS vs MLFQ vs Self-Scheduler on 10 requests. | **Checkpoint M1B:** Predictor-Free prototype successfully executes bimodal batches. |
| **Day 7** | Monitor GPU memory fragmentation during stress runs; tune batch limits. | Profile MLFQ vs FCFS queue wait times; verify long jobs yield to short jobs. | Tune self-scheduling queue check frequency (every iteration vs every $K$ steps). | Run primary benchmark suite: 3 runs each across **FCFS vs MLFQ vs Predictor-Free**. | Complete raw benchmark CSV dataset gathered. |
| **Day 8** | Instrument precise scheduler overhead logging (in ms) per iteration step. | Analyze MLFQ trace data; verify quantum distribution matches theoretical predictions. | Characterize failure modes: identify conditions where ranking overhead exceeds queue wait savings. | Calculate metrics: Normalized Latency ($JCT / \text{tokens}$), P95 TTFT, and plot CDFs. | Comparison charts, latency tables, and breakdown figures generated. |
| **Day 9** | Code cleanup: package repo into `engine/`, `schedulers/`, `benchmarks/`, `eval/`. | Write code documentation and inline comments for FCFS and MLFQ implementations. | Write Methodology & Algorithmic Formulation sections for M1 technical report. | Write Evaluation & Findings section for M1 report; format figures and tables. | Codebase frozen; 4-page technical report drafted. |
| **Day 10** | Write `README.md` with one-command run script (`run_m1_benchmarks.sh`). | Prepare terminal demo showing short requests jumping long ones under self-scheduling. | Finalize equations, mathematical rigor, and Kendall's $\tau_b$ correlation notes in report. | Compile final **4-page M1 Report PDF** and assemble **6-slide checkpoint deck**. | **Milestone 1 Complete:** Working code, reproducible demo, report, and deck ready. |

---

## 📚 Literature Reference Matrix

```
┌────────────────────────────────────────────────────────────────────────┐
│                        SHARED CORE READING                             │
│ 1. Zhen et al. (OpenReview: hZmG4z68Je): Training-Free Self-Scheduling │
│ 2. Orca (OSDI 2022): Continuous Batching & Iteration Scheduling        │
└────────────────────────────────────────────────────────────────────────┘
│                    │                    │                    │
▼                    ▼                    ▼                    ▼
┌───────────┐        ┌───────────┐        ┌───────────┐        ┌───────────┐
│ Member 1  │        │ Member 2  │        │ Member 3  │        │ Member 4  │
│Engine Core│        │MLFQ Base  │        │Self-Sched │        │Workload & │
│ & Hooking │        │& Preemption│       │ Prototype │        │Evaluation │
└───────────┘        └───────────┘        └───────────┘        └───────────┘
```

### Shared Core Papers
* **Training-Free Self-Scheduling for Efficient LLM Inference Serving** (Zhen et al., ICLR 2026 submission, OpenReview `hZmG4z68Je`) — *Focus:* Sec 3 (Self-scheduling algorithm, rank prompt formulation) & Sec 4 (Kendall's $\tau_b$ rank correlation).
* **Orca: A Distributed Serving System for Transformer-Based Generative Models** (Yu et al., OSDI 2022) — *Focus:* Sec 3 & 4 (Iteration-level continuous batching, prefill vs. decode step loops).

### Functional Specialty Papers
* **Member 1:** *vLLM: Efficient Memory Management with PagedAttention* (Kwon et al., SOSP 2023)
* **Member 2:** *FastServe: Iteration-Level Preemptive Scheduling for LLM Inference* (Wu et al., NSDI 2026)
* **Member 3:** *Efficient LLM Scheduling by Learning to Rank* (Fu et al., arXiv:2408.15792)
* **Member 4:** *Splitwise: Efficient Generative LLM Inference Using Phase Splitting* (Patel et al., ISCA 2024)

---

## 📊 Target Metrics & Evaluation

1. **Normalized Latency (Primary Metric):**
   $$\text{Normalized Latency} = \frac{\text{Job Completion Time (JCT)}}{\text{Total Generated Tokens}}$$

2. **Time to First Token (TTFT):**
   $$\text{TTFT} = \text{Timestamp}_{\text{first\_token}} - \text{Timestamp}_{\text{arrival}}$$
   *Evaluated at P50, P90, and P99 via Cumulative Distribution Function (CDF) curves.*

3. **Queue Wait Time:**
   $$\text{Wait Time} = \text{Timestamp}_{\text{execution\_start}} - \text{Timestamp}_{\text{arrival}}$$

4. **Scheduling Decision Overhead:**
   *Runtime cost (ms) per iteration step to ensure schedule ranking does not bottleneck GPU throughput.*

---

## 🚀 Quick Start & Setup

### Prerequisites
* Python 3.10+
* CUDA 12.x compatible GPU environment
* PyTorch 2.x

### Installation
```bash
# Clone the repository
git clone https://github.com/your-org/llm-self-scheduling.git
cd llm-self-scheduling

# Install dependencies
pip install -r requirements.txt
```

### Running Benchmark Suite
To execute the full M1 benchmark sweep across FCFS, MLFQ, and Predictor-Free schedulers:

```bash
chmod +x run_m1_benchmarks.sh
./run_m1_benchmarks.sh
```

### Plotting Results
To generate comparison CDFs and Normalized Latency plots:

```bash
python eval/plot_results.py --input-dir eval/results --output-dir eval/plots
```
