# RL LINER SHIPPING

An independent reinforcement-learning implementation of the Liner Shipping Network Design Problem (LSNDP), based on the published methodology of Dutta, Lin & Jin (2024), for comparison against an existing GA/MILP-based vessel routing system.

## Purpose

This repository implements a research-grade, reproducible RL solver for liner shipping network design. The objective is not to replace existing methods but to provide an independent benchmark for comparing RL-based approaches against traditional operations-research techniques (genetic algorithms / mixed-integer linear programming) on the same problem instances, using the same evaluation metrics.

## Research Objective

**Primary question:** How does a published RL-based LSNDP method compare with an existing GA/MILP-based architecture under a common LINERLIB evaluation framework?

## Source Paper

- **Title:** Liner Shipping Network Design with Reinforcement Learning
- **Authors:** Utsav Dutta, Yifan Lin, Zhaoyang Larry Jin (C3 AI)
- **arXiv:** [2411.09068](https://arxiv.org/abs/2411.09068)
- **PDF:** `RL_paper.pdf` (repository root)

## Benchmark Data

This project uses the official [LINERLIB](https://github.com/linerlib/linerlib) benchmark suite (Brouer et al., 2014, Transportation Science), version 1.2.

Seven instances are available:

| Instance | Ports | Demands | Fleet Classes |
|----------|-------|---------|---------------|
| Baltic | 12 | 22 | 2 |
| WAF (West Africa) | 20 | 37 | 2 |
| Mediterranean | 39 | 365 | 3 |
| Pacific | 45 | 722 | 4 |
| WorldSmall | 47 | 1,764 | 6 |
| EuropeAsia | 114 | 4,000 | 6 |
| WorldLarge | 201 | 9,622 | 6 |

Data is located in `data/` and is treated as immutable raw source material. See `docs/DATA_PROVENANCE.md` for full inventory and SHA-256 hashes.

## Architecture Overview

```
RL Agent (Network Design)          MCF Heuristic (Flow Evaluation)
        ↓                                      ↓
  Selects vessel class                 Routes demand through
  + Determines port sequence           generated service network
        ↓                                      ↓
  Service s = (vessel, ports)          Profit η computed
        ↓                                      ↓
  Added to network S                   Reward R = (η_t+1 - η_t) / η_1
        ↓                                      ↓
  PPO updates policy θ
```

The RL agent operates at the **service/network design level**. It never routes individual containers — that is the role of the heuristic Multi-Commodity Flow (MCF) procedure, which evaluates each candidate network and produces the profit signal used for training.

## P0 Status

**Current phase: P0 — Research & Specification Freeze** ✅ COMPLETE

P0 documentation/specification foundation is frozen. The following documents are the authoritative reference for all subsequent phases. Scientific ambiguities and unresolved questions are explicitly tracked in `docs/ASSUMPTIONS.md` and `docs/DEVIATIONS.md` and are NOT hidden.

P0 has produced the following frozen documentation:

| Document | Description |
|----------|-------------|
| `docs/P0_RESEARCH_SPECIFICATION.md` | Master specification; research questions; scope; modes |
| `docs/PAPER_METHOD_SPECIFICATION.md` | Precise methodology from target paper (state, action, reward, policy, PPO) |
| `docs/PROBLEM_FORMULATION.md` | Frozen mathematical interpretation of LSNDP |
| `docs/DATA_PROVENANCE.md` | Complete data inventory, SHA-256 hashes, provenance verification |
| `docs/ASSUMPTIONS.md` | All assumptions tagged CONFIRMED / INFERRED / PENDING |
| `docs/DEVIATIONS.md` | Formal deviation register (paper vs. our implementation) |
| `docs/METRIC_DEFINITIONS.md` | Mathematically frozen metric formulas |
| `docs/EXPERIMENT_PROTOCOL.md` | Five experiments (A–E) with strict ordering |
| `docs/VALIDATION_ACCEPTANCE_CRITERIA.md` | 13 hard validation gates (P1–P13) |
| `docs/RESEARCH_REFERENCE_MATRIX.md` | All source articles mapped to project components |
| `config/benchmark.yaml` | Frozen benchmark instance metadata |

### What P0 Has Decided

1. **Benchmark set:** 7 LINERLIB instances; paper reproduction focuses on Baltic, WAF, WorldSmall
2. **Two policies:** Encoder-only (Bernoulli) and encoder-decoder (LSTM autoregressive)
3. **Reward:** Incremental normalized profit, R = (η_{t+1} - η_t) / η_1
4. **MCF role:** Heuristic evaluation only — not part of the policy
5. **Two research modes:** Mode A (paper-faithful reproduction) and Mode B (common comparison)
6. **Data immutability:** Raw LINERLIB data in `data/` will not be modified
7. **Repository independence:** No dependencies on any external project's code or configuration

### What P0 Has NOT Done

- ❌ No RL engine implementation
- ❌ No GAT / Transformer / LSTM code
- ❌ No PPO implementation
- ❌ No Gymnasium environment
- ❌ No MCF solver
- ❌ No training performed
- ❌ No benchmarks run
- ❌ No experimental results fabricated
- ❌ No Git operations performed
- ❌ No existing project code modified

## Future Phases

| Phase | Description |
|-------|-------------|
| **P0** | Research & Specification Freeze — COMPLETE |
| **P1** | LINERLIB Data Foundation — load, validate, hash, provenance audit |
| **P2** | Mathematical Problem Formulation — freeze equations, toy verification |
| **P3** | MCF / Network Evaluation Engine — greedy heuristic implementation |
| **P4** | RL Environment — Gymnasium interface, reset/step/termination |
| **P5** | State Representation — port/edge/vessel feature construction |
| **P6** | Action & Service Generation — encoder-only + encoder-decoder policies |
| **P7** | GAT + Transformer Architecture — shared encoder backbone |
| **P8** | Encoder-Only Policy — Bernoulli port selection + approximate TSP |
| **P9** | Encoder-Decoder Policy — LSTM autoregressive decoder |
| **P10** | PPO Training Engine — clipped surrogate, GAE, AdamW |
| **P11** | Unit + Integration + Numerical Validation — Gates P1–P11 |
| **P12** | Real LINERLIB Training — Baltic ±10% perturbation, 16,000 instances |
| **P13** | Inference / Solver — checkpoint, service generation, MCF evaluation |
| **P14** | Paper Reproduction — Baltic/WAF/WorldSmall vs. published results |
| **P15** | Common Evaluation — shared metric definitions, RL vs GA/MILP pipeline |
| **P16** | GA/MILP vs RL Benchmark — side-by-side comparison on fixed instances |
| **P17** | Network Scaling Experiments — all 7 instances, quality & time scaling |
| **P18** | Ablation & Sensitivity Studies — policy variants, hyperparameter sweeps |
| **P19** | Final Research Analysis — synthesis, paper preparation |
| **P20** | Final System Packaging — documentation, reproducibility package |

### Major Validation Gates

| Gate | Phases | Description |
|------|--------|-------------|
| **Gate 1 — Engine Correctness** | P0–P6 | Data loads, formulation matches hand calculations, MCF constraints satisfied, environment valid, state/action tensors clean |
| **Gate 2 — RL Correctness** | P7–P11 | Forward/backward pass works, PPO converges on toy problem, integration pipeline functional, no numerical instability |
| **Gate 3 — Scientific Validation** | P12–P14 | LINERLIB training succeeds, paper reproduction results consistent, benchmark suite produces valid outputs |
| **Gate 4 — Research Comparison** | P15–P19 | Common evaluator operational; RL vs GA/MILP comparison produced; scaling characterization complete; ablation studies concluded |

## Repository Structure

```
RL_LINER_SHIPPING/
├── docs/                         # P0 documentation (frozen)
│   ├── P0_RESEARCH_SPECIFICATION.md
│   ├── PAPER_METHOD_SPECIFICATION.md
│   ├── PROBLEM_FORMULATION.md
│   ├── DATA_PROVENANCE.md
│   ├── ASSUMPTIONS.md
│   ├── DEVIATIONS.md
│   ├── METRIC_DEFINITIONS.md
│   ├── EXPERIMENT_PROTOCOL.md
│   ├── VALIDATION_ACCEPTANCE_CRITERIA.md
│   └── RESEARCH_REFERENCE_MATRIX.md
├── config/
│   └── benchmark.yaml            # Frozen benchmark configuration
├── data/                         # Immutable raw LINERLIB data
│   ├── LINERLIB-master/          # Original benchmark repository
│   ├── ports.csv                 # 435 global ports
│   ├── fleet_data.csv            # 6 vessel class specifications
│   ├── dist_sparse.csv           # Sparse distance matrix
│   ├── Demand_*.csv              # 7 instance demand files
│   ├── fleet_*.csv               # 7 instance fleet files
