# P0 Research Specification

**Document Status:** FROZEN — Specification & Benchmark Freeze Complete
**Version:** 0.2 (P0 Fixation Pass — 2026-09-12)
**Date:** 2026-09-12
**Repository:** RL_LINER_SHIPPING

---

## 1. Research Objective

### Primary Research Question

> How does a published reinforcement-learning-based Liner Shipping Network Design method compare with an existing GA/MILP-based architecture under a common LINERLIB evaluation framework?

### Secondary Research Questions

1. **Reproducibility:** Can the published RL methodology of Dutta, Lin & Jin (arXiv:2411.09068) be faithfully reproduced from the paper and its supplementary materials?
2. **Solution quality scaling:** How does RL solution quality scale with network size across the LINERLIB benchmark instances (12 → 201 ports)?
3. **Computational cost scaling:** How does RL training and inference time scale with network size?
4. **Economic comparison:** How does RL compare economically with GA/MILP on identical instances?
5. **Operational comparison:** How do the generated networks differ operationally (transshipment share, vessel utilization, service structure)?
6. **Trade-offs:** What are the trade-offs between solution quality and compute time for RL vs. GA/MILP?

### Distinctive Positioning

This repository is **intentionally independent** from any existing AI Vessel Routing System. The RL implementation must eventually produce solution artifacts that can be evaluated by a neutral common evaluator alongside GA/MILP solution artifacts. No shared codebase, configuration, or runtime dependency exists at this stage.

---

## 2. Scope

### In Scope

| Category | Description |
|----------|-------------|
| Benchmark data | Official LINERLIB benchmark instances (7 instances) |
| LSNDP | Full Liner Shipping Network Design Problem formulation |
| NDP | Network Design Problem sub-component (RL's domain) |
| MCF | Multi-Commodity Flow evaluation procedure |
| GAT | Graph Attention Network encoder |
| Transformer | Transformer encoder policy component |
| Encoder-only policy | Non-autoregressive port selection via Bernoulli sampling |
| Encoder-decoder policy | Autoregressive LSTM decoder-based port selection |
| PPO | Proximal Policy Optimization training algorithm |
| Training | Policy gradient optimization loop |
| Inference | Online service construction from trained policy |
| Benchmarking | Performance measurement on LINERLIB instances |
| Scaling experiments | Cross-instance performance analysis |
| Common evaluation | Shared metric definitions for RL vs. GA/MILP comparison |

### Out of Scope for Initial Paper Reproduction

| Exclusion | Rationale |
|-----------|-----------|
| Modifying the existing GA/MILP algorithm | Comparison requires identical, unmodified baselines |
| Integrating RL into production orchestration | Research focus only; integration is future work |
| End-to-end container-level RL | Paper operates at network design level, not container level |
| Arbitrary reward engineering | Paper-faithful reward formula must be preserved first |
| Adding transit-time optimization to paper-faithful model | Transit-time constraints are not part of the paper's LSNDP formulation |
| Replacing the paper methodology before reproduction | Paper-faithful baseline required before extensions |

---

## 3. Research Modes

Two distinct operational modes are defined:

### MODE A: Paper-Faithful Reproduction

**Purpose:** Verify that the methodology described in Dutta et al. (2024) can be independently implemented and produces results consistent with the published claims.

**Constraints:**
- Use the paper's stated hyperparameters (Table 5, Appendix D).
- Use the paper's stated state, action, and reward formulations.
- Use the paper's stated MCF heuristic (Algorithm 1, Appendix B).
- Use only the three benchmark instances reported in the paper: Baltic (n=12), WAF (n=20), WorldSmall (n=47).
- Report results with the same metrics as the paper: network profit.
- Document every deviation as it arises.

**Acceptance criterion for Mode A:** Results must fall within a reasoned envelope of the published values (profit figures in Tables 1–2). Exact numerical reproduction is not expected due to stochasticity, but qualitative and directional consistency is required.

### MODE B: Common Project Evaluation

**Purpose:** Place the RL method on a common measurement platform alongside the existing GA/MILP system so that apples-to-apples comparison is possible.

**Constraints:**
- Both RL and GA/MILP must use identical input instances, demand data, vessel data, and evaluation definitions.
- Both methods must report all agreed metrics (see METRIC_DEFINITIONS.md).
- Random seeds must be documented and multiple runs aggregated.
- Compute-time measurement boundaries must be identical between systems.

**Acceptance criterion for Mode B:** Both systems must produce traceable solution artifacts that can be independently evaluated by the common evaluator using the frozen metric definitions.

### Why Both Modes Are Necessary

Mode A establishes **what the paper claims and whether we understand it correctly**. Mode B establishes **where the method stands relative to an alternative approach on a level playing field**. Without Mode A, Mode B results are meaningless (we cannot trust our implementation). Without Mode B, Mode A results lack comparative context (paper reproduction alone does not answer the primary research question).

---

## 4. Paper Source

**Target paper:**
- Title: "Liner Shipping Network Design with Reinforcement Learning"
- Authors: Utsav Dutta, Yifan Lin, Zhaoyang Larry Jin
- Affiliation: C3 AI, Redwood City, California, US
- arXiv identifier: 2411.09068
- PDF: `RL_paper.pdf` (in repository root)
- Pages: 27

**Supporting source (benchmark data):**
- LINERLIB benchmark suite (Brouer et al., 2014, Transportation Science)
- Version 1.2 released February 2024
- Repository: `data/LINERLIB-master/`
- Original publication: Brouer, Berit D.; Alvarez, J. Fernando; Plum, Christian Edinger Munk; Pisinger, David; Sigurd, Mikkel M. "A base integer programming model and benchmark suite for liner shipping network design." Transportation Science 48(2):281–312, 2014.

---

## 5. Architecture Overview

The system follows a strict two-stage decomposition:

```
RL Agent (Network Design)
    ↓ generates services {s₁, s₂, ..., sₖ}
    ↓ assigns vessels to each service
MCF Heuristic (Flow Evaluation)
    ↓ routes demand through generated network
    ↓ computes profit η
Reward Signal
    ↓ R(t+1) = [η(t+1) - η(t)] / η(1)
PPO Update
    ↓ updates policy parameters θ
```

Key principle: **The RL agent designs the network structure. An MCF procedure evaluates how well that network serves demand.** The RL agent never directly routes individual containers.

---

## 6. Document Index

| Document | Purpose |
|----------|---------|
| `PAPER_METHOD_SPECIFICATION.md` | Precise methodology from target paper |
| `PROBLEM_FORMULATION.md` | Frozen mathematical interpretation |
| `DATA_PROVENANCE.md` | Complete data inventory and hash trail |
| `ASSUMPTIONS.md` | Every assumption with evidence tag |
| `DEVIATIONS.md` | Formal register of paper-vs-implementation differences |
| `METRIC_DEFINITIONS.md` | Mathematically frozen metric formulas |
| `EXPERIMENT_PROTOCOL.md` | Frozen experimental methodology |
| `VALIDATION_ACCEPTANCE_CRITERIA.md` | Hard validation gates |
| `RESEARCH_REFERENCE_MATRIX.md` | Article relevance matrix |

---

## 7. P0 Self-Review Checklist

All checklist items must be addressed within the document set:

- [x] Contradictions — reviewed internally during authoring
- [x] Undefined terminology — all terms defined at first use
- [x] Missing metrics — all requested metrics documented in METRIC_DEFINITIONS.md
- [x] Ambiguous denominators — explicitly resolved in METRIC_DEFINITIONS.md
- [x] Unsupported assumptions — tagged and classified in ASSUMPTIONS.md
- [x] Paper-vs-implementation confusion — evidence tags used throughout
- [x] Benchmark identity confusion — verified against LINERLIB readme and actual data
- [x] Accidental fabricated results — no results claimed in P0
- [x] Missing acceptance criteria — gates defined in VALIDATION_ACCEPTANCE_CRITERIA.md
- [x] Missing reproducibility requirements — seed protocol defined in EXPERIMENT_PROTOCOL.md

---

## 8. Open Questions Before P1

These questions must be resolved before implementation begins:

1. **[PENDING]** WAF instance port count: LINERLIB readme states n=19 but actual data contains 20 ports. Which count should the paper-faithful reproduction use? The paper's Table 2 states "WAF (n=20)" — this matches the actual data file, suggesting the original article had a typo corrected in the LINERLIB v1.2 release.

2. **[PENDING]** EuropeAsia vs. AsiaEurope naming: LINERLIB refers to this as "AsiaEurope" (111 ports per article), but the data file is named `Demand_EuropeAsia.csv` and contains 114 ports. This discrepancy needs resolution before claiming any result.

3. **[PENDING]** WorldLarge port count: Article declares n=197 but actual data contains 201 ports. This is a second instance where actual data diverges from the original article declaration.

4. **[PENDING]** Maximum number of services |S|: The paper defines edge feature dimensionality as De = 6 + |S| but does not explicitly state what value of |S| is used for each instance. This must be determined from the paper's experimental setup or inferred.

5. **[PENDING]** Perturbation details for non-Baltic training: The paper reports training on perturbed Baltic data (±10%) for the reproduction experiments, then applies the trained policy to WAF and WorldSmall without retraining. For the full scaling experiment (Experiment C), it is unclear whether perturbation is applied to each instance or if the policy is trained separately per instance.

6. **[PENDING]** Specific hyperparameter selections from ranges: Table 5 gives ranges (e.g., learning rate [1e-4, 3e-4], clip [0.15, 0.25]) but not the exact values used. The paper does not specify which point in each range was selected.

7. **[PENDING]** Demand file for WorldSmall: LINERLIB v1.2 added `Demand_WorldSmall_Fixed_Sep.csv` to correct 7 demand entries that were truncated to decimal values (e.g., "1.86" instead of "1860"). The paper was published after v1.2. Which file should be used for paper-faithful reproduction?

8. **[PENDING]** MCF implementation language: The paper states the MCF is "written in Rust" but does not provide the source. For paper-faithful reproduction, an equivalent algorithm must be implemented in Python. The exact Rust implementation details (e.g., Dijkstra variant, tie-breaking rules) are not specified.

9. **[PENDING]** Transshipment cost handling in expanded graph: Algorithm 1 uses "marginal unit cost (from Chandle)" for path sorting, but the exact per-edge weight derivation from Chandle (Eq. 31) for the expanded graph is not fully detailed in the pseudocode.

10. **[PENDING]** Unused vessel cost sign convention: Eq. 34 defines `Cunused = -Σ_v (vn - Σ_r nv,r) · vTC`. This implies unused vessels generate negative cost (i.e., they reduce total cost / increase profit). This is counter-intuitive and must be verified before implementation.

11. **[PENDING]** Whether the paper's η₁ normalization uses the profit after the first service or the profit of an empty network (η₀ = 0): If η₁ is the profit after step 1, normalization is well-defined. But if η₁ equals zero initially, normalization would be undefined. The paper implies η₁ > 0 from the first service, but this should be confirmed.

12. **[PENDING]** Hardware specifics for training: The paper mentions "A100 GPU" for training and "Apple M2 CPU with 12 cores" for inference, but does not specify memory configuration, CUDA version, or PyTorch version.

---

## 9. P0 Completion Status

| Checkpoint | Status |
|------------|--------|
| Research objective frozen | ✅ Complete |
| Paper methodology documented | ✅ Complete (PAPER_METHOD_SPECIFICATION.md) |
| Problem formulation documented | ✅ Complete (PROBLEM_FORMULATION.md) |
| Data provenance documented | ✅ Complete (DATA_PROVENANCE.md) |
| Benchmark instances frozen | ✅ Complete |
| Paper reproduction instances frozen | ✅ Complete |
| Scaling instances frozen | ✅ Complete |
| State representation frozen | ✅ Complete |
| Action representation frozen | ✅ Complete |
| Reward formulation frozen | ✅ Complete |
| Termination logic specified | ✅ Complete |
| MCF role specified | ✅ Complete |
| Policy architectures specified | ✅ Complete |
| PPO methodology specified | ✅ Complete |
| Paper-faithful mode defined | ✅ Complete |
| Common-evaluation mode defined | ✅ Complete |
| All requested metrics defined | ✅ Complete (METRIC_DEFINITIONS.md) |
| Compute-time methodology defined | ✅ Complete |
| Seed/reproducibility protocol defined | ✅ Complete |
| Assumptions documented | ✅ Complete (ASSUMPTIONS.md) |
| Deviations register created | ✅ Complete (DEVIATIONS.md) |
| Validation gates created | ✅ Complete (VALIDATION_ACCEPTANCE_CRITERIA.md) |
| Acceptance criteria created | ✅ Complete |
| Benchmark YAML created | ✅ Complete (config/benchmark.yaml) |
| README created | ✅ Complete |
| Research reference matrix created | ✅ Complete (RESEARCH_REFERENCE_MATRIX.md) |
| No RL implementation written | ✅ Confirmed |
| No Git actions performed | ✅ Confirmed |
| No existing project modified | ✅ Confirmed |
| No experimental results fabricated | ✅ Confirmed |
