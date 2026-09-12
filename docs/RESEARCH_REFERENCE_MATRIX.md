# Research Reference Matrix

**Status:** FROZEN
**Version:** 0.1
**Date:** 2026-09-12

---

## Matrix

| Source | Topic | Relevant Sections | Technical Contribution | Used in P0? | Intended Phase |
|--------|-------|-------------------|----------------------|-------------|----------------|
| **Dutta et al. (2024)** — arXiv:2411.09068 (RL_paper.pdf) | LSNDP problem definition | Section 3, Appendix A | Full mathematical formulation of LSNDP decomposed into NDP + MCF | ✅ Yes | P0-P13 (primary methodology) |
| Same | RL framework / MDP formulation | Section 4, Appendix C (Eqs. 1-39, Algorithm 2) | State, action, reward, transition model for NDP as MDP | ✅ Yes | P0-P13 |
| Same | Encoder-only policy architecture | Section 4, Eqs. 6-13 | GAT → Transformer → Bernoulli port selection | ✅ Yes | P0-P13 |
| Same | Encoder-decoder policy architecture | Section 4, Eqs. 15-27 | GAT → Transformer → LSTM autoregressive decoder | ✅ Yes | P0-P13 |
| Same | PPO hyperparameters | Section 5, Table 5 | Full hyperparameter table (ranges and fixed values) | ✅ Yes | P0-P8 |
| Same | Heuristic MCF algorithm | Appendix B, Algorithm 1, Figure 4 | Greedy sequential commodity flow with expanded graph | ✅ Yes | P0-P3 |
| Same | Reward formulation | Section 3 (Eq. 1), Appendix A (Eqs. 28-36), Section 5 | Incremental profit reward, normalized by η₁ | ✅ Yes | P0-P8 |
| Same | Experimental results | Section 6, Tables 1-4, Figures 6-11 | Baltic/WAF/WorldSmall profits, inference times, perturbation studies | ✅ Yes | P0-P12 |
| Same | Perturbation methodology | Appendix E | ±10% and ±50% demand perturbation details | ⚠️ Partial | P0, P1-P3 |
| **Brouer et al. (2014)** — LINERLIB benchmark | Benchmark instances | LINERLIB README, data files | 7-instance benchmark suite: Baltic, WAF, Med, Pacific, WS, EUAS, WL | ✅ Yes | P0-P13 |
| Same | Vessel class specifications | fleet_data.csv | 6 vessel classes with 11 features each | ✅ Yes | P0-P13 |
| Same | Port specifications | ports.csv | 435 ports with cost parameters | ✅ Yes | P0-P13 |
| Same | Distance data | dist_sparse.csv, dist_dense.csv | Waypoint-based distance matrix | ✅ Yes | P0-P3 |
| Same | Benchmark solutions | results/BrouerDesaulniersPisinger2014/*.log | Optimal/near-optimal MILP solutions for comparison | ✅ Yes | P0, P11-P12 |
| Same | Instance corrections | readme.txt (v1.2) | WAF 20 ports (not 19), WorldSmall demand fix, transittime revisions | ✅ Yes | P0, P1 |
| **Christianse et al. (2020)** — LINER shipping network design review | Literature context | Referenced in paper Section 2 | Comprehensive LSNDP review; standardized formulation | ❌ No (background) | Future |
| **Plum et al. (2014)** | Holistic MIP formulation | Referenced in paper Section 2 | Alternative OR approach to LSNDP | ❌ No (background) | Future |
| **Wang & Meng (2014)** | Liner shipping with deadlines | Referenced in paper Section 2 | Related problem variant | ❌ No (background) | Future |
| **Krogsgaard et al. (2018)** | Advanced MCF heuristics | Referenced in paper Section 3 | State-of-the-art MCF implementation (not used in paper) | ❌ No | Future (deviation consideration) |
| **Bello et al. (2016)** | Neural combinatorial optimization with RL | Referenced in paper Section 2 | Pointer Networks + RL foundation | ❌ No (historical) | Future |
| **Drori et al. (2020)** | GNN-based combinatorial optimization | Referenced in paper Section 2 | Line graph conversion approach | ❌ No (alternative) | Future |
| **Fellek et al. (2023)** | Graph transformer for VRP | Referenced in paper Section 2 | Multi-head attention with edge embeddings | ❌ No (architectural inspiration) | Future |
| **LINERLIB-master/include/bm_data.hpp** | C++ data structures | Header file | Internal data representation (not used directly) | ❌ No | Never (reference only) |
| **LINERLIB-master/src/** | C++ solver source | Source files | Implementation reference for understanding LINERLIB behavior | ⚠️ Partial | P2 (formulation validation) |
| ScienceDirect articles (2020 EJOR issues) | Operations research methods | Various | General OR background; none specifically about liner shipping or RL | ❌ No | Never |

---

## Evidence Priority Order

As stated in the research article library rules, sources are prioritized as follows:

1. **Target paper** (Dutta et al., 2024) — definitive for methodology
2. **Official LINERLIB** — definitive for benchmark data
3. **Other provided research articles** — supportive/background only
4. **Existing project documentation** — for understanding GA/MILP comparison requirements
5. **General model knowledge** — ONLY when sources are silent, explicitly marked as inference

---

## Conflict Log

No unresolved **direct methodological contradiction** has been identified between the target paper and the LINERLIB benchmark data regarding the core RL methodology (state representation, action space, reward formula, policy architecture, PPO configuration).

The following categories of discrepancy exist and are tracked separately:

| Category | Items | Resolution Status |
|----------|-------|-------------------|
| **Dataset/version discrepancies** | WAF port count (article: 19, data: 20), EuropeAsia port count (article: 111, data: 114), WorldLarge port count (article: 197, data: 201), WorldSmall demand file choice (original vs. Fixed_Sep) | Documented in DATA_PROVENANCE.md and DEVIATIONS.md; not methodological contradictions |
| **Implementation ambiguities** | Exact hyperparameter values within ranges (Table 5), maximum services \|S\| for edge feature dimension, Y_d penalty rate, MCF tie-breaking behavior, transshipment weight derivation from Eq. 31 to expanded graph | Tracked as PENDING VERIFICATION in ASSUMPTIONS.md and DEVIATIONS.md |
| **Engineering decisions** | MCF language (Rust → Python), demand file selection, training strategy per instance, timing boundaries | Documented as ENGINEERING DECISION in DEVIATIONS.md |
| **Unresolved questions** | 12 items listed in P0_RESEARCH_SPECIFICATION.md Section 8 | Open until resolved before or during P1 |
