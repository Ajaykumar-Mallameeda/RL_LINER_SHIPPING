# Problem Formulation

**Status:** FROZEN — Subject to change only after paper-faithful reproduction is validated
**Version:** 0.1
**Date:** 2026-09-12

---

## 1. Mathematical Interpretation

This document freezes the mathematical interpretation used throughout the project.
It is derived from the target paper (Dutta et al., 2024) and the LINERLIB
benchmark specification (Brouer et al., 2014). Where interpretations are needed
beyond what the sources state explicitly, they are marked with [INFERENCE].

---

## 2. Core Sets and Indices

| Symbol | Meaning | Source |
|--------|---------|--------|
| P | Set of all ports in the network, |P| = P_count | [LINERLIB] |
| p, q, r | Individual ports ∈ P | [LINERLIB] |
| V | Set of vessel classes, V = {1, ..., V_count} | [LINERLIB] |
| v | Individual vessel class ∈ V | [LINERLIB] |
| D | Set of commodity demands, D = {1, ..., |D|} | [LINERLIB] |
| d | Individual demand ∈ D | [LINERLIB] |
| S | Set of services (rotations), S = {1, ..., |S|} | [PAPER] |
| s | Individual service ∈ S | [PAPER] |
| t | Time step in MDP episode, t = 0, 1, 2, ... | [PAPER] |

---

## 3. Port Parameters

Each port p ∈ P has the following attributes (from `ports.csv`):

| Attribute | Symbol | Unit | Source File Column |
|-----------|--------|------|-------------------|
| UNLOCODE | p_id | — | `UNLocode` |
| Name | p_name | — | `name` |
| Country | p_country | — | `Country` |
| Cabotage Region | p_cabregion | — | `Cabotage_Region` |
| D-region | p_dregion | — | `D_Region` |
| Longitude | p_lon | degrees | `Longitude` |
| Latitude | p_lat | degrees | `Latitude` |
| Max draft | p_draft | meters | `Draft` |
| Loading cost per FEU | p_l | USD/FFE | `CostPerFULL` |
| Transshipment cost per FEU | p_t | USD/FFE | `CostPerFULLTrnsf` |
| Fixed port call cost | p_f | USD/call | `PortCallCostFixed` |
| Variable port call cost per FEU | p_v | USD/FFE | `PortCallCostPerFFE` |

**Evidence tag:** [LINERLIB] — confirmed from `data/ports.csv` headers and LINERLIB readme

---

## 4. Vessel Parameters

Each vessel class v ∈ V has the following attributes (from `fleet_data.csv`):

| Attribute | Symbol | Unit | Column |
|-----------|--------|------|--------|
| Capacity | v_cap | FFE | `Capacity FFE` |
| Daily TC rate | v_TC | USD/day | `TC rate daily (fixed Cost)` |
| Draft | v_draft | m | `draft` |
| Min speed | v_min | knots | `minSpeed` |
| Max speed | v_max | knots | `maxSpeed` |
| Design speed | v_s | knots | `designSpeed` |
| Bunker consumption at design speed | v_fish | tons/day | `Bunker ton per day at designSpeed` |
| Idle consumption | v_fi | tons/day | `Idle Consumption ton/day` |
| Panama Canal fee | v_panama | USD | `panamaFee` |
| Suez Canal fee | v_suez | USD | `suezFee` |

The quantity of available vessels of class v for instance i is denoted v_n^{(i)} and is specified in `fleet_<Instance>.csv`.

**Evidence tag:** [LINERLIB] + [PAPER] — confirmed from `data/fleet_data.csv` and paper Appendix A.1

---

## 5. Demand Parameters

Each demand d ∈ D has the following attributes (from `Demand_<Instance>.csv`):

| Attribute | Symbol | Unit | Column |
|-----------|--------|------|--------|
| Origin port | d_o | UNLOCODE | `Origin` |
| Destination port | d_d | UNLOCODE | `Destination` |
| Weekly quantity | d_q | FFE/week | `FFEPerWeek` |
| Revenue per FFE | d_R | USD/FFE | `Revenue_1` |
| Maximum transit time | d_tt | days | `TransitTime` |

**Evidence tag:** [LINERLIB] — confirmed from demand file headers

---

## 6. Distance/Edge Parameters

Each potential edge e = (p, q) ∈ E (where E is the set of all directed port pairs) has:

| Attribute | Symbol | Unit | Source |
|-----------|--------|------|--------|
| Origin port | e_o | UNLOCODE | `dist_sparse.csv` |
| Destination port | e_d | UNLOCODE | `dist_sparse.csv` |
| Distance | e_dist | nautical miles | `dist_sparse.csv` / `dist_dense.csv` |
| Suez traversal | e_suez | binary {0,1} | `dist_dense.csv` |
| Panama traversal | e_panama | binary {0,1} | `dist_dense.csv` |

Distance is computed via shortest-path routing over waypoints from `dist_sparse.csv`.
The LINERLIB readme notes a 5% error margin on distances due to NIMA generation method.

**Evidence tag:** [LINERLIB] — confirmed from `data/dist_sparse.csv` and readme

---

## 7. Service Definition

A service s ∈ S is defined as:

```
s = (s_V, s_P, s_E)
```

Where:
- `s_V ⊆ V` — set of vessel classes deployed on this service
- `s_P = (p_1, p_2, ..., p_m)` — ordered sequence of port calls (round-trip rotation)
- `s_E = {(p_1,p_2), (p_2,p_3), ..., (p_m,p_1)}` — set of legs (edges) in the rotation

The number of vessels of class v deployed on service s is denoted n_{v,s}.

**Vessel count calculation** [PAPER — Appendix A.3, confirmed]:
The paper explicitly states that vessel assignments are **fractional** (not rounded up). See Table 8 in the paper, which reports values such as 24.11, 2.03, and 3.58 vessels. The required number of vessels is:

```
n_{v,s} = (Σ_{e∈s_E} e_dist) / v_s / 7
```

where the result is in weeks of vessel service, which equals the number of vessels of class v deployed on service s. The result is fractional — no ceiling is applied.

**[PENDING SOURCE VERIFICATION]**: It is not explicitly stated whether n_{v,s} in Eq. 33-35 refers to the continuous fractional value or a rounded value used for fleet-constraint checking. The cost equations clearly use the fractional value (as evidenced by Table 1's profit breakdown matching fractional vessel counts). Fleet-constraint enforcement (whether n_{v,s} must sum to ≤ v_n with fractional or integer comparison) is **NOT specified in the extracted text** and must be verified before P3 implementation.

---

## 8. Multi-Commodity Flow Variables

| Symbol | Meaning |
|--------|---------|
| f^d_e | Flow of commodity d on edge e (FFE/week) |
| Dm | Set of rejected (missed) demands |
| Dm_d | Quantity of demand d that is rejected |

**Evidence tag:** [PAPER] — Appendix A.2, Eqs. 29-31

---

## 9. Cost Components

### 9.1 Revenue
```
R_total = Σ_d d_R · Σ_{e: e_d = d_d} f^d_e
```
Total revenue from all satisfied demand.

[EVIDENCE: PAPER Eq. 29]

### 9.2 Rejected Demand Penalty
```
C_reject = Y_d · Σ_d (d_q - Σ_{e: e_d=d_d} f^d_e)
```
Where Y_d is the penalty rate per FFE of rejected demand.

**[PENDING]**: The paper references Y_d but does not specify its value in the extracted text.
This must be confirmed from the full paper or LINERLIB documentation.

[EVIDENCE: PAPER Eq. 30]

### 9.3 Handling Cost
```
C_handle = Σ_p p_l · (Σ_{e:e_d=p} Σ_d f^d_e + Σ_{e:e_o=p} Σ_d f^d_e)
         + Σ_p p_t · Σ_{e',e''∈E_s: e'_d=p, e''_o=p, e'_d≠e''_d} |Σ_d f^d_{e'} - Σ_d f^d_{e''}|
```

First term: loading + unloading cost at each port.
Second term: transshipment cost (difference in inflow/outflow at transshipment ports).

[EVIDENCE: PAPER Eq. 31]

### 9.4 Service Cost (Fixed Operating Cost)
```
C_service = Σ_s Σ_{v∈s_V} n_{v,s} · v_TC
```

[EVIDENCE: PAPER Eq. 32]

### 9.5 Unused Vessel Credit
```
C_unused = -Σ_v (v_n - Σ_{s} n_{v,s}) · v_TC
```

**Sign convention [PAPER CONFIRMED]:** The negative sign is deliberate and correct. When Σ_s n_{v,s} < v_n (under-utilization), C_unused is positive, contributing profit (unused vessels can be sub-let). When Σ_s n_{v,s} > v_n (over-utilization), C_unused is negative, contributing cost (additional vessels must be chartered). This is explicitly stated in Appendix A.3 and confirmed by Table 1 (p.9) where LINERLIB shows +6,823 and RL shows −12,596.

**Evidence tag:** [PAPER] — Appendix A.3 (p.16), Table 1 (p.9), Eq. 34

### 9.6 Voyage Cost
```
C_voyage = Σ_s Σ_{p∈s_P} Σ_{v∈s_V} (p_f + p_v · v_cap) · n_{v,s}
         + Σ_s Σ_{v∈s_V} [ (Σ_{e∈s_E} e_dist/v_s · v_fish + Σ_{p∈s_P} 1 · v_fi) · n_{v,s} ]
         + Σ_s Σ_{v∈s_V} Σ_{e∈s_E} (e_suez · v_suez + e_panama · v_panama)
```

Components:
1. Port call costs (fixed + variable per capacity)
2. Fuel costs (sailing + idle at ports)
3. Canal fees

[EVIDENCE: PAPER Eq. 35]

---

## 10. Network Profit

```
η = R_total - C_reject - C_handle - C_service - C_unused - C_voyage
```

Or equivalently, grouping NDP costs:
```
η = R_total - C_reject - C_handle - C_NDP
```
where C_NDP = C_service + C_unused + C_voyage.

**[PAPER] CONFIRMED — Eq. 28**

---

## 11. Critical Distinction: Paper Objective vs. Project Contribution Margin

The paper's optimization objective is **network profit η** as defined above.

The project's later evaluation may use a **contribution margin** metric that could differ
from η due to:
- Different cost accounting conventions
- Additional operational constraints not in the paper
- Different treatment of fixed overheads

**This distinction must be preserved.** Until the common evaluator is specified,
the project's primary metric is η (paper profit). The contribution margin formula
is PENDING COMMON-EVALUATOR SPECIFICATION.

---

## 12. Notation Summary

| Symbol | Domain | Meaning |
|--------|--------|---------|
| η | ℝ | Network profit (single scalar) |
| η_t | ℝ | Network profit at MDP step t |
| R_t | ℝ | Reward at step t |
| S_t | object | State at step t |
| A_t | object | Action at step t |
| π_θ | function | Policy parameterized by θ |
| f^d_e | ℝ≥0 | Flow of demand d on edge e |
| Dm | set | Rejected demand set |
| n_{v,s} | ℝ≥0 | Number of class-v vessels on service s |
| γ | ℝ | Discount factor (= 1.0) |
| λ | ℝ | GAE lambda (= 0.9) |
| ε | ℝ | PPO clip coefficient |
