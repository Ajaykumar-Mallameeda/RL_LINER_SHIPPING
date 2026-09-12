# Validation & Acceptance Criteria

**Status:** FROZEN
**Version:** 0.1
**Date:** 2026-09-12

A phase is not "complete" merely because its files exist. Each gate must be physically validated.

---

## Gate Summary

| Gate | Phase | Description | Hard Requirement |
|------|-------|-------------|-----------------|
| P1 | Data | All benchmark instances load correctly | 0 errors, correct row counts |
| P2 | Formulation | Toy problems match hand-calculated results | Exact match on toy instance |
| P3 | MCF | Flow conservation and capacity constraints pass | 0 violations |
| P4 | Environment | reset/step/termination behave correctly | All transitions valid |
| P5 | State | No NaN/Inf; dimensions correct | Clean tensors, correct shapes |
| P6 | Action | Generated services are legal | All services feasible |
| P7 | Model | Forward/backward pass works | No numerical errors |
| P8 | PPO | Training updates occur without instability | Loss decreases, no NaN |
| P9 | Integration | End-to-end pipeline works | Environment → policy → MCF → reward → PPO update |
| P10 | Inference | Checkpoint → solution works | Valid solution from saved model |
| P11 | Benchmark | LINERLIB benchmark runs successfully | All 7 instances produce results |
| P12 | Reproduction | Paper comparison can be reported | Results consistent with paper |
| P13 | Common Eval | RL and GA/MILP evaluable under identical definitions | Shared evaluator produces same results for both |

---

## P1 — Data Gate

**Requirement:** All benchmark instances load without error and contain expected data.

### Checklist
- [ ] `ports.csv` loads: 435 unique ports
- [ ] `fleet_data.csv` loads: 6 vessel classes with all 11 features
- [ ] All 7 `Demand_*.csv` files load: correct row counts, all integer FFE values
- [ ] All 7 `fleet_*.csv` files load: vessel class quantities parse correctly
- [ ] `dist_sparse.csv` loads: bidirectional edge pairs present
- [ ] Instance-specific demand files contain only ports present in `ports.csv`
- [ ] SHA-256 hashes of all files match the values recorded in DATA_PROVENANCE.md
- [ ] `Demand_WorldSmall_Fixed_Sep.csv` has no non-integer FFE values
- [ ] `transittime_revision/*.csv` files load correctly

### Pass Criterion
All checklist items checked. Zero loading errors. Zero hash mismatches.

---

## P2 — Formulation Gate

**Requirement:** On a hand-crafted toy problem, the profit calculation matches manual computation exactly.

### Test Fixture (SYNTHETIC TEST FIXTURE)
Create a minimal instance:
- 2 ports: A, B
- 1 demand: A→B, quantity=100 FFE/week, revenue=10 USD/FFE
- 1 vessel class: capacity=200 FFE, TC_rate=100 USD/day, design_speed=10 knots
- Distance A↔B: 100 nautical miles
- Port costs: p_l=1 USD/FFE, p_t=0.5 USD/FFE, p_f=5000 USD, p_v=0.5 USD/FFE

### Hand Calculation
Service A→B→A:
- Tour distance = 200 nm
- Tour duration = 200/10 = 20 hours = 0.833 weeks
- Vessels needed = 0.833 (fractional, per paper Appendix A.3 — no ceiling applied)
- C_service = 0.833 × 100 = 83.33 USD
- C_voyage port calls = 2 × (5000 + 0.5×200) = 2 × 5100 = 10,200 USD (with 0.833 vessels)
- C_voyage fuel = (200/10) × (bunker_cost) × 0.833 vessels
- R_total = 100 × 10 = 1000 USD
- C_handle = loading + unloading = 100 × 1 + 100 × 1 = 200 USD
- C_reject = 0 (all demand satisfied)

**Exact numeric verification required.**

### Pass Criterion
Computed profit matches hand calculation within floating-point tolerance (1e-6 USD).

---

## P3 — MCF Gate

### Checklist
- [ ] Flow conservation: for each commodity d and each port p, inflow - outflow = satisfied demand (at destination) or -satisfied demand (at origin) or 0 (at transshipment ports)
- [ ] Capacity constraints: flow on each edge ≤ edge capacity
- [ ] Non-negativity: all flows ≥ 0
- [ ] Greedy priority: higher-revenue demands are processed first
- [ ] Path finding: Dijkstra produces shortest paths by marginal handling cost
- [ ] Capacity update: edge capacities decrease correctly after each flow assignment
- [ ] Rejected demand tracking: unsatisfied demand is correctly accumulated in Dm
- [ ] Expanded graph: proxy nodes correctly model port handling and transshipment costs

### Pass Criterion
100% constraint satisfaction on test instances (toy + at least one LINERLIB instance).

---

## P4 — Environment Gate

### Checklist
- [ ] `reset()` returns a valid initial state with all vessels available and all demand unsatisfied
- [ ] `step(action)` executes the action, runs MCF, computes profit, returns (next_state, reward, done, info)
- [ ] Reward computation: R_{t+1} = (η_{t+1} - η_t) / η_1
- [ ] State transition: vessel counts decremented, service list appended
- [ ] Termination condition 1: all vessel counts ≤ 0 triggers done=True
- [ ] Termination condition 2: all demand satisfied (Dm empty) triggers done=True
- [ ] State features updated correctly after each step (dynamic edge features reflect remaining capacity and demand)
- [ ] No crash on edge cases: empty fleet, zero demand, single port

### Pass Criterion
All checks pass on toy instance and Baltic instance.

---

## P5 — State Gate

### Checklist
- [ ] Port feature matrix shape: (P+1, 2) where P = instance port count
- [ ] Edge feature matrix shape: (6+|S_max|, E) where E = number of directed port pairs
- [ ] Vessel feature matrix shape: (V, 11) where V = number of vessel classes
- [ ] No NaN values in any feature tensor
- [ ] No Inf values in any feature tensor
- [ ] Port features correctly reflect remaining (unsatisfied) demand at each step
- [ ] Dynamic edge features correctly track remaining capacity per service
- [ ] Global node features remain [0, 0] throughout
- [ ] Feature ranges are reasonable (no extreme values indicating bugs)

### Pass Criterion
All shape and value checks pass for all 7 instances.

---

## P6 — Action Gate

### Encoder-only:
- [ ] Vessel class selection is valid (class has remaining vessels)
- [ ] Port subset selection produces non-empty set
- [ ] Generated port sequence is a valid permutation of the selected subset
- [ ] First and last port in sequence are the same (round-trip)
- [ ] All consecutive port pairs have defined distances

### Encoder-decoder:
- [ ] Vessel selection probability distribution sums to 1 (after masking)
- [ ] Port selection probability distribution sums to 1 (after masking)
- [ ] Masking rules correctly applied at each sub-step
- [ ] BOS embedding only active at τ=1, t=1
- [ ] Already-visited ports masked (except first port for service completion)
- [ ] Generated service is a valid round-trip rotation

### Pass Criterion
100% legal actions on 1000 random/learned actions per instance.

---

## P7 — Model Gate

### Checklist
- [ ] Encoder-only forward pass completes without error
- [ ] Encoder-decoder forward pass completes without error
- [ ] Gradient computation works (backward pass)
- [ ] No NaN gradients
- [ ] No exploding gradients (gradient norm within reasonable bounds)
- [ ] GAT layers produce valid embeddings
- [ ] Transformer encoder produces valid embeddings
- [ ] LSTM decoder produces valid hidden states
- [ ] Output probability distributions are valid (non-negative, sum to 1 after masking)
- [ ] Model parameters are properly shared across steps within an episode

### Pass Criterion
All checks pass. Model trains for at least 100 update steps without numerical failure.

---

## P8 — PPO Gate

### Checklist
- [ ] Clip range computation is correct: ratio = new_prob / old_prob
- [ ] Clipped surrogate loss is computed correctly
- [ ] Value function loss uses target values (not current estimates)
- [ ] Entropy bonus is computed and added correctly
- [ ] GAE advantage computation is correct (λ=0.9, γ=1.0)
- [ ] Mini-batch sampling is random without replacement
- [ ] Learning rate adaptation works (target KL=0.1)
- [ ] Multiple update epochs (10) are applied correctly
- [ ] Optimizer (AdamW) performs parameter updates
- [ ] Loss values are finite and show decreasing trend over early training

### Pass Criterion
Training progresses for at least 1000 update steps with finite, non-increasing (on average) loss.

---

## P9 — Integration Gate

### Checklist
- [ ] Complete episode: reset → [step × N] → termination executes without error
- [ ] Reward signal flows correctly: profit change → normalized reward
- [ ] Policy gradient update uses correct rewards and advantages
- [ ] State is correctly passed between steps (dynamic features updated)
- [ ] Parallel environment rollout collects correct trajectories
- [ ] Checkpoint save/load preserves full model + optimizer state
- [ ] Inference from loaded checkpoint produces identical results to saving immediately after training

### Pass Criterion
Full training loop runs for at least one complete epoch on Baltic instance without error.

---

## P10 — Inference Gate

### Checklist
- [ ] Saved checkpoint loads successfully
- [ ] Inference on Baltic produces a valid network (set of services with vessel assignments)
- [ ] Inference on WAF produces a valid network
- [ ] Inference on WorldSmall produces a valid network
- [ ] All inferred services pass Action Gate checks (P6)
- [ ] MCF evaluation of inferred services produces finite profit
- [ ] Inference time is measured and recorded

### Pass Criterion
Valid solutions produced on all three paper reproduction instances.

---

## P11 — Benchmark Gate

### Checklist
- [ ] All 7 LINERLIB instances produce valid solutions
- [ ] Network profit computed for each instance
- [ ] Profit values are compared against LINERLIB benchmark logs where available
- [ ] Compute times recorded for each instance
- [ ] Demand coverage computed for each instance
- [ ] Vessel utilization computed for each instance
- [ ] Transshipment volume computed for each instance

### Pass Criterion
All 7 instances produce complete result sets.

---

## P12 — Reproduction Gate

### Checklist
- [ ] Encoder-decoder profit on Baltic ≥ LINERLIB Baltic profit ($260,948)
- [ ] Encoder-decoder profit on WAF ≥ LINERLIB WAF profit ($5,202,534)
- [ ] Encoder-decoder profit on WorldSmall ≥ LINERLIB WorldSmall profit ($32,280,000)
- [ ] Inference times are within an order of magnitude of paper's reported times
- [ ] Direction of improvement over LINERLIB is correct (RL ≥ LINERLIB)
- [ ] Results documented with full reproducibility information (seed, hardware, config)

### Pass Criterion
All three instances meet or exceed LINERLIB benchmark profits. Qualitative consistency with paper's reported trends.

---

## P13 — Common Evaluation Gate

### Checklist
- [ ] Common evaluator can process RL solution artifacts
- [ ] Common evaluator can process GA/MILP solution artifacts
- [ ] Both artifacts are evaluated using identical metric definitions
- [ ] Metric values are identical when the same solution is evaluated twice
- [ ] All metrics from METRIC_DEFINITIONS.md are computable for both systems

### Pass Criterion
Side-by-side evaluation of both systems on at least one shared instance produces comparable, consistent results.
