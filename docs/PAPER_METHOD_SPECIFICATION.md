# Paper Method Specification

**Source:** Dutta, Lin & Jin (2024), "Liner Shipping Network Design with Reinforcement Learning", arXiv:2411.09068
**Evidence basis:** Target paper sections 3–5, Appendices A–C, Tables 1–5

---

## 1. Problem Decomposition

The LSNDP is decomposed into two sequential sub-problems:

```
LSNDP
 ├── Network Design Problem (NDP)   ← RL agent's domain
 │       Input:  graph G(P,E), fleet V, demand D
 │       Output: set of services S = {s₁, ..., sₖ}
 │       Each service s: (vessel class, ordered port sequence)
 │
 └── Multi-Commodity Flow Problem (MCF)   ← evaluation heuristic
         Input:  services S, graph G, demand D
         Output: flow assignment f^d_e, rejected demand Dm
         Computes: η(S) = network profit
```

**Critical distinction:** The RL agent only decides *which services to build* and *what vessels to assign*. It never decides how individual containers flow through the network. That decision is made by the MCF heuristic, whose output determines the reward signal.

---

## 2. State Representation

State at step t: **S_t = (S*_t, V_t, p_t, f^s_e, f^d_e)**

### 2.1 Port Features — p ∈ ℝ^(P+1)×2  [PAPER, Eq. 37-38]

Each node in the graph represents a port. An additional global node is included.

For each port p_i (i = 1, ..., P):
```
p_i = [ Σ_{d: destination=d_i} d_q ,  Σ_{d: origin=d_i} d_q ]
```
- Element 1: total incoming demand (FFE/week) destined for port i
- Element 2: total outgoing demand (FFE/week) originating from port i

Global node:
```
p_{P+1} = [0, 0]
```

**Status of dynamic features:** Port features are recomputed at each step based on remaining unsatisfied demand. When demand is partially satisfied by the MCF, the feature values change.

[PAPER: CONFIRMED — Eqs. 37-38]

### 2.2 Edge Features — f_e ∈ ℝ^(D_e × E)  [PAPER, Eq. 39]

Where E = total number of possible port pairs, D_e = 6 + |S|, |S| = maximum number of services.

**Static features** f^s_e ∈ ℝ^(4×E):
| Feature | Description | Time-varying? |
|---------|-------------|---------------|
| Origin port index | Integer index of origin port | No |
| Destination port index | Integer index of destination port | No |
| Distance | Nautical miles between ports (from dist_sparse.csv routing) | No |
| Revenue per unit | Revenue_1 / FFEPerWeek for the OD pair on this edge | No (constant) |

**Dynamic features** f^d_e ∈ ℝ^((2+|S|)×E):
| Feature | Description | Time-varying? |
|---------|-------------|---------------|
| Remaining unsatisfied demand | Remaining FFE/week for this OD pair | Yes |
| Remaining edge capacity | Remaining vessel capacity on this edge | Yes |
| Service inclusion indicators | |S| binary flags: is this edge in service s_j? | Yes (sparse) |

[PAPER: CONFIRMED — Eq. 39]

### 2.3 Vessel State — v_t ∈ ℝ^(V×D_v)  [PAPER, Appendix A.1, Eq. 10]

D_v = 11 vessel features per class, corresponding to columns of Table 3 in Brouer et al. (2014):

| # | Feature | Symbol | Unit |
|---|---------|--------|------|
| 1 | Capacity | v_cap | FFE |
| 2 | Quantity (remaining) | v_n | vessels |
| 3 | TC rate (daily) | v_TC | USD/day |
| 4 | Draft | v_draft | m |
| 5 | Minimum speed | v_minSpeed | knots |
| 6 | Maximum speed | v_maxSpeed | knots |
| 7 | Design speed | v_s | knots |
| 8 | Fuel consumption at design speed | v_fish | USD/day |
| 9 | Idle fuel consumption | v_fi | USD/day |
| 10 | Panama Canal fee | v_panama | USD |
| 11 | Suez Canal fee | v_suez | USD |

Note: Feature #2 (quantity) is **dynamic** — it decreases as vessels are assigned to services.

[PAPER: CONFIRMED — Appendix A.1]

---

## 3. Action Representation

Action at step t: **A_t = (vessel selection, port sequence)**  [PAPER, Eq. 3]

A single action generates **one complete round-trip service**:

```
A_t = (A_v,t, A_p,t)
A_v,t ∈ {1, ..., V}          — index of vessel class selected
A_p,t = (p_1, p_2, ..., p_m)  — ordered sequence of ports in the service
```

### 3.1 Vessel Selection

The agent selects a vessel class from those still having remaining capacity (v_n > 0).

### 3.2 Port Sequence Generation

Two approaches are used, differing in how A_p,t is generated:

**Encoder-only (one-shot):**
- Generate all port memberships simultaneously via Bernoulli sampling
- Determine port sequence separately (approximate TSP)
- Port selection is independent per port

**Encoder-decoder (autoregressive):**
- Select vessel class via neural policy (soft decision)
- Then sequentially select ports one at a time via LSTM decoder
- Service completes when the first selected port is revisited

The paper notes that a service must return to its origin port to form a valid round-trip (rotation).

[PAPER: CONFIRMED — Section 4, Eqs. 2-3, 6, 25-26]

---

## 4. Policy Architectures

Both approaches share the same encoder backbone, differing only in the decoder.

### 4.1 Shared Encoder

```
Input:  port features fp, edge features fe
    ↓
GAT layers (L=3 layers, multi-head attention)
    ↓
h_p^(L) ∈ ℝ^((P+1)×H)     — port + global embeddings
    ↓
Transformer encoder (3 layers, 8 heads, H=512)
    ↓
h̃_p ∈ ℝ^(P×H)              — contextualized port embeddings
h̃_v ∈ ℝ^(V×H)              — contextualized vessel embeddings
```

**GAT layers** (Eq. 7-8):
```
h_p^(1) = GAT^(1)(fp, fe)
h_p^(l) = GAT^(l)(h_p^(l-1), fe)   for l = 2, 3
```
Edge features remain unchanged through GAT. Output: h_p^(L) = [h_p,port^(L), h_p,global^(L)].

**Transformer encoder** (Eq. 11):
```
[h̃_p, h̃_v] = Transformer(h_p,port^(L), hv)
```
Where hv is the encoded vessel state (selected vessel class for encoder-only; all classes for encoder-decoder).

Positional embeddings are **omitted** (no inherent temporal ordering in the graph).

[PAPER: CONFIRMED — Section 4, Eqs. 7-11]

### 4.2 Policy A: Encoder-Only

**Port selection** (Eq. 12-13):
```
[ñ_p]_i = σ(W_p · h̃_p_i)        ← sigmoid over port embedding
X_p ~ Bernoulli(ñ_p)               ← independent Bernoulli sampling
Ã_p = {i | (X_p)_i = 1}           ← selected ports
```

**Sequence determination:** After selecting which ports are included, the order is determined by an approximate TSP procedure (not specified in detail in the paper). The paper states this is a heuristic applied after the neural policy outputs the port subset.

**Vessel selection:** Rule-based — select the vessel class with the highest remaining count among those that can physically visit all selected ports (capacity constraint check).

[PAPER: CONFIRMED — Eqs. 12-13, Section 4]

### 4.3 Policy B: Encoder-Decoder (Autoregressive)

**Decoding process** (Eqs. 15-27):

```
h̃_p = Transformer(h_p,port^(L))         ← port embeddings for decoder
h̃_v = W'_v · v_t                        ← all vessel class embeddings
h_BOS ∈ ℝ^H                              ← fixed "begin of service" embedding

h_embed = [h̃_p; h̃_v; h_BOS]             ← concatenated, size (P+V+1)×H
N̄ = P + V + 1
```

**Sub-step τ decoding** (Eqs. 19-26):
```
h'_τ = LSTM(h_{τ-1}, x_t)               ← LSTM with previous hidden state
b_τ = LN(FF(h'_τ))                       ← feed-forward + layer norm
ê_τ = Softmax(b_τ_n)                     ← probability over N̄ candidates

i ~ Uniform(mask(ê_τ))                   ← masked sampling
x ← [x, h_embed_i]                       ← append selected embedding
```

**Masking rule:**
- τ = 1 (first sub-step of step t): only vessels unmasked, all ports masked
- τ ≥ 2: only ports unmasked, all vessels masked
- Already-visited ports masked, except the first port (revisiting it closes the service)
- BOS embedding only active at τ = 1, t = 1

**Probability factorization** (Eq. 27):
```
π_θ(A_t | S_t) = Π_{τ=1}^{n_τ} P(A_t(τ) | A_t(τ'<τ), S_t)
```

**Architecture details** (Table 5):
- LSTM layers: 1
- FF layer: ReLU activation, maps H → N̄
- Layer normalization after FF
- Softmax for probability output

[PAPER: CONFIRMED — Section 4, Eqs. 15-27, Table 5]

---

## 5. RL Algorithm: PPO

**Framework:** PyTorch, AdamW optimizer (default β₁, β₂, weight decay)
**Environment interface:** Gymnasium

### Hyperparameters  [PAPER, Table 5]

| Parameter | Value | Type |
|-----------|-------|------|
| Hidden layer size (H) | 512 | Fixed |
| Transformer heads | 8 | Fixed |
| Transformer layers | 3 | Fixed |
| GAT layers | 3 | Fixed |
| LSTM layers | 1 | Fixed |
| Learning rate | [1e-4, 3e-4] | Range — exact value NOT specified |
| Parallel environments | [8, 16] | Range — exact value NOT specified |
| Steps per environment per update | [50, 100] | Range — exact value NOT specified |
| Discount factor (γ) | 1.0 | Fixed |
| GAE λ (TD lambda) | 0.9 | Fixed |
| Mini-batch size | [64, 128] | Range — exact value NOT specified |
| Update epochs | 10 | Fixed |
| Clip coefficient (ε) | [0.15, 0.25] | Range — exact value NOT specified |
| Target KL | 0.1 | Fixed |
| Entropy coefficient | [0.01, 0.1] | Range — exact value NOT specified |
| Value function coefficient | 0.5 | Fixed |

**PPO loss components:**
- Clipped surrogate objective
- Entropy bonus (coefficient α_entropy)
- Value function loss (coefficient 0.5)

**KL adjustment:** Target KL = 0.1 is used to adaptively adjust learning rate.

[PAPER: CONFIRMED — Table 5, Section 5]
**NOTE:** The exact hyperparameter values chosen within the reported ranges are NOT specified in the paper. These are ranges of tested configurations, not necessarily the final selected values.

---

## 6. Reward Formulation

### 6.1 Network Profit  [PAPER, Eq. 28]

```
η = R_total - C_reject - C_handle - C_NDP
```

Where:
- **R_total** = Σ_d d_R · Σ_{e: e_d=d_d} f^d_e  (total revenue from routed demand)
- **C_reject** = Y_d · Σ_d (d_q - Σ_{e: e_d=d_d} f^d_e)  (penalty for rejected demand)
- **C_handle** = Σ_p [p_l · (Σ_inflows + Σ_outflows) + p_t · Σ_transshipment flow difference]
- **C_NDP** = C_service + C_unused + C_voyage  (network design costs)

### 6.2 Network Design Costs  [PAPER, Eqs. 32-35]

```
C_service = Σ_s Σ_{v∈s_V} n_{v,s} · v_TC

C_unused = -Σ_v (v_n - Σ_{r∈R} n_{v,r}) · v_TC

C_voyage = Σ_s Σ_{p∈s_P} Σ_{v∈s_V} [(p_f + p_v · v_cap) · n_{v,s}]
         + Σ_s Σ_{v∈s_V} [ (Σ_{e∈s_E} e_dist/v_s · v_fish + Σ_{p∈s_P} 1 · v_fi) · n_{v,s} ]
         + Σ_s Σ_{v∈s_V} Σ_{e∈s_E} (e_suez · v_suez + e_panama · v_panama)
```

Note on C_unused: The paper's Eq. 34 includes a **negative sign**, meaning unused vessels *reduce cost* (increase profit). This is **confirmed intentional**. Paper Appendix A.3 (p.16) states: *"If the generated services do not utilize all available vessels, the remaining vessels can be rented out at the time charter (or TC) rates. Conversely, if the services require more vessels than are available, additional vessels must be acquired at the same rate."* Table 1 (p.9) corroborates: the LINERLIB solution shows "Unused vessel profit = +6,823" (under-utilized fleet generates profit), while the RL solution shows "Unused vessel profit = −12,596" (over-utilized fleet incurs cost). The negative sign in Eq. 34 correctly models both cases.

**[PAPER] CONFIRMED — Eqs. 32-34, Appendix A.3, Table 1 (p.9)**

### 6.3 Reward Signal  [PAPER, Eq. 1, Eq. 36]

```
R_{t+1} = η_{t+1} - η_t          ← raw incremental profit
R_{t+1} = (η_{t+1} - η_t) / η_1  ← normalized incremental profit
```

The normalized form is used for training stability. η_1 is the network profit after the first service is added.

[PAPER: CONFIRMED — Eqs. 1, 28-36]

---

## 7. Episode Termination

Termination occurs when **either** condition is met  [PAPER, Algorithm 2]:

1. **Vessel exhaustion:** v_n < 0 for all vessel classes v (no remaining vessels of any class)
2. **Demand satisfaction:** Dm = 0 (all demand has been satisfied; no rejected demand remains)

The episode may also terminate if the maximum number of services |S| is reached, though this bound is not explicitly enforced in Algorithm 2.

[PAPER: CONFIRMED — Algorithm 2, Section C]

---

## 8. MCF Role and Algorithm

The MCF is a **heuristic evaluation procedure**, not part of the RL policy. Its role is to compute η(S*) given a set of services S*.

### MCF Algorithm (Heuristic, Greedy)  [PAPER, Appendix B, Algorithm 1]

```
Input: Services S, Graph G(P,E), Demand D, Revenue R
Output: Flow assignment f^d_e, Rejected demand Dm

1. Expand graph: each edge (p,q) in service s becomes edges (p_s, q_s) with
   capacity = vessel capacity of service s, weight = 0
2. Add proxy nodes at each port for loading (w=p_l), offloading (w=p_q),
   and transshipment (w=p_t)
3. Initialize: f^d_e = ∅, Dm = ∅, Capacity[e] from services
4. Sort D descending by revenue per FFE
5. For each demand d (by priority):
   a. Set remaining demand d_r = d_q
   b. Find all paths T from d_o to d_d using expanded graph
   c. Sort T ascending by marginal unit cost (from handling costs)
   d. For each path t in T:
      - capacity_t = min(Capacity[e] for e in t.edges)
      - q = min(d_r, capacity_t)
      - d_r -= q
      - Record flow f^d_e for e in t
      - Update Capacity[e] -= q
      - If d_r = 0: break
   e. If d_r > 0: add d_r to Dm (rejected)
6. Return f^d_e, Dm
```

The paper states the MCF is implemented in Rust for performance (executed millions of times during training). Our implementation will be in Python; performance will be benchmarked separately.

[PAPER: CONFIRMED — Appendix B, Algorithm 1, Figure 4]

---

## 9. Training Setup

**Training data (paper's approach):**
- Baltic instance with ±10% demand perturbation
- 16,000 perturbed training instances
- Demand quantities randomized; origins and destinations fixed
- Standard deviation of perturbation distribution = 10% of original demand
- Samples truncated at 0 (no negative demand)
- Test set: single unperturbed LINERLIB Baltic instance

**Generalization test:** Policy trained on perturbed Baltic is applied to WAF and WorldSmall without retraining.

**Hardware:**
- Training: NVIDIA A100 GPU
- Inference benchmarking: Apple M2 CPU, 12 cores

[PAPER: CONFIRMED — Section 6.1, Appendix E]
