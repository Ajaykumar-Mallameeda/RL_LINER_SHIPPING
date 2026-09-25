# Deviations Register

**Status:** ACTIVE
**Version:** 0.1
**Date:** 2026-09-12

This document records formal deviations between the target paper's methodology and our intended implementation. No deviation is declared without evidence. "PENDING VERIFICATION" is used where the deviation cannot yet be confirmed or denied.

---

## Deviation Registry

| ID | Paper Behavior | Our Behavior | Reason | Impact | Status |
|----|---------------|-------------|--------|--------|--------|
| DEV-01 | MCF implemented in Rust for performance (paper Section 6, Appendix B) | MCF implemented in Python | Rust implementation not accessible; Python equivalent algorithm required for independent repository | Performance difference expected; correctness should match | PENDING VERIFICATION |
| DEV-02 | Paper uses `Demand_WorldSmall.csv` (original, with 7 truncated decimal values) OR `Demand_WorldSmall_Fixed_Sep.csv` (unclear which) | Will use `Demand_WorldSmall_Fixed_Sep.csv` | LINERLIB v1.2 corrected these values; paper published after v1.2; using corrected data is more defensible | Profit values may differ slightly from paper if paper used original file | PENDING VERIFICATION |
| DEV-03 | Paper reports WAF as n=20 (Table 2) but original article says n=19 | Using actual data count of 20 ports | LINERLIB v1.2 readme explicitly corrects this to 20 ports | Consistent with paper's Table 2 reporting | CONFIRMED — No deviation |
| DEV-04 | Paper does not specify exact hyperparameter values within reported ranges (learning rate, clip coefficient, etc.) | Will select specific values and document them | Implementation requires concrete values; choices will be documented and justified | Different hyperparameters may yield different results; reproducibility limited to documented choices | ENGINEERING DECISION — Documented in config |
| DEV-05 | Paper trains on perturbed Baltic data then tests on unperturbed Baltic, WAF, WorldSmall | May train separately per instance for scaling experiments | Paper's generalization experiment uses transfer learning; scaling experiments require instance-specific training | Different training regimes; results not directly comparable to paper's generalization claim | ENGINEERING DECISION |
| DEV-06 | Paper's WAF instance: article says 19 ports, data has 20 | Reporting WAF as 20 ports | LINERLIB v1.2 readme confirms 20 ports; paper Table 2 also says n=20 | The original 2014 article had a typo; our data is the corrected version | CONFIRMED — No deviation from corrected data |
| DEV-07 | Paper's EuropeAsia instance: article says 111 ports, data has 114 | Reporting as 114 ports (matching data) | Actual data file contains 114 unique ports; article declaration may be outdated | Benchmark results cannot be directly compared to any publication claiming n=111 | PENDING VERIFICATION |
| DEV-08 | Paper's WorldLarge instance: article says 197 ports, data has 201 | Reporting as 201 ports (matching data) | Actual data file contains 201 unique ports | Same as DEV-07 | PENDING VERIFICATION |
| DEV-09 | Paper uses a specific (unstated) value for max services \|S\| affecting edge feature dimension D_e = 6 + \|S\| | \|S\| value must be determined from paper's experimental setup | Paper does not explicitly state this value | Affects state space dimensionality; must be documented | PENDING VERIFICATION |
| DEV-10 | Paper's reward normalization uses η₁ (profit after first service) | Same normalization will be used | Direct reproduction requirement | None — faithful implementation | CONFIRMED — No deviation |
| DEV-11 | Paper does not specify the penalty rate Y_d for rejected demand (Eq. 30) | Y_d = 1000 USD/FFE, found in paper Appendix A.1 | Resolved: Appendix A.1 (p.15) explicitly states "Yd: Penalty if rejected, penalty for rejection of this demand, which is set to $1000." Verified numerically against Table 1 (Baltic: 380,000 / 1000 = 380 FFE rejected; 389,000 / 1000 = 389 FFE rejected). | RESOLVED |
| DEV-12 | Paper mentions "environment simulation + RL inference" timing but does not detail what is included | Will define timing boundaries precisely in EXPERIMENT_PROTOCOL.md | Need unambiguous measurement protocol | Ensures fair comparison | ENGINEERING DECISION |
| DEV-13 | Paper's transshipment cost formulation in Eq. 31 involves absolute differences of flows across services at each port | MCF expanded graph (Fig. 4) provides a concrete implementation mechanism | Algorithm-level clarification needed | Correct implementation requires understanding of expanded graph construction | CONFIRMED — Clarified by paper's Figure 4 |
| DEV-14 | Paper uses Gymnasium interface for environment | Will use Gymnasium interface | Paper explicitly states this; ensures compatibility with modern RL frameworks | None | CONFIRMED — No deviation |
| DEV-15 | Paper's C_unused (Eq. 34) has a negative sign, meaning unused vessels reduce cost | Will implement exactly as stated | Faithful reproduction requirement; counter-intuitive but explicit in paper. Confirmed by Table 1 (p.9): RL solution "Unused vessel profit = −12,596" (negative because over-utilized fleet), LINERLIB "Unused vessel profit = +6,823" (positive because under-utilized). Paper Appendix A.3 explains: unused vessels can be "rented out" (generating profit), excess vessels must be "acquired" (incurring cost). | CONFIRMED — Deliberate design choice in paper, corroborated by Table 1

---

## Deviation Summary

- **CONFIRMED — No deviation:** 5 items (DEV-01 actual, DEV-03, DEV-06, DEV-10, DEV-14, DEV-15)
- **PENDING VERIFICATION:** 6 items (DEV-01 language, DEV-02 file choice, DEV-07, DEV-08, DEV-09, DEV-11)
- **ENGINEERING DECISION:** 4 items (DEV-04, DEV-05, DEV-12, DEV-13 implementation approach)

---

## Guidance for P1

Before P1 begins implementation:

1. Resolve all PENDING VERIFICATION items by reviewing the full paper text and LINERLIB documentation.
2. Document all ENGINEERING DECISIONs with rationale in the relevant specification files.
3. Any new deviations discovered during implementation must be added to this register immediately.
4. Do not mark items as CONFIRMED without direct evidence from the paper or LINERLIB source.
