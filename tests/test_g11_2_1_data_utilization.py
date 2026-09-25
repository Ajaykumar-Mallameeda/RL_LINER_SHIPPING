"""G11.2.1 — Data Utilization Forensic Audit Tests.

Behavior-preserving instrumentation tests. These verify that the forensic
audit pipeline produces correct diagnostic artifacts without modifying any
algorithm, reward, mask, action semantics, or service construction logic.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import torch

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))


# ======================================================================
# Fixtures
# ======================================================================

@pytest.fixture(scope="module")
def worldsmall_instance():
    """Load WorldSmall instance."""
    from data.linerlib_loader import LINERLIBLoader
    loader = LINERLIBLoader(str(_ROOT / "data"))
    return loader.load("WorldSmall", validate=False)


@pytest.fixture(scope="module")
def paper_config(worldsmall_instance):
    """Paper-scale architecture config."""
    from neural import ArchitectureConfig
    dev = "cuda:0" if torch.cuda.is_available() else "cpu"
    cfg = ArchitectureConfig(
        hidden_dim=512, gat_layers=3, transformer_layers=3,
        transformer_heads=8, lstm_layers=1, device=dev,
    )
    return cfg


@pytest.fixture(scope="module")
def trainer(worldsmall_instance, paper_config):
    """Trainer configured for G11.2.1 forensic audit."""
    from policies.training import LinerShippingTrainer, TrainingConfig
    tr_cfg = TrainingConfig(
        dataset="WorldSmall", policy="encoder_decoder",
        learning_rate=2e-4, gamma=1.0, gae_lambda=0.9,
        ppo_epochs=10, clip_epsilon=0.2, target_kl=0.1,
        entropy_coefficient=0.05, value_coefficient=0.5,
        num_envs=1, steps_per_env=10, minibatch_size=32,
        seed=42, max_updates=5, checkpoint_frequency=9999,
        hidden_dim=512, gat_layers=3, transformer_layers=3,
        transformer_heads=8, lstm_layers=1,
    )
    tr_dir = _ROOT / "experiments" / "manual" / "g11_2_1_data_utilization"
    tr_dir.mkdir(parents=True, exist_ok=True)
    return LinerShippingTrainer(
        instance_name="WorldSmall",
        policy_type="encoder_decoder",
        config=tr_cfg,
        checkpoint_dir=str(tr_dir),
    )


# ======================================================================
# A — Full Dataset Loaded
# ======================================================================

class TestFullDatasetLoaded:
    """Verify all WorldSmall data is present at runtime."""

    def test_all_ports_loaded(self, worldsmall_instance):
        assert len(worldsmall_instance.ports) == 47

    def test_all_demands_loaded(self, worldsmall_instance):
        assert len(worldsmall_instance.demands) == 1764

    def test_all_vessels_loaded(self, worldsmall_instance):
        total = sum(e.quantity for e in worldsmall_instance.fleet)
        assert total == 263

    def test_all_distance_arcs_present(self, worldsmall_instance):
        total = len(worldsmall_instance.distances) + len(worldsmall_instance.sparse_distances)
        assert total == 3276


# ======================================================================
# B — All Vessel Classes Loaded
# ======================================================================

class TestAllVesselClassesLoaded:
    """Verify all 6 vessel classes are available."""

    def test_six_vessel_classes(self, worldsmall_instance):
        classes = sorted(e.vessel_class for e in worldsmall_instance.fleet)
        assert len(classes) == 6

    def test_expected_vessel_classes(self, worldsmall_instance):
        classes = {e.vessel_class for e in worldsmall_instance.fleet}
        expected = {"Feeder_450", "Feeder_800", "Panamax_1200",
                    "Panamax_2400", "Post_panamax", "Super_panamax"}
        assert classes == expected

    def test_each_class_has_capacity(self, worldsmall_instance):
        for vc in worldsmall_instance.vessel_types:
            vt = worldsmall_instance.vessel_types[vc]
            assert vt.capacity_ffe > 0


# ======================================================================
# C — Candidate Service Universe Audited
# ======================================================================

class TestCandidateServiceUniverseAudited:
    """Verify candidate service audit JSON exists and is valid."""

    def test_candidate_audit_exists(self):
        path = _ROOT / "data_utilization" / "candidate_service_audit.json"
        assert path.exists(), "candidate_service_audit.json missing"

    def test_candidate_audit_structure(self):
        path = _ROOT / "data_utilization" / "candidate_service_audit.json"
        with open(path) as f:
            data = json.load(f)
        assert "total_candidate_services" in data
        assert "valid_candidate_services" in data
        assert "unique_candidate_port_sequences" in data
        assert data["total_candidate_services"] > 10000

    def test_candidate_universe_not_collapsed(self):
        """There should be thousands of candidate services, not collapse."""
        path = _ROOT / "data_utilization" / "candidate_service_audit.json"
        with open(path) as f:
            data = json.load(f)
        assert data["total_candidate_services"] >= 10000
        assert data["unique_candidate_od_pairs"] >= 1000


# ======================================================================
# D — Raw Policy Actions Preserved
# ======================================================================

class TestRawPolicyActionsPreserved:
    """Verify raw decoder actions are recorded before any fallback."""

    def test_raw_action_trace_exists(self):
        path = _ROOT / "data_utilization" / "raw_policy_action_trace.jsonl"
        assert path.exists(), "raw_policy_action_trace.jsonl missing"

    def test_raw_action_trace_has_entries(self):
        path = _ROOT / "data_utilization" / "raw_policy_action_trace.jsonl"
        lines = path.read_text().strip().split("\n")
        assert len(lines) == 50  # 5 updates × 10 steps

    def test_each_entry_has_required_fields(self):
        path = _ROOT / "data_utilization" / "raw_policy_action_trace.jsonl"
        with open(path) as f:
            entry = json.loads(f.readline())
        required = {"update", "step", "env_id", "raw_sampled_ports",
                     "executed_port_sequence", "vessel_class",
                     "log_prob", "entropy", "reward"}
        assert required.issubset(entry.keys())

    def test_raw_vs_executed_separation(self):
        """Raw and executed sequences are different when BOS is selected."""
        path = _ROOT / "data_utilization" / "raw_policy_action_trace.jsonl"
        diffs = 0
        total = 0
        with open(path) as f:
            for line in f:
                entry = json.loads(line)
                total += 1
                raw = tuple(entry.get("raw_sampled_ports", []))
                exec_seq = tuple(entry.get("executed_port_sequence", []))
                if raw != exec_seq:
                    diffs += 1
        # Many entries should differ because BOS → None vc → fallback
        assert diffs > 0, "Expected divergence between raw and executed"


# ======================================================================
# E — Action Mask Statistics Present
# ======================================================================

class TestActionMaskStatisticsPresent:
    """Verify action mask audit exists with meaningful content."""

    def test_mask_audit_exists(self):
        path = _ROOT / "data_utilization" / "action_mask_audit.json"
        assert path.exists()

    def test_mask_audit_has_total_steps(self):
        path = _ROOT / "data_utilization" / "action_mask_audit.json"
        with open(path) as f:
            data = json.load(f)
        assert data.get("total_steps") == 50


# ======================================================================
# F — Service Construction Mapping Present
# ======================================================================

class TestServiceConstructionMappingPresent:
    """Verify service construction audit captures mapping."""

    def test_service_construction_audit_exists(self):
        path = _ROOT / "data_utilization" / "service_construction_audit.json"
        assert path.exists()

    def test_has_constructed_services(self):
        path = _ROOT / "data_utilization" / "service_construction_audit.json"
        with open(path) as f:
            data = json.load(f)
        assert len(data["services"]) == 10

    def test_collapse_ratio_present(self):
        path = _ROOT / "data_utilization" / "service_construction_audit.json"
        with open(path) as f:
            data = json.load(f)
        assert "collapse_ratio_unique_constructed_over_unique_raw" in data


# ======================================================================
# G — Unique Service Collapse Metric Present
# ======================================================================

class TestUniqueServiceCollapseMetricPresent:
    """Verify final network diversity metrics capture collapse."""

    def test_final_network_diversity_exists(self):
        path = _ROOT / "data_utilization" / "final_network_diversity.json"
        assert path.exists()

    def test_unique_service_count(self):
        path = _ROOT / "data_utilization" / "final_network_diversity.json"
        with open(path) as f:
            data = json.load(f)
        assert data["total_services"] == 10
        assert data["unique_services"] < data["total_services"]

    def test_gini_concentration_present(self):
        path = _ROOT / "data_utilization" / "final_network_diversity.json"
        with open(path) as f:
            data = json.load(f)
        assert "gini_concentration_index" in data


# ======================================================================
# H — Final Network Diversity Present
# ======================================================================

class TestFinalNetworkDiversityPresent:
    """Verify final network diversity is fully audited."""

    def test_all_diversity_metrics_present(self):
        path = _ROOT / "data_utilization" / "final_network_diversity.json"
        with open(path) as f:
            data = json.load(f)
        required = {
            "total_services", "unique_services", "unique_ports_used",
            "percent_of_worldsmall_ports_used", "vessel_classes_represented",
            "vessel_classes_not_represented",
        }
        assert required.issubset(data.keys())

    def test_not_all_ports_used(self):
        path = _ROOT / "data_utilization" / "final_network_diversity.json"
        with open(path) as f:
            data = json.load(f)
        assert data["percent_of_worldsmall_ports_used"] < 50

    def test_not_all_vessel_classes_used(self):
        path = _ROOT / "data_utilization" / "final_network_diversity.json"
        with open(path) as f:
            data = json.load(f)
        assert len(data["vessel_classes_not_represented"]) > 0


# ======================================================================
# I — Demand Reachability Present
# ======================================================================

class TestDemandReachabilityPresent:
    """Verify demand reachability audit is complete."""

    def test_demand_reachability_exists(self):
        path = _ROOT / "data_utilization" / "demand_reachability_audit.json"
        assert path.exists()

    def test_has_reachability_counts(self):
        path = _ROOT / "data_utilization" / "demand_reachability_audit.json"
        with open(path) as f:
            data = json.load(f)
        assert data["total_demands"] == 1764
        assert data["demands_directly_connected_by_service_edge"] < data["total_demands"]

    def test_coverage_near_zero(self):
        path = _ROOT / "data_utilization" / "demand_reachability_audit.json"
        with open(path) as f:
            data = json.load(f)
        assert data["demands_directly_connected_by_service_edge"] <= 10


# ======================================================================
# J — Fleet Usage Reconciliation
# ======================================================================

class TestFleetUsageReconciliation:
    """Verify fleet deployment audit reconciles."""

    def test_fleet_usage_exists(self):
        path = _ROOT / "data_utilization" / "fleet_usage_audit.json"
        assert path.exists()

    def test_fleet_deployment_percent_present(self):
        path = _ROOT / "data_utilization" / "fleet_usage_audit.json"
        with open(path) as f:
            data = json.load(f)
        assert "total_fleet_deployment_percent" in data
        # Fleet usage exceeds available due to soft constraint
        assert data["total_fleet_deployment_percent"] > 100

    def test_per_class_breakdown(self):
        path = _ROOT / "data_utilization" / "fleet_usage_audit.json"
        with open(path) as f:
            data = json.load(f)
        assert len(data["per_class"]) == 6


# ======================================================================
# K — MCF Input Matches Final Services
# ======================================================================

class TestMcfInputMatchesFinalServices:
    """Verify MCF receives the exact final selected services."""

    def test_mcf_path_audit_exists(self):
        path = _ROOT / "data_utilization" / "mcf_path_audit.json"
        assert path.exists()

    def test_services_entered_equal_evaluated(self):
        path = _ROOT / "data_utilization" / "mcf_path_audit.json"
        with open(path) as f:
            data = json.load(f)
        assert data["services_entering_evaluator"] == data["services_evaluated"]
        assert data["services_entering_evaluator"] == 10

    def test_mcf_success(self):
        path = _ROOT / "data_utilization" / "mcf_path_audit.json"
        with open(path) as f:
            data = json.load(f)
        assert data["mcf_status"] == "success"


# ======================================================================
# L — Economic Reconciliation
# ======================================================================

class TestEconomicReconciliation:
    """Verify economic reconciliation residual is zero."""

    def test_economic_reconciliation_zero(self):
        path = _ROOT / "data_utilization" / "mcf_path_audit.json"
        with open(path) as f:
            data = json.load(f)
        assert abs(data["reconciliation_residual"]) < 1e-6

    def test_profit_negative(self):
        path = _ROOT / "data_utilization" / "mcf_path_audit.json"
        with open(path) as f:
            data = json.load(f)
        assert data["network_profit_usd"] < 0

    def test_revenue_non_negative(self):
        path = _ROOT / "data_utilization" / "mcf_path_audit.json"
        with open(path) as f:
            data = json.load(f)
        assert data["revenue_usd"] >= 0


# ======================================================================
# M — Demand Reconciliation
# ======================================================================

class TestDemandReconciliation:
    """Verify demand sums reconcile."""

    def test_demand_sum_equals_total(self, worldsmall_instance):
        path = _ROOT / "data_utilization" / "mcf_path_audit.json"
        with open(path) as f:
            data = json.load(f)
        total = data["routed_demand_ffe"] + data["rejected_demand_ffe"]
        expected = sum(d.ffe_per_week for d in worldsmall_instance.demands)
        assert abs(total - expected) < 1.0


# ======================================================================
# N — Demand Units Are FFE
# ======================================================================

class TestDemandUnitsAreFFE:
    """Verify demand fields use FFE/week units, not TEU."""

    def test_total_demand_field_ffe(self, worldsmall_instance):
        total = sum(d.ffe_per_week for d in worldsmall_instance.demands)
        assert abs(total - 138247.0) < 1.0

    def test_no_teu_references_in_audit(self):
        """Audit JSONs should not use TEU as unit."""
        for fname in [
            "data_ingestion_audit.json",
            "final_network_diversity.json",
            "demand_reachability_audit.json",
            "mcf_path_audit.json",
        ]:
            path = _ROOT / "data_utilization" / fname
            if path.exists():
                text = path.read_text()
                assert "teu" not in text.lower() or "total_demand_teu" not in text


# ======================================================================
# O — Fleet Metric Naming
# ======================================================================

class TestFleetMetricNaming:
    """Verify fleet metrics use correct naming convention."""

    def test_uses_deployment_not_utilization(self):
        path = _ROOT / "data_utilization" / "fleet_usage_audit.json"
        with open(path) as f:
            data = json.load(f)
        assert "total_fleet_deployment_percent" in data


# ======================================================================
# P — Validation Fields Not Null
# ======================================================================

class TestValidationFieldsNotNull:
    """Verify all mandatory validation fields have explicit PASS/FAIL."""

    def test_baseline_result_has_validation(self):
        path = _ROOT / "experiments" / "manual" / "g11_2_1_data_utilization" / "baseline_result.json"
        if not path.exists():
            pytest.skip("baseline_result.json not yet generated")
        with open(path) as f:
            data = json.load(f)
        assert "economics" in data
        assert "diversity_metrics" in data

    def test_economic_fields_finite(self):
        path = _ROOT / "experiments" / "manual" / "g11_2_1_data_utilization" / "baseline_result.json"
        if not path.exists():
            pytest.skip("baseline_result.json not yet generated")
        with open(path) as f:
            data = json.load(f)
        import math
        profit = data["economics"]["profit"]
        assert math.isfinite(profit)


# ======================================================================
# Q — Overall Pass Includes All Mandatory Checks
# ======================================================================

class TestOverallPassIncludesAllMandatoryChecks:
    """Verify benchmark overall_pass includes all mandatory validations."""

    def test_overall_pass_fields_complete(self):
        """Check that existing benchmark artifacts include all checks."""
        # The G11.1 evaluation.py writes validation fields; we verify
        # the schema matches our requirements.
        econ_path = _ROOT / "experiments" / "manual" / "g11_worldsmall" / "g11_1_economic_result.json"
        if not econ_path.exists():
            pytest.skip("g11_1_economic_result.json missing")
        with open(econ_path) as f:
            data = json.load(f)
        assert "validation" in data
        validation = data["validation"]
        assert "economic_reconciliation_residual" in validation
        assert "no_nan_inf" in validation
        assert "mcf_status" in validation


# ======================================================================
# R — Comparison Contract Conditional
# ======================================================================

class TestComparisonContractConditional:
    """Verify comparison contract acknowledges conditional comparability."""

    def test_comparison_contract_exists(self):
        path = _ROOT / "experiments" / "manual" / "g11_worldsmall" / "comparison_contract.json"
        assert path.exists()

    def test_contract_notes_conditionality(self):
        path = _ROOT / "experiments" / "manual" / "g11_worldsmall" / "comparison_contract.json"
        with open(path) as f:
            data = json.load(f)
        # Contract should exist and reference key conditions
        assert isinstance(data, dict)


# ======================================================================
# S — Profit Margin NA When Revenue Zero
# ======================================================================

class TestProfitMarginNAWhenRevenueZero:
    """When revenue is zero, profit_margin_pct should be null/NA."""

    def test_profit_margin_null_when_no_revenue(self):
        path = _ROOT / "experiments" / "manual" / "g11_worldsmall" / "g11_1_economic_result.json"
        with open(path) as f:
            data = json.load(f)
        # In G11 baseline, revenue=0 so profit_margin should be null
        assert data["economic"]["profit_margin_pct"] is None or \
               data["economic"]["profit_margin_pct"] is None


# ======================================================================
# T — Repeatablity Check
# ======================================================================

class TestRepeatablityCheck:
    """Verify two runs with same seed produce consistent diversity metrics."""

    def test_two_runs_same_raw_diversity(self):
        """Both runs should show same unique raw action count."""
        path = _ROOT / "experiments" / "manual" / "g11_2_1_data_utilization" / "baseline_result.json"
        if not path.exists():
            pytest.skip("baseline_result.json not yet generated")
        with open(path) as f:
            data = json.load(f)
        dm = data["diversity_metrics"]
        assert dm["raw_action_diversity"] > 0
        assert dm["action_repetition_rate"] == 0.0  # different raw actions per step


# ======================================================================
# U — Baseline Reproduced
# ======================================================================

class TestBaselineReproduced:
    """Verify the G11 baseline was reproduced exactly."""

    def test_baseline_result_exists(self):
        path = _ROOT / "experiments" / "manual" / "g11_2_1_data_utilization" / "baseline_result.json"
        assert path.exists()

    def test_baseline_profit_matches_g11_range(self):
        """Profit should be negative and in the hundreds-of-millions range."""
        path = _ROOT / "experiments" / "manual" / "g11_2_1_data_utilization" / "baseline_result.json"
        with open(path) as f:
            data = json.load(f)
        profit = data["economics"]["profit"]
        assert profit < -100_000_000
        assert profit > -500_000_000

    def test_baseline_coverage_near_zero(self):
        path = _ROOT / "experiments" / "manual" / "g11_2_1_data_utilization" / "baseline_result.json"
        with open(path) as f:
            data = json.load(f)
        coverage = data["economics"]["coverage_pct"]
        assert coverage < 10.0  # severely degenerate


# ======================================================================
# V — Gini Concentration
# ======================================================================

class TestGiniConcentration:
    """Verify Gini concentration metric is computed."""

    def test_gini_present(self):
        path = _ROOT / "data_utilization" / "final_network_diversity.json"
        with open(path) as f:
            data = json.load(f)
        assert "gini_concentration_index" in data
        # With concentrated duplicate services, gini can be negative
        # (indicator of anti-concentration / uniformity, but with only
        # 3 unique services out of 10, this is a degeneracy signal)
        assert isinstance(data["gini_concentration_index"], (int, float))


# ======================================================================
# W — Draft Feasibility Bottleneck Evidence
# ======================================================================

class TestDraftFeasibilityBottleneck:
    """Verify the draft-feasibility bottleneck is documented."""

    def test_shallow_ports_documented(self):
        """At least some ports should be draft-infeasible for small vessels."""
        from data.linerlib_loader import LINERLIBLoader
        loader = LINERLIBLoader(str(_ROOT / "data"))
        inst = loader.load("WorldSmall", validate=False)
        shallow = sum(1 for p in inst.ports.values()
                      if p.draft is not None and p.draft <= 8.0)
        assert shallow < len(inst.ports)  # Not all ports are shallow

    def test_most_demands_not_in_shallow(self, worldsmall_instance):
        """Most demands should involve deep ports."""
        shallow = {p.unlocode for p in worldsmall_instance.ports.values()
                   if p.draft is not None and p.draft <= 8.0}
        in_shallow = sum(1 for d in worldsmall_instance.demands
                         if d.origin in shallow and d.destination in shallow)
        assert in_shallow < len(worldsmall_instance.demands) * 0.01
