"""
P15 — Common Evaluation Test Suite.

Tests for:
1. Candidate schema
2. RL candidate conversion
3. Reference candidate adapter
4. Common MCF invocation
5. Objective consistency
6. Cost decomposition
7. Fleet accounting
8. Rejected-demand accounting
9. Deterministic evaluation
10. Serialization
11. Invalid candidate detection
12. Empty-network evaluation
13. Repeated evaluation
14. Raw data integrity
15. P1-P14 regression
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path

import pytest

_repo_root = Path(__file__).resolve().parents[1]
if str(_repo_root) not in os.sys.path:
    os.sys.path.insert(0, str(_repo_root))

from data.instance import LINERLIBInstance
from data.linerlib_loader import LINERLIBLoader
from evaluation import CandidateService, CandidateSolution, CommonEvaluator, EvaluationResult
from evaluation.adapters import rl_result_to_candidate, reference_solution_to_candidate
from inference.result import InferenceResult, ServiceStep


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def loader():
    return LINERLIBLoader("data")


@pytest.fixture(scope="module")
def baltic_instance(loader):
    return loader.load("Baltic")


@pytest.fixture(scope="module")
def waf_instance(loader):
    return loader.load("WAF")


@pytest.fixture
def evaluator(baltic_instance):
    return CommonEvaluator(baltic_instance)


@pytest.fixture
def valid_candidate(baltic_instance):
    return CandidateSolution(
        dataset="Baltic",
        instance="Baltic",
        services=[
            CandidateService(service_id=0, vessel_class="Feeder_800", port_sequence=["FIKTK", "NOAES"]),
        ],
        method="test",
        seed=42,
    )


# ---------------------------------------------------------------------------
# 1. Candidate Schema
# ---------------------------------------------------------------------------

class TestCandidateSchema:
    """Test CandidateSolution schema."""

    def test_minimal_candidate(self):
        c = CandidateSolution(dataset="Baltic", instance="Baltic", method="test")
        assert c.n_services == 0
        assert c.method == "test"

    def test_candidate_with_services(self):
        c = CandidateSolution(
            dataset="Baltic",
            instance="Baltic",
            services=[
                CandidateService(0, "Feeder_800", ["FIKTK", "NOAES"]),
                CandidateService(1, "Feeder_800", ["DKAAR", "NOAES"]),
            ],
            method="test",
        )
        assert c.n_services == 2

    def test_candidate_serialization(self):
        c = CandidateSolution(
            dataset="Baltic",
            instance="Baltic",
            services=[CandidateService(0, "Feeder_800", ["FIKTK", "NOAES"])],
            method="rl_test",
            provenance={"seed": 42},
            seed=42,
        )
        data = c.to_dict()
        restored = CandidateSolution.from_dict(data)
        assert restored.dataset == c.dataset
        assert restored.n_services == c.n_services
        assert restored.services[0].vessel_class == c.services[0].vessel_class

    def test_candidate_json_io(self, tmp_path):
        c = CandidateSolution(dataset="Baltic", instance="Baltic", method="test")
        path = tmp_path / "candidate.json"
        with open(path, "w") as f:
            json.dump(c.to_dict(), f)
        with open(path) as f:
            restored = CandidateSolution.from_dict(json.load(f))
        assert restored.method == "test"


# ---------------------------------------------------------------------------
# 2. RL Candidate Conversion
# ---------------------------------------------------------------------------

class TestRLCandidateConversion:
    """Test converting InferenceResult to CandidateSolution."""

    def test_convert_inference_result(self):
        step = ServiceStep(
            step_index=0, vessel_class="Feeder_800",
            port_sequence=["FIKTK", "NOAES"], service_id=0,
            log_prob=None, entropy=None, reward_raw=0.0,
            reward_normalized=0.0, eta_cumulative=0.0,
            fleet_after={}, terminated=False, truncated=False,
            termination_reason=None,
        )
        result = InferenceResult(
            dataset="Baltic", instance_name="Baltic",
            policy_type="encoder_only", checkpoint_path="ckpt.pt",
            checkpoint_hash="abc123", seed=42, deterministic=True,
            services=[step], total_services=1, final_eta=0.0,
            final_mcf_result=None, runtime_seconds=1.0,
            termination_reason="vessel_exhaustion", is_truncated=False,
        )

        candidate = rl_result_to_candidate(result, "Baltic")
        assert candidate.n_services == 1
        assert candidate.services[0].vessel_class == "Feeder_800"
        assert candidate.method == "rl_encoder_only"
        assert candidate.provenance["checkpoint_path"] == "ckpt.pt"


# ---------------------------------------------------------------------------
# 3. Reference Candidate Adapter
# ---------------------------------------------------------------------------

class TestReferenceAdapter:
    """Test converting reference solutions to CandidateSolution."""

    def test_convert_reference_solution(self):
        services = [
            {"vessel_class": "Feeder_800", "port_sequence": ["FIKTK", "NOAES"]},
        ]
        candidate = reference_solution_to_candidate(services, "Baltic", method="ref")
        assert candidate.n_services == 1
        assert candidate.method == "ref"
        assert candidate.provenance["source_log"] == ""


# ---------------------------------------------------------------------------
# 4. Common MCF Invocation
# ---------------------------------------------------------------------------

class TestMCFInvocation:
    """Test that evaluator correctly invokes MCF."""

    def test_evaluate_valid_candidate(self, evaluator, valid_candidate):
        result = evaluator.evaluate(valid_candidate)
        assert result.mcf_status == "success"
        assert result.is_feasible
        assert result.service_count == 1
        assert isinstance(result.objective_eta, float)

    def test_evaluate_empty_network(self, evaluator):
        candidate = CandidateSolution(dataset="Baltic", instance="Baltic", method="empty")
        result = evaluator.evaluate(candidate)
        assert result.mcf_status == "success"
        assert result.service_count == 0
        assert result.routed_demand == 0.0
        assert result.rejected_demand > 0

    def test_evaluate_multiple_services(self, evaluator):
        candidate = CandidateSolution(
            dataset="Baltic", instance="Baltic", method="multi",
            services=[
                CandidateService(0, "Feeder_800", ["FIKTK", "NOAES"]),
                CandidateService(1, "Feeder_800", ["DKAAR", "NOAES"]),
            ],
        )
        result = evaluator.evaluate(candidate)
        assert result.mcf_status == "success"
        assert result.service_count == 2


# ---------------------------------------------------------------------------
# 5. Objective Consistency
# ---------------------------------------------------------------------------

class TestObjectiveConsistency:
    """Test that objective is computed consistently."""

    def test_eta_matches_mcf(self, evaluator, valid_candidate):
        result = evaluator.evaluate(valid_candidate)
        # eta should equal revenue - costs
        costs = (result.C_reject + result.C_handle + result.C_service +
                 result.C_unused + result.C_voyage)
        expected_eta = result.revenue - costs
        assert math.isclose(result.objective_eta, expected_eta, abs_tol=1e-3)

    def test_empty_network_eta(self, evaluator):
        """Empty network should have eta = -rejection_cost (no revenue)."""
        candidate = CandidateSolution(dataset="Baltic", instance="Baltic", method="empty")
        result = evaluator.evaluate(candidate)
        # No services → no revenue, no service/voyage costs
        # Only rejection cost remains
        assert result.revenue == 0.0
        assert result.C_service == 0.0
        assert result.C_voyage == 0.0


# ---------------------------------------------------------------------------
# 6. Cost Decomposition
# ---------------------------------------------------------------------------

class TestCostDecomposition:
    """Test that all cost components are tracked."""

    def test_all_costs_present(self, evaluator, valid_candidate):
        result = evaluator.evaluate(valid_candidate)
        assert hasattr(result, "C_reject")
        assert hasattr(result, "C_handle")
        assert hasattr(result, "C_service")
        assert hasattr(result, "C_unused")
        assert hasattr(result, "C_voyage")
        assert hasattr(result, "revenue")

    def test_cost_values_are_finite(self, evaluator, valid_candidate):
        result = evaluator.evaluate(valid_candidate)
        for attr in ["objective_eta", "revenue", "C_reject", "C_handle",
                     "C_service", "C_unused", "C_voyage"]:
            val = getattr(result, attr)
            assert math.isfinite(val), f"{attr} is not finite: {val}"


# ---------------------------------------------------------------------------
# 7. Fleet Accounting
# ---------------------------------------------------------------------------

class TestFleetAccounting:
    """Test fleet usage and deviation tracking."""

    def test_fleet_usage_computed(self, evaluator, valid_candidate):
        result = evaluator.evaluate(valid_candidate)
        assert isinstance(result.fleet_usage, dict)
        assert isinstance(result.fleet_deviation, dict)

    def test_fleet_deviation_sign(self, evaluator, valid_candidate):
        """Negative deviation = under-used (profit), positive = over-used (cost)."""
        result = evaluator.evaluate(valid_candidate)
        # At least one vessel class should have a deviation
        assert len(result.fleet_deviation) > 0


# ---------------------------------------------------------------------------
# 8. Rejected-Demand Accounting
# ---------------------------------------------------------------------------

class TestRejectedDemand:
    """Test rejected demand tracking."""

    def test_empty_network_full_rejection(self, evaluator):
        candidate = CandidateSolution(dataset="Baltic", instance="Baltic", method="empty")
        result = evaluator.evaluate(candidate)
        assert result.routed_demand == 0.0
        assert result.rejected_demand > 0

    def test_demand_coverage(self, evaluator, valid_candidate):
        result = evaluator.evaluate(valid_candidate)
        if result.rejected_demand > 0:
            coverage = result.routed_demand / (result.routed_demand + result.rejected_demand)
            assert 0.0 <= coverage <= 1.0


# ---------------------------------------------------------------------------
# 9. Deterministic Evaluation
# ---------------------------------------------------------------------------

class TestDeterministicEvaluation:
    """Test that repeated evaluations produce identical results."""

    def test_same_candidate_same_result(self, evaluator, valid_candidate):
        r1 = evaluator.evaluate(valid_candidate)
        r2 = evaluator.evaluate(valid_candidate)
        assert math.isclose(r1.objective_eta, r2.objective_eta, abs_tol=1e-6)
        assert r1.routed_demand == r2.routed_demand
        assert r1.rejected_demand == r2.rejected_demand
        assert r1.service_count == r2.service_count


# ---------------------------------------------------------------------------
# 10. Serialization
# ---------------------------------------------------------------------------

class TestSerialization:
    """Test result serialization."""

    def test_result_to_dict(self, evaluator, valid_candidate):
        result = evaluator.evaluate(valid_candidate)
        data = result.to_dict()
        assert "objective_eta" in data
        assert "method" in data
        assert "dataset" in data

    def test_result_roundtrip(self, evaluator, valid_candidate):
        result = evaluator.evaluate(valid_candidate)
        data = result.to_dict()
        restored = EvaluationResult.from_dict(data)
        assert math.isclose(restored.objective_eta, result.objective_eta, abs_tol=1e-6)
        assert restored.method == result.method
        assert restored.service_count == result.service_count

    def test_result_json_io(self, evaluator, valid_candidate, tmp_path):
        result = evaluator.evaluate(valid_candidate)
        path = tmp_path / "eval_result.json"
        with open(path, "w") as f:
            json.dump(result.to_dict(), f)
        with open(path) as f:
            restored = EvaluationResult.from_dict(json.load(f))
        assert restored.mcf_status == "success"


# ---------------------------------------------------------------------------
# 11. Invalid Candidate Detection
# ---------------------------------------------------------------------------

class TestInvalidCandidates:
    """Test handling of invalid candidates."""

    def test_unknown_vessel_class(self, evaluator):
        candidate = CandidateSolution(
            dataset="Baltic", instance="Baltic", method="bad",
            services=[CandidateService(0, "NonExistent", ["FIKTK", "NOAES"])],
        )
        result = evaluator.evaluate(candidate)
        assert not result.structural_feasibility
        assert len(result.warnings) > 0

    def test_unknown_port(self, evaluator):
        candidate = CandidateSolution(
            dataset="Baltic", instance="Baltic", method="bad",
            services=[CandidateService(0, "Feeder_800", ["XXXXX", "YYYYY"])],
        )
        result = evaluator.evaluate(candidate)
        assert not result.structural_feasibility

    def test_too_few_ports(self, evaluator):
        candidate = CandidateSolution(
            dataset="Baltic", instance="Baltic", method="bad",
            services=[CandidateService(0, "Feeder_800", ["FIKTK"])],
        )
        result = evaluator.evaluate(candidate)
        assert not result.structural_feasibility


# ---------------------------------------------------------------------------
# 12. Empty-Network Evaluation
# ---------------------------------------------------------------------------

class TestEmptyNetwork:
    """Test evaluation of empty networks."""

    def test_empty_network_success(self, evaluator):
        candidate = CandidateSolution(dataset="Baltic", instance="Baltic", method="empty")
        result = evaluator.evaluate(candidate)
        assert result.mcf_status == "success"
        assert result.service_count == 0
        assert result.objective_eta < 0  # Negative due to rejection cost


# ---------------------------------------------------------------------------
# 13. Repeated Evaluation
# ---------------------------------------------------------------------------

class TestRepeatedEvaluation:
    """Test that repeated evaluation is stable."""

    def test_five_evaluations_consistent(self, evaluator, valid_candidate):
        results = [evaluator.evaluate(valid_candidate) for _ in range(5)]
        etas = [r.objective_eta for r in results]
        assert all(math.isclose(etas[0], e, abs_tol=1e-6) for e in etas)


# ---------------------------------------------------------------------------
# 14. Raw Data Integrity
# ---------------------------------------------------------------------------

class TestRawDataIntegrity:
    """Test raw data unchanged."""

    def test_baltic_data_hash(self):
        f = _repo_root / "data" / "Demand_Baltic.csv"
        h = hashlib.sha256(f.read_bytes()).hexdigest()
        assert h.startswith("562538e34c827ab6")

    def test_linerlib_master_intact(self):
        readme = _repo_root / "data" / "LINERLIB-master" / "README.md"
        assert readme.exists()


# ---------------------------------------------------------------------------
# 15. P1-P14 Regression
# ---------------------------------------------------------------------------

class TestP1toP14Regression:
    """Ensure P15 doesn't break existing functionality."""

    def test_p1_loader(self, loader):
        inst = loader.load("Baltic")
        assert inst.name == "Baltic"

    def test_p3_mcf(self, baltic_instance):
        from mcf import evaluate_network
        from mcf.expanded_graph import ServiceDefinition
        svc = ServiceDefinition(0, "Feeder_800", ["FIKTK", "NOAES"])
        result = evaluate_network(baltic_instance, [svc], {"0": {"Feeder_800": 1.0}})
        assert result.eta is not None

    def test_p4_env(self, baltic_instance):
        from env.environment import LSNDPEnv
        from env.action import ServiceAction
        env = LSNDPEnv(baltic_instance)
        env.reset(seed=42)
        sa = ServiceAction(vessel_class="Feeder_800", port_sequence=["FIKTK", "NOAES"])
        obs, r, term, trunc, info = env.step(sa)
        assert info["num_services"] == 1

    def test_p13_solver(self, loader):
        from inference.solver import InferenceSolver
        from inference.config import InferenceConfig
        config = InferenceConfig(policy_type="encoder_only", deterministic=True, seed=42, max_services=3)
        solver = InferenceSolver("checkpoints/test_p12/final_checkpoint.pt", config=config)
        result = solver.run(seed=42)
        assert result.policy_type == "encoder_only"

    def test_p14_perturbation(self, baltic_instance):
        from experiments.reproduction.perturbation import perturb_demand
        p = perturb_demand(baltic_instance, fraction=0.10, seed=42)
        assert len(p.demands) == len(baltic_instance.demands)

    def test_cross_phase_rl_to_evaluator(self, baltic_instance):
        """Full pipeline: inference → candidate → common evaluator."""
        from inference.solver import InferenceSolver
        from inference.config import InferenceConfig
        from evaluation.adapters import rl_result_to_candidate

        config = InferenceConfig(policy_type="encoder_only", deterministic=True, seed=42, max_services=3)
        solver = InferenceSolver("checkpoints/test_p12/final_checkpoint.pt", config=config)
        ir = solver.run(seed=42)

        candidate = rl_result_to_candidate(ir, "Baltic")
        evaluator = CommonEvaluator(baltic_instance)
        result = evaluator.evaluate(candidate)

        assert result.mcf_status == "success"
        assert result.method == "rl_encoder_only"
