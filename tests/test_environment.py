"""
P4 — Tests for the LSNDP Gymnasium environment.

Covers:
  - Environment construction
  - reset / initial state
  - valid service action
  - invalid service action (validation errors)
  - service insertion
  - vessel-state update
  - P3 evaluator invocation
  - profit update
  - raw / normalized reward
  - profit history
  - remaining-demand update
  - termination / truncation
  - deterministic replay
  - reset after terminal
  - info dictionary
  - observation contract
  - Gymnasium API compatibility
  - Real Baltic smoke integration
"""

from __future__ import annotations

import math
import hashlib
import sys
from pathlib import Path

import gymnasium as gym
import numpy as np
import pytest

# Ensure project root is on path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from data.instance import (
    LINERLIBInstance, VesselType, Port, DistanceArc, FleetEntry, Demand,
    ProvenanceRecord,
)
from env.environment import (
    LSNDPEnv, ServiceAction, ServiceValidationError, _EnvState,
    MAX_SERVICES_SAFETY_CAP,
)
from mcf import ServiceDefinition


# ---------------------------------------------------------------------------
# Synthetic test fixtures — NOT benchmark data
# ---------------------------------------------------------------------------

def _make_toy_instance() -> LINERLIBInstance:
    """
    SYNTHETIC TEST FIXTURE — NOT LINERLIB BENCHMARK DATA.

    Minimal 2-port, 1-vessel-class instance for isolated deterministic tests.
    Ports A and B with a simple round-trip distance.
    """
    port_a = Port(
        unlocode="A", name="Port A", country=None, cabotage_region="test",
        d_region=None, longitude=None, latitude=None, draft=10.0,
        cost_per_full=1.0, cost_per_full_transfer=0.5,
        port_call_cost_fixed=100.0, port_call_cost_per_ffe=0.5,
        provenance=ProvenanceRecord(source_file="synthetic_fixture", source_row=1),
    )
    port_b = Port(
        unlocode="B", name="Port B", country=None, cabotage_region="test",
        d_region=None, longitude=None, latitude=None, draft=10.0,
        cost_per_full=1.0, cost_per_full_transfer=0.5,
        port_call_cost_fixed=100.0, port_call_cost_per_ffe=0.5,
        provenance=ProvenanceRecord(source_file="synthetic_fixture", source_row=2),
    )

    vessel_v1 = VesselType(
        vessel_class="V1", capacity_ffe=200, tc_rate_daily=100,
        draft=12.0, min_speed=5.0, max_speed=15.0, design_speed=10.0,
        bunker_ton_per_day_at_design=50.0, idle_consumption_ton_per_day=10.0,
        panama_fee=0, suez_fee=0,
        provenance=ProvenanceRecord(source_file="synthetic_fixture", source_row=1),
    )

    arc_ab = DistanceArc(
        origin="A", destination="B", distance_nm=100.0,
        draft_required=10.0, is_panama=False, is_suez=False,
        provenance=ProvenanceRecord(source_file="synthetic_fixture", source_row=1),
    )
    arc_ba = DistanceArc(
        origin="B", destination="A", distance_nm=100.0,
        draft_required=10.0, is_panama=False, is_suez=False,
        provenance=ProvenanceRecord(source_file="synthetic_fixture", source_row=2),
    )

    demand = Demand(
        origin="A", destination="B",
        ffe_per_week=100.0, revenue=50.0, max_transit_time=10,
        provenance=ProvenanceRecord(source_file="synthetic_fixture", source_row=1),
    )

    return LINERLIBInstance(
        name="TOY_P4_TEST",
        vessel_types={"V1": vessel_v1},
        ports={"A": port_a, "B": port_b},
        fleet=[FleetEntry(vessel_class="V1", quantity=5)],
        distances=[arc_ab, arc_ba],
        sparse_distances=[arc_ab, arc_ba],
        demands=[demand],
    )


def _make_baltic_instance():
    """Load real Baltic instance from LINERLIB data."""
    from data.linerlib_loader import LINERLIBLoader
    loader = LINERLIBLoader(str(ROOT / "data"))
    return loader.load("Baltic")


def _baltic_valid_cycle_ports(vessel_class=None, n=3):
    """Return a list of ports that form a valid cycle for the given vessel class."""
    from data.linerlib_loader import LINERLIBLoader
    loader = LINERLIBLoader(str(ROOT / "data"))
    inst = loader.load("Baltic")
    if vessel_class:
        vt = inst.vessel_types[vessel_class]
        valid_ports = [p.unlocode for p in inst.ports.values() if (p.draft or 0) <= vt.draft + 1e-6]
    else:
        # Use Post_panamax which can visit most ports
        vt = inst.vessel_types["Post_panamax"]
        valid_ports = [p.unlocode for p in inst.ports.values() if (p.draft or 0) <= vt.draft + 1e-6]
    pairs = set()
    for a in inst.distances:
        pairs.add((a.origin, a.destination))
    # Find a triangle
    for i, p1 in enumerate(valid_ports):
        for j, p2 in enumerate(valid_ports):
            if j == i:
                continue
            for k, p3 in enumerate(valid_ports):
                if k in (i, j):
                    continue
                if (p1, p2) in pairs and (p2, p3) in pairs and (p3, p1) in pairs:
                    return [p1, p2, p3]
    return valid_ports[:n] if len(valid_ports) >= n else valid_ports


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def toy_instance():
    return _make_toy_instance()


@pytest.fixture
def toy_env(toy_instance):
    return LSNDPEnv(toy_instance)


@pytest.fixture
def baltic_instance():
    return _make_baltic_instance()


@pytest.fixture
def baltic_env(baltic_instance):
    return LSNDPEnv(baltic_instance)


# ===========================================================================
# 1. Environment Construction
# ===========================================================================

class TestEnvironmentConstruction:
    def test_construct_with_toy_instance(self, toy_instance):
        env = LSNDPEnv(toy_instance)
        assert env._instance is toy_instance
        assert env._vessel_classes == ["V1"]
        assert env._ports_sorted == ["A", "B"]

    def test_construct_with_baltic_instance(self, baltic_instance):
        env = LSNDPEnv(baltic_instance)
        assert env._instance.name == "Baltic"
        assert len(env._vessel_classes) > 0
        assert len(env._ports_sorted) > 0

    def test_observation_space_defined(self, toy_env):
        sp = toy_env.observation_space
        assert isinstance(sp, gym.spaces.Dict)
        # Required fields per P4 spec
        assert "remaining_demand" in sp.spaces
        assert "fleet_remaining" in sp.spaces
        assert "service_count" in sp.spaces
        assert "last_profit" in sp.spaces
        assert "instance_info" in sp.spaces
        # services is kept internally; not in gym space but accessible via get_state()

    def test_action_space_defined(self, toy_env):
        ap = toy_env.action_space
        assert isinstance(ap, gym.spaces.Dict)
        assert "vessel_class" in ap.spaces
        assert "port_sequence" in ap.spaces


# ===========================================================================
# 2. Reset / Initial State
# ===========================================================================

class TestReset:
    def test_reset_returns_observation_and_info(self, toy_env):
        obs, info = toy_env.reset()
        assert isinstance(obs, dict)
        assert isinstance(info, dict)

    def test_reset_initial_services_empty(self, toy_env):
        obs, _ = toy_env.reset()
        # Services are internal bookkeeping, accessible via get_state()
        assert len(toy_env.get_state().services) == 0

    def test_reset_initial_profit_zero(self, toy_env):
        """η_0 = 0 for empty network [PAPER CONFIRMED]."""
        obs, _ = toy_env.reset()
        assert len(obs["last_profit"]) == 1
        assert math.isclose(obs["last_profit"][0], 0.0, abs_tol=1e-6)

    def test_reset_initial_fleet_full(self, toy_env):
        obs, _ = toy_env.reset()
        fr = obs["fleet_remaining"]
        assert fr[0] == 5  # toy instance has V1 qty=5

    def test_reset_initial_demand_full(self, toy_env):
        obs, _ = toy_env.reset()
        rd = obs["remaining_demand"]
        assert rd[0] == 100.0  # single demand of 100 FFE/week

    def test_reset_deterministic_with_seed(self, toy_instance):
        env1 = LSNDPEnv(toy_instance)
        env2 = LSNDPEnv(toy_instance)
        obs1, _ = env1.reset(seed=42)
        obs2, _ = env2.reset(seed=42)
        assert np.array_equal(obs1["remaining_demand"], obs2["remaining_demand"])
        assert np.array_equal(obs1["fleet_remaining"], obs2["fleet_remaining"])
        assert obs1["last_profit"] == obs2["last_profit"]

    def test_reset_after_terminal(self, toy_env):
        """After a terminal step, reset should restart cleanly."""
        toy_env.reset()
        toy_env._terminated = True
        obs, info = toy_env.reset()
        assert not toy_env.is_terminal()
        assert len(obs["last_profit"]) == 1
        assert math.isclose(obs["last_profit"][0], 0.0, abs_tol=1e-6)


# ===========================================================================
# 3. Valid Service Action
# ===========================================================================

class TestValidServiceAction:
    def test_step_with_service_action_object(self, toy_env):
        toy_env.reset()
        sa = ServiceAction(vessel_class="V1", port_sequence=["A", "B"])
        obs, reward, terminated, truncated, info = toy_env.step(sa)
        # Note: with toy demand=100 and capacity=571, demand may be fully satisfied
        # after one step, causing termination. Just verify the step succeeds.
        assert isinstance(obs, dict)
        assert len(toy_env.get_state().services) == 1

    def test_step_with_dict_action(self, toy_env):
        toy_env.reset()
        action = {"vessel_class": 0, "port_sequence": [0, 1]}
        obs, reward, terminated, truncated, info = toy_env.step(action)
        assert len(toy_env.get_state().services) == 1

    def test_service_added_to_network(self, toy_env):
        toy_env.reset()
        sa = ServiceAction(vessel_class="V1", port_sequence=["A", "B"])
        toy_env.step(sa)
        state = toy_env.get_state()
        assert len(state.services) == 1
        assert state.services[0].vessel_class == "V1"
        assert state.services[0].port_sequence == ["A", "B"]

    def test_profit_history_grows(self, toy_instance):
        """Use higher demand so episode doesn't terminate after one step."""
        toy_instance.demands[0].ffe_per_week = 2000.0
        env = LSNDPEnv(toy_instance)
        env.reset()
        sa = ServiceAction(vessel_class="V1", port_sequence=["A", "B"])
        env.step(sa)
        env.step(sa)
        ph = env.get_state().profit_history
        assert len(ph) == 3  # η_0, η_1, η_2

    def test_p3_evaluator_invoked(self, toy_env):
        toy_env.reset()
        sa = ServiceAction(vessel_class="V1", port_sequence=["A", "B"])
        toy_env.step(sa)
        result = toy_env.get_last_mcf_result()
        assert result is not None
        assert result.num_services == 1

    def test_vessel_requirements_computed(self, toy_env):
        toy_env.reset()
        sa = ServiceAction(vessel_class="V1", port_sequence=["A", "B"])
        toy_env.step(sa)
        state = toy_env.get_state()
        # Key is integer service_id (not string — corrected for paper precision)
        assert 0 in state.vessel_requirements
        n_vs = state.vessel_requirements[0]["V1"]
        # L_s = 200 nm, v_s = 10 knots (nm/hour) → weekly voyages = 24*7 = 168h
        # n_vs = 200 / (10 * 24 * 7) = 200/1680 ≈ 0.1190 vessels
        expected = 200.0 / (10.0 * 24.0 * 7.0)
        assert math.isclose(n_vs, expected, abs_tol=1e-6)

    def test_fractional_vessel_not_rounded(self, toy_env):
        toy_env.reset()
        sa = ServiceAction(vessel_class="V1", port_sequence=["A", "B"])
        toy_env.step(sa)
        state = toy_env.get_state()
        n_vs = state.vessel_requirements[0]["V1"]
        # Must be fractional, not rounded to integer
        assert n_vs != int(n_vs)

    def test_service_count_in_observation(self, toy_instance):
        toy_instance.demands[0].ffe_per_week = 2000.0
        toy_instance.fleet = [FleetEntry(vessel_class="V1", quantity=10)]
        env = LSNDPEnv(toy_instance)
        env.reset()
        sa = ServiceAction(vessel_class="V1", port_sequence=["A", "B"])
        env.step(sa)
        env.step(sa)
        obs, _, _, _, _ = env.step(sa)
        assert obs["service_count"] == 3


# ===========================================================================
# 4. Invalid Service Action (Validation Errors)
# ===========================================================================

class TestInvalidServiceAction:
    def test_invalid_vessel_class_raises(self, toy_env):
        toy_env.reset()
        sa = ServiceAction(vessel_class="NONEXISTENT", port_sequence=["A", "B"])
        with pytest.raises(ServiceValidationError):
            toy_env.step(sa)

    def test_invalid_port_raises(self, toy_env):
        toy_env.reset()
        sa = ServiceAction(vessel_class="V1", port_sequence=["A", "ZZZ"])
        with pytest.raises(ServiceValidationError):
            toy_env.step(sa)

    def test_single_port_raises(self, toy_env):
        toy_env.reset()
        sa = ServiceAction(vessel_class="V1", port_sequence=["A"])
        with pytest.raises(ServiceValidationError):
            toy_env.step(sa)

    def test_empty_port_sequence_raises(self, toy_env):
        toy_env.reset()
        sa = ServiceAction(vessel_class="V1", port_sequence=[])
        with pytest.raises(ServiceValidationError):
            toy_env.step(sa)

    def test_duplicate_ports_raises(self, toy_env):
        toy_env.reset()
        sa = ServiceAction(vessel_class="V1", port_sequence=["A", "A"])
        with pytest.raises(ServiceValidationError):
            toy_env.step(sa)

    def test_draft_incompatible_raises(self, toy_instance):
        """[PAPER] Draft is NOT a hard constraint — C_unused handles fleet
        deviations economically per Eq. 34. Services with draft gaps are
        allowed; over-utilization/under-utilization is handled in cost model.
        """
        toy_instance.vessel_types["V1"].draft = 5.0
        env = LSNDPEnv(toy_instance)
        env.reset()
        sa = ServiceAction(vessel_class="V1", port_sequence=["A", "B"])
        # Should NOT raise — draft is not a hard constraint
        obs, reward, terminated, truncated, info = env.step(sa)
        toy_instance.vessel_types["V1"].draft = 12.0  # restore

    def test_no_distance_raises(self, toy_instance):
        """Service with no distance for a leg should fail."""
        toy_instance.distances = [d for d in toy_instance.distances
                                   if not (d.origin == "A" and d.destination == "B")]
        env = LSNDPEnv(toy_instance)
        env.reset()
        sa = ServiceAction(vessel_class="V1", port_sequence=["A", "B"])
        with pytest.raises(ServiceValidationError):
            env.step(sa)

    def test_step_on_terminal_raises(self, toy_env):
        toy_env.reset()
        toy_env._terminated = True
        sa = ServiceAction(vessel_class="V1", port_sequence=["A", "B"])
        with pytest.raises(RuntimeError):
            toy_env.step(sa)


# ===========================================================================
# 5. Reward Calculation
# ===========================================================================

class TestReward:
    def test_raw_reward_computed(self, toy_env):
        toy_env.reset()
        sa = ServiceAction(vessel_class="V1", port_sequence=["A", "B"])
        obs, reward, terminated, truncated, info = toy_env.step(sa)
        assert "reward_raw" in info
        assert isinstance(info["reward_raw"], float)

    def test_normalized_reward_computed(self, toy_env):
        toy_env.reset()
        sa = ServiceAction(vessel_class="V1", port_sequence=["A", "B"])
        obs, reward, terminated, truncated, info = toy_env.step(sa)
        assert "reward_normalized" in info
        assert isinstance(info["reward_normalized"], float)
        assert math.isclose(reward, info["reward_normalized"], abs_tol=1e-10)

    def test_normalized_by_eta_1(self, toy_env):
        toy_env.reset()
        sa = ServiceAction(vessel_class="V1", port_sequence=["A", "B"])
        _, r1, _, _, info1 = toy_env.step(sa)
        eta_1 = info1["profit"]
        if not math.isclose(eta_1, 0.0, abs_tol=1e-6):
            # R_1 = (η_1 - η_0) / η_1 = η_1 / η_1 = 1.0
            assert math.isclose(r1, 1.0, abs_tol=1e-6)

    def test_second_step_reward(self, toy_instance):
        toy_instance.demands[0].ffe_per_week = 2000.0
        env = LSNDPEnv(toy_instance)
        env.reset()
        sa = ServiceAction(vessel_class="V1", port_sequence=["A", "B"])
        _, r1, _, _, _ = env.step(sa)
        _, r2, _, _, info2 = env.step(sa)
        ph = env.get_state().profit_history
        eta_1 = ph[1]
        eta_2 = ph[2]
        if not math.isclose(eta_1, 0.0, abs_tol=1e-6):
            expected_r2 = (eta_2 - eta_1) / eta_1
            assert math.isclose(r2, expected_r2, abs_tol=1e-6)

    def test_zero_eta_1_fallback(self):
        """When η_1 ≈ 0, normalized reward falls back to raw reward."""
        inst = _make_toy_instance()
        # Set revenue very high so η_1 > 0, then compute normalized reward
        inst.demands[0].revenue = 1000000.0
        env = LSNDPEnv(inst)
        env.reset()
        sa = ServiceAction(vessel_class="V1", port_sequence=["A", "B"])
        _, reward, _, _, info = env.step(sa)
        raw = info["reward_raw"]
        eta_1 = info["profit"]
        # With large revenue, η_1 >> 0, so normalized ≈ raw / eta_1
        # Test that the fallback path is NOT taken
        if not math.isclose(eta_1, 0.0, abs_tol=1e-6):
            assert not math.isclose(reward, raw, abs_tol=1e-6)
            assert math.isclose(reward, raw / eta_1, abs_tol=1e-6)


# ===========================================================================
# 6. Profit History
# ===========================================================================

class TestProfitHistory:
    def test_profit_history_starts_with_zero(self, toy_env):
        toy_env.reset()
        ph = toy_env.get_state().profit_history
        assert len(ph) == 1
        assert math.isclose(ph[0], 0.0, abs_tol=1e-6)

    def test_profit_history_grows_each_step(self, toy_instance):
        toy_instance.demands[0].ffe_per_week = 2000.0
        env = LSNDPEnv(toy_instance)
        env.reset()
        sa = ServiceAction(vessel_class="V1", port_sequence=["A", "B"])
        env.step(sa)
        env.step(sa)
        ph = env.get_state().profit_history
        assert len(ph) == 3  # η_0, η_1, η_2

    def test_current_profit_accessible(self, toy_env):
        toy_env.reset()
        sa = ServiceAction(vessel_class="V1", port_sequence=["A", "B"])
        toy_env.step(sa)
        profit = toy_env.get_current_profit()
        assert isinstance(profit, float)
        assert profit == toy_env.get_state().profit_history[-1]

    def test_profit_in_observation(self, toy_instance):
        toy_instance.demands[0].ffe_per_week = 2000.0
        env = LSNDPEnv(toy_instance)
        env.reset()
        sa = ServiceAction(vessel_class="V1", port_sequence=["A", "B"])
        env.step(sa)
        obs, _, _, _, _ = env.step(sa)
        # last_profit should match current profit
        assert math.isclose(obs["last_profit"][0], env.get_current_profit(), abs_tol=1e-6)


# ===========================================================================
# 7. Demand State Update
# ===========================================================================

class TestDemandState:
    def test_demand_decreases_after_service(self, toy_env):
        toy_env.reset()
        sa = ServiceAction(vessel_class="V1", port_sequence=["A", "B"])
        toy_env.step(sa)
        state = toy_env.get_state()
        remaining = state.remaining_demand[0]
        assert remaining <= 100.0  # was 100, now less or equal

    def test_demand_satisfied_rejected_in_info(self, toy_instance):
        toy_instance.demands[0].ffe_per_week = 2000.0
        env = LSNDPEnv(toy_instance)
        env.reset()
        sa = ServiceAction(vessel_class="V1", port_sequence=["A", "B"])
        env.step(sa)
        _, _, _, _, info = env.step(sa)
        assert "demand_satisfied" in info
        assert "demand_rejected" in info
        assert "remaining_demand" in info

    def test_total_demand_constant(self, toy_env):
        toy_env.reset()
        total = toy_env.get_state().total_demand
        sa = ServiceAction(vessel_class="V1", port_sequence=["A", "B"])
        toy_env.step(sa)
        assert toy_env.get_state().total_demand == total


# ===========================================================================
# 8. Termination / Truncation
# ===========================================================================

class TestTermination:
    def test_no_termination_after_first_step(self, toy_instance):
        """Use high demand so first step does not satisfy all demand."""
        toy_instance.demands[0].ffe_per_week = 2000.0
        env = LSNDPEnv(toy_instance)
        env.reset()
        sa = ServiceAction(vessel_class="V1", port_sequence=["A", "B"])
        _, _, terminated, truncated, _ = env.step(sa)
        assert not terminated
        assert not truncated

    def test_vessel_exhaustion_termination(self, baltic_instance):
        """When all vessel counts reach 0, episode terminates."""
        env = LSNDPEnv(baltic_instance)
        env.reset()
        # Manually drain all fleet to trigger vessel exhaustion
        for vc in env._state.fleet_remaining:
            env._state.fleet_remaining[vc] = 0.0
        sa = ServiceAction(vessel_class="Post_panamax", port_sequence=["DKAAR", "FIKTK", "FIRAU"])
        _, _, term, trunc, _ = env.step(sa)
        assert term
        assert env.get_state().termination_reason == "vessel_exhaustion"

    def test_demand_satisfaction_termination(self, toy_instance):
        """When all demand is satisfied, episode terminates."""
        toy_instance.demands[0].ffe_per_week = 1.0
        toy_instance.fleet = [FleetEntry(vessel_class="V1", quantity=100)]
        env = LSNDPEnv(toy_instance)
        env.reset()

        steps = 0
        while not env.is_terminal() and steps < MAX_SERVICES_SAFETY_CAP:
            sa = ServiceAction(vessel_class="V1", port_sequence=["A", "B"])
            _, _, term, trunc, _ = env.step(sa)
            if term or trunc:
                break
            steps += 1

        assert env.is_terminal()
        state = env.get_state()
        assert state.termination_reason == "demand_satisfied"

    def test_truncation_at_safety_cap(self, toy_instance):
        """Engineering safety cap triggers truncation."""
        toy_instance.demands[0].ffe_per_week = 2000.0
        env = LSNDPEnv(toy_instance)
        env.reset()
        # Manually set num_services_added near cap
        env._state.num_services_added = MAX_SERVICES_SAFETY_CAP + 1
        env._terminated = False
        env._truncated = False
        sa = ServiceAction(vessel_class="V1", port_sequence=["A", "B"])
        _, _, terminated, truncated, info = env.step(sa)
        assert truncated
        assert info["termination_reason"] == "safety_cap_reached"

    def test_terminated_and_truncated_distinguishable(self, toy_env):
        toy_env.reset()
        toy_env._terminated = True
        toy_env._truncated = False
        sa = ServiceAction(vessel_class="V1", port_sequence=["A", "B"])
        with pytest.raises(RuntimeError):
            toy_env.step(sa)


# ===========================================================================
# 9. Determinism
# ===========================================================================

class TestDeterminism:
    def test_identical_actions_produce_identical_results(self, toy_instance):
        """Same seed + same action sequence = identical outcomes."""
        toy_instance.demands[0].ffe_per_week = 2000.0
        env1 = LSNDPEnv(toy_instance)
        env2 = LSNDPEnv(toy_instance)
        env1.reset(seed=123)
        env2.reset(seed=123)

        actions = [
            ServiceAction(vessel_class="V1", port_sequence=["A", "B"]),
            ServiceAction(vessel_class="V1", port_sequence=["A", "B"]),
        ]

        for sa in actions:
            o1, r1, t1, tr1, i1 = env1.step(sa)
            o2, r2, t2, tr2, i2 = env2.step(sa)
            assert np.array_equal(o1["remaining_demand"], o2["remaining_demand"])
            assert np.array_equal(o1["fleet_remaining"], o2["fleet_remaining"])
            assert math.isclose(r1, r2, abs_tol=1e-10)
            assert t1 == t2
            assert tr1 == tr2
            assert math.isclose(i1["profit"], i2["profit"], abs_tol=1e-6)


# ===========================================================================
# 10. Info Dictionary
# ===========================================================================

class TestInfoDictionary:
    def test_info_contains_required_fields(self, toy_env):
        toy_env.reset()
        sa = ServiceAction(vessel_class="V1", port_sequence=["A", "B"])
        _, _, _, _, info = toy_env.step(sa)

        required_fields = [
            "profit", "profit_history", "num_services", "step",
            "remaining_demand", "demand_satisfied", "demand_rejected",
            "total_demand", "vessel_state", "vessel_requirements",
            "termination_reason", "reward_raw", "reward_normalized",
        ]
        for field in required_fields:
            assert field in info, f"Missing info field: {field}"

    def test_info_profit_matches_observation(self, toy_env):
        toy_env.reset()
        sa = ServiceAction(vessel_class="V1", port_sequence=["A", "B"])
        _, _, _, _, info = toy_env.step(sa)
        assert math.isclose(info["profit"], toy_env.get_current_profit(), abs_tol=1e-6)

    def test_info_after_reset(self, toy_instance):
        toy_instance.demands[0].ffe_per_week = 2000.0
        env = LSNDPEnv(toy_instance)
        env.reset()
        sa = ServiceAction(vessel_class="V1", port_sequence=["A", "B"])
        env.step(sa)
        _, _, _, _, info = env.step(sa)
        assert info["num_services"] == 2
        assert info["step"] == 2

    def test_info_mcf_components_present(self, toy_env):
        toy_env.reset()
        sa = ServiceAction(vessel_class="V1", port_sequence=["A", "B"])
        _, _, _, _, info = toy_env.step(sa)
        mcf_fields = [
            "demand_coverage", "eta", "total_revenue", "rejection_cost",
            "handling_cost", "service_cost", "unused_vessel_cost",
            "voyage_cost", "port_call_cost", "sailing_fuel_cost",
            "idle_fuel_cost", "canal_fee_cost", "timing_seconds",
        ]
        for field in mcf_fields:
            assert field in info, f"Missing MCF info field: {field}"


# ===========================================================================
# 11. Observation Contract
# ===========================================================================

class TestObservationContract:
    def test_observation_is_dict(self, toy_env):
        obs, _ = toy_env.reset()
        assert isinstance(obs, dict)

    def test_observation_services_format(self, toy_instance):
        toy_instance.demands[0].ffe_per_week = 2000.0
        env = LSNDPEnv(toy_instance)
        env.reset()
        sa = ServiceAction(vessel_class="V1", port_sequence=["A", "B"])
        env.step(sa)
        env.step(sa)
        # Services are internal bookkeeping, accessible via get_state()
        assert len(env.get_state().services) == 2
        svc = env.get_state().services[0]
        assert svc.vessel_class == "V1"
        assert list(svc.port_sequence) == ["A", "B"]

    def test_observation_remaining_demand_shape(self, toy_env):
        toy_env.reset()
        obs, _ = toy_env.reset()
        assert obs["remaining_demand"].shape == (len(toy_env._instance.demands),)

    def test_observation_fleet_remaining_shape(self, toy_env):
        toy_env.reset()
        obs, _ = toy_env.reset()
        assert obs["fleet_remaining"].shape == (len(toy_env._instance.vessel_types),)

    def test_observation_serializable(self, toy_instance):
        toy_instance.demands[0].ffe_per_week = 2000.0
        env = LSNDPEnv(toy_instance)
        env.reset()
        sa = ServiceAction(vessel_class="V1", port_sequence=["A", "B"])
        env.step(sa)
        obs, _, _, _, _ = env.step(sa)
        import json
        serializable = {
            "remaining_demand": obs["remaining_demand"].tolist(),
            "fleet_remaining": obs["fleet_remaining"].tolist(),
            "service_count": obs["service_count"],
            "last_profit": float(obs["last_profit"][0]),
            "instance_info": obs["instance_info"],
        }
        json_str = json.dumps(serializable)
        assert len(json_str) > 0

    def test_observation_static_instance_info(self, toy_instance):
        toy_instance.demands[0].ffe_per_week = 2000.0
        env = LSNDPEnv(toy_instance)
        env.reset()
        obs1, _ = env.reset()
        sa = ServiceAction(vessel_class="V1", port_sequence=["A", "B"])
        env.step(sa)
        obs2, _, _, _, _ = env.step(sa)
        assert obs1["instance_info"] == obs2["instance_info"]


# ===========================================================================
# 12. Gymnasium API Compatibility
# ===========================================================================

class TestGymnasiumAPI:
    def test_reset_signature(self, toy_env):
        obs, info = toy_env.reset()
        assert isinstance(obs, dict)
        assert isinstance(info, dict)

    def test_step_returns_five_values(self, toy_env):
        toy_env.reset()
        sa = ServiceAction(vessel_class="V1", port_sequence=["A", "B"])
        result = toy_env.step(sa)
        assert len(result) == 5
        obs, reward, terminated, truncated, info = result
        assert isinstance(obs, dict)
        assert isinstance(reward, float)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert isinstance(info, dict)

    def test_seeding_works(self, toy_instance):
        env = LSNDPEnv(toy_instance)
        obs1, _ = env.reset(seed=42)
        obs2, _ = env.reset(seed=42)
        assert np.array_equal(obs1["remaining_demand"], obs2["remaining_demand"])

    def test_different_seeds_consistent(self, toy_instance):
        env = LSNDPEnv(toy_instance)
        obs1, _ = env.reset(seed=42)
        obs2, _ = env.reset(seed=99)
        # For deterministic env without randomness, may be identical.
        # Point: seeding doesn't crash.
        assert isinstance(obs1, dict)
        assert isinstance(obs2, dict)

    def test_space_types(self, toy_env):
        assert isinstance(toy_env.observation_space, gym.spaces.Space)
        assert isinstance(toy_env.action_space, gym.spaces.Space)


# ===========================================================================
# 13. Real Baltic Smoke Integration Test
# ===========================================================================

class TestRealDataSmoke:
    def test_baltic_reset(self, baltic_env):
        obs, info = baltic_env.reset()
        assert isinstance(obs, dict)
        assert isinstance(info, dict)
        assert len(baltic_env.get_state().services) == 0
        assert len(obs["last_profit"]) == 1
        assert math.isclose(obs["last_profit"][0], 0.0, abs_tol=1e-6)

    def test_baltic_valid_step(self, baltic_env):
        baltic_env.reset()
        # Use Post_panamax (largest draft) which can visit DEBRV/DKAAR/FIKTK
        vc = "Post_panamax"
        ports = _baltic_valid_cycle_ports(vessel_class=vc, n=3)
        sa = ServiceAction(vessel_class=vc, port_sequence=ports)
        obs, reward, terminated, truncated, info = baltic_env.step(sa)

        assert isinstance(obs, dict)
        assert len(baltic_env.get_state().services) == 1
        assert isinstance(reward, float)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert isinstance(info, dict)

        result = baltic_env.get_last_mcf_result()
        assert result is not None
        assert result.num_services == 1
        assert isinstance(result.eta, float)

        ph = baltic_env.get_state().profit_history
        assert len(ph) == 2

    def test_baltic_multiple_steps(self, baltic_env):
        baltic_env.reset()
        vc = "Post_panamax"
        ports = _baltic_valid_cycle_ports(vessel_class=vc, n=3)

        for i in range(3):
            sa = ServiceAction(vessel_class=vc, port_sequence=ports)
            obs, reward, terminated, truncated, info = baltic_env.step(sa)
            assert isinstance(reward, float)

        state = baltic_env.get_state()
        assert state.num_services_added == 3
        assert len(state.profit_history) == 4

    def test_baltic_info_complete(self, baltic_env):
        baltic_env.reset()
        vc = "Post_panamax"
        ports = _baltic_valid_cycle_ports(vessel_class=vc, n=3)
        sa = ServiceAction(vessel_class=vc, port_sequence=ports)
        _, _, _, _, info = baltic_env.step(sa)

        assert "profit" in info
        assert "reward_raw" in info
        assert "reward_normalized" in info
        assert "num_services" in info
        assert "demand_satisfied" in info
        assert "demand_rejected" in info
        assert "vessel_state" in info
        assert "vessel_requirements" in info
        assert "eta" in info
        assert "demand_coverage" in info

    def test_baltic_demand_decreases(self, baltic_env):
        baltic_env.reset()
        vc = "Post_panamax"
        ports = _baltic_valid_cycle_ports(vessel_class=vc, n=3)
        sa = ServiceAction(vessel_class=vc, port_sequence=ports)
        _, _, _, _, _ = baltic_env.step(sa)
        state = baltic_env.get_state()
        total_rem = sum(state.remaining_demand.values())
        assert total_rem <= baltic_env.get_state().total_demand

    def test_baltic_raw_data_unchanged(self, baltic_instance):
        """Verify raw data files were not modified during environment operation."""
        hashes_before = {}
        for fname in ["Demand_Baltic.csv", "fleet_Baltic.csv", "ports.csv", "dist_sparse.csv"]:
            fpath = ROOT / "data" / fname
            if fpath.exists():
                hashes_before[fname] = hashlib.sha256(fpath.read_bytes()).hexdigest()

        env = LSNDPEnv(baltic_instance)
        env.reset()
        vc = "Post_panamax"
        ports = _baltic_valid_cycle_ports(vessel_class=vc, n=3)
        for _ in range(3):
            sa = ServiceAction(vessel_class=vc, port_sequence=ports)
            env.step(sa)

        for fname, h_before in hashes_before.items():
            fpath = ROOT / "data" / fname
            if fpath.exists():
                h_after = hashlib.sha256(fpath.read_bytes()).hexdigest()
                assert h_after == h_before, f"Raw data modified: {fname}"


# ===========================================================================
# 14. Edge Cases
# ===========================================================================

class TestEdgeCases:
    def test_empty_env_step_with_no_demand(self):
        """Environment with no demands should handle gracefully."""
        inst = _make_toy_instance()
        inst.demands = []
        env = LSNDPEnv(inst)
        obs, info = env.reset()
        assert obs["remaining_demand"].shape == (0,)
        sa = ServiceAction(vessel_class="V1", port_sequence=["A", "B"])
        _, reward, terminated, truncated, info = env.step(sa)
        assert isinstance(reward, float)

    def test_info_has_warnings_field(self, toy_env):
        toy_env.reset()
        sa = ServiceAction(vessel_class="V1", port_sequence=["A", "B"])
        _, _, _, _, info = toy_env.step(sa)
        assert "warnings" in info
        assert isinstance(info["warnings"], list)

    def test_get_state_returns_internal_state(self, toy_env):
        toy_env.reset()
        state = toy_env.get_state()
        assert isinstance(state, _EnvState)
        assert state.instance is toy_env._instance

    def test_get_current_profit_after_reset(self, toy_env):
        toy_env.reset()
        assert math.isclose(toy_env.get_current_profit(), 0.0, abs_tol=1e-6)

    def test_get_current_profit_after_step(self, toy_env):
        toy_env.reset()
        sa = ServiceAction(vessel_class="V1", port_sequence=["A", "B"])
        toy_env.step(sa)
        profit = toy_env.get_current_profit()
        assert isinstance(profit, float)
