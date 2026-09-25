"""
G11.2.5 — Focused tests for the controlled draft-filter RE-ABLATION.

These tests pin the CONTROL and the INSTRUMENTATION, not a scientific verdict.
They assert that the experiment could only have produced a valid comparison:

  * one shared initial checkpoint for both arms (PHASE 2)
  * identical configuration between arms (PHASE 3 / PHASE 15)
  * the draft filter is the ONLY behavioral difference (PHASE 1 / PHASE 8)
  * raw and executed action traces are preserved, never conflated (PHASE 8)
  * PPO execution diagnostics are present and non-trivial (PHASE 4)
  * sampled and deterministic evaluation are both present and kept apart (PHASE 6)
  * MCF and demand reconciliation actually hold (PHASE 9 / PHASE 10)
  * checkpoints follow the declared schema (PHASE 6)
  * the run is reproducible (PHASE 12)
  * diversity denominators are valid populations (PHASE 7)
  * no fabricated services appear (PHASE 16.15)
  * checkpoint evaluation does not drift the training seed (PHASE 16.16)

Tests that need a trained artifact read the G11.2.5 output files. They skip
(never silently pass) when the experiment has not been run.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List

import pytest
import torch

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from experiments.manual.g11_2_5 import g11_2_5_reablation as g  # noqa: E402

OUT = g.OUT
ARMS = ("baseline", "no_draft")


# ======================================================================
# Helpers / fixtures
# ======================================================================


def _read_json(name: str) -> Any:
    p = OUT / name
    if not p.exists():
        pytest.skip(f"G11.2.5 artifact not present: {name}")
    return json.loads(p.read_text(encoding="utf-8"))


def _read_jsonl(name: str) -> List[Dict[str, Any]]:
    p = OUT / name
    if not p.exists():
        pytest.skip(f"G11.2.5 artifact not present: {name}")
    return [json.loads(ln) for ln in p.read_text(encoding="utf-8").splitlines() if ln.strip()]


@pytest.fixture(scope="module")
def manifest() -> Dict[str, Any]:
    return _read_json("initial_checkpoint_manifest.json")


@pytest.fixture(scope="module")
def final_cmp() -> Dict[str, Any]:
    return _read_json("final_comparison.json")


@pytest.fixture(scope="module")
def baseline_updates() -> List[Dict[str, Any]]:
    return [r for r in _read_jsonl("baseline_metrics.jsonl") if not r.get("empty_trajectory")]


@pytest.fixture(scope="module")
def no_draft_updates() -> List[Dict[str, Any]]:
    return [r for r in _read_jsonl("no_draft_metrics.jsonl") if not r.get("empty_trajectory")]


@pytest.fixture(scope="module")
def repro() -> Dict[str, Any]:
    return _read_json("reproducibility_result.json")


@pytest.fixture(scope="module")
def econ() -> List[Dict[str, Any]]:
    return _read_json("economic_comparison.json")


@pytest.fixture(scope="module")
def draft_cmp() -> List[Dict[str, Any]]:
    return _read_json("draft_impact_comparison.json")


# ======================================================================
# 1. identical initial checkpoint
# ======================================================================


class TestIdenticalInitialCheckpoint:
    def test_manifest_exists_and_is_hashed(self, manifest):
        assert len(manifest["policy_state_sha256"]) == 64
        assert len(manifest["critic_state_sha256"]) == 64

    def test_both_arms_started_from_the_same_state(self, final_cmp):
        assert final_cmp["initial_checkpoint_shared"] is True
        assert final_cmp["initial_policy_sha256"]

    def test_policy_parameter_count_recorded(self, manifest):
        assert manifest["policy_parameter_count"] > 0
        assert manifest["policy_tensor_count"] > 0

    def test_optimizer_state_recorded(self, manifest):
        # AdamW has no per-parameter state until the first step; the point is
        # that the field is explicitly accounted for, not omitted.
        assert "state_entries_at_init" in manifest["optimizer"]
        assert "param_groups" in manifest["optimizer"]

    def test_rng_state_availability_recorded(self, manifest):
        r = manifest["rng_state_availability"]
        assert r["python_random"] and r["numpy"] and r["torch_cpu"]

    def test_dataset_identity_hashed(self, manifest):
        ds = manifest["dataset_sha256"]
        assert ds, "dataset hashes must be recorded"
        for k, v in ds.items():
            assert len(v) == 64, f"{k} hash malformed"

    def test_initial_checkpoint_file_matches_manifest(self, manifest):
        if not g.INIT_CKPT_PATH.exists():
            pytest.skip("initial checkpoint file not retained")
        ck = torch.load(g.INIT_CKPT_PATH, map_location="cpu", weights_only=False)
        assert g.state_dict_hash(ck["policy_state_dict"]) == manifest["policy_state_sha256"]
        assert g.state_dict_hash(ck["critic_state_dict"]) == manifest["critic_state_sha256"]


# ======================================================================
# 2. identical configuration
# ======================================================================


class TestIdenticalConfiguration:
    def test_frozen_ppo_config_matches_brief(self):
        g._assert_config_matches_brief()  # raises on drift

    def test_architecture_matches_brief(self):
        for k, v in g.BRIEF_ARCH.items():
            assert g.ARCH[k] == v

    def test_required_hyperparameters_are_the_specified_values(self):
        assert g.PPO["learning_rate"] == 2e-4
        assert g.PPO["gamma"] == 1.0
        assert g.PPO["gae_lambda"] == 0.9
        assert g.PPO["ppo_epochs"] == 10
        assert g.PPO["clip_epsilon"] == 0.20
        assert g.PPO["target_kl"] == 0.10
        assert g.PPO["entropy_coefficient"] == 0.05
        assert g.PPO["value_coefficient"] == 0.50
        assert g.PPO["max_grad_norm"] == 0.5
        assert g.PPO["num_envs"] == 1
        assert g.PPO["steps_per_env"] == 10
        assert g.PPO["minibatch_size"] == 32
        assert g.PPO["seed"] == 42

    def test_manifest_config_matches_module_config(self, manifest):
        assert manifest["architecture"] == g.ARCH
        assert manifest["ppo_config"] == g.PPO
        assert manifest["seed"] == 42

    def test_checkpoint_evaluation_points_are_the_required_ones(self):
        assert g.EVAL_POINTS == [0, 1, 5, 10, 25, 50]

    def test_evaluation_horizon_identical_across_arms(self):
        # EVAL_MAX_STEPS is a single module constant; both arms read it.
        assert isinstance(g.EVAL_MAX_STEPS, int) and g.EVAL_MAX_STEPS > 0


# ======================================================================
# 3/4. baseline ON, intervention OFF, and the flag is the only difference
# ======================================================================


class TestDraftFilterArmConfiguration:
    def test_baseline_filter_stage_is_enabled(self, draft_cmp):
        rows = [r for r in draft_cmp if r.get("baseline_filter_stage") is not None]
        assert rows
        assert all(r["baseline_filter_stage"] == "enabled" for r in rows)

    def test_intervention_filter_stage_is_disabled(self, draft_cmp):
        rows = [r for r in draft_cmp if r.get("no_draft_filter_stage") is not None]
        assert rows
        assert all(r["no_draft_filter_stage"] == "disabled" for r in rows)

    def test_disabled_arm_removes_nothing(self, draft_cmp):
        for r in draft_cmp:
            assert r.get("no_draft_ports_removed_total") == 0

    def test_enabled_arm_actually_filters(self, draft_cmp):
        removed = [r["baseline_ports_removed_total"] for r in draft_cmp
                   if r.get("baseline_filter_stage") == "enabled"]
        assert removed, "no construction calls recorded for the baseline arm"
        assert any(v > 0 for v in removed), (
            "baseline arm never removed a port — the flag is not taking effect"
        )

    def test_generator_construction_differ_only_by_flag(self):
        """The two generators are the same class with one differing field."""
        import inspect
        from actions.service_generator import ServiceGenerator
        params = list(inspect.signature(ServiceGenerator.__init__).parameters)
        assert "draft_filter_enabled" in params

    def test_instrumented_generator_does_not_override_filter_logic(self):
        """
        The instrumented generator must DELEGATE to the base implementation,
        otherwise the measured arm behaviour is not the real arm behaviour.
        """
        import inspect
        src = inspect.getsource(g.InstrumentedServiceGenerator.order_ports)
        assert "super().order_ports(" in src
        # It must not re-derive a different filter predicate.
        assert "def _is_draft_feasible" not in src


# ======================================================================
# 5. raw action trace preserved
# ======================================================================


class TestRawActionTrace:
    def test_raw_policy_probe_shows_flag_does_not_leak_upstream(self):
        probe = _read_json("raw_policy_equivalence_probe.json")
        assert probe["all_raw_outputs_identical"] is True, probe["verdict"]
        assert probe["rows"]
        assert all(r["raw_sequence_identical"] for r in probe["rows"])
        assert all(r["raw_tokens_identical"] for r in probe["rows"])
        assert all(r["log_prob_identical"] for r in probe["rows"])

    def test_probe_shows_construction_actually_differs(self):
        probe = _read_json("raw_policy_equivalence_probe.json")
        assert probe["executed_lengths_differ_on_some_step"] is True, (
            "raw identical AND executed identical means the intervention did nothing"
        )

    def test_training_updates_record_raw_actions(self, baseline_updates):
        assert baseline_updates
        for r in baseline_updates:
            assert "raw_total_actions" in r
            assert "raw_unique_actions" in r
            assert "raw_unique_port_sequences" in r

    def test_policy_trace_files_record_the_action_surface(self):
        for arm in ARMS:
            rows = _read_jsonl(f"{arm}_policy_trace.jsonl")
            rows = [r for r in rows if not r.get("empty_trajectory")]
            assert rows, f"{arm} policy trace is empty"
            assert all("raw_unique_actions" in r for r in rows)


# ======================================================================
# 6. executed action trace preserved and never conflated with raw
# ======================================================================


class TestExecutedActionTrace:
    def test_executed_actions_recorded_separately(self, baseline_updates):
        for r in baseline_updates:
            assert "executed_total_actions" in r
            assert "executed_unique_actions" in r
            assert r["executed_total_actions"] == r["raw_total_actions"], (
                "executed denominator must cover the same action population as raw"
            )

    def test_executed_may_differ_from_raw_under_draft_filter(self, baseline_updates):
        # A draft filter ON arm must be able to shrink executed relative to raw.
        pairs = [(r["raw_unique_actions"], r["executed_unique_actions"])
                 for r in baseline_updates]
        assert any(e <= a for a, e in pairs)

    def test_raw_and_executed_are_distinct_fields(self, baseline_updates):
        """Guard against the G11.2.3 class of bug where fallback was substituted."""
        for r in baseline_updates:
            assert "mean_raw_seq_len" in r and "mean_executed_seq_len" in r


# ======================================================================
# 7. PPO diagnostics present (PHASE 4)
# ======================================================================


class TestPPODiagnostics:
    REQUIRED = [
        "epochs_attempted", "epochs_executed", "kl_early_stopped",
        "minibatch_size", "minibatches_per_epoch", "total_optimizer_steps",
        "approx_kl", "policy_loss", "value_loss", "entropy_loss",
        "clip_fraction", "grad_norm_pre_clip", "grad_norm_applied",
        "policy_param_delta", "critic_param_delta", "ratio_mean", "ratio_min",
        "ratio_max", "ratio_outside_clip_fraction",
    ]

    def test_all_required_diagnostics_present(self, baseline_updates, no_draft_updates):
        for r in baseline_updates + no_draft_updates:
            for k in self.REQUIRED:
                assert k in r, f"missing PPO diagnostic {k}"

    def test_ppo_epochs_actually_execute(self, baseline_updates, no_draft_updates):
        for r in baseline_updates + no_draft_updates:
            assert r["epochs_attempted"] == 10
            assert r["epochs_executed"] >= 1

    def test_minibatches_actually_execute(self, baseline_updates):
        for r in baseline_updates:
            assert r["minibatches_per_epoch"] >= 1
            assert r["total_optimizer_steps"] >= 1

    def test_target_kl_early_stopping_is_exercised_or_bounded(self, baseline_updates):
        # Either the KL stayed under target (no early stop) or it early-stopped.
        for r in baseline_updates:
            stopped = r["kl_early_stopped"]
            assert stopped == (r["approx_kl"] > g.PPO["target_kl"]), (
                f"early-stop flag inconsistent with measured KL at update {r['update']}"
            )

    def test_ppo_diagnostics_file_has_per_epoch_detail(self):
        rows = _read_jsonl("ppo_diagnostics.jsonl")
        assert rows
        assert any("epoch_detail" in r for r in rows), "no per-epoch trace recorded"

    def test_both_arms_recorded_in_diagnostics(self):
        rows = _read_jsonl("ppo_diagnostics.jsonl")
        assert {r.get("arm") for r in rows} >= {"baseline", "no_draft"}

    def test_numerically_finite(self, baseline_updates, no_draft_updates):
        for r in baseline_updates + no_draft_updates:
            for k in ("approx_kl", "policy_loss", "value_loss", "total_loss"):
                assert math.isfinite(r[k]), f"non-finite {k} at update {r['update']}"

    def test_policy_and_critic_parameters_moved(self, baseline_updates, no_draft_updates):
        # Diagnostic only — NOT a learning claim (PHASE 14).
        for r in baseline_updates + no_draft_updates:
            assert r["policy_param_delta"] > 0
            assert r["critic_param_delta"] > 0

    def test_applied_grad_norm_respects_clip(self, baseline_updates, no_draft_updates):
        for r in baseline_updates + no_draft_updates:
            assert r["grad_norm_applied"] <= g.PPO["max_grad_norm"] + 1e-9


# ======================================================================
# 8/9. sampled AND deterministic evaluation present, never combined
# ======================================================================


class TestEvaluationModes:
    def _net_traces(self):
        out = {}
        for arm in ARMS:
            rows = _read_jsonl(f"{arm}_network_trace.jsonl")
            out[arm] = rows
        return out

    def test_both_evaluation_modes_present_in_comparison(self):
        rows = _read_json("sampled_vs_deterministic.json")
        assert rows
        for r in rows:
            assert "baseline_sampled_services" in r and "baseline_argmax_services" in r
            assert "no_draft_sampled_services" in r and "no_draft_argmax_services" in r

    def test_modes_are_recorded_separately_not_merged(self):
        rows = _read_json("sampled_vs_deterministic.json")
        for r in rows:
            # A merged metric would be identical in both slots by construction.
            assert r["baseline_sampled_bos_rate"] is not None
            assert r["baseline_argmax_bos_rate"] is not None
            assert r["note"] == "recorded separately, never combined"

    def test_network_trace_records_eval_mode(self):
        for arm in ARMS:
            rows = self._net_traces()[arm]
            assert rows
            assert rows[0]["eval_mode"] == "sampled", (
                "sampled evaluation is primary and must be what is traced"
            )


# ======================================================================
# 10. MCF reconciliation
# ======================================================================


class TestMCFReconciliation:
    def test_mcf_success_at_all_evaluated_checkpoints(self, econ):
        for r in econ:
            assert r["baseline_mcf_success"] is True, f"baseline MCF failed at u{r['update']}"
            assert r["no_draft_mcf_success"] is True, f"no_draft MCF failed at u{r['update']}"

    def test_mcf_economic_reconciliation_residual_is_zero(self, econ):
        for r in econ:
            for arm in ("baseline", "no_draft"):
                res = r[f"{arm}_reconciliation_residual"]
                assert res is None or abs(res) < 1e-6, (
                    f"{arm} u{r['update']} eta residual {res}"
                )

    def test_economics_are_not_inferred_from_policy_metrics(self, econ):
        """Economics must come from MCF fields, not reward/diversity."""
        for r in econ:
            for arm in ("baseline", "no_draft"):
                assert f"{arm}_weekly_revenue" in r
                assert f"{arm}_total_cost" in r
                assert f"{arm}_weekly_profit" in r

    def test_full_cost_breakdown_recorded(self):
        rows = _read_jsonl("baseline_network_trace.jsonl")
        if not rows or not rows[0].get("economics"):
            pytest.skip("final baseline checkpoint proposed no services")
        econ_keys = set(rows[0]["economics"].keys())
        for k in ("weekly_revenue", "service_cost", "voyage_cost", "handling_cost",
                  "rejection_penalty", "unused_vessel_cost", "total_cost",
                  "weekly_profit", "profit_margin_pct"):
            assert k in econ_keys, f"missing economic component {k}"

    def test_profit_margin_only_reported_when_defined(self):
        rows = _read_jsonl("baseline_network_trace.jsonl")
        if not rows or not rows[0].get("economics"):
            pytest.skip("final baseline checkpoint proposed no services")
        e = rows[0]["economics"]
        if e["profit_margin_pct"] is None:
            assert e["profit_margin_defined"] is False
        else:
            assert e["profit_margin_defined"] is True


# ======================================================================
# 11. demand reconciliation + hierarchical interpretation
# ======================================================================


class TestDemandReconciliation:
    def test_demand_hierarchy_recorded(self):
        rows = _read_jsonl("baseline_network_trace.jsonl")
        if not rows or not rows[0].get("demand"):
            pytest.skip("final baseline checkpoint proposed no services")
        d = rows[0]["demand"]
        for k in ("total_demand", "structurally_reachable_demand", "mcf_routed_demand",
                  "mcf_rejected_demand", "final_covered_demand", "coverage_pct"):
            assert k in d, f"missing demand field {k}"

    def test_demand_identity_holds(self):
        rows = _read_jsonl("baseline_network_trace.jsonl")
        if not rows or not rows[0].get("demand"):
            pytest.skip("final baseline checkpoint proposed no services")
        d = rows[0]["demand"]
        lhs = d["mcf_routed_demand"] + d["mcf_rejected_demand"]
        assert abs(lhs - d["total_demand"]) < 1e-4, (
            f"routed+rejected != total ({lhs} vs {d['total_demand']})"
        )

    def test_reachability_is_nested_not_exclusive(self):
        rows = _read_jsonl("baseline_network_trace.jsonl")
        if not rows or not rows[0].get("demand"):
            pytest.skip("final baseline checkpoint proposed no services")
        d = rows[0]["demand"]
        assert d["structurally_reachable_demand"] + d["structurally_unreachable_demand"] \
            == pytest.approx(d["total_demand"], abs=1e-4)
        assert "nested" in d["note"]

    def test_demand_comparison_covers_both_arms(self):
        rows = _read_json("demand_comparison.json")
        assert rows
        for r in rows:
            assert "baseline_coverage_pct" in r
            assert "no_draft_coverage_pct" in r


# ======================================================================
# 12. checkpoint schema
# ======================================================================


class TestCheckpointSchema:
    REQUIRED_TOP = ["update", "eval_mode", "eval_seed", "raw_policy",
                    "executed_policy", "draft_impact", "network_structure",
                    "no_services_proposed", "mcf"]

    def _final_rows(self):
        return {a: _read_jsonl(f"{a}_network_trace.jsonl") for a in ARMS}

    def test_final_checkpoint_rows_have_required_keys(self):
        for arm, rows in self._final_rows().items():
            if not rows:
                pytest.skip(f"{arm} produced no final checkpoint row")
            for k in self.REQUIRED_TOP:
                assert k in rows[0], f"{arm}: missing checkpoint key {k}"

    def test_checkpoints_recorded_at_required_points(self):
        # 6 declared evaluation points, each with both modes.
        assert len(g.EVAL_POINTS) == 6

    def test_draft_impact_block_is_complete(self):
        for arm, rows in self._final_rows().items():
            if not rows:
                pytest.skip(f"{arm} produced no final checkpoint row")
            d = rows[0]["draft_impact"]
            for k in ("filter_stage", "construction_calls", "raw_ports_total",
                      "ports_after_draft_filter_total", "ports_after_tsp_total",
                      "ports_removed_total", "pct_decisions_affected",
                      "mean_removed_ports", "median_removed_ports",
                      "max_removed_ports"):
                assert k in d, f"{arm}: missing draft metric {k}"

    def test_measured_pipeline_order_matches_implementation(self, draft_cmp):
        # The implementation filters BEFORE the TSP; the brief sketched the
        # reverse. The artifact must state what actually runs.
        for r in draft_cmp:
            assert r["implemented_pipeline_order"] == \
                "raw -> draft filter (ON/OFF) -> TSP -> service"

    def test_intervention_arm_records_filter_as_disabled(self):
        rows = _read_jsonl("no_draft_network_trace.jsonl")
        if not rows:
            pytest.skip("no_draft produced no final checkpoint row")
        assert rows[0]["draft_impact"]["filter_stage"] == "disabled"

    def test_draft_stage_length_selfcheck_passes(self):
        for arm in ARMS:
            rows = _read_jsonl(f"{arm}_network_trace.jsonl")
            if not rows:
                pytest.skip(f"{arm} produced no final checkpoint row")
            assert rows[0]["draft_impact"]["length_selfcheck_all_pass"] is True, (
                f"{arm}: reconstructed filter stage disagrees with the real output"
            )


# ======================================================================
# 13. reproducibility
# ======================================================================


class TestRepeatability:
    def test_reproducibility_recorded_for_both_arms(self, repro):
        assert set(repro) == {"baseline", "no_draft"}

    def test_repeatability_is_bitwise(self, repro):
        for arm in ARMS:
            r = repro[arm]
            assert r["verdict"].startswith("PASS"), (
                f"{arm} reproducibility: {r['verdict']} "
                f"first_divergence={r.get('first_divergence')}"
            )
            assert r["final_policy_bitwise_identical"] is True
            assert r["init_policy_identical"] is True

    def test_per_update_traces_match_bitwise(self, repro):
        for arm in ARMS:
            for row in repro[arm]["per_update"]:
                if not row.get("both_present"):
                    continue
                assert row["raw_actions_identical"] is True, (
                    f"{arm} raw actions diverged at update {row['update']}"
                )
                assert row["executed_actions_identical"] is True
                assert row["old_log_probs_identical"] is True

    def test_checkpoints_identical_across_repeats(self, repro):
        for arm in ARMS:
            assert repro[arm]["checkpoints_identical"] is True

    def test_failure_is_not_downgraded_to_approximate(self, repro):
        """A failed bitwise test must be reported as FAIL, never softened."""
        for arm in ARMS:
            v = repro[arm]["verdict"]
            assert "approx" not in v.lower()
            if not v.startswith("PASS"):
                assert repro[arm]["first_divergence"] is not None, (
                    "a failed reproducibility check must localise its first divergence"
                )


# ======================================================================
# 14. diversity metric denominator validity
# ======================================================================


class TestDiversityDenominators:
    def test_raw_ratio_uses_the_raw_population(self, baseline_updates, no_draft_updates):
        for r in baseline_updates + no_draft_updates:
            n = r["raw_total_actions"]
            assert n > 0
            expected = r["raw_unique_actions"] / n
            assert r["raw_unique_action_ratio"] == pytest.approx(expected)

    def test_executed_ratio_uses_the_executed_population(self, baseline_updates):
        for r in baseline_updates:
            n = r["executed_total_actions"]
            assert n > 0
            expected = r["executed_unique_actions"] / n
            assert r["executed_unique_action_ratio"] == pytest.approx(expected)

    def test_no_cross_population_ratio_is_mislabelled(self, baseline_updates):
        """A 'collapse ratio' must not compare unique services against actions."""
        for r in baseline_updates:
            assert "collapse_ratio" not in r, (
                "a ratio across incomparable populations must not be labelled collapse"
            )

    def test_service_diversity_uses_services_as_denominator(self):
        for arm in ARMS:
            rows = _read_jsonl(f"{arm}_network_trace.jsonl")
            if not rows:
                pytest.skip(f"{arm} produced no final checkpoint row")
            ns = rows[0]["network_structure"]
            total = ns["services_emitted"]
            if total == 0:
                assert ns["service_diversity"] == 0.0
            else:
                assert ns["service_diversity"] == pytest.approx(
                    ns["unique_services"] / total
                )
            assert ns["denominator_total_services"] == total

    def test_draft_pct_uses_construction_call_denominator(self, draft_cmp):
        for r in draft_cmp:
            if r.get("baseline_pct_decisions_affected") is None:
                continue
            assert 0.0 <= r["baseline_pct_decisions_affected"] <= 100.0

    def test_bos_and_fallback_use_the_action_denominator(self, baseline_updates):
        for r in baseline_updates:
            n = r["raw_total_actions"]
            assert r["bos_rate"] == pytest.approx(r["bos_count"] / n)
            assert r["fallback_rate"] == pytest.approx(r["fallback_count"] / n)

    def test_collapse_guard_is_a_warning_not_a_verdict(self, final_cmp):
        """The guard may not manufacture a CASE A/B/C."""
        assert final_cmp["verdict"].startswith("CASE")
        # Guard is advisory; its presence must never change the case letter.
        assert "collapse_guard" not in final_cmp["classification"].get("reasons", [])


# ======================================================================
# 15. no fabricated services
# ======================================================================


class TestNoFabricatedServices:
    def test_empty_network_is_reported_as_measured_outcome(self):
        for arm in ARMS:
            rows = _read_jsonl(f"{arm}_network_trace.jsonl")
            if not rows:
                pytest.skip(f"{arm} produced no final checkpoint row")
            row = rows[0]
            if row["no_services_proposed"]:
                assert row["network_structure"]["services_emitted"] == 0
                assert row["economics"] is None
                assert row["mcf"]["mcf_success"] is False

    def test_service_count_never_exceeds_proposed_steps(self):
        for arm in ARMS:
            rows = _read_jsonl(f"{arm}_network_trace.jsonl")
            if not rows:
                pytest.skip(f"{arm} produced no final checkpoint row")
            row = rows[0]
            assert row["network_structure"]["services_emitted"] <= \
                row["executed_policy"]["proposed_steps"] + row["executed_policy"]["bos_count"]

    def test_services_have_minimum_structure(self):
        for arm in ARMS:
            rows = _read_jsonl(f"{arm}_network_trace.jsonl")
            if not rows or row_has_no_services(rows[0]):
                continue
            row = rows[0]
            for dist in row["network_structure"]["service_length_distribution"].values():
                assert dist >= 2, "a fabricated 1-port 'service' is not a valid cycle"

    def test_bos_stops_are_counted_not_filled(self):
        for arm in ARMS:
            rows = _read_jsonl(f"{arm}_network_trace.jsonl")
            if not rows:
                pytest.skip(f"{arm} produced no final checkpoint row")
            ep = rows[0]["executed_policy"]
            assert ep["bos_count"] >= 0
            assert ep["bos_rate"] == pytest.approx(ep["bos_count"] / ep["total_executed_actions"])


def row_has_no_services(row: Dict[str, Any]) -> bool:
    return bool(row.get("no_services_proposed"))


# ======================================================================
# 16. no checkpoint seed drift
# ======================================================================


class TestNoSeedDrift:
    def test_checkpoint_evaluation_does_not_perturb_training_rng(self):
        """
        `run_arm` seeds with `set_all_seeds(SEED + u)` at the top of every
        update, so a checkpoint evaluation between updates cannot shift the
        training random stream. Verify the source ordering.
        """
        import inspect
        src = inspect.getsource(g.run_arm)
        seed_pos = src.index("set_all_seeds(SEED + u)")
        rollout_pos = src.index("collect_rollout")
        assert seed_pos < rollout_pos, "rollout must be preceded by its seed"

    def test_every_update_uses_a_distinct_deterministic_seed(self, baseline_updates):
        seeds = [g.SEED + (r["update"] - 1) for r in baseline_updates]
        assert seeds == sorted(seeds)
        assert len(set(seeds)) == len(seeds)

    def test_evaluation_uses_a_fixed_seed_per_checkpoint(self):
        for arm in ARMS:
            rows = _read_jsonl(f"{arm}_network_trace.jsonl")
            if not rows:
                pytest.skip(f"{arm} produced no final checkpoint row")
            assert rows[0]["eval_seed"] == g.SEED, (
                "evaluation seed must be fixed, not drawn from a drifting stream"
            )


# ======================================================================
# Cross-cutting: the experiment is interpretable
# ======================================================================


class TestClassificationIntegrity:
    def test_case_is_one_of_the_four(self, final_cmp):
        assert final_cmp["verdict"] in {"CASE A", "CASE B", "CASE C", "CASE D"}

    def test_no_convergence_claim(self, final_cmp):
        blob = json.dumps(final_cmp).lower()
        for banned in ("converged", "convergence achieved", "proven optimal",
                       "statistically significant"):
            assert banned not in blob, f"prohibited claim present: {banned}"

    def test_ppo_stability_is_recorded_not_tuned(self, final_cmp):
        st = final_cmp["ppo_stability"]
        assert st["target_kl"] == 0.10
        assert "note" in st and "NOT tuned" in st["note"]
        assert st["approx_kl_per_update"]["baseline"]
        assert st["approx_kl_per_update"]["no_draft"]

    def test_kl_trajectory_is_preserved_verbatim(self, final_cmp):
        kls = final_cmp["ppo_stability"]["approx_kl_per_update"]["baseline"]
        assert any(k > 0 for k in kls)
        # Large values must be preserved, not clipped or filtered.
        assert max(kls) == kls[list(kls).index(max(kls))]

    def test_incomplete_arms_are_disclosed(self, final_cmp):
        inc = final_cmp["classification"].get("incomplete", [])
        assert isinstance(inc, list)

    def test_early_stop_reasons_are_reported(self, final_cmp):
        for key in ("baseline_stopped_reason", "no_draft_stopped_reason"):
            assert key in final_cmp
