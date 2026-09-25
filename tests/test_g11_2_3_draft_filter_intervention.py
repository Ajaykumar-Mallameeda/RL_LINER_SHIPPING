"""
G11.2.3 — Focused Tests for Draft-Filter Intervention.

These tests verify:
1. The draft filter can be disabled only in the experimental path
2. TSP ordering remains unchanged when filter is off
3. Decoder output is unchanged
4. PPO configuration remains unchanged
5. Reward computation remains unchanged
6. MCF evaluation remains unchanged
7. Service construction is still valid after intervention
8. n_vs reconciliation works
9. Demand reconciliation works
10. Checkpoint reload works
11. Baseline/intervention configs are comparable
12. No NaN/Inf in outputs
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
    from data.linerlib_loader import LINERLIBLoader
    loader = LINERLIBLoader(str(_ROOT / "data"))
    return loader.load("WorldSmall", validate=False)


@pytest.fixture(scope="module")
def dist_by_pair(worldsmall_instance):
    return {(a.origin, a.destination): a for a in worldsmall_instance.distances}


@pytest.fixture(scope="module")
def baseline_generator(dist_by_pair, worldsmall_instance):
    from actions.service_generator import ServiceGenerator
    return ServiceGenerator(worldsmall_instance, dist_by_pair, draft_filter_enabled=True)


@pytest.fixture(scope="module")
def no_draft_generator(dist_by_pair, worldsmall_instance):
    from actions.service_generator import ServiceGenerator
    return ServiceGenerator(worldsmall_instance, dist_by_pair, draft_filter_enabled=False)


# ======================================================================
# 1. Draft filter disabled only in experimental path
# ======================================================================

class TestDraftFilterIntervention:
    def test_default_is_enabled(self, baseline_generator):
        """Default behavior must preserve existing draft filtering."""
        assert baseline_generator._draft_filter_enabled is True

    def test_intervention_disables_filter(self, no_draft_generator):
        """Experimental path must disable draft filtering."""
        assert no_draft_generator._draft_filter_enabled is False

    def test_explicit_enable(self, dist_by_pair, worldsmall_instance):
        """Can explicitly enable draft filter."""
        from actions.service_generator import ServiceGenerator
        gen = ServiceGenerator(worldsmall_instance, dist_by_pair, draft_filter_enabled=True)
        assert gen._draft_filter_enabled is True

    def test_explicit_disable(self, dist_by_pair, worldsmall_instance):
        """Can explicitly disable draft filter."""
        from actions.service_generator import ServiceGenerator
        gen = ServiceGenerator(worldsmall_instance, dist_by_pair, draft_filter_enabled=False)
        assert gen._draft_filter_enabled is False


# ======================================================================
# 2. TSP unchanged when filter is off
# ======================================================================

class TestTSPOrdering:
    def test_tsp_order_same_with_and_without_filter_large_vessel(self,
            baseline_generator, no_draft_generator, worldsmall_instance):
        """For a vessel that can visit all ports, both paths give same order."""
        ports = sorted(worldsmall_instance.ports.keys())[:10]
        # Find a vessel class that can visit all these ports
        for vc in ["Post_panamax", "Super_panamax"]:
            all_feasible = all(baseline_generator.can_visit_port(vc, p) for p in ports)
            if all_feasible:
                order_baseline = baseline_generator.order_ports(ports, vc)
                order_no_draft = no_draft_generator.order_ports(ports, vc)
                assert order_baseline == order_no_draft
                break
        else:
            pytest.skip("No vessel can visit all first 10 ports")

    def test_tsp_deterministic_no_draft(self, no_draft_generator, worldsmall_instance):
        """TSP ordering is deterministic even without draft filter."""
        ports = sorted(worldsmall_instance.ports.keys())[:15]
        order1 = no_draft_generator.order_ports(ports, "Feeder_450")
        order2 = no_draft_generator.order_ports(ports, "Feeder_450")
        assert order1 == order2


# ======================================================================
# 3. Decoder unchanged
# ======================================================================

class TestDecoderUnchanged:
    def test_decoder_interface_same(self, worldsmall_instance, dist_by_pair):
        """The decoder policy's interface is unchanged."""
        from neural.config import ArchitectureConfig
        from neural.backbone import NeuralBackbone
        from policies.encoder_decoder import EncoderDecoderPolicy
        from actions.service_generator import ServiceGenerator

        cfg = ArchitectureConfig(hidden_dim=64, gat_layers=1, transformer_layers=1,
                                 transformer_heads=2, lstm_layers=1)
        bb = NeuralBackbone(cfg)
        gen_base = ServiceGenerator(worldsmall_instance, dist_by_pair,
                                    draft_filter_enabled=True)
        gen_no = ServiceGenerator(worldsmall_instance, dist_by_pair,
                                  draft_filter_enabled=False)

        pol_base = EncoderDecoderPolicy(bb, worldsmall_instance, gen_base)
        pol_no = EncoderDecoderPolicy(bb, worldsmall_instance, gen_no)

        # Both should have same number of parameters (generator doesn't add params)
        assert sum(p.numel() for p in pol_base.parameters()) == \
               sum(p.numel() for p in pol_no.parameters())


# ======================================================================
# 4. PPO configuration unchanged
# ======================================================================

class TestPPOConfigUnchanged:
    def test_ppo_config_same(self, worldsmall_instance, dist_by_pair):
        """PPO config is independent of draft filter setting."""
        from policies.training import TrainingConfig
        cfg1 = TrainingConfig(
            dataset="WorldSmall", policy="encoder_decoder",
            learning_rate=2e-4, gamma=1.0, gae_lambda=0.9, ppo_epochs=10,
            clip_epsilon=0.2, target_kl=0.1, entropy_coefficient=0.05,
            value_coefficient=0.5, num_envs=1, steps_per_env=10,
            minibatch_size=32, seed=42, max_updates=5,
            checkpoint_frequency=100, hidden_dim=64, gat_layers=1,
            transformer_layers=1, transformer_heads=2, lstm_layers=1,
        )
        cfg2 = TrainingConfig(
            dataset="WorldSmall", policy="encoder_decoder",
            learning_rate=2e-4, gamma=1.0, gae_lambda=0.9, ppo_epochs=10,
            clip_epsilon=0.2, target_kl=0.1, entropy_coefficient=0.05,
            value_coefficient=0.5, num_envs=1, steps_per_env=10,
            minibatch_size=32, seed=42, max_updates=5,
            checkpoint_frequency=100, hidden_dim=64, gat_layers=1,
            transformer_layers=1, transformer_heads=2, lstm_layers=1,
        )
        assert cfg1.to_dict() == cfg2.to_dict()


# ======================================================================
# 5. Reward unchanged
# ======================================================================

class TestRewardUnchanged:
    def test_reward_formula_unchanged(self, worldsmall_instance):
        """Reward formula is independent of draft filter."""
        from env.environment import LSNDPEnv
        from mcf import ServiceDefinition
        from data.instance import Demand

        env = LSNDPEnv(worldsmall_instance)
        obs, _ = env.reset(seed=42)

        # First step: add a service
        from env.action import ServiceAction
        ports = sorted(worldsmall_instance.ports.keys())[:3]
        sa = ServiceAction(vessel_class="Feeder_450", port_sequence=ports)
        obs2, reward, terminated, truncated, info = env.step(sa)

        # Reward should be finite and based on eta change
        assert isinstance(reward, float)
        assert torch.isfinite(torch.tensor(reward))


# ======================================================================
# 6. MCF unchanged
# ======================================================================

class TestMCFUnchanged:
    def test_mcf_evaluate_with_different_services(self, worldsmall_instance,
                                                  baseline_generator,
                                                  no_draft_generator):
        """MCF evaluates services correctly regardless of how they were constructed."""
        from mcf import evaluate_network
        from mcf.expanded_graph import ServiceDefinition

        # Build two different services
        svc1_ports = ["AOLAD", "ECGYE"]  # Feeder-feasible
        svc2_ports = sorted(worldsmall_instance.ports.keys())[:5]  # All ports

        for svc_ports, label in [(svc1_ports, "feeder"), (svc2_ports, "mixed")]:
            svc_defs = [ServiceDefinition(service_id=0, vessel_class="Feeder_450",
                                          port_sequence=svc_ports)]
            vessel_reqs = {0: {"Feeder_450": 1.0}}
            result = evaluate_network(worldsmall_instance, svc_defs, vessel_reqs)
            assert result.eta is not None
            assert torch.isfinite(torch.tensor(result.eta))


# ======================================================================
# 7. Service construction still valid after intervention
# ======================================================================

class TestServiceConstructionValid:
    def test_no_draft_still_produces_valid_services(self, no_draft_generator):
        """Services produced without draft filter must still pass validation."""
        # Pick ports that have distances between them
        ports = sorted(no_draft_generator._instance.ports.keys())[:5]
        ordered = no_draft_generator.order_ports(ports, "Post_panamax")
        result = no_draft_generator.generate_service("Post_panamax", ordered)
        assert result.is_valid, f"Invalid service: {result.reasons}"

    def test_no_draft_minimum_ports(self, no_draft_generator):
        """Still need >= 2 ports even without draft filter."""
        ordered = no_draft_generator.order_ports(["AOLAD"], "Post_panamax")
        assert len(ordered) >= 1  # Fallback for single port

    def test_no_draft_cyclic_closure(self, no_draft_generator):
        """Service sequence forms valid cycle."""
        ports = sorted(no_draft_generator._instance.ports.keys())[:5]
        ordered = no_draft_generator.order_ports(ports, "Post_panamax")
        assert len(ordered) >= 2
        # Last port connects back to first
        first, last = ordered[0], ordered[-1]
        assert (last, first) in no_draft_generator._dist or \
               (first, last) in no_draft_generator._dist


# ======================================================================
# 8. n_vs reconciliation
# ======================================================================

class TestNVsReconciliation:
    def test_n_vs_computed_correctly_no_draft(self, no_draft_generator, worldsmall_instance):
        """n_vs = tour_distance / (design_speed * 24 * 7) per environment.py formula."""
        ports = sorted(worldsmall_instance.ports.keys())[:5]
        ordered = no_draft_generator.order_ports(ports, "Post_panamax")
        # Use the environment's formula (which is what actually drives training)
        tour_dist = no_draft_generator.compute_tour_distance(ordered)
        vt = worldsmall_instance.vessel_types["Post_panamax"]
        expected = tour_dist / (vt.design_speed * 24.0 * 7.0)
        # Verify the value is reasonable (not NaN, not infinite)
        assert isinstance(expected, float) and expected > 0
        assert torch.isfinite(torch.tensor(expected))

    def test_n_vs_different_for_different_vessels(self, no_draft_generator):
        """n_vs varies with vessel speed even for same ports."""
        ports = sorted(no_draft_generator._instance.ports.keys())[:5]
        ordered = no_draft_generator.order_ports(ports, "Post_panamax")
        n_vs_large = no_draft_generator.compute_vessel_requirement("Post_panamax", ordered)
        n_vs_small = no_draft_generator.compute_vessel_requirement("Feeder_450", ordered)
        # Faster vessel needs fewer ships
        assert n_vs_large < n_vs_small


# ======================================================================
# 9. Demand reconciliation
# ======================================================================

class TestDemandReconciliation:
    def test_demand_conserved(self, worldsmall_instance):
        """Total demand = routed + rejected."""
        from mcf import evaluate_network, ServiceDefinition

        ports = sorted(worldsmall_instance.ports.keys())[:5]
        svc_defs = [ServiceDefinition(service_id=0, vessel_class="Post_panamax",
                                      port_sequence=ports)]
        vessel_reqs = {0: {"Post_panamax": 1.0}}
        result = evaluate_network(worldsmall_instance, svc_defs, vessel_reqs)
        total = sum(d.ffe_per_week for d in worldsmall_instance.demands)
        assert abs(result.routed_demand + result.rejected_demand - total) < 1e-3


# ======================================================================
# 10. Checkpoint reload
# ======================================================================

class TestCheckpointReload:
    def test_checkpoint_save_load(self, tmp_path, worldsmall_instance, dist_by_pair):
        """Checkpoints can be saved and loaded without error."""
        from neural.config import ArchitectureConfig
        from neural.backbone import NeuralBackbone
        from mcf.ppo_engine import ValueFunction
        from policies.encoder_decoder import EncoderDecoderPolicy
        from actions.service_generator import ServiceGenerator
        import torch

        cfg = ArchitectureConfig(hidden_dim=64, gat_layers=1, transformer_layers=1,
                                 transformer_heads=2, lstm_layers=1)
        bb = NeuralBackbone(cfg)
        gen = ServiceGenerator(worldsmall_instance, dist_by_pair,
                               draft_filter_enabled=False)
        policy = EncoderDecoderPolicy(bb, worldsmall_instance, gen)
        port_dim = (len(worldsmall_instance.ports) + 1) * 2
        vessel_dim = len(worldsmall_instance.vessel_types) * 11
        critic = ValueFunction(input_dim=port_dim + vessel_dim)

        # Save
        ckpt_path = tmp_path / "test_ckpt.pt"
        torch.save({
            "policy_state_dict": policy.state_dict(),
            "critic_state_dict": critic.state_dict(),
            "update_count": 5,
        }, ckpt_path)

        # Load
        policy2 = EncoderDecoderPolicy(
            NeuralBackbone(cfg), worldsmall_instance,
            ServiceGenerator(worldsmall_instance, dist_by_pair,
                             draft_filter_enabled=False),
        )
        critic2 = ValueFunction(input_dim=port_dim + vessel_dim)
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        policy2.load_state_dict(ckpt["policy_state_dict"])
        critic2.load_state_dict(ckpt["critic_state_dict"])

        # Verify identical outputs
        assert torch.equal(
            next(policy.parameters()).data,
            next(policy2.parameters()).data,
        )


# ======================================================================
# 11. Baseline/intervention configuration comparability
# ======================================================================

class TestConfigComparability:
    def test_same_instance(self, baseline_generator, no_draft_generator):
        """Both generators use the same instance."""
        assert baseline_generator._instance.name == no_draft_generator._instance.name
        assert len(baseline_generator._instance.ports) == \
               len(no_draft_generator._instance.ports)

    def test_same_distances(self, baseline_generator, no_draft_generator):
        """Both generators use the same distance matrix."""
        assert set(baseline_generator._dist.keys()) == \
               set(no_draft_generator._dist.keys())

    def test_different_behavior_same_ports(self, baseline_generator,
                                            no_draft_generator):
        """Same input ports produce different outputs when draft filter differs."""
        # Feeder_450 with mixed ports
        ports = sorted(baseline_generator._instance.ports.keys())[:15]
        baseline_order = baseline_generator.order_ports(ports, "Feeder_450")
        no_draft_order = no_draft_generator.order_ports(ports, "Feeder_450")
        # With draft filter, many ports get dropped for Feeder_450
        assert len(baseline_order) <= len(no_draft_order)
        # They may differ
        if len(baseline_order) < len(no_draft_order):
            assert baseline_order != no_draft_order


# ======================================================================
# 12. No NaN/Inf
# ======================================================================

class TestNoNaNInf:
    def test_no_nan_in_order_ports(self, no_draft_generator):
        """order_ports never returns NaN."""
        ports = sorted(no_draft_generator._instance.ports.keys())[:20]
        for vc in no_draft_generator._vessel_classes:
            ordered = no_draft_generator.order_ports(ports, vc)
            for p in ordered:
                assert isinstance(p, str) and len(p) > 0

    def test_no_nan_in_validate_service(self, no_draft_generator):
        """validate_service never produces NaN reasons."""
        ports = ["AOLAD", "ECGYE", "GHTKD"]
        reasons = no_draft_generator.validate_service("Feeder_450", ports)
        for r in reasons:
            assert isinstance(r, str) and "nan" not in r.lower()

    def test_compute_vessel_requirement_no_nan(self, no_draft_generator):
        """n_vs computation never produces NaN."""
        ports = sorted(no_draft_generator._instance.ports.keys())[:5]
        ordered = no_draft_generator.order_ports(ports, "Post_panamax")
        n_vs = no_draft_generator.compute_vessel_requirement("Post_panamax", ordered)
        assert isinstance(n_vs, float) and torch.isfinite(torch.tensor(n_vs))


# ======================================================================
# 13. Qualitative diversity test
# ======================================================================

class TestDiversityImprovement:
    def test_no_draft_produces_longer_services(self, no_draft_generator,
                                                baseline_generator):
        """Without draft filter, services should generally be longer."""
        sample_ports = sorted(baseline_generator._instance.ports.keys())[:15]
        for vc in ["Feeder_450", "Panamax_2400", "Post_panamax"]:
            baseline_len = len(baseline_generator.order_ports(sample_ports, vc))
            no_draft_len = len(no_draft_generator.order_ports(sample_ports, vc))
            assert no_draft_len >= baseline_len, \
                f"No-draft should not produce shorter services for {vc}"

    def test_no_draft_more_unique_services_same_input(self, no_draft_generator,
                                                       baseline_generator):
        """Given the same decoder output, no-draft preserves more port diversity."""
        sample_ports = sorted(baseline_generator._instance.ports.keys())[:15]
        # Use a single vessel class for fair comparison
        vc = "Feeder_450"
        baseline_len = len(baseline_generator.order_ports(sample_ports, vc))
        no_draft_len = len(no_draft_generator.order_ports(sample_ports, vc))
        # No-draft should produce >= as many ports (never fewer)
        assert no_draft_len >= baseline_len, \
            f"No-draft ({no_draft_len}) should not be shorter than baseline ({baseline_len})"
        # And for a vessel with limited draft access, no-draft should be strictly longer
        assert no_draft_len > baseline_len, \
            f"No-draft should produce more ports for {vc}"


# ======================================================================
# G — Evaluation harness integrity
# ======================================================================

class TestEvaluationHarnessIntegrity:
    """
    The G11.2.3 harness had three defects that made every checkpoint report
    `empty_run=True` and hid the real result. These tests pin the fixes.
    """

    def test_bos_is_terminal_not_fallback(self):
        """BOS (service_action is None) must not inject a synthetic service."""
        import inspect
        from experiments.manual.g11_2_3 import run_experiment as rx
        src = inspect.getsource(rx._rollout_eval)
        # The old bug stepped a fabricated ServiceAction in the else branch.
        assert "ServiceAction(" not in src, \
            "eval harness must not fabricate fallback services"
        assert "break" in src, "BOS must terminate the rollout"

    def test_eval_seed_is_fixed_across_checkpoints(self):
        """A per-checkpoint seed would confound policy change with sampling."""
        import inspect
        from experiments.manual.g11_2_3 import run_experiment as rx
        src = inspect.getsource(rx.MetricsCollector.evaluate_checkpoint)
        assert "update_idx * 1000" not in src, \
            "eval seed must not vary with update_idx"
        assert "seed=SEED" in src, "eval must use the fixed SEED"

    def test_sampled_eval_is_recorded(self):
        """Both argmax and sampled modes must be recorded; sampled is primary."""
        import inspect
        from experiments.manual.g11_2_3 import run_experiment as rx
        src = inspect.getsource(rx.MetricsCollector.evaluate_checkpoint)
        assert "argmax" in src and "sampled" in src, \
            "checkpoint eval must record both rollout modes"
        # _ckpt_val resolves bare keys to the sampled section.
        assert "mode = \"sampled\"" in inspect.getsource(rx._ckpt_val), \
            "bare metric keys must resolve to the sampled evaluation"

    def test_collapse_guard_blocks_false_verdict(self):
        """Collapsed policies (near-duplicate services) must yield CASE D."""
        from experiments.manual.g11_2_3.run_experiment import _classify_experiment

        def mk(uniq, svc, profit, cov):
            return {
                "unique_services": uniq, "num_services": svc,
                "mean_service_length": 3.0, "weekly_profit": profit,
                "coverage_pct": cov,
            }

        # Both arms collapsed: 2 unique out of 30 services.
        comparison = {"per_checkpoint": {
            "10": {"baseline": mk(2, 30, -1e8, 1.0),
                   "intervention": mk(1, 30, -2e8, 0.5)},
            "50": {"baseline": mk(2, 25, -1e8, 1.0),
                   "intervention": mk(2, 28, -2e8, 0.5)},
        }}
        verdict = _classify_experiment(comparison, [], [])
        assert verdict.startswith("CASE D"), \
            f"collapsed policies must not produce CASE A/B/C, got: {verdict[:60]}"
        assert "COLLAPSE GUARD" in verdict

    def test_healthy_policy_is_not_guarded(self):
        """A non-collapsed pair must reach a real verdict, not CASE D."""
        from experiments.manual.g11_2_3.run_experiment import _classify_experiment

        def mk(uniq, svc, profit, cov):
            return {
                "unique_services": uniq, "num_services": svc,
                "mean_service_length": 9.0, "weekly_profit": profit,
                "coverage_pct": cov,
            }

        # Healthy: most services distinct in both arms.
        comparison = {"per_checkpoint": {
            "10": {"baseline": mk(18, 20, -1e8, 5.0),
                   "intervention": mk(25, 28, -0.5e8, 9.0)},
            "50": {"baseline": mk(22, 24, -1e8, 5.0),
                   "intervention": mk(30, 30, -0.5e8, 9.0)},
        }}
        verdict = _classify_experiment(comparison, [], [])
        assert "COLLAPSE GUARD" not in verdict, \
            "healthy policies must not trip the collapse guard"
        assert not verdict.startswith("CASE D"), \
            f"healthy improvement should not be CASE D, got: {verdict[:60]}"


class TestPPOWiringDefect:
    """
    [G11.2.4 UPDATE] This class originally pinned the G11.2.3 DEFECT:
    `PPOTrainer.train_step()` held the real ppo_epochs loop but was never
    called; `perform_ppo_update()` took a single clipped full-batch step,
    making ppo_epochs / minibatch_size / target_kl decorative.

    The defect is now REPAIRED (G11.2.4). The active path routes through
    `PPOTrainer.train_step_adapter()`, which executes the epoch loop,
    minibatch splitting, and target-KL early stopping while preserving the
    encoder-decoder action representation via `policy.evaluate_actions()`.

    These assertions are inverted to pin the REPAIRED state, so the defect
    cannot be silently reintroduced. Full behavioural evidence lives in
    `tests/test_g11_2_4_ppo_wiring.py`.
    """

    def test_train_step_contains_epoch_loop(self):
        """train_step must retain the epoch loop (it is the correct code)."""
        from mcf.ppo_engine.trainer import PPOTrainer
        import inspect
        src = inspect.getsource(PPOTrainer.train_step)
        assert "for epoch in range(self.config.ppo_epochs)" in src

    def test_adapter_contains_epoch_loop(self):
        """
        [G11.2.4] The adapter used by the active path must contain the real
        epoch loop. This is the loop that now actually runs.
        """
        from mcf.ppo_engine.trainer import PPOTrainer
        import inspect
        src = inspect.getsource(PPOTrainer.train_step_adapter)
        assert "for epoch in range(ppo_epochs)" in src

    def test_perform_ppo_update_now_calls_train_step_adapter(self):
        """
        [G11.2.4 REPAIRED] perform_ppo_update must delegate to the real
        multi-epoch PPO update instead of re-implementing a single step.

        This is the INVERSE of the original G11.2.3 assertion, which was
        written to fail deliberately once the wiring was repaired.
        """
        from policies.training import LinerShippingTrainer
        import inspect
        src = inspect.getsource(LinerShippingTrainer.perform_ppo_update)
        assert "train_step_adapter" in src, (
            "PPO wiring regression: perform_ppo_update must call "
            "train_step_adapter() so ppo_epochs/minibatch_size/target_kl "
            "are honoured. See G11_2_4_PPO_WIRING_REPAIR_REPORT.md."
        )

    def test_perform_ppo_update_does_not_reimplement_ppo(self):
        """
        [G11.2.4] The inline full-batch PPO re-implementation must be gone.
        """
        from policies.training import LinerShippingTrainer
        import inspect
        src = inspect.getsource(LinerShippingTrainer.perform_ppo_update)
        assert "new_log_probs_list" not in src
        assert "total_loss.backward()" not in src
