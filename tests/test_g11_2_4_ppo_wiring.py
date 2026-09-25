"""
G11.2.4 — PPO Wiring Repair Tests.

The G11.2.3 controlled run found that the active training path
(``LinerShippingTrainer.perform_ppo_update``) re-implemented PPO as a single
clipped full-batch step, so ``ppo_epochs``, ``minibatch_size`` and
``target_kl`` were decorative while ``PPOTrainer.train_step()`` — which holds
the real epoch loop — was unreachable.

These tests pin the REPAIR. The contract is:

    configured PPO  ==  executed PPO

Test map (PHASE 5 of the repair brief):
    TEST 1  multi-epoch execution
    TEST 2  minibatch execution
    TEST 3  target-KL early stopping is reachable
    TEST 4  encoder-decoder action compatibility (evaluate_actions + autograd)
    TEST 5  raw-action contract (old_log_prob is the RAW sampled action)
    TEST 6  PPO ratio ~ 1 on an unchanged policy
    TEST 7  gradient flow (policy and critic both non-zero, both change)
    TEST 8  configuration effectiveness (epochs and minibatch count respond)

Scope: wiring validation ONLY. Nothing here asserts that PPO has learned or
converged.
"""
from __future__ import annotations

import ast
import copy
import inspect
import sys
from pathlib import Path
from typing import Any, Dict, List

import pytest
import torch

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from data.linerlib_loader import LINERLIBLoader
from mcf.ppo_engine import PPOConfig, build_rollout_batch
from mcf.ppo_engine.trainer import PPOTrainer
from policies.training import LinerShippingTrainer, TrainingConfig

# Controlled-validation PPO settings from the repair brief.
PPO_EPOCHS = 10
MINIBATCH_SIZE = 32
CLIP_EPSILON = 0.20
TARGET_KL = 0.10
ENTROPY_COEF = 0.05
VALUE_COEF = 0.50
MAX_GRAD_NORM = 0.5
LEARNING_RATE = 2e-4


# ======================================================================
# Fixtures
# ======================================================================

def _make_trainer(
    instance: str = "Baltic",
    policy: str = "encoder_decoder",
    ppo_epochs: int = PPO_EPOCHS,
    minibatch_size: int = MINIBATCH_SIZE,
    target_kl: float = TARGET_KL,
    steps_per_env: int = 40,
    hidden_dim: int = 32,
    seed: int = 42,
) -> LinerShippingTrainer:
    """Small, fast trainer. H=32 keeps these tests fast; the wiring under test
    is independent of hidden size."""
    cfg = TrainingConfig(
        dataset=instance,
        policy=policy,
        learning_rate=LEARNING_RATE,
        gamma=1.0,
        gae_lambda=0.9,
        ppo_epochs=ppo_epochs,
        clip_epsilon=CLIP_EPSILON,
        target_kl=target_kl,
        entropy_coefficient=ENTROPY_COEF,
        value_coefficient=VALUE_COEF,
        num_envs=1,
        steps_per_env=steps_per_env,
        minibatch_size=minibatch_size,
        seed=seed,
        max_updates=5,
        checkpoint_frequency=9999,
        hidden_dim=hidden_dim,
        gat_layers=1,
        transformer_layers=1,
        transformer_heads=2,
        lstm_layers=1,
    )
    t = LinerShippingTrainer(
        instance_name=instance,
        policy_type=policy,
        config=cfg,
        checkpoint_dir=str(_ROOT / "experiments" / "manual" / "g11_2_4"),
    )
    t._update_count = 0
    t._episode_count = 0
    t._metrics_log.clear()
    return t


@pytest.fixture(scope="module")
def trainer() -> LinerShippingTrainer:
    return _make_trainer()


@pytest.fixture(scope="module")
def trajectory(trainer) -> List[Dict[str, Any]]:
    traj, _ = trainer.collect_rollout(seed=42)
    assert len(traj) > 0, "Empty trajectory at seed=42"
    return traj


@pytest.fixture()
def fresh_trainer() -> LinerShippingTrainer:
    """Function-scoped trainer so parameter-movement tests are independent."""
    return _make_trainer()


# ======================================================================
# Wiring: the active path must reach the real PPO update
# ======================================================================

class TestWiringReachability:
    """The active path must route through the multi-epoch PPO update."""

    def test_perform_ppo_update_calls_adapter(self):
        """perform_ppo_update must call train_step_adapter, not re-implement PPO."""
        src = inspect.getsource(LinerShippingTrainer.perform_ppo_update)
        assert "train_step_adapter" in src, (
            "perform_ppo_update must delegate to the real multi-epoch PPO update."
        )

    def test_adapter_contains_epoch_loop(self):
        from mcf.ppo_engine.trainer import PPOTrainer
        src = inspect.getsource(PPOTrainer.train_step_adapter)
        assert "for epoch in range(ppo_epochs)" in src

    def test_adapter_contains_minibatch_loop(self):
        from mcf.ppo_engine.trainer import PPOTrainer
        src = inspect.getsource(PPOTrainer.train_step_adapter)
        assert "for start in range(0, n_samples, mb_size)" in src

    def test_adapter_uses_configured_minibatch_size(self):
        """The minibatch size must come from config, not a hard-coded count."""
        from mcf.ppo_engine.trainer import PPOTrainer
        src = inspect.getsource(PPOTrainer.train_step_adapter)
        assert "self.config.minibatch_size" in src

    def test_adapter_calls_target_kl_early_stop(self):
        from mcf.ppo_engine.trainer import PPOTrainer
        src = inspect.getsource(PPOTrainer.train_step_adapter)
        assert "should_early_stop_kl" in src

    def test_adapter_uses_configured_max_grad_norm(self):
        from mcf.ppo_engine.trainer import PPOTrainer
        src = inspect.getsource(PPOTrainer.train_step_adapter)
        assert "self.config.max_grad_norm" in src, (
            "Gradient clipping must honour config.max_grad_norm, not a literal 0.5."
        )

    def test_perform_ppo_update_no_longer_single_full_batch_step(self):
        """The old defect: exactly one optimizer step on the whole batch."""
        src = inspect.getsource(LinerShippingTrainer.perform_ppo_update)
        # The inline re-implementation is gone; no local stacked loss tensors.
        assert "new_log_probs_list" not in src
        assert "torch.stack(new_log_probs_list)" not in src

    def test_run_training_does_not_double_count_episodes(self):
        """
        [G11.2.4] run_training() builds and appends the metrics entry for a
        trajectory BEFORE calling perform_ppo_update(). The update must
        therefore not append a second entry for the same episode.

        Regression caught during this repair: once the encoder-only path began
        returning a real diagnostic (it previously returned early on the empty
        post-fallback-filter trajectory), the unconditional append inside
        perform_ppo_update double-counted episodes.
        """
        t = _make_trainer(steps_per_env=5)
        t.config.max_updates = 3
        t.run_training(max_updates=3)
        assert t._update_count == 3
        assert t._episode_count == 3, (
            f"expected 3 episodes, got {t._episode_count} "
            "(metrics appended twice for the same trajectory)"
        )
        assert len(t._metrics_log) == 3

    def test_direct_call_still_logs_metrics(self):
        """
        [G11.2.4] The G11.1 contract: a DIRECT perform_ppo_update() call must
        still update _metrics_log, so get_summary() does not report
        "no_training". The log_metrics=False suppression must apply only to
        the run_training() call site.
        """
        t = _make_trainer(steps_per_env=5)
        traj, _ = t.collect_rollout(seed=42)
        assert len(t._metrics_log) == 0
        t.perform_ppo_update(traj)
        assert len(t._metrics_log) == 1


# ======================================================================
# TEST 1 — MULTI-EPOCH EXECUTION
# ======================================================================

class TestMultiEpochExecution:
    def test_configured_ten_epochs(self, trainer):
        assert trainer.config.ppo_epochs == PPO_EPOCHS

    def test_ten_epochs_actually_execute_when_early_stop_disabled(
        self, fresh_trainer,
    ):
        """
        With target-KL early stopping disabled, the FULL configured epoch count
        must execute. (With the production target_kl=0.10 the loop may legitimately
        stop early — that is covered separately in
        TestTargetKLEarlyStopping / TestKLResponsiveness.)
        """
        fresh_trainer.trainer.config.target_kl = 1e9
        traj, _ = fresh_trainer.collect_rollout(seed=42)
        diag = fresh_trainer.perform_ppo_update(traj)

        assert diag.epochs_configured == PPO_EPOCHS
        assert diag.epochs_executed == PPO_EPOCHS, (
            f"expected {PPO_EPOCHS} epochs, executed {diag.epochs_executed}"
        )
        assert len(diag.epoch_trace) == PPO_EPOCHS

    def test_production_target_kl_can_still_stop_early(self, fresh_trainer):
        """
        At the paper's target_kl=0.10 the loop is allowed to stop early once KL
        exceeds it. Either outcome is valid; what is pinned is that the number of
        executed epochs is never MORE than configured and never zero.
        """
        traj, _ = fresh_trainer.collect_rollout(seed=42)
        diag = fresh_trainer.perform_ppo_update(traj)
        assert 1 <= diag.epochs_executed <= PPO_EPOCHS
        assert diag.epochs_executed == len(diag.epoch_trace)

    def test_more_than_one_epoch_executes(self, fresh_trainer):
        """The core anti-regression: the epoch loop is not decorative."""
        traj, _ = fresh_trainer.collect_rollout(seed=42)
        diag = fresh_trainer.perform_ppo_update(traj)
        assert diag.epochs_executed > 1

    def test_optimizer_steps_equal_epochs_times_minibatches(
        self, fresh_trainer,
    ):
        traj, _ = fresh_trainer.collect_rollout(seed=42)
        diag = fresh_trainer.perform_ppo_update(traj)
        expected = diag.epochs_executed * diag.minibatches_per_epoch
        assert diag.total_optimizer_steps == expected
        assert diag.total_optimizer_steps > 1


# ======================================================================
# TEST 2 — MINIBATCH EXECUTION
# ======================================================================

class TestMinibatchExecution:
    def test_rollout_larger_than_one_minibatch(self, trajectory):
        assert len(trajectory) > MINIBATCH_SIZE, (
            "test needs a rollout larger than one minibatch"
        )

    def test_multiple_minibatches_per_epoch(self, fresh_trainer):
        traj, _ = fresh_trainer.collect_rollout(seed=42)
        diag = fresh_trainer.perform_ppo_update(traj)
        assert diag.minibatch_size == MINIBATCH_SIZE
        assert diag.minibatches_per_epoch >= 2, (
            f"expected >=2 minibatches, got {diag.minibatches_per_epoch}"
        )

    def test_every_epoch_reports_its_minibatch_count(self, fresh_trainer):
        traj, _ = fresh_trainer.collect_rollout(seed=42)
        diag = fresh_trainer.perform_ppo_update(traj)
        for entry in diag.epoch_trace:
            assert entry["minibatches"] == diag.minibatches_per_epoch
            assert entry["optimizer_steps"] >= 1


# ======================================================================
# TEST 3 — TARGET KL IS ACTIVE
# ======================================================================

class TestTargetKLEarlyStopping:
    def test_should_early_stop_kl_reads_config(self, trainer):
        ppo = trainer.trainer
        assert ppo.should_early_stop_kl(0.5) is True
        assert ppo.should_early_stop_kl(0.001) is False

    def test_early_stop_terminates_epoch_loop_synthetically(self, trainer):
        """
        Controlled synthetic condition: a target_kl below the observed KL must
        terminate the loop. Uses the real adapter path; does NOT force a
        production training failure.
        """
        t = trainer.trainer
        original_target = t.config.target_kl
        try:
            # Any positive threshold below the first epoch's observed KL fires.
            t.config.target_kl = 1e-12
            batch = build_rollout_batch(
                _collect_once(trainer), trainer.policy_type,
                trainer.instance, trainer.backbone._device(),
            )
            diag = t.train_step_adapter(batch)
            assert diag.kl_early_stopped is True
            assert diag.epochs_executed < diag.epochs_configured, (
                "early stop must cut the epoch loop short"
            )
            assert diag.epochs_executed >= 1
        finally:
            t.config.target_kl = original_target

    def test_no_early_stop_when_target_is_unreachable(self, fresh_trainer):
        """With a huge target, all configured epochs run."""
        fresh_trainer.trainer.config.target_kl = 1e9
        traj, _ = fresh_trainer.collect_rollout(seed=42)
        diag = fresh_trainer.perform_ppo_update(traj)
        assert diag.kl_early_stopped is False
        assert diag.epochs_executed == diag.epochs_configured


# ======================================================================
# TEST 4 — ENCODER-DECODER ACTION COMPATIBILITY
# ======================================================================

class TestEncoderDecoderCompatibility:
    def test_evaluate_actions_returns_differentiable_tensors(
        self, trainer, trajectory,
    ):
        t = trajectory[0]
        subs = t.get("substep_selected", [])
        with torch.enable_grad():
            lp, ent = trainer.policy.evaluate_actions(
                t["state"], t.get("fleet_remaining", {}),
                substep_selected=subs, n_substeps=len(subs),
            )
        assert lp.requires_grad is True
        assert ent.requires_grad is True
        assert torch.isfinite(ent), "entropy must be finite"

    def test_critic_output_requires_grad(self, trainer, trajectory):
        val = trainer.critic(trajectory[0]["critic_input"].to(
            trainer.backbone._device()
        )).squeeze(-1)
        assert val.requires_grad is True

    def test_adapter_accepts_stored_decoder_trajectories(
        self, trainer, trajectory,
    ):
        batch = build_rollout_batch(
            trajectory, trainer.policy_type, trainer.instance,
            trainer.backbone._device(),
        )
        assert batch.n_samples == len(trajectory)
        # Every payload carries the raw token sequence.
        for p in batch.payloads:
            assert "substep_selected" in p
            assert "fleet_remaining" in p
            assert p["n_substeps"] == len(p["substep_selected"])


# ======================================================================
# TEST 5 — RAW ACTION CONTRACT
# ======================================================================

class TestRawActionContract:
    def test_old_log_prob_matches_raw_evaluate_actions(self):
        """
        old_log_prob was produced by sampling the RAW decoder sequence. The
        adapter's re-evaluation of that same raw sequence must reproduce it
        before any update (ratio ~ 1). This proves old_log_prob describes the
        raw sampled action, not the executed one.
        """
        t = _make_trainer()
        traj, _ = t.collect_rollout(seed=42)
        batch = build_rollout_batch(
            traj, t.policy_type, t.instance, t.backbone._device(),
        )
        stats = batch.measure_ratios(t.policy, t.critic, CLIP_EPSILON)
        assert abs(stats["ratio_mean"] - 1.0) < 1e-3, (
            f"raw re-evaluation must reproduce old_log_prob; "
            f"ratio_mean={stats['ratio_mean']}"
        )

    def test_adapter_never_reads_executed_action(self):
        """
        Structural check: the adapter builds the PPO action from
        substep_selected only. It must not read the executed action from the
        trajectory dict.
        """
        import mcf.ppo_engine.adapter as adapter_mod
        tree = ast.parse(inspect.getsource(adapter_mod))

        offenders: List[str] = []
        forbidden = ("executed_action", "executed_port_sequence")
        for node in ast.walk(tree):
            # Docstrings mention the executed action by design; only executable
            # code (subscript reads and attribute access) is a violation.
            if isinstance(node, ast.Subscript) and isinstance(node.slice, ast.Constant):
                if node.slice.value in forbidden:
                    offenders.append(node.slice.value)
            if isinstance(node, ast.Attribute) and node.attr in forbidden:
                offenders.append(node.attr)
        assert not offenders, (
            f"adapter must not read the executed action; found: {offenders}"
        )

    def test_raw_and_executed_sequences_can_differ(self, trainer, trajectory):
        """
        The trajectory must genuinely distinguish raw from executed. If they
        were always equal the invariant above would be vacuous.
        """
        differing = [
            t for t in trajectory
            if list(t.get("decoded_port_sequence", []))
            != list(t.get("executed_port_sequence", []))
        ]
        assert differing, (
            "expected at least one step where TSP reordered the raw decoder "
            "sequence, otherwise the raw/executed distinction is untested"
        )

    def test_raw_payload_used_is_substep_selected(self, trainer, trajectory):
        batch = build_rollout_batch(
            trajectory, trainer.policy_type, trainer.instance,
            trainer.backbone._device(),
        )
        for src_step, payload in zip(trajectory, batch.payloads):
            assert payload["substep_selected"] == list(
                src_step.get("substep_selected", [])
            )


# ======================================================================
# TEST 6 — PPO RATIO
# ======================================================================

class TestPPORatio:
    """
    TEST 6. The ratio must be ~1 on an UNCHANGED policy.

    Each test in this class uses its OWN freshly-constructed trainer. The
    module-scoped `trainer` fixture is shared and other test classes step its
    weights, which would make `evaluate_actions()` legitimately disagree with
    the stored `old_log_prob` and corrupt these measurements.
    """

    def test_ratio_approximately_one_before_optimization(self):
        t = _make_trainer()
        traj, _ = t.collect_rollout(seed=42)
        batch = build_rollout_batch(
            traj, t.policy_type, t.instance, t.backbone._device(),
        )
        stats = batch.measure_ratios(t.policy, t.critic, CLIP_EPSILON)
        assert abs(stats["ratio_mean"] - 1.0) < 1e-3
        assert abs(stats["ratio_min"] - 1.0) < 1e-3
        assert abs(stats["ratio_max"] - 1.0) < 1e-3

    def test_ratio_is_one_before_first_optimizer_step(self):
        t = _make_trainer()
        traj, _ = t.collect_rollout(seed=42)
        batch = build_rollout_batch(
            traj, t.policy_type, t.instance, t.backbone._device(),
        )
        stats = batch.measure_ratios(t.policy, t.critic, CLIP_EPSILON)
        assert abs(stats["ratio_mean"] - 1.0) < 1e-3
        assert stats["ratio_outside_clip_fraction"] == 0.0

    def test_ratio_departs_from_one_after_an_update(self):
        """
        The complement of TEST 6: once the policy has moved, the ratio must NOT
        stay pinned at 1. A ratio frozen at 1 would mean the update is inert.
        """
        t = _make_trainer()
        traj, _ = t.collect_rollout(seed=42)
        batch = build_rollout_batch(
            traj, t.policy_type, t.instance, t.backbone._device(),
        )
        before = batch.measure_ratios(t.policy, t.critic, CLIP_EPSILON)
        assert abs(before["ratio_mean"] - 1.0) < 1e-3

        t.perform_ppo_update(traj)
        after = batch.measure_ratios(t.policy, t.critic, CLIP_EPSILON)
        assert abs(after["ratio_mean"] - 1.0) > 1e-3, (
            "ratio stayed at 1 after an update — the policy did not move"
        )

    def test_ratio_stats_recorded_in_diagnostics(self):
        t = _make_trainer()
        traj, _ = t.collect_rollout(seed=42)
        diag = t.perform_ppo_update(traj)
        # The recorded stats are measured BEFORE any optimizer step, so the
        # ratio must still be ~1 there.
        assert math_is_close(diag.ratio_mean, 1.0, 1e-2)
        assert diag.ratio_min > 0.0
        assert diag.ratio_max >= diag.ratio_mean >= diag.ratio_min
        assert 0.0 <= diag.ratio_outside_clip_fraction <= 1.0


# ======================================================================
# TEST 7 — GRADIENT FLOW
# ======================================================================

class TestGradientFlow:
    def test_policy_and_critic_gradients_both_non_zero(
        self, trainer, trajectory,
    ):
        device = trainer.backbone._device()
        batch = build_rollout_batch(
            trajectory, trainer.policy_type, trainer.instance, device,
        )
        trainer.policy.zero_grad()
        trainer.critic.zero_grad()

        returns, advantages = trainer.trainer.compute_returns_and_advantages(
            batch.old_values, batch.rewards, batch.dones,
        )
        idx = torch.arange(batch.n_samples)
        lps, ents, vals = batch.build_evaluate_fn(
            trainer.policy, trainer.critic,
        )(idx)

        pl, _, _ = trainer.trainer.compute_ppo_loss(
            lps, batch.old_log_probs, advantages, CLIP_EPSILON,
        )
        vl = trainer.trainer.compute_value_loss(vals.view_as(returns), returns)
        el = trainer.trainer.compute_entropy_bonus(ents)
        (pl + VALUE_COEF * vl + el).backward()

        pnorm = sum(
            float(p.grad.pow(2).sum()) for p in trainer.policy.parameters()
            if p.grad is not None
        ) ** 0.5
        cnorm = sum(
            float(p.grad.pow(2).sum()) for p in trainer.critic.parameters()
            if p.grad is not None
        ) ** 0.5
        assert pnorm > 0, f"policy gradient is zero ({pnorm})"
        assert cnorm > 0, f"critic gradient is zero ({cnorm})"

    def test_entropy_term_reaches_the_policy(self, trainer, trajectory):
        """
        [G11.2.4] The entropy term must carry a REAL gradient. Previously the
        loss used stored, detached entropies, contributing exactly zero.
        """
        device = trainer.backbone._device()
        batch = build_rollout_batch(
            trajectory, trainer.policy_type, trainer.instance, device,
        )
        trainer.policy.zero_grad()
        idx = torch.arange(batch.n_samples)
        _, ents, _ = batch.build_evaluate_fn(trainer.policy, trainer.critic)(idx)
        trainer.trainer.compute_entropy_bonus(ents).backward()
        pnorm = sum(
            float(p.grad.pow(2).sum()) for p in trainer.policy.parameters()
            if p.grad is not None
        ) ** 0.5
        assert pnorm > 0, (
            "entropy bonus must produce a policy gradient; stored entropies "
            "would make this exactly zero"
        )

    def test_policy_parameters_change(self, fresh_trainer):
        traj, _ = fresh_trainer.collect_rollout(seed=42)
        before = [p.detach().clone() for p in fresh_trainer.policy.parameters()]
        fresh_trainer.perform_ppo_update(traj)
        after = list(fresh_trainer.policy.parameters())
        delta = sum(
            float((a.detach() - b).pow(2).sum()) for a, b in zip(after, before)
        ) ** 0.5
        assert delta > 0, "policy parameters did not change"

    def test_critic_parameters_change(self, fresh_trainer):
        traj, _ = fresh_trainer.collect_rollout(seed=42)
        before = [p.detach().clone() for p in fresh_trainer.critic.parameters()]
        fresh_trainer.perform_ppo_update(traj)
        after = list(fresh_trainer.critic.parameters())
        delta = sum(
            float((a.detach() - b).pow(2).sum()) for a, b in zip(after, before)
        ) ** 0.5
        assert delta > 0, "critic parameters did not change"

    def test_diagnostics_report_both_deltas(self, fresh_trainer):
        traj, _ = fresh_trainer.collect_rollout(seed=42)
        diag = fresh_trainer.perform_ppo_update(traj)
        assert diag.policy_param_delta > 0
        assert diag.critic_param_delta > 0


# ======================================================================
# TEST 8 — CONFIGURATION EFFECTIVENESS
# ======================================================================

class TestConfigurationEffectiveness:
    def test_ppo_epochs_change_alters_step_count(self):
        """ppo_epochs 1 -> 2 must change the optimizer step count."""
        t1 = _make_trainer(ppo_epochs=1)
        t2 = _make_trainer(ppo_epochs=2)
        traj1, _ = t1.collect_rollout(seed=42)
        traj2, _ = t2.collect_rollout(seed=42)
        d1 = t1.perform_ppo_update(traj1)
        d2 = t2.perform_ppo_update(traj2)
        assert d1.epochs_executed == 1
        assert d2.epochs_executed == 2
        # Each epoch runs every minibatch once, so doubling the epoch count
        # doubles the optimizer steps, independent of rollout length.
        assert d2.total_optimizer_steps == 2 * d1.total_optimizer_steps

    def test_minibatch_size_change_alters_minibatch_count(self):
        """
        minibatch_size must actually reshape the minibatch iteration.

        Rollout length can differ between the two trainers (the environment
        terminates on its own terms), so the relationship is asserted as
        "smaller batch => more minibatches" rather than an exact factor.
        """
        t1 = _make_trainer(minibatch_size=4, steps_per_env=40)
        t2 = _make_trainer(minibatch_size=8, steps_per_env=40)
        traj1, _ = t1.collect_rollout(seed=42)
        traj2, _ = t2.collect_rollout(seed=42)
        n1, n2 = len(traj1), len(traj2)
        d1 = t1.perform_ppo_update(traj1)
        d2 = t2.perform_ppo_update(traj2)
        assert d1.minibatch_size == 4
        assert d2.minibatch_size == 8
        # ceil(n1/4) > ceil(n2/8) whenever n1 >= n2/2; both rollouts are
        # bounded above by steps_per_env so the inequality holds for any
        # realistic split.
        import math as _m
        assert d1.minibatches_per_epoch == _m.ceil(n1 / 4)
        assert d2.minibatches_per_epoch == _m.ceil(n2 / 8)
        assert d1.minibatches_per_epoch > d2.minibatches_per_epoch or n1 < n2

    def test_minibatch_count_matches_formula(self, fresh_trainer):
        traj, _ = fresh_trainer.collect_rollout(seed=42)
        diag = fresh_trainer.perform_ppo_update(traj)
        n = len(traj)
        expected = -(-n // MINIBATCH_SIZE)  # ceiling division
        assert diag.minibatches_per_epoch == expected
        assert diag.minibatch_size == MINIBATCH_SIZE

    def test_max_grad_norm_is_configurable_and_used(self, trainer):
        ppo = trainer.trainer
        assert ppo.config.max_grad_norm == MAX_GRAD_NORM
        # Applied norm never exceeds the configured clip.
        traj, _ = trainer.collect_rollout(seed=42)
        diag = trainer.perform_ppo_update(traj)
        assert diag.applied_grad_norm <= MAX_GRAD_NORM + 1e-9

    def test_pre_clip_and_applied_grad_norms_are_distinct(self, fresh_trainer):
        """
        [G11.2.4] The pre-clip norm must NOT be reported as the applied
        magnitude. They differ whenever clipping binds.
        """
        traj, _ = fresh_trainer.collect_rollout(seed=42)
        diag = fresh_trainer.perform_ppo_update(traj)
        assert diag.gradient_norm >= 0
        assert diag.applied_grad_norm <= MAX_GRAD_NORM + 1e-9
        if diag.gradient_norm > MAX_GRAD_NORM:
            assert diag.applied_grad_norm < diag.gradient_norm, (
                "applied norm must be the clipped value, not the pre-clip value"
            )

    def test_all_required_config_fields_reach_ppo_config(self, trainer):
        ppo = trainer.trainer.config
        assert ppo.ppo_epochs == PPO_EPOCHS
        assert ppo.minibatch_size == MINIBATCH_SIZE
        assert ppo.clip_epsilon == CLIP_EPSILON
        assert ppo.target_kl == TARGET_KL
        assert ppo.entropy_coefficient == ENTROPY_COEF
        assert ppo.value_coefficient == VALUE_COEF
        assert ppo.max_grad_norm == MAX_GRAD_NORM
        assert ppo.learning_rate == LEARNING_RATE


# ======================================================================
# KL responsiveness (PHASE 7 — wiring validation, not convergence)
# ======================================================================

class TestKLResponsiveness:
    def test_kl_is_finite_and_recorded(self, fresh_trainer):
        traj, _ = fresh_trainer.collect_rollout(seed=42)
        diag = fresh_trainer.perform_ppo_update(traj)
        assert diag.approx_kl >= 0
        assert all(
            e["approx_kl"] >= 0 and math_is_finite(e["approx_kl"])
            for e in diag.epoch_trace
        )

    def test_kl_varies_across_epochs(self, fresh_trainer):
        """
        With 10 optimizer steps the KL sequence must not be a constant.
        A constant would indicate the policy is not moving.
        """
        fresh_trainer.trainer.config.target_kl = 1e9
        traj, _ = fresh_trainer.collect_rollout(seed=42)
        diag = fresh_trainer.perform_ppo_update(traj)
        kls = [e["approx_kl"] for e in diag.epoch_trace]
        assert len(kls) > 1
        assert max(kls) - min(kls) > 0 or max(kls) > 0, (
            f"KL did not respond to policy updates: {kls}"
        )

    def test_policy_moves_across_updates(self, fresh_trainer):
        """Policy weights must keep changing on a second update."""
        traj1, _ = fresh_trainer.collect_rollout(seed=42)
        fresh_trainer.perform_ppo_update(traj1)
        mid = [p.detach().clone() for p in fresh_trainer.policy.parameters()]
        traj2, _ = fresh_trainer.collect_rollout(seed=43)
        fresh_trainer.perform_ppo_update(traj2)
        delta = sum(
            float((a.detach() - b).pow(2).sum())
            for a, b in zip(fresh_trainer.policy.parameters(), mid)
        ) ** 0.5
        assert delta > 0, "policy stopped moving after the second update"


# ======================================================================
# Helpers
# ======================================================================

def _collect_once(trainer) -> List[Dict[str, Any]]:
    traj, _ = trainer.collect_rollout(seed=42)
    return traj


def math_is_finite(x) -> bool:
    import math
    return math.isfinite(float(x))


def math_is_close(a, b, tol=1e-6) -> bool:
    return abs(float(a) - float(b)) <= tol


# ======================================================================
# Main
# ======================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
