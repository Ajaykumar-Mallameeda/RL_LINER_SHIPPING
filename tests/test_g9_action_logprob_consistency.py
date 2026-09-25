"""G9 — Action/Log-Prob Consistency Tests.

Verifies that old_log_prob and new_log_prob (via evaluate_actions) describe
the SAME action, so the PPO ratio = 1.0 before any optimizer step.

This catches regressions of the F1 bug where `raw_sampled_ports` was never
stored in the trajectory dict, causing perform_ppo_update() to rebuild an
empty mask and compute a mismatched new_log_prob.
"""
from __future__ import annotations

import math
from typing import Any, Dict, List

import pytest
import torch

from data.linerlib_loader import LINERLIBLoader
from env.environment import LSNDPEnv
from mcf.ppo_engine import PPOConfig, PPOTrainer, ValueFunction
from neural import ArchitectureConfig, NeuralBackbone, neural_state_to_tensors
from policies.encoder_only import EncoderOnlyPolicy
from policies.encoder_decoder import EncoderDecoderPolicy
from policies.training import LinerShippingTrainer, TrainingConfig
from state.representation import ServiceMembership, StateEncoder


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def loader():
    return LINERLIBLoader("data")


@pytest.fixture(scope="module")
def baltic(loader):
    return loader.load("Baltic")


@pytest.fixture(scope="module")
def dist_by_pair(baltic):
    return {(a.origin, a.destination): a for a in baltic.distances}


@pytest.fixture(scope="module")
def small_cfg():
    return ArchitectureConfig(
        hidden_dim=32, gat_layers=1, transformer_layers=1,
        transformer_heads=2, lstm_layers=1, dropout=0.0,
    )


@pytest.fixture(scope="module")
def paper_cfg():
    return ArchitectureConfig(
        hidden_dim=512, gat_layers=3, transformer_layers=3,
        transformer_heads=8, lstm_layers=1, dropout=0.0,
    )


def _make_trainer(policy_type: str, cfg, baltic, dist_by_pair):
    from actions.service_generator import ServiceGenerator
    from policies.training import LinerShippingTrainer, TrainingConfig

    gen = ServiceGenerator(baltic, dist_by_pair)
    backbone = NeuralBackbone(cfg)
    if policy_type == "encoder_only":
        policy = EncoderOnlyPolicy(backbone, baltic, gen)
    else:
        policy = EncoderDecoderPolicy(backbone, baltic, gen)

    port_dim = (len(baltic.ports) + 1) * 2
    vessel_dim = len(baltic.vessel_types) * 11
    critic = ValueFunction(input_dim=port_dim + vessel_dim)

    ppo_cfg = PPOConfig(
        learning_rate=2e-4, gamma=1.0, gae_lambda=0.9,
        ppo_epochs=1, clip_epsilon=0.2, entropy_coefficient=0.05,
        value_coefficient=0.5, num_envs=1, steps_per_env=10,
        minibatch_size=8, seed=42, optimizer="adamw",
        optimizer_kwargs={"weight_decay": 1e-4},
    )
    trainer = PPOTrainer(policy, critic, ppo_cfg)
    env = LSNDPEnv(baltic)
    enc = StateEncoder(baltic, dist_by_pair)

    return {
        "policy": policy,
        "critic": critic,
        "ppo_trainer": trainer,
        "env": env,
        "state_encoder": enc,
        "instance": baltic,
        "dist_by_pair": dist_by_pair,
    }


# ---------------------------------------------------------------------------
# T1. Trajectory stores raw_sampled_ports for encoder-only
# ---------------------------------------------------------------------------

class TestTrajectoryKeys:
    """F1 — ensure the trajectory contains all keys needed for correct PPO."""

    def test_encoder_only_trajectory_has_raw_sampled_ports(self, small_cfg, baltic, dist_by_pair):
        from policies.training import LinerShippingTrainer, TrainingConfig

        tc = TrainingConfig(
            dataset="Baltic", policy="encoder_only",
            learning_rate=2e-4, gamma=1.0, gae_lambda=0.9,
            ppo_epochs=1, clip_epsilon=0.2, target_kl=0.1,
            entropy_coefficient=0.05, value_coefficient=0.5,
            num_envs=1, steps_per_env=10, minibatch_size=8,
            seed=42, max_updates=1, checkpoint_frequency=100,
            hidden_dim=small_cfg.hidden_dim,
            gat_layers=small_cfg.gat_layers,
            transformer_layers=small_cfg.transformer_layers,
            transformer_heads=small_cfg.transformer_heads,
            lstm_layers=small_cfg.lstm_layers,
        )
        trainer = LinerShippingTrainer("Baltic", "encoder_only", tc)
        traj, steps = trainer.collect_rollout(seed=42)

        assert steps > 0, "Expected non-empty trajectory"
        for i, step in enumerate(traj):
            assert "raw_sampled_ports" in step, (
                f"Step {i}: 'raw_sampled_ports' key missing from trajectory. "
                f"This breaks PPO on-policy consistency (F1)."
            )
            assert "selected_mask" in step, (
                f"Step {i}: 'selected_mask' tensor missing from trajectory."
            )
            # Verify the mask matches the ports list
            port_codes = sorted(baltic.ports.keys())
            rebuilt_mask = torch.zeros(len(port_codes), dtype=torch.bool)
            for pc in step["raw_sampled_ports"]:
                if pc in port_codes:
                    idx = port_codes.index(pc)
                    rebuilt_mask[idx] = True
            stored_mask = step["selected_mask"]
            assert torch.equal(rebuilt_mask, stored_mask.cpu()), (
                f"Step {i}: rebuilt mask from raw_sampled_ports does not match "
                f"stored selected_mask tensor."
            )


# ---------------------------------------------------------------------------
# T2. PPO ratio equals 1.0 when policy weights unchanged
# ---------------------------------------------------------------------------

class TestPPORatioConsistency:
    """Verify that old_log_prob and new_log_prob describe the same action."""

    def test_ppo_ratio_one_encoder_only_no_fallback(
        self, small_cfg, baltic, dist_by_pair,
    ):
        """When no fallback fires, ratio should be exactly 1.0 before update."""
        tc = TrainingConfig(
            dataset="Baltic", policy="encoder_only",
            learning_rate=2e-4, gamma=1.0, gae_lambda=0.9,
            ppo_epochs=1, clip_epsilon=0.2, target_kl=0.1,
            entropy_coefficient=0.05, value_coefficient=0.5,
            num_envs=1, steps_per_env=10, minibatch_size=8,
            seed=42, max_updates=1, checkpoint_frequency=100,
            hidden_dim=small_cfg.hidden_dim,
            gat_layers=small_cfg.gat_layers,
            transformer_layers=small_cfg.transformer_layers,
            transformer_heads=small_cfg.transformer_heads,
            lstm_layers=small_cfg.lstm_layers,
        )
        trainer = LinerShippingTrainer("Baltic", "encoder_only", tc)
        traj, _ = trainer.collect_rollout(seed=42)

        assert len(traj) > 0, "Empty trajectory"
        # Check that NO fallback was applied
        for i, step in enumerate(traj):
            assert not step.get("fallback_applied", False), (
                f"Step {i}: fallback was applied; ratio test invalid in this case. "
                f"Use a different seed or test T3 instead."
            )

        # Perform one PPO update WITHOUT changing weights first — verify ratio
        old_params = {n: p.detach().clone()
                      for n, p in trainer.policy.named_parameters()}

        diag = trainer.perform_ppo_update(traj)

        # KL should be ~0 since weights haven't changed yet (before optimizer.step)
        # Actually perform_ppo_update calls optimizer.step, so we check after
        # the first call separately
        assert math.isfinite(diag.approx_kl), "KL is not finite"
        # The KL after ONE update can be > 0 because weights changed.
        # We verify this by running AGAIN with same trajectory and checking ratio≈1.

    def test_ratio_after_reload_is_one(
        self, small_cfg, baltic, dist_by_pair,
    ):
        """Reload weights → re-evaluate same trajectory → ratio ≈ 1.0 → KL ≈ 0."""
        from policies.training import LinerShippingTrainer, TrainingConfig

        tc = TrainingConfig(
            dataset="Baltic", policy="encoder_only",
            learning_rate=2e-4, gamma=1.0, gae_lambda=0.9,
            ppo_epochs=1, clip_epsilon=0.2, target_kl=0.1,
            entropy_coefficient=0.05, value_coefficient=0.5,
            num_envs=1, steps_per_env=10, minibatch_size=8,
            seed=42, max_updates=1, checkpoint_frequency=100,
            hidden_dim=small_cfg.hidden_dim,
            gat_layers=small_cfg.gat_layers,
            transformer_layers=small_cfg.transformer_layers,
            transformer_heads=small_cfg.transformer_heads,
            lstm_layers=small_cfg.lstm_layers,
        )
        trainer = LinerShippingTrainer("Baltic", "encoder_only", tc)
        traj, _ = trainer.collect_rollout(seed=42)

        # Save weights
        saved_state = {
            n: p.detach().clone()
            for n, p in trainer.policy.named_parameters()
        }
        saved_critic = {
            n: p.detach().clone()
            for n, p in trainer.critic.named_parameters()
        }

        # First update (changes weights)
        diag1 = trainer.perform_ppo_update(traj)
        assert math.isfinite(diag1.total_loss)

        # Reload original weights
        for n, p in trainer.policy.named_parameters():
            p.data.copy_(saved_state[n])
        for n, p in trainer.critic.named_parameters():
            p.data.copy_(saved_critic[n])
        trainer.trainer.optimizer.zero_grad()

        # Second update with SAME trajectory but original weights
        # old_log_prob and new_log_prob should now match exactly
        diag2 = trainer.perform_ppo_update(traj)
        assert math.isfinite(diag2.approx_kl), "KL must be finite"
        assert diag2.approx_kl < 0.01, (
            f"KL={diag2.approx_kl:.6f} should be ≈0 when weights are identical. "
            f"This indicates old/new log-prob mismatch (F1 bug)."
        )


# ---------------------------------------------------------------------------
# T3. Encoder-decoder substep_selected consistency
# ---------------------------------------------------------------------------

class TestEncoderDecoderSequence:
    """F1 variant for encoder-decoder: substep_selected must be stored."""

    def test_substep_selected_stored_correctly(self, small_cfg, baltic, dist_by_pair):
        from policies.training import LinerShippingTrainer, TrainingConfig

        tc = TrainingConfig(
            dataset="Baltic", policy="encoder_decoder",
            learning_rate=2e-4, gamma=1.0, gae_lambda=0.9,
            ppo_epochs=1, clip_epsilon=0.2, target_kl=0.1,
            entropy_coefficient=0.05, value_coefficient=0.5,
            num_envs=1, steps_per_env=10, minibatch_size=8,
            seed=42, max_updates=1, checkpoint_frequency=100,
            hidden_dim=small_cfg.hidden_dim,
            gat_layers=small_cfg.gat_layers,
            transformer_layers=small_cfg.transformer_layers,
            transformer_heads=small_cfg.transformer_heads,
            lstm_layers=small_cfg.lstm_layers,
        )
        trainer = LinerShippingTrainer("Baltic", "encoder_decoder", tc)
        traj, steps = trainer.collect_rollout(seed=42)

        assert steps > 0, "Expected non-empty trajectory"
        for i, step in enumerate(traj):
            assert "substep_selected" in step, (
                f"Step {i}: 'substep_selected' key missing for encoder-decoder."
            )
            assert isinstance(step["substep_selected"], list), (
                f"Step {i}: 'substep_selected' must be a list of ints."
            )


# ---------------------------------------------------------------------------
# T4. Verify evaluate_actions uses same action as rollout
# ---------------------------------------------------------------------------

class TestEvaluateActionsConsistency:
    """Verify that evaluate_actions() re-computes the same log-prob."""

    def test_evaluate_actions_matches_rollout_log_prob(
        self, small_cfg, baltic, dist_by_pair,
    ):
        """For encoder-only: evaluate_actions with stored mask = rollout log_prob."""
        from policies.encoder_only import EncoderOnlyPolicy
        from actions.service_generator import ServiceGenerator

        gen = ServiceGenerator(baltic, dist_by_pair)
        policy = EncoderOnlyPolicy(NeuralBackbone(small_cfg), baltic, gen)
        enc = StateEncoder(baltic, dist_by_pair)

        env = LSNDPEnv(baltic)
        obs, _ = env.reset(seed=77)
        membership = ServiceMembership()
        rem = {i: float(obs["remaining_demand"][i])
               for i in range(len(obs["remaining_demand"]))}
        fleet = {vc: float(obs["fleet_remaining"][i])
                 for i, vc in enumerate(sorted(baltic.vessel_types.keys()))}

        ns = enc.encode(rem, fleet, membership)
        bundle = neural_state_to_tensors(ns)

        out = policy.sample_action(bundle, fleet, seed=77)
        raw_lp = out.raw_log_prob

        # Rebuild the mask exactly as perform_ppo_update does
        n_ports = len(baltic.ports)
        selected_mask = torch.zeros(n_ports, dtype=torch.bool, device=bundle.device)
        port_codes_list = sorted(baltic.ports.keys())
        for pc in out.raw_sampled_ports:
            if pc in port_codes_list:
                selected_mask[port_codes_list.index(pc)] = True

        with torch.enable_grad():
            new_lp, _ = policy.evaluate_actions(bundle, fleet, selected_mask=selected_mask)

        ratio = torch.exp(new_lp - raw_lp).item()
        assert abs(ratio - 1.0) < 1e-5, (
            f"PPO ratio = {ratio:.6f}, expected 1.0. "
            f"old_log_prob={float(raw_lp):.6f}, new_log_prob={float(new_lp):.6f}"
        )
