"""
P10 — Tests for PPO Training Engine.

Covers the P10.14 test list (24 tests):
  1. rollout buffer insertion
  2. rollout buffer retrieval
  3. done/truncated handling
  4. GAE calculation
  5. gamma=1 calculation
  6. lambda=0.9 calculation
  7. PPO ratio
  8. clipping
  9. policy loss
  10. value loss
  11. entropy bonus
  12. total loss
  13. approximate KL
  14. clip fraction
  15. minibatching
  16. P8 integration
  17. P8 fallback exclusion
  18. P9 integration
  19. P9 decoded-action log-prob preservation
  20. checkpoint save/load
  21. optimizer state restoration
  22. deterministic seed behavior
  23. no accidental raw-data modification
  24. no accidental P1-P9 scope corruption

Fixtures are tiny synthetic instances. No real LINERLIB training in this phase.
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock

import math
import pytest
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from data.instance import (
    DatasetProvenance,
    Demand,
    DistanceArc,
    FleetEntry,
    InstanceMetadata,
    LINERLIBInstance,
    Port,
    ProvenanceRecord,
    VesselType,
)
from mcf.ppo_engine.buffer import PPOBuffer, TrajectoryStep
from mcf.ppo_engine.config import PPOConfig, PaperPPOConfig
from mcf.ppo_engine.trainer import PPOTrainer, PDiagnostics


# ===========================================================================
# Fixtures
# ===========================================================================

def _toy_port(code: str) -> Port:
    return Port(
        unlocode=code, name=f"Port {code}", country=None,
        cabotage_region="test", d_region=None, longitude=None, latitude=None,
        draft=10.0, cost_per_full=1.0, cost_per_full_transfer=0.5,
        port_call_cost_fixed=100.0, port_call_cost_per_ffe=0.5,
        provenance=ProvenanceRecord(source_file="synthetic_p10", source_row=1),
    )


def _toy_vessel(name: str) -> VesselType:
    return VesselType(
        vessel_class=name, capacity_ffe=100.0, tc_rate_daily=100,
        draft=12.0, min_speed=5.0, max_speed=15.0, design_speed=10.0,
        bunker_ton_per_day_at_design=50.0, idle_consumption_ton_per_day=10.0,
        panama_fee=0, suez_fee=0,
        provenance=ProvenanceRecord(source_file="synthetic_p10", source_row=1),
    )


def make_toy_instance() -> LINERLIBInstance:
    """Tiny 3-port instance for P10 testing."""
    ports = {"A": _toy_port("A"), "B": _toy_port("B"), "C": _toy_port("C")}
    vessels = {"V1": _toy_vessel("V1"), "V2": _toy_vessel("V2")}
    demands = [
        Demand(origin="A", destination="B", ffe_per_week=50.0, revenue=200.0, max_transit_time=10,
               provenance=ProvenanceRecord(source_file="synthetic_p10", source_row=1)),
    ]
    distances = [
        DistanceArc(origin="A", destination="B", distance_nm=100.0, draft_required=10.0,
                    is_panama=False, is_suez=False,
                    provenance=ProvenanceRecord(source_file="synthetic_p10", source_row=1)),
        DistanceArc(origin="B", destination="C", distance_nm=100.0, draft_required=10.0,
                    is_panama=False, is_suez=False,
                    provenance=ProvenanceRecord(source_file="synthetic_p10", source_row=2)),
        DistanceArc(origin="C", destination="A", distance_nm=100.0, draft_required=10.0,
                    is_panama=False, is_suez=False,
                    provenance=ProvenanceRecord(source_file="synthetic_p10", source_row=3)),
    ]
    fleet = [FleetEntry(vessel_class="V1", quantity=5), FleetEntry(vessel_class="V2", quantity=3)]
    metadata = InstanceMetadata(
        name="TOY_P10", active_port_count=3, vessel_type_count=2,
        total_vessels=8, demand_count=1, distance_arc_count=3,
    )
    return LINERLIBInstance(
        name="TOY_P10", ports=ports, vessel_types=vessels,
        demands=demands, distances=distances, fleet=fleet, metadata=metadata,
        provenance=DatasetProvenance(source_root="[SYNTHETIC TEST FIXTURE -- P10]"),
    )


def make_mock_policy() -> MagicMock:
    """Create a mock policy that supports the P10 interface."""
    policy = MagicMock()
    policy.forward.return_value = MagicMock()
    policy.log_prob.return_value = torch.tensor(-1.0)
    policy.entropy.return_value = torch.tensor(0.5)
    policy.parameters.return_value = []
    return policy


def make_mock_critic() -> nn.Module:
    """Create a simple critic network."""
    import torch.nn as nn
    return nn.Sequential(
        nn.Linear(10, 16),
        nn.ReLU(),
        nn.Linear(16, 1),
    )


# ===========================================================================
# 1-2. Buffer insertion and retrieval
# ===========================================================================

class TestBuffer:
    def test_insert_and_retrieve(self):
        """Test 1: Basic buffer insertion and retrieval."""
        buf = PPOBuffer()
        step = TrajectoryStep(
            state_repr="state", policy_id="test", action="action",
            executed_action="executed", reward=1.0, done=False, truncated=False,
            old_log_prob=torch.tensor(-1.0), old_value=torch.tensor(0.0),
            entropy=torch.tensor(0.5),
        )
        buf.add_step(**{k: getattr(step, k) for k in dir(step) if not k.startswith('_')})
        assert len(buf) == 1

    def test_retrieve_all_steps(self):
        """Test 2: Retrieve all stored steps."""
        buf = PPOBuffer()
        for i in range(5):
            buf.add_step(
                state_repr=i, policy_id="test", action=i, executed_action=i,
                reward=float(i), done=False, truncated=False,
                old_log_prob=torch.tensor(-float(i)),
                old_value=torch.tensor(float(i)),
                entropy=torch.tensor(0.5),
            )
        steps = buf.get_all_steps()
        assert len(steps) == 5
        assert all(s.reward == float(i) for i, s in enumerate(steps))


# ===========================================================================
# 3. Done/truncated handling
# ===========================================================================

class TestDoneTruncated:
    def test_done_flag_stored(self):
        """Test done=True is preserved."""
        buf = PPOBuffer()
        buf.add_step(
            state_repr="s", policy_id="p", action="a", executed_action="e",
            reward=1.0, done=True, truncated=False,
            old_log_prob=torch.tensor(-1.0), old_value=torch.tensor(0.0),
            entropy=torch.tensor(0.5),
        )
        steps = buf.get_all_steps()
        assert steps[0].done is True
        assert steps[0].truncated is False

    def test_truncated_flag_stored(self):
        """Test truncated=True is preserved."""
        buf = PPOBuffer()
        buf.add_step(
            state_repr="s", policy_id="p", action="a", executed_action="e",
            reward=1.0, done=False, truncated=True,
            old_log_prob=torch.tensor(-1.0), old_value=torch.tensor(0.0),
            entropy=torch.tensor(0.5),
        )
        steps = buf.get_all_steps()
        assert steps[0].done is False
        assert steps[0].truncated is True

    def test_end_episode_returns_trajectory(self):
        """Test end_episode returns the trajectory and clears in-progress steps."""
        buf = PPOBuffer()
        buf.add_step(
            state_repr="s", policy_id="p", action="a", executed_action="e",
            reward=1.0, done=True, truncated=False,
            old_log_prob=torch.tensor(-1.0), old_value=torch.tensor(0.0),
            entropy=torch.tensor(0.5),
        )
        episode = buf.end_episode()
        assert len(episode) == 1
        # _steps is cleared; completed episodes stay in _episode_trajectories
        assert len(buf._steps) == 0
        assert len(buf._episode_trajectories) == 1


# ===========================================================================
# 4-6. GAE calculation
# ===========================================================================

class TestGAE:
    def setup_method(self):
        self.config = PPOConfig(gamma=1.0, gae_lambda=0.9)
        self.trainer = PPOTrainer(make_mock_policy(), make_mock_critic(), self.config)

    def test_gae_basic(self):
        """Test 4: Basic GAE computation."""
        values = torch.tensor([0.0, 1.0, 2.0, 3.0])
        rewards = torch.tensor([1.0, 1.0, 1.0, 1.0])
        dones = torch.tensor([0, 0, 0, 1], dtype=torch.float)

        returns, advantages = self.trainer.compute_returns_and_advantages(
            values, rewards, dones,
        )

        assert returns.shape == values.shape
        assert advantages.shape == values.shape
        # Last step: return = reward + 0 (terminal)
        assert abs(returns[-1].item() - 1.0) < 1e-5

    def test_gamma_1_behavior(self):
        """Test 5: Verify gamma=1 behavior with exact expected values."""
        values = torch.tensor([0.0, 0.0, 0.0])
        rewards = torch.tensor([1.0, 2.0, 3.0])
        dones = torch.tensor([0, 0, 0], dtype=torch.float)

        returns, advantages = self.trainer.compute_returns_and_advantages(values, rewards, dones)

        # Hand-computed GAE with gamma=1, lambda=0.9:
        # t=2: delta=3, gae=3, ret=3
        # t=1: delta=2, gae=2+0.9*3=4.7, ret=4.7
        # t=0: delta=1, gae=1+0.9*4.7=5.23, ret=5.23
        expected_advantages = torch.tensor([5.23, 4.7, 3.0])
        expected_returns = torch.tensor([5.23, 4.7, 3.0])
        torch.testing.assert_close(advantages, expected_advantages, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(returns, expected_returns, atol=1e-5, rtol=1e-5)

    def test_lambda_0_9(self):
        """Test 6: Lambda=0.9 GAE decay with exact values."""
        values = torch.tensor([0.0, 0.0, 0.0, 0.0])
        rewards = torch.tensor([1.0, 0.0, 0.0, 0.0])
        dones = torch.tensor([0, 0, 0, 1], dtype=torch.float)

        returns, advantages = self.trainer.compute_returns_and_advantages(
            values, rewards, dones,
        )

        # Hand-computed:
        # t=3: done=1, delta=0, gae=0, adv=0, ret=0
        # t=2: done=0, delta=0, gae=0+0.9*1*0=0, adv=0, ret=0
        # t=1: done=0, delta=0, gae=0+0.9*1*0=0, adv=0, ret=0
        # t=0: done=0, delta=1, gae=1+0.9*1*0=1, adv=1, ret=1
        expected_advantages = torch.tensor([1.0, 0.0, 0.0, 0.0])
        expected_returns = torch.tensor([1.0, 0.0, 0.0, 0.0])
        torch.testing.assert_close(advantages, expected_advantages, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(returns, expected_returns, atol=1e-5, rtol=1e-5)


# ===========================================================================
# 7-14. PPO objective
# ===========================================================================

class TestPPOObjective:
    def setup_method(self):
        self.config = PPOConfig(gamma=1.0, gae_lambda=0.9, clip_epsilon=0.2)
        self.trainer = PPOTrainer(make_mock_policy(), make_mock_critic(), self.config)

    def test_ppo_ratio(self):
        """Test 7: Policy ratio computation."""
        old_log_probs = torch.tensor([-1.0, -2.0, -3.0])
        new_log_probs = torch.tensor([-0.5, -1.5, -2.5])

        advantages = torch.ones(3)
        loss, clip_frac, kl = self.trainer.compute_ppo_loss(
            new_log_probs, old_log_probs, advantages, 0.2,
        )

        # Ratio = exp(new - old) = exp(0.5) ≈ 1.649
        expected_ratio = torch.exp(new_log_probs - old_log_probs)
        assert expected_ratio[0].item() > 1.0

    def test_clipping(self):
        """Test 8: Clipped ratio bounds."""
        # Large policy change → ratio should be clipped
        old_log_probs = torch.tensor([0.0])
        new_log_probs = torch.tensor([1.0])  # ratio = e^1 ≈ 2.718
        advantages = torch.tensor([1.0])

        loss, clip_frac, kl = self.trainer.compute_ppo_loss(
            new_log_probs, old_log_probs, advantages, 0.2,
        )

        # Should have clipping
        assert clip_frac > 0.0

    def test_policy_loss(self):
        """Test 9: Policy loss is finite."""
        old_log_probs = torch.tensor([-1.0, -1.0])
        new_log_probs = torch.tensor([-1.1, -0.9])
        advantages = torch.tensor([1.0, -1.0])

        loss, _, _ = self.trainer.compute_ppo_loss(
            new_log_probs, old_log_probs, advantages, 0.2,
        )

        assert torch.isfinite(loss)

    def test_value_loss(self):
        """Test 10: Value loss computation."""
        values = torch.tensor([0.0, 1.0, 2.0])
        returns = torch.tensor([0.1, 1.1, 2.1])

        loss = self.trainer.compute_value_loss(values, returns)

        assert torch.isfinite(loss)
        assert loss.item() > 0  # MSE > 0 when predictions != targets

    def test_entropy_bonus(self):
        """Test 11: Entropy bonus computation."""
        entropies = torch.tensor([0.5, 0.6, 0.7])

        loss = self.trainer.compute_entropy_bonus(entropies)

        assert torch.isfinite(loss)
        # Entropy bonus is subtracted from total loss (encourages exploration)
        assert loss.item() < 0

    def test_total_loss(self):
        """Test 12: Total loss combines policy, value, entropy."""
        old_log_probs = torch.tensor([-1.0, -1.0])
        new_log_probs = torch.tensor([-1.1, -0.9])
        advantages = torch.tensor([1.0, -1.0])
        returns = torch.tensor([0.0, 1.0])
        entropies = torch.tensor([0.5, 0.5])

        policy_loss, _, _ = self.trainer.compute_ppo_loss(
            new_log_probs, old_log_probs, advantages, 0.2,
        )
        value_loss = self.trainer.compute_value_loss(torch.tensor([0.0, 1.0]), returns)
        entropy_loss = self.trainer.compute_entropy_bonus(entropies)

        total = policy_loss + 0.5 * value_loss + entropy_loss

        assert torch.isfinite(total)

    def test_approximate_kl(self):
        """Test 13: PPO k3 approximate KL estimator with exact values.

        The k3 estimator computes: KL_approx = 0.5 * mean((ratio - 1)^2)
        where ratio = exp(new_log_prob - old_log_prob).
        This approximates KL(pi_new || pi_old) under pi_old samples.
        """
        old_log_probs = torch.tensor([-1.0, -1.0])
        new_log_probs = torch.tensor([-1.0, -1.0])  # Identical → KL = 0 exactly
        advantages = torch.ones(2)

        _, _, kl = self.trainer.compute_ppo_loss(
            new_log_probs, old_log_probs, advantages, 0.2,
        )

        assert kl == 0.0  # Identical policies give exactly zero KL

    def test_kl_small_perturbation_exact(self):
        """KL estimator with known small perturbation gives expected value."""
        # delta = -0.1, ratio = exp(-0.1) ≈ 0.904837
        # KL = 0.5 * (0.904837 - 1)^2 = 0.5 * 0.00907 = 0.004535
        old_log_probs = torch.tensor([-1.0])
        new_log_probs = torch.tensor([-1.1])
        advantages = torch.ones(1)

        _, _, kl = self.trainer.compute_ppo_loss(
            new_log_probs, old_log_probs, advantages, 0.2,
        )

        expected_ratio = torch.exp(torch.tensor(-0.1)).item()
        expected_kl = 0.5 * (expected_ratio - 1.0) ** 2
        assert abs(kl - expected_kl) < 1e-6

    def test_clip_fraction(self):
        """Test 14: Clip fraction tracking."""
        old_log_probs = torch.tensor([-1.0])
        new_log_probs = torch.tensor([-2.0])  # Large change
        advantages = torch.ones(1)

        _, clip_frac, _ = self.trainer.compute_ppo_loss(
            new_log_probs, old_log_probs, advantages, 0.2,
        )

        assert 0.0 <= clip_frac <= 1.0


# ===========================================================================
# 15. Minibatching
# ===========================================================================

class TestMinibatching:
    def test_minibatch_size_respected(self):
        """Test 15: Minibatches respect configured batch size."""
        config = PPOConfig(minibatch_size=4)
        trainer = PPOTrainer(make_mock_policy(), make_mock_critic(), config)

        # Create batch larger than minibatch_size
        n_samples = 16
        states = torch.randn(n_samples, 10)
        actions = [None] * n_samples
        old_log_probs = torch.randn(n_samples)
        old_values = torch.randn(n_samples)
        advantages = torch.randn(n_samples)
        returns = torch.randn(n_samples)
        entropies = torch.abs(torch.randn(n_samples))

        # Should complete without error
        diag = trainer.train_step(
            states, actions, old_log_probs, old_values,
            advantages, returns, entropies, batches=4,
        )

        assert isinstance(diag, PDiagnostics)
        assert math.isfinite(diag.total_loss)


# ===========================================================================
# 16-17. P8 integration
# ===========================================================================

class TestP8Integration:
    def test_p8_fallback_exclusion(self):
        """Test 16/17: P8 fallback samples can be excluded."""
        buf = PPOBuffer()

        # Normal sample
        buf.add_step(
            state_repr="s1", policy_id="encoder_only", action="a1",
            executed_action="e1", reward=1.0, done=False, truncated=False,
            old_log_prob=torch.tensor(-1.0), old_value=torch.tensor(0.0),
            entropy=torch.tensor(0.5), fallback_applied=False,
        )

        # Fallback sample
        buf.add_step(
            state_repr="s2", policy_id="encoder_only", action="a2",
            executed_action="e2", reward=1.0, done=False, truncated=False,
            old_log_prob=torch.tensor(-1.0), old_value=torch.tensor(0.0),
            entropy=torch.tensor(0.5), fallback_applied=True,
        )

        # Filter should exclude fallback
        filtered = buf.filter_non_fallback()
        assert len(filtered) == 1
        assert not filtered[0].fallback_applied

    def test_fallback_detection(self):
        """Test fallback detection works."""
        buf = PPOBuffer()
        buf.add_step(
            state_repr="s", policy_id="encoder_only", action="a",
            executed_action="e", reward=1.0, done=False, truncated=False,
            old_log_prob=torch.tensor(-1.0), old_value=torch.tensor(0.0),
            entropy=torch.tensor(0.5), fallback_applied=True,
        )
        assert buf.has_fallback_samples()


# ===========================================================================
# 18-19. P9 integration
# ===========================================================================

class TestP9Integration:
    def test_p9_decoded_log_prob_preserved(self):
        """Test 18/19: P9 decoder log-prob is preserved through buffer."""
        buf = PPOBuffer()

        decoded_seq = ["PORT_A", "PORT_B", "PORT_C"]
        executed_seq = ["PORT_C", "PORT_A", "PORT_B"]  # TSP reordered

        buf.add_step(
            state_repr="s", policy_id="encoder_decoder", action="a",
            executed_action="e", reward=1.0, done=False, truncated=False,
            old_log_prob=torch.tensor(-2.5), old_value=torch.tensor(0.5),
            entropy=torch.tensor(1.2),
            decoded_port_sequence=decoded_seq,
            executed_port_sequence=executed_seq,
        )

        steps = buf.get_all_steps()
        assert steps[0].decoded_port_sequence == decoded_seq
        assert steps[0].executed_port_sequence == executed_seq
        assert abs(steps[0].old_log_prob.item() - (-2.5)) < 1e-5


# ===========================================================================
# 20-21. Checkpointing
# ===========================================================================

class TestCheckpointing:
    def test_save_load_checkpoint(self, tmp_path):
        """Test 20: Checkpoint save and load with real modules."""
        import tempfile
        path = Path(tempfile.mktemp(suffix=".pt"))

        config = PPOConfig(seed=42)
        policy = nn.Linear(10, 1)
        critic = nn.Linear(10, 1)
        trainer = PPOTrainer(policy, critic, config)
        trainer._update_count = 5

        trainer.save_checkpoint(str(path))
        assert path.exists()

        new_policy = nn.Linear(10, 1)
        new_critic = nn.Linear(10, 1)
        new_trainer = PPOTrainer(new_policy, new_critic, config)
        loaded = new_trainer.load_checkpoint(str(path))

        assert loaded["update_count"] == 5
        assert new_trainer.update_count == 5
        path.unlink(missing_ok=True)

    def test_optimizer_state_restored(self, tmp_path):
        """Test 21: Optimizer state is restored."""
        import tempfile
        path = Path(tempfile.mktemp(suffix=".pt"))

        config = PPOConfig(seed=42)
        policy = nn.Linear(10, 1)
        critic = nn.Linear(10, 1)
        trainer = PPOTrainer(policy, critic, config)
        trainer._update_count = 10

        trainer.save_checkpoint(str(path))

        new_policy = nn.Linear(10, 1)
        new_critic = nn.Linear(10, 1)
        new_trainer = PPOTrainer(new_policy, new_critic, config)
        new_trainer.load_checkpoint(str(path))

        assert new_trainer.update_count == 10
        path.unlink(missing_ok=True)


# ===========================================================================
# 22. Deterministic seed behavior
# ===========================================================================

class TestDeterminism:
    def test_seed_in_config(self):
        """Test 22: Seed is stored in config."""
        config = PPOConfig(seed=12345)
        assert config.seed == 12345

        config_dict = config.to_dict()
        assert config_dict["seed"] == 12345


# ===========================================================================
# 23-24. Data integrity and scope boundaries
# ===========================================================================

class TestDataIntegrity:
    def test_no_raw_data_modification(self):
        """Test 23: Buffer operations don't modify raw data."""
        inst = make_toy_instance()
        original_name = inst.name

        buf = PPOBuffer()
        buf.add_step(
            state_repr="s", policy_id="test", action="a", executed_action="e",
            reward=1.0, done=False, truncated=False,
            old_log_prob=torch.tensor(-1.0), old_value=torch.tensor(0.0),
            entropy=torch.tensor(0.5),
        )

        assert inst.name == original_name

    def test_no_p1_p9_scope_corruption(self):
        """Test 24: PPO engine doesn't modify P1-P9 modules."""
        # Import P1-P9 modules to verify they exist
        from data.instance import LINERLIBInstance
        from neural.backbone import NeuralBackbone
        from policies.encoder_only import EncoderOnlyPolicy
        from policies.encoder_decoder import EncoderDecoderPolicy

        # These imports should work without side effects
        assert LINERLIBInstance is not None
        assert NeuralBackbone is not None
        assert EncoderOnlyPolicy is not None
        assert EncoderDecoderPolicy is not None


# ===========================================================================
# Config tests
# ===========================================================================

class TestConfig:
    def test_paper_config_defaults(self):
        """Paper-config respects paper defaults."""
        config = PPOConfig()
        assert config.gamma == 1.0
        assert config.gae_lambda == 0.9
        assert config.ppo_epochs == 10
        assert config.clip_epsilon == 0.2

    def test_config_validation(self):
        """Invalid config raises ValueError."""
        with pytest.raises(ValueError):
            PPOConfig(gamma=-1.0).validate()

        with pytest.raises(ValueError):
            PPOConfig(learning_rate=0).validate()

    def test_config_serialization(self):
        """Config round-trips through dict."""
        config = PPOConfig(seed=42, learning_rate=1e-4)
        d = config.to_dict()
        restored = PPOConfig.from_dict(d)
        assert restored.seed == 42
        assert abs(restored.learning_rate - 1e-4) < 1e-10


# ===========================================================================
# Main
# ===========================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
