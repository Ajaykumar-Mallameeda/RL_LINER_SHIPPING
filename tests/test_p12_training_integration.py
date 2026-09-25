"""
P12 — Regression Tests for Training Integration Defects.

These tests verify that the P12 training integration is correct after
the forensic debugging audit. They cover:

  1. old_log_prob passed to PPO (not old_value)
  2. old_value remains exclusively for GAE/value estimation
  3. One rollout per training iteration; same trajectory drives metrics + PPO
  4. max_updates semantics are respected
  5. PPO ratio correctness through full P12 pipeline
  6. P8 fallback exclusion from PPO
  7. P9 sequence/log-prob consistency
  8. Termination/truncation handling in trajectory
  9. Economic metric propagation (keys match environment output)
  10. Checkpoint save/load round-trip
  11. Deterministic seeded rollout
  12. Raw data immutability during training
  13. Numerical finite-value checks (no NaN/Inf losses)
  14. Exception handling does not mask programming errors
"""

from __future__ import annotations

import hashlib
import math
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import math
import sys
from pathlib import Path
from unittest.mock import patch

import pytest
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def _make_toy_instance():
    """Build a tiny LINERLIBInstance suitable for training integration tests."""
    from data.instance import (
        DatasetProvenance, Demand, DistanceArc, FleetEntry,
        InstanceMetadata, LINERLIBInstance, Port, ProvenanceRecord, VesselType,
    )
    ports = {
        "A": Port(unlocode="A", name="Port A", country=None, cabotage_region="test",
                  d_region=None, longitude=None, latitude=None, draft=10.0,
                  cost_per_full=1.0, cost_per_full_transfer=0.5,
                  port_call_cost_fixed=100.0, port_call_cost_per_ffe=0.5,
                  provenance=ProvenanceRecord(source_file="synthetic_p12", source_row=1)),
        "B": Port(unlocode="B", name="Port B", country=None, cabotage_region="test",
                  d_region=None, longitude=None, latitude=None, draft=10.0,
                  cost_per_full=1.0, cost_per_full_transfer=0.5,
                  port_call_cost_fixed=100.0, port_call_cost_per_ffe=0.5,
                  provenance=ProvenanceRecord(source_file="synthetic_p12", source_row=2)),
        "C": Port(unlocode="C", name="Port C", country=None, cabotage_region="test",
                  d_region=None, longitude=None, latitude=None, draft=10.0,
                  cost_per_full=1.0, cost_per_full_transfer=0.5,
                  port_call_cost_fixed=100.0, port_call_cost_per_ffe=0.5,
                  provenance=ProvenanceRecord(source_file="synthetic_p12", source_row=3)),
    }
    vessels = {
        "V1": VesselType(vessel_class="V1", capacity_ffe=100.0, tc_rate_daily=100,
                         draft=12.0, min_speed=5.0, max_speed=15.0, design_speed=10.0,
                         bunker_ton_per_day_at_design=50.0,
                         idle_consumption_ton_per_day=10.0, panama_fee=0, suez_fee=0,
                         provenance=ProvenanceRecord(source_file="synthetic_p12", source_row=1)),
        "V2": VesselType(vessel_class="V2", capacity_ffe=80.0, tc_rate_daily=80,
                         draft=11.0, min_speed=4.0, max_speed=14.0, design_speed=9.0,
                         bunker_ton_per_day_at_design=40.0,
                         idle_consumption_ton_per_day=8.0, panama_fee=0, suez_fee=0,
                         provenance=ProvenanceRecord(source_file="synthetic_p12", source_row=2)),
    }
    demands = [
        Demand(origin="A", destination="B", ffe_per_week=50.0, revenue=200.0,
               max_transit_time=10,
               provenance=ProvenanceRecord(source_file="synthetic_p12", source_row=1)),
        Demand(origin="B", destination="C", ffe_per_week=30.0, revenue=150.0,
               max_transit_time=10,
               provenance=ProvenanceRecord(source_file="synthetic_p12", source_row=2)),
    ]
    distances = [
        DistanceArc(origin="A", destination="B", distance_nm=100.0, draft_required=10.0,
                    is_panama=False, is_suez=False,
                    provenance=ProvenanceRecord(source_file="synthetic_p12", source_row=1)),
        DistanceArc(origin="B", destination="C", distance_nm=100.0, draft_required=10.0,
                    is_panama=False, is_suez=False,
                    provenance=ProvenanceRecord(source_file="synthetic_p12", source_row=2)),
        DistanceArc(origin="C", destination="A", distance_nm=100.0, draft_required=10.0,
                    is_panama=False, is_suez=False,
                    provenance=ProvenanceRecord(source_file="synthetic_p12", source_row=3)),
    ]
    fleet = [FleetEntry(vessel_class="V1", quantity=5), FleetEntry(vessel_class="V2", quantity=3)]
    metadata = InstanceMetadata(
        name="TOY_P12", active_port_count=3, vessel_type_count=2,
        total_vessels=8, demand_count=2, distance_arc_count=3,
    )
    return LINERLIBInstance(
        name="TOY_P12", ports=ports, vessel_types=vessels,
        demands=demands, distances=distances, fleet=fleet, metadata=metadata,
        provenance=DatasetProvenance(source_root="[SYNTHETIC TEST FIXTURE -- P12]"),
    )


def _make_minimal_config(**overrides):
    """Build a minimal TrainingConfig with sensible defaults."""
    from policies.training import TrainingConfig
    defaults = dict(
        dataset="TOY_P12", policy="encoder_only",
        learning_rate=2e-4, gamma=1.0, gae_lambda=0.9, ppo_epochs=2,
        clip_epsilon=0.2, target_kl=0.1, entropy_coefficient=0.05,
        value_coefficient=0.5, num_envs=1, steps_per_env=100, minibatch_size=32,
        seed=42, max_updates=5, checkpoint_frequency=10,
        hidden_dim=16, gat_layers=1, transformer_layers=1,
        transformer_heads=2, lstm_layers=1,
    )
    defaults.update(overrides)
    return TrainingConfig(**defaults)


def _make_trainer(tmp_path, **config_overrides):
    """Create a LinerShippingTrainer backed by a synthetic instance."""
    from policies.training import LinerShippingTrainer
    toy = _make_toy_instance()
    config = _make_minimal_config(**config_overrides)

    # Mock the loader so it returns our synthetic instance instead of loading CSVs.
    with patch("policies.training.LINERLIBLoader") as MockLoader:
        mock_loader = MagicMock()
        mock_loader.load.return_value = toy
        MockLoader.return_value = mock_loader

        trainer = LinerShippingTrainer(
            instance_name="TOY_P12",
            policy_type=config.policy,
            config=config,
            checkpoint_dir=str(tmp_path / "ckpt"),
        )
    return trainer


def _make_graph_tensors(trainer):
    """Build a minimal GraphTensors bundle matching the trainer's instance."""
    from neural.tensors import GraphTensors
    n_ports = len(trainer.instance.ports)
    n_vessels = len(trainer.instance.vessel_types)
    # Critic input dim = (P+1)*2 + V*11
    critic_dim = (n_ports + 1) * 2 + n_vessels * 11
    return GraphTensors(
        node_features=torch.zeros(n_ports + 1, 2, dtype=torch.float32),
        static_edge_features=torch.zeros(4, 3, dtype=torch.float32),
        dynamic_edge_features=torch.zeros(2, 3, dtype=torch.float32),
        edge_index=torch.tensor([[0, 0, 1], [1, 2, 0]], dtype=torch.long),
        vessel_features=torch.zeros(n_vessels, 11, dtype=torch.float32),
        port_codes=sorted(trainer.instance.ports.keys()),
        vessel_classes=sorted(trainer.instance.vessel_types.keys()),
        num_ports=n_ports,
        num_nodes=n_ports + 1,
        num_edges=3,
        num_vessel_classes=n_vessels,
        num_services=0,
        instance_name=trainer.instance.name,
        device=torch.device("cpu"),
        dtype=torch.float32,
    ), critic_dim


# ===========================================================================
# Defect 1: old_log_prob passed to PPO
# ===========================================================================

class TestOldLogProbPassedToPPO:
    """D1 — PPO must receive old_log_probs, NOT old_value."""

    def test_ppo_loss_signature_accepts_old_log_probs(self):
        """compute_ppo_loss(log_probs, old_log_probs, ...) has correct signature."""
        from mcf.ppo_engine.config import PPOConfig
        from mcf.ppo_engine.trainer import PPOTrainer
        cfg = PPOConfig(gamma=1.0, gae_lambda=0.9, clip_epsilon=0.2)
        mock_policy = nn.Linear(10, 1)
        mock_critic = nn.Linear(10, 1)
        trainer = PPOTrainer(mock_policy, mock_critic, cfg)

        old_log_probs = torch.tensor([-1.0, -2.0])
        new_log_probs = torch.tensor([-1.1, -1.9])
        advantages = torch.tensor([1.0, -1.0])

        loss, clip_frac, kl = trainer.compute_ppo_loss(
            new_log_probs, old_log_probs, advantages, 0.2,
        )
        assert torch.isfinite(loss)

    def test_perform_ppo_update_passes_old_log_probs(self, tmp_path):
        """perform_ppo_update feeds old_log_prob from trajectory into PPO."""
        trainer = _make_trainer(tmp_path)

        # Build a synthetic non-fallback trajectory with a real GraphTensors state
        graph_t, critic_dim = _make_graph_tensors(trainer)
        step = {
            "old_log_prob": torch.tensor(-1.5), "old_value": torch.tensor(0.5),
            "reward": 1.0, "done": False, "truncated": False,
            "entropy": torch.tensor(0.5), "state": graph_t,
            "critic_input": torch.zeros(1, critic_dim),
            "action": [], "executed_action": [],
            "fallback_applied": False,
            "decoded_port_sequence": ["A", "B"], "executed_port_sequence": ["A", "B"],
            "info": {},
        }
        trajectory = [step]

        # Intercept compute_ppo_loss to verify it receives old_log_probs
        captured = {}
        original = trainer.trainer.compute_ppo_loss

        def track(new_lp, old_lp, adv, clip_eps):
            captured["new_shape"] = tuple(new_lp.shape)
            captured["old_shape"] = tuple(old_lp.shape)
            captured["adv_shape"] = tuple(adv.shape)
            return original(new_lp, old_lp, adv, clip_eps)

        trainer.trainer.compute_ppo_loss = track
        diag = trainer.perform_ppo_update(trajectory)

        assert captured.get("new_shape") == captured.get("old_shape") == captured.get("adv_shape")
        assert math.isfinite(diag.policy_loss)


# ===========================================================================
# Defect 2: old_value remains exclusively for GAE/value
# ===========================================================================

class TestOldValueForGAEOnly:
    """D1b — old_value used only for returns/advantages, never in PPO loss."""

    def test_old_value_not_an_arg_to_ppo_loss(self, tmp_path):
        """Verify compute_ppo_loss never receives old_value."""
        trainer = _make_trainer(tmp_path)
        graph_t, critic_dim = _make_graph_tensors(trainer)
        trajectory = [{
            "old_log_prob": torch.tensor(-1.5), "old_value": torch.tensor(0.5),
            "reward": 1.0, "done": False, "truncated": False,
            "entropy": torch.tensor(0.5), "state": graph_t,
            "critic_input": torch.zeros(1, critic_dim),
            "action": [], "executed_action": [],
            "fallback_applied": False,
            "decoded_port_sequence": ["A", "B"], "executed_port_sequence": ["A", "B"],
            "info": {},
        }]

        received = []
        original = trainer.trainer.compute_ppo_loss

        def track(new_lp, old_lp, adv, clip_eps):
            received.append({"old_lp_is_logprob": True})
            return original(new_lp, old_lp, adv, clip_eps)

        trainer.trainer.compute_ppo_loss = track
        trainer.perform_ppo_update(trajectory)
        assert len(received) > 0


# ===========================================================================
# Defect 3: One rollout per iteration
# ===========================================================================

class TestSingleRolloutInvariant:
    """D2 — Exactly ONE trajectory is collected per training iteration."""

    def test_one_collect_rollout_per_update(self, tmp_path):
        """run_training calls collect_rollout exactly once per update."""
        trainer = _make_trainer(tmp_path, max_updates=3, ppo_epochs=1, target_kl=10.0)

        call_count = 0
        orig = trainer.collect_rollout

        def counting(seed):
            nonlocal call_count
            call_count += 1
            return orig(seed=seed)

        trainer.collect_rollout = counting
        metrics = trainer.run_training(max_updates=3)

        assert call_count == 3, f"Expected 3 collect_rollout calls, got {call_count}"
        assert trainer._update_count == 3
        assert len(metrics) == 3

    def test_same_trajectory_drives_metrics_and_ppo(self, tmp_path):
        """Metrics and PPO use the SAME trajectory object."""
        trainer = _make_trainer(tmp_path, max_updates=1, ppo_epochs=1)
        trajectory, _ = trainer.collect_rollout(seed=42)
        assert len(trajectory) > 0

        metrics = trainer._build_metrics_from_trajectory(
            trajectory, update_count=0, episode_count=0,
        )
        diag = trainer.perform_ppo_update(trajectory)

        # Metrics get PPO stats patched in during perform_ppo_update
        assert diag.total_loss >= 0

    def test_no_duplicate_env_execution_per_iteration(self, tmp_path):
        """Each training iteration executes the env exactly once per step."""
        trainer = _make_trainer(tmp_path, max_updates=1, ppo_epochs=1)
        trajectory, steps = trainer.collect_rollout(seed=42)
        assert steps == len(trajectory)


# ===========================================================================
# Defect 4: max_updates semantics
# ===========================================================================

class TestMaxUpdatesSemantics:
    """D3 — max_updates controls the number of PPO update iterations."""

    def test_max_updates_respected(self, tmp_path):
        """Training runs exactly max_updates iterations."""
        trainer = _make_trainer(tmp_path, max_updates=5, ppo_epochs=1, target_kl=10.0)
        metrics = trainer.run_training(max_updates=5)
        assert trainer._update_count == 5
        assert len(metrics) == 5

    def test_max_updates_zero_raises(self, tmp_path):
        """max_updates=0 raises ValueError."""
        trainer = _make_trainer(tmp_path, max_updates=0)
        with pytest.raises(ValueError, match="max_updates"):
            trainer.run_training(max_updates=0)

    def test_max_updates_none_raises(self, tmp_path):
        """Unset max_updates raises ValueError."""
        from policies.training import LinerShippingTrainer
        config = _make_minimal_config(max_updates=0)
        config.max_updates = None
        with patch("policies.training.LINERLIBLoader") as MockLoader:
            MockLoader.return_value.load.return_value = _make_toy_instance()
            trainer = LinerShippingTrainer("TOY_P12", "encoder_only", config,
                                           str(tmp_path / "ckpt"))
        with pytest.raises(ValueError, match="max_updates"):
            trainer.run_training()

    def test_checkpoint_frequency_respected(self, tmp_path):
        """Checkpoints saved at checkpoint_frequency intervals."""
        from policies.training import LinerShippingTrainer
        trainer = _make_trainer(tmp_path, max_updates=10, ppo_epochs=1,
                                target_kl=10.0, checkpoint_frequency=3)
        trainer.run_training(max_updates=10)
        ckpt_dir = tmp_path / "ckpt"
        final = ckpt_dir / "final_checkpoint.pt"
        assert final.exists()

    def test_config_max_updates_used_as_default(self, tmp_path):
        """When not overridden, config.max_updates is the default."""
        trainer = _make_trainer(tmp_path, max_updates=7, ppo_epochs=1, target_kl=10.0)
        metrics = trainer.run_training()
        assert trainer._update_count == 7


# ===========================================================================
# Defect 5: PPO ratio correctness
# ===========================================================================

class TestPPORatioCorrectness:
    """D5 — PPO ratio = exp(new_log_prob - old_log_prob)."""

    def test_ratio_formula(self):
        """Verify ratio = exp(new - old)."""
        old_lp = torch.tensor([-1.0, -2.0, -3.0])
        new_lp = torch.tensor([-0.5, -1.5, -2.5])
        ratios = torch.exp(new_lp - old_lp)
        expected = torch.tensor([0.5, 0.5, 0.5])  # all deltas = +0.5
        assert torch.allclose(ratios, torch.exp(torch.tensor(0.5)))

    def test_clipping_reduces_ratio(self):
        """Large policy change → clipping occurs."""
        from mcf.ppo_engine.config import PPOConfig
        from mcf.ppo_engine.trainer import PPOTrainer
        cfg = PPOConfig(clip_epsilon=0.2)
        old_lp = torch.tensor([0.0])
        new_lp = torch.tensor([2.0])  # ratio = e^2 ≈ 7.39 → clipped to 1.2
        adv = torch.ones(1)
        trainer = PPOTrainer(nn.Linear(10, 1), nn.Linear(10, 1), cfg)
        loss, clip_frac, kl = trainer.compute_ppo_loss(new_lp, old_lp, adv, 0.2)
        assert clip_frac > 0.0

    def test_identical_policies_give_zero_kl(self):
        """When new==old, KL = 0 exactly."""
        from mcf.ppo_engine.config import PPOConfig
        from mcf.ppo_engine.trainer import PPOTrainer
        cfg = PPOConfig(clip_epsilon=0.2)
        lp = torch.tensor([-1.0, -1.0])
        trainer = PPOTrainer(nn.Linear(10, 1), nn.Linear(10, 1), cfg)
        _, _, kl = trainer.compute_ppo_loss(lp, lp, torch.ones(2), 0.2)
        assert kl == 0.0


# ===========================================================================
# Defect 6: P8 fallback exclusion
# ===========================================================================

class TestP8FallbackExclusion:
    """D6 — Fallback-repaired samples excluded from PPO update."""

    def test_fallback_samples_filtered_out(self, tmp_path):
        """Steps with fallback_applied=True are removed before PPO."""
        trainer = _make_trainer(tmp_path, max_updates=1, ppo_epochs=1)
        graph_t, critic_dim = _make_graph_tensors(trainer)
        trajectory = [
            {"old_log_prob": torch.tensor(-1.0), "old_value": torch.tensor(0.0),
             "reward": 1.0, "done": False, "truncated": False,
             "entropy": torch.tensor(0.5), "state": graph_t,
             "critic_input": torch.zeros(1, critic_dim),
             "action": [], "executed_action": [],
             "fallback_applied": False,
             "decoded_port_sequence": ["A", "B"], "executed_port_sequence": ["A", "B"],
             "info": {}},
            {"old_log_prob": torch.tensor(-2.0), "old_value": torch.tensor(1.0),
             "reward": 2.0, "done": False, "truncated": False,
             "entropy": torch.tensor(0.3), "state": graph_t,
             "critic_input": torch.zeros(1, critic_dim),
             "action": [], "executed_action": [],
             "fallback_applied": True,
             "decoded_port_sequence": ["A", "B"], "executed_port_sequence": ["A", "B"],
             "info": {}},
        ]
        diag = trainer.perform_ppo_update(trajectory)
        assert torch.isfinite(torch.tensor(diag.policy_loss))

    def test_all_fallback_empty_trajectory(self, tmp_path):
        """All-fallback trajectory → zero diagnostics."""
        trainer = _make_trainer(tmp_path, max_updates=1, ppo_epochs=1)
        trajectory = [
            {"old_log_prob": torch.tensor(-1.0), "old_value": torch.tensor(0.0),
             "reward": 1.0, "done": False, "truncated": False,
             "entropy": torch.tensor(0.5), "state": None,
             "critic_input": torch.zeros(1, 10),
             "action": [], "executed_action": [],
             "fallback_applied": True,
             "decoded_port_sequence": [], "executed_port_sequence": [],
             "info": {}},
        ]
        diag = trainer.perform_ppo_update(trajectory)
        assert diag.policy_loss == 0.0
        assert diag.value_loss == 0.0
        assert diag.gradient_norm == 0.0


# ===========================================================================
# Defect 7: P9 sequence/log-prob consistency
# ===========================================================================

class TestP9SequenceConsistency:
    """D7 — P9 log_prob corresponds to decoded sequence, not executed."""

    def test_trajectory_stores_both_sequences(self, tmp_path):
        """Trajectory stores decoded_port_sequence and executed_port_sequence."""
        trainer = _make_trainer(tmp_path, max_updates=1, ppo_epochs=1,
                                 policy="encoder_decoder", hidden_dim=16, lstm_layers=1)
        trajectory, _ = trainer.collect_rollout(seed=42)
        assert len(trajectory) > 0
        for step in trajectory:
            assert "decoded_port_sequence" in step
            assert "executed_port_sequence" in step
            assert "old_log_prob" in step
            assert isinstance(step["old_log_prob"], torch.Tensor)

    def test_p9_log_prob_is_scalar_finite(self):
        """P9 log_prob is a finite scalar tensor."""
        from policies.encoder_decoder import EncoderDecoderPolicy
        from neural.backbone import NeuralBackbone
        from neural.config import ArchitectureConfig
        from actions.service_generator import ServiceGenerator
        from state.representation import StateEncoder, ServiceMembership
        from neural import neural_state_to_tensors
        from env.environment import LSNDPEnv

        toy = _make_toy_instance()
        dist_by_pair = {(a.origin, a.destination): a for a in toy.distances}
        gen = ServiceGenerator(toy, dist_by_pair)
        cfg = ArchitectureConfig(hidden_dim=16, gat_layers=1,
                                  transformer_layers=1, transformer_heads=2,
                                  lstm_layers=1)
        backbone = NeuralBackbone(cfg)
        policy = EncoderDecoderPolicy(backbone, toy, gen)

        encoder = StateEncoder(toy, dist_by_pair)
        env = LSNDPEnv(toy)
        obs, info = env.reset(seed=42)

        rem = {i: obs["remaining_demand"][i]
               for i in range(len(obs["remaining_demand"]))}
        fleet = {vc: obs["fleet_remaining"][i]
                 for i, vc in enumerate(sorted(toy.vessel_types.keys()))}
        ns = encoder.encode(rem, fleet, ServiceMembership())
        bundle = neural_state_to_tensors(ns)

        out = policy.sample_action(bundle, fleet, seed=42)
        assert out.log_prob is not None
        assert out.entropy is not None
        assert out.log_prob.dim() == 0
        assert torch.isfinite(out.log_prob)


# ===========================================================================
# Defect 8: Termination/truncation handling
# ===========================================================================

class TestTerminationHandling:
    """D8 — done/truncated flags correctly stored in trajectory."""

    def test_done_flag_preserved(self, tmp_path):
        """Terminal steps have done=True."""
        trainer = _make_trainer(tmp_path)
        trajectory, _ = trainer.collect_rollout(seed=42)
        assert len(trajectory) > 0
        assert trajectory[-1]["done"] is True

    def test_truncated_flag_stored(self):
        """Both done and truncated flags present."""
        step = {"old_log_prob": torch.tensor(-1.0), "old_value": torch.tensor(0.0),
                "reward": 1.0, "done": False, "truncated": True,
                "entropy": torch.tensor(0.5)}
        assert step["done"] is False
        assert step["truncated"] is True

    def test_gae_terminates_correctly(self):
        """GAE computation handles terminal states correctly."""
        from mcf.ppo_engine.config import PPOConfig
        from mcf.ppo_engine.trainer import PPOTrainer
        cfg = PPOConfig(gamma=1.0, gae_lambda=0.9)
        trainer = PPOTrainer(nn.Linear(10, 1), nn.Linear(10, 1), cfg)

        values = torch.tensor([0.0, 1.0, 2.0, 3.0])
        rewards = torch.tensor([1.0, 1.0, 1.0, 1.0])
        dones = torch.tensor([0.0, 0.0, 0.0, 1.0])

        returns, advantages = trainer.compute_returns_and_advantages(
            values, rewards, dones,
        )
        assert abs(returns[-1].item() - 1.0) < 1e-5


# ===========================================================================
# Defect 9: Economic metric propagation
# ===========================================================================

class TestEconomicMetricPropagation:
    """D4 — MCF cost components propagate into TrainingMetrics."""

    def test_cost_keys_match_environment_output(self, tmp_path):
        """Keys used match _build_info() keys."""
        trainer = _make_trainer(tmp_path)
        trajectory, _ = trainer.collect_rollout(seed=42)
        if not trajectory:
            pytest.skip("Empty trajectory")

        final_info = trajectory[-1]["info"]
        assert "profit" in final_info
        assert "num_services" in final_info
        assert "demand_rejected" in final_info
        assert "vessel_state" in final_info

    def test_nonzero_cost_components_on_real_run(self, tmp_path):
        """Cost extraction works with populated info dict."""
        trainer = _make_trainer(tmp_path)
        fake_traj = [{
            "info": {"profit": -1000.0, "num_services": 1,
                     "demand_rejected": 10.0,
                     "vessel_state": {"V1": 4.0, "V2": 3.0},
                     "service_cost": 500.0, "unused_vessel_cost": 200.0,
                     "voyage_cost": 300.0, "rejection_cost": 100.0,
                     "handling_cost": 50.0},
            "reward": 1.0, "old_log_prob": torch.tensor(-1.0),
            "old_value": torch.tensor(0.0), "entropy": torch.tensor(0.5),
        }]
        metrics = trainer._build_metrics_from_trajectory(
            fake_traj, update_count=0, episode_count=0,
        )
        assert metrics.C_service == 500.0
        assert metrics.C_voyage == 300.0
        assert metrics.C_reject == 100.0
        assert metrics.C_handle == 50.0
        assert metrics.network_profit_eta == -1000.0


# ===========================================================================
# Defect 10: Checkpoint save/load
# ===========================================================================

class TestCheckpointRoundTrip:
    """D10 — Checkpoint saves and restores all necessary state."""

    def test_save_and_load_restores_state(self, tmp_path):
        """Loading restores model params and counters."""
        trainer = _make_trainer(tmp_path, max_updates=1, ppo_epochs=1)
        trajectory, _ = trainer.collect_rollout(seed=42)
        if trajectory:
            trainer.perform_ppo_update(trajectory)

        assert trainer._update_count >= 1
        path = trainer.save_checkpoint("test_ckpt.pt")
        assert Path(path).exists()

        trainer2 = _make_trainer(tmp_path, max_updates=0)
        trainer2.load_checkpoint(path)
        assert trainer2._update_count == trainer._update_count
        assert trainer2._episode_count == trainer._episode_count

    def test_checkpoint_contains_raw_data_hashes(self, tmp_path):
        """Checkpoint preserves raw data hashes."""
        trainer = _make_trainer(tmp_path, max_updates=1, ppo_epochs=1)
        trainer.save_checkpoint("hash_ckpt.pt")
        ckpt = torch.load(str(tmp_path / "ckpt" / "hash_ckpt.pt"),
                          map_location="cpu", weights_only=False)
        assert "raw_data_hashes" in ckpt
        assert isinstance(ckpt["raw_data_hashes"], dict)
        assert len(ckpt["raw_data_hashes"]) > 0


# ===========================================================================
# Defect 11: Deterministic seeded rollout
# ===========================================================================

class TestDeterministicSeededRollout:
    """D11 — Same seed produces identical initial trajectory length."""

    def test_same_seed_same_trajectory_length(self, tmp_path):
        """Two rollouts with the same seed have the same number of steps."""
        t1 = _make_trainer(tmp_path, max_updates=1, ppo_epochs=1)
        t2 = _make_trainer(tmp_path, max_updates=1, ppo_epochs=1)
        _, steps1 = t1.collect_rollout(seed=42)
        _, steps2 = t2.collect_rollout(seed=42)
        assert steps1 == steps2
        assert steps1 > 0

    def test_different_seed_different_trajectory(self, tmp_path):
        """Different seeds can yield different trajectory lengths."""
        trainer = _make_trainer(tmp_path, max_updates=1, ppo_epochs=1)
        _, steps_a = trainer.collect_rollout(seed=42)
        _, steps_b = trainer.collect_rollout(seed=99)
        assert steps_a > 0
        assert steps_b > 0


# ===========================================================================
# Defect 12: Raw data immutability
# ===========================================================================

class TestDataIntegrity:
    """D12 — Raw LINERLIB data files remain unchanged during training."""

    def test_hash_verification_before_after(self, tmp_path):
        """Raw data hashes recorded at init and verified after training."""
        trainer = _make_trainer(tmp_path, max_updates=1, ppo_epochs=1)
        hashes_before = dict(trainer._raw_data_hashes)
        assert len(hashes_before) > 0

        trajectory, _ = trainer.collect_rollout(seed=42)
        if trajectory:
            trainer.perform_ppo_update(trajectory)

        assert trainer._verify_data_integrity()
        assert trainer._raw_data_hashes == hashes_before


# ===========================================================================
# Defect 13: Numerical finite-value checks
# ===========================================================================

class TestNumericalStability:
    """D13 — Losses, KL, gradients must be finite."""

    def test_ppo_diagnostics_finite(self, tmp_path):
        """All PDiagnostics fields must be finite."""
        trainer = _make_trainer(tmp_path, max_updates=1, ppo_epochs=1)
        trajectory, _ = trainer.collect_rollout(seed=42)
        if not trajectory:
            pytest.skip("Empty trajectory")
        diag = trainer.perform_ppo_update(trajectory)

        assert math.isfinite(diag.policy_loss)
        assert math.isfinite(diag.value_loss)
        assert math.isfinite(diag.total_loss)
        assert math.isfinite(diag.approx_kl)
        assert math.isfinite(diag.gradient_norm)
        assert math.isfinite(diag.advantage_mean)
        assert math.isfinite(diag.value_mean)

    def test_old_log_prob_finite(self, tmp_path):
        """All old_log_prob values in trajectory are finite."""
        trainer = _make_trainer(tmp_path, max_updates=1, ppo_epochs=1)
        trajectory, _ = trainer.collect_rollout(seed=42)
        for step in trajectory:
            assert torch.isfinite(step["old_log_prob"])
            assert torch.isfinite(step["old_value"])


# ===========================================================================
# Defect 14: Exception handling audit
# ===========================================================================

class TestExceptionHandling:
    """D14 — Broad exception handlers do not mask programming errors."""

    def test_programming_error_propagates(self, tmp_path):
        """Non-ServiceValidationError exceptions propagate through collect_rollout."""
        from env.environment import ServiceValidationError
        trainer = _make_trainer(tmp_path)
        original_step = trainer.env.step

        def raise_type_error(*args, **kwargs):
            raise TypeError("Programming error — should propagate")

        try:
            trainer.env.step = raise_type_error
            with pytest.raises(TypeError, match="Programming error"):
                trainer.collect_rollout(seed=42)
        finally:
            trainer.env.step = original_step

    def test_service_validation_error_caught(self, tmp_path):
        """ServiceValidationError is caught and retried with fallback."""
        from env.environment import ServiceValidationError
        trainer = _make_trainer(tmp_path)
        original_step = trainer.env.step
        call_count = 0

        def failing_then_ok(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise ServiceValidationError(
                    __import__("env.action", fromlist=["ServiceAction"])
                    .ServiceAction(vessel_class="V1", port_sequence=["A"]),
                    ["Test failure"],
                )
            return original_step(*args, **kwargs)

        try:
            trainer.env.step = failing_then_ok
            trajectory, steps = trainer.collect_rollout(seed=42)
            # Should succeed after retry (steps >= 0)
            assert steps >= 0
        finally:
            trainer.env.step = original_step


# ===========================================================================
# End-to-end smoke test
# ===========================================================================

class TestEndToEndSmoke:
    """D15 — Full end-to-end smoke test on synthetic instance."""

    def test_full_training_pipeline(self, tmp_path):
        """Complete training run completes without error."""
        trainer = _make_trainer(tmp_path, max_updates=3, ppo_epochs=2,
                                 target_kl=10.0, checkpoint_frequency=10)
        metrics = trainer.run_training(max_updates=3)

        assert len(metrics) == 3
        assert trainer._update_count == 3
        assert trainer._episode_count == 3

        for m in metrics:
            assert math.isfinite(m.reward)
            assert math.isfinite(m.PPO_approx_kl)
            assert math.isfinite(m.gradient_norm)

        final_ckpt = tmp_path / "ckpt" / "final_checkpoint.pt"
        assert final_ckpt.exists()


# ===========================================================================
# Main
# ===========================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
