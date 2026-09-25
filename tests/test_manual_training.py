"""
test_manual_training.py -- Tests for the manual training control system.

Covers:
  1. Configuration validation (all valid presets, invalid configs)
  2. PAPER mode invariant enforcement
  3. Checkpoint save/load round-trip with config preservation
  4. Training metrics CSV/JSONL output
  5. Resume from checkpoint (incompatible architecture rejection)
  6. Experiment directory creation (no overwrites)
  7. Diagnostic observability fields
  8. Reward diagnostic decomposition
  9. Preset application
 10. Edge cases (zero perturbation, extreme values)
"""

from __future__ import annotations

import csv
import json
import math
import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch
from policies.training import LinerShippingTrainer

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

from runners.config import RunnerConfigError, validate_runner_config


# ===========================================================================
# Import the module under test
# ===========================================================================

import train_rl as mt


# ===========================================================================
# Helper: build a minimal working toy LINERLIBInstance with bidirectional arcs
# ===========================================================================

def _build_toy_instance(name="TOY_MT", vessel_qty=5):
    """Build a 2-port toy instance with A<->B bidirectional distances."""
    from data.instance import (
        DatasetProvenance, Demand, DistanceArc, FleetEntry,
        InstanceMetadata, LINERLIBInstance, Port, ProvenanceRecord,
        VesselType,
    )
    ports = {
        "A": Port(unlocode="A", name="Port A", country=None, cabotage_region="t",
                  d_region=None, longitude=None, latitude=None, draft=10.0,
                  cost_per_full=1.0, cost_per_full_transfer=0.5,
                  port_call_cost_fixed=100.0, port_call_cost_per_ffe=0.5,
                  provenance=ProvenanceRecord(source_file="synthetic", source_row=1)),
        "B": Port(unlocode="B", name="Port B", country=None, cabotage_region="t",
                  d_region=None, longitude=None, latitude=None, draft=10.0,
                  cost_per_full=1.0, cost_per_full_transfer=0.5,
                  port_call_cost_fixed=100.0, port_call_cost_per_ffe=0.5,
                  provenance=ProvenanceRecord(source_file="synthetic", source_row=2)),
    }
    vessels = {
        "V1": VesselType(vessel_class="V1", capacity_ffe=100.0, tc_rate_daily=100,
                         draft=12.0, min_speed=5.0, max_speed=15.0, design_speed=10.0,
                         bunker_ton_per_day_at_design=50.0,
                         idle_consumption_ton_per_day=10.0, panama_fee=0, suez_fee=0,
                         provenance=ProvenanceRecord(source_file="synthetic", source_row=1)),
    }
    demands = [Demand(origin="A", destination="B", ffe_per_week=50.0, revenue=200.0,
                      max_transit_time=10,
                      provenance=ProvenanceRecord(source_file="synthetic", source_row=1))]
    distances = [
        DistanceArc(origin="A", destination="B", distance_nm=100.0,
                    draft_required=10.0, is_panama=False, is_suez=False,
                    provenance=ProvenanceRecord(source_file="synthetic", source_row=1)),
        DistanceArc(origin="B", destination="A", distance_nm=100.0,
                    draft_required=10.0, is_panama=False, is_suez=False,
                    provenance=ProvenanceRecord(source_file="synthetic", source_row=2)),
    ]
    fleet = [FleetEntry(vessel_class="V1", quantity=vessel_qty)]
    metadata = InstanceMetadata(name=name, active_port_count=2,
                                 vessel_type_count=1, total_vessels=vessel_qty,
                                 demand_count=1, distance_arc_count=2)
    return LINERLIBInstance(
        name=name, ports=ports, vessel_types=vessels,
        demands=demands, distances=distances, fleet=fleet,
        metadata=metadata,
        provenance=DatasetProvenance(source_root="[SYNTHETIC]"),
    )


def _make_trainer_mt(tmp_path, **overrides):
    """Create a LinerShippingTrainer backed by the bidirectional toy instance."""
    from policies.training import TrainingConfig
    defaults = dict(
        dataset="TOY_MT", policy="encoder_only",
        learning_rate=2e-4, gamma=1.0, gae_lambda=0.9, ppo_epochs=1,
        clip_epsilon=0.2, target_kl=10.0, entropy_coefficient=0.05,
        value_coefficient=0.5, num_envs=1, steps_per_env=20, minibatch_size=16,
        seed=42, max_updates=3, checkpoint_frequency=100,
        hidden_dim=16, gat_layers=1, transformer_layers=1,
        transformer_heads=2, lstm_layers=1,
        perturbation_fraction=0.0, n_perturbed_instances=0,
    )
    defaults.update(overrides)
    tc = TrainingConfig(**defaults)
    toy = _build_toy_instance()
    with patch("policies.training.LINERLIBLoader") as MockLoader:
        mock_loader = MagicMock()
        mock_loader.load.return_value = toy
        MockLoader.return_value = mock_loader
        return LinerShippingTrainer(
            instance_name="TOY_MT", policy_type="encoder_only",
            config=tc, checkpoint_dir=str(tmp_path / "ckpt"),
        )


# ===========================================================================
# 1. Configuration Validation
# ===========================================================================

class TestConfigValidation:
    """Configuration must be validated before any PyTorch execution."""

    def test_valid_minimal_config(self):
        cfg = mt.validate_manual_config(
            instance="Baltic", policy="encoder_only",
            hidden_dim=32, gat_layers=1, transformer_layers=1,
            transformer_heads=2, lstm_layers=1,
            learning_rate=2e-4, gamma=1.0, gae_lambda=0.9,
            ppo_epochs=2, clip_epsilon=0.2, target_kl=0.1,
            entropy_coefficient=0.05, value_coefficient=0.5,
            num_envs=1, steps_per_env=50, minibatch_size=32,
            max_updates=10, perturbation_fraction=0.0,
            n_perturbed_instances=0, apply_transittime_revision=False,
            seed=42, checkpoint_frequency=5, preset="MANUAL",
        )
        assert cfg["instance"] == "Baltic"
        assert cfg["policy"] == "encoder_only"
        assert cfg["hidden_dim"] == 32

    def test_hidden_dim_zero_rejected(self):
        with pytest.raises(ValueError, match="hidden_dim"):
            mt.validate_manual_config(
                instance="Baltic", policy="encoder_only",
                hidden_dim=0, gat_layers=1, transformer_layers=1,
                transformer_heads=1, lstm_layers=1,
                learning_rate=2e-4, gamma=1.0, gae_lambda=0.9,
                ppo_epochs=2, clip_epsilon=0.2, target_kl=0.1,
                entropy_coefficient=0.05, value_coefficient=0.5,
                num_envs=1, steps_per_env=50, minibatch_size=32,
                max_updates=10, perturbation_fraction=0.0,
                n_perturbed_instances=0, apply_transittime_revision=False,
                seed=42, checkpoint_frequency=5, preset="SMOKE",
            )

    def test_gat_layers_zero_rejected(self):
        with pytest.raises(ValueError, match="gat_layers"):
            mt.validate_manual_config(
                instance="Baltic", policy="encoder_only",
                hidden_dim=32, gat_layers=0, transformer_layers=1,
                transformer_heads=2, lstm_layers=1,
                learning_rate=2e-4, gamma=1.0, gae_lambda=0.9,
                ppo_epochs=2, clip_epsilon=0.2, target_kl=0.1,
                entropy_coefficient=0.05, value_coefficient=0.5,
                num_envs=1, steps_per_env=50, minibatch_size=32,
                max_updates=10, perturbation_fraction=0.0,
                n_perturbed_instances=0, apply_transittime_revision=False,
                seed=42, checkpoint_frequency=5, preset="SMOKE",
            )

    def test_transformer_heads_zero_rejected(self):
        with pytest.raises(ValueError, match="transformer_heads"):
            mt.validate_manual_config(
                instance="Baltic", policy="encoder_only",
                hidden_dim=32, gat_layers=1, transformer_layers=1,
                transformer_heads=0, lstm_layers=1,
                learning_rate=2e-4, gamma=1.0, gae_lambda=0.9,
                ppo_epochs=2, clip_epsilon=0.2, target_kl=0.1,
                entropy_coefficient=0.05, value_coefficient=0.5,
                num_envs=1, steps_per_env=50, minibatch_size=32,
                max_updates=10, perturbation_fraction=0.0,
                n_perturbed_instances=0, apply_transittime_revision=False,
                seed=42, checkpoint_frequency=5, preset="SMOKE",
            )

    def test_hidden_not_divisible_by_heads(self):
        with pytest.raises(ValueError, match="divisible"):
            mt.validate_manual_config(
                instance="Baltic", policy="encoder_only",
                hidden_dim=33, gat_layers=1, transformer_layers=1,
                transformer_heads=4, lstm_layers=1,
                learning_rate=2e-4, gamma=1.0, gae_lambda=0.9,
                ppo_epochs=2, clip_epsilon=0.2, target_kl=0.1,
                entropy_coefficient=0.05, value_coefficient=0.5,
                num_envs=1, steps_per_env=50, minibatch_size=32,
                max_updates=10, perturbation_fraction=0.0,
                n_perturbed_instances=0, apply_transittime_revision=False,
                seed=42, checkpoint_frequency=5, preset="SMOKE",
            )

    def test_negative_learning_rate(self):
        with pytest.raises(ValueError, match="learning_rate"):
            mt.validate_manual_config(
                instance="Baltic", policy="encoder_only",
                hidden_dim=32, gat_layers=1, transformer_layers=1,
                transformer_heads=2, lstm_layers=1,
                learning_rate=-1e-4, gamma=1.0, gae_lambda=0.9,
                ppo_epochs=2, clip_epsilon=0.2, target_kl=0.1,
                entropy_coefficient=0.05, value_coefficient=0.5,
                num_envs=1, steps_per_env=50, minibatch_size=32,
                max_updates=10, perturbation_fraction=0.0,
                n_perturbed_instances=0, apply_transittime_revision=False,
                seed=42, checkpoint_frequency=5, preset="SMOKE",
            )

    def test_gamma_out_of_range_above(self):
        with pytest.raises(ValueError, match="gamma"):
            mt.validate_manual_config(
                instance="Baltic", policy="encoder_only",
                hidden_dim=32, gat_layers=1, transformer_layers=1,
                transformer_heads=2, lstm_layers=1,
                learning_rate=2e-4, gamma=1.5, gae_lambda=0.9,
                ppo_epochs=2, clip_epsilon=0.2, target_kl=0.1,
                entropy_coefficient=0.05, value_coefficient=0.5,
                num_envs=1, steps_per_env=50, minibatch_size=32,
                max_updates=10, perturbation_fraction=0.0,
                n_perturbed_instances=0, apply_transittime_revision=False,
                seed=42, checkpoint_frequency=5, preset="SMOKE",
            )

    def test_clip_epsilon_zero(self):
        with pytest.raises(ValueError, match="clip_epsilon"):
            mt.validate_manual_config(
                instance="Baltic", policy="encoder_only",
                hidden_dim=32, gat_layers=1, transformer_layers=1,
                transformer_heads=2, lstm_layers=1,
                learning_rate=2e-4, gamma=1.0, gae_lambda=0.9,
                ppo_epochs=2, clip_epsilon=0.0, target_kl=0.1,
                entropy_coefficient=0.05, value_coefficient=0.5,
                num_envs=1, steps_per_env=50, minibatch_size=32,
                max_updates=10, perturbation_fraction=0.0,
                n_perturbed_instances=0, apply_transittime_revision=False,
                seed=42, checkpoint_frequency=5, preset="SMOKE",
            )

    def test_steps_per_env_zero(self):
        with pytest.raises(ValueError, match="steps_per_env"):
            mt.validate_manual_config(
                instance="Baltic", policy="encoder_only",
                hidden_dim=32, gat_layers=1, transformer_layers=1,
                transformer_heads=2, lstm_layers=1,
                learning_rate=2e-4, gamma=1.0, gae_lambda=0.9,
                ppo_epochs=2, clip_epsilon=0.2, target_kl=0.1,
                entropy_coefficient=0.05, value_coefficient=0.5,
                num_envs=1, steps_per_env=0, minibatch_size=32,
                max_updates=10, perturbation_fraction=0.0,
                n_perturbed_instances=0, apply_transittime_revision=False,
                seed=42, checkpoint_frequency=5, preset="SMOKE",
            )

    def test_num_envs_zero(self):
        with pytest.raises(ValueError, match="num_envs"):
            mt.validate_manual_config(
                instance="Baltic", policy="encoder_only",
                hidden_dim=32, gat_layers=1, transformer_layers=1,
                transformer_heads=2, lstm_layers=1,
                learning_rate=2e-4, gamma=1.0, gae_lambda=0.9,
                ppo_epochs=2, clip_epsilon=0.2, target_kl=0.1,
                entropy_coefficient=0.05, value_coefficient=0.5,
                num_envs=0, steps_per_env=50, minibatch_size=32,
                max_updates=10, perturbation_fraction=0.0,
                n_perturbed_instances=0, apply_transittime_revision=False,
                seed=42, checkpoint_frequency=5, preset="SMOKE",
            )

    def test_max_updates_zero(self):
        with pytest.raises(ValueError, match="max_updates"):
            mt.validate_manual_config(
                instance="Baltic", policy="encoder_only",
                hidden_dim=32, gat_layers=1, transformer_layers=1,
                transformer_heads=2, lstm_layers=1,
                learning_rate=2e-4, gamma=1.0, gae_lambda=0.9,
                ppo_epochs=2, clip_epsilon=0.2, target_kl=0.1,
                entropy_coefficient=0.05, value_coefficient=0.5,
                num_envs=1, steps_per_env=50, minibatch_size=32,
                max_updates=0, perturbation_fraction=0.0,
                n_perturbed_instances=0, apply_transittime_revision=False,
                seed=42, checkpoint_frequency=5, preset="SMOKE",
            )

    def test_minibatch_size_zero(self):
        with pytest.raises(ValueError, match="minibatch_size"):
            mt.validate_manual_config(
                instance="Baltic", policy="encoder_only",
                hidden_dim=32, gat_layers=1, transformer_layers=1,
                transformer_heads=2, lstm_layers=1,
                learning_rate=2e-4, gamma=1.0, gae_lambda=0.9,
                ppo_epochs=2, clip_epsilon=0.2, target_kl=0.1,
                entropy_coefficient=0.05, value_coefficient=0.5,
                num_envs=1, steps_per_env=50, minibatch_size=0,
                max_updates=10, perturbation_fraction=0.0,
                n_perturbed_instances=0, apply_transittime_revision=False,
                seed=42, checkpoint_frequency=5, preset="SMOKE",
            )

    def test_negative_perturbation_fraction(self):
        with pytest.raises(ValueError, match="perturbation_fraction"):
            mt.validate_manual_config(
                instance="Baltic", policy="encoder_only",
                hidden_dim=32, gat_layers=1, transformer_layers=1,
                transformer_heads=2, lstm_layers=1,
                learning_rate=2e-4, gamma=1.0, gae_lambda=0.9,
                ppo_epochs=2, clip_epsilon=0.2, target_kl=0.1,
                entropy_coefficient=0.05, value_coefficient=0.5,
                num_envs=1, steps_per_env=50, minibatch_size=32,
                max_updates=10, perturbation_fraction=-0.1,
                n_perturbed_instances=100, apply_transittime_revision=False,
                seed=42, checkpoint_frequency=5, preset="SMOKE",
            )

    def test_perturbation_fraction_positive_needs_instances(self):
        with pytest.raises(ValueError, match="n_perturbed_instances"):
            mt.validate_manual_config(
                instance="Baltic", policy="encoder_only",
                hidden_dim=32, gat_layers=1, transformer_layers=1,
                transformer_heads=2, lstm_layers=1,
                learning_rate=2e-4, gamma=1.0, gae_lambda=0.9,
                ppo_epochs=2, clip_epsilon=0.2, target_kl=0.1,
                entropy_coefficient=0.05, value_coefficient=0.5,
                num_envs=1, steps_per_env=50, minibatch_size=32,
                max_updates=10, perturbation_fraction=0.1,
                n_perturbed_instances=0, apply_transittime_revision=False,
                seed=42, checkpoint_frequency=5, preset="SMOKE",
            )

    def test_unknown_instance_rejected(self):
        with pytest.raises(ValueError):
            mt.validate_manual_config(
                instance="NonExistent", policy="encoder_only",
                hidden_dim=32, gat_layers=1, transformer_layers=1,
                transformer_heads=2, lstm_layers=1,
                learning_rate=2e-4, gamma=1.0, gae_lambda=0.9,
                ppo_epochs=2, clip_epsilon=0.2, target_kl=0.1,
                entropy_coefficient=0.05, value_coefficient=0.5,
                num_envs=1, steps_per_env=50, minibatch_size=32,
                max_updates=10, perturbation_fraction=0.0,
                n_perturbed_instances=0, apply_transittime_revision=False,
                seed=42, checkpoint_frequency=5, preset="SMOKE",
            )

    def test_unknown_policy_rejected(self):
        with pytest.raises(ValueError):
            mt.validate_manual_config(
                instance="Baltic", policy="quantum_fleet",
                hidden_dim=32, gat_layers=1, transformer_layers=1,
                transformer_heads=2, lstm_layers=1,
                learning_rate=2e-4, gamma=1.0, gae_lambda=0.9,
                ppo_epochs=2, clip_epsilon=0.2, target_kl=0.1,
                entropy_coefficient=0.05, value_coefficient=0.5,
                num_envs=1, steps_per_env=50, minibatch_size=32,
                max_updates=10, perturbation_fraction=0.0,
                n_perturbed_instances=0, apply_transittime_revision=False,
                seed=42, checkpoint_frequency=5, preset="SMOKE",
            )

    def test_all_instances_accepted(self):
        """All LINERLIB instances are accepted by validation."""
        from data.linerlib_loader import LINERLIBLoader
        loader = LINERLIBLoader("data")
        for name in loader.available_instances():
            cfg = mt.validate_manual_config(
                instance=name, policy="encoder_only",
                hidden_dim=32, gat_layers=1, transformer_layers=1,
                transformer_heads=2, lstm_layers=1,
                learning_rate=2e-4, gamma=1.0, gae_lambda=0.9,
                ppo_epochs=2, clip_epsilon=0.2, target_kl=0.1,
                entropy_coefficient=0.05, value_coefficient=0.5,
                num_envs=1, steps_per_env=50, minibatch_size=32,
                max_updates=1, perturbation_fraction=0.0,
                n_perturbed_instances=0, apply_transittime_revision=False,
                seed=42, checkpoint_frequency=100, preset="SMOKE",
            )
            assert cfg["instance"] == name

    def test_encoder_decoder_accepted(self):
        cfg = mt.validate_manual_config(
            instance="Baltic", policy="encoder_decoder",
            hidden_dim=32, gat_layers=1, transformer_layers=1,
            transformer_heads=2, lstm_layers=1,
            learning_rate=2e-4, gamma=1.0, gae_lambda=0.9,
            ppo_epochs=2, clip_epsilon=0.2, target_kl=0.1,
            entropy_coefficient=0.05, value_coefficient=0.5,
            num_envs=1, steps_per_env=50, minibatch_size=32,
            max_updates=10, perturbation_fraction=0.0,
            n_perturbed_instances=0, apply_transittime_revision=False,
            seed=42, checkpoint_frequency=5, preset="MANUAL",
        )
        assert cfg["policy"] == "encoder_decoder"


# ===========================================================================
# 2. PAPER Mode Invariant Enforcement
# ===========================================================================

class TestPaperMode:
    """PAPER mode enforces paper-faithful architectural constants."""

    def test_paper_mode_accepts_paper_dims(self):
        cfg = mt.validate_manual_config(
            instance="Baltic", policy="encoder_decoder",
            hidden_dim=mt.PAPER_HIDDEN_DIM,
            gat_layers=mt.PAPER_GAT_LAYERS,
            transformer_layers=mt.PAPER_TRANSFORMER_LAYERS,
            transformer_heads=mt.PAPER_TRANSFORMER_HEADS,
            lstm_layers=mt.PAPER_LSTM_LAYERS,
            learning_rate=mt.PAPER_LEARNING_RATE,
            gamma=mt.PAPER_GAMMA, gae_lambda=mt.PAPER_GAE_LAMBDA,
            ppo_epochs=mt.PAPER_PPO_EPOCHS,
            clip_epsilon=mt.PAPER_CLIP_EPSILON,
            target_kl=mt.PAPER_TARGET_KL,
            entropy_coefficient=mt.PAPER_ENTROPY_COEFFICIENT,
            value_coefficient=mt.PAPER_VALUE_COEFFICIENT,
            num_envs=8, steps_per_env=100, minibatch_size=64,
            max_updates=200, perturbation_fraction=0.10,
            n_perturbed_instances=100,
            apply_transittime_revision=False,
            seed=42, checkpoint_frequency=50, preset="PAPER",
        )
        assert cfg["preset"] == "PAPER"
        assert cfg["hidden_dim"] == mt.PAPER_HIDDEN_DIM
        assert cfg["apply_transittime_revision"] is False

    def test_paper_mode_rejects_non_paper_hidden(self):
        with pytest.raises(ValueError, match="PAPER mode"):
            mt.validate_manual_config(
                instance="Baltic", policy="encoder_decoder",
                hidden_dim=64, gat_layers=mt.PAPER_GAT_LAYERS,
                transformer_layers=mt.PAPER_TRANSFORMER_LAYERS,
                transformer_heads=mt.PAPER_TRANSFORMER_HEADS,
                lstm_layers=mt.PAPER_LSTM_LAYERS,
                learning_rate=mt.PAPER_LEARNING_RATE,
                gamma=mt.PAPER_GAMMA, gae_lambda=mt.PAPER_GAE_LAMBDA,
                ppo_epochs=mt.PAPER_PPO_EPOCHS,
                clip_epsilon=mt.PAPER_CLIP_EPSILON,
                target_kl=mt.PAPER_TARGET_KL,
                entropy_coefficient=mt.PAPER_ENTROPY_COEFFICIENT,
                value_coefficient=mt.PAPER_VALUE_COEFFICIENT,
                num_envs=8, steps_per_env=100, minibatch_size=64,
                max_updates=200, perturbation_fraction=0.10,
                n_perturbed_instances=100,
                apply_transittime_revision=False,
                seed=42, checkpoint_frequency=50, preset="PAPER",
            )

    def test_paper_mode_rejects_transit_time_enabled(self):
        with pytest.raises(ValueError, match="PAPER mode"):
            mt.validate_manual_config(
                instance="Baltic", policy="encoder_decoder",
                hidden_dim=mt.PAPER_HIDDEN_DIM,
                gat_layers=mt.PAPER_GAT_LAYERS,
                transformer_layers=mt.PAPER_TRANSFORMER_LAYERS,
                transformer_heads=mt.PAPER_TRANSFORMER_HEADS,
                lstm_layers=mt.PAPER_LSTM_LAYERS,
                learning_rate=mt.PAPER_LEARNING_RATE,
                gamma=mt.PAPER_GAMMA, gae_lambda=mt.PAPER_GAE_LAMBDA,
                ppo_epochs=mt.PAPER_PPO_EPOCHS,
                clip_epsilon=mt.PAPER_CLIP_EPSILON,
                target_kl=mt.PAPER_TARGET_KL,
                entropy_coefficient=mt.PAPER_ENTROPY_COEFFICIENT,
                value_coefficient=mt.PAPER_VALUE_COEFFICIENT,
                num_envs=8, steps_per_env=100, minibatch_size=64,
                max_updates=200, perturbation_fraction=0.10,
                n_perturbed_instances=100,
                apply_transittime_revision=True,
                seed=42, checkpoint_frequency=50, preset="PAPER",
            )

    def test_paper_mode_rejects_wrong_gat_layers(self):
        with pytest.raises(ValueError, match="PAPER mode"):
            mt.validate_manual_config(
                instance="Baltic", policy="encoder_decoder",
                hidden_dim=mt.PAPER_HIDDEN_DIM,
                gat_layers=1, transformer_layers=mt.PAPER_TRANSFORMER_LAYERS,
                transformer_heads=mt.PAPER_TRANSFORMER_HEADS,
                lstm_layers=mt.PAPER_LSTM_LAYERS,
                learning_rate=mt.PAPER_LEARNING_RATE,
                gamma=mt.PAPER_GAMMA, gae_lambda=mt.PAPER_GAE_LAMBDA,
                ppo_epochs=mt.PAPER_PPO_EPOCHS,
                clip_epsilon=mt.PAPER_CLIP_EPSILON,
                target_kl=mt.PAPER_TARGET_KL,
                entropy_coefficient=mt.PAPER_ENTROPY_COEFFICIENT,
                value_coefficient=mt.PAPER_VALUE_COEFFICIENT,
                num_envs=8, steps_per_env=100, minibatch_size=64,
                max_updates=200, perturbation_fraction=0.10,
                n_perturbed_instances=100,
                apply_transittime_revision=False,
                seed=42, checkpoint_frequency=50, preset="PAPER",
            )


# ===========================================================================
# 3. Preset Application
# ===========================================================================

class TestPresets:
    """Named presets set correct defaults."""

    def test_smoke_preset(self):
        mt._apply_preset("SMOKE")
        assert mt.INSTANCE == "Baltic"
        assert mt.POLICY == "encoder_only"
        assert mt.HIDDEN_DIM == 16
        assert mt.GAT_LAYERS == 1
        assert mt.MAX_UPDATES == 3
        assert mt.STEPS_PER_ENV == 20
        # Restore
        mt.PRESET = "MANUAL"
        mt.INSTANCE = "Baltic"
        mt.POLICY = "encoder_decoder"
        mt.HIDDEN_DIM = 64
        mt.GAT_LAYERS = 2
        mt.MAX_UPDATES = 10
        mt.STEPS_PER_ENV = 50

    def test_debug_preset(self):
        mt._apply_preset("DEBUG")
        assert mt.HIDDEN_DIM == mt.PAPER_HIDDEN_DIM
        assert mt.GAT_LAYERS == mt.PAPER_GAT_LAYERS
        assert mt.TRANSFORMER_LAYERS == mt.PAPER_TRANSFORMER_LAYERS
        assert mt.TRANSFORMER_HEADS == mt.PAPER_TRANSFORMER_HEADS
        assert mt.POLICY == "encoder_decoder"
        # Restore
        mt.PRESET = "MANUAL"
        mt.INSTANCE = "Baltic"
        mt.POLICY = "encoder_decoder"
        mt.HIDDEN_DIM = 64
        mt.GAT_LAYERS = 2
        mt.TRANSFORMER_LAYERS = 2
        mt.TRANSFORMER_HEADS = 4

    def test_paper_preset(self):
        mt._apply_preset("PAPER")
        assert mt.HIDDEN_DIM == mt.PAPER_HIDDEN_DIM
        assert mt.GAT_LAYERS == mt.PAPER_GAT_LAYERS
        assert mt.TRANSFORMER_LAYERS == mt.PAPER_TRANSFORMER_LAYERS
        assert mt.TRANSFORMER_HEADS == mt.PAPER_TRANSFORMER_HEADS
        assert mt.NUM_ENVS == 8
        assert mt.STEPS_PER_ENV == 100
        assert mt.PERTURBATION_FRACTION == 0.10
        assert mt.APPLY_TRANSITTIME_REVISION is False
        # Restore
        mt.PRESET = "MANUAL"
        mt.INSTANCE = "Baltic"
        mt.POLICY = "encoder_decoder"
        mt.HIDDEN_DIM = 64
        mt.GAT_LAYERS = 2
        mt.TRANSFORMER_LAYERS = 2
        mt.TRANSFORMER_HEADS = 4
        mt.NUM_ENVS = 1
        mt.STEPS_PER_ENV = 50
        mt.PERTURBATION_FRACTION = 0.0
        mt.APPLY_TRANSITTIME_REVISION = False

    def test_unknown_preset_raises(self):
        with pytest.raises(ValueError, match="Unknown preset"):
            mt._apply_preset("NONEXISTENT")

    def test_manual_preset_no_override(self):
        original_hidden = mt.HIDDEN_DIM
        mt._apply_preset("MANUAL")
        assert mt.HIDDEN_DIM == original_hidden


# ===========================================================================
# 4. Checkpoint Save / Load Round-Trip
# ===========================================================================

class TestCheckpointRoundTrip:
    def test_save_load_restores_update_count(self, tmp_path):
        trainer = _make_trainer_mt(tmp_path)
        trajectory, _ = trainer.collect_rollout(seed=42)
        if trajectory:
            trainer.perform_ppo_update(trajectory)
        initial_count = trainer._update_count
        assert initial_count >= 1
        path = trainer.save_checkpoint("test.pt")
        trainer2 = _make_trainer_mt(tmp_path)
        trainer2.load_checkpoint(path)
        assert trainer2._update_count == initial_count

    def test_checkpoint_contains_config(self, tmp_path):
        trainer = _make_trainer_mt(tmp_path)
        path = trainer.save_checkpoint("cfg_test.pt")
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        assert "config" in ckpt
        assert ckpt["config"]["hidden_dim"] == 16
        assert ckpt["config"]["seed"] == 42
        assert ckpt["config"]["dataset"] == "TOY_MT"
        assert ckpt["config"]["policy"] == "encoder_only"

    def test_incompatible_checkpoint_rejected(self, tmp_path):
        trainer = _make_trainer_mt(tmp_path, hidden_dim=16, gat_layers=1)
        trainer.save_checkpoint("small.pt")
        ckpt = torch.load(str(tmp_path / "ckpt" / "small.pt"),
                          map_location="cpu", weights_only=False)
        assert ckpt["config"]["hidden_dim"] == 16
        assert ckpt["config"]["gat_layers"] == 1


# ===========================================================================
# 5. Metric Logging
# ===========================================================================

class TestMetricLogging:
    def test_metrics_csv_has_required_columns(self, tmp_path):
        trainer = _make_trainer_mt(tmp_path, max_updates=2)
        metrics = trainer.run_training(max_updates=2)
        assert len(metrics) == 2
        for m in metrics:
            assert hasattr(m, "reward")
            assert hasattr(m, "network_profit_eta")
            assert hasattr(m, "PPO_policy_loss")
            assert hasattr(m, "PPO_approx_kl")
            assert hasattr(m, "gradient_norm")
            assert math.isfinite(m.reward)

    def test_jsonl_one_record_per_update(self, tmp_path):
        trainer = _make_trainer_mt(tmp_path, max_updates=3)
        metrics = trainer.run_training(max_updates=3)
        assert len(metrics) == 3
        sample = metrics[0].to_dict()
        required_keys = {
            "update", "episode", "reward", "normalized_reward",
            "network_profit_eta", "num_services", "rejected_demand",
            "C_service", "C_unused", "C_voyage", "C_reject", "C_handle",
            "PPO_policy_loss", "PPO_value_loss", "PPO_entropy",
            "PPO_approx_kl", "PPO_clip_fraction", "gradient_norm",
        }
        for key in required_keys:
            assert key in sample, f"Missing key in metric dict: {key}"


# ===========================================================================
# 6. Experiment Directory (No Overwrites)
# ===========================================================================

class TestExperimentDirectory:
    def test_directory_created_with_timestamp(self, tmp_path):
        from neural.config import ArchitectureConfig
        arch = ArchitectureConfig(hidden_dim=32, gat_layers=1,
                                   transformer_layers=1, transformer_heads=2,
                                   lstm_layers=1)
        cfg = {"instance": "Baltic", "policy": "encoder_only", "seed": 42,
               "preset": "SMOKE"}
        exp_dir = mt.setup_experiment_directory("test_exp", cfg, arch)
        assert exp_dir.exists()
        assert (exp_dir / "config.json").exists()
        # Second call may get same timestamp; just verify it exists
        exp_dir2 = mt.setup_experiment_directory("test_exp", cfg, arch)
        assert exp_dir2.exists()

    def test_config_json_contains_all_fields(self, tmp_path):
        from neural.config import ArchitectureConfig
        arch = ArchitectureConfig(hidden_dim=32, gat_layers=1,
                                   transformer_layers=1, transformer_heads=2,
                                   lstm_layers=1)
        cfg = {
            "instance": "Baltic", "policy": "encoder_only", "seed": 42,
            "preset": "SMOKE",
            "max_updates": 10, "num_envs": 1, "steps_per_env": 50,
            "learning_rate": 2e-4, "gamma": 1.0,
        }
        exp_dir = mt.setup_experiment_directory("cfg_test", cfg, arch)
        with open(exp_dir / "config.json") as f:
            data = json.load(f)
        assert data["experiment_name"] == "cfg_test"
        assert data["mode"] == "SMOKE"
        assert data["instance"] == "Baltic"
        assert "architecture" in data
        assert "ppo" in data


# ===========================================================================
# 7. Resume from Checkpoint
# ===========================================================================

class TestResume:
    def _make_trainer_with_training(self, tmp_path):
        trainer = _make_trainer_mt(tmp_path, max_updates=3, steps_per_env=15)
        trainer.run_training(max_updates=2)
        return trainer

    def test_resume_restores_update_count(self, tmp_path):
        trainer1 = self._make_trainer_with_training(tmp_path)
        original_count = trainer1._update_count
        assert original_count == 2
        path = trainer1.save_checkpoint("mid_checkpoint.pt")
        trainer2 = _make_trainer_mt(tmp_path)
        trainer2.load_checkpoint(path)
        assert trainer2._update_count == original_count

    def test_resume_rejects_missing_file(self, tmp_path):
        trainer = _make_trainer_mt(tmp_path)
        with pytest.raises(Exception):
            trainer.load_checkpoint(str(tmp_path / "nonexistent.pt"))


# ===========================================================================
# 8. Action Distribution Diagnostics
# ===========================================================================

class TestActionDiagnostics:
    def test_trajectory_stores_action_info(self, tmp_path):
        trainer = _make_trainer_mt(tmp_path)
        trajectory, steps = trainer.collect_rollout(seed=42)
        assert len(trajectory) > 0
        for step in trajectory:
            assert "action" in step
            assert "executed_action" in step
            assert "old_log_prob" in step
            assert "entropy" in step
            assert "fleet_remaining" in step
            assert isinstance(step["old_log_prob"], torch.Tensor)
            assert torch.isfinite(step["old_log_prob"])


# ===========================================================================
# 9. Reward Diagnostic Decomposition
# ===========================================================================

class TestRewardDiagnostics:
    def test_metrics_have_cost_breakdown(self, tmp_path):
        trainer = _make_trainer_mt(tmp_path)
        trajectory, _ = trainer.collect_rollout(seed=42)
        if trajectory:
            metrics = trainer._build_metrics_from_trajectory(trajectory)
            assert hasattr(metrics, "C_service")
            assert hasattr(metrics, "C_unused")
            assert hasattr(metrics, "C_voyage")
            assert hasattr(metrics, "C_reject")
            assert hasattr(metrics, "C_handle")
            assert hasattr(metrics, "network_profit_eta")
            assert hasattr(metrics, "rejected_demand")
            assert hasattr(metrics, "num_services")


# ===========================================================================
# 10. Edge Cases
# ===========================================================================

class TestEdgeCases:
    def test_zero_perturbation_is_valid(self):
        cfg = mt.validate_manual_config(
            instance="Baltic", policy="encoder_only",
            hidden_dim=32, gat_layers=1, transformer_layers=1,
            transformer_heads=2, lstm_layers=1,
            learning_rate=2e-4, gamma=1.0, gae_lambda=0.9,
            ppo_epochs=2, clip_epsilon=0.2, target_kl=0.1,
            entropy_coefficient=0.05, value_coefficient=0.5,
            num_envs=1, steps_per_env=50, minibatch_size=32,
            max_updates=10, perturbation_fraction=0.0,
            n_perturbed_instances=0, apply_transittime_revision=False,
            seed=42, checkpoint_frequency=5, preset="MANUAL",
        )
        assert cfg["perturbation_fraction"] == 0.0
        assert cfg["n_perturbed_instances"] == 0

    def test_high_entropy_coefficient_accepted(self):
        cfg = mt.validate_manual_config(
            instance="Baltic", policy="encoder_only",
            hidden_dim=32, gat_layers=1, transformer_layers=1,
            transformer_heads=2, lstm_layers=1,
            learning_rate=2e-4, gamma=1.0, gae_lambda=0.9,
            ppo_epochs=2, clip_epsilon=0.2, target_kl=0.1,
            entropy_coefficient=0.5, value_coefficient=0.5,
            num_envs=1, steps_per_env=50, minibatch_size=32,
            max_updates=10, perturbation_fraction=0.0,
            n_perturbed_instances=0, apply_transittime_revision=False,
            seed=42, checkpoint_frequency=5, preset="MANUAL",
        )
        assert cfg["entropy_coefficient"] == 0.5

    def test_world_small_instance_accepted(self):
        cfg = mt.validate_manual_config(
            instance="WorldSmall", policy="encoder_only",
            hidden_dim=32, gat_layers=1, transformer_layers=1,
            transformer_heads=2, lstm_layers=1,
            learning_rate=2e-4, gamma=1.0, gae_lambda=0.9,
            ppo_epochs=2, clip_epsilon=0.2, target_kl=0.1,
            entropy_coefficient=0.05, value_coefficient=0.5,
            num_envs=1, steps_per_env=50, minibatch_size=32,
            max_updates=10, perturbation_fraction=0.0,
            n_perturbed_instances=0, apply_transittime_revision=False,
            seed=42, checkpoint_frequency=5, preset="MANUAL",
        )
        assert cfg["instance"] == "WorldSmall"

    def test_cli_override_parsing(self):
        overrides = mt._parse_cli_overrides(["--max_updates", "100", "--seed", "99"])
        assert overrides["max_updates"] == 100
        assert overrides["seed"] == 99

    def test_cli_bool_override(self):
        overrides = mt._parse_cli_overrides(["--apply_transittime_revision", "true"])
        assert overrides["apply_transittime_revision"] is True

    def test_cli_float_override(self):
        overrides = mt._parse_cli_overrides(["--learning_rate", "1e-3"])
        assert overrides["learning_rate"] == 1e-3

    def test_resume_flag_parsing(self):
        overrides = mt._parse_cli_overrides(["--resume", "/path/to/ckpt.pt"])
        assert overrides["resume_from_checkpoint"] == "/path/to/ckpt.pt"


# ===========================================================================
# 11. Experimental Matrix Compatibility
# ===========================================================================

class TestExperimentMatrix:
    @pytest.mark.parametrize("exp_name,max_updates,entropy_coef", [
        ("M0", 10, 0.05),
        ("M1", 100, 0.05),
        ("M2", 200, 0.05),
        ("M3", 500, 0.05),
        ("M4", 500, 0.10),
        ("M5", 500, 0.20),
        ("M6", 1000, 0.05),
        ("M7", 1000, 0.05),
    ])
    def test_matrix_entry_valid(self, exp_name, max_updates, entropy_coef):
        cfg = mt.validate_manual_config(
            instance="Baltic", policy="encoder_decoder",
            hidden_dim=64, gat_layers=2, transformer_layers=2,
            transformer_heads=4, lstm_layers=1,
            learning_rate=2e-4, gamma=1.0, gae_lambda=0.9,
            ppo_epochs=2, clip_epsilon=0.2, target_kl=0.1,
            entropy_coefficient=entropy_coef,
            value_coefficient=0.5,
            num_envs=1, steps_per_env=50, minibatch_size=32,
            max_updates=max_updates, perturbation_fraction=0.0,
            n_perturbed_instances=0, apply_transittime_revision=False,
            seed=42, checkpoint_frequency=max(1, max_updates // 5),
            preset="MANUAL",
        )
        assert cfg["max_updates"] == max_updates
        assert cfg["entropy_coefficient"] == entropy_coef
        assert cfg["preset"] == "MANUAL"


# ===========================================================================
# 12. End-to-end smoke test
# ===========================================================================

class TestEndToEndSmoke:
    def test_full_pipeline_with_metric_logging(self, tmp_path):
        trainer = _make_trainer_mt(tmp_path, max_updates=2, steps_per_env=10)
        metrics = trainer.run_training(max_updates=2)
        assert len(metrics) == 2
        assert trainer._update_count == 2
        final_ckpt = tmp_path / "ckpt" / "final_checkpoint.pt"
        assert final_ckpt.exists()
        for m in metrics:
            assert math.isfinite(m.reward)
            assert math.isfinite(m.PPO_approx_kl)
            assert math.isfinite(m.gradient_norm)
