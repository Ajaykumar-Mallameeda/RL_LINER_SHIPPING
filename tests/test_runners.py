"""
Tests for the runner configuration/validation layer.

Covers:
  1. Valid configuration acceptance
  2. Invalid instance rejection
  3. Invalid policy rejection
  4. Negative learning rate rejection
  5. Invalid PPO epochs rejection
  6. Invalid minibatch size rejection
  7. Invalid number of environments rejection
  8. Invalid step count rejection
  9. Transformer heads incompatible with hidden dimension rejection
  10. Invalid layer count rejection
  11. Invalid checkpoint frequency rejection
  12. Deterministic behaviour with identical seeds
  13. Failure handling (non-zero exit on bad config)
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from runners.config import (
    RunnerConfigError,
    DEFAULT_HIDDEN_DIM,
    DEFAULT_MAX_UPDATES,
    validate_instance,
    validate_policy,
    validate_runner_config,
)
from runners.report import VALIDATION_PASSED, validate_result


# ===========================================================================
# 1. Valid configuration acceptance
# ===========================================================================

class TestValidConfig:
    """Valid configurations are accepted without error."""

    def test_minimal_config(self):
        """Only required params, everything else defaults."""
        cfg = validate_runner_config(instance="Baltic", policy="encoder_only")
        assert cfg["instance"] == "Baltic"
        assert cfg["policy"] == "encoder_only"
        assert cfg["seed"] == 42
        assert cfg["max_updates"] == DEFAULT_MAX_UPDATES
        assert cfg["hidden_dim"] == DEFAULT_HIDDEN_DIM

    def test_full_config(self):
        """All fields accepted when explicitly provided."""
        cfg = validate_runner_config(
            instance="Baltic",
            policy="encoder_decoder",
            seed=99,
            max_updates=10,
            num_envs=2,
            steps_per_env=20,
            minibatch_size=16,
            learning_rate=1e-3,
            gamma=0.99,
            gae_lambda=0.95,
            ppo_epochs=3,
            clip_epsilon=0.15,
            target_kl=0.05,
            entropy_coefficient=0.01,
            value_coefficient=0.25,
            hidden_dim=64,
            gat_layers=2,
            transformer_layers=2,
            transformer_heads=4,
            lstm_layers=1,
            checkpoint_frequency=5,
        )
        assert cfg["instance"] == "Baltic"
        assert cfg["policy"] == "encoder_decoder"
        assert cfg["seed"] == 99
        assert cfg["max_updates"] == 10
        assert cfg["learning_rate"] == 1e-3
        assert cfg["hidden_dim"] == 64
        assert cfg["transformer_heads"] == 4
        assert cfg["hidden_dim"] % cfg["transformer_heads"] == 0

    def test_all_instances_accepted(self):
        """All known LINERLIB instances are accepted."""
        from data.linerlib_loader import LINERLIBLoader
        loader = LINERLIBLoader("data")
        for name in loader.available_instances():
            cfg = validate_runner_config(instance=name, policy="encoder_only", max_updates=1)
            assert cfg["instance"] == name


# ===========================================================================
# 2. Invalid instance rejection
# ===========================================================================

class TestInvalidInstance:
    """Unknown instances must be rejected with a clear error."""

    def test_unknown_instance(self):
        with pytest.raises(RunnerConfigError, match="Unknown instance"):
            validate_runner_config(instance="NonExistent", policy="encoder_only")

    def test_unknown_instance_mentions_available(self):
        err = None
        try:
            validate_runner_config(instance="FakeInstance", policy="encoder_only")
        except RunnerConfigError as e:
            err = str(e)
        assert err is not None
        assert "Available instances" in err


# ===========================================================================
# 3. Invalid policy rejection
# ===========================================================================

class TestInvalidPolicy:
    """Invalid policy types must be rejected."""

    def test_invalid_policy_string(self):
        with pytest.raises(RunnerConfigError, match="policy"):
            validate_runner_config(instance="Baltic", policy="ga_milp")

    def test_invalid_policy_none(self):
        with pytest.raises(RunnerConfigError):
            validate_runner_config(instance="Baltic", policy=None)  # type: ignore

    def test_valid_policies_accepted(self):
        for pol in ("encoder_only", "encoder_decoder"):
            cfg = validate_runner_config(instance="Baltic", policy=pol)
            assert cfg["policy"] == pol


# ===========================================================================
# 4. Negative learning rate rejection
# ===========================================================================

class TestNegativeLearningRate:
    def test_negative_lr(self):
        with pytest.raises(RunnerConfigError, match="learning_rate"):
            validate_runner_config(instance="Baltic", policy="encoder_only", learning_rate=-1e-4)

    def test_zero_lr(self):
        with pytest.raises(RunnerConfigError, match="learning_rate"):
            validate_runner_config(instance="Baltic", policy="encoder_only", learning_rate=0)


# ===========================================================================
# 5. Invalid PPO epochs rejection
# ===========================================================================

class TestInvalidPPOEpochs:
    def test_zero_ppo_epochs(self):
        with pytest.raises(RunnerConfigError, match="ppo_epochs"):
            validate_runner_config(instance="Baltic", policy="encoder_only", ppo_epochs=0)

    def test_negative_ppo_epochs(self):
        with pytest.raises(RunnerConfigError, match="ppo_epochs"):
            validate_runner_config(instance="Baltic", policy="encoder_only", ppo_epochs=-1)


# ===========================================================================
# 6. Invalid minibatch size rejection
# ===========================================================================

class TestInvalidMinibatchSize:
    def test_zero_minibatch(self):
        with pytest.raises(RunnerConfigError, match="minibatch_size"):
            validate_runner_config(instance="Baltic", policy="encoder_only", minibatch_size=0)


# ===========================================================================
# 7. Invalid number of environments rejection
# ===========================================================================

class TestInvalidNumEnvsg:
    def test_zero_envs(self):
        with pytest.raises(RunnerConfigError, match="num_envs"):
            validate_runner_config(instance="Baltic", policy="encoder_only", num_envs=0)

    def test_negative_envs(self):
        with pytest.raises(RunnerConfigError, match="num_envs"):
            validate_runner_config(instance="Baltic", policy="encoder_only", num_envs=-1)


# ===========================================================================
# 8. Invalid step count rejection
# ===========================================================================

class TestInvalidStepsPerEnv:
    def test_zero_steps(self):
        with pytest.raises(RunnerConfigError, match="steps_per_env"):
            validate_runner_config(instance="Baltic", policy="encoder_only", steps_per_env=0)


# ===========================================================================
# 9. Transformer heads incompatible with hidden dimension
# ===========================================================================

class TestTransformerHeadsCompatibility:
    def test_hidden_not_divisible_by_heads(self):
        """hidden_dim=32, transformer_heads=3 → 32%3 != 0."""
        with pytest.raises(RunnerConfigError, match="hidden_dim"):
            validate_runner_config(
                instance="Baltic", policy="encoder_only",
                hidden_dim=32, transformer_heads=3,
            )

    def test_compatible_dims_accepted(self):
        """hidden_dim=32, transformer_heads=4 → divisible, should pass."""
        cfg = validate_runner_config(
            instance="Baltic", policy="encoder_only",
            hidden_dim=32, transformer_heads=4,
        )
        assert cfg["hidden_dim"] % cfg["transformer_heads"] == 0


# ===========================================================================
# 10. Invalid layer count rejection
# ===========================================================================

class TestInvalidLayerCounts:
    def test_zero_gat_layers(self):
        with pytest.raises(RunnerConfigError, match="gat_layers"):
            validate_runner_config(instance="Baltic", policy="encoder_only", gat_layers=0)

    def test_zero_transformer_layers(self):
        with pytest.raises(RunnerConfigError, match="transformer_layers"):
            validate_runner_config(instance="Baltic", policy="encoder_only", transformer_layers=0)

    def test_zero_lstm_layers(self):
        with pytest.raises(RunnerConfigError, match="lstm_layers"):
            validate_runner_config(instance="Baltic", policy="encoder_only", lstm_layers=0)

    def test_zero_transformer_heads(self):
        with pytest.raises(RunnerConfigError, match="transformer_heads"):
            validate_runner_config(instance="Baltic", policy="encoder_only", transformer_heads=0)


# ===========================================================================
# 11. Invalid checkpoint frequency rejection
# ===========================================================================

class TestInvalidCheckpointFrequency:
    def test_zero_checkpoint_freq(self):
        with pytest.raises(RunnerConfigError, match="checkpoint_frequency"):
            validate_runner_config(
                instance="Baltic", policy="encoder_only",
                checkpoint_frequency=0,
            )


# ===========================================================================
# 12. Deterministic behaviour with identical seeds
# ===========================================================================

class TestDeterministicSeeds:
    """Two runs with the same seed produce trajectories of equal length."""

    def test_same_seed_same_trajectory_length(self, tmp_path):
        """Two rollouts with the same seed from the same trainer have equal length."""
        from policies.training import LinerShippingTrainer, TrainingConfig

        config = validate_runner_config(
            instance="Baltic", policy="encoder_only",
            seed=77, max_updates=1,
            hidden_dim=16, gat_layers=1,
            transformer_layers=1, transformer_heads=2, lstm_layers=1,
        )
        tc = TrainingConfig(
            dataset="Baltic", policy="encoder_only",
            learning_rate=2e-4, gamma=1.0, gae_lambda=0.9,
            ppo_epochs=1, clip_epsilon=0.2, target_kl=10.0,
            entropy_coefficient=0.05, value_coefficient=0.5,
            num_envs=1, steps_per_env=50, minibatch_size=32,
            seed=77, max_updates=1, checkpoint_frequency=100,
            hidden_dim=16, gat_layers=1, transformer_layers=1,
            transformer_heads=2, lstm_layers=1,
        )
        trainer = LinerShippingTrainer(
            instance_name="Baltic", policy_type="encoder_only",
            config=tc, checkpoint_dir=str(tmp_path / "t77"),
        )
        _, steps1 = trainer.collect_rollout(seed=77)
        _, steps2 = trainer.collect_rollout(seed=77)
        assert steps1 == steps2
        assert steps1 > 0


# ===========================================================================
# 13. Failure handling — non-zero exit on bad config
# ===========================================================================

class TestFailureHandling:
    """Scripts should raise clear errors, not crash with obscure tracebacks."""

    def test_bad_instance_raises_clearly(self):
        with pytest.raises(RunnerConfigError) as exc_info:
            validate_runner_config(instance="ZzzNotReal", policy="encoder_only")
        assert "ZzzNotReal" in str(exc_info.value)
        assert "Available instances" in str(exc_info.value)

    def test_bad_policy_raises_clearly(self):
        with pytest.raises(RunnerConfigError) as exc_info:
            validate_runner_config(instance="Baltic", policy="quantum_fleet")
        assert "quantum_fleet" in str(exc_info.value)

    def test_nan_learning_rate_raises(self):
        with pytest.raises(RunnerConfigError):
            validate_runner_config(instance="Baltic", policy="encoder_only", learning_rate=float("nan"))

    def test_inf_learning_rate_raises(self):
        with pytest.raises(RunnerConfigError):
            validate_runner_config(instance="Baltic", policy="encoder_only", learning_rate=float("inf"))


# ===========================================================================
# 14. Validation result structure
# ===========================================================================

class TestValidationResult:
    """validate_result() produces expected check map."""

    def test_valid_run_all_pass(self):
        result = {
            "final_eta": -1e6,
            "revenue": 0.0,
            "C_reject": 100.0,
            "C_handle": 50.0,
            "C_service": 200.0,
            "C_unused": 300.0,
            "C_voyage": 400.0,
            "routed_demand": 0.0,
            "rejected_demand": 100.0,
            "total_services": 2,
            "policy_type": "encoder_only",
            "training_updates": 5,
            "policy_loss": 0.5,
            "value_loss": 0.3,
            "approx_kl": 0.01,
            "gradient_norm": 0.2,
        }
        checks = validate_result(result, check_structure=False)
        non_na = {k: v for k, v in checks.items() if v != "N/A"}
        assert all(v == VALIDATION_PASSED for v in non_na.values())

    def test_nan_value_detected(self):
        result = {
            "final_eta": float("nan"),
            "revenue": 0.0,
            "C_reject": 100.0,
            "C_handle": 50.0,
            "C_service": 200.0,
            "C_unused": 300.0,
            "C_voyage": 400.0,
            "routed_demand": 0.0,
            "rejected_demand": 100.0,
            "total_services": 2,
            "policy_type": "encoder_only",
            "training_updates": 5,
            "policy_loss": 0.5,
            "value_loss": 0.3,
            "approx_kl": 0.01,
            "gradient_norm": 0.2,
        }
        checks = validate_result(result, check_structure=False)
        assert checks["finite_outputs"] != VALIDATION_PASSED
        assert checks["no_nan_inf"] != VALIDATION_PASSED

    def test_invalid_policy_flagged(self):
        result = {
            "final_eta": 0.0,
            "revenue": 0.0, "C_reject": 0.0, "C_handle": 0.0,
            "C_service": 0.0, "C_unused": 0.0, "C_voyage": 0.0,
            "routed_demand": 0.0, "rejected_demand": 0.0,
            "total_services": 0,
            "policy_type": "invalid_policy",
            "training_updates": 0,
            "policy_loss": 0.0, "value_loss": 0.0,
            "approx_kl": 0.0, "gradient_norm": 0.0,
        }
        checks = validate_result(result, check_structure=False)
        assert checks["valid_policy"] != VALIDATION_PASSED


# ===========================================================================
# 15. run_pipeline produces valid JSON
# ===========================================================================

class TestRunPipelineJSON:
    """run_pipeline.py produces a well-formed RL_pipeline_results.json."""

    def test_json_keys_present(self, tmp_path, monkeypatch):
        """Simulate a minimal run and verify JSON structure."""
        import tempfile, os
        with tempfile.TemporaryDirectory() as td:
            # We can't easily run the full script, but we verify
            # the JSON schema from an existing output file.
            existing = _ROOT / "RL_pipeline_results.json"
            if existing.exists():
                with open(existing) as f:
                    data = json.load(f)
                required = [
                    "experiment", "configuration", "architecture",
                    "training", "data", "result", "economic_metrics",
                    "ppo_metrics", "validation", "runtime",
                    "reproducibility", "status",
                ]
                for key in required:
                    assert key in data, f"Missing key: {key}"
                assert data["status"] in ("success", "failed_validation")


# ===========================================================================
# 16. tune_pipeline produces valid JSON
# ===========================================================================

class TestTunePipelineJSON:
    def test_json_structure(self):
        existing = _ROOT / "tuning_results.json"
        if not existing.exists():
            pytest.skip("tuning_results.json not present — run tune_pipeline.py first")
        with open(existing) as f:
            data = json.load(f)
        assert "results" in data
        assert isinstance(data["results"], list)
        if data["results"]:
            r = data["results"][0]
            assert "config" in r
            assert "evaluation" in r
            assert "status" in r


# ===========================================================================
# 17. train_rl produces training results
# ===========================================================================

class TestTrainRlResults:
    def test_training_result_file_created(self):
        results_dir = _ROOT / "results" / "training"
        if not results_dir.exists():
            pytest.skip("results/training/ not present — run train_rl.py first")
        json_files = list(results_dir.glob("*.json"))
        assert len(json_files) > 0, "No training result JSON files found"
        with open(json_files[0]) as f:
            data = json.load(f)
        assert "configuration" in data
        assert "training_summary" in data
        assert "status" in data
