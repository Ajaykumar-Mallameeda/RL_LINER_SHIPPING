"""
run_pipeline.py — Full end-to-end RL pipeline runner.

Runs the complete RL training + inference + evaluation pipeline and
produces a machine-readable JSON result at RL_pipeline_results.json.

Usage:
    python run_pipeline.py

This script does NOT duplicate any RL algorithm implementation. It
wires together existing engine components through validated user
configuration.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

# Ensure project root is on path.
_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from runners.config import (
    DEFAULT_CHECKPOINT_FREQUENCY,
    DEFAULT_GAT_LAYERS,
    DEFAULT_HIDDEN_DIM,
    DEFAULT_LSTM_LAYERS,
    DEFAULT_MAX_UPDATES,
    DEFAULT_POLICY,
    DEFAULT_SEED,
    DEFAULT_TRANSFORMER_HEADS,
    DEFAULT_TRANSFORMER_LAYERS,
    validate_instance,
    validate_policy,
    validate_runner_config,
)
from runners.report import (
    collect_data_hashes,
    collect_reproducibility_info,
    validate_result,
    write_json,
    _border,
    _check,
    _field,
    _section,
    VALIDATION_PASSED,
)


def main() -> int:
    """Run the full pipeline and return exit code."""
    print()
    _border()
    print("  RL LINER SHIPPING — FULL PIPELINE RUN")
    _border()

    # ------------------------------------------------------------------
    # [1] CONFIGURATION
    # ------------------------------------------------------------------
    _section("[1] CONFIGURATION")

    # Default configuration — users can override by editing this block.
    INSTANCE = "Baltic"
    POLICY = "encoder_only"
    SEED = 42
    MAX_UPDATES = 3   # Pipeline smoke: minimal training
    NUM_ENVS = 1
    STEPS_PER_ENV = 50
    MINIBATCH_SIZE = 32
    LEARNING_RATE = 2e-4
    GAMMA = 1.0
    GAE_LAMBDA = 0.9
    PPO_EPOCHS = 2
    CLIP_EPSILON = 0.2
    TARGET_KL = 0.1
    ENTROPY_COEFFICIENT = 0.05
    VALUE_COEFFICIENT = 0.5
    HIDDEN_DIM = 32
    GAT_LAYERS = 1
    TRANSFORMER_LAYERS = 1
    TRANSFORMER_HEADS = 2
    LSTM_LAYERS = 1
    CHECKPOINT_FREQUENCY = DEFAULT_CHECKPOINT_FREQUENCY

    config = validate_runner_config(
        instance=INSTANCE,
        policy=POLICY,
        seed=SEED,
        max_updates=MAX_UPDATES,
        num_envs=NUM_ENVS,
        steps_per_env=STEPS_PER_ENV,
        minibatch_size=MINIBATCH_SIZE,
        learning_rate=LEARNING_RATE,
        gamma=GAMMA,
        gae_lambda=GAE_LAMBDA,
        ppo_epochs=PPO_EPOCHS,
        clip_epsilon=CLIP_EPSILON,
        target_kl=TARGET_KL,
        entropy_coefficient=ENTROPY_COEFFICIENT,
        value_coefficient=VALUE_COEFFICIENT,
        hidden_dim=HIDDEN_DIM,
        gat_layers=GAT_LAYERS,
        transformer_layers=TRANSFORMER_LAYERS,
        transformer_heads=TRANSFORMER_HEADS,
        lstm_layers=LSTM_LAYERS,
        checkpoint_frequency=CHECKPOINT_FREQUENCY,
    )

    print("  Configuration validated successfully.")
    for k, v in sorted(config.items()):
        print(f"    {k}: {v}")

    # ------------------------------------------------------------------
    # [2] DATA FOUNDATION
    # ------------------------------------------------------------------
    _section("[2] DATA FOUNDATION")

    inst_name = config["instance"]
    from data.linerlib_loader import LINERLIBLoader
    loader = LINERLIBLoader("data")
    instance = loader.load(inst_name)

    print(f"  Instance : {instance.name}")
    print(f"  Ports    : {len(instance.ports)}")
    print(f"  Vessels  : {len(instance.vessel_types)}")
    print(f"  Demands  : {len(instance.demands)}")
    print(f"  Fleet    : {[(e.vessel_class, e.quantity) for e in instance.fleet]}")

    data_hashes = collect_data_hashes(inst_name)
    print(f"  Data files hashed: {len(data_hashes)}")
    print(f"  All hashes verified against raw files.")

    # ------------------------------------------------------------------
    # [3] RL PIPELINE
    # ------------------------------------------------------------------
    _section("[3] RL PIPELINE")

    from policies.training import LinerShippingTrainer, TrainingConfig
    from neural.config import ArchitectureConfig

    arch_cfg = ArchitectureConfig(
        hidden_dim=config["hidden_dim"],
        gat_layers=config["gat_layers"],
        transformer_layers=config["transformer_layers"],
        transformer_heads=config["transformer_heads"],
        lstm_layers=config["lstm_layers"],
    )

    training_config = TrainingConfig(
        dataset=inst_name,
        policy=config["policy"],
        learning_rate=config["learning_rate"],
        gamma=config["gamma"],
        gae_lambda=config["gae_lambda"],
        ppo_epochs=config["ppo_epochs"],
        clip_epsilon=config["clip_epsilon"],
        target_kl=config["target_kl"],
        entropy_coefficient=config["entropy_coefficient"],
        value_coefficient=config["value_coefficient"],
        num_envs=config["num_envs"],
        steps_per_env=config["steps_per_env"],
        minibatch_size=config["minibatch_size"],
        seed=config["seed"],
        max_updates=config["max_updates"],
        checkpoint_frequency=config["checkpoint_frequency"],
        hidden_dim=config["hidden_dim"],
        gat_layers=config["gat_layers"],
        transformer_layers=config["transformer_layers"],
        transformer_heads=config["transformer_heads"],
        lstm_layers=config["lstm_layers"],
    )

    print(f"  Policy     : {config['policy']}")
    print(f"  Architecture: hidden={arch_cfg.hidden_dim}, "
          f"GAT={arch_cfg.gat_layers}, "
          f"Transformer={arch_cfg.transformer_layers}x{arch_cfg.transformer_heads}, "
          f"LSTM={arch_cfg.lstm_layers}")
    print(f"  PPO config : lr={config['learning_rate']}, "
          f"gamma={config['gamma']}, "
          f"epochs={config['ppo_epochs']}, "
          f"clip={config['clip_epsilon']}")
    print(f"  Budget     : {config['max_updates']} updates, "
          f"seed={config['seed']}")

    checkpoint_dir = _ROOT / "checkpoints" / "pipeline_run"
    trainer = LinerShippingTrainer(
        instance_name=inst_name,
        policy_type=config["policy"],
        config=training_config,
        checkpoint_dir=str(checkpoint_dir),
    )

    # ------------------------------------------------------------------
    # [4] TRAINING / INFERENCE RESULT
    # ------------------------------------------------------------------
    _section("[4] TRAINING / INFERENCE RESULT")

    train_start = time.time()
    metrics = trainer.run_training(max_updates=config["max_updates"])
    train_time = time.time() - train_start

    summary = trainer.get_summary()
    print(f"\n  Training complete: {summary['updates']} updates, "
          f"{summary['episodes']} episodes")
    print(f"  Final reward  : {summary.get('final_reward', 'N/A')}")
    print(f"  Mean reward   : {summary.get('mean_reward', 'N/A'):.4f}")
    print(f"  Final KL      : {summary.get('final_kl', 'N/A'):.6f}")
    print(f"  Wall clock    : {train_time:.2f}s")

    # Save final checkpoint
    ckpt_path = trainer.save_checkpoint("final_pipeline.pt")

    # ------------------------------------------------------------------
    # [5] ECONOMIC RESULT (P15 CommonEvaluator)
    # ------------------------------------------------------------------
    _section("[5] ECONOMIC RESULT")

    from inference.solver import InferenceSolver
    from inference.config import InferenceConfig
    from evaluation import CommonEvaluator, CandidateSolution, CandidateService
    from evaluation.adapters import rl_result_to_candidate

    inf_config = InferenceConfig(
        policy_type=config["policy"],
        deterministic=True,
        max_services=10,
        seed=config["seed"],
    )
    solver = InferenceSolver(
        checkpoint_path=ckpt_path,
        config=inf_config,
        dataset_root="data",
    )
    inf_result = solver.run(instance_name=inst_name, seed=config["seed"])

    print(f"  Inference services : {inf_result.total_services}")
    print(f"  Inference eta      : {inf_result.final_eta:,.2f}")
    print(f"  Termination        : {inf_result.termination_reason}")

    # Convert to candidate and evaluate with P15
    candidate = rl_result_to_candidate(inf_result, inst_name)
    evaluator = CommonEvaluator(instance, data_hash="")
    eval_result = evaluator.evaluate(candidate)

    print(f"\n  P15 Evaluation:")
    print(f"    Objective (eta)     : {eval_result.objective_eta:,.2f}")
    print(f"    Revenue             : {eval_result.revenue:,.2f}")
    print(f"    C_service           : {eval_result.C_service:,.2f}")
    print(f"    C_unused            : {eval_result.C_unused:,.2f}")
    print(f"    C_voyage            : {eval_result.C_voyage:,.2f}")
    print(f"    C_reject            : {eval_result.C_reject:,.2f}")
    print(f"    C_handle            : {eval_result.C_handle:,.2f}")
    print(f"    Routed demand       : {eval_result.routed_demand:,.2f}")
    print(f"    Rejected demand     : {eval_result.rejected_demand:,.2f}")
    print(f"    Service count       : {eval_result.service_count}")
    print(f"    Feasible            : {eval_result.is_feasible}")
    print(f"    MCF status          : {eval_result.mcf_status}")

    # ------------------------------------------------------------------
    # [6] VALIDATION
    # ------------------------------------------------------------------
    _section("[6] VALIDATION")

    # Build result dict for validation
    last_metric = metrics[-1] if metrics else None

    result_dict = {
        "final_eta": eval_result.objective_eta,
        "revenue": eval_result.revenue,
        "C_reject": eval_result.C_reject,
        "C_handle": eval_result.C_handle,
        "C_service": eval_result.C_service,
        "C_unused": eval_result.C_unused,
        "C_voyage": eval_result.C_voyage,
        "routed_demand": eval_result.routed_demand,
        "rejected_demand": eval_result.rejected_demand,
        "total_services": inf_result.total_services,
        "policy_type": config["policy"],
        "training_updates": summary.get("updates", 0),
        "policy_loss": last_metric.PPO_policy_loss if last_metric else 0.0,
        "value_loss": last_metric.PPO_value_loss if last_metric else 0.0,
        "approx_kl": last_metric.PPO_approx_kl if last_metric else 0.0,
        "gradient_norm": last_metric.gradient_norm if last_metric else 0.0,
    }

    checks = validate_result(result_dict, check_structure=False)
    # N/A checks (result_structure when using validation from report) are not failures
    passed_checks = {k: v for k, v in checks.items() if v != "N/A"}
    all_passed = all(v == VALIDATION_PASSED for v in passed_checks.values())
    if all_passed:
        print(f"\n  ALL VALIDATION CHECKS PASSED")
    else:
        failed = [k for k, v in checks.items() if v != VALIDATION_PASSED]
        print(f"\n  VALIDATION ISSUES: {', '.join(failed)}")

    # ------------------------------------------------------------------
    # [7] REPRODUCIBILITY
    # ------------------------------------------------------------------
    _section("[7] REPRODUCIBILITY")

    repro_info = collect_reproducibility_info(config)
    repro_info["checkpoint_path"] = ckpt_path
    repro_info["checkpoint_hash"] = solver._metadata.checkpoint_hash if solver._metadata else None
    repro_info["instance_hash"] = data_hashes.get(f"Demand_{inst_name}.csv", "")

    for k, v in sorted(repro_info.items()):
        val_str = str(v)[:80] if v else "null"
        print(f"  {k}: {val_str}")

    # ------------------------------------------------------------------
    # [8] OUTPUT
    # ------------------------------------------------------------------
    _section("[8] OUTPUT")

    # Build complete result structure
    full_result = {
        "experiment": {
            "name": "full_pipeline",
            "timestamp": repro_info["timestamp"],
        },
        "configuration": {
            "instance": config["instance"],
            "policy": config["policy"],
            "seed": config["seed"],
            "max_updates": config["max_updates"],
            "num_envs": config["num_envs"],
            "steps_per_env": config["steps_per_env"],
            "minibatch_size": config["minibatch_size"],
            "architecture": arch_cfg.to_dict(),
        },
        "architecture": {
            "hidden_dim": arch_cfg.hidden_dim,
            "gat_layers": arch_cfg.gat_layers,
            "transformer_layers": arch_cfg.transformer_layers,
            "transformer_heads": arch_cfg.transformer_heads,
            "lstm_layers": arch_cfg.lstm_layers,
            "paper_frozen": arch_cfg.matches_paper(),
        },
        "training": {
            "updates": summary.get("updates", 0),
            "episodes": summary.get("episodes", 0),
            "final_reward": summary.get("final_reward"),
            "mean_reward": summary.get("mean_reward"),
            "final_profit": summary.get("final_profit"),
            "final_kl": summary.get("final_kl"),
            "wall_clock_seconds": train_time,
            "checkpoint_path": ckpt_path,
        },
        "data": {
            "instance_name": inst_name,
            "port_count": len(instance.ports),
            "vessel_type_count": len(instance.vessel_types),
            "demand_count": len(instance.demands),
            "fleet": [(e.vessel_class, e.quantity) for e in instance.fleet],
            "data_hashes": data_hashes,
        },
        "result": {
            "total_services": inf_result.total_services,
            "inference_eta": inf_result.final_eta,
            "termination_reason": inf_result.termination_reason,
            "is_truncated": inf_result.is_truncated,
            "runtime_seconds": inf_result.runtime_seconds,
        },
        "economic_metrics": {
            "objective_eta": eval_result.objective_eta,
            "revenue": eval_result.revenue,
            "C_service": eval_result.C_service,
            "C_unused": eval_result.C_unused,
            "C_voyage": eval_result.C_voyage,
            "C_reject": eval_result.C_reject,
            "C_handle": eval_result.C_handle,
            "routed_demand": eval_result.routed_demand,
            "rejected_demand": eval_result.rejected_demand,
            "service_count": eval_result.service_count,
            "fleet_usage": eval_result.fleet_usage,
            "fleet_deviation": eval_result.fleet_deviation,
        },
        "ppo_metrics": {
            "final_policy_loss": result_dict["policy_loss"],
            "final_value_loss": result_dict["value_loss"],
            "final_approx_kl": result_dict["approx_kl"],
            "final_gradient_norm": result_dict["gradient_norm"],
        },
        "validation": dict(checks),
        "runtime": {
            "training_seconds": train_time,
            "inference_seconds": inf_result.runtime_seconds,
            "evaluation_seconds": eval_result.runtime_seconds,
        },
        "reproducibility": repro_info,
        "status": "success" if all_passed else "failed_validation",
    }

    # Write JSON result
    output_path = _ROOT / "RL_pipeline_results.json"
    write_json(str(output_path), full_result)
    print(f"  Results written to: {output_path}")

    # Print services summary
    if inf_result.services:
        print(f"\n  Services constructed:")
        for svc in inf_result.services[:10]:  # Limit display
            print(f"    S{svc.service_id}: {svc.vessel_class} -> {' -> '.join(svc.port_sequence)}")
        if len(inf_result.services) > 10:
            print(f"    ... and {len(inf_result.services) - 10} more")

    _border()
    print(f"  STATUS: {'SUCCESS' if all_passed else 'VALIDATION ISSUES'}")
    _border()
    print()

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
