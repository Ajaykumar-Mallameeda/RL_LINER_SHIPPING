"""
P13 — Command-line interface for inference/solver.

Provides a CLI entry point for running inference on trained checkpoints.
Usage:
    python -m inference.cli --checkpoint checkpoints/final.pt --instance Baltic
    python -m inference.cli --checkpoint checkpoints/final.pt --deterministic
    python -m inference.cli --checkpoint checkpoints/final.pt --stochastic --seed 42 --n-runs 5
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Optional

from .config import InferenceConfig
from .result import InferenceResult
from .solver import InferenceSolver

logger = logging.getLogger(__name__)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="P13 Inference Solver for LSNDP RL Engine",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--checkpoint", "-c",
        required=True,
        help="Path to training checkpoint (.pt file).",
    )
    parser.add_argument(
        "--instance", "-i",
        default=None,
        help="Instance name to run on (default: checkpoint's instance).",
    )
    parser.add_argument(
        "--policy-type", "-p",
        choices=["encoder_only", "encoder_decoder"],
        default=None,
        help="Policy type (default: inferred from checkpoint).",
    )
    parser.add_argument(
        "--deterministic", "-d",
        action="store_true",
        default=True,
        help="Use deterministic (argmax) inference.",
    )
    parser.add_argument(
        "--stochastic", "-s",
        action="store_true",
        default=False,
        help="Use stochastic (sampled) inference.",
    )
    parser.add_argument(
        "--seed", type=int,
        default=42,
        help="Random seed for stochastic inference.",
    )
    parser.add_argument(
        "--n-runs", type=int, default=1,
        help="Number of independent runs (for stochastic evaluation).",
    )
    parser.add_argument(
        "--output", "-o",
        default=None,
        help="Output path for results JSON (default: stdout).",
    )
    parser.add_argument(
        "--dataset-root",
        default="data",
        help="Root directory for LINERLIB data files.",
    )
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging verbosity.",
    )
    return parser


def main(argv: Optional[list] = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    # Determine determinism mode.
    deterministic = args.deterministic and not args.stochastic

    # Build config.
    config = InferenceConfig(
        policy_type=args.policy_type,
        deterministic=deterministic,
        seed=args.seed if not deterministic else None,
        max_services=100,
        validate_actions=True,
        check_numerical_stability=True,
        record_diagnostics=False,
    )

    # Allow override of policy type from command line.
    if args.policy_type:
        config.policy_type = args.policy_type

    # Run inference.
    try:
        solver = InferenceSolver(
            checkpoint_path=args.checkpoint,
            config=config,
            dataset_root=args.dataset_root,
        )
    except Exception as e:
        logger.error(f"Failed to initialize solver: {e}")
        return 1

    if args.n_runs == 1:
        result = solver.run(instance_name=args.instance, seed=args.seed)
        results = [result]
    else:
        results = solver.run_multiple(
            n_runs=args.n_runs,
            instance_name=args.instance,
            base_seed=args.seed,
        )

    # Output results.
    output_data = {
        "num_runs": len(results),
        "results": [r.to_dict() for r in results],
    }

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(output_data, f, indent=2, default=str)
        logger.info(f"Results written to {out_path}")
    else:
        print(json.dumps(output_data, indent=2, default=str))

    # Summary.
    for i, r in enumerate(results):
        status = "OK" if r.is_success else "FAILED"
        print(
            f"Run {i+1}: eta={r.final_eta:,.2f}, services={r.total_services}, "
            f"term={r.termination_reason}, trunc={r.is_truncated}, "
            f"runtime={r.runtime_seconds:.2f}s [{status}]"
        )
        if r.errors:
            for e in r.errors:
                print(f"  ERROR: {e}")
        if r.warnings:
            for w in r.warnings:
                print(f"  WARN: {w}")

    return 0 if all(r.is_success for r in results) else 1


if __name__ == "__main__":
    sys.exit(main())
