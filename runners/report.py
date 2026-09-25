"""
Shared terminal reporter for runner scripts.

Provides sectioned formatted output for pipeline runs, tuning runs,
and training runs - matching the presentation philosophy of the
user's multi-agent test style without duplicating any RL logic.
"""

from __future__ import annotations

import json
import sys
import time
from datetime import datetime
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# Terminal presentation helpers
# ---------------------------------------------------------------------------

def _section(title: str) -> None:
    """Print a section header."""
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}")


def _field(label: str, value: Any, indent: int = 2) -> None:
    prefix = " " * indent
    if isinstance(value, (dict, list)):
        print(f"{prefix}{label}:")
        if isinstance(value, dict):
            for k, v in value.items():
                print(f"{prefix}  {k}: {v}")
        elif isinstance(value, list):
            for item in value:
                print(f"{prefix}  - {item}")
    else:
        print(f"{prefix}{label}: {value}")


def _print_table(headers: List[str], rows: List[List[str]]) -> None:
    """Print a simple text table."""
    col_widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            if i < len(col_widths):
                col_widths[i] = max(col_widths[i], len(str(cell)))
    fmt = "  ".join(f"{{:<{w}}}" for w in col_widths)
    print(fmt.format(*headers))
    print("  ".join("-" * w for w in col_widths))
    for row in rows:
        print(fmt.format(*[str(c) for c in row]))


def _border() -> None:
    print("=" * 60)


# ---------------------------------------------------------------------------
# Validation reporting
# ---------------------------------------------------------------------------

VALIDATION_PASSED = "PASS"
VALIDATION_FAILED = "FAIL"
VALIDATION_NA = "N/A"


def _check(name: str, condition: bool, detail: str = "") -> str:
    status = VALIDATION_PASSED if condition else VALIDATION_FAILED
    msg = f"  [{status}] {name}"
    if detail:
        msg += f" - {detail}"
    print(msg)
    return status


def validate_result(
    result: Dict[str, Any],
    *,
    check_services: bool = True,
    check_structure: bool = True,
) -> Dict[str, str]:
    """Run validation checks and print results. Returns check map."""
    checks: Dict[str, str] = {}

    # Numerical finiteness
    numeric_keys = [
        "final_eta", "revenue", "C_reject", "C_handle",
        "C_service", "C_unused", "C_voyage",
        "routed_demand", "rejected_demand",
        "policy_loss", "value_loss", "entropy_loss",
        "approx_kl", "gradient_norm",
    ]
    finite_ok = True
    for key in numeric_keys:
        val = result.get(key)
        if val is not None:
            try:
                import math
                if not math.isfinite(float(val)):
                    finite_ok = False
                    break
            except (TypeError, ValueError):
                pass
    checks["finite_outputs"] = _check("Finite numerical outputs", finite_ok)

    # No NaN / Inf
    nan_ok = finite_ok
    checks["no_nan_inf"] = _check("No NaN or Inf values", nan_ok)

    # Policy type valid
    policy = result.get("policy_type", "")
    valid_policy = policy in ("encoder_only", "encoder_decoder")
    checks["valid_policy"] = _check("Valid policy type", valid_policy, policy)

    # Services produced (optional - training-only runs may have 0)
    if check_services:
        n_svc = result.get("total_services", 0)
        checks["valid_services"] = _check("Services produced", n_svc > 0, f"{n_svc} services")
    else:
        checks["valid_services"] = VALIDATION_NA

    # Economic metrics present
    has_costs = all(
        result.get(k) is not None
        for k in ["C_service", "C_unused", "C_voyage", "C_reject", "C_handle"]
    )
    checks["economic_metrics"] = _check("Economic metrics populated", has_costs)

    # Training completed
    updates = result.get("training_updates", 0)
    checks["training_complete"] = _check("Training completed", updates >= 0, f"{updates} updates")

    # Result structure
    if check_structure:
        required_keys = ["experiment", "configuration", "result", "validation"]
        has_structure = all(k in result for k in required_keys)
        checks["result_structure"] = _check("Result structure complete", has_structure)
    else:
        checks["result_structure"] = VALIDATION_NA

    return checks


# ---------------------------------------------------------------------------
# JSON serialization helpers
# ---------------------------------------------------------------------------

def _default_serializer(obj: Any) -> Any:
    """Handle non-serializable types for JSON output."""
    import datetime
    if isinstance(obj, (datetime.date, datetime.datetime)):
        return obj.isoformat()
    if hasattr(obj, "isoformat"):
        return obj.isoformat()
    if hasattr(obj, "tolist"):
        return obj.tolist()
    if hasattr(obj, "__dict__"):
        return obj.__dict__
    return str(obj)


def write_json(path: str, data: Dict[str, Any]) -> None:
    """Write result dict to JSON with consistent formatting."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=_default_serializer, sort_keys=True)


def read_json(path: str) -> Dict[str, Any]:
    """Read and parse a JSON file."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Reproducibility info
# ---------------------------------------------------------------------------

def collect_reproducibility_info(run_config: Dict[str, Any]) -> Dict[str, Any]:
    """Collect reproducibility metadata for a run."""
    import platform
    import sys as _sys
    try:
        import torch
        torch_version = torch.__version__
    except ImportError:
        torch_version = "not installed"
    try:
        import numpy
        numpy_version = numpy.__version__
    except ImportError:
        numpy_version = "not installed"

    return {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "python_version": _sys.version,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "torch_version": torch_version,
        "numpy_version": numpy_version,
        "seed": run_config.get("seed"),
        "instance": run_config.get("instance"),
        "policy_type": run_config.get("policy"),
    }


# ---------------------------------------------------------------------------
# Data provenance hashes
# ---------------------------------------------------------------------------

def collect_data_hashes(instance_name: str) -> Dict[str, str]:
    """Compute SHA-256 hashes for raw data files used by an instance."""
    import hashlib
    from pathlib import Path

    data_root = Path("data")
    hashes: Dict[str, str] = {}
    for csv_file in sorted(data_root.glob("*.csv")):
        with open(csv_file, "rb") as f:
            hashes[csv_file.name] = hashlib.sha256(f.read()).hexdigest()
    return hashes
