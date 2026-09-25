"""
P13 — Checkpoint Loading and Compatibility Validation.

Loads a P10/P12 training checkpoint and validates that its architecture
matches what is expected for inference. Fails loudly on any mismatch.

Checkpoint format (produced by P12 LinerShippingTrainer.save_checkpoint):
{
    "instance_name": str,
    "policy_type": str,              # "encoder_only" or "encoder_decoder"
    "config": { ... },               # TrainingConfig as dict
    "update_count": int,
    "episode_count": int,
    "metrics_log": [...],
    "raw_data_hashes": {...},
    "timestamp": str,
    "backbone_state_dict": {...},
    "policy_state_dict": {...},
    "critic_state_dict": {...},
    "optimizer_state_dict": {...},
}

The checkpoint's architecture metadata is cross-checked against the
requested inference configuration to prevent silent mismatches.
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch

from neural.config import ArchitectureConfig

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Checkpoint validation errors
# ---------------------------------------------------------------------------

class CheckpointError(Exception):
    """Raised when a checkpoint cannot be loaded or validated."""

    def __init__(self, message: str, details: Optional[List[str]] = None):
        self.message = message
        self.details = details or []
        msg_lines = [message]
        for d in self.details:
            msg_lines.append(f"  - {d}")
        super().__init__("\n".join(msg_lines))


# ---------------------------------------------------------------------------
# Checkpoint metadata
# ---------------------------------------------------------------------------

@dataclass
class CheckpointMetadata:
    """Machine-readable metadata extracted from a checkpoint file."""

    instance_name: str
    policy_type: str
    update_count: int
    episode_count: int
    timestamp: str
    raw_data_hashes: Dict[str, str]
    backbone_config: Dict[str, Any]
    backbone_params: int
    policy_params: int
    critic_params: int
    total_params: int
    checkpoint_path: str
    checkpoint_hash: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "instance_name": self.instance_name,
            "policy_type": self.policy_type,
            "update_count": self.update_count,
            "episode_count": self.episode_count,
            "timestamp": self.timestamp,
            "raw_data_hashes": self.raw_data_hashes,
            "backbone_config": self.backbone_config,
            "backbone_params": self.backbone_params,
            "policy_params": self.policy_params,
            "critic_params": self.critic_params,
            "total_params": self.total_params,
            "checkpoint_path": self.checkpoint_path,
            "checkpoint_hash": self.checkpoint_hash,
        }


# ---------------------------------------------------------------------------
# Checkpoint loading
# ---------------------------------------------------------------------------

def compute_checkpoint_hash(path: str) -> str:
    """Compute SHA-256 hash of a checkpoint file for traceability."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def load_checkpoint(path: str) -> Dict[str, Any]:
    """
    Load a checkpoint file and return its payload dict.

    Parameters
    ----------
    path : str
        Path to the .pt checkpoint file.

    Returns
    -------
    dict
        The checkpoint payload.

    Raises
    ------
    CheckpointError
        If the file is missing, corrupted, or malformed.
    """
    p = Path(path)
    if not p.exists():
        raise CheckpointError(
            f"Checkpoint file not found: {path}",
            details=["Use a valid path to an existing .pt file."],
        )

    try:
        payload = torch.load(p, map_location="cpu", weights_only=False)
    except Exception as e:
        raise CheckpointError(
            f"Failed to load checkpoint: {path}",
            details=[f"PyTorch loading error: {type(e).__name__}: {e}"],
        )

    if not isinstance(payload, dict):
        raise CheckpointError(
            f"Malformed checkpoint: expected dict, got {type(payload).__name__}",
        )

    return payload


def validate_checkpoint_structure(
    payload: Dict[str, Any],
    required_keys: List[str],
) -> None:
    """
    Validate that a checkpoint payload contains all required keys.

    Parameters
    ----------
    payload : dict
        The loaded checkpoint payload.
    required_keys : list[str]
        Keys that must be present.
    """
    missing = [k for k in required_keys if k not in payload]
    if missing:
        raise CheckpointError(
            f"Checkpoint missing required keys: {missing}",
            details=[f"Available keys: {list(payload.keys())}"],
        )


# ---------------------------------------------------------------------------
# Architecture compatibility checking
# ---------------------------------------------------------------------------

def extract_architecture_from_checkpoint(
    payload: Dict[str, Any],
) -> Tuple[ArchitectureConfig, Dict[str, int], str]:
    """
    Extract architecture config and parameter counts from checkpoint.

    Returns
    -------
    config : ArchitectureConfig
        The architecture configuration stored in the checkpoint.
    param_counts : dict
        Parameter counts per component (backbone, policy, critic).
    policy_type : str
        Policy type stored in the checkpoint.
    """
    policy_type = payload.get("policy_type", "unknown")

    # Try to reconstruct config from the training config stored in checkpoint.
    training_config = payload.get("config", {})
    hidden_dim = training_config.get("hidden_dim", 512)
    gat_layers = training_config.get("gat_layers", 3)
    transformer_layers = training_config.get("transformer_layers", 3)
    transformer_heads = training_config.get("transformer_heads", 8)
    lstm_layers = training_config.get("lstm_layers", 1)

    config = ArchitectureConfig(
        hidden_dim=hidden_dim,
        gat_layers=gat_layers,
        transformer_layers=transformer_layers,
        transformer_heads=transformer_heads,
        lstm_layers=lstm_layers,
    )

    # Count parameters.
    backbone_params = sum(
        v.numel() for k, v in payload.get("backbone_state_dict", {}).items()
    )
    policy_params = sum(
        v.numel() for k, v in payload.get("policy_state_dict", {}).items()
    )
    critic_params = sum(
        v.numel() for k, v in payload.get("critic_state_dict", {}).items()
    )
    total = backbone_params + policy_params + critic_params

    param_counts = {
        "backbone": backbone_params,
        "policy": policy_params,
        "critic": critic_params,
        "total": total,
    }

    return config, param_counts, policy_type


def compare_architecture(
    checkpoint_config: ArchitectureConfig,
    requested_config: Optional[ArchitectureConfig],
    expected_policy_type: str,
    actual_policy_type: str,
) -> List[str]:
    """
    Compare checkpoint architecture against requested/inferred expectations.

    Returns a list of incompatibility messages (empty = compatible).
    """
    issues: List[str] = []

    # Policy type check.
    if actual_policy_type != expected_policy_type:
        issues.append(
            f"Policy type mismatch: checkpoint has '{actual_policy_type}', "
            f"expected '{expected_policy_type}'."
        )

    # Architecture config comparison.
    if requested_config is not None:
        checks = [
            ("hidden_dim", checkpoint_config.hidden_dim, requested_config.hidden_dim),
            ("gat_layers", checkpoint_config.gat_layers, requested_config.gat_layers),
            ("transformer_layers", checkpoint_config.transformer_layers,
             requested_config.transformer_layers),
            ("transformer_heads", checkpoint_config.transformer_heads,
             requested_config.transformer_heads),
            ("lstm_layers", checkpoint_config.lstm_layers, requested_config.lstm_layers),
        ]
        for name, ckpt_val, req_val in checks:
            if ckpt_val != req_val:
                issues.append(
                    f"Architecture mismatch on '{name}': checkpoint={ckpt_val}, "
                    f"requested={req_val}."
                )

    return issues


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_and_validate_checkpoint(
    path: str,
    expected_policy_type: str,
    expected_config: Optional[ArchitectureConfig] = None,
) -> Tuple[Dict[str, Any], CheckpointMetadata]:
    """
    Load and fully validate a training checkpoint.

    This is the canonical entry point for checkpoint loading in P13. It:
    1. Loads the checkpoint file.
    2. Validates required keys are present.
    3. Extracts architecture metadata.
    4. Compares architecture against expectations.
    5. Computes file hash for traceability.
    6. Returns the payload and metadata.

    Parameters
    ----------
    path : str
        Path to the checkpoint .pt file.
    expected_policy_type : str
        Expected policy type ("encoder_only" or "encoder_decoder").
    expected_config : ArchitectureConfig, optional
        Expected architecture config. If None, validation is relaxed.

    Returns
    -------
    payload : dict
        The loaded checkpoint payload.
    metadata : CheckpointMetadata
        Structured metadata about the checkpoint.

    Raises
    ------
    CheckpointError
        If any validation step fails.
    """
    # 1. Load.
    payload = load_checkpoint(path)

    # 2. Validate structure.
    required_keys = [
        "backbone_state_dict",
        "policy_state_dict",
        "critic_state_dict",
        "instance_name",
        "policy_type",
        "config",
    ]
    validate_checkpoint_structure(payload, required_keys)

    # 3. Extract architecture.
    arch_config, param_counts, actual_policy_type = (
        extract_architecture_from_checkpoint(payload)
    )

    # 4. Compare against expectations.
    issues = compare_architecture(
        arch_config, expected_config, expected_policy_type, actual_policy_type,
    )
    if issues:
        raise CheckpointError(
            "Checkpoint architecture incompatible with requested configuration.",
            details=issues,
        )

    # 5. Compute hash.
    ckpt_hash = compute_checkpoint_hash(path)

    # 6. Build metadata.
    metadata = CheckpointMetadata(
        instance_name=payload["instance_name"],
        policy_type=actual_policy_type,
        update_count=payload.get("update_count", 0),
        episode_count=payload.get("episode_count", 0),
        timestamp=payload.get("timestamp", "unknown"),
        raw_data_hashes=payload.get("raw_data_hashes", {}),
        backbone_config=arch_config.to_dict(),
        backbone_params=param_counts["backbone"],
        policy_params=param_counts["policy"],
        critic_params=param_counts["critic"],
        total_params=param_counts["total"],
        checkpoint_path=path,
        checkpoint_hash=ckpt_hash,
    )

    logger.info(
        f"Checkpoint loaded: {path} | policy={actual_policy_type} | "
        f"params={param_counts['total']:,} | updates={payload.get('update_count', 0)}"
    )

    return payload, metadata
