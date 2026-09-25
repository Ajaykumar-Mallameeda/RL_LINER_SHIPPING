"""
PPO Training Configuration.

All hyperparameters are configurable, with defaults matching the paper's
specified ranges where available. Where the paper gives options rather than
one uniquely selected value, the selected configuration is documented as
[ENGINEERING DECISION].
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass
class PPOConfig:
    """
    Configuration for PPO training.

    All values are configurable. Defaults align with paper specifications
    where stated; where the paper gives ranges or options, an explicit
    engineering decision is recorded in the docstring.

    Attributes
    ----------
    learning_rate : float
        [PAPER] Range: 1e-4 to 3e-4. Default: 2e-4 [ENGINEERING DECISION].
    gamma : float
        Discount factor. [PAPER] gamma = 1.0 (no discount).
    gae_lambda : float
        GAE lambda for advantage estimation. [PAPER] lambda = 0.9.
    ppo_epochs : int
        Number of PPO update epochs per rollout batch. [PAPER] 10.
    clip_epsilon : float
        PPO clip epsilon. [PAPER] Range: 0.15 or 0.25. Default: 0.2 [ENGINEERING DECISION].
    target_kl : float
        Target KL for early stopping. [PAPER] 0.1.
    entropy_coefficient : float
        Entropy bonus coefficient. [PAPER] Range: 0.01 or 0.1. Default: 0.05 [ENGINEERING DECISION].
    value_coefficient : float
        Value function loss coefficient. [PAPER] 0.5.
    optimizer : str
        Optimizer type. Default: "adamw" [IMPLEMENTATION].
    optimizer_kwargs : dict
        Additional optimizer parameters (e.g., weight_decay).
    num_envs : int
        Number of parallel environments. [PAPER] 8 or 16. Default: 8 [ENGINEERING DECISION].
    steps_per_env : int
        Steps per environment per update. [PAPER] 50 or 100. Default: 100 [ENGINEERING DECISION].
    minibatch_size : int
        Minibatch size for PPO updates. [PAPER] 64 or 128. Default: 64 [ENGINEERING DECISION].
    seed : Optional[int]
        Random seed for reproducibility.
    max_updates : Optional[int]
        Maximum number of PPO update iterations (None = unlimited).
    checkpoint_frequency : int
        Save checkpoint every N updates. Default: 100.
    """

    # ---- PPO core ----
    learning_rate: float = 2e-4
    gamma: float = 1.0
    gae_lambda: float = 0.9
    ppo_epochs: int = 10
    clip_epsilon: float = 0.2
    target_kl: float = 0.1
    entropy_coefficient: float = 0.05
    value_coefficient: float = 0.5

    # ---- Optimizer ----
    optimizer: str = "adamw"
    optimizer_kwargs: Dict[str, Any] = field(default_factory=lambda: {"weight_decay": 1e-4})

    # ---- Environment ----
    num_envs: int = 8
    steps_per_env: int = 100

    # ---- Minibatching ----
    minibatch_size: int = 64

    # ---- Gradient clipping ----
    max_grad_norm: float = 0.5
    """Global grad-norm clip applied to policy + critic before optimizer.step().

    Previously hard-coded as 0.5 at both call sites, which made any other value
    impossible to configure. [G11.2.4]
    """

    # ---- Reproducibility ----
    seed: Optional[int] = None

    # ---- Checkpointing ----
    max_updates: Optional[int] = None
    checkpoint_frequency: int = 100

    def validate(self) -> None:
        """Raise ValueError on invalid configuration."""
        if self.learning_rate <= 0:
            raise ValueError(f"learning_rate must be > 0, got {self.learning_rate}")
        if not 0.0 <= self.gamma <= 1.0:
            raise ValueError(f"gamma must be in [0, 1], got {self.gamma}")
        if not 0.0 <= self.gae_lambda <= 1.0:
            raise ValueError(f"gae_lambda must be in [0, 1], got {self.gae_lambda}")
        if self.ppo_epochs < 1:
            raise ValueError(f"ppo_epochs must be >= 1, got {self.ppo_epochs}")
        if not 0.0 < self.clip_epsilon <= 1.0:
            raise ValueError(f"clip_epsilon must be in (0, 1], got {self.clip_epsilon}")
        if self.target_kl <= 0:
            raise ValueError(f"target_kl must be > 0, got {self.target_kl}")
        if self.entropy_coefficient < 0:
            raise ValueError(f"entropy_coefficient must be >= 0, got {self.entropy_coefficient}")
        if self.value_coefficient < 0:
            raise ValueError(f"value_coefficient must be >= 0, got {self.value_coefficient}")
        if self.minibatch_size < 1:
            raise ValueError(f"minibatch_size must be >= 1, got {self.minibatch_size}")
        if self.max_grad_norm <= 0:
            raise ValueError(f"max_grad_norm must be > 0, got {self.max_grad_norm}")
        if self.num_envs < 1:
            raise ValueError(f"num_envs must be >= 1, got {self.num_envs}")
        if self.steps_per_env < 1:
            raise ValueError(f"steps_per_env must be >= 1, got {self.steps_per_env}")

    def to_dict(self) -> Dict[str, Any]:
        """Serialize config to dict."""
        return {
            "learning_rate": self.learning_rate,
            "gamma": self.gamma,
            "gae_lambda": self.gae_lambda,
            "ppo_epochs": self.ppo_epochs,
            "clip_epsilon": self.clip_epsilon,
            "target_kl": self.target_kl,
            "entropy_coefficient": self.entropy_coefficient,
            "value_coefficient": self.value_coefficient,
            "optimizer": self.optimizer,
            "optimizer_kwargs": self.optimizer_kwargs,
            "num_envs": self.num_envs,
            "steps_per_env": self.steps_per_env,
            "minibatch_size": self.minibatch_size,
            "max_grad_norm": self.max_grad_norm,
            "seed": self.seed,
            "max_updates": self.max_updates,
            "checkpoint_frequency": self.checkpoint_frequency,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PPOConfig":
        """Deserialize config from dict."""
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


class PaperPPOConfig(PPOConfig):
    """
    Paper-faithful configuration with engineering decisions documented.

    This class provides a named constant for the paper's configuration,
    making it clear which values come from the paper vs. engineering decisions.
    """

    def __init__(self) -> None:
        """Initialize with paper-specified defaults."""
        super().__init__()
        # Override with paper-specified values where available.
        # Note: Some values have engineering decisions due to paper ambiguity.
