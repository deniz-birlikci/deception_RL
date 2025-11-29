from __future__ import annotations

"""
Training configuration using OmegaConf with dataclass support.

Provides structured, typed configuration for Secret Hitler RL training.
"""

from dataclasses import dataclass, field

from omegaconf import MISSING


@dataclass
class ModelConfig:
    """Model configuration."""

    name: str = MISSING
    """Name of the model (e.g., 'secret-hitler-v1')."""

    project: str = MISSING
    """Project name for logging and organization (e.g., 'secret-hitler')."""

    base_model: str = MISSING
    """Base model to fine-tune (e.g., 'Qwen/Qwen2.5-3B-Instruct')."""


@dataclass
class RolloutConfig:
    """Configuration for game rollouts."""

    max_turns: int = 100
    """Maximum number of turns per game before forcing termination."""

    max_retries: int = 3
    """Maximum retries for getting valid tool calls from the model."""

    simultaneous_games: int = 18
    """Number of games to run simultaneously per training step."""

    enable_thinking: bool = True
    """Enable internal thinking for Qwen models (via chat_template_kwargs)."""

    verbose: bool = False
    """Print debug information during rollouts."""

    trainable_fascist_prob: float = 0.6
    """Probability the trainable agent is assigned Fascist/Hitler role (0.0-1.0).
    Default 0.6 means 60% chance of being Fascist team, vs 40% uniform (which gives ~24% Fascist team).
    Set to 0.4 for uniform distribution (since Fascist team is 2/5 = 40% naturally)."""


@dataclass
class TrainLoopConfig:
    """Configuration for the training loop."""

    train_steps: int = 40
    """Total number of training steps to run."""

    learning_rate: float = 1e-5
    """Learning rate for training."""

    random_seed: int = 42
    """Random seed for reproducibility."""

    max_seq_length: int = 8192
    """Maximum sequence length for the model."""

    gpu_memory_utilization: float = 0.7
    """GPU memory utilization for the model."""

    tensor_parallel_size: int = 1
    """Tensor parallel size for the model."""


@dataclass
class CheckpointConfig:
    """Configuration for checkpoint handling."""

    mode: str = "latest"
    """Options: 'latest' (resume), 'scratch' (delete existing), 'specific' (resume from checkpoint.step)."""

    step: int | None = None
    """Checkpoint step to resume from when mode='specific'."""


@dataclass
class TrainingConfig:
    """Top-level training configuration."""

    model: ModelConfig = field(default_factory=ModelConfig)
    """Model configuration (name, project, base_model)."""

    rollout: RolloutConfig = field(default_factory=RolloutConfig)
    """Rollout configuration (max_turns, retries, etc)."""

    train: TrainLoopConfig = field(default_factory=TrainLoopConfig)
    """Training loop configuration (steps, learning_rate, etc)."""

    checkpoint: CheckpointConfig = field(default_factory=CheckpointConfig)
    """Checkpoint handling configuration."""
