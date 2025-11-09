"""
Training configuration using OmegaConf with dataclass support.

Provides structured, typed configuration for 2048 RL training.
"""

from dataclasses import dataclass, field

from omegaconf import MISSING


@dataclass
class ModelConfig:
    """Model configuration."""

    name: str = MISSING
    """Name of the model (e.g., '2048-v1')."""

    project: str = MISSING
    """Project name for logging and organization (e.g., '2048')."""

    base_model: str = MISSING
    """Base model to fine-tune (e.g., 'Qwen/Qwen2.5-3B-Instruct')."""


@dataclass
class RolloutConfig:
    """Configuration for game rollouts."""

    max_turns: int = 200
    """Maximum number of moves per game before forcing termination."""

    max_retries: int = 3
    """Maximum retries for getting valid tool calls from the model."""

    simultaneous_games: int = 18
    """Number of games to run simultaneously per training step."""

    enable_ruler: bool = True
    """Enable RULER scoring for trajectory quality assessment."""

    enable_thinking: bool = True
    """Enable internal thinking for Qwen3 models (via chat_template_kwargs).
    When True, Qwen3-4B-Thinking models will generate explicit reasoning
    in <think></think> blocks before making tool calls."""

    verbose: bool = False
    """Print debug information during rollouts."""


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
    """Tensor parallel size (1 means no tensor parallelism)."""


@dataclass
class ModalConfig:
    """Configuration for Modal deployment."""

    gpu_type: str = "H100"
    """GPU type to use (e.g., 'H100', 'A100', 'T4')."""

    timeout: int = 7200
    """Timeout in seconds for the training job."""

    num_parallel_jobs: int = 1
    """Number of parallel training jobs to run (for data parallelism)."""


@dataclass
class TrainingConfig:
    """Top-level training configuration."""

    model: ModelConfig = field(default_factory=ModelConfig)
    """Model configuration (name, project, base_model)."""

    rollout: RolloutConfig = field(default_factory=RolloutConfig)
    """Rollout configuration (max_turns, retries, etc)."""

    train: TrainLoopConfig = field(default_factory=TrainLoopConfig)
    """Training loop configuration (steps, learning_rate, etc)."""

    modal: ModalConfig = field(default_factory=ModalConfig)
    """Modal deployment configuration."""

