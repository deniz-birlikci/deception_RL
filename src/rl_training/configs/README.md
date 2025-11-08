# Training Configuration

This directory contains YAML configuration files for Secret Hitler RL training.

## Usage

### Basic Usage

Run with default config:
```bash
modal run src/rl_training/train_modal.py
```

Run with custom config:
```bash
modal run src/rl_training/train_modal.py --config-path configs/train_small.yaml
```

### Config Structure

Configs use OmegaConf with structured dataclasses for type safety and validation.

#### Access Patterns

In Python code, access nested configs using dot notation:

```python
# Access rollout settings
max_turns = config.rollout.max_turns
enable_thinking = config.rollout.enable_thinking
simultaneous_games = config.rollout.simultaneous_games

# Access model settings
model_name = config.model.name
base_model = config.model.base_model

# Access training settings
learning_rate = config.train.learning_rate
train_steps = config.train.train_steps
```

### Config Sections

#### `model`
Model configuration (name, project, base model)
- `name`: Model identifier (e.g., "secret-hitler-v1")
- `project`: Project name for logging
- `base_model`: HuggingFace model to fine-tune

#### `rollout`
Game rollout configuration
- `max_turns`: Max turns per game before forced termination
- `max_retries`: Max retries for getting valid tool calls
- `simultaneous_games`: Number of parallel games per training step
- `enable_thinking`: Enable Qwen internal thinking
- `verbose`: Print debug information

#### `train`
Training loop configuration
- `train_steps`: Total number of training steps
- `learning_rate`: Learning rate for training
- `random_seed`: Random seed for reproducibility

## Example Configs

### `train_default.yaml`
Production config with full settings:
- 40 training steps
- 18 simultaneous games
- Qwen/Qwen2.5-3B-Instruct base model

### `train_small.yaml`
Test config for quick iteration:
- 5 training steps
- 4 simultaneous games
- Verbose mode enabled
- Shorter max turns (50 vs 100)

## Creating Custom Configs

1. Copy an existing config:
   ```bash
   cp configs/train_default.yaml configs/my_experiment.yaml
   ```

2. Edit the YAML file with your settings

3. Run with your config:
   ```bash
   modal run src/rl_training/train_modal.py --config-path configs/my_experiment.yaml
   ```

## Validation

Configs are validated on load:
- Required fields must be provided (model.name, model.project, model.base_model)
- Type checking ensures correct types for all fields
- Invalid configs will raise clear error messages

## Configuration in Code

The config system uses dataclasses defined in `src/rl_training/config.py`:

```python
from omegaconf import OmegaConf
from src.rl_training.config import TrainingConfig

# Load and merge
yaml_config = OmegaConf.load("configs/train_default.yaml")
config = OmegaConf.merge(OmegaConf.structured(TrainingConfig), yaml_config)

# Access nested values
print(config.rollout.max_turns)  # 100
print(config.model.name)  # "secret-hitler-v1"

# Pass to rollout function
rollout(
    model,
    step,
    max_turns=config.rollout.max_turns,
    enable_thinking=config.rollout.enable_thinking,
)
```

