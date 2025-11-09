import asyncio
import random
from pathlib import Path

import modal
from omegaconf import OmegaConf

import art

# Create Modal app
app = modal.App("SecretHitler-training")

# Define the Modal image with all dependencies
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .apt_install("procps")
    .pip_install(
        "openpipe-art[backend]",
        "python-dotenv",
        "openai>=1.65.5",
        "requests",
        "weave>=0.51.51",
        "numpy==1.26.4",
        "s3fs",
        "omegaconf",
    )
    # .add_local_dir("example_2048", "/root/example_2048")
    .add_local_dir("src", remote_path="/root/src")
)


# Define Modal secrets
# You'll need to set these in Modal dashboard or via CLI
# modal secret create art-secrets \
#   OPENROUTER_API_KEY=<your-key> \
#   AWS_ACCESS_KEY_ID=<your-key> \
#   AWS_SECRET_ACCESS_KEY=<your-key> \
#   WANDB_API_KEY=<your-key>


@app.function(
    image=image,
    gpu=modal.gpu.H100(),  # Will be overridden by config
    timeout=7200,  # Will be overridden by config
    secrets=[modal.Secret.from_name("wandb-secret")],  # Will be overridden by config
    # mounts=[modal.Mount.from_local_dir(".", remote_path="/root/2048_example")],
)
async def train(config_dict: dict):
    """
    Train the Secret Hitler RL agent.

    Args:
        config_dict: Configuration dictionary (loaded from YAML and serialized)
    """
    # NOW import these - they run in Modal's environment with all deps
    from .rollout import rollout
    from .config import TrainingConfig
    from art.local import LocalBackend

    # Reconstruct config from dict
    config = OmegaConf.structured(TrainingConfig)
    config = OmegaConf.merge(config, config_dict)

    print("=== Training Configuration ===")
    print(OmegaConf.to_yaml(config))
    print("=" * 50)

    # Set random seed
    random.seed(config.train.random_seed)

    print("Starting training")

    # Declare the model
    model = art.TrainableModel(
        name=config.model.name,
        project=config.model.project,
        base_model=config.model.base_model,
    )
    print(f"Model declared: {config.model.name}")
    model._internal_config = art.dev.InternalModelConfig(
        init_args=art.dev.InitArgs(
            max_seq_length=config.train.max_seq_length,
        ),
        engine_args=art.dev.EngineArgs(
            gpu_memory_utilization=config.train.gpu_memory_utilization,
            tensor_parallel_size=config.train.tensor_parallel_size,
        ),
    )

    # Initialize the backend
    backend = LocalBackend()
    print("Backend initialized")

    # Register the model with the local backend (sets up logging, inference, and training)
    await model.register(backend)
    print("Model registered")

    # Train for specified steps
    for i in range(await model.get_step(), config.train.train_steps):
        print(f"Starting training step {i}/{config.train.train_steps}")

        train_groups = await art.gather_trajectory_groups(
            (
                art.TrajectoryGroup(
                    # for each step, rollout simultaneous games
                    rollout(
                        model,
                        i,
                        is_validation=False,
                        max_turns=config.rollout.max_turns,
                        enable_thinking=config.rollout.enable_thinking,
                        verbose=config.rollout.verbose,
                    )
                    for _ in range(config.rollout.simultaneous_games)
                )
                for _ in range(1)
            ),
            pbar_desc="gather",
            max_exceptions=10,
        )

        await model.train(
            train_groups,
            config=art.TrainConfig(learning_rate=config.train.learning_rate),
        )

        print(f"Completed training step {i}/{config.train.train_steps}")


@app.local_entrypoint()
def main(config_path: str = "configs/train_default.yaml"):
    """
    Local entrypoint for running the training on Modal.

    Args:
        config_path: Path to YAML config file (default: configs/train_default.yaml)

    Usage:
        modal run src/rl_training/train_modal.py
        modal run src/rl_training/train_modal.py --config-path configs/my_config.yaml
    """
    from .config import TrainingConfig

    # Load config from YAML
    config_file = Path(config_path)
    if not config_file.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    print(f"Loading config from: {config_path}")

    # Load YAML and merge with structured config
    yaml_config = OmegaConf.load(config_file)
    structured_config = OmegaConf.structured(TrainingConfig)
    config = OmegaConf.merge(structured_config, yaml_config)

    # Validate config (will raise error if MISSING values not provided)
    OmegaConf.to_container(config, throw_on_missing=True)

    print("Config loaded successfully")

    # Convert to dict for serialization to Modal
    config_dict = OmegaConf.to_container(config, resolve=True)

    # Run the train function with config
    print("Submitting training job to Modal...")
    train.remote(config_dict)


if __name__ == "__main__":
    # For local testing (not on Modal)
    # This won't actually run on Modal, just locally
    from .config import TrainingConfig

    config_file = Path("configs/train_default.yaml")
    yaml_config = OmegaConf.load(config_file)
    structured_config = OmegaConf.structured(TrainingConfig)
    config = OmegaConf.merge(structured_config, yaml_config)
    config_dict = OmegaConf.to_container(config, resolve=True)

    asyncio.run(train(config_dict))
