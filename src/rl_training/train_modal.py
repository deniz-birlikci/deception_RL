import asyncio
import random
from pathlib import Path

import modal
from omegaconf import OmegaConf

import art
from src.rl_training.word_list import WORD_LIST


def generate_experiment_name(base_name: str) -> str:
    """
    Generate a random experiment name with format: {word1}_{word2}_{base_name}

    Args:
        base_name: The base model name from config

    Returns:
        Random experiment name
    """
    word1, word2 = random.sample(WORD_LIST, 2)
    return f"{word1}_{word2}_{base_name}"


# Create Modal app
app = modal.App("SecretHitler-training")

checkpoint_volume = modal.Volume.from_name(
    "art-secret-hitler-checkpoints", create_if_missing=True
)

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
        "weave==0.52.16",
        "numpy==1.26.4",
        "s3fs",
        "omegaconf",
        "pydantic",
        "pydantic-settings",
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


async def gather_rollouts_with_timeout(
    model,
    step: int,
    is_validation: bool,
    oversampling_concurrency: int,
    timeout_seconds: float,
    max_turns: int,
    enable_thinking: bool,
    verbose: bool,
):
    """
    Launch rollouts with oversampling and a global timeout.

    Args:
        model: The ART model to use
        step: Training step number
        is_validation: Whether this is a validation run
        oversampling_concurrency: Total number of rollouts to launch
        timeout_seconds: Global timeout for all rollouts
        max_turns: Maximum turns per game
        enable_thinking: Enable thinking for Qwen models
        verbose: Print debug info

    Returns:
        Tuple of (trajectory_groups, oversampling_metrics)
    """
    import time
    from .rollout import rollout

    start_time = time.perf_counter()

    # Launch all rollouts concurrently
    tasks = [
        asyncio.create_task(
            rollout(
                model=model,
                step=step,
                is_validation=is_validation,
                verbose=verbose,
                max_turns=max_turns,
                enable_thinking=enable_thinking,
            )
        )
        for _ in range(oversampling_concurrency)
    ]

    # Wait for rollouts to complete or timeout
    completed_trajectories = []
    abandoned_count = 0

    try:
        # Wait with timeout - returns when ALL complete OR timeout is hit
        done, pending = await asyncio.wait(tasks, timeout=timeout_seconds)

        # Collect completed trajectories
        for task in done:
            try:
                trajectory = await task
                completed_trajectories.append(trajectory)
            except Exception as e:
                print(f"Rollout failed with exception: {e}")
                abandoned_count += 1

        # Cancel pending tasks (these timed out)
        abandoned_count += len(pending)
        for task in pending:
            task.cancel()
        
        # Give cancelled tasks a brief grace period to clean up
        if pending:
            try:
                await asyncio.wait(pending, timeout=5.0)
            except asyncio.CancelledError:
                pass
            # After 5s, move on regardless - don't wait forever

    except Exception as e:
        # Unexpected error - cancel all tasks
        print(f"Error during rollout gathering: {e}")
        abandoned_count = len(tasks)
        for task in tasks:
            if not task.done():
                task.cancel()
        
        # Brief grace period for cleanup
        not_done = [t for t in tasks if not t.done()]
        if not_done:
            try:
                await asyncio.wait(not_done, timeout=5.0)
            except asyncio.CancelledError:
                pass

    elapsed_time = time.perf_counter() - start_time

    # Compute metrics
    total_launched = oversampling_concurrency
    total_completed = len(completed_trajectories)
    completion_rate = total_completed / total_launched if total_launched > 0 else 0
    abandonment_rate = abandoned_count / total_launched if total_launched > 0 else 0

    oversampling_metrics = {
        "total_launched": total_launched,
        "total_completed": total_completed,
        "total_abandoned": abandoned_count,
        "completion_rate": completion_rate,
        "abandonment_rate": abandonment_rate,
        "elapsed_time_seconds": elapsed_time,
        "oversampling_concurrency": oversampling_concurrency,
    }

    # Wrap in TrajectoryGroup as expected by ART
    train_groups = [art.TrajectoryGroup(completed_trajectories)]

    return train_groups, oversampling_metrics


@app.function(
    image=image,
    gpu="H100:4",  # Will be overridden by config
    timeout=7200,  # Will be overridden by config
    secrets=[
        modal.Secret.from_name("wandb-secret"),
        modal.Secret.from_name("art-secrets"),
    ],  # Will be overridden by config
    volumes={"/root/.art": checkpoint_volume},
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
    from .metrics_utils import compute_role_based_metrics, compute_oversampling_role_metrics
    from art.local import LocalBackend
    import weave
    import wandb

    # Reconstruct config from dict
    config = OmegaConf.structured(TrainingConfig)
    config = OmegaConf.merge(config, config_dict)

    print("=== Training Configuration ===")
    print(OmegaConf.to_yaml(config))
    print("=" * 50)

    # Generate random experiment name BEFORE setting seed (so it's random across runs)
    experiment_name = generate_experiment_name(config.model.name)
    print(f"Generated experiment name: {experiment_name}")

    # Set random seed for reproducible training
    random.seed(config.train.random_seed)

    print("Starting training")

    # Declare the model
    model = art.TrainableModel(
        name=experiment_name,
        project=config.model.project,
        base_model=config.model.base_model,
    )
    print(f"Model declared: {experiment_name}")
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

    # Set up weave logging, control that openai is not autotracked
    weave.init(
        project_name=config.model.project,
        autopatch_settings={"openai": {"enabled": False}},
        settings={"implicitly_patch_integrations": False},
    )

    # Train for specified steps
    for i in range(await model.get_step(), config.train.train_steps):
        print(f"Starting training step {i}/{config.train.train_steps}")

        # Determine rollout strategy based on oversampling config
        if config.rollout.enable_oversampling:
            print(
                f"Oversampling enabled: launching {config.rollout.oversampling_concurrency} rollouts, "
                f"timeout={config.rollout.oversampling_timeout_seconds}s"
            )
            train_groups, oversampling_metrics = await gather_rollouts_with_timeout(
                model=model,
                step=i,
                is_validation=False,
                oversampling_concurrency=config.rollout.oversampling_concurrency,
                timeout_seconds=config.rollout.oversampling_timeout_seconds,
                max_turns=config.rollout.max_turns,
                enable_thinking=config.rollout.enable_thinking,
                verbose=config.rollout.verbose,
            )
            print(f"Oversampling metrics: {oversampling_metrics}")
            
            # Compute role-based oversampling metrics
            completed_trajs = []
            for group in train_groups:
                completed_trajs.extend(group.trajectories)
            
            oversampling_role_metrics = compute_oversampling_role_metrics(
                completed_trajectories=completed_trajs,
                total_launched=oversampling_metrics["total_launched"],
                total_abandoned=oversampling_metrics["total_abandoned"],
            )
            
            # Log oversampling metrics to wandb
            wandb.log({
                f"oversampling/{k}": v 
                for k, v in oversampling_metrics.items()
            }, step=i)
            
            # Log oversampling role metrics
            if oversampling_role_metrics:
                wandb.log(oversampling_role_metrics, step=i)
        else:
            # Standard rollout without oversampling
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

        # Compute and log role-based metrics for all trajectories
        role_metrics = compute_role_based_metrics(train_groups)
        if role_metrics:
            wandb.log(role_metrics, step=i)
            print(f"Role-based metrics logged: {len(role_metrics)} metrics")

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
