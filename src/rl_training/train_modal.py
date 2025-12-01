from __future__ import annotations

import asyncio
import math
import os
import random
import shutil
from pathlib import Path
from collections import Counter

import modal
from omegaconf import OmegaConf

import art
from art.utils.output_dirs import get_model_dir
import weave

# Create Modal app
app = modal.App("SecretImpostor-training")

checkpoint_volume = modal.Volume.from_name(
    "art-secret-impostor-checkpoints", create_if_missing=True
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
        "weave>=0.51.51",
        "wandb",
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
    gpu="H100",  # Will be overridden by config
    timeout=43200,  # 12 hours, can be overridden by config
    secrets=[
        modal.Secret.from_name("wandb-secret"),
        modal.Secret.from_name("art-secrets"),
    ],  # Will be overridden by config
    volumes={"/root/.art": checkpoint_volume},
)
async def train(config_dict: dict):
    """
    Train the Secret Impostor RL agent.

    Args:
        config_dict: Configuration dictionary (loaded from YAML and serialized)
    """
    # NOW import these - they run in Modal's environment with all deps
    import logging
    from .rollout import rollout_with_timeout
    from .config import TrainingConfig
    from .metrics_utils import compute_role_based_metrics, compute_oversampling_role_metrics
    from art.local import LocalBackend
    import wandb

    # Silence noisy HTTP loggers - only show gather progress
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)

    # Reconstruct config from dict
    config = OmegaConf.structured(TrainingConfig)
    config = OmegaConf.merge(config, config_dict)

    print("=== Training Configuration ===")
    print(OmegaConf.to_yaml(config))
    print("=" * 50)

    experiment_name = config.model.name

    # Initialize W&B for logging
    wandb.init(
        project=config.model.project,
        name=experiment_name,
        config=OmegaConf.to_container(config, resolve=True),
        reinit=True,
    )

    # Set random seed
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

    # Initialize the backend (prefer ART_STORAGE_PATH or Modal volume)
    storage_path = os.environ.get("ART_STORAGE_PATH")
    default_mount = Path("/root/.art")
    if storage_path is None and default_mount.exists():
        storage_path = str(default_mount)

    if storage_path:
        backend = LocalBackend(path=storage_path)
        backend_path = storage_path
    else:
        backend = LocalBackend()
        backend_path = backend._path  # type: ignore[attr-defined]
    print("Backend initialized")

    # Handle checkpoint start behavior
    checkpoint_mode = (config.checkpoint.mode or "latest").lower()
    checkpoint_step = config.checkpoint.step
    model_dir = Path(get_model_dir(model=model, art_path=backend_path))
    checkpoint_dir = model_dir / "checkpoints"

    if checkpoint_mode not in {"latest", "scratch", "specific"}:
        raise ValueError(
            f"Unknown checkpoint.mode='{config.checkpoint.mode}'. "
            "Valid options: latest, scratch, specific."
        )

    if checkpoint_mode == "scratch":
        if model_dir.exists():
            shutil.rmtree(model_dir)
            print(f"Removed existing model directory at {model_dir} to start from scratch.")
        else:
            print("No existing model directory found; starting from scratch.")
    elif checkpoint_mode == "specific":
        if checkpoint_step is None:
            raise ValueError("checkpoint.step must be set when checkpoint.mode='specific'.")
        specific_dir = checkpoint_dir / f"{checkpoint_step:04d}"
        if not specific_dir.exists():
            raise FileNotFoundError(
                f"Checkpoint for step {checkpoint_step} not found at {specific_dir}."
            )
        if checkpoint_dir.exists():
            for child in checkpoint_dir.iterdir():
                if child.is_dir() and child.name.isdigit() and int(child.name) != checkpoint_step:
                    shutil.rmtree(child)
        print(f"Resuming from checkpoint step {checkpoint_step:04d}.")
    else:
        print("Checkpoint mode set to 'latest' (default resume).")

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
    # Each game has a 5-min timeout (in rollout_with_timeout), so stalled games fail individually
    # This means we keep completed games even if some timeout!
    
    current_step = await model.get_step()
    if config.checkpoint.mode.lower() == "specific" and config.checkpoint.step is not None:
        initial_step = config.checkpoint.step
        if current_step < initial_step:
            current_step = initial_step
            print(f"Overriding start step to {initial_step} based on checkpoint config.")

    for i in range(current_step, config.train.train_steps):
        print(f"Starting training step {i}/{config.train.train_steps}")

        # Per-game timeouts mean stalled games fail individually, keeping completed ones
        train_groups = await art.gather_trajectory_groups(
            (
                art.TrajectoryGroup(
                    # for each step, rollout simultaneous games (with per-game timeout)
                    rollout_with_timeout(
                        model,
                        i,
                        is_validation=False,
                        max_turns=config.rollout.max_turns,
                        enable_thinking=config.rollout.enable_thinking,
                        verbose=config.rollout.verbose,
                        trainable_impostor_prob=config.rollout.trainable_impostor_prob,
                    )
                    for _ in range(config.rollout.simultaneous_games)
                )
                for _ in range(1)
            ),
            pbar_desc="gather",
            max_exceptions=config.rollout.simultaneous_games,  # Allow all to fail individually
        )

        # Log step-level aggregate metrics
        all_trajectories = [t for group in train_groups for t in group.trajectories]
        games_completed = len(all_trajectories)
        games_expected = config.rollout.simultaneous_games
        min_games_for_training = max(1, math.ceil(games_expected * 0.5))
        impostor_starts = sum(
            int(t.metrics.get("trainable_impostor_start", 0)) for t in all_trajectories
        )
        winning_team_counts = Counter(
            (t.metadata.get("winning_team") or "unknown") for t in all_trajectories
        )
        role_counts = Counter(
            (t.metadata.get("trainable_role") or "unknown") for t in all_trajectories
        )

        exception_counts = Counter(
            exc.type for group in train_groups for exc in group.exceptions
        )
        total_exceptions = sum(exception_counts.values())

        if games_completed >= min_games_for_training:
            rewards = [t.reward for t in all_trajectories]
            wins = sum(1 for r in rewards if r > 0)
            wandb.log({
                "step/mean_reward": sum(rewards) / len(rewards),
                "step/max_reward": max(rewards),
                "step/min_reward": min(rewards),
                "step/win_rate": wins / len(rewards),
                "step/exceptions": total_exceptions,
                "step/games_completed": games_completed,
                "step/games_expected": games_expected,
                "step/trainable_impostor_starts": impostor_starts,
                "step/impostor_wins": winning_team_counts.get("impostor", 0),
                "step/crewmate_wins": winning_team_counts.get("crewmate", 0),
                "step/trainable_role_master_impostor": role_counts.get("master_impostor", 0),
                "step/trainable_role_impostor": role_counts.get("impostor", 0),
                "step/trainable_role_crewmate": role_counts.get("crewmate", 0),
            }, step=i)
            print(f"Step {i}: {games_completed}/{games_expected} games completed, training...")
            print(f"Winning teams: {dict(winning_team_counts)} | Roles: {dict(role_counts)}")
        else:
            wandb.log({
                "step/games_completed": games_completed,
                "step/games_expected": games_expected,
                "step/exceptions": total_exceptions,
                "step/trainable_impostor_starts": impostor_starts,
                "step/impostor_wins": winning_team_counts.get("impostor", 0),
                "step/crewmate_wins": winning_team_counts.get("crewmate", 0),
                "step/trainable_role_master_impostor": role_counts.get("master_impostor", 0),
                "step/trainable_role_impostor": role_counts.get("impostor", 0),
                "step/trainable_role_crewmate": role_counts.get("crewmate", 0),
            }, step=i)
            failure_breakdown = ""
            if total_exceptions:
                top_failures = ", ".join(
                    f"{name.split('.')[-1]}: {count}"
                    for name, count in exception_counts.most_common(3)
                )
                failure_breakdown = f" Failures: {top_failures}"
            print(
                f"Step {i}: Only {games_completed}/{games_expected} games completed "
                f"(need ≥{min_games_for_training}), skipping training."
                f"{failure_breakdown}"
            )
            if winning_team_counts:
                print(f"Winning teams: {dict(winning_team_counts)} | Roles: {dict(role_counts)}")
            continue

        # Compute and log role-based metrics for all trajectories
        role_metrics = compute_role_based_metrics(train_groups)
        oversample_metrics = compute_oversampling_role_metrics(train_groups)
        combined_metrics = {**role_metrics, **oversample_metrics}
        if combined_metrics:
            wandb.log(combined_metrics, step=i)
            print(f"Role-based metrics logged: {len(combined_metrics)} metrics")

        await model.train(
            train_groups,
            config=art.TrainConfig(learning_rate=config.train.learning_rate),
        )

        print(f"Completed training step {i}/{config.train.train_steps}")


@app.local_entrypoint()
def main(config_path: str | None = None):
    """
    Local entrypoint for running the training on Modal.

    Args:
        config_path: Path to YAML config file (default: src/rl_training/configs/train_default.yaml)

    Usage:
        modal run -m src.rl_training.train_modal
        modal run -m src.rl_training.train_modal --config-path src/rl_training/configs/my_config.yaml
    """
    from .config import TrainingConfig

    # Default to config relative to this script
    if config_path is None:
        script_dir = Path(__file__).parent
        config_file = script_dir / "configs" / "train_default.yaml"
    else:
        config_file = Path(config_path)
    
    if not config_file.exists():
        raise FileNotFoundError(f"Config file not found: {config_file}")

    print(f"Loading config from: {config_file}")

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
    from .config import TrainingConfig

    script_dir = Path(__file__).parent
    config_file = script_dir / "configs" / "train_default.yaml"
    yaml_config = OmegaConf.load(config_file)
    structured_config = OmegaConf.structured(TrainingConfig)
    config = OmegaConf.merge(structured_config, yaml_config)
    config_dict = OmegaConf.to_container(config, resolve=True)

    asyncio.run(train(config_dict))
