import asyncio
import random
from pathlib import Path

import modal
from omegaconf import OmegaConf

import art

# Create Modal app
app = modal.App("2048-training")

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
)


# Define Modal secrets
# You'll need to set these in Modal dashboard or via CLI
# modal secret create art-secrets \
#   OPENROUTER_API_KEY=<your-key> \
#   AWS_ACCESS_KEY_ID=<your-key> \
#   AWS_SECRET_ACCESS_KEY=<your-key> \
#   WANDB_API_KEY=<your-key>


# Shared training logic (DRY - Don't Repeat Yourself)
async def _train_impl(config_dict: dict, job_index: int = 0):
    """
    Core training implementation shared by both single and multi-GPU functions.
    
    Args:
        config_dict: Configuration dictionary (loaded from YAML and serialized)
        job_index: Index of this training job (for parallel runs with different seeds)
    """
    # Import inside function - runs in Modal's environment with all deps
    from .rollout import rollout
    from .config import TrainingConfig
    from art.local import LocalBackend
    from art.rewards import ruler_score_group

    # Reconstruct config from dict
    config = OmegaConf.structured(TrainingConfig)
    config = OmegaConf.merge(config, config_dict)

    print("=== Training Configuration ===")
    print(OmegaConf.to_yaml(config))
    print(f"Job Index: {job_index}")
    print("=" * 50)

    # Set random seed (offset by job_index for parallel runs)
    random.seed(config.train.random_seed + job_index)

    print("Starting training")

    # Declare the model with unique name for each job if running parallel
    model_name = config.model.name
    if config.modal.num_parallel_jobs > 1:
        model_name = f"{config.model.name}-job{job_index}"

    model = art.TrainableModel(
        name=model_name,
        project=config.model.project,
        base_model=config.model.base_model,
    )
    print(f"Model declared: {model_name}")
    
    model._internal_config = art.dev.InternalModelConfig(
        init_args=art.dev.InitArgs(
            max_seq_length=config.train.max_seq_length,
        ),
        engine_args=art.dev.EngineArgs(
            gpu_memory_utilization=config.train.gpu_memory_utilization,
            tensor_parallel_size=config.train.tensor_parallel_size,
            # Enable vLLM reasoning parser for Qwen3-4B-Thinking
            additional_config={
                "enable_auto_tool_choice": True,
                "tool_call_parser": "hermes",
                "reasoning_backend": "qwen3",
            },
        ),
    )
    print("Internal model config declared")

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
                        verbose=config.rollout.verbose,
                        max_turns=config.rollout.max_turns,
                        enable_thinking=config.rollout.enable_thinking,
                    )
                    for _ in range(config.rollout.simultaneous_games)
                )
                for _ in range(1)
            ),
            after_each=lambda group: (
                ruler_score_group(
                    group,
                    "openai/o4-mini",
                    debug=True,
                    swallow_exceptions=True,  # Return None on error, filtering out the group
                )
                if config.rollout.enable_ruler
                else None
            ),
            pbar_desc="gather",
            max_exceptions=10,
        )

        await model.train(
            train_groups,
            config=art.TrainConfig(learning_rate=config.train.learning_rate),
        )

        print(f"Completed training step {i}/{config.train.train_steps}")


# Single GPU training function (for data parallelism)
# Modal requires @app.function at global scope, so we can't dynamically create this
@app.function(
    image=image,
    gpu="H100",  # Single H100 GPU per job
    timeout=7200,
    secrets=[modal.Secret.from_name("wandb-secret"), modal.Secret.from_name("art-secrets")],
)
async def train_single_gpu(config_dict: dict, job_index: int = 0):
    """Train on single GPU (data parallelism) - calls shared implementation."""
    await _train_impl(config_dict, job_index)


# Multi-GPU training function (for tensor parallelism)
@app.function(
    image=image,
    gpu="H100:8",  # 8 H100 GPUs in one container
    timeout=7200,
    secrets=[modal.Secret.from_name("wandb-secret"), modal.Secret.from_name("art-secrets")],
)
async def train_multi_gpu(config_dict: dict, job_index: int = 0):
    """Train on 8 GPUs (tensor parallelism) - calls shared implementation."""
    await _train_impl(config_dict, job_index)


@app.local_entrypoint()
def main(config_path: str = "configs/train_default.yaml"):
    """
    Local entrypoint for running the training on Modal.

    Args:
        config_path: Path to YAML config file (default: configs/train_default.yaml)

    Usage:
        # Single GPU training:
        modal run example_2048/train_modal.py
        modal run example_2048/train_modal.py --config-path configs/train_default.yaml

        # 8xH100 parallel training:
        modal run example_2048/train_modal.py --config-path configs/train_8xh100.yaml
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
    print(f"GPU type: {config.modal.gpu_type}")
    print(f"Number of parallel jobs: {config.modal.num_parallel_jobs}")
    print(f"Tensor parallel size: {config.train.tensor_parallel_size}")

    # Convert to dict for serialization to Modal
    config_dict = OmegaConf.to_container(config, resolve=True)

    print()
    print("=" * 80)
    print("PARALLELIZATION STRATEGY")
    print("=" * 80)
    
    # Determine the parallelization strategy
    if config.train.tensor_parallel_size > 1:
        # Tensor parallelism: one job with multiple GPUs (use train_multi_gpu)
        print("✓ Tensor Parallelism Mode")
        print(f"  - 1 training job with {config.train.tensor_parallel_size} GPUs")
        print(f"  - GPU specification: {config.modal.gpu_type}:{config.train.tensor_parallel_size}")
        print(f"  - Model split across {config.train.tensor_parallel_size} GPUs")
        print("  - Use case: Large models that don't fit on 1 GPU")
        print()
        print("Submitting to Modal...")
        print("=" * 80)
        train_multi_gpu.remote(config_dict, 0)
        
    elif config.modal.num_parallel_jobs > 1:
        # Data parallelism: multiple jobs, each with 1 GPU (use train_single_gpu)
        print("✓ Data Parallelism Mode")
        print(f"  - {config.modal.num_parallel_jobs} independent training jobs")
        print(f"  - Each job uses: 1x {config.modal.gpu_type}")
        print(f"  - Total GPUs: {config.modal.num_parallel_jobs} × 1 = {config.modal.num_parallel_jobs} {config.modal.gpu_type}s")
        print("  - Each job trains independently with different seed")
        print("  - Use case: Faster data collection, exploration diversity")
        print()
        print(f"Submitting {config.modal.num_parallel_jobs} parallel jobs to Modal...")
        print("=" * 80)
        # Run multiple jobs in parallel with different job indices
        train_single_gpu.map(
            [(config_dict, job_idx) for job_idx in range(config.modal.num_parallel_jobs)]
        )
        
    else:
        # Single job, single GPU (use train_single_gpu)
        print("✓ Single GPU Mode")
        print("  - 1 training job with 1 GPU")
        print(f"  - GPU specification: {config.modal.gpu_type}")
        print("  - Use case: Testing, small experiments")
        print()
        print("Submitting to Modal...")
        print("=" * 80)
        train_single_gpu.remote(config_dict, 0)


if __name__ == "__main__":
    # For local testing (not on Modal)
    # This won't actually run on Modal, just locally
    from .config import TrainingConfig

    config_file = Path("configs/train_default.yaml")
    yaml_config = OmegaConf.load(config_file)
    structured_config = OmegaConf.structured(TrainingConfig)
    config = OmegaConf.merge(structured_config, yaml_config)
    config_dict = OmegaConf.to_container(config, resolve=True)

    # Use train_single_gpu for local testing
    asyncio.run(train_single_gpu.local(config_dict, 0))
