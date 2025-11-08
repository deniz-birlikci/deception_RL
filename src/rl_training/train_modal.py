import asyncio
import random
import modal

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

TRAIN_STEPS = 40
SIMULTANEOUS_GAMES = 18
ENABLE_RULER = True


@app.function(
    image=image,
    gpu="H100",  # or "T4", "A100", etc.
    timeout=7200,  # 2 hours
    secrets=[modal.Secret.from_name("wandb-secret")],
    # mounts=[modal.Mount.from_local_dir(".", remote_path="/root/2048_example")],
)
async def train():
    # Change to the mounted directory so imports work
    # os.chdir("/root/2048_example")
    # sys.path.insert(0, "/root/2048_example")

    # NOW import these - they run in Modal's environment with all deps
    from .rollout import rollout
    from art.local import LocalBackend
    from art.rewards import ruler_score_group

    # Set random seed
    random.seed(42)

    print("Starting training")

    # Declare the model
    model = art.TrainableModel(
        name="tutorial-001",
        project="2048",
        base_model="Qwen/Qwen2.5-3B-Instruct",
    )
    print("Model declared")
    model._internal_config = art.dev.InternalModelConfig(
        init_args=art.dev.InitArgs(
            max_seq_length=8192,
        ),
    )
    print("Internal model config declared")

    # Initialize the backend
    backend = LocalBackend()
    print("Backend declared")

    # Register the model with the local backend (sets up logging, inference, and training)
    await model.register(backend)
    print("Model registered")

    # await backend._experimental_pull_from_s3(
    #     model,
    #     verbose=True,
    # )
    print("Model pulled from S3")

    # Train for TRAIN_STEPS steps
    for i in range(await model.get_step(), TRAIN_STEPS):
        print(f"Starting training step {i}/{TRAIN_STEPS}")

        train_groups = await art.gather_trajectory_groups(
            (
                art.TrajectoryGroup(
                    # for each step, rollout SIMULTANEOUS_GAMES trajectories
                    rollout(model, i, is_validation=False)
                    for _ in range(SIMULTANEOUS_GAMES)
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
                if ENABLE_RULER
                else None
            ),
            pbar_desc="gather",
            max_exceptions=10,
        )

        # save the model to S3
        # await backend._experimental_push_to_s3(
        #     model,
        # )

        await model.train(
            train_groups,
            config=art.TrainConfig(learning_rate=1e-5),
        )

        print(f"Completed training step {i}/{TRAIN_STEPS}")


@app.local_entrypoint()
def main():
    """
    Local entrypoint for running the training on Modal.
    Run with: modal run train_modal.py
    """
    # Run the train function
    print("Running train function")
    train.remote()


if __name__ == "__main__":
    # For local testing (not on Modal)
    # This won't actually run on Modal, just locally
    asyncio.run(train())
