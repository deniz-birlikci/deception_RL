# 2048 Training Example for Modal Labs

This example shows how to train a language model to play 2048 using reinforcement learning on Modal Labs infrastructure.

## Setup

### 1. Install Modal

```bash
pip install modal
```

### 2. Authenticate with Modal

```bash
modal setup
```

This will open a browser window to authenticate with Modal.

### 3. Set up Modal Secrets

You need to create a Modal secret with the following environment variables:

```bash
modal secret create art-secrets \
  OPENROUTER_API_KEY=<your-openrouter-api-key> \
  AWS_ACCESS_KEY_ID=<your-aws-access-key> \
  AWS_SECRET_ACCESS_KEY=<your-aws-secret-key> \
  WANDB_API_KEY=<your-wandb-api-key>
```

**Required Environment Variables:**

- `OPENROUTER_API_KEY`: Your OpenRouter API key for model inference (get from https://openrouter.ai/)
- `AWS_ACCESS_KEY_ID`: AWS access key for S3 checkpoint storage
- `AWS_SECRET_ACCESS_KEY`: AWS secret key for S3 checkpoint storage  
- `WANDB_API_KEY`: Weights & Biases API key for logging (optional but recommended, get from https://wandb.ai/)

### 4. Run Training on Modal

```bash
modal run train_modal.py
```

This will:
1. Spin up a GPU instance on Modal (A10G by default)
2. Load the base model (Qwen2.5-3B-Instruct)
3. Run 40 training steps, each with 18 simultaneous game rollouts
4. Save checkpoints to S3 after each step
5. Log metrics to Weights & Biases

## Configuration

You can modify these parameters in `train_modal.py`:

- `TRAIN_STEPS`: Number of training iterations (default: 40)
- `SIMULTANEOUS_GAMES`: Number of parallel game rollouts per step (default: 18)
- `ENABLE_RULER`: Enable RULER scoring for trajectory quality (default: True)
- `gpu`: GPU type (default: "A10G", options: "T4", "A100", "H100", etc.)
- `timeout`: Maximum runtime in seconds (default: 7200 = 2 hours)

## Local Testing

To test the rollout locally without Modal:

```bash
# Install dependencies
pip install openpipe-art[backend] python-dotenv openai requests weave

# Create a .env file with your API keys
echo "OPENROUTER_API_KEY=your-key" > .env

# Run a test rollout
python rollout.py
```

## File Structure

- `train_modal.py`: Main Modal training script
- `rollout.py`: Defines the 2048 game rollout logic
- `utils.py`: 2048 game implementation
- `README.md`: This file

## How It Works

1. **Model Setup**: Starts with Qwen2.5-3B-Instruct as the base model
2. **Game Rollout**: Agent plays 2048 by outputting moves in XML format (`<move>left</move>`)
3. **Reward Function**: 
   - Win bonus: 2.0 for reaching 128
   - Max value reward: Logarithmic scale based on highest tile
   - Board value reward: Logarithmic scale based on total board value
4. **Training**: Uses the ART framework to fine-tune the model on successful trajectories
5. **Checkpointing**: Saves model state to S3 after each training step

## Cost Estimation

Running on Modal with an A10G GPU:
- Compute: ~$1.10/hour
- 40 training steps typically take 1-2 hours
- Total estimated cost: $1-3 per full training run

## Monitoring

View your training progress:
- Modal dashboard: https://modal.com/apps
- Weights & Biases: https://wandb.ai/ (if enabled)

## Troubleshooting

**GPU Out of Memory**: Reduce `SIMULTANEOUS_GAMES` or use a larger GPU
**Timeout**: Increase `timeout` parameter or reduce `TRAIN_STEPS`
**S3 Errors**: Check your AWS credentials and permissions

