# QWOP World Record Training

Training reinforcement learning agents to play QWOP using the [qwop-gym](https://github.com/smanolloff/qwop-gym) environment.

**Goal:** Beat the current world record of [100 meters in 45.530 seconds](https://www.speedrun.com/qwop/runs/y9vk0k2m).

## Repository Structure

```
qwop-wr/
├── qwop-gym/          # Reference copy of the qwop-gym source code
│                      # (for understanding the environment implementation)
└── qwop-training/     # Standalone training setup for Lambda Labs
    ├── setup_lambda.sh    # Automated setup script
    ├── lambda_train.sh    # Training helper with virtual display
    ├── verify_setup.sh    # Installation verification
    ├── config/            # Training configuration files
    └── README.txt         # Quick start guide
```

## About qwop-gym

This project uses [qwop-gym](https://github.com/smanolloff/qwop-gym) as the Gym environment for training. The qwop-gym package provides:

- A Gymnasium-compatible environment for QWOP
- Deterministic gameplay with configurable frame stepping
- State extraction (60 bytes observation)
- Support for multiple RL algorithms (PPO, DQN, QRDQN, etc.)
- High performance (~2000+ observations/sec)

The `qwop-gym/` directory contains a reference copy of the source code for understanding how the environment works. The actual package is installed via PyPI during setup.

## Quick Start (Lambda Labs)

```bash
# Clone this repo
git clone <your-repo-url>
cd qwop-wr/qwop-training

# Run automated setup
./setup_lambda.sh

# Activate environment
conda activate qwop

# Start training
./lambda_train.sh train_ppo
```

## Training Algorithms

The environment supports multiple RL algorithms:

- **PPO** (Proximal Policy Optimization) - Recommended for beginners
- **DQN** (Deep Q Network)
- **QRDQN** (Quantile Regression DQN)
- **GAIL** (Generative Adversarial Imitation Learning)
- **AIRL** (Adversarial Inverse Reinforcement Learning)
- **BC** (Behavioral Cloning)

## Configuration

Training parameters can be customized by editing the YAML files in `qwop-training/config/`:

- `train_ppo.yml` - PPO hyperparameters
- `train_dqn.yml` - DQN hyperparameters
- `train_qrdqn.yml` - QRDQN hyperparameters
- `spectate.yml` - Model evaluation settings

## Monitoring

Use TensorBoard to monitor training progress:

```bash
tensorboard --logdir qwop-training/data/ --host 0.0.0.0 --port 6006
```

From your local machine:
```bash
ssh -L 6006:localhost:6006 ubuntu@<lambda-ip>
# Open http://localhost:6006
```

## Credits

- QWOP game by Bennett Foddy
- qwop-gym environment by [Simeon Manolov](https://github.com/smanolloff/qwop-gym)
