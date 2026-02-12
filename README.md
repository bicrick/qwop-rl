# QWOP World Record Attempt

![banner-3](./doc/banner-3.gif)

This project implements the gym environment from [smanolloff/qwop-gym](https://github.com/smanolloff/qwop-gym) with a singular goal: achieve the QWOP world record.

## The Goal

**100 Meters in 0m 45s 530ms**

Current world record held by [kurodo1916](https://www.speedrun.com/users/kurodo1916) - [Watch the run](https://www.speedrun.com/qwop/runs/y9vk0k2m)

## Training Approach

We trained our agent using **QRDQN (Quantile Regression Deep Q-Network)** over 32 million timesteps with 8 parallel environments running concurrently.

### Key Hyperparameters

Based on our training configuration in `qwop_gym/tools/templates/wandb/qrdqn.yml`:

```yaml
Algorithm: QRDQN (Quantile Regression DQN)
Total Timesteps: 32,000,000
Parallel Environments: 8
Max Episode Steps: 1,000

# Learning Parameters
Batch Size: 64
Buffer Size: 100,000
Learning Rate: 0.001 (constant)
Gamma (Discount Factor): 0.997

# Exploration
Exploration Initial Epsilon: 0.2
Exploration Final Epsilon: 0.0
Exploration Fraction: 0.3 (30% of training)

# Training Schedule
Learning Starts: 100,000 timesteps
Train Frequency: Every 4 steps
Gradient Steps: 1
Target Update Interval: 512 steps
Tau: 1.0
```

## Why QRDQN Over PPO?

QRDQN significantly outperforms policy-based methods like PPO for QWOP. Here's why:

### 1. Discrete Action Space Advantage
QWOP has a discrete action space (key combinations Q, W, O, P). Q-learning methods like QRDQN are naturally suited for discrete actions, directly learning the value of each specific action rather than learning a probability distribution over actions.

### 2. Value-Based vs Policy-Based Learning
QRDQN learns the expected return of each action directly, while PPO learns a probability distribution over actions. In QWOP's physics-heavy, deterministic environment, knowing exact action values is more beneficial than sampling from a distribution. The game requires precise, repeatable action sequences, which value-based methods excel at discovering.

### 3. Sample Efficiency Through Experience Replay
QRDQN uses experience replay with a buffer of 100,000 transitions, allowing it to learn from past experiences multiple times. PPO is on-policy and discards experiences after each update, making it significantly less sample-efficient. In QWOP, where good runs are rare early in training, being able to replay successful episodes is crucial.

### 4. Distributional Reinforcement Learning
QRDQN doesn't just learn the mean expected return - it models the full distribution of returns across quantiles. This is powerful for QWOP's high-variance reward structure where small action differences lead to vastly different outcomes (smooth running vs catastrophic falling). By understanding the full distribution, QRDQN can better navigate risk and make more informed decisions.

### 5. Superior Exploration Strategy
QRDQN's epsilon-greedy exploration is simpler and more effective than PPO's entropy-based exploration for discovering the precise action sequences needed in QWOP. Starting with 20% random exploration and gradually reducing to pure exploitation allows the agent to thoroughly explore the action space early while converging to optimal policies later.

### 6. Off-Policy Learning Advantage
QRDQN can learn from any past experience regardless of which policy generated it, while PPO must learn from recent on-policy data. This makes QRDQN more robust to the challenging credit assignment problem in QWOP: which actions earlier in a run (when the character is at 10m) led to success or failure later (when reaching 50m or falling at 80m)?

## Acknowledgments

- [smanolloff/qwop-gym](https://github.com/smanolloff/qwop-gym) - The excellent gym environment implementation that made this project possible
- [Wesleyliao/QWOP-RL](https://github.com/Wesleyliao/QWOP-RL) - Another notable QWOP reinforcement learning project
- [kurodo1916](https://www.speedrun.com/users/kurodo1916) - Current world record holder who set the bar at 45.530 seconds
