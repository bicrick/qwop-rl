# Discrete SAC Implementation for QWOP

This document describes the implementation of Discrete Soft Actor-Critic (DSAC) for the QWOP environment.

## Overview

Discrete SAC is a maximum entropy reinforcement learning algorithm adapted for discrete action spaces. It addresses key limitations of QRDQN:

- **Superior Exploration**: Entropy regularization prevents convergence to "scooting" local optimum
- **Stable Learning**: Twin Q-networks reduce overestimation bias
- **Automatic Tuning**: Temperature parameter auto-adjusts exploration-exploitation trade-off
- **Off-Policy Efficiency**: Learns from replay buffer like QRDQN

## Architecture

### Policy Network (Actor)
- **Input**: QWOP state (60 floats)
- **Hidden**: 2 layers × 256 units with ReLU
- **Output**: Softmax distribution over 9 actions
- **Initialization**: Orthogonal weights

### Q-Networks (Critics)
- **Twin Networks**: Q1 and Q2 to reduce overestimation
- **Input**: QWOP state (60 floats)
- **Hidden**: 2 layers × 256 units with ReLU
- **Output**: Q-values for all 9 actions
- **Target Networks**: Soft updated with τ=0.005

### Temperature Parameter (α)
- **Auto-tuned**: Maintains target entropy H = -log(1/9) × 0.98 ≈ 2.15 nats
- **Learned**: Optimized alongside policy and critics
- **Purpose**: Balances exploration vs exploitation

## Training Objective

```python
# Policy loss: maximize Q - α·entropy
L_π = E[(action_probs * (α * log_probs - Q_values)).sum()]

# Q-function loss: minimize Bellman error
L_Q = E[(Q(s,a) - (r + γ(min(Q₁', Q₂') - α·log π')))²]

# Temperature loss: maintain target entropy
L_α = E[-α * (H(π) - H_target)]
```

## Files

- `qwop_gym/algorithms/discrete_sac.py` - Main algorithm implementation
- `qwop_gym/wrappers/reward_shaping_wrapper.py` - Reward shaping for anti-scooting
- `config/train_dsac.yml` - Basic training configuration
- `config/train_dsac_shaped.yml` - Config with reward shaping
- `test_discrete_sac.py` - Validation test suite

## Usage

### 1. Test Implementation (Sanity Check)

Test on CartPole-v1 to verify the algorithm works:

```bash
python test_discrete_sac.py
```

This runs a quick 5000-step training session on CartPole and verifies:
- Network initialization
- Forward passes (actor & critics)
- Action selection
- Training loop
- Model save/load

### 2. Train on QWOP (Basic)

Train with vanilla DSAC (no reward shaping):

```bash
python qwop-gym.py train_dsac -c config/train_dsac.yml
```

**Hyperparameters**:
- Buffer: 500K transitions
- Batch: 256
- Learning rate: 0.0003
- Tau: 0.005 (soft update)
- Gamma: 0.99
- Entropy: Auto-tuned to H_target = 2.15 nats

### 3. Train with Reward Shaping (Recommended)

Train with shaped rewards to discourage scooting:

```bash
python qwop-gym.py train_dsac -c config/train_dsac_shaped.yml
```

**Reward Components**:
- **Base reward**: Velocity - time cost - fall penalty
- **Posture penalty**: -1.0 × (height_deficit / min_height) if torso too low
- **Energy cost**: -0.1 per action change (anti-jitter)
- **Joint limit penalty**: -0.5 × (angle_excess) for extreme angles
- **Style bonus**: +0.3 for upright posture

### 4. Monitor Training

View TensorBoard logs:

```bash
tensorboard --logdir data/DSAC-*/
```

**Key Metrics**:
- `train/actor_loss` - Policy gradient loss
- `train/critic_loss` - Q-function TD error
- `train/ent_coef` - Temperature parameter α
- `train/target_entropy` - Target entropy (should be ~2.15)
- `user/distance` - Distance achieved per episode
- `reward_components/*` - Individual reward components (if using shaping)

## Reward Shaping Configuration

The `RewardShapingWrapper` supports several strategies:

### Static Weights (Basic)

```yaml
env_wrappers:
  - module: qwop_gym.wrappers.reward_shaping_wrapper
    cls: RewardShapingWrapper
    kwargs:
      posture_weight: 1.0      # Penalize low torso
      energy_weight: 0.1        # Penalize action switching
      joint_limit_weight: 0.5   # Penalize splits
      style_weight: 0.3         # Reward upright posture
```

### Progressive Decay (Advanced)

Gradually reduce shaping weights as agent learns:

```yaml
env_wrappers:
  - module: qwop_gym.wrappers.reward_shaping_wrapper
    cls: ProgressiveRewardShapingWrapper
    kwargs:
      initial_posture_weight: 1.0
      initial_energy_weight: 0.1
      initial_joint_limit_weight: 0.5
      initial_style_weight: 0.5
      decay_steps: 5_000_000        # Decay over 5M steps
      final_weight_ratio: 0.1       # End at 10% of initial
```

## Expected Performance

Based on the theoretical analysis:

1. **Phase 1 (0-1M steps)**: Random exploration, discovering basic actions
2. **Phase 2 (1-3M steps)**: Learning to stand without falling immediately
3. **Phase 3 (3-7M steps)**: Developing locomotion strategies (hopefully not scooting!)
4. **Phase 4 (7-10M steps)**: Refining gait, increasing distance

**Success Criteria**:
- Episode distance > 20m consistently
- Upright posture maintained (no scooting)
- Smooth, repeatable gait pattern
- Better than QRDQN baseline performance

## Comparison to QRDQN

| Metric | QRDQN | Discrete SAC |
|--------|-------|--------------|
| Exploration | ε-greedy | Entropy regularization |
| Stability | Prone to collapse | Twin Q-networks |
| Local optima | Gets stuck in scooting | Entropy prevents scooting |
| Sample efficiency | Medium | High (with PER) |
| Hyperparameter tuning | Manual | Auto (temperature) |

## Troubleshooting

### Agent learns to scoot despite reward shaping

- Increase `posture_weight` (try 2.0 or 3.0)
- Lower `min_torso_height` threshold (try 180 or 150)
- Add progressive reward decay

### Training is unstable (high variance)

- Reduce learning rate (try 1e-4)
- Increase `batch_size` (try 512)
- Enable reward normalization: `normalize_rewards: true`

### Agent doesn't explore enough

- Lower `target_entropy_ratio` (try 0.95 to allow more exploration)
- Increase `learning_starts` (more random exploration initially)

### Training is too slow

- Reduce `buffer_size` (try 200K)
- Disable PER: `use_per: false`
- Reduce network size: `net_arch: [128, 128]`

## Next Steps

After validating basic DSAC performance:

1. **Imitation Learning**: Pre-train on human speedrun data
2. **Curriculum Learning**: Implement 3-stage training (stability → motion → speed)
3. **Hyperparameter Tuning**: Sweep learning rate, entropy ratio, reward weights
4. **Architecture Search**: Try deeper/wider networks, dueling architecture
5. **Advanced Techniques**: Add n-step returns, multi-step Q-learning

## References

- Christodoulou (2019): "Soft Actor-Critic for Discrete Action Settings" - [arXiv:1910.07207](https://arxiv.org/abs/1910.07207)
- Haarnoja et al. (2018): "Soft Actor-Critic Algorithms and Applications" - [arXiv:1812.05905](https://arxiv.org/abs/1812.05905)
- Schaul et al. (2015): "Prioritized Experience Replay" - [arXiv:1511.05952](https://arxiv.org/abs/1511.05952)
