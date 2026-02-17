# DQNfD World Record Training Pipeline

This guide walks through the complete DQN-from-Demonstrations (DQNfD) training pipeline for achieving superhuman performance in QWOP, following Wesley Liao's proven methodology.

**Target: Break the 45-second world record** (requires ~13.3 m/s average speed)

## Overview

Wesley Liao's breakthrough came from a specific training approach:
1. Get expert demonstrations
2. Inject demonstrations into the DQN replay buffer **during** training (not just pre-training)
3. After internalizing techniques, remove demonstrations and optimize further
4. Fine-tune purely for speed with velocity incentives

We replicate this exactly, using **QRDQN-PROVEN** (100% success rate, 7.52s game time) as our expert.

## Prerequisites

1. **Bootstrap the environment** (if not already done):
   ```bash
   qwop-gym bootstrap
   ```

2. **Verify PROVEN model exists**:
   ```bash
   ls -lh data/QRDQN-PROVEN-k3jlgned/model.zip
   ```

3. **Install dependencies**:
   ```bash
   pip install qwop-gym[sb3]
   ```

## Pipeline Stages

### Stage 0: Collect Expert Demonstrations

Collect full transition tuples (obs, action, reward, next_obs, done) from the PROVEN model:

```bash
qwop-gym -c config/collect_demos.yml collect_demos
```

**What this does:**
- Runs PROVEN model for 50 episodes
- Saves ~10,000-15,000 transitions to `data/demonstrations/qrdqn_proven_demos.npz`
- Takes ~5-10 minutes

**Expected output:**
```
Loaded 50000 demonstration transitions
Total episodes:     50
Successful:         50 (100.0%)
Total transitions:  12345
Output file:        data/demonstrations/qrdqn_proven_demos.npz
```

### Stage 1: Warm-up with Demonstrations (~10M steps, 3-4 hours)

Train EQRDQN with hybrid replay buffer (50% expert, 50% self-play):

```bash
qwop-gym -c config/train_dqnfd_stage1.yml train_eqrdqn
```

**What this does:**
- Loads PROVEN weights as initialization
- Injects 1 expert transition per 2 agent steps (Wesley's ratio)
- Uses Prioritized Experience Replay (PER)
- 300k buffer size (3x PROVEN's capacity)
- Low exploration (0.1 → 0.01) since starting from good policy

**Success criteria:**
- Maintains 100% success rate throughout training
- Episode reward increases gradually (learning to blend expert + self-play)
- No catastrophic forgetting

**Monitor with TensorBoard:**
```bash
tensorboard --logdir data/DQNfD-Stage1-*/
```

Watch for:
- `rollout/ep_rew_mean` should stay positive and increase
- `user/is_success` should stay at or near 1.0
- `train/mean_td_error` should decrease over time

### Stage 2: Self-Play Refinement (~15M steps, 5-6 hours)

Remove demonstrations and refine policy through pure self-play:

**FIRST: Update config with Stage 1 model path**
```bash
# Edit config/train_dqnfd_stage2.yml
# Change: model_load_file: "data/DQNfD-Stage1-XXXXX/model.zip"
# To:     model_load_file: "data/DQNfD-Stage1-<your-run-id>/model.zip"
```

Then train:
```bash
qwop-gym -c config/train_dqnfd_stage2.yml train_eqrdqn
```

**What this does:**
- NO demonstration injection (pure self-play)
- Even lower exploration (0.05 → 0.001)
- Lower learning rate (0.0001 → 0.00001)
- Agent consolidates techniques without "fighting" expert data

**Success criteria:**
- Maintains success rate (should not drop below 95%)
- Policy stabilizes (episode variance decreases)
- Average speed improves slightly

### Stage 3: Speed Optimization (~10M steps, 3-4 hours)

Add aggressive velocity incentives to optimize for pure speed:

**FIRST: Update config with Stage 2 model path**
```bash
# Edit config/train_dqnfd_stage3.yml
# Change: model_load_file: "data/DQNfD-Stage2-XXXXX/model.zip"
# To:     model_load_file: "data/DQNfD-Stage2-<your-run-id>/model.zip"
```

Then train:
```bash
qwop-gym -c config/train_dqnfd_stage3.yml train_eqrdqn
```

**What this does:**
- Adds `VelocityIncentiveWrapper` with exponential scaling (v^2.5)
- Milestone bonuses at 10, 15, 20, 25 m/s
- Almost zero exploration (0.02 → 0.0)
- Very low learning rate for fine-tuning

**Velocity bonus breakdown:**
- 5 m/s: +112 reward/step (baseline running)
- 8 m/s: +362 reward/step (good technique)
- 12 m/s: +995 reward/step (fast running)
- 15 m/s: +1746 reward/step (elite speed)
- 20 m/s: +4000 reward/step (WORLD RECORD territory!)

**Success criteria:**
- Average speed increases significantly (target: 13+ m/s)
- Best runs approach or break 45 seconds
- Success rate may drop slightly (80-90%) due to risk-taking

## Evaluation

After Stage 3, evaluate the final model over many seeds to find the best run:

```bash
# Edit config/evaluate_qrdqn_proven.yml to point to Stage 3 model
qwop-gym -c config/evaluate_dqnfd_stage3.yml evaluate
```

Run 500+ episodes with different seeds to capture best-case performance:
```yaml
n_runs: 500
model_file: "data/DQNfD-Stage3-WR-<your-run-id>/model.zip"
keep_top_n: 20
out_dir: "data/evaluations/DQNfD-Stage3-WR"
```

**World record criteria:**
- Best run completes 100m in < 45 seconds
- Average speed > 13.3 m/s (100m ÷ 7.5s × 1/frame_skip)

## Troubleshooting

### Stage 1: Agent forgets how to run
**Symptom:** Success rate drops, knee-scraping returns  
**Solution:** Increase `demo_injection_ratio` to 0.7 or 0.8 (more expert guidance)

### Stage 2: Performance degrades without demos
**Symptom:** Success rate drops below 90%  
**Solution:** Temporarily re-enable demos at lower ratio (0.2-0.3) for 2-3M steps

### Stage 3: Agent falls frequently
**Symptom:** Success rate drops below 70%, agent takes excessive risks  
**Solution:** 
- Reduce velocity incentive weights (`velocity_weight: 1.5`)
- Increase `failure_cost` in env_kwargs (penalize falling more)
- Add `reward_clip_max: 5000` to prevent runaway rewards

### Insufficient speed improvement in Stage 3
**Symptom:** Average speed plateaus below 10 m/s  
**Solution:**
- Increase `velocity_exponent` to 3.0 (more aggressive scaling)
- Add milestone bonuses at lower thresholds (7m/s, 9m/s)
- Increase `acceleration_weight` to reward speed increases

## Architecture Summary

```
QRDQN-PROVEN (expert)
       ↓
  collect_demos → expert_demos.npz
       ↓
Stage 1: EQRDQN + DQNfD (50% demos, 50% self-play, 10M steps)
       ↓
Stage 2: EQRDQN (100% self-play, 15M steps)
       ↓
Stage 3: EQRDQN + VelocityIncentiveWrapper (speed optimization, 10M steps)
       ↓
  Evaluation (500+ runs) → WORLD RECORD
```

## Key Differences from Wesley's Approach

| Aspect | Wesley (QWOP-RL-1) | This Pipeline |
|--------|-------------------|---------------|
| Expert source | Human (Kurodo) | QRDQN-PROVEN (100% success) |
| Algorithm | Prioritized DDQN (TF1) | Enhanced QRDQN (PyTorch) |
| Network | [256, 128] | [256, 128] (same) |
| Buffer size | 300k | 300k (same) |
| Demo injection | 50% | 50% (same) |
| Stages | ACER → DQNfD → Self-play | DQNfD → Self-play → Velocity |
| Environment | Selenium (slow) | WebSocket (fast) |
| Observation | 71-dim | 60-dim (normalized) |

## Expected Timeline

- **Stage 0 (demos):** 10 minutes
- **Stage 1 (DQNfD):** 3-4 hours
- **Stage 2 (self-play):** 5-6 hours
- **Stage 3 (velocity):** 3-4 hours
- **Evaluation:** 1-2 hours

**Total:** ~13-17 hours of training time

## References

- Wesley Liao's article: https://wesleyliao.medium.com/achieving-super-human-performance-in-qwop-9a4b968c4cc9
- DQNfD paper: https://arxiv.org/abs/1704.03732
- Prioritized Experience Replay: https://arxiv.org/abs/1511.05952
- QRDQN: https://arxiv.org/abs/1710.10044

## Next Steps

Once you break the world record:
1. Record the best run with `qwop-gym spectate`
2. Save the recording as evidence
3. Publish results with model weights
4. Consider curriculum learning for even faster convergence
5. Experiment with frame-level control (higher frame rate)

Good luck breaking the world record!
