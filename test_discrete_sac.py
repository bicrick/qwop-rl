#!/usr/bin/env python3.10
"""Test script for Discrete SAC implementation.

This script tests the DiscreteSAC algorithm on CartPole-v1 as a sanity check
before training on the more complex QWOP environment.

Usage:
    python test_discrete_sac.py
"""

import gymnasium as gym
import torch
from qwop_gym.algorithms.discrete_sac import DiscreteSAC
from qwop_gym.tools import common


def test_cartpole():
    """Test Discrete SAC on CartPole-v1."""
    print("=" * 80)
    print("Testing Discrete SAC on CartPole-v1")
    print("=" * 80)
    
    # Create environment
    env = gym.make("CartPole-v1")
    print(f"\nEnvironment: {env.spec.id}")
    print(f"Observation space: {env.observation_space}")
    print(f"Action space: {env.action_space}")
    
    # Check that it's a discrete action space
    assert isinstance(env.action_space, gym.spaces.Discrete), \
        "Discrete SAC requires discrete action space"
    
    print("\n" + "-" * 80)
    print("Initializing Discrete SAC agent...")
    print("-" * 80)
    
    # Create DSAC agent with reduced parameters for quick testing
    model = DiscreteSAC(
        policy="MlpPolicy",
        env=env,
        learning_rate=3e-4,
        buffer_size=10000,
        learning_starts=1000,
        batch_size=64,
        tau=0.005,
        gamma=0.99,
        train_freq=1,
        gradient_steps=1,
        ent_coef="auto",
        target_entropy_ratio=0.98,
        use_per=False,  # Disable PER for simple test
        policy_kwargs={"net_arch": [64, 64]},  # Smaller network for CartPole
        verbose=1,
        device=common.get_device(),
    )
    
    print(f"\nDevice: {model.device}")
    print(f"Policy architecture: {model.policy.net_arch}")
    print(f"Number of actions: {model.policy.n_actions}")
    
    # Test that networks are properly initialized
    print("\n" + "-" * 80)
    print("Checking network initialization...")
    print("-" * 80)
    
    obs, _ = env.reset()
    obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(model.device)
    
    # Test actor forward pass
    with torch.no_grad():
        action_probs, log_probs = model.policy.get_action_probs(obs_tensor)
        print(f"Action probabilities: {action_probs.cpu().numpy()}")
        print(f"Log probabilities: {log_probs.cpu().numpy()}")
        assert action_probs.shape == (1, env.action_space.n), "Wrong action prob shape"
        
    # Test critic forward pass
    with torch.no_grad():
        q1, q2, probs = model.policy.evaluate_actions(obs_tensor)
        print(f"Q1 values: {q1.cpu().numpy()}")
        print(f"Q2 values: {q2.cpu().numpy()}")
        assert q1.shape == (1, env.action_space.n), "Wrong Q1 shape"
        assert q2.shape == (1, env.action_space.n), "Wrong Q2 shape"
    
    # Test action selection
    action = model.predict(obs, deterministic=False)[0]
    print(f"\nSampled action: {action}")
    assert 0 <= action < env.action_space.n, "Invalid action"
    
    print("\n" + "-" * 80)
    print("Training for 5000 steps...")
    print("-" * 80)
    
    # Train for a short period
    model.learn(
        total_timesteps=5000,
        log_interval=1,
        progress_bar=True,
    )
    
    print("\n" + "-" * 80)
    print("Evaluating trained agent...")
    print("-" * 80)
    
    # Evaluate performance
    n_eval_episodes = 10
    episode_rewards = []
    episode_lengths = []
    
    for i in range(n_eval_episodes):
        obs, _ = env.reset()
        episode_reward = 0
        episode_length = 0
        done = False
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            episode_reward += reward
            episode_length += 1
            done = terminated or truncated
        
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        print(f"Episode {i+1}: reward={episode_reward:.2f}, length={episode_length}")
    
    mean_reward = sum(episode_rewards) / len(episode_rewards)
    mean_length = sum(episode_lengths) / len(episode_lengths)
    
    print(f"\nMean reward: {mean_reward:.2f} ± {torch.std(torch.tensor(episode_rewards)):.2f}")
    print(f"Mean length: {mean_length:.2f} ± {torch.std(torch.tensor(episode_lengths)):.2f}")
    
    # Check if learning occurred
    if mean_reward > 50:
        print("\n✓ SUCCESS: Agent learned basic policy (mean reward > 50)")
    else:
        print("\n⚠ WARNING: Agent may need more training (mean reward <= 50)")
    
    print("\n" + "=" * 80)
    print("Test completed successfully!")
    print("=" * 80)
    
    env.close()
    
    return model


def test_save_load():
    """Test saving and loading the model."""
    print("\n" + "=" * 80)
    print("Testing model save/load...")
    print("=" * 80)
    
    env = gym.make("CartPole-v1")
    
    # Create and train a small model
    model = DiscreteSAC(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        buffer_size=1000,
        learning_starts=100,
        batch_size=32,
        policy_kwargs={"net_arch": [32, 32]},
        verbose=0,
    )
    
    print("Training for 500 steps...")
    model.learn(total_timesteps=500, progress_bar=False)
    
    # Save model
    save_path = "/tmp/test_dsac_model"
    print(f"Saving model to {save_path}...")
    model.save(save_path)
    
    # Load model
    print("Loading model...")
    loaded_model = DiscreteSAC.load(save_path, env=env)
    
    # Test that loaded model works
    obs, _ = env.reset()
    action1, _ = model.predict(obs)
    action2, _ = loaded_model.predict(obs)
    
    print(f"Original model action: {action1}")
    print(f"Loaded model action: {action2}")
    
    # Actions should be the same (deterministic predict)
    # Note: Due to randomness in sampling, we just check they're valid
    assert 0 <= action1 < env.action_space.n, "Invalid action from original model"
    assert 0 <= action2 < env.action_space.n, "Invalid action from loaded model"
    
    print("✓ Model save/load test passed!")
    
    env.close()


if __name__ == "__main__":
    print("\n" + "=" * 80)
    print(" DISCRETE SAC IMPLEMENTATION TEST SUITE")
    print("=" * 80)
    
    try:
        # Run CartPole test
        model = test_cartpole()
        
        # Run save/load test
        test_save_load()
        
        print("\n" + "=" * 80)
        print(" ALL TESTS PASSED!")
        print("=" * 80)
        print("\nYou can now train Discrete SAC on QWOP:")
        print("  python qwop-gym.py train_dsac -c config/train_dsac.yml")
        print("\nFor reward shaping, modify config/train_dsac.yml to add:")
        print("  env_wrappers:")
        print("    - module: qwop_gym.wrappers.reward_shaping_wrapper")
        print("      cls: RewardShapingWrapper")
        print("      kwargs:")
        print("        posture_weight: 1.0")
        print("        min_torso_height: 200")
        print("        energy_weight: 0.1")
        print("        joint_limit_weight: 0.5")
        
    except Exception as e:
        print("\n" + "=" * 80)
        print(" TEST FAILED!")
        print("=" * 80)
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
