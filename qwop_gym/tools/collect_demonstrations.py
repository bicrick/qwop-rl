# =============================================================================
# Copyright 2023 Simeon Manolov <s.manolloff@gmail.com>.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =============================================================================

import gymnasium as gym
import importlib
import numpy as np
import os


def load_model(mod_name, cls_name, file):
    """Load a trained model from file."""
    print(f"Loading {cls_name} model from {file}")
    mod = importlib.import_module(mod_name)
    
    if cls_name == "BC":
        return mod.reconstruct_policy(file)
    
    return getattr(mod, cls_name).load(file)


def collect_demonstrations(
    model_file,
    model_mod,
    model_cls,
    n_episodes,
    out_file,
    seed_start=10000,
    steps_per_step=1,
):
    """
    Run expert model for N episodes and save full transition tuples.
    
    Creates a .npz file containing:
    - obs: observations (N, obs_dim)
    - actions: actions taken (N,)
    - rewards: rewards received (N,)
    - next_obs: next observations (N, obs_dim)
    - dones: episode termination flags (N,)
    
    This format matches Wesley's DQNfD expectations and can be used for
    demonstration injection during training.
    
    Args:
        model_file: Path to the trained model file
        model_mod: Module name (e.g., "sb3_contrib")
        model_cls: Algorithm class name (e.g., "QRDQN")
        n_episodes: Number of episodes to collect
        out_file: Output .npz file path
        seed_start: Starting seed value for episodes
        steps_per_step: Number of env.step() calls per model action
    """
    # Load the expert model
    model = load_model(model_mod, model_cls, model_file)
    
    # Create environment
    env = gym.make("local/QWOP-v1", seed=seed_start)
    
    # Storage for all transitions
    all_obs = []
    all_actions = []
    all_rewards = []
    all_next_obs = []
    all_dones = []
    
    print(f"\n{'='*70}")
    print(f"Collecting demonstrations: {n_episodes} episodes")
    print(f"Model: {model_file}")
    print(f"Output: {out_file}")
    print(f"{'='*70}\n")
    
    try:
        # Initial resets (gymnasium requirement)
        env.reset()
        env.reset()
        
        total_transitions = 0
        successful_episodes = 0
        
        for ep in range(n_episodes):
            # Reset without seed to get natural variation (no page reload)
            obs, info = env.reset()
            
            episode_transitions = 0
            terminated = False
            
            while not terminated:
                # Get action from expert model
                action, _ = model.predict(obs, deterministic=True)
                
                # Execute action (potentially multiple times per steps_per_step)
                for _ in range(steps_per_step):
                    next_obs, reward, terminated, truncated, info = env.step(action)
                    
                    # Store transition
                    all_obs.append(obs.copy())
                    all_actions.append(int(action))
                    all_rewards.append(float(reward))
                    all_next_obs.append(next_obs.copy())
                    all_dones.append(bool(terminated or truncated))
                    
                    episode_transitions += 1
                    total_transitions += 1
                    
                    if terminated or truncated:
                        break
                    
                    obs = next_obs
            
            # Episode complete
            success = info.get('is_success', False)
            if success:
                successful_episodes += 1
            
            print(f"Episode {ep+1}/{n_episodes}: "
                  f"time={info['time']:.2f}s, "
                  f"distance={info['distance']:.1f}m, "
                  f"success={success}, "
                  f"transitions={episode_transitions} "
                  f"(natural variation, no reload)")
        
    finally:
        env.close()
    
    # Convert to numpy arrays
    obs_array = np.array(all_obs, dtype=np.float32)
    actions_array = np.array(all_actions, dtype=np.int32)
    rewards_array = np.array(all_rewards, dtype=np.float32)
    next_obs_array = np.array(all_next_obs, dtype=np.float32)
    dones_array = np.array(all_dones, dtype=bool)
    
    # Save to .npz file
    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    np.savez_compressed(
        out_file,
        obs=obs_array,
        actions=actions_array,
        rewards=rewards_array,
        next_obs=next_obs_array,
        dones=dones_array,
    )
    
    # Print summary
    print(f"\n{'='*70}")
    print(f"COLLECTION SUMMARY")
    print(f"{'='*70}")
    print(f"Total episodes:     {n_episodes}")
    print(f"Successful:         {successful_episodes} ({successful_episodes/n_episodes*100:.1f}%)")
    print(f"Total transitions:  {total_transitions}")
    print(f"Avg per episode:    {total_transitions/n_episodes:.1f}")
    print(f"Output file:        {out_file}")
    print(f"Output size:        {os.path.getsize(out_file) / 1024 / 1024:.2f} MB")
    print(f"{'='*70}\n")
    
    print(f"Demonstrations saved successfully!")
    print(f"Array shapes:")
    print(f"  obs:      {obs_array.shape}")
    print(f"  actions:  {actions_array.shape}")
    print(f"  rewards:  {rewards_array.shape}")
    print(f"  next_obs: {next_obs_array.shape}")
    print(f"  dones:    {dones_array.shape}")
