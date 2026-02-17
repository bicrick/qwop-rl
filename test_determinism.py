#!/usr/bin/env python3
"""Test if QWOP physics are deterministic"""

import sys
sys.path.insert(0, '.')

import gymnasium as gym
from qwop_gym.tools import common
import importlib

# Load model
print("Loading model...")
mod = importlib.import_module("sb3_contrib")
model = mod.QRDQN.load("data/QRDQN-PROVEN-k3jlgned/model.zip")

# Setup environment exactly like evaluation
print("Registering environment...")
env_kwargs = {
    "frames_per_step": 4,
    "reduced_action_set": True,
    "auto_draw": False,
    "stat_in_browser": False,
    "game_in_browser": False,
}

# Expand from env.yml
import yaml
with open('config/env.yml') as f:
    base_cfg = yaml.safe_load(f)
    
for k, v in base_cfg.items():
    if k not in env_kwargs:
        env_kwargs[k] = v

common.register_env(env_kwargs, [])

TEST_SEED = 30000

print(f"\n{'='*70}")
print(f"TEST 1: Run episode and record actions")
print(f"{'='*70}\n")

env1 = gym.make("local/QWOP-v1", seed=TEST_SEED)
env1.reset()
env1.reset()

obs, info = env1.reset(seed=TEST_SEED)
actions1 = []
terminated = False

while not terminated:
    action, _ = model.predict(obs)
    actions1.append(int(action))
    obs, reward, terminated, truncated, info = env1.step(action)

print(f"Run 1 completed:")
print(f"  Actions taken: {len(actions1)}")
print(f"  Final time: {info['time']:.2f}s")
print(f"  Final distance: {info['distance']:.1f}m")
print(f"  Success: {info['is_success']}")

env1.close()

print(f"\n{'='*70}")
print(f"TEST 2: Replay same actions with same seed")
print(f"{'='*70}\n")

env2 = gym.make("local/QWOP-v1", seed=TEST_SEED)
env2.reset()
obs2, info2 = env2.reset(seed=TEST_SEED)

action_iter = iter(actions1)
action_count = 0
terminated = False

while not terminated and action_count < len(actions1):
    action = next(action_iter)
    action_count += 1
    obs2, reward, terminated, truncated, info2 = env2.step(action)
    
    if terminated:
        print(f"\n⚠️  Episode terminated early at action {action_count}/{len(actions1)}")

print(f"\nReplay completed:")
print(f"  Actions consumed: {action_count}/{len(actions1)}")
print(f"  Final time: {info2['time']:.2f}s")
print(f"  Final distance: {info2['distance']:.1f}m")
print(f"  Success: {info2['is_success']}")
print(f"  Remaining actions: {len(actions1) - action_count}")

env2.close()

print(f"\n{'='*70}")
print(f"RESULTS")
print(f"{'='*70}\n")

if info['distance'] == info2['distance'] and info['is_success'] == info2['is_success']:
    print("✓ Deterministic: Both runs had identical outcomes")
else:
    print("✗ Non-deterministic: Outcomes differed!")
    print(f"  Run 1: {info['distance']:.1f}m, success={info['is_success']}")
    print(f"  Run 2: {info2['distance']:.1f}m, success={info2['is_success']}")
    print("\nThis explains why replay fails - QWOP physics are not perfectly deterministic")
