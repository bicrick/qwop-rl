"""Velocity Incentive Wrapper for QWOP.

This wrapper adds exponential velocity rewards to strongly incentivize high-speed runs.
The key insight: velocity bonuses accumulate over the episode, so staying alive
longer at high speeds yields massive cumulative rewards.

Reward Components:
- Base reward: Original environment reward (velocity × speed_rew_mult - time_cost)
- Velocity bonus: velocity_weight × (velocity ** velocity_exponent)
- Milestone bonuses: One-time bonuses for reaching speed thresholds
- Acceleration bonus: Reward for increasing velocity

The exponential scaling makes high speeds extremely valuable:
- At v^2.5 with weight=2.0:
  - 5 m/s: 112 reward/step
  - 8 m/s: 362 reward/step (3.2x more!)
  - 12 m/s: 995 reward/step (8.9x more!)
  - 15 m/s: 1746 reward/step (15.6x more!)

This creates a strong incentive to maximize speed while maintaining survival
(falling early loses all potential accumulated bonuses).
"""

import gymnasium as gym
import numpy as np
from typing import Any, Dict, Optional, SupportsFloat


class VelocityIncentiveWrapper(gym.Wrapper):
    """
    Wraps QWOP environment to add exponential velocity rewards.
    
    :param env: QWOP environment to wrap
    :param velocity_weight: Base multiplier for velocity bonus (default: 1.0)
    :param velocity_exponent: Exponent for velocity (default: 2.0, try 2.5-3.0 for aggressive)
    :param use_exponential: Use e^(v/scale) instead of v^exp (default: False)
    :param velocity_scale: Scale factor for exponential mode (default: 5.0)
    :param milestone_bonuses: Dict of {speed_threshold: bonus_value} (default: None)
    :param acceleration_weight: Multiplier for acceleration bonus (default: 0.5)
    :param min_velocity_for_bonus: Don't give bonuses below this speed (default: 3.0)
    :param normalize_rewards: Normalize rewards using running statistics (default: False)
    :param reward_clip_max: Maximum absolute reward value (default: None for no clipping)
    """
    
    def __init__(
        self,
        env: gym.Env,
        velocity_weight: float = 1.0,
        velocity_exponent: float = 2.0,
        use_exponential: bool = False,
        velocity_scale: float = 5.0,
        milestone_bonuses: Optional[Dict[float, float]] = None,
        acceleration_weight: float = 0.5,
        min_velocity_for_bonus: float = 3.0,
        normalize_rewards: bool = False,
        reward_clip_max: Optional[float] = None,
    ):
        super().__init__(env)
        
        # Velocity bonus parameters
        self.velocity_weight = velocity_weight
        self.velocity_exponent = velocity_exponent
        self.use_exponential = use_exponential
        self.velocity_scale = velocity_scale
        self.min_velocity_for_bonus = min_velocity_for_bonus
        
        # Milestone bonuses (sorted by threshold for efficient checking)
        if milestone_bonuses is None:
            self.milestone_bonuses = {}
        else:
            self.milestone_bonuses = dict(sorted(milestone_bonuses.items()))
        
        # Acceleration parameters
        self.acceleration_weight = acceleration_weight
        
        # Reward normalization and clipping
        self.normalize_rewards = normalize_rewards
        self.reward_clip_max = reward_clip_max
        
        # Episode tracking
        self.last_velocity = 0.0
        self.last_distance = 0.0
        self.last_time = 0.0
        self.reached_milestones = set()
        self.cumulative_velocity_bonus = 0.0
        self.episode_steps = 0
        
        # Normalization tracking (running mean/std)
        if normalize_rewards:
            self.reward_mean = 0.0
            self.reward_var = 1.0
            self.reward_count = 0
    
    def reset(self, **kwargs) -> tuple:
        """Reset environment and tracking state."""
        obs, info = self.env.reset(**kwargs)
        
        # Reset episode tracking
        self.last_velocity = 0.0
        self.last_distance = 0.0
        self.last_time = 0.0
        self.reached_milestones = set()
        self.cumulative_velocity_bonus = 0.0
        self.episode_steps = 0
        
        return obs, info
    
    def step(self, action: int) -> tuple:
        """
        Execute action and compute velocity-incentivized reward.
        
        :param action: Action to execute
        :return: (observation, shaped_reward, terminated, truncated, info)
        """
        # Execute action in base environment
        obs, base_reward, terminated, truncated, info = self.env.step(action)
        
        self.episode_steps += 1
        
        # Extract velocity information from info
        current_distance = info.get('distance', 0.0)
        current_time = info.get('time', 0.0001)  # Avoid division by zero
        current_velocity = info.get('avgspeed', 0.0)
        
        # Calculate instantaneous velocity (distance delta / time delta)
        if self.last_time > 0:
            time_delta = current_time - self.last_time
            if time_delta > 0:
                instant_velocity = (current_distance - self.last_distance) / time_delta
            else:
                instant_velocity = current_velocity
        else:
            instant_velocity = current_velocity
        
        # Use instantaneous velocity for bonuses (more responsive)
        velocity = max(0.0, instant_velocity)
        
        # Compute shaped reward components
        shaped_reward = base_reward
        reward_components = {
            "base": float(base_reward),
            "velocity_bonus": 0.0,
            "milestone_bonus": 0.0,
            "acceleration_bonus": 0.0,
        }
        
        # 1. Velocity bonus: exponential scaling for high speeds
        if velocity >= self.min_velocity_for_bonus:
            if self.use_exponential:
                # Exponential form: weight × e^(v/scale)
                velocity_bonus = self.velocity_weight * np.exp(velocity / self.velocity_scale)
            else:
                # Power form: weight × v^exponent
                velocity_bonus = self.velocity_weight * (velocity ** self.velocity_exponent)
            
            shaped_reward += velocity_bonus
            reward_components["velocity_bonus"] = float(velocity_bonus)
            self.cumulative_velocity_bonus += velocity_bonus
        
        # 2. Milestone bonuses: one-time rewards for reaching speed thresholds
        milestone_bonus = 0.0
        for threshold, bonus in self.milestone_bonuses.items():
            if velocity >= threshold and threshold not in self.reached_milestones:
                milestone_bonus += bonus
                self.reached_milestones.add(threshold)
        
        if milestone_bonus > 0:
            shaped_reward += milestone_bonus
            reward_components["milestone_bonus"] = float(milestone_bonus)
        
        # 3. Acceleration bonus: reward for increasing speed
        if self.last_velocity > 0 and velocity > self.last_velocity:
            acceleration = velocity - self.last_velocity
            acceleration_bonus = self.acceleration_weight * acceleration
            shaped_reward += acceleration_bonus
            reward_components["acceleration_bonus"] = float(acceleration_bonus)
        
        # Clip rewards if specified
        if self.reward_clip_max is not None:
            shaped_reward = np.clip(shaped_reward, -self.reward_clip_max, self.reward_clip_max)
        
        # Normalize rewards if enabled
        if self.normalize_rewards:
            shaped_reward = self._normalize_reward(shaped_reward)
        
        # Update tracking
        self.last_velocity = velocity
        self.last_distance = current_distance
        self.last_time = current_time
        
        # Add detailed info for logging
        info["reward_components"] = reward_components
        info["shaped_reward"] = float(shaped_reward)
        info["instant_velocity"] = float(velocity)
        info["cumulative_velocity_bonus"] = float(self.cumulative_velocity_bonus)
        info["milestones_reached"] = list(self.reached_milestones)
        
        return obs, shaped_reward, terminated, truncated, info
    
    def _normalize_reward(self, reward: float) -> float:
        """
        Normalize reward using running mean and standard deviation.
        
        Uses Welford's online algorithm for numerical stability.
        """
        self.reward_count += 1
        delta = reward - self.reward_mean
        self.reward_mean += delta / self.reward_count
        delta2 = reward - self.reward_mean
        self.reward_var += delta * delta2
        
        if self.reward_count > 1:
            std = np.sqrt(self.reward_var / (self.reward_count - 1))
            if std > 1e-6:
                return (reward - self.reward_mean) / std
        
        return reward


class ProgressiveVelocityIncentiveWrapper(VelocityIncentiveWrapper):
    """
    Progressive velocity incentive wrapper that increases intensity over time.
    
    Gradually increases velocity bonus weights as training progresses, allowing
    the agent to first learn basic locomotion, then progressively optimize for speed.
    
    :param env: QWOP environment to wrap
    :param initial_velocity_weight: Starting weight for velocity bonus
    :param final_velocity_weight: Final weight for velocity bonus
    :param ramp_steps: Number of steps over which to ramp up weights
    :param initial_exponent: Starting exponent for velocity
    :param final_exponent: Final exponent for velocity
    :param kwargs: Additional arguments passed to VelocityIncentiveWrapper
    """
    
    def __init__(
        self,
        env: gym.Env,
        initial_velocity_weight: float = 0.5,
        final_velocity_weight: float = 2.0,
        ramp_steps: int = 2_000_000,
        initial_exponent: float = 1.5,
        final_exponent: float = 2.5,
        **kwargs,
    ):
        # Start with initial weights
        super().__init__(
            env,
            velocity_weight=initial_velocity_weight,
            velocity_exponent=initial_exponent,
            **kwargs,
        )
        
        self.initial_velocity_weight = initial_velocity_weight
        self.final_velocity_weight = final_velocity_weight
        self.initial_exponent = initial_exponent
        self.final_exponent = final_exponent
        self.ramp_steps = ramp_steps
        self.current_step = 0
    
    def step(self, action: int) -> tuple:
        """Step with progressive weight increase."""
        # Update weights based on progress
        progress = min(1.0, self.current_step / self.ramp_steps)
        
        # Linear interpolation for weights
        self.velocity_weight = (
            self.initial_velocity_weight
            + progress * (self.final_velocity_weight - self.initial_velocity_weight)
        )
        self.velocity_exponent = (
            self.initial_exponent
            + progress * (self.final_exponent - self.initial_exponent)
        )
        
        self.current_step += 1
        
        return super().step(action)
