"""Reward Shaping Wrapper for QWOP.

This wrapper adds composite reward components to encourage proper locomotion
and discourage degenerate strategies like "scooting" (knee-crawling).

Based on recommendations from "Locomotion Synthesis in High-Dimensional 
Contact-Rich Environments" for designing reward functions that guide RL
agents toward human-like bipedal locomotion.
"""

import gymnasium as gym
import numpy as np
from typing import Any, Dict, Optional, SupportsFloat


class RewardShapingWrapper(gym.Wrapper):
    """
    Wraps QWOP environment to add shaped reward components.
    
    Composite reward function:
    R = R_base + R_posture + R_energy + R_joint_limit + R_style
    
    Components:
    - R_base: Original environment reward (velocity - time cost - fall penalty)
    - R_posture: Penalty for low torso height (discourages scooting)
    - R_energy: Penalty for excessive action changes (discourages jitter)
    - R_joint_limit: Penalty for extreme joint angles (prevents splits)
    - R_style: Bonus for maintaining upright running posture (optional)
    
    :param env: QWOP environment to wrap
    :param posture_weight: Weight for torso height penalty (default: 0.0)
    :param min_torso_height: Minimum acceptable torso height in pixels (default: 200)
    :param energy_weight: Weight for action change penalty (default: 0.0)
    :param joint_limit_weight: Weight for joint limit penalty (default: 0.0)
    :param style_weight: Weight for style reward (default: 0.0)
    :param normalize_rewards: Whether to track and normalize shaped rewards (default: False)
    """
    
    def __init__(
        self,
        env: gym.Env,
        posture_weight: float = 0.0,
        min_torso_height: float = 200.0,
        energy_weight: float = 0.0,
        joint_limit_weight: float = 0.0,
        style_weight: float = 0.0,
        normalize_rewards: bool = False,
    ):
        super().__init__(env)
        
        # Reward component weights
        self.posture_weight = posture_weight
        self.min_torso_height = min_torso_height
        self.energy_weight = energy_weight
        self.joint_limit_weight = joint_limit_weight
        self.style_weight = style_weight
        self.normalize_rewards = normalize_rewards
        
        # State tracking
        self.last_action = None
        self.last_obs = None
        
        # Normalization tracking (running mean/std)
        if normalize_rewards:
            self.reward_mean = 0.0
            self.reward_var = 1.0
            self.reward_count = 0
    
    def reset(self, **kwargs) -> tuple:
        """Reset environment and tracking state."""
        obs, info = self.env.reset(**kwargs)
        self.last_action = None
        self.last_obs = obs.copy()
        return obs, info
    
    def step(self, action: int) -> tuple:
        """
        Execute action and compute shaped reward.
        
        :param action: Action to execute
        :return: (observation, shaped_reward, terminated, truncated, info)
        """
        # Execute action in base environment
        obs, base_reward, terminated, truncated, info = self.env.step(action)
        
        # Compute shaped reward components
        shaped_reward = base_reward
        reward_components = {"base": base_reward}
        
        # 1. Posture reward: penalize low torso height (anti-scooting)
        if self.posture_weight > 0:
            posture_reward = self._compute_posture_reward(obs)
            shaped_reward += posture_reward
            reward_components["posture"] = posture_reward
        
        # 2. Energy cost: penalize large action changes (anti-jitter)
        if self.energy_weight > 0 and self.last_action is not None:
            energy_cost = self._compute_energy_cost(action)
            shaped_reward += energy_cost
            reward_components["energy"] = energy_cost
        
        # 3. Joint limit penalty: penalize extreme joint angles
        if self.joint_limit_weight > 0:
            joint_penalty = self._compute_joint_penalty(obs)
            shaped_reward += joint_penalty
            reward_components["joint_limit"] = joint_penalty
        
        # 4. Style reward: bonus for maintaining upright posture
        if self.style_weight > 0:
            style_reward = self._compute_style_reward(obs)
            shaped_reward += style_reward
            reward_components["style"] = style_reward
        
        # Normalize rewards if enabled
        if self.normalize_rewards:
            shaped_reward = self._normalize_reward(shaped_reward)
        
        # Update tracking
        self.last_action = action
        self.last_obs = obs.copy()
        
        # Add reward components to info for logging
        info["reward_components"] = reward_components
        info["shaped_reward"] = shaped_reward
        
        return obs, shaped_reward, terminated, truncated, info
    
    def _compute_posture_reward(self, obs: np.ndarray) -> float:
        """
        Compute posture reward based on torso height.
        
        The torso is body part index 0 (first 5 values in observation).
        Observation format per body part: [pos_x, pos_y, angle, vel_x, vel_y]
        
        We want to penalize when torso y-position (obs[1]) is too low.
        Lower y = lower torso = closer to ground = scooting behavior.
        
        Note: Observations are normalized to [-1, 1]. We need to denormalize
        to get actual pixel coordinates.
        """
        # Get torso y-position (normalized)
        torso_y_norm = obs[1]
        
        # Denormalize using the env's normalizer
        # pos_y range is typically (-10, 10) pixels based on env code
        env_unwrapped = self.env.unwrapped
        torso_y_actual = env_unwrapped.pos_y.denormalize(torso_y_norm)
        
        # Penalize if below minimum height
        if torso_y_actual < self.min_torso_height:
            height_deficit = self.min_torso_height - torso_y_actual
            penalty = -self.posture_weight * (height_deficit / self.min_torso_height)
            return penalty
        
        return 0.0
    
    def _compute_energy_cost(self, action: int) -> float:
        """
        Compute energy cost for action changes.
        
        Penalizes frequent action switching, which creates jittery,
        unrealistic movement and can exploit physics engine errors.
        
        We compute a simple "action change" penalty: if action differs
        from last action, apply a small penalty.
        """
        if action != self.last_action:
            return -self.energy_weight
        return 0.0
    
    def _compute_joint_penalty(self, obs: np.ndarray) -> float:
        """
        Compute penalty for extreme joint angles.
        
        Penalizes the "splits" failure mode where legs extend too far apart.
        
        Observation format: 12 body parts × 5 values each
        Body parts include thighs (indices 3, 4) and calves (indices 5, 6).
        The angle field (index 2 in each 5-tuple) represents rotation.
        
        We penalize when thigh or calf angles exceed safe limits.
        """
        penalty = 0.0
        
        # Check thigh angles (body parts 3 and 4)
        left_thigh_angle = obs[3 * 5 + 2]  # Part 3, field 2 (angle)
        right_thigh_angle = obs[4 * 5 + 2]  # Part 4, field 2 (angle)
        
        # Normalized angles are in [-1, 1], representing actual range [-6, 6] radians
        # Angles beyond ±0.8 (normalized) = ±4.8 radians = extreme splits
        angle_threshold = 0.8
        
        if abs(left_thigh_angle) > angle_threshold:
            penalty -= self.joint_limit_weight * (abs(left_thigh_angle) - angle_threshold)
        
        if abs(right_thigh_angle) > angle_threshold:
            penalty -= self.joint_limit_weight * (abs(right_thigh_angle) - angle_threshold)
        
        return penalty
    
    def _compute_style_reward(self, obs: np.ndarray) -> float:
        """
        Compute style reward for maintaining upright running posture.
        
        Rewards the agent for:
        - Keeping torso upright (angle close to 0)
        - Maintaining forward velocity
        - Having feet at appropriate positions relative to torso
        
        This is a simplified version of DeepMimic-style imitation rewards.
        """
        reward = 0.0
        
        # Reward upright torso (angle close to 0)
        torso_angle = obs[2]  # Torso angle (normalized)
        upright_bonus = self.style_weight * (1.0 - abs(torso_angle))
        reward += upright_bonus
        
        # Reward forward velocity
        torso_vel_x = obs[3]  # Torso x velocity (normalized)
        if torso_vel_x > 0:  # Only reward forward movement
            velocity_bonus = self.style_weight * 0.5 * torso_vel_x
            reward += velocity_bonus
        
        return reward
    
    def _normalize_reward(self, reward: float) -> float:
        """
        Normalize reward using running mean and standard deviation.
        
        This helps stabilize training when shaped rewards have different scales.
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


class ProgressiveRewardShapingWrapper(RewardShapingWrapper):
    """
    Progressive reward shaping that adjusts weights over time.
    
    Gradually reduces shaping rewards as the agent learns, allowing it to
    transition from shaped rewards (easier to learn) to true task rewards
    (optimal performance).
    
    :param env: QWOP environment to wrap
    :param initial_posture_weight: Initial weight for posture reward
    :param initial_energy_weight: Initial weight for energy cost
    :param initial_joint_limit_weight: Initial weight for joint limit penalty
    :param initial_style_weight: Initial weight for style reward
    :param decay_steps: Number of steps over which to decay shaping weights
    :param final_weight_ratio: Final weight as ratio of initial (default: 0.1)
    """
    
    def __init__(
        self,
        env: gym.Env,
        initial_posture_weight: float = 1.0,
        initial_energy_weight: float = 0.1,
        initial_joint_limit_weight: float = 0.5,
        initial_style_weight: float = 0.5,
        decay_steps: int = 1_000_000,
        final_weight_ratio: float = 0.1,
        **kwargs,
    ):
        super().__init__(
            env,
            posture_weight=initial_posture_weight,
            energy_weight=initial_energy_weight,
            joint_limit_weight=initial_joint_limit_weight,
            style_weight=initial_style_weight,
            **kwargs,
        )
        
        self.initial_weights = {
            "posture": initial_posture_weight,
            "energy": initial_energy_weight,
            "joint_limit": initial_joint_limit_weight,
            "style": initial_style_weight,
        }
        self.decay_steps = decay_steps
        self.final_weight_ratio = final_weight_ratio
        self.current_step = 0
    
    def step(self, action: int) -> tuple:
        """Step with progressive weight decay."""
        # Update weights based on progress
        progress = min(1.0, self.current_step / self.decay_steps)
        decay_factor = 1.0 - progress * (1.0 - self.final_weight_ratio)
        
        self.posture_weight = self.initial_weights["posture"] * decay_factor
        self.energy_weight = self.initial_weights["energy"] * decay_factor
        self.joint_limit_weight = self.initial_weights["joint_limit"] * decay_factor
        self.style_weight = self.initial_weights["style"] * decay_factor
        
        self.current_step += 1
        
        return super().step(action)
