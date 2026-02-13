"""Curriculum Learning Wrapper for QWOP.

Implements a multi-stage training curriculum that gradually increases difficulty:
1. Stage 1: Basic Stability - Learn to stand without falling
2. Stage 2: Forward Motion - Learn to walk forward
3. Stage 3: Speed Optimization - Maximize velocity

This approach is based on curriculum learning principles where the agent
first masters simple skills before progressing to more complex ones.
"""

import gymnasium as gym
import numpy as np
from typing import Any, Dict, Optional, SupportsFloat, List
from enum import IntEnum


class CurriculumStage(IntEnum):
    """Curriculum training stages."""
    STABILITY = 0      # Learn basic balance
    FORWARD_MOTION = 1  # Learn to move forward
    SPEED_OPTIMIZATION = 2  # Maximize speed


class CurriculumLearningWrapper(gym.Wrapper):
    """
    Wrapper that implements curriculum learning for QWOP.
    
    The curriculum progresses through stages based on performance thresholds:
    
    Stage 1 (Stability):
    - Goal: Stand upright without falling
    - Reward: Based on time survived and maintaining torso angle
    - Progression: Move to stage 2 after N successful episodes (e.g., 50 episodes > 100 steps)
    
    Stage 2 (Forward Motion):
    - Goal: Move forward consistently
    - Reward: Based on distance covered
    - Progression: Move to stage 3 after achieving target distance (e.g., 10m)
    
    Stage 3 (Speed Optimization):
    - Goal: Maximize velocity
    - Reward: Based on speed and final distance
    - Progression: Continue indefinitely
    
    :param env: QWOP environment to wrap
    :param stability_episodes: Episodes to complete stage 1 (default: 100)
    :param stability_threshold: Min steps per episode to progress (default: 100)
    :param motion_episodes: Episodes to complete stage 2 (default: 200)
    :param motion_threshold: Min distance to progress (default: 10.0)
    :param manual_stage: Force a specific stage (None for automatic progression)
    """
    
    def __init__(
        self,
        env: gym.Env,
        stability_episodes: int = 100,
        stability_threshold: float = 100.0,  # steps
        motion_episodes: int = 200,
        motion_threshold: float = 10.0,  # meters
        manual_stage: Optional[int] = None,
    ):
        super().__init__(env)
        
        # Curriculum parameters
        self.stability_episodes = stability_episodes
        self.stability_threshold = stability_threshold
        self.motion_episodes = motion_episodes
        self.motion_threshold = motion_threshold
        self.manual_stage = manual_stage
        
        # Progress tracking
        self.current_stage = CurriculumStage.STABILITY if manual_stage is None else manual_stage
        self.episode_count = 0
        self.successful_episodes = 0
        self.best_distance = 0.0
        self.stage_history: List[Dict] = []
        
        # Episode metrics
        self.episode_steps = 0
        self.episode_distance = 0.0
        self.episode_max_torso_y = 0.0
        
        print(f"[Curriculum] Initialized at stage {self.current_stage.name}")
    
    def reset(self, **kwargs) -> tuple:
        """Reset environment and update curriculum stage."""
        # Update stage history if not first episode
        if self.episode_count > 0:
            self._record_episode()
            if self.manual_stage is None:
                self._check_progression()
        
        self.episode_count += 1
        self.episode_steps = 0
        self.episode_distance = 0.0
        self.episode_max_torso_y = 0.0
        
        obs, info = self.env.reset(**kwargs)
        
        # Add curriculum info
        info["curriculum_stage"] = int(self.current_stage)
        info["curriculum_stage_name"] = self.current_stage.name
        
        return obs, info
    
    def step(self, action: int) -> tuple:
        """
        Execute action and compute curriculum-appropriate reward.
        
        :param action: Action to execute
        :return: (observation, reward, terminated, truncated, info)
        """
        obs, base_reward, terminated, truncated, info = self.env.step(action)
        
        self.episode_steps += 1
        self.episode_distance = info.get("distance", 0.0)
        
        # Track torso height for stability assessment
        torso_y = obs[1]  # Normalized torso y position
        env_unwrapped = self.env.unwrapped
        torso_y_actual = env_unwrapped.pos_y.denormalize(torso_y)
        self.episode_max_torso_y = max(self.episode_max_torso_y, torso_y_actual)
        
        # Compute curriculum-specific reward
        curriculum_reward = self._compute_curriculum_reward(
            obs, base_reward, terminated, info
        )
        
        # Add curriculum info
        info["curriculum_stage"] = int(self.current_stage)
        info["curriculum_stage_name"] = self.current_stage.name
        info["curriculum_reward"] = curriculum_reward
        info["base_reward"] = base_reward
        
        return obs, curriculum_reward, terminated, truncated, info
    
    def _compute_curriculum_reward(
        self,
        obs: np.ndarray,
        base_reward: float,
        terminated: bool,
        info: Dict,
    ) -> float:
        """Compute stage-specific reward."""
        
        if self.current_stage == CurriculumStage.STABILITY:
            return self._stability_reward(obs, terminated, info)
        
        elif self.current_stage == CurriculumStage.FORWARD_MOTION:
            return self._motion_reward(obs, base_reward, terminated, info)
        
        else:  # SPEED_OPTIMIZATION
            return base_reward  # Use original reward for speed optimization
    
    def _stability_reward(
        self,
        obs: np.ndarray,
        terminated: bool,
        info: Dict,
    ) -> float:
        """
        Stage 1: Reward for maintaining stability.
        
        Components:
        - Survival bonus: +1 per step alive
        - Upright bonus: +0.5 for keeping torso upright
        - Height bonus: +0.5 for maintaining torso height
        - Fall penalty: -10 for falling
        """
        reward = 0.0
        
        # Survival bonus (encourages staying alive longer)
        reward += 1.0
        
        # Upright torso bonus
        torso_angle = obs[2]  # Normalized torso angle
        upright_bonus = 0.5 * (1.0 - abs(torso_angle))
        reward += upright_bonus
        
        # Height maintenance bonus
        torso_y = obs[1]  # Normalized torso y
        if torso_y > 0.5:  # Above median height
            reward += 0.5
        
        # Fall penalty
        if terminated and not info.get("is_success", False):
            reward -= 10.0
        
        return reward
    
    def _motion_reward(
        self,
        obs: np.ndarray,
        base_reward: float,
        terminated: bool,
        info: Dict,
    ) -> float:
        """
        Stage 2: Reward for forward motion.
        
        Uses base reward (velocity-based) but with reduced fall penalty
        to encourage exploration of forward movement.
        """
        reward = base_reward
        
        # Reduce fall penalty to encourage trying to move forward
        # (agent might fall while learning to walk)
        if terminated and not info.get("is_success", False):
            # Original penalty is -10, make it -5 in this stage
            reward += 5.0
        
        # Add small bonus for maintaining upright posture while moving
        torso_angle = obs[2]
        if abs(torso_angle) < 0.3:  # Reasonably upright
            reward += 0.5
        
        return reward
    
    def _record_episode(self) -> None:
        """Record episode statistics."""
        episode_data = {
            "episode": self.episode_count,
            "stage": self.current_stage.name,
            "steps": self.episode_steps,
            "distance": self.episode_distance,
            "max_torso_height": self.episode_max_torso_y,
        }
        self.stage_history.append(episode_data)
        
        # Update best distance
        if self.episode_distance > self.best_distance:
            self.best_distance = self.episode_distance
    
    def _check_progression(self) -> None:
        """Check if agent should progress to next stage."""
        
        if self.current_stage == CurriculumStage.STABILITY:
            # Check if agent can consistently stay alive
            recent_episodes = self.stage_history[-min(50, len(self.stage_history)):]
            
            if len(recent_episodes) >= 50:
                avg_steps = np.mean([ep["steps"] for ep in recent_episodes])
                
                if avg_steps >= self.stability_threshold:
                    self._progress_stage()
        
        elif self.current_stage == CurriculumStage.FORWARD_MOTION:
            # Check if agent can move forward consistently
            recent_episodes = self.stage_history[-min(50, len(self.stage_history)):]
            
            if len(recent_episodes) >= 50:
                avg_distance = np.mean([ep["distance"] for ep in recent_episodes])
                
                if avg_distance >= self.motion_threshold:
                    self._progress_stage()
    
    def _progress_stage(self) -> None:
        """Progress to next curriculum stage."""
        old_stage = self.current_stage
        
        if self.current_stage == CurriculumStage.STABILITY:
            self.current_stage = CurriculumStage.FORWARD_MOTION
        elif self.current_stage == CurriculumStage.FORWARD_MOTION:
            self.current_stage = CurriculumStage.SPEED_OPTIMIZATION
        else:
            return  # Already at final stage
        
        print(f"\n{'='*80}")
        print(f"[Curriculum] STAGE PROGRESSION!")
        print(f"  From: {old_stage.name}")
        print(f"  To:   {self.current_stage.name}")
        print(f"  Episode: {self.episode_count}")
        print(f"  Best distance: {self.best_distance:.2f}m")
        print(f"{'='*80}\n")
    
    def get_curriculum_stats(self) -> Dict:
        """Get current curriculum statistics."""
        recent = self.stage_history[-100:] if self.stage_history else []
        
        return {
            "current_stage": self.current_stage.name,
            "episode_count": self.episode_count,
            "successful_episodes": self.successful_episodes,
            "best_distance": self.best_distance,
            "recent_avg_steps": np.mean([ep["steps"] for ep in recent]) if recent else 0,
            "recent_avg_distance": np.mean([ep["distance"] for ep in recent]) if recent else 0,
        }


class PhysicsModifiedCurriculumWrapper(CurriculumLearningWrapper):
    """
    Advanced curriculum wrapper that modifies physics parameters.
    
    WARNING: This requires modifications to the QWOP environment to support
    dynamic physics parameter adjustment. Use CurriculumLearningWrapper for
    a reward-only curriculum that works with the standard environment.
    
    Stage 1: Reduced gravity to make balancing easier
    Stage 2: Normal gravity with stability assistance
    Stage 3: Full normal physics
    
    :param env: QWOP environment (must support set_gravity method)
    :param stage1_gravity_multiplier: Gravity multiplier for stage 1 (default: 0.5)
    :param stage2_gravity_multiplier: Gravity multiplier for stage 2 (default: 0.8)
    """
    
    def __init__(
        self,
        env: gym.Env,
        stage1_gravity_multiplier: float = 0.5,
        stage2_gravity_multiplier: float = 0.8,
        **kwargs,
    ):
        super().__init__(env, **kwargs)
        
        self.stage1_gravity = stage1_gravity_multiplier
        self.stage2_gravity = stage2_gravity_multiplier
        
        # Check if environment supports physics modification
        if not hasattr(env.unwrapped, "set_gravity"):
            print("[Warning] Environment does not support set_gravity()")
            print("          Physics modification will be skipped.")
            print("          Consider using CurriculumLearningWrapper instead.")
            self.physics_supported = False
        else:
            self.physics_supported = True
            self._apply_physics()
    
    def _apply_physics(self) -> None:
        """Apply stage-specific physics parameters."""
        if not self.physics_supported:
            return
        
        if self.current_stage == CurriculumStage.STABILITY:
            gravity_mult = self.stage1_gravity
        elif self.current_stage == CurriculumStage.FORWARD_MOTION:
            gravity_mult = self.stage2_gravity
        else:
            gravity_mult = 1.0  # Normal gravity
        
        self.env.unwrapped.set_gravity(gravity_mult)
        print(f"[Curriculum] Applied gravity multiplier: {gravity_mult}")
    
    def _progress_stage(self) -> None:
        """Progress stage and update physics."""
        super()._progress_stage()
        self._apply_physics()
