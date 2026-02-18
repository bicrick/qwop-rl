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

"""
DQN from Demonstrations (DQNfD) Callback for Stable-Baselines3.

This callback injects expert demonstrations into the replay buffer during
training, following Wesley Liao's approach to break through the knee-scraping
plateau in QWOP.

Based on: https://wesleyliao.medium.com/achieving-super-human-performance-in-qwop-9a4b968c4cc9
"""

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback
from typing import Optional


class DQNfDCallback(BaseCallback):
    """
    Injects expert demonstrations into the replay buffer during training.
    
    This implements the hybrid replay buffer approach from Wesley Liao's
    DQNfD agent, where expert transitions are mixed with the agent's own
    experience during training (not just pre-training).
    
    The key insight: continuously exposing the agent to expert behavior
    during training prevents it from forgetting good techniques while it
    explores variations.
    
    Args:
        demo_file: Path to .npz file containing demonstrations
            Expected keys: 'obs', 'actions', 'rewards', 'next_obs', 'dones'
        injection_ratio: Fraction of steps that inject expert transitions
            0.5 = inject 1 expert transition per 2 agent steps (Wesley's ratio)
            1.0 = inject 1 expert transition per agent step
            0.25 = inject 1 expert transition per 4 agent steps
        prefill_count: Number of demo transitions to inject into the replay
            buffer BEFORE any agent steps (at training_start). Cycles through
            demos as many times as needed. Use this to ensure the buffer is
            populated with correctly-scaled rewards before training begins.
            0 = no pre-fill (default).
        verbose: Whether to print injection progress (default: 1)
    
    Example:
        >>> callback = DQNfDCallback(
        ...     demo_file="data/expert_demos.npz",
        ...     injection_ratio=0.5,  # 50/50 mix during training
        ...     prefill_count=300000, # fill entire buffer before training
        ... )
        >>> model.learn(total_timesteps=10_000_000, callback=callback)
    """
    
    def __init__(
        self,
        demo_file: str,
        injection_ratio: float = 0.5,
        prefill_count: int = 0,
        verbose: int = 1,
    ):
        super().__init__(verbose)
        
        # Load demonstrations
        print(f"Loading demonstrations from {demo_file}")
        demos = np.load(demo_file)
        
        self.demo_obs = demos['obs']
        self.demo_actions = demos['actions']
        self.demo_rewards = demos['rewards']
        self.demo_next_obs = demos['next_obs']
        self.demo_dones = demos['dones']
        
        self.n_demos = len(self.demo_actions)
        self.demo_idx = 0
        self.demo_epoch = 0
        self.prefill_count = prefill_count
        
        # Injection parameters
        self.injection_ratio = injection_ratio
        self.injection_interval = max(1, int(1 / injection_ratio)) if injection_ratio > 0 else 0
        
        # Statistics
        self.total_injections = 0
        self.last_log_step = 0
        
        print(f"Loaded {self.n_demos} demonstration transitions")
        if prefill_count > 0:
            print(f"Pre-fill count: {prefill_count} (will cycle {prefill_count // self.n_demos + 1}x through demos)")
        if injection_ratio > 0:
            print(f"Injection ratio: {injection_ratio} (1 demo per {self.injection_interval} steps during training)")
    
    def _on_training_start(self) -> None:
        """Called at the beginning of training - pre-fills the replay buffer."""
        if not hasattr(self.model, 'replay_buffer') or self.model.replay_buffer is None:
            raise ValueError(
                "DQNfDCallback requires a model with a replay buffer. "
                "This callback is compatible with DQN, QRDQN, and similar off-policy algorithms."
            )
        
        if self.prefill_count > 0:
            buf_size = self.model.replay_buffer.buffer_size
            actual_fill = min(self.prefill_count, buf_size)
            print(f"\nPre-filling replay buffer with {actual_fill} demo transitions "
                  f"(buffer capacity: {buf_size})...")
            
            for i in range(actual_fill):
                idx = i % self.n_demos
                self._inject_at(idx)
                
                if self.verbose > 0 and (i + 1) % 50000 == 0:
                    print(f"  Pre-fill progress: {i + 1}/{actual_fill}")
            
            print(f"Pre-fill complete. Buffer is now {actual_fill / buf_size * 100:.0f}% full "
                  f"with demo transitions.\n")
        
        if self.verbose > 0:
            print(f"DQNfD training started:")
            print(f"  Replay buffer size: {self.model.replay_buffer.buffer_size}")
            print(f"  Demonstrations: {self.n_demos} transitions")
            if self.injection_interval > 0:
                print(f"  Injection interval: {self.injection_interval} steps")
    
    def _on_step(self) -> bool:
        """
        Called after each environment step.
        
        Injects expert demonstrations into the replay buffer at the configured
        injection_ratio. Skipped if injection_ratio is 0.
        
        Returns:
            True to continue training, False to stop
        """
        if self.injection_interval > 0 and self.num_timesteps % self.injection_interval == 0:
            self._inject_demonstration()
            self.total_injections += 1
        
        # Log progress periodically
        if self.verbose > 0 and self.num_timesteps - self.last_log_step >= 50000:
            self._log_progress()
            self.last_log_step = self.num_timesteps
        
        return True
    
    def _inject_at(self, idx: int) -> None:
        """Inject the demonstration at index `idx` into the replay buffer."""
        self.model.replay_buffer.add(
            obs=self.demo_obs[idx].reshape(1, -1),
            next_obs=self.demo_next_obs[idx].reshape(1, -1),
            action=np.array([self.demo_actions[idx]]),
            reward=np.array([self.demo_rewards[idx]]),
            done=np.array([self.demo_dones[idx]]),
            infos=[{}],
        )

    def _inject_demonstration(self) -> None:
        """Inject the next demo transition into the replay buffer and advance the cursor."""
        self._inject_at(self.demo_idx)
        
        self.demo_idx += 1
        if self.demo_idx >= self.n_demos:
            self.demo_idx = 0
            self.demo_epoch += 1
            if self.verbose > 0:
                print(f"  [DQNfD] Completed epoch {self.demo_epoch} of demonstration data")
    
    def _log_progress(self) -> None:
        """Log injection statistics."""
        agent_steps = self.num_timesteps
        demo_ratio = self.total_injections / agent_steps if agent_steps > 0 else 0
        
        print(f"  [DQNfD] Step {agent_steps}: "
              f"injected {self.total_injections} demos "
              f"(ratio: {demo_ratio:.3f})")
    
    def _on_training_end(self) -> None:
        """Called at the end of training."""
        if self.verbose > 0:
            print(f"\nDQNfD training complete:")
            print(f"  Total agent steps: {self.num_timesteps}")
            print(f"  Total demonstrations injected: {self.total_injections}")
            print(f"  Final demo/agent ratio: {self.total_injections / self.num_timesteps:.3f}")
            print(f"  Completed {self.demo_epoch} epochs of demonstration data")


class ProgressiveDQNfDCallback(DQNfDCallback):
    """
    Progressive DQNfD callback that gradually reduces demonstration injection.
    
    Starts with high injection ratio (heavy expert guidance) and gradually
    reduces it over training (more self-play). This can help the agent
    transition from imitation to innovation.
    
    Args:
        demo_file: Path to .npz file containing demonstrations
        initial_injection_ratio: Starting injection ratio (default: 0.5)
        final_injection_ratio: Final injection ratio (default: 0.1)
        decay_steps: Number of steps over which to decay (default: 5M)
        verbose: Whether to print injection progress (default: 1)
    
    Example:
        >>> callback = ProgressiveDQNfDCallback(
        ...     demo_file="data/expert_demos.npz",
        ...     initial_injection_ratio=0.5,  # Start at 50/50
        ...     final_injection_ratio=0.1,    # End at 10/90
        ...     decay_steps=5_000_000,
        ... )
    """
    
    def __init__(
        self,
        demo_file: str,
        initial_injection_ratio: float = 0.5,
        final_injection_ratio: float = 0.1,
        decay_steps: int = 5_000_000,
        verbose: int = 1,
    ):
        # Initialize with initial ratio
        super().__init__(demo_file, initial_injection_ratio, verbose)
        
        self.initial_injection_ratio = initial_injection_ratio
        self.final_injection_ratio = final_injection_ratio
        self.decay_steps = decay_steps
    
    def _on_step(self) -> bool:
        """Update injection ratio based on training progress."""
        # Calculate current injection ratio
        progress = min(1.0, self.num_timesteps / self.decay_steps)
        current_ratio = (
            self.initial_injection_ratio
            - progress * (self.initial_injection_ratio - self.final_injection_ratio)
        )
        
        # Update injection interval
        self.injection_ratio = current_ratio
        self.injection_interval = max(1, int(1 / current_ratio))
        
        # Call parent's _on_step (which does the actual injection)
        return super()._on_step()
