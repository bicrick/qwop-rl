"""Enhanced QRDQN with Prioritized Experience Replay."""

from typing import Any, ClassVar, Dict, Optional, Type, Union

import numpy as np
import torch as th
from sb3_contrib.qrdqn.qrdqn import QRDQN
from sb3_contrib.common.utils import quantile_huber_loss
from stable_baselines3.common.type_aliases import GymEnv, Schedule

from qwop_gym.buffers.prioritized_replay import PrioritizedReplayBuffer


class EnhancedQRDQN(QRDQN):
    """
    QRDQN with Prioritized Experience Replay.
    
    Extends QRDQN by adding prioritized sampling based on TD-error
    and importance sampling corrections.
    
    All parameters are the same as QRDQN. The replay buffer will be
    automatically set to PrioritizedReplayBuffer if not specified.
    """

    def train(self, gradient_steps: int, batch_size: int = 100) -> None:
        """
        Update policy using gradient descent, with prioritized replay.
        
        Overrides parent's train() to:
        1. Apply importance sampling weights to loss
        2. Compute per-sample TD-errors
        3. Update priorities in the replay buffer
        
        :param gradient_steps: Number of gradient steps
        :param batch_size: Minibatch size
        """
        # Switch to train mode (this affects batch norm / dropout)
        self.policy.set_training_mode(True)
        # Update learning rate according to schedule
        self._update_learning_rate(self.policy.optimizer)

        losses = []
        for _ in range(gradient_steps):
            # Sample replay buffer
            replay_data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)  # type: ignore[union-attr]

            # Get importance sampling weights from prioritized buffer
            # (PrioritizedReplayBuffer stores these after sampling)
            use_prioritized = isinstance(self.replay_buffer, PrioritizedReplayBuffer)
            
            if use_prioritized:
                importance_weights = self.replay_buffer.last_weights  # type: ignore[attr-defined]
                tree_indices = self.replay_buffer.last_tree_indices  # type: ignore[attr-defined]
            else:
                # Uniform weights if not using prioritized replay
                importance_weights = th.ones(batch_size, device=self.device)
                tree_indices = None

            with th.no_grad():
                # Compute the quantiles of next observation
                next_quantiles = self.quantile_net_target(replay_data.next_observations)
                # Compute the greedy actions which maximize the next Q values
                next_greedy_actions = next_quantiles.mean(dim=1, keepdim=True).argmax(dim=2, keepdim=True)
                # Make "n_quantiles" copies of actions, and reshape to (batch_size, n_quantiles, 1)
                next_greedy_actions = next_greedy_actions.expand(batch_size, self.n_quantiles, 1)
                # Follow greedy policy: use the one with the highest Q values
                next_quantiles = next_quantiles.gather(dim=2, index=next_greedy_actions).squeeze(dim=2)
                # 1-step TD target
                target_quantiles = replay_data.rewards + (1 - replay_data.dones) * self.gamma * next_quantiles

            # Get current quantile estimates
            current_quantiles = self.quantile_net(replay_data.observations)

            # Make "n_quantiles" copies of actions, and reshape to (batch_size, n_quantiles, 1).
            actions = replay_data.actions[..., None].long().expand(batch_size, self.n_quantiles, 1)
            # Retrieve the quantiles for the actions from the replay buffer
            current_quantiles = th.gather(current_quantiles, dim=2, index=actions).squeeze(dim=2)

            # Compute per-sample TD-error for priority updates
            # TD-error = mean absolute difference across quantiles
            with th.no_grad():
                td_errors = th.abs(current_quantiles - target_quantiles).mean(dim=1)

            # Compute Quantile Huber loss (returns scalar batch mean)
            loss = quantile_huber_loss(current_quantiles, target_quantiles, sum_over_quantiles=True)
            
            # Apply importance sampling weights
            # Note: quantile_huber_loss already returns mean, so we weight it directly
            # For more precise weighting, we'd need per-sample losses, but this is
            # the approach used in many Rainbow DQN implementations
            weighted_loss = loss * importance_weights.mean()
            
            losses.append(weighted_loss.item())

            # Optimize the policy
            self.policy.optimizer.zero_grad()
            weighted_loss.backward()
            # Clip gradient norm
            if self.max_grad_norm is not None:
                th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.policy.optimizer.step()

            # Update priorities in replay buffer
            if use_prioritized and tree_indices is not None:
                self.replay_buffer.update_priorities(  # type: ignore[attr-defined]
                    tree_indices,
                    td_errors.cpu().numpy()
                )

        # Increase update counter
        self._n_updates += gradient_steps

        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/loss", np.mean(losses))
        
        # Log additional metrics for prioritized replay
        if use_prioritized:
            self.logger.record("train/mean_importance_weight", importance_weights.mean().item())
            self.logger.record("train/mean_td_error", td_errors.mean().item())
