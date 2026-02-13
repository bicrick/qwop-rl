"""Discrete Soft Actor-Critic (SAC) for discrete action spaces.

Based on "Soft Actor-Critic for Discrete Action Settings" (Christodoulou, 2019)
https://arxiv.org/abs/1910.07207

This implementation adapts SAC for discrete action spaces like QWOP by:
- Using Softmax policy instead of Gaussian
- Computing Q-values for all actions in parallel
- Entropy computed as -sum(p * log(p))
- No reparameterization trick needed
"""

from typing import Any, ClassVar, Dict, List, Optional, Tuple, Type, Union

import numpy as np
import torch as th
import torch.nn.functional as F
from gymnasium import spaces
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.off_policy_algorithm import OffPolicyAlgorithm
from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.type_aliases import GymEnv, Schedule, MaybeCallback
from stable_baselines3.common.utils import polyak_update
from torch import nn

from qwop_gym.buffers.prioritized_replay import PrioritizedReplayBuffer


class DiscreteSACPolicy(BasePolicy):
    """
    Policy network for Discrete SAC.
    
    Consists of:
    - Actor: Maps state to action probabilities (Softmax)
    - Twin Critics: Map state to Q-values for all actions
    """
    
    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Discrete,
        lr_schedule: Schedule,
        net_arch: Optional[List[int]] = None,
        activation_fn: Type[nn.Module] = nn.ReLU,
        **kwargs,
    ):
        super().__init__(
            observation_space,
            action_space,
            features_extractor_class=None,
            **kwargs,
        )
        
        if net_arch is None:
            net_arch = [256, 256]
        
        self.net_arch = net_arch
        
        # Handle activation_fn as string or class
        if isinstance(activation_fn, str):
            # Parse string like "torch.nn.ReLU" to actual class
            import importlib
            module_name, class_name = activation_fn.rsplit(".", 1)
            module = importlib.import_module(module_name)
            self.activation_fn = getattr(module, class_name)
        else:
            self.activation_fn = activation_fn
        
        self.n_actions = int(action_space.n)
        
        # Build networks
        self.actor = self._build_actor()
        self.critic_1 = self._build_critic()
        self.critic_2 = self._build_critic()
        self.critic_1_target = self._build_critic()
        self.critic_2_target = self._build_critic()
        
        # Copy parameters to target networks
        self.critic_1_target.load_state_dict(self.critic_1.state_dict())
        self.critic_2_target.load_state_dict(self.critic_2.state_dict())
        
        # Initialize weights with orthogonal initialization
        self._init_weights()
    
    def _build_actor(self) -> nn.Module:
        """Build actor network: state -> action probabilities."""
        layers = []
        input_dim = int(np.prod(self.observation_space.shape))
        
        for hidden_dim in self.net_arch:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(self.activation_fn())
            input_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(input_dim, self.n_actions))
        
        return nn.Sequential(*layers)
    
    def _build_critic(self) -> nn.Module:
        """Build critic network: state -> Q-values for all actions."""
        layers = []
        input_dim = int(np.prod(self.observation_space.shape))
        
        for hidden_dim in self.net_arch:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(self.activation_fn())
            input_dim = hidden_dim
        
        # Output Q-values for all actions
        layers.append(nn.Linear(input_dim, self.n_actions))
        
        return nn.Sequential(*layers)
    
    def _init_weights(self):
        """Initialize network weights with orthogonal initialization."""
        for module in [self.actor, self.critic_1, self.critic_2]:
            for layer in module:
                if isinstance(layer, nn.Linear):
                    nn.init.orthogonal_(layer.weight, gain=np.sqrt(2))
                    nn.init.constant_(layer.bias, 0.0)
    
    def forward(self, obs: th.Tensor, deterministic: bool = False) -> th.Tensor:
        """
        Forward pass: select action given observation.
        
        :param obs: Observation
        :param deterministic: If True, return argmax action; if False, sample from policy
        :return: Action
        """
        logits = self.actor(obs)
        
        if deterministic:
            return th.argmax(logits, dim=1)
        else:
            # Sample from categorical distribution
            probs = F.softmax(logits, dim=1)
            dist = th.distributions.Categorical(probs)
            return dist.sample()
    
    def get_action_probs(self, obs: th.Tensor) -> Tuple[th.Tensor, th.Tensor]:
        """
        Get action probabilities and log probabilities.
        
        :param obs: Observation
        :return: (action_probs, log_action_probs)
        """
        logits = self.actor(obs)
        action_probs = F.softmax(logits, dim=1)
        log_action_probs = F.log_softmax(logits, dim=1)
        return action_probs, log_action_probs
    
    def evaluate_actions(self, obs: th.Tensor) -> Tuple[th.Tensor, th.Tensor, th.Tensor]:
        """
        Evaluate Q-values for all actions and compute policy distribution.
        
        :param obs: Observation
        :return: (q1_values, q2_values, action_probs)
        """
        q1_values = self.critic_1(obs)
        q2_values = self.critic_2(obs)
        action_probs, _ = self.get_action_probs(obs)
        return q1_values, q2_values, action_probs
    
    def _predict(self, observation: th.Tensor, deterministic: bool = False) -> th.Tensor:
        return self.forward(observation, deterministic)


class DiscreteSAC(OffPolicyAlgorithm):
    """
    Discrete Soft Actor-Critic (SAC) algorithm.
    
    SAC with maximum entropy reinforcement learning for discrete action spaces.
    
    :param policy: Policy class (DiscreteSACPolicy)
    :param env: Gymnasium environment
    :param learning_rate: Learning rate for actor and critics
    :param buffer_size: Replay buffer size
    :param learning_starts: Number of steps before training starts
    :param batch_size: Minibatch size
    :param tau: Soft update coefficient for target networks
    :param gamma: Discount factor
    :param train_freq: Update frequency
    :param gradient_steps: Number of gradient steps per update
    :param ent_coef: Entropy regularization coefficient (or "auto" for automatic tuning)
    :param target_entropy_ratio: Target entropy ratio for automatic tuning (default 0.98)
    :param use_per: Use Prioritized Experience Replay
    :param per_alpha: PER alpha parameter
    :param per_beta_start: PER beta start value
    :param per_beta_frames: PER beta annealing frames
    :param per_eps: PER epsilon for numerical stability
    """
    
    policy_aliases: ClassVar[Dict[str, Type[BasePolicy]]] = {
        "MlpPolicy": DiscreteSACPolicy,
    }
    
    def __init__(
        self,
        policy: Union[str, Type[DiscreteSACPolicy]],
        env: Union[GymEnv, str],
        learning_rate: Union[float, Schedule] = 3e-4,
        buffer_size: int = 500000,
        learning_starts: int = 50000,
        batch_size: int = 256,
        tau: float = 0.005,
        gamma: float = 0.99,
        train_freq: Union[int, Tuple[int, str]] = 1,
        gradient_steps: int = 1,
        ent_coef: Union[str, float] = "auto",
        target_entropy_ratio: float = 0.98,
        use_per: bool = False,
        per_alpha: float = 0.6,
        per_beta_start: float = 0.4,
        per_beta_frames: int = 100000,
        per_eps: float = 1e-6,
        replay_buffer_class: Optional[Type[ReplayBuffer]] = None,
        replay_buffer_kwargs: Optional[Dict[str, Any]] = None,
        policy_kwargs: Optional[Dict[str, Any]] = None,
        tensorboard_log: Optional[str] = None,
        verbose: int = 0,
        device: Union[th.device, str] = "auto",
        seed: Optional[int] = None,
        _init_setup_model: bool = True,
    ):
        super().__init__(
            policy,
            env,
            learning_rate,
            buffer_size,
            learning_starts,
            batch_size,
            tau,
            gamma,
            train_freq,
            gradient_steps,
            action_noise=None,
            replay_buffer_class=replay_buffer_class,
            replay_buffer_kwargs=replay_buffer_kwargs,
            policy_kwargs=policy_kwargs,
            stats_window_size=100,
            tensorboard_log=tensorboard_log,
            verbose=verbose,
            device=device,
            seed=seed,
            sde_support=False,
            optimize_memory_usage=False,
            supported_action_spaces=(spaces.Discrete,),
            support_multi_env=True,
        )
        
        self.target_entropy_ratio = target_entropy_ratio
        self.use_per = use_per
        
        # Handle PER setup
        if use_per:
            if replay_buffer_class is None:
                self.replay_buffer_class = PrioritizedReplayBuffer
            if replay_buffer_kwargs is None:
                replay_buffer_kwargs = {}
            replay_buffer_kwargs.update({
                "alpha": per_alpha,
                "beta_start": per_beta_start,
                "beta_frames": per_beta_frames,
                "eps": per_eps,
            })
            self.replay_buffer_kwargs = replay_buffer_kwargs
        
        # Entropy coefficient (temperature parameter)
        if ent_coef == "auto":
            # Automatic entropy tuning
            # Target entropy: -log(1/|A|) * target_entropy_ratio
            n_actions = int(self.action_space.n)
            self.target_entropy = -np.log(1.0 / n_actions) * target_entropy_ratio
            self.log_ent_coef = th.log(th.ones(1, device=self.device)).requires_grad_(True)
            self.ent_coef_optimizer = th.optim.Adam([self.log_ent_coef], lr=learning_rate)
            self.ent_coef_auto = True
        else:
            self.ent_coef_auto = False
            self.ent_coef = ent_coef
        
        if _init_setup_model:
            self._setup_model()
    
    def _setup_model(self) -> None:
        super()._setup_model()
        self._create_aliases()
        
        # Create separate optimizers for actor and critics
        self.actor_optimizer = th.optim.Adam(self.policy.actor.parameters(), lr=self.lr_schedule(1))
        self.critic_optimizer = th.optim.Adam(
            list(self.policy.critic_1.parameters()) + list(self.policy.critic_2.parameters()),
            lr=self.lr_schedule(1)
        )
    
    def _create_aliases(self) -> None:
        """Create aliases for networks."""
        self.actor = self.policy.actor
        self.critic_1 = self.policy.critic_1
        self.critic_2 = self.policy.critic_2
        self.critic_1_target = self.policy.critic_1_target
        self.critic_2_target = self.policy.critic_2_target
    
    def train(self, gradient_steps: int, batch_size: int = 64) -> None:
        """
        Update policy using gradient descent.
        
        :param gradient_steps: Number of gradient steps
        :param batch_size: Minibatch size
        """
        # Switch to train mode
        self.policy.set_training_mode(True)
        
        # Update learning rate
        self._update_learning_rate([self.actor_optimizer, self.critic_optimizer])
        
        actor_losses = []
        critic_losses = []
        ent_coef_losses = []
        ent_coefs = []
        
        for gradient_step in range(gradient_steps):
            # Sample from replay buffer
            replay_data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)
            
            # Get importance sampling weights if using PER
            use_per = isinstance(self.replay_buffer, PrioritizedReplayBuffer)
            if use_per:
                importance_weights = self.replay_buffer.last_weights
                tree_indices = self.replay_buffer.last_tree_indices
            else:
                importance_weights = th.ones(batch_size, device=self.device)
                tree_indices = None
            
            # === Update Critics ===
            with th.no_grad():
                # Get next action probabilities and log probs
                next_action_probs, next_log_probs = self.policy.get_action_probs(replay_data.next_observations)
                
                # Compute target Q-values
                next_q1_target = self.critic_1_target(replay_data.next_observations)
                next_q2_target = self.critic_2_target(replay_data.next_observations)
                
                # Take minimum to reduce overestimation bias
                next_q_target = th.min(next_q1_target, next_q2_target)
                
                # Add entropy term: Q = r + γ * (min_Q - α * log π)
                ent_coef = th.exp(self.log_ent_coef.detach()) if self.ent_coef_auto else self.ent_coef
                next_q_value = next_action_probs * (next_q_target - ent_coef * next_log_probs)
                next_q_value = next_q_value.sum(dim=1, keepdim=True)
                
                # Compute TD target
                target_q_value = replay_data.rewards + (1 - replay_data.dones) * self.gamma * next_q_value
            
            # Get current Q-values
            current_q1 = self.critic_1(replay_data.observations)
            current_q2 = self.critic_2(replay_data.observations)
            
            # Gather Q-values for taken actions
            current_q1 = current_q1.gather(1, replay_data.actions.long())
            current_q2 = current_q2.gather(1, replay_data.actions.long())
            
            # Compute TD errors for PER priority updates
            with th.no_grad():
                td_errors = th.abs(current_q1 - target_q_value).squeeze(1)
            
            # Compute critic loss
            critic_1_loss = F.mse_loss(current_q1, target_q_value, reduction='none')
            critic_2_loss = F.mse_loss(current_q2, target_q_value, reduction='none')
            
            # Apply importance sampling weights
            critic_1_loss = (critic_1_loss * importance_weights.unsqueeze(1)).mean()
            critic_2_loss = (critic_2_loss * importance_weights.unsqueeze(1)).mean()
            critic_loss = critic_1_loss + critic_2_loss
            
            critic_losses.append(critic_loss.item())
            
            # Optimize critics
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()
            
            # Update priorities in replay buffer
            if use_per and tree_indices is not None:
                self.replay_buffer.update_priorities(tree_indices, td_errors.cpu().numpy())
            
            # === Update Actor ===
            # Get current action probabilities
            action_probs, log_probs = self.policy.get_action_probs(replay_data.observations)
            
            # Get Q-values from critics (no gradient through critics)
            with th.no_grad():
                q1_values = self.critic_1(replay_data.observations)
                q2_values = self.critic_2(replay_data.observations)
                q_values = th.min(q1_values, q2_values)
            
            # Compute actor loss: maximize Q - α * entropy
            ent_coef = th.exp(self.log_ent_coef.detach()) if self.ent_coef_auto else self.ent_coef
            actor_loss = (action_probs * (ent_coef * log_probs - q_values)).sum(dim=1).mean()
            
            actor_losses.append(actor_loss.item())
            
            # Optimize actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()
            
            # === Update Temperature (Entropy Coefficient) ===
            if self.ent_coef_auto:
                # Compute current entropy
                with th.no_grad():
                    action_probs, log_probs = self.policy.get_action_probs(replay_data.observations)
                    entropy = -(action_probs * log_probs).sum(dim=1).mean()
                
                # Entropy loss: minimize difference from target entropy
                ent_coef_loss = -(self.log_ent_coef * (entropy - self.target_entropy)).mean()
                
                ent_coef_losses.append(ent_coef_loss.item())
                ent_coefs.append(th.exp(self.log_ent_coef.detach()).item())
                
                # Optimize temperature
                self.ent_coef_optimizer.zero_grad()
                ent_coef_loss.backward()
                self.ent_coef_optimizer.step()
            else:
                ent_coefs.append(self.ent_coef)
            
            # === Soft Update Target Networks ===
            polyak_update(self.critic_1.parameters(), self.critic_1_target.parameters(), self.tau)
            polyak_update(self.critic_2.parameters(), self.critic_2_target.parameters(), self.tau)
        
        # Update counter
        self._n_updates += gradient_steps
        
        # Log metrics
        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/actor_loss", np.mean(actor_losses))
        self.logger.record("train/critic_loss", np.mean(critic_losses))
        self.logger.record("train/ent_coef", np.mean(ent_coefs))
        
        if self.ent_coef_auto:
            self.logger.record("train/ent_coef_loss", np.mean(ent_coef_losses))
            self.logger.record("train/target_entropy", self.target_entropy)
        
        if use_per:
            self.logger.record("train/mean_importance_weight", importance_weights.mean().item())
            self.logger.record("train/mean_td_error", td_errors.mean().item())
    
    def learn(
        self,
        total_timesteps: int,
        callback: MaybeCallback = None,
        log_interval: int = 4,
        tb_log_name: str = "DiscreteSAC",
        reset_num_timesteps: bool = True,
        progress_bar: bool = False,
    ):
        return super().learn(
            total_timesteps=total_timesteps,
            callback=callback,
            log_interval=log_interval,
            tb_log_name=tb_log_name,
            reset_num_timesteps=reset_num_timesteps,
            progress_bar=progress_bar,
        )
    
    def _excluded_save_params(self) -> List[str]:
        return super()._excluded_save_params() + ["actor", "critic_1", "critic_2", "critic_1_target", "critic_2_target"]
    
    def _get_torch_save_params(self) -> Tuple[List[str], List[str]]:
        state_dicts = ["policy", "actor_optimizer", "critic_optimizer"]
        
        if self.ent_coef_auto:
            state_dicts.extend(["log_ent_coef", "ent_coef_optimizer"])
        
        return state_dicts, []
