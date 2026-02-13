"""Prioritized Experience Replay Buffer for stable-baselines3."""

from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch as th
from gymnasium import spaces
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.type_aliases import ReplayBufferSamples
from stable_baselines3.common.vec_env import VecNormalize


class SumTree:
    """
    Sum Tree data structure for efficient prioritized sampling.
    
    Provides O(log N) updates and sampling by storing priorities in a binary tree
    where each parent node contains the sum of its children.
    
    Based on: https://arxiv.org/abs/1511.05952 (Prioritized Experience Replay)
    """

    def __init__(self, size: int):
        """
        Initialize Sum Tree.
        
        :param size: Maximum number of leaf nodes (buffer capacity)
        """
        self.nodes = np.zeros(2 * size - 1)  # Total nodes in complete binary tree
        self.data = np.zeros(size, dtype=int)  # Stores data indices
        self.size = size  # Maximum capacity
        self.count = 0  # Current position for new data
        self.real_size = 0  # Actual number of elements stored

    @property
    def total(self) -> float:
        """Get total sum of all priorities."""
        return self.nodes[0]

    def update(self, data_idx: int, value: float) -> None:
        """
        Update priority value for a specific data index.
        
        :param data_idx: Index in the data array (0 to size-1)
        :param value: New priority value
        """
        idx = data_idx + self.size - 1  # Convert to tree index
        change = value - self.nodes[idx]
        self.nodes[idx] = value

        # Propagate change up the tree
        parent = (idx - 1) // 2
        while parent >= 0:
            self.nodes[parent] += change
            if parent == 0:
                break
            parent = (parent - 1) // 2

    def add(self, value: float, data: int) -> None:
        """
        Add new data with priority value.
        
        :param value: Priority value
        :param data: Data index to store
        """
        self.data[self.count] = data
        self.update(self.count, value)
        self.count = (self.count + 1) % self.size
        self.real_size = min(self.size, self.real_size + 1)

    def get(self, cumsum: float) -> tuple:
        """
        Retrieve data index and priority for a given cumulative sum.
        
        :param cumsum: Target cumulative sum
        :return: (data_idx, priority, data_value)
        """
        assert cumsum <= self.total, f"cumsum {cumsum} > total {self.total}"

        idx = 0
        while 2 * idx + 1 < len(self.nodes):
            left = 2 * idx + 1
            right = 2 * idx + 2

            if cumsum <= self.nodes[left]:
                idx = left
            else:
                idx = right
                cumsum -= self.nodes[left]

        data_idx = idx - self.size + 1
        return data_idx, self.nodes[idx], self.data[data_idx]


class PrioritizedReplayBuffer(ReplayBuffer):
    """
    Prioritized Experience Replay Buffer.
    
    Samples transitions with probability proportional to their TD error,
    as described in https://arxiv.org/abs/1511.05952
    
    :param buffer_size: Max number of element in the buffer
    :param observation_space: Observation space
    :param action_space: Action space
    :param device: PyTorch device
    :param n_envs: Number of parallel environments
    :param optimize_memory_usage: Enable memory efficient variant
    :param handle_timeout_termination: Handle timeout termination separately
    :param alpha: Prioritization exponent (0 = uniform, 1 = full prioritization)
    :param beta_start: Initial importance sampling exponent
    :param beta_frames: Number of frames over which to anneal beta to 1.0
    :param eps: Small positive constant to prevent zero priorities
    """

    def __init__(
        self,
        buffer_size: int,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        device: Union[th.device, str] = "auto",
        n_envs: int = 1,
        optimize_memory_usage: bool = False,
        handle_timeout_termination: bool = True,
        alpha: float = 0.6,
        beta_start: float = 0.4,
        beta_frames: int = 100000,
        eps: float = 1e-6,
    ):
        super().__init__(
            buffer_size,
            observation_space,
            action_space,
            device,
            n_envs,
            optimize_memory_usage,
            handle_timeout_termination,
        )

        # Prioritized replay parameters
        self.alpha = alpha
        self.beta_start = beta_start
        self.beta_frames = beta_frames
        self.eps = eps
        self.max_priority = eps
        self.frame_count = 0

        # Initialize sum tree for prioritized sampling
        # Note: We need a tree for each parallel environment's buffer positions
        self.tree = SumTree(size=self.buffer_size)

        # Store last sampled indices and weights for priority updates
        self.last_tree_indices: Optional[np.ndarray] = None
        self.last_weights: Optional[th.Tensor] = None

    def add(
        self,
        obs: np.ndarray,
        next_obs: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        done: np.ndarray,
        infos: List[Dict[str, Any]],
    ) -> None:
        """
        Add new transition to buffer with maximum priority.
        
        New experiences are assigned max priority to ensure they're replayed at least once.
        """
        # Store transition using parent class
        super().add(obs, next_obs, action, reward, done, infos)

        # Assign maximum priority to new experience
        # Note: self.pos has already been incremented by parent's add()
        # so we use (self.pos - 1) % self.buffer_size
        idx = (self.pos - 1) % self.buffer_size
        self.tree.add(self.max_priority, idx)

    def sample(self, batch_size: int, env: Optional[VecNormalize] = None) -> ReplayBufferSamples:
        """
        Sample a batch of transitions using prioritized sampling.
        
        :param batch_size: Number of transitions to sample
        :param env: VecNormalize wrapper for normalization
        :return: ReplayBufferSamples with sampled transitions
        """
        # Determine valid range for sampling
        upper_bound = self.buffer_size if self.full else self.pos

        # Sample indices using prioritized sampling
        batch_inds = np.zeros(batch_size, dtype=np.int32)
        tree_indices = np.zeros(batch_size, dtype=np.int32)
        priorities = np.zeros(batch_size, dtype=np.float32)

        # Divide total priority into batch_size segments
        segment = self.tree.total / batch_size

        for i in range(batch_size):
            # Sample uniformly from each segment
            a, b = segment * i, segment * (i + 1)
            cumsum = np.random.uniform(a, b)

            # Get transition index and priority from tree
            tree_idx, priority, data_idx = self.tree.get(cumsum)
            
            # Ensure we don't sample invalid indices
            if data_idx >= upper_bound:
                # If we hit an invalid index, resample from valid range
                data_idx = np.random.randint(0, upper_bound)
                tree_idx = data_idx
                priority = self.tree.nodes[tree_idx + self.tree.size - 1]

            batch_inds[i] = data_idx
            tree_indices[i] = tree_idx
            priorities[i] = priority

        # Compute importance sampling weights
        # Anneal beta from beta_start to 1.0 over beta_frames
        self.frame_count += 1
        beta = min(1.0, self.beta_start + (1.0 - self.beta_start) * self.frame_count / self.beta_frames)

        # Compute sampling probabilities
        probs = priorities / self.tree.total

        # Importance sampling weights: (N * P(i))^(-beta)
        weights = (upper_bound * probs) ** (-beta)

        # Normalize weights so max weight = 1.0 (for stability)
        weights = weights / weights.max()

        # Store for later priority updates
        self.last_tree_indices = tree_indices
        self.last_weights = th.tensor(weights, dtype=th.float32, device=self.device)

        # Get actual samples using parent's method
        return self._get_samples(batch_inds, env=env)

    def update_priorities(self, indices: np.ndarray, priorities: np.ndarray) -> None:
        """
        Update priorities for sampled transitions.
        
        :param indices: Tree indices of transitions (from last_tree_indices)
        :param priorities: New priority values (typically TD errors)
        """
        if isinstance(priorities, th.Tensor):
            priorities = priorities.detach().cpu().numpy()

        for tree_idx, priority in zip(indices, priorities):
            # Compute priority: (|TD_error| + eps) ^ alpha
            priority = (np.abs(priority) + self.eps) ** self.alpha

            # Update max priority for new samples
            self.max_priority = max(self.max_priority, priority)

            # Update tree
            self.tree.update(int(tree_idx), priority)
