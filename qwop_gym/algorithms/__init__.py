"""Custom RL algorithms for QWOP training."""

from qwop_gym.algorithms.enhanced_qrdqn import EnhancedQRDQN
from qwop_gym.algorithms.discrete_sac import DiscreteSAC, DiscreteSACPolicy

__all__ = ["EnhancedQRDQN", "DiscreteSAC", "DiscreteSACPolicy"]
