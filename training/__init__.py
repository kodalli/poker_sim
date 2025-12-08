"""Training utilities for reinforcement learning."""

from training.buffer import Experience, RolloutBuffer
from training.equity import estimate_equity, heuristic_equity
from training.ppo import PPOConfig, PPOTrainer
from training.rewards import RewardConfig, compute_reward
from training.trainer import RLTrainer, TrainingConfig, create_trainer
from training.logging import MetricsLogger

# JAX imports (optional - may not be available)
try:
    from training.jax_ppo import PPOConfig as JAXPPOConfig
    from training.jax_trainer import JAXTrainer, JAXTrainingConfig, create_jax_trainer
    _JAX_AVAILABLE = True
except ImportError:
    _JAX_AVAILABLE = False

__all__ = [
    # PyTorch training
    "Experience",
    "RolloutBuffer",
    "estimate_equity",
    "heuristic_equity",
    "PPOConfig",
    "PPOTrainer",
    "RewardConfig",
    "compute_reward",
    "RLTrainer",
    "TrainingConfig",
    "create_trainer",
    # Logging
    "MetricsLogger",
]

# Add JAX exports if available
if _JAX_AVAILABLE:
    __all__.extend([
        "JAXPPOConfig",
        "JAXTrainer",
        "JAXTrainingConfig",
        "create_jax_trainer",
    ])
