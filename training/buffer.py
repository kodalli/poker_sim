"""Experience buffer for PPO training."""

from dataclasses import dataclass, field
from typing import Generator

import numpy as np
import torch


@dataclass
class Experience:
    """A single experience tuple from the environment."""

    state: np.ndarray  # Encoded state (433-dim)
    action: int  # Action index (0-4)
    action_log_prob: float  # Log probability of the action taken
    bet_fraction: float  # Bet sizing fraction (0-1)
    reward: float  # Immediate reward
    value: float  # Value estimate at this state
    done: bool  # Whether this ends the episode


@dataclass
class Batch:
    """A batch of experiences for training."""

    states: torch.Tensor  # (batch, state_dim)
    actions: torch.Tensor  # (batch,) - action indices
    action_log_probs: torch.Tensor  # (batch,) - old log probs
    bet_fractions: torch.Tensor  # (batch,)
    returns: torch.Tensor  # (batch,) - discounted returns
    advantages: torch.Tensor  # (batch,) - GAE advantages
    values: torch.Tensor  # (batch,) - old value estimates


@dataclass
class RolloutBuffer:
    """Buffer for storing and processing rollout experiences.

    Supports GAE (Generalized Advantage Estimation) computation
    and batch sampling for PPO updates.
    """

    experiences: list[Experience] = field(default_factory=list)
    gamma: float = 0.99  # Discount factor
    lambda_: float = 0.95  # GAE lambda

    # Computed after rollout
    returns: np.ndarray | None = None
    advantages: np.ndarray | None = None

    def add(self, exp: Experience) -> None:
        """Add an experience to the buffer."""
        self.experiences.append(exp)

    def clear(self) -> None:
        """Clear all experiences."""
        self.experiences.clear()
        self.returns = None
        self.advantages = None

    def __len__(self) -> int:
        return len(self.experiences)

    def compute_returns_and_advantages(
        self,
        last_value: float = 0.0,
        normalize_advantages: bool = True,
    ) -> None:
        """Compute returns and advantages using GAE.

        Args:
            last_value: Value estimate for the state after last experience
                       (0 if episode ended, otherwise bootstrap)
            normalize_advantages: Whether to normalize advantages
        """
        n = len(self.experiences)
        if n == 0:
            return

        # Extract values and rewards
        rewards = np.array([e.reward for e in self.experiences])
        values = np.array([e.value for e in self.experiences])
        dones = np.array([e.done for e in self.experiences])

        # Initialize arrays
        self.returns = np.zeros(n, dtype=np.float32)
        self.advantages = np.zeros(n, dtype=np.float32)

        # GAE computation (backward pass)
        gae = 0.0
        next_value = last_value

        for t in reversed(range(n)):
            # If done, next_value should be 0
            if dones[t]:
                next_value = 0.0
                gae = 0.0

            # TD error: r + gamma * V(s') - V(s)
            delta = rewards[t] + self.gamma * next_value - values[t]

            # GAE: delta + gamma * lambda * GAE(t+1)
            gae = delta + self.gamma * self.lambda_ * gae

            self.advantages[t] = gae
            self.returns[t] = gae + values[t]

            next_value = values[t]

        # Normalize advantages
        if normalize_advantages and n > 1:
            adv_mean = self.advantages.mean()
            adv_std = self.advantages.std() + 1e-8
            self.advantages = (self.advantages - adv_mean) / adv_std

    def get_batches(
        self,
        batch_size: int,
        device: torch.device | None = None,
        shuffle: bool = True,
    ) -> Generator[Batch, None, None]:
        """Generate batches for training.

        Args:
            batch_size: Size of each batch
            device: Torch device for tensors
            shuffle: Whether to shuffle experiences

        Yields:
            Batch objects for training
        """
        if self.returns is None or self.advantages is None:
            raise ValueError("Must call compute_returns_and_advantages first")

        device = device or torch.device("cpu")
        n = len(self.experiences)

        # Create indices
        indices = np.arange(n)
        if shuffle:
            np.random.shuffle(indices)

        # Extract all data
        states = np.array([self.experiences[i].state for i in range(n)])
        actions = np.array([self.experiences[i].action for i in range(n)])
        log_probs = np.array([self.experiences[i].action_log_prob for i in range(n)])
        bet_fracs = np.array([self.experiences[i].bet_fraction for i in range(n)])
        values = np.array([self.experiences[i].value for i in range(n)])

        # Generate batches
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            batch_indices = indices[start:end]

            yield Batch(
                states=torch.tensor(states[batch_indices], dtype=torch.float32, device=device),
                actions=torch.tensor(actions[batch_indices], dtype=torch.long, device=device),
                action_log_probs=torch.tensor(log_probs[batch_indices], dtype=torch.float32, device=device),
                bet_fractions=torch.tensor(bet_fracs[batch_indices], dtype=torch.float32, device=device),
                returns=torch.tensor(self.returns[batch_indices], dtype=torch.float32, device=device),
                advantages=torch.tensor(self.advantages[batch_indices], dtype=torch.float32, device=device),
                values=torch.tensor(values[batch_indices], dtype=torch.float32, device=device),
            )

    def get_all(self, device: torch.device | None = None) -> Batch:
        """Get all experiences as a single batch."""
        batches = list(self.get_batches(len(self.experiences), device, shuffle=False))
        if not batches:
            raise ValueError("Buffer is empty")
        return batches[0]

    def get_stats(self) -> dict[str, float]:
        """Get buffer statistics for logging."""
        if not self.experiences:
            return {}

        rewards = [e.reward for e in self.experiences]
        values = [e.value for e in self.experiences]

        stats = {
            "buffer_size": len(self.experiences),
            "reward_mean": np.mean(rewards),
            "reward_std": np.std(rewards),
            "reward_min": np.min(rewards),
            "reward_max": np.max(rewards),
            "value_mean": np.mean(values),
            "value_std": np.std(values),
        }

        if self.returns is not None:
            stats["return_mean"] = float(self.returns.mean())
            stats["return_std"] = float(self.returns.std())

        if self.advantages is not None:
            stats["advantage_mean"] = float(self.advantages.mean())
            stats["advantage_std"] = float(self.advantages.std())

        return stats
