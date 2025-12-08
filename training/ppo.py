"""Proximal Policy Optimization (PPO) implementation."""

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam

from agents.neural.network import ActorCriticMLP
from training.buffer import Batch, RolloutBuffer


@dataclass
class PPOConfig:
    """Configuration for PPO training."""

    # Optimizer
    learning_rate: float = 3e-4
    max_grad_norm: float = 0.5  # Gradient clipping

    # PPO hyperparameters
    gamma: float = 0.99  # Discount factor
    lambda_: float = 0.95  # GAE parameter
    epsilon: float = 0.2  # Clipping parameter
    ppo_epochs: int = 4  # Update epochs per batch

    # Loss weights
    value_coef: float = 0.5  # Value loss weight
    entropy_coef: float = 0.01  # Entropy bonus weight

    # Batch settings
    batch_size: int = 64
    normalize_advantages: bool = True


@dataclass
class PPOMetrics:
    """Metrics from a PPO update."""

    policy_loss: float
    value_loss: float
    entropy: float
    total_loss: float
    approx_kl: float  # Approximate KL divergence
    clip_fraction: float  # Fraction of clipped ratios


class PPOTrainer:
    """PPO trainer for ActorCriticMLP networks."""

    def __init__(
        self,
        network: ActorCriticMLP,
        config: PPOConfig | None = None,
        device: torch.device | None = None,
    ) -> None:
        self.network = network
        self.config = config or PPOConfig()
        self.device = device or torch.device("cpu")

        self.optimizer = Adam(
            network.parameters(),
            lr=self.config.learning_rate,
        )

        # Move network to device
        self.network.to(self.device)

    def update(self, buffer: RolloutBuffer) -> PPOMetrics:
        """Run PPO update on collected experiences.

        Args:
            buffer: RolloutBuffer with computed advantages

        Returns:
            PPOMetrics with training statistics
        """
        # Compute advantages if not already done
        if buffer.advantages is None:
            buffer.compute_returns_and_advantages(
                normalize_advantages=self.config.normalize_advantages
            )

        # Track metrics across epochs
        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0
        total_approx_kl = 0.0
        total_clip_frac = 0.0
        num_batches = 0

        # Run multiple epochs
        for epoch in range(self.config.ppo_epochs):
            for batch in buffer.get_batches(
                self.config.batch_size,
                device=self.device,
                shuffle=True,
            ):
                metrics = self._update_batch(batch)

                total_policy_loss += metrics["policy_loss"]
                total_value_loss += metrics["value_loss"]
                total_entropy += metrics["entropy"]
                total_approx_kl += metrics["approx_kl"]
                total_clip_frac += metrics["clip_fraction"]
                num_batches += 1

        # Average metrics
        n = max(num_batches, 1)
        return PPOMetrics(
            policy_loss=total_policy_loss / n,
            value_loss=total_value_loss / n,
            entropy=total_entropy / n,
            total_loss=(total_policy_loss + total_value_loss) / n,
            approx_kl=total_approx_kl / n,
            clip_fraction=total_clip_frac / n,
        )

    def _update_batch(self, batch: Batch) -> dict[str, float]:
        """Update network on a single batch."""
        self.network.train()

        # Forward pass
        action_logits, bet_fraction, values = self.network(batch.states)
        values = values.squeeze(-1)

        # Create action distribution
        action_probs = F.softmax(action_logits, dim=-1)
        dist = torch.distributions.Categorical(action_probs)

        # Get log probs and entropy
        new_log_probs = dist.log_prob(batch.actions)
        entropy = dist.entropy().mean()

        # Compute ratio for PPO
        log_ratio = new_log_probs - batch.action_log_probs
        ratio = torch.exp(log_ratio)

        # Approximate KL divergence
        approx_kl = ((ratio - 1) - log_ratio).mean().item()

        # Clipped surrogate objective
        advantages = batch.advantages
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - self.config.epsilon, 1 + self.config.epsilon) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()

        # Value loss (clipped)
        value_pred = values
        value_target = batch.returns

        # Option 1: Simple MSE
        value_loss = F.mse_loss(value_pred, value_target)

        # Option 2: Clipped value loss (can help stability)
        # value_pred_clipped = batch.values + torch.clamp(
        #     value_pred - batch.values, -self.config.epsilon, self.config.epsilon
        # )
        # value_loss1 = F.mse_loss(value_pred, value_target)
        # value_loss2 = F.mse_loss(value_pred_clipped, value_target)
        # value_loss = torch.max(value_loss1, value_loss2)

        # Entropy bonus (encourages exploration)
        entropy_loss = -entropy

        # Total loss
        total_loss = (
            policy_loss
            + self.config.value_coef * value_loss
            + self.config.entropy_coef * entropy_loss
        )

        # Backward pass
        self.optimizer.zero_grad()
        total_loss.backward()

        # Gradient clipping
        if self.config.max_grad_norm > 0:
            nn.utils.clip_grad_norm_(
                self.network.parameters(),
                self.config.max_grad_norm,
            )

        self.optimizer.step()

        # Compute clip fraction
        with torch.no_grad():
            clip_fraction = (
                (torch.abs(ratio - 1) > self.config.epsilon).float().mean().item()
            )

        return {
            "policy_loss": policy_loss.item(),
            "value_loss": value_loss.item(),
            "entropy": entropy.item(),
            "approx_kl": approx_kl,
            "clip_fraction": clip_fraction,
        }

    def save(self, path: str) -> None:
        """Save trainer state."""
        torch.save(
            {
                "network_state": self.network.state_dict(),
                "optimizer_state": self.optimizer.state_dict(),
                "config": self.config,
            },
            path,
        )

    def load(self, path: str) -> None:
        """Load trainer state."""
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.network.load_state_dict(checkpoint["network_state"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state"])
        if "config" in checkpoint:
            self.config = checkpoint["config"]
