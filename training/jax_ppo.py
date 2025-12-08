"""PPO implementation in JAX for GPU-accelerated training."""

from dataclasses import dataclass
from functools import partial
from typing import NamedTuple

import jax
import jax.numpy as jnp
import optax
from jax import Array
from flax import linen as nn


@dataclass(frozen=True)
class PPOConfig:
    """Configuration for PPO training."""

    # Learning rate
    learning_rate: float = 3e-4
    max_grad_norm: float = 0.5

    # PPO hyperparameters
    gamma: float = 0.99  # Discount factor
    lambda_: float = 0.95  # GAE parameter
    epsilon: float = 0.2  # Clipping parameter
    ppo_epochs: int = 4  # Update epochs per batch
    num_minibatches: int = 4  # Number of minibatches

    # Loss weights
    value_coef: float = 0.5
    entropy_coef: float = 0.01

    # Normalization
    normalize_advantages: bool = True


class Trajectory(NamedTuple):
    """Batch of trajectory data from rollout."""

    obs: Array  # [T, N, obs_dim]
    actions: Array  # [T, N]
    log_probs: Array  # [T, N]
    values: Array  # [T, N]
    rewards: Array  # [T, N]
    dones: Array  # [T, N]
    valid_masks: Array  # [T, N, num_actions]


class PPOMetrics(NamedTuple):
    """Metrics from PPO update."""

    policy_loss: float
    value_loss: float
    entropy: float
    total_loss: float
    approx_kl: float
    clip_fraction: float
    # RL diagnostics
    explained_variance: float
    grad_norm: float
    value_pred_error: float


def create_optimizer(config: PPOConfig) -> optax.GradientTransformation:
    """Create optimizer with gradient clipping."""
    return optax.chain(
        optax.clip_by_global_norm(config.max_grad_norm),
        optax.adam(config.learning_rate),
    )


@jax.jit
def compute_gae(
    rewards: Array,
    values: Array,
    dones: Array,
    gamma: float = 0.99,
    lambda_: float = 0.95,
) -> tuple[Array, Array]:
    """Compute Generalized Advantage Estimation.

    Args:
        rewards: [T, N] rewards at each step
        values: [T, N] value estimates
        dones: [T, N] done flags
        gamma: Discount factor
        lambda_: GAE parameter

    Returns:
        Tuple of (advantages [T, N], returns [T, N])
    """
    T, N = rewards.shape

    # Bootstrap value (assume 0 for terminal states)
    # For continuing episodes, we'd need the next value
    next_value = jnp.zeros(N)

    def scan_fn(carry, t):
        gae, next_val = carry
        done = dones[t]
        reward = rewards[t]
        value = values[t]

        # TD error
        delta = reward + gamma * next_val * (1 - done) - value

        # GAE
        gae = delta + gamma * lambda_ * (1 - done) * gae

        return (gae, value), gae

    # Scan backwards through time
    _, advantages = jax.lax.scan(
        scan_fn,
        (jnp.zeros(N), next_value),
        jnp.arange(T - 1, -1, -1),
    )

    # Reverse to get correct time order
    advantages = advantages[::-1]

    # Returns = advantages + values
    returns = advantages + values

    return advantages, returns


@partial(jax.jit, static_argnums=(0, 8))
def ppo_loss(
    network: nn.Module,
    params: dict,
    obs: Array,
    actions: Array,
    old_log_probs: Array,
    advantages: Array,
    returns: Array,
    valid_masks: Array,
    config: PPOConfig,
    rng_key: Array,
) -> tuple[Array, dict]:
    """Compute PPO loss.

    Args:
        network: Flax network module
        params: Network parameters
        obs: [batch, obs_dim] observations
        actions: [batch] action indices
        old_log_probs: [batch] log probabilities from rollout
        advantages: [batch] advantage estimates
        returns: [batch] return estimates
        valid_masks: [batch, num_actions] valid action masks
        config: PPO config
        rng_key: PRNG key for dropout

    Returns:
        Tuple of (total_loss, metrics_dict)
    """
    # Forward pass
    action_logits, _, values = network.apply(
        {"params": params},
        obs,
        training=True,
        rngs={"dropout": rng_key},
    )
    values = values.squeeze(-1)

    # Mask invalid actions (use -1e8 for better numerical stability)
    masked_logits = jnp.where(valid_masks, action_logits, -1e8)

    # Compute new log probs
    log_probs = jax.nn.log_softmax(masked_logits, axis=-1)
    new_log_probs = jnp.take_along_axis(log_probs, actions[:, None], axis=1).squeeze(-1)

    # Compute entropy with numerical stability
    # Note: No need to multiply by valid_masks again - softmax already zeroes invalid actions
    probs = jax.nn.softmax(masked_logits, axis=-1)
    # Add epsilon to avoid log(0) for invalid actions (their probs are ~0 but not exactly 0)
    safe_log_probs = jnp.log(probs + 1e-10)
    entropy = -jnp.sum(probs * safe_log_probs, axis=-1).mean()

    # PPO policy loss
    ratio = jnp.exp(new_log_probs - old_log_probs)
    surr1 = ratio * advantages
    surr2 = jnp.clip(ratio, 1 - config.epsilon, 1 + config.epsilon) * advantages
    policy_loss = -jnp.minimum(surr1, surr2).mean()

    # Value loss
    value_loss = ((values - returns) ** 2).mean()

    # Entropy bonus (negative because we want to maximize entropy)
    entropy_loss = -entropy

    # Total loss
    total_loss = (
        policy_loss
        + config.value_coef * value_loss
        + config.entropy_coef * entropy_loss
    )

    # Metrics
    approx_kl = ((ratio - 1) - jnp.log(ratio)).mean()
    clip_fraction = (jnp.abs(ratio - 1) > config.epsilon).mean()

    # RL diagnostics
    # Explained variance: how well value function predicts returns (1.0 = perfect)
    returns_var = jnp.var(returns)
    explained_variance = jnp.where(
        returns_var > 1e-8,
        1.0 - jnp.var(returns - values) / returns_var,
        0.0,
    )
    # Value prediction error: mean absolute error
    value_pred_error = jnp.abs(returns - values).mean()

    metrics = {
        "policy_loss": policy_loss,
        "value_loss": value_loss,
        "entropy": entropy,
        "total_loss": total_loss,
        "approx_kl": approx_kl,
        "clip_fraction": clip_fraction,
        "explained_variance": explained_variance,
        "value_pred_error": value_pred_error,
    }

    return total_loss, metrics


@partial(jax.jit, static_argnums=(0, 3, 10))
def ppo_update_step(
    network: nn.Module,
    params: dict,
    opt_state: optax.OptState,
    optimizer: optax.GradientTransformation,
    obs: Array,
    actions: Array,
    old_log_probs: Array,
    advantages: Array,
    returns: Array,
    valid_masks: Array,
    config: PPOConfig,
    rng_key: Array,
) -> tuple[dict, optax.OptState, dict]:
    """Single PPO update step.

    Returns:
        Tuple of (new_params, new_opt_state, metrics)
    """
    # Compute loss and gradients
    grad_fn = jax.value_and_grad(ppo_loss, argnums=1, has_aux=True)
    (loss, metrics), grads = grad_fn(
        network, params, obs, actions, old_log_probs,
        advantages, returns, valid_masks, config, rng_key
    )

    # Compute gradient norm before clipping
    grad_norm = optax.global_norm(grads)
    metrics["grad_norm"] = grad_norm

    # Update parameters
    updates, new_opt_state = optimizer.update(grads, opt_state, params)
    new_params = optax.apply_updates(params, updates)

    return new_params, new_opt_state, metrics


def ppo_update(
    network: nn.Module,
    params: dict,
    opt_state: optax.OptState,
    optimizer: optax.GradientTransformation,
    trajectory: Trajectory,
    config: PPOConfig,
    rng_key: Array,
) -> tuple[dict, optax.OptState, PPOMetrics]:
    """Full PPO update on trajectory data.

    Args:
        network: Flax network
        params: Network parameters
        opt_state: Optimizer state
        optimizer: Optax optimizer
        trajectory: Trajectory data from rollout
        config: PPO config
        rng_key: PRNG key

    Returns:
        Tuple of (new_params, new_opt_state, metrics)
    """
    T, N = trajectory.rewards.shape

    # Compute GAE
    advantages, returns = compute_gae(
        trajectory.rewards,
        trajectory.values,
        trajectory.dones,
        config.gamma,
        config.lambda_,
    )

    # Normalize advantages
    if config.normalize_advantages:
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    # Flatten trajectory
    obs_flat = trajectory.obs.reshape(-1, trajectory.obs.shape[-1])
    actions_flat = trajectory.actions.reshape(-1)
    log_probs_flat = trajectory.log_probs.reshape(-1)
    advantages_flat = advantages.reshape(-1)
    returns_flat = returns.reshape(-1)
    valid_masks_flat = trajectory.valid_masks.reshape(-1, trajectory.valid_masks.shape[-1])

    # Track metrics
    total_metrics = {
        "policy_loss": 0.0,
        "value_loss": 0.0,
        "entropy": 0.0,
        "total_loss": 0.0,
        "approx_kl": 0.0,
        "clip_fraction": 0.0,
        # RL diagnostics
        "explained_variance": 0.0,
        "grad_norm": 0.0,
        "value_pred_error": 0.0,
    }
    num_updates = 0

    batch_size = T * N
    minibatch_size = batch_size // config.num_minibatches

    # Multiple epochs
    for epoch in range(config.ppo_epochs):
        # Shuffle indices
        rng_key, shuffle_key = jax.random.split(rng_key)
        indices = jax.random.permutation(shuffle_key, batch_size)

        # Process minibatches
        for start in range(0, batch_size, minibatch_size):
            end = start + minibatch_size
            mb_indices = indices[start:end]

            # Get minibatch data
            mb_obs = obs_flat[mb_indices]
            mb_actions = actions_flat[mb_indices]
            mb_log_probs = log_probs_flat[mb_indices]
            mb_advantages = advantages_flat[mb_indices]
            mb_returns = returns_flat[mb_indices]
            mb_valid_masks = valid_masks_flat[mb_indices]

            # Update
            rng_key, update_key = jax.random.split(rng_key)
            params, opt_state, metrics = ppo_update_step(
                network, params, opt_state, optimizer,
                mb_obs, mb_actions, mb_log_probs,
                mb_advantages, mb_returns, mb_valid_masks,
                config, update_key
            )

            # Accumulate metrics
            for k, v in metrics.items():
                total_metrics[k] += float(v)
            num_updates += 1

    # Average metrics
    for k in total_metrics:
        total_metrics[k] /= max(num_updates, 1)

    return params, opt_state, PPOMetrics(**total_metrics)
