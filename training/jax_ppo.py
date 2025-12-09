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
    num_minibatches: int = 8  # Number of minibatches (optimized for large batches)

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
    model_masks: Array | None = None  # [T, N] - 1.0 when model acted, 0.0 on opponent turns (optional for backward compat)


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
    model_masks: Array | None = None,
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
        model_masks: [batch] mask (1.0 = model's turn, 0.0 = opponent's turn). If None, all steps are used.

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
    per_sample_entropy = -jnp.sum(probs * safe_log_probs, axis=-1)

    # PPO policy loss (per sample)
    ratio = jnp.exp(new_log_probs - old_log_probs)
    surr1 = ratio * advantages
    surr2 = jnp.clip(ratio, 1 - config.epsilon, 1 + config.epsilon) * advantages
    per_sample_policy_loss = -jnp.minimum(surr1, surr2)

    # Value loss (per sample)
    per_sample_value_loss = (values - returns) ** 2

    # Apply model_masks if provided (only backprop on model's turns)
    if model_masks is not None:
        # Masked mean for policy loss and entropy (only train on model's decisions)
        mask_sum = jnp.maximum(model_masks.sum(), 1.0)  # Avoid div by 0
        policy_loss = (per_sample_policy_loss * model_masks).sum() / mask_sum
        entropy = (per_sample_entropy * model_masks).sum() / mask_sum
        # Value loss can use all samples (value function learns from all observations)
        value_loss = per_sample_value_loss.mean()
    else:
        # No masking - use all samples (backward compat for self-play)
        policy_loss = per_sample_policy_loss.mean()
        entropy = per_sample_entropy.mean()
        value_loss = per_sample_value_loss.mean()

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
    model_masks: Array | None = None,
) -> tuple[dict, optax.OptState, dict]:
    """Single PPO update step.

    Returns:
        Tuple of (new_params, new_opt_state, metrics)
    """
    # Compute loss and gradients
    grad_fn = jax.value_and_grad(ppo_loss, argnums=1, has_aux=True)
    (loss, metrics), grads = grad_fn(
        network, params, obs, actions, old_log_probs,
        advantages, returns, valid_masks, config, rng_key, model_masks
    )

    # Compute gradient norm before clipping
    grad_norm = optax.global_norm(grads)
    metrics["grad_norm"] = grad_norm

    # Update parameters
    updates, new_opt_state = optimizer.update(grads, opt_state, params)
    new_params = optax.apply_updates(params, updates)

    return new_params, new_opt_state, metrics


def _create_ppo_epoch_fn(network, optimizer, config, use_model_masks=False):
    """Create JIT-compiled epoch function for PPO updates.

    This is separated to allow JIT compilation with static arguments.
    """
    num_minibatches = config.num_minibatches

    @jax.jit
    def single_epoch(carry, _):
        """Process one epoch of minibatch updates."""
        params, opt_state, metrics_sum, rng_key, data = carry
        if use_model_masks:
            obs_flat, actions_flat, log_probs_flat, advantages_flat, returns_flat, valid_masks_flat, model_masks_flat = data
        else:
            obs_flat, actions_flat, log_probs_flat, advantages_flat, returns_flat, valid_masks_flat = data
            model_masks_flat = None

        batch_size = obs_flat.shape[0]
        minibatch_size = batch_size // num_minibatches

        # Shuffle indices for this epoch
        rng_key, shuffle_key = jax.random.split(rng_key)
        indices = jax.random.permutation(shuffle_key, batch_size)

        def single_minibatch(mb_carry, mb_idx):
            """Process one minibatch update."""
            params, opt_state, metrics_sum, rng_key = mb_carry

            # Get minibatch indices
            start = mb_idx * minibatch_size
            mb_indices = jax.lax.dynamic_slice(indices, [start], [minibatch_size])

            # Get minibatch data
            mb_obs = obs_flat[mb_indices]
            mb_actions = actions_flat[mb_indices]
            mb_log_probs = log_probs_flat[mb_indices]
            mb_advantages = advantages_flat[mb_indices]
            mb_returns = returns_flat[mb_indices]
            mb_valid_masks = valid_masks_flat[mb_indices]
            mb_model_masks = model_masks_flat[mb_indices] if model_masks_flat is not None else None

            # Update
            rng_key, update_key = jax.random.split(rng_key)
            params, opt_state, metrics = ppo_update_step(
                network, params, opt_state, optimizer,
                mb_obs, mb_actions, mb_log_probs,
                mb_advantages, mb_returns, mb_valid_masks,
                config, update_key, mb_model_masks
            )

            # Accumulate metrics (keep as JAX arrays, no float() conversion!)
            new_metrics_sum = {k: metrics_sum[k] + metrics[k] for k in metrics_sum.keys()}

            return (params, opt_state, new_metrics_sum, rng_key), None

        # Process all minibatches in this epoch
        (params, opt_state, metrics_sum, rng_key), _ = jax.lax.scan(
            single_minibatch,
            (params, opt_state, metrics_sum, rng_key),
            jnp.arange(num_minibatches)
        )

        return (params, opt_state, metrics_sum, rng_key, data), None

    return single_epoch


# Cache for compiled epoch functions
_epoch_fn_cache = {}


def ppo_update(
    network: nn.Module,
    params: dict,
    opt_state: optax.OptState,
    optimizer: optax.GradientTransformation,
    trajectory: Trajectory,
    config: PPOConfig,
    rng_key: Array,
) -> tuple[dict, optax.OptState, PPOMetrics]:
    """Full PPO update on trajectory data (JIT-optimized).

    This version uses jax.lax.scan instead of Python for-loops,
    keeping all computation on GPU without device->host transfers.

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

    # Handle model_masks (optional, for mixed opponent training)
    use_model_masks = trajectory.model_masks is not None
    if use_model_masks:
        model_masks_flat = trajectory.model_masks.reshape(-1)
        data = (obs_flat, actions_flat, log_probs_flat, advantages_flat, returns_flat, valid_masks_flat, model_masks_flat)
    else:
        data = (obs_flat, actions_flat, log_probs_flat, advantages_flat, returns_flat, valid_masks_flat)

    # Initialize metrics accumulator (JAX arrays)
    init_metrics = {
        "policy_loss": jnp.array(0.0),
        "value_loss": jnp.array(0.0),
        "entropy": jnp.array(0.0),
        "total_loss": jnp.array(0.0),
        "approx_kl": jnp.array(0.0),
        "clip_fraction": jnp.array(0.0),
        "explained_variance": jnp.array(0.0),
        "grad_norm": jnp.array(0.0),
        "value_pred_error": jnp.array(0.0),
    }

    # Get or create cached epoch function (separate cache for with/without model_masks)
    cache_key = (id(network), id(optimizer), config.num_minibatches, use_model_masks)
    if cache_key not in _epoch_fn_cache:
        _epoch_fn_cache[cache_key] = _create_ppo_epoch_fn(network, optimizer, config, use_model_masks)
    epoch_fn = _epoch_fn_cache[cache_key]

    # Run all epochs using scan (fully on GPU)
    (params, opt_state, metrics_sum, _, _), _ = jax.lax.scan(
        epoch_fn,
        (params, opt_state, init_metrics, rng_key, data),
        jnp.arange(config.ppo_epochs)
    )

    # Average metrics and convert to Python floats (only once at the end!)
    num_updates = config.ppo_epochs * config.num_minibatches
    final_metrics = {k: float(v / num_updates) for k, v in metrics_sum.items()}

    return params, opt_state, PPOMetrics(**final_metrics)
