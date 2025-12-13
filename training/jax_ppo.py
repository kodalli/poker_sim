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

    # === v9 opponent model fields (optional) ===
    opp_actions: Array | None = None  # [T, N, opp_action_dim] encoded opponent actions
    lstm_hidden: Array | None = None  # [T, N, lstm_hidden_dim] LSTM hidden states
    lstm_cell: Array | None = None  # [T, N, lstm_hidden_dim] LSTM cell states


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
        # Masked mean for policy loss, entropy, AND value loss (only train on model's decisions)
        # Training value function on opponent turns corrupts it - it learns to predict
        # opponent's returns rather than its own future returns
        mask_sum = jnp.maximum(model_masks.sum(), 1.0)  # Avoid div by 0
        policy_loss = (per_sample_policy_loss * model_masks).sum() / mask_sum
        entropy = (per_sample_entropy * model_masks).sum() / mask_sum
        value_loss = (per_sample_value_loss * model_masks).sum() / mask_sum
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


@partial(jax.jit, static_argnums=(0, 10))
def ppo_loss_sequential(
    network: nn.Module,
    params: dict,
    obs_seq: Array,
    opp_actions_seq: Array,
    initial_carry: tuple[Array, Array],
    actions: Array,
    old_log_probs: Array,
    advantages: Array,
    returns: Array,
    valid_masks: Array,
    config: PPOConfig,
    rng_key: Array,
    model_masks: Array | None = None,
) -> tuple[Array, dict]:
    """Compute PPO loss for sequential data with LSTM unrolling.

    Args:
        network: Flax network module with opponent model
        params: Network parameters
        obs_seq: [seq_len, batch, obs_dim] observation sequence
        opp_actions_seq: [seq_len, batch, opp_action_dim] opponent action sequence
        initial_carry: (hidden, cell) LSTM carry at start of sequence
        actions: [seq_len, batch] action indices
        old_log_probs: [seq_len, batch] log probabilities from rollout
        advantages: [seq_len, batch] advantage estimates
        returns: [seq_len, batch] return estimates
        valid_masks: [seq_len, batch, num_actions] valid action masks
        config: PPO config
        rng_key: PRNG key for dropout
        model_masks: [seq_len, batch] mask (1.0 = model's turn). If None, all steps used.

    Returns:
        Tuple of (total_loss, metrics_dict)
    """
    seq_len = obs_seq.shape[0]

    def step_fn(carry, inputs):
        """Single step of LSTM unrolling."""
        lstm_carry, rng = carry
        obs_t, opp_action_t = inputs

        # Split RNG for dropout
        rng, step_rng = jax.random.split(rng)

        # Forward pass through network
        action_logits, _, values, new_carry = network.apply(
            {"params": params},
            obs_t,
            opp_action_t,
            lstm_carry,
            training=True,
            rngs={"dropout": step_rng},
        )

        return (new_carry, rng), (action_logits, values)

    # Unroll LSTM through sequence using scan
    rng_key, unroll_key = jax.random.split(rng_key)
    _, (action_logits_seq, values_seq) = jax.lax.scan(
        step_fn,
        (initial_carry, unroll_key),
        (obs_seq, opp_actions_seq)
    )

    # action_logits_seq: [seq_len, batch, num_actions]
    # values_seq: [seq_len, batch, 1]
    values_seq = values_seq.squeeze(-1)  # [seq_len, batch]

    # Flatten to [seq_len * batch]
    batch_size = obs_seq.shape[1]
    action_logits_flat = action_logits_seq.reshape(-1, action_logits_seq.shape[-1])
    values_flat = values_seq.reshape(-1)
    actions_flat = actions.reshape(-1)
    old_log_probs_flat = old_log_probs.reshape(-1)
    advantages_flat = advantages.reshape(-1)
    returns_flat = returns.reshape(-1)
    valid_masks_flat = valid_masks.reshape(-1, valid_masks.shape[-1])

    # Mask invalid actions
    masked_logits = jnp.where(valid_masks_flat, action_logits_flat, -1e8)

    # Compute new log probs
    log_probs = jax.nn.log_softmax(masked_logits, axis=-1)
    new_log_probs = jnp.take_along_axis(log_probs, actions_flat[:, None], axis=1).squeeze(-1)

    # Compute entropy
    probs = jax.nn.softmax(masked_logits, axis=-1)
    safe_log_probs = jnp.log(probs + 1e-10)
    per_sample_entropy = -jnp.sum(probs * safe_log_probs, axis=-1)

    # PPO policy loss (per sample)
    ratio = jnp.exp(new_log_probs - old_log_probs_flat)
    surr1 = ratio * advantages_flat
    surr2 = jnp.clip(ratio, 1 - config.epsilon, 1 + config.epsilon) * advantages_flat
    per_sample_policy_loss = -jnp.minimum(surr1, surr2)

    # Value loss (per sample)
    per_sample_value_loss = (values_flat - returns_flat) ** 2

    # Apply model_masks if provided
    if model_masks is not None:
        model_masks_flat = model_masks.reshape(-1)
        mask_sum = jnp.maximum(model_masks_flat.sum(), 1.0)
        policy_loss = (per_sample_policy_loss * model_masks_flat).sum() / mask_sum
        entropy = (per_sample_entropy * model_masks_flat).sum() / mask_sum
        value_loss = (per_sample_value_loss * model_masks_flat).sum() / mask_sum
    else:
        policy_loss = per_sample_policy_loss.mean()
        entropy = per_sample_entropy.mean()
        value_loss = per_sample_value_loss.mean()

    # Entropy bonus
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
    returns_var = jnp.var(returns_flat)
    explained_variance = jnp.where(
        returns_var > 1e-8,
        1.0 - jnp.var(returns_flat - values_flat) / returns_var,
        0.0,
    )
    value_pred_error = jnp.abs(returns_flat - values_flat).mean()

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


def _create_ppo_sequential_epoch_fn(network, optimizer, config, window_size, use_model_masks=False):
    """Create JIT-compiled epoch function for sequential PPO updates with LSTM.

    This is separated to allow JIT compilation with static arguments.
    The data arrays are passed through the carry to allow caching.
    """
    num_minibatches = config.num_minibatches

    @jax.jit
    def single_epoch(carry, _):
        """Process one epoch of minibatch updates."""
        params, opt_state, metrics_sum, rng_key, data = carry

        # Unpack data tuple
        (obs_batch, opp_actions_batch, initial_carry_h, initial_carry_c,
         actions_batch, log_probs_batch, advantages_batch, returns_batch,
         valid_masks_batch, model_masks_batch) = data

        initial_carry = (initial_carry_h, initial_carry_c)
        batch_size = obs_batch.shape[1]
        minibatch_size = batch_size // num_minibatches

        # Shuffle indices for this epoch
        rng_key, shuffle_key = jax.random.split(rng_key)
        indices = jax.random.permutation(shuffle_key, batch_size)

        def single_minibatch(mb_carry, mb_idx):
            """Process one minibatch update."""
            params, opt_state, metrics_sum, rng_key = mb_carry

            # Get minibatch indices using dynamic_slice
            start = mb_idx * minibatch_size
            mb_indices = jax.lax.dynamic_slice(indices, [start], [minibatch_size])

            # Extract minibatch data
            mb_obs = obs_batch[:, mb_indices, :]
            mb_opp_actions = opp_actions_batch[:, mb_indices, :]
            mb_initial_carry = (
                initial_carry[0][mb_indices, :],
                initial_carry[1][mb_indices, :]
            )
            mb_actions = actions_batch[:, mb_indices]
            mb_log_probs = log_probs_batch[:, mb_indices]
            mb_advantages = advantages_batch[:, mb_indices]
            mb_returns = returns_batch[:, mb_indices]
            mb_valid_masks = valid_masks_batch[:, mb_indices, :]
            mb_model_masks = model_masks_batch[:, mb_indices] if use_model_masks else None

            # Compute loss and gradients
            rng_key, loss_key = jax.random.split(rng_key)
            grad_fn = jax.value_and_grad(ppo_loss_sequential, argnums=1, has_aux=True)
            (loss, metrics), grads = grad_fn(
                network, params, mb_obs, mb_opp_actions, mb_initial_carry,
                mb_actions, mb_log_probs, mb_advantages, mb_returns,
                mb_valid_masks, config, loss_key, mb_model_masks
            )

            # Compute gradient norm
            grad_norm = optax.global_norm(grads)
            metrics["grad_norm"] = grad_norm

            # Update parameters
            updates, new_opt_state = optimizer.update(grads, opt_state, params)
            new_params = optax.apply_updates(params, updates)

            # Accumulate metrics (keep as JAX arrays, no float() conversion!)
            new_metrics_sum = {k: metrics_sum[k] + metrics[k] for k in metrics_sum.keys()}

            return (new_params, new_opt_state, new_metrics_sum, rng_key), None

        # Process all minibatches in this epoch
        (params, opt_state, metrics_sum, rng_key), _ = jax.lax.scan(
            single_minibatch,
            (params, opt_state, metrics_sum, rng_key),
            jnp.arange(num_minibatches)
        )

        return (params, opt_state, metrics_sum, rng_key, data), None

    return single_epoch


# Separate cache for sequential epoch functions
_sequential_epoch_fn_cache = {}


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




def ppo_update_sequential(
    network: nn.Module,
    params: dict,
    opt_state: optax.OptState,
    optimizer: optax.GradientTransformation,
    trajectory: Trajectory,
    config: PPOConfig,
    rng_key: Array,
    window_size: int = 64,
) -> tuple[dict, optax.OptState, PPOMetrics]:
    """PPO update for sequential data with truncated BPTT.

    Splits trajectory into windows, shuffles windows (not individual steps),
    and unrolls LSTM through each window for gradient computation.

    LSTM states are recomputed from the opp_actions sequence rather than stored,
    saving ~2.4GB GPU memory per rollout.

    Args:
        network: Flax network with opponent model
        params: Network parameters
        opt_state: Optimizer state
        optimizer: Optax optimizer
        trajectory: Trajectory with v9 fields (opp_actions). LSTM states are recomputed.
        config: PPO config
        rng_key: PRNG key
        window_size: Truncated BPTT window size (default 64)

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

    # Truncate to fit window_size
    n_windows = T // window_size

    # Handle case where trajectory is shorter than window_size
    if n_windows == 0:
        # Trajectory too short for BPTT - just use window_size=T
        window_size = T
        n_windows = 1

    truncated_T = n_windows * window_size

    # Truncate all trajectory arrays
    obs_trunc = trajectory.obs[:truncated_T]
    opp_actions_trunc = trajectory.opp_actions[:truncated_T]
    actions_trunc = trajectory.actions[:truncated_T]
    log_probs_trunc = trajectory.log_probs[:truncated_T]
    advantages_trunc = advantages[:truncated_T]
    returns_trunc = returns[:truncated_T]
    valid_masks_trunc = trajectory.valid_masks[:truncated_T]
    model_masks_trunc = trajectory.model_masks[:truncated_T] if trajectory.model_masks is not None else None

    # Reshape into windows: [n_windows, window_size, N, ...]
    def reshape_to_windows(arr):
        # arr: [truncated_T, N, ...] or [truncated_T, N]
        if arr.ndim == 2:
            # [T, N] -> [n_windows, window_size, N]
            return arr.reshape(n_windows, window_size, N)
        else:
            # [T, N, ...] -> [n_windows, window_size, N, ...]
            rest_dims = arr.shape[2:]
            return arr.reshape(n_windows, window_size, N, *rest_dims)

    obs_windows = reshape_to_windows(obs_trunc)
    opp_actions_windows = reshape_to_windows(opp_actions_trunc)
    actions_windows = reshape_to_windows(actions_trunc)
    log_probs_windows = reshape_to_windows(log_probs_trunc)
    advantages_windows = reshape_to_windows(advantages_trunc)
    returns_windows = reshape_to_windows(returns_trunc)
    valid_masks_windows = reshape_to_windows(valid_masks_trunc)
    model_masks_windows = reshape_to_windows(model_masks_trunc) if model_masks_trunc is not None else None

    # Extract LSTM states at window boundaries from stored trajectory
    # trajectory.lstm_hidden: [T, N, lstm_hidden_dim]
    # We need the state at the START of each window (step 0, window_size, 2*window_size, ...)
    lstm_hidden_trunc = trajectory.lstm_hidden[:truncated_T]
    lstm_cell_trunc = trajectory.lstm_cell[:truncated_T]

    # Reshape to windows: [n_windows, window_size, N, lstm_hidden_dim]
    lstm_hidden_windows = reshape_to_windows(lstm_hidden_trunc)
    lstm_cell_windows = reshape_to_windows(lstm_cell_trunc)

    # Extract initial carry for each window (state at index 0 of each window)
    # [n_windows, window_size, N, dim] -> [n_windows, N, dim] (take index 0 along window_size axis)
    initial_hidden = lstm_hidden_windows[:, 0, :, :]  # [n_windows, N, lstm_hidden_dim]
    initial_cell = lstm_cell_windows[:, 0, :, :]  # [n_windows, N, lstm_hidden_dim]

    # Flatten windows and games: [n_windows * N]
    # We'll process each (window, game) pair as a separate sequence
    batch_size = n_windows * N

    def flatten_window_and_game(arr):
        # arr: [n_windows, window_size, N, ...] -> [window_size, n_windows * N, ...]
        # or [n_windows, N, ...] -> [n_windows * N, ...]
        if arr.ndim == 3:
            # [n_windows, window_size, N] -> [window_size, n_windows * N]
            return arr.transpose(1, 0, 2).reshape(window_size, -1)
        elif arr.ndim == 4:
            # [n_windows, window_size, N, feature_dim] -> [window_size, n_windows * N, feature_dim]
            return arr.transpose(1, 0, 2, 3).reshape(window_size, -1, arr.shape[-1])
        elif arr.ndim == 2:
            # [n_windows, N, ...] -> [n_windows * N, ...]
            return arr.reshape(-1, *arr.shape[2:])
        else:
            raise ValueError(f"Unexpected ndim: {arr.ndim}")

    # Reshape to [window_size, batch_size, ...]
    obs_batch = obs_windows.transpose(1, 0, 2, 3).reshape(window_size, batch_size, -1)
    opp_actions_batch = opp_actions_windows.transpose(1, 0, 2, 3).reshape(window_size, batch_size, -1)
    actions_batch = actions_windows.transpose(1, 0, 2).reshape(window_size, batch_size)
    log_probs_batch = log_probs_windows.transpose(1, 0, 2).reshape(window_size, batch_size)
    advantages_batch = advantages_windows.transpose(1, 0, 2).reshape(window_size, batch_size)
    returns_batch = returns_windows.transpose(1, 0, 2).reshape(window_size, batch_size)
    valid_masks_batch = valid_masks_windows.transpose(1, 0, 2, 3).reshape(window_size, batch_size, -1)
    model_masks_batch = model_masks_windows.transpose(1, 0, 2).reshape(window_size, batch_size) if model_masks_windows is not None else None

    # Initial carry: [n_windows, N, lstm_hidden_dim] -> [batch_size, lstm_hidden_dim]
    initial_carry = (
        initial_hidden.reshape(batch_size, -1),
        initial_cell.reshape(batch_size, -1)
    )

    # Now we have sequences: [window_size, batch_size, ...]
    # Run PPO updates with shuffling at batch level (not window level)
    # We'll do minibatch shuffling across the batch dimension

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

    # Check if we need model_masks
    use_model_masks = model_masks_batch is not None

    # Pack data into tuple for carry (enables JIT caching)
    # Use zeros if model_masks is None to maintain consistent pytree structure
    model_masks_for_data = model_masks_batch if use_model_masks else jnp.zeros_like(actions_batch)
    data = (
        obs_batch, opp_actions_batch, initial_carry[0], initial_carry[1],
        actions_batch, log_probs_batch, advantages_batch, returns_batch,
        valid_masks_batch, model_masks_for_data
    )

    # Get or create cached epoch function
    cache_key = (id(network), id(optimizer), config.num_minibatches, window_size, use_model_masks)
    if cache_key not in _sequential_epoch_fn_cache:
        _sequential_epoch_fn_cache[cache_key] = _create_ppo_sequential_epoch_fn(
            network, optimizer, config, window_size, use_model_masks
        )
    epoch_fn = _sequential_epoch_fn_cache[cache_key]

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
