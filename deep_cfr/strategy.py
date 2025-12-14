"""Strategy computation for Single Deep CFR.

Converts advantage values from the neural network into action strategies
using regret matching (CFR+ style with positive regret clipping).
"""

import jax
import jax.numpy as jnp
from jax import Array

from poker_jax.state import NUM_ACTIONS


@jax.jit
def regret_match(advantages: Array, valid_mask: Array | None = None) -> Array:
    """Convert advantages to strategy using regret matching.

    CFR+ style regret matching:
    1. Clip advantages to positive values
    2. Normalize to get probability distribution
    3. Fall back to uniform over valid actions if all zero

    Args:
        advantages: [batch, num_actions] advantage values
        valid_mask: [batch, num_actions] bool mask (True = valid action)
                   If None, all actions are considered valid

    Returns:
        [batch, num_actions] action probabilities
    """
    # Default to all valid if no mask provided
    if valid_mask is None:
        valid_mask = jnp.ones_like(advantages, dtype=jnp.bool_)

    # Clip to positive (CFR+ style)
    positive_adv = jnp.maximum(advantages, 0.0)

    # Mask invalid actions
    masked_adv = jnp.where(valid_mask, positive_adv, 0.0)

    # Sum for normalization
    total = masked_adv.sum(axis=-1, keepdims=True)

    # Normalize to get strategy
    strategy = jnp.where(total > 0, masked_adv / total, 0.0)

    # Fallback: uniform over valid actions if all advantages are zero
    num_valid = valid_mask.sum(axis=-1, keepdims=True).astype(jnp.float32)
    uniform = jnp.where(valid_mask, 1.0 / jnp.maximum(num_valid, 1.0), 0.0)

    # Use uniform when total positive advantage is zero
    strategy = jnp.where(total > 0, strategy, uniform)

    return strategy


@jax.jit
def get_strategy(
    params: dict,
    network,
    obs: Array,
    valid_mask: Array,
) -> Array:
    """Compute strategy from advantage network.

    Full pipeline:
    1. Get advantages from network
    2. Apply regret matching
    3. Return strategy probabilities

    Args:
        params: Network parameters
        network: AdvantageNetwork module
        obs: [batch, obs_dim] game state observations
        valid_mask: [batch, num_actions] valid action mask

    Returns:
        [batch, num_actions] action probabilities
    """
    # Get advantage predictions
    advantages = network.apply({"params": params}, obs, training=False)

    # Convert to strategy via regret matching
    strategy = regret_match(advantages, valid_mask)

    return strategy


@jax.jit
def sample_action_from_strategy(
    rng_key: Array,
    strategy: Array,
) -> Array:
    """Sample action from strategy distribution.

    Args:
        rng_key: PRNG key for sampling
        strategy: [batch, num_actions] action probabilities

    Returns:
        [batch] sampled actions
    """
    # Add small epsilon to avoid numerical issues
    strategy = strategy + 1e-8
    strategy = strategy / strategy.sum(axis=-1, keepdims=True)

    # Sample from categorical
    actions = jax.random.categorical(rng_key, jnp.log(strategy), axis=-1)

    return actions


@jax.jit
def compute_expected_value(
    strategy: Array,
    action_values: Array,
    valid_mask: Array | None = None,
) -> Array:
    """Compute expected value under strategy.

    Args:
        strategy: [batch, num_actions] action probabilities
        action_values: [batch, num_actions] value of each action
        valid_mask: [batch, num_actions] valid action mask (optional)

    Returns:
        [batch] expected value
    """
    if valid_mask is not None:
        strategy = jnp.where(valid_mask, strategy, 0.0)
        action_values = jnp.where(valid_mask, action_values, 0.0)

    ev = (strategy * action_values).sum(axis=-1)
    return ev


@jax.jit
def compute_advantages_from_values(
    action_values: Array,
    strategy: Array,
    valid_mask: Array | None = None,
) -> Array:
    """Compute counterfactual advantages from action values.

    Advantage of action a = value(a) - expected_value
    where expected_value = sum(strategy[a] * value[a])

    Args:
        action_values: [batch, num_actions] value of each action
        strategy: [batch, num_actions] current strategy probabilities
        valid_mask: [batch, num_actions] valid action mask (optional)

    Returns:
        [batch, num_actions] advantage values
    """
    ev = compute_expected_value(strategy, action_values, valid_mask)
    advantages = action_values - ev[:, None]

    # Mask invalid actions
    if valid_mask is not None:
        advantages = jnp.where(valid_mask, advantages, 0.0)

    return advantages
