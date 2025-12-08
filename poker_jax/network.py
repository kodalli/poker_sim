"""Flax neural network for JAX poker agent."""

from typing import Sequence

import jax
import jax.numpy as jnp
import flax.linen as nn
from jax import Array

from poker_jax.encoding import OBS_DIM


class ActorCriticMLP(nn.Module):
    """Actor-critic MLP network for poker.

    Architecture:
    - Shared backbone: obs -> hidden layers -> features
    - Actor head: features -> action_logits (5) + bet_fraction (1)
    - Critic head: features -> value (1)
    """

    hidden_dims: Sequence[int] = (256, 128, 64)
    num_actions: int = 5  # fold, check, call, raise, all_in
    dropout_rate: float = 0.1

    @nn.compact
    def __call__(self, x: Array, training: bool = True) -> tuple[Array, Array, Array]:
        """Forward pass.

        Args:
            x: [batch, OBS_DIM] observations
            training: Whether in training mode (affects dropout)

        Returns:
            Tuple of:
                action_logits: [batch, 5] logits for each action
                bet_fraction: [batch, 1] bet sizing (0-1)
                value: [batch, 1] state value estimate
        """
        # Shared backbone
        for dim in self.hidden_dims:
            x = nn.Dense(dim)(x)
            x = nn.LayerNorm()(x)
            x = nn.relu(x)
            x = nn.Dropout(rate=self.dropout_rate, deterministic=not training)(x)

        # Actor head - action logits
        action_logits = nn.Dense(self.num_actions, name="action_head")(x)

        # Actor head - bet fraction
        bet_hidden = nn.Dense(32)(x)
        bet_hidden = nn.relu(bet_hidden)
        bet_fraction = nn.Dense(1)(bet_hidden)
        bet_fraction = nn.sigmoid(bet_fraction)

        # Critic head - value
        value_hidden = nn.Dense(64)(x)
        value_hidden = nn.relu(value_hidden)
        value = nn.Dense(1, name="value_head")(value_hidden)

        return action_logits, bet_fraction, value


class ActorCriticTransformer(nn.Module):
    """Transformer-based actor-critic for poker.

    Treats cards as tokens and uses self-attention.
    Reserved for v3 architecture.
    """

    num_heads: int = 4
    num_layers: int = 2
    embed_dim: int = 64
    mlp_dim: int = 128
    num_actions: int = 5
    dropout_rate: float = 0.1

    @nn.compact
    def __call__(self, x: Array, training: bool = True) -> tuple[Array, Array, Array]:
        """Forward pass with transformer architecture."""
        batch_size = x.shape[0]

        # Project input to embed_dim
        x = nn.Dense(self.embed_dim)(x)

        # Transformer layers
        for _ in range(self.num_layers):
            # Self-attention
            residual = x
            x = nn.LayerNorm()(x)
            # Reshape for attention: [batch, 1, embed_dim] since we have single sequence
            x = x[:, None, :]
            x = nn.SelfAttention(
                num_heads=self.num_heads,
                deterministic=not training,
                dropout_rate=self.dropout_rate,
            )(x)
            x = x[:, 0, :]  # Back to [batch, embed_dim]
            x = x + residual

            # MLP block
            residual = x
            x = nn.LayerNorm()(x)
            x = nn.Dense(self.mlp_dim)(x)
            x = nn.gelu(x)
            x = nn.Dropout(rate=self.dropout_rate, deterministic=not training)(x)
            x = nn.Dense(self.embed_dim)(x)
            x = x + residual

        x = nn.LayerNorm()(x)

        # Heads (same as MLP)
        action_logits = nn.Dense(self.num_actions)(x)

        bet_hidden = nn.Dense(32)(x)
        bet_hidden = nn.relu(bet_hidden)
        bet_fraction = nn.sigmoid(nn.Dense(1)(bet_hidden))

        value_hidden = nn.Dense(64)(x)
        value_hidden = nn.relu(value_hidden)
        value = nn.Dense(1)(value_hidden)

        return action_logits, bet_fraction, value


def create_network(
    network_type: str = "mlp",
    hidden_dims: Sequence[int] = (256, 128, 64),
) -> nn.Module:
    """Factory function to create network.

    Args:
        network_type: "mlp" or "transformer"
        hidden_dims: Hidden layer dimensions for MLP

    Returns:
        Flax Module
    """
    if network_type == "mlp":
        return ActorCriticMLP(hidden_dims=hidden_dims)
    elif network_type == "transformer":
        return ActorCriticTransformer()
    else:
        raise ValueError(f"Unknown network type: {network_type}")


def init_network(
    network: nn.Module,
    rng_key: Array,
    obs_dim: int = OBS_DIM,
) -> dict:
    """Initialize network parameters.

    Args:
        network: Flax module
        rng_key: PRNG key for initialization
        obs_dim: Observation dimension

    Returns:
        Initialized parameters
    """
    dummy_input = jnp.zeros((1, obs_dim))
    variables = network.init(rng_key, dummy_input, training=False)
    return variables["params"]


def count_parameters(params: dict) -> int:
    """Count total number of parameters."""
    return sum(x.size for x in jax.tree_util.tree_leaves(params))


@jax.jit
def apply_network(
    network: nn.Module,
    params: dict,
    obs: Array,
    training: bool = False,
    rngs: dict | None = None,
) -> tuple[Array, Array, Array]:
    """Apply network to observations.

    Args:
        network: Flax module
        params: Network parameters
        obs: [batch, obs_dim] observations
        training: Whether in training mode
        rngs: RNG dict for dropout (required if training=True)

    Returns:
        Tuple of (action_logits, bet_fraction, value)
    """
    return network.apply(
        {"params": params},
        obs,
        training=training,
        rngs=rngs or {},
    )


@jax.jit
def sample_action(
    rng_key: Array,
    action_logits: Array,
    valid_mask: Array,
    temperature: float = 1.0,
) -> tuple[Array, Array]:
    """Sample action from policy with masking.

    Args:
        rng_key: PRNG key
        action_logits: [batch, num_actions] raw logits
        valid_mask: [batch, num_actions] bool mask (True = valid)
        temperature: Sampling temperature

    Returns:
        Tuple of (actions [batch], log_probs [batch])
    """
    # Mask invalid actions with large negative value
    masked_logits = jnp.where(valid_mask, action_logits, -1e9)

    # Apply temperature
    scaled_logits = masked_logits / temperature

    # Sample from categorical distribution
    actions = jax.random.categorical(rng_key, scaled_logits)

    # Compute log probability
    log_probs = jax.nn.log_softmax(scaled_logits)
    action_log_probs = jnp.take_along_axis(
        log_probs, actions[:, None], axis=1
    ).squeeze(-1)

    return actions, action_log_probs


@jax.jit
def get_action_probs(
    action_logits: Array,
    valid_mask: Array,
) -> Array:
    """Get action probabilities with masking.

    Args:
        action_logits: [batch, num_actions] raw logits
        valid_mask: [batch, num_actions] bool mask

    Returns:
        [batch, num_actions] action probabilities
    """
    masked_logits = jnp.where(valid_mask, action_logits, -1e9)
    probs = jax.nn.softmax(masked_logits, axis=-1)
    return probs
