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
    - Actor head: features -> action_logits (9) + bet_fraction (1)
    - Critic head: features -> value (1)

    v3 changes:
    - Increased hidden dims for more capacity
    - 9 actions: fold, check, call, raise_33, raise_66, raise_100, raise_150, all_in
    """

    hidden_dims: Sequence[int] = (512, 256, 128)  # Bigger for v3
    num_actions: int = 9  # v3: 9 discrete actions
    dropout_rate: float = 0.1

    @nn.compact
    def __call__(self, x: Array, training: bool = True) -> tuple[Array, Array, Array]:
        """Forward pass.

        Args:
            x: [batch, OBS_DIM] observations (433 dims in v3)
            training: Whether in training mode (affects dropout)

        Returns:
            Tuple of:
                action_logits: [batch, 9] logits for each action
                bet_fraction: [batch, 1] bet sizing (0-1, unused in v3)
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


# Constants for opponent model
OPPONENT_LSTM_HIDDEN = 64
OPPONENT_EMBED_DIM = 32
OPPONENT_ACTION_DIM = 13  # 9 action types one-hot + bet_amount + round + pot_odds + position


class OpponentLSTM(nn.Module):
    """LSTM that builds opponent model from observed actions.

    Maintains a hidden state that persists across hands, building up a
    representation of the opponent's playing style over time.

    v9 addition for learned opponent modeling.
    """

    hidden_dim: int = OPPONENT_LSTM_HIDDEN
    embed_dim: int = OPPONENT_EMBED_DIM

    @nn.compact
    def __call__(
        self, action_features: Array, carry: tuple[Array, Array]
    ) -> tuple[Array, tuple[Array, Array]]:
        """Update opponent model with new action observation.

        Args:
            action_features: [batch, OPPONENT_ACTION_DIM] encoded opponent action
            carry: (hidden, cell) tuple from previous step, each [batch, hidden_dim]

        Returns:
            opp_embed: [batch, embed_dim] opponent embedding for policy conditioning
            new_carry: (new_hidden, new_cell) for next step
        """
        # LSTM update
        new_carry, _ = nn.LSTMCell(features=self.hidden_dim)(carry, action_features)

        # Project hidden state to embedding
        opp_embed = nn.Dense(self.embed_dim, name="opp_embed_proj")(new_carry[0])

        return opp_embed, new_carry

    @staticmethod
    def init_carry(batch_size: int, hidden_dim: int = OPPONENT_LSTM_HIDDEN) -> tuple[Array, Array]:
        """Initialize LSTM carry state to zeros."""
        return (
            jnp.zeros((batch_size, hidden_dim)),
            jnp.zeros((batch_size, hidden_dim)),
        )


class ActorCriticMLPWithOpponentModel(nn.Module):
    """Actor-critic with learned opponent modeling.

    Extends ActorCriticMLP with an OpponentLSTM that encodes opponent behavior
    into a latent embedding. The embedding is concatenated with observations
    before the policy backbone.

    v9: Adds cross-hand memory for opponent modeling.
    """

    hidden_dims: Sequence[int] = (512, 256, 128)
    num_actions: int = 9
    dropout_rate: float = 0.1
    opponent_lstm_hidden: int = OPPONENT_LSTM_HIDDEN
    opponent_embed_dim: int = OPPONENT_EMBED_DIM

    @nn.compact
    def __call__(
        self,
        obs: Array,
        opp_action: Array,
        lstm_carry: tuple[Array, Array],
        training: bool = True,
    ) -> tuple[Array, Array, Array, tuple[Array, Array]]:
        """Forward pass with opponent modeling.

        Args:
            obs: [batch, OBS_DIM] observations
            opp_action: [batch, OPPONENT_ACTION_DIM] encoded last opponent action
                        (zeros if no opponent action yet this hand)
            lstm_carry: (hidden, cell) tuple for opponent LSTM
            training: Whether in training mode

        Returns:
            action_logits: [batch, 9] action logits
            bet_fraction: [batch, 1] bet sizing
            value: [batch, 1] state value
            new_carry: Updated LSTM carry for next step
        """
        # Update opponent model
        opp_lstm = OpponentLSTM(
            hidden_dim=self.opponent_lstm_hidden,
            embed_dim=self.opponent_embed_dim,
        )
        opp_embed, new_carry = opp_lstm(opp_action, lstm_carry)

        # Concatenate opponent embedding with observation
        x = jnp.concatenate([obs, opp_embed], axis=-1)

        # Shared backbone (same as ActorCriticMLP)
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

        return action_logits, bet_fraction, value, new_carry


class ActorCriticTransformer(nn.Module):
    """Transformer-based actor-critic for poker.

    WARNING: EXPERIMENTAL - This implementation is currently broken!
    The self-attention operates on a single token (the entire observation),
    which makes it equivalent to a simple linear layer. For meaningful
    self-attention, the input should be tokenized (e.g., per card).

    Use ActorCriticMLP instead for production training.

    Reserved for v3 architecture with proper tokenization.
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
    elif network_type == "mlp_opponent":
        return ActorCriticMLPWithOpponentModel(hidden_dims=hidden_dims)
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
        obs_dim: Observation dimension (should match OBS_DIM from encoding)

    Returns:
        Initialized parameters

    Raises:
        AssertionError: If obs_dim doesn't match expected OBS_DIM
    """
    if obs_dim != OBS_DIM:
        raise ValueError(
            f"obs_dim mismatch: got {obs_dim}, expected {OBS_DIM}. "
            f"Ensure encoding and network dimensions are consistent."
        )
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
        temperature: Sampling temperature (only affects sampling, not log_probs)

    Returns:
        Tuple of (actions [batch], log_probs [batch])

    Note:
        Log probabilities are computed from UNSCALED logits to ensure correct
        PPO importance weights. Temperature only affects the sampling distribution,
        not the returned log_probs used for training.
    """
    # Mask invalid actions with large negative value
    masked_logits = jnp.where(valid_mask, action_logits, -1e9)

    # Compute log probabilities from UNSCALED masked logits
    # This is critical for correct PPO importance sampling ratios
    log_probs = jax.nn.log_softmax(masked_logits, axis=-1)

    # Apply temperature ONLY for sampling (exploration)
    scaled_logits = masked_logits / temperature

    # Sample from categorical distribution using temperature-scaled logits
    actions = jax.random.categorical(rng_key, scaled_logits)

    # Extract log_probs for selected actions (from UNSCALED distribution)
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
