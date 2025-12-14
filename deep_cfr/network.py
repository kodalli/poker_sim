"""Advantage network for Single Deep CFR.

The advantage network predicts counterfactual advantages for each action
given a game state observation. Unlike the actor-critic network used in PPO,
this network:
- Has no value head (advantages are relative, not absolute)
- Has no bet_fraction output (discrete actions only)
- Output represents counterfactual advantages, not action logits
"""

from typing import Sequence

import jax
import jax.numpy as jnp
import flax.linen as nn
from jax import Array

from poker_jax.encoding import OBS_DIM
from poker_jax.state import NUM_ACTIONS


class AdvantageNetwork(nn.Module):
    """Neural network predicting counterfactual advantages.

    Architecture:
    - Input: Game state observation (443 dims)
    - Hidden layers with LayerNorm and ReLU
    - Output: Advantage value per action (9 dims)

    The network learns to predict the expected utility difference
    between taking each action and the weighted average strategy value.
    """

    hidden_dims: Sequence[int] = (512, 256, 128)
    num_actions: int = NUM_ACTIONS  # 9 discrete actions
    dropout_rate: float = 0.1

    @nn.compact
    def __call__(self, x: Array, training: bool = False) -> Array:
        """Forward pass.

        Args:
            x: [batch, OBS_DIM] game state observations (443 dims)
            training: Whether in training mode (affects dropout)

        Returns:
            advantages: [batch, num_actions] advantage estimate per action
        """
        # Hidden layers with LayerNorm and ReLU
        for dim in self.hidden_dims:
            x = nn.Dense(dim)(x)
            x = nn.LayerNorm()(x)
            x = nn.relu(x)
            x = nn.Dropout(rate=self.dropout_rate, deterministic=not training)(x)

        # Output layer - raw advantages
        advantages = nn.Dense(self.num_actions, name="advantage_head")(x)

        return advantages


def create_advantage_network(
    hidden_dims: Sequence[int] = (512, 256, 128),
) -> AdvantageNetwork:
    """Factory function to create advantage network.

    Args:
        hidden_dims: Hidden layer dimensions

    Returns:
        AdvantageNetwork module
    """
    return AdvantageNetwork(hidden_dims=hidden_dims)


def init_advantage_network(
    network: AdvantageNetwork,
    rng_key: Array,
    obs_dim: int = OBS_DIM,
) -> dict:
    """Initialize network parameters.

    Args:
        network: AdvantageNetwork module
        rng_key: PRNG key for initialization
        obs_dim: Observation dimension (should match OBS_DIM)

    Returns:
        Initialized parameters
    """
    if obs_dim != OBS_DIM:
        raise ValueError(
            f"obs_dim mismatch: got {obs_dim}, expected {OBS_DIM}. "
            f"Ensure encoding and network dimensions are consistent."
        )
    dummy_input = jnp.zeros((1, obs_dim))
    variables = network.init(rng_key, dummy_input, training=False)
    return variables["params"]


@jax.jit
def apply_advantage_network(
    network: AdvantageNetwork,
    params: dict,
    obs: Array,
    training: bool = False,
    rngs: dict | None = None,
) -> Array:
    """Apply network to observations.

    Args:
        network: AdvantageNetwork module
        params: Network parameters
        obs: [batch, obs_dim] observations
        training: Whether in training mode
        rngs: RNG dict for dropout (required if training=True)

    Returns:
        [batch, num_actions] advantage values
    """
    return network.apply(
        {"params": params},
        obs,
        training=training,
        rngs=rngs or {},
    )


def count_parameters(params: dict) -> int:
    """Count total number of parameters."""
    return sum(x.size for x in jax.tree_util.tree_leaves(params))
