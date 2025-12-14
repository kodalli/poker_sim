"""Single Deep CFR opponent for evaluation.

Loads a trained advantage network and plays according to the
derived strategy (regret matching on advantages).
"""

from pathlib import Path
from typing import Optional
import pickle

import jax
import jax.numpy as jnp
from jax import Array

from poker_jax.state import GameState, NUM_ACTIONS
from poker_jax.encoding import encode_state_for_current_player, OBS_DIM

from deep_cfr.network import AdvantageNetwork, create_advantage_network
from deep_cfr.strategy import regret_match


# Global model storage
_SDCFR_NETWORK: Optional[AdvantageNetwork] = None
_SDCFR_PARAMS: Optional[dict] = None


def load_sdcfr_model(params_path: str, hidden_dims: tuple = (512, 256, 128)) -> None:
    """Load SD-CFR model from file.

    Args:
        params_path: Path to .pkl file containing network parameters
        hidden_dims: Network architecture (must match training)
    """
    global _SDCFR_NETWORK, _SDCFR_PARAMS

    model_path = Path(params_path)
    if not model_path.exists():
        raise FileNotFoundError(f"SD-CFR model not found at {params_path}")

    with open(model_path, 'rb') as f:
        _SDCFR_PARAMS = pickle.load(f)

    _SDCFR_NETWORK = create_advantage_network(hidden_dims=hidden_dims)
    print(f"Loaded SD-CFR model from {params_path}")


def get_sdcfr_model() -> tuple:
    """Get the loaded SD-CFR model.

    Returns:
        (network, params) tuple

    Raises:
        RuntimeError: If model hasn't been loaded
    """
    if _SDCFR_NETWORK is None or _SDCFR_PARAMS is None:
        raise RuntimeError("SD-CFR model not loaded. Call load_sdcfr_model() first.")
    return _SDCFR_NETWORK, _SDCFR_PARAMS


def create_sdcfr_opponent(params_path: str, hidden_dims: tuple = (512, 256, 128)):
    """Create a SD-CFR opponent function with embedded model.

    Args:
        params_path: Path to params .pkl file
        hidden_dims: Network architecture

    Returns:
        JAX-jittable opponent function
    """
    with open(params_path, 'rb') as f:
        params = pickle.load(f)

    network = create_advantage_network(hidden_dims=hidden_dims)
    print(f"Creating SD-CFR opponent with model from {params_path}")

    @jax.jit
    def sdcfr_opponent_with_model(
        state: GameState,
        valid_mask: Array,
        rng_key: Array,
        obs: Array,
    ) -> Array:
        """SD-CFR opponent using preloaded model.

        Args:
            state: GameState [N games]
            valid_mask: [N, NUM_ACTIONS] valid action mask
            rng_key: PRNG key
            obs: [N, OBS_DIM] observations

        Returns:
            [N] action indices
        """
        # Get advantages from network
        advantages = network.apply({"params": params}, obs, training=False)

        # Convert to strategy via regret matching
        strategy = regret_match(advantages, valid_mask)

        # Mask invalid actions and renormalize (safety)
        masked = jnp.where(valid_mask, strategy, 0.0)
        total = masked.sum(axis=-1, keepdims=True)
        probs = masked / (total + 1e-9)

        # Handle case where all probs are zero (use uniform over valid)
        uniform = valid_mask.astype(jnp.float32)
        uniform = uniform / (uniform.sum(axis=-1, keepdims=True) + 1e-9)
        probs = jnp.where(total > 0, probs, uniform)

        # Sample actions
        return jax.random.categorical(rng_key, jnp.log(probs + 1e-9))

    return sdcfr_opponent_with_model


@jax.jit
def sdcfr_opponent(
    state: GameState,
    valid_mask: Array,
    rng_key: Array,
    obs: Array,
) -> Array:
    """SD-CFR opponent using globally loaded model.

    Note: load_sdcfr_model() must be called before using this function.

    Args:
        state: GameState [N games]
        valid_mask: [N, NUM_ACTIONS] valid action mask
        rng_key: PRNG key
        obs: [N, OBS_DIM] observations

    Returns:
        [N] action indices
    """
    network, params = get_sdcfr_model()

    # Get advantages from network
    advantages = network.apply({"params": params}, obs, training=False)

    # Convert to strategy via regret matching
    strategy = regret_match(advantages, valid_mask)

    # Mask invalid actions and renormalize
    masked = jnp.where(valid_mask, strategy, 0.0)
    total = masked.sum(axis=-1, keepdims=True)
    probs = masked / (total + 1e-9)

    # Handle case where all probs are zero
    uniform = valid_mask.astype(jnp.float32)
    uniform = uniform / (uniform.sum(axis=-1, keepdims=True) + 1e-9)
    probs = jnp.where(total > 0, probs, uniform)

    # Sample actions
    return jax.random.categorical(rng_key, jnp.log(probs + 1e-9))


def get_sdcfr_action_deterministic(
    obs: Array,
    valid_mask: Array,
) -> Array:
    """Get SD-CFR action deterministically (argmax instead of sampling).

    Args:
        obs: [N, OBS_DIM] observations
        valid_mask: [N, NUM_ACTIONS] valid action mask

    Returns:
        [N] action indices (most probable valid action)
    """
    network, params = get_sdcfr_model()

    # Get advantages from network
    advantages = network.apply({"params": params}, obs, training=False)

    # Convert to strategy
    strategy = regret_match(advantages, valid_mask)

    # Mask invalid actions
    masked = jnp.where(valid_mask, strategy, -jnp.inf)

    # Argmax
    return jnp.argmax(masked, axis=-1)


def get_sdcfr_strategy(
    obs: Array,
    valid_mask: Array,
) -> Array:
    """Get SD-CFR strategy probabilities.

    Args:
        obs: [N, OBS_DIM] observations
        valid_mask: [N, NUM_ACTIONS] valid action mask

    Returns:
        [N, NUM_ACTIONS] strategy probabilities
    """
    network, params = get_sdcfr_model()

    # Get advantages from network
    advantages = network.apply({"params": params}, obs, training=False)

    # Convert to strategy
    strategy = regret_match(advantages, valid_mask)

    return strategy


def register_sdcfr_opponent(params_path: str, hidden_dims: tuple = (512, 256, 128)) -> None:
    """Register SD-CFR opponent in the opponent registry.

    Args:
        params_path: Path to params .pkl file
        hidden_dims: Network architecture
    """
    from evaluation.opponents import OPPONENT_TYPES, NEEDS_OBS

    sdcfr_opponent_fn = create_sdcfr_opponent(params_path, hidden_dims)
    OPPONENT_TYPES["sdcfr"] = sdcfr_opponent_fn
    NEEDS_OBS.add("sdcfr")
    print(f"Registered SD-CFR opponent as 'sdcfr'")
