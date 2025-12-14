"""CFR opponent for use in training and evaluation.

Loads a precomputed CFR strategy and plays according to it.
Fully JAX-jittable for efficient integration with the training loop.
"""

from pathlib import Path
from typing import Optional

import jax
import jax.numpy as jnp
from jax import Array

from poker_jax.state import GameState, NUM_ACTIONS

from .abstraction import (
    compute_info_key_batch_v2,
    info_key_to_index,
    MAX_INFO_SETS,
)

# Global strategy storage (loaded at runtime)
_CFR_STRATEGY: Optional[Array] = None

# Hand strength index in observation
HAND_STRENGTH_OFFSET = 104 + 260 + 4 + 2 + 6 + 9  # = 385
NORMALIZED_STRENGTH_IDX = HAND_STRENGTH_OFFSET + 10  # = 395


def load_cfr_strategy(path: str) -> None:
    """Load CFR strategy from file.

    Args:
        path: Path to .npy file containing strategy array
    """
    global _CFR_STRATEGY
    strategy_path = Path(path)
    if not strategy_path.exists():
        raise FileNotFoundError(f"CFR strategy not found at {path}")

    _CFR_STRATEGY = jnp.load(str(strategy_path))
    print(f"Loaded CFR strategy from {path}, shape: {_CFR_STRATEGY.shape}")


def get_cfr_strategy() -> Array:
    """Get the loaded CFR strategy.

    Returns:
        [MAX_INFO_SETS, NUM_ACTIONS] strategy array

    Raises:
        RuntimeError: If strategy hasn't been loaded
    """
    if _CFR_STRATEGY is None:
        raise RuntimeError("CFR strategy not loaded. Call load_cfr_strategy() first.")
    return _CFR_STRATEGY


def create_cfr_opponent(strategy_path: str):
    """Create a CFR opponent function with embedded strategy.

    This creates a closure that captures the strategy, which can then
    be JIT compiled.

    Args:
        strategy_path: Path to strategy .npy file

    Returns:
        JAX-jittable opponent function
    """
    strategy = jnp.load(str(strategy_path))
    print(f"Creating CFR opponent with strategy from {strategy_path}")

    @jax.jit
    def cfr_opponent_with_strategy(
        state: GameState,
        valid_mask: Array,
        rng_key: Array,
        obs: Array,
    ) -> Array:
        """CFR opponent using preloaded strategy.

        Args:
            state: GameState [N games]
            valid_mask: [N, NUM_ACTIONS] valid action mask
            rng_key: PRNG key
            obs: [N, OBS_DIM] observations

        Returns:
            [N] action indices
        """
        n_games = state.done.shape[0]
        game_idx = jnp.arange(n_games)

        # Get current player's hole cards
        player_idx = state.current_player
        hole_cards = state.hole_cards[game_idx, player_idx, :]

        # Get hand strength from observation
        normalized_strength = obs[:, NORMALIZED_STRENGTH_IDX]

        # Compute info set indices using V2 abstraction
        info_keys = compute_info_key_batch_v2(
            hole_cards=hole_cards,
            street=state.round,
            normalized_strength=normalized_strength,
            pot=state.pot,
            bets=state.bets,
            current_player=state.current_player,
            button=state.button,
            starting_chips=jnp.full(n_games, 200.0),
            action_history=state.action_history,
            history_len=state.history_len,
        )
        info_indices = info_key_to_index(info_keys)

        # Lookup strategies
        strategies = strategy[info_indices]  # [N, NUM_ACTIONS]

        # Mask invalid actions and renormalize
        masked = jnp.where(valid_mask, strategies, 0.0)
        total = masked.sum(axis=-1, keepdims=True)
        probs = masked / (total + 1e-9)

        # Handle case where all probs are zero (use uniform over valid)
        uniform = valid_mask.astype(jnp.float32)
        uniform = uniform / (uniform.sum(axis=-1, keepdims=True) + 1e-9)
        probs = jnp.where(total > 0, probs, uniform)

        # Sample actions
        return jax.random.categorical(rng_key, jnp.log(probs + 1e-9))

    return cfr_opponent_with_strategy


@jax.jit
def cfr_opponent(
    state: GameState,
    valid_mask: Array,
    rng_key: Array,
    obs: Array,
) -> Array:
    """CFR opponent using globally loaded strategy.

    Note: load_cfr_strategy() must be called before using this function.

    Args:
        state: GameState [N games]
        valid_mask: [N, NUM_ACTIONS] valid action mask
        rng_key: PRNG key
        obs: [N, OBS_DIM] observations

    Returns:
        [N] action indices
    """
    strategy = get_cfr_strategy()

    n_games = state.done.shape[0]
    game_idx = jnp.arange(n_games)

    # Get current player's hole cards
    player_idx = state.current_player
    hole_cards = state.hole_cards[game_idx, player_idx, :]

    # Get hand strength from observation
    normalized_strength = obs[:, NORMALIZED_STRENGTH_IDX]

    # Compute info set indices using V2 abstraction
    info_keys = compute_info_key_batch_v2(
        hole_cards=hole_cards,
        street=state.round,
        normalized_strength=normalized_strength,
        pot=state.pot,
        bets=state.bets,
        current_player=state.current_player,
        button=state.button,
        starting_chips=jnp.full(n_games, 200.0),
        action_history=state.action_history,
        history_len=state.history_len,
    )
    info_indices = info_key_to_index(info_keys)

    # Lookup strategies
    strategies = strategy[info_indices]  # [N, NUM_ACTIONS]

    # Mask invalid actions and renormalize
    masked = jnp.where(valid_mask, strategies, 0.0)
    total = masked.sum(axis=-1, keepdims=True)
    probs = masked / (total + 1e-9)

    # Handle case where all probs are zero (use uniform over valid)
    uniform = valid_mask.astype(jnp.float32)
    uniform = uniform / (uniform.sum(axis=-1, keepdims=True) + 1e-9)
    probs = jnp.where(total > 0, probs, uniform)

    # Sample actions
    return jax.random.categorical(rng_key, jnp.log(probs + 1e-9))


def get_cfr_action_deterministic(
    state: GameState,
    valid_mask: Array,
    obs: Array,
) -> Array:
    """Get CFR action deterministically (argmax instead of sampling).

    Useful for evaluation/debugging.

    Args:
        state: GameState [N games]
        valid_mask: [N, NUM_ACTIONS] valid action mask
        obs: [N, OBS_DIM] observations

    Returns:
        [N] action indices (most probable valid action)
    """
    strategy = get_cfr_strategy()

    n_games = state.done.shape[0]
    game_idx = jnp.arange(n_games)

    # Get current player's hole cards
    player_idx = state.current_player
    hole_cards = state.hole_cards[game_idx, player_idx, :]

    # Get hand strength from observation
    normalized_strength = obs[:, NORMALIZED_STRENGTH_IDX]

    # Compute info set indices using V2 abstraction
    info_keys = compute_info_key_batch_v2(
        hole_cards=hole_cards,
        street=state.round,
        normalized_strength=normalized_strength,
        pot=state.pot,
        bets=state.bets,
        current_player=state.current_player,
        button=state.button,
        starting_chips=jnp.full(n_games, 200.0),
        action_history=state.action_history,
        history_len=state.history_len,
    )
    info_indices = info_key_to_index(info_keys)

    # Lookup strategies
    strategies = strategy[info_indices]  # [N, NUM_ACTIONS]

    # Mask invalid actions
    masked = jnp.where(valid_mask, strategies, -jnp.inf)

    # Argmax
    return jnp.argmax(masked, axis=-1)
