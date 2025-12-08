"""JAX-accelerated poker game engine for GPU training."""

from poker_jax.state import (
    GameState,
    PlayerState,
    create_initial_state,
    get_valid_actions_mask,
    ACTION_FOLD,
    ACTION_CHECK,
    ACTION_CALL,
    ACTION_RAISE,
    ACTION_ALL_IN,
    ROUND_PREFLOP,
    ROUND_FLOP,
    ROUND_TURN,
    ROUND_RIVER,
    ROUND_SHOWDOWN,
)
from poker_jax.game import reset, step, get_rewards
from poker_jax.encoding import encode_state, encode_state_for_current_player, OBS_DIM
from poker_jax.network import (
    ActorCriticMLP,
    create_network,
    init_network,
    sample_action,
    get_action_probs,
)
from poker_jax.hands import evaluate_hand, evaluate_hands_batch, determine_winner

__all__ = [
    # State
    "GameState",
    "PlayerState",
    "create_initial_state",
    "get_valid_actions_mask",
    # Actions
    "ACTION_FOLD",
    "ACTION_CHECK",
    "ACTION_CALL",
    "ACTION_RAISE",
    "ACTION_ALL_IN",
    # Rounds
    "ROUND_PREFLOP",
    "ROUND_FLOP",
    "ROUND_TURN",
    "ROUND_RIVER",
    "ROUND_SHOWDOWN",
    # Game
    "reset",
    "step",
    "get_rewards",
    # Encoding
    "encode_state",
    "encode_state_for_current_player",
    "OBS_DIM",
    # Network
    "ActorCriticMLP",
    "create_network",
    "init_network",
    "sample_action",
    "get_action_probs",
    # Hands
    "evaluate_hand",
    "evaluate_hands_batch",
    "determine_winner",
]
