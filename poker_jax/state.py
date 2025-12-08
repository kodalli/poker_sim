"""JAX-compatible game state for vectorized poker simulation."""

from typing import NamedTuple

import jax.numpy as jnp
from jax import Array


class PlayerState(NamedTuple):
    """State for a single player (vectorized over N games)."""

    chips: Array  # [N] chip stack
    bet: Array  # [N] current round bet
    total_invested: Array  # [N] total chips invested this hand
    folded: Array  # [N] bool - has player folded
    all_in: Array  # [N] bool - is player all-in


class GameState(NamedTuple):
    """Complete game state for N parallel heads-up poker games.

    All arrays have first dimension N (number of parallel games).
    Cards are represented as integers 0-51:
        card = rank * 4 + suit
        rank: 0=2, 1=3, ..., 12=A
        suit: 0=clubs, 1=diamonds, 2=hearts, 3=spades
    """

    # === Card state ===
    # Deck represented as remaining cards (for efficient sampling)
    deck: Array  # [N, 52] shuffled deck indices
    deck_idx: Array  # [N] next card index to deal

    # Hole cards: [N, 2, 2] = [games, players, cards]
    hole_cards: Array

    # Community cards: [N, 5] (-1 = not yet dealt)
    community: Array

    # === Player state ===
    # Using separate arrays instead of nested struct for JAX compatibility
    chips: Array  # [N, 2] chip stacks
    bets: Array  # [N, 2] current round bets
    total_invested: Array  # [N, 2] total invested this hand
    folded: Array  # [N, 2] bool
    all_in: Array  # [N, 2] bool

    # === Pot ===
    pot: Array  # [N] main pot

    # === Game flow ===
    # Round: 0=preflop, 1=flop, 2=turn, 3=river, 4=showdown
    round: Array  # [N] int

    # Current player to act (0 or 1)
    current_player: Array  # [N] int

    # Button position (0 or 1) - determines who posts blinds
    button: Array  # [N] int

    # Number of actions taken this round (for checking if round complete)
    actions_this_round: Array  # [N] int

    # Last action type taken (for determining valid actions)
    # 0=none, 1=fold, 2=check, 3=call, 4=raise, 5=all_in
    last_action: Array  # [N] int

    # Last raise amount (for min-raise calculation)
    last_raise_amount: Array  # [N] int

    # === Game completion ===
    done: Array  # [N] bool - is game complete
    winner: Array  # [N] int - winning player (0, 1, or -1 for split)

    # === Blinds (constant per game) ===
    small_blind: Array  # [N] int
    big_blind: Array  # [N] int
    starting_chips: Array  # [N] int

    # === Random state ===
    rng_key: Array  # [N, 2] PRNG keys

    # === Action history (v3) ===
    # Stores last MAX_HISTORY actions: [N, MAX_HISTORY, 3]
    # Each action: [player_id, action_type, bet_amount_normalized]
    action_history: Array  # [N, 10, 3]
    history_len: Array  # [N] number of actions recorded this hand


# Action constants
ACTION_NONE = 0
ACTION_FOLD = 1
ACTION_CHECK = 2
ACTION_CALL = 3
ACTION_RAISE_33 = 4   # 33% pot (probe/blocking bet)
ACTION_RAISE_66 = 5   # 66% pot (standard bet)
ACTION_RAISE_100 = 6  # 100% pot (polarized)
ACTION_RAISE_150 = 7  # 150% pot (overbet)
ACTION_ALL_IN = 8

# For backwards compat and simple raise references
ACTION_RAISE = ACTION_RAISE_66  # Default raise is 66% pot

NUM_ACTIONS = 9  # Total action count for network output

# Action history constants
MAX_HISTORY = 10  # Track last 10 actions

# Round constants
ROUND_PREFLOP = 0
ROUND_FLOP = 1
ROUND_TURN = 2
ROUND_RIVER = 3
ROUND_SHOWDOWN = 4


def create_initial_state(
    rng_key: Array,
    n_games: int,
    starting_chips: int = 200,
    small_blind: int = 1,
    big_blind: int = 2,
) -> GameState:
    """Create initial game state for N parallel games.

    Args:
        rng_key: JAX PRNG key
        n_games: Number of parallel games
        starting_chips: Starting chip count per player
        small_blind: Small blind amount
        big_blind: Big blind amount

    Returns:
        GameState with all games ready to start (but not yet dealt)
    """
    import jax.random as jrandom

    # Split keys for each game
    keys = jrandom.split(rng_key, n_games)

    return GameState(
        # Cards - will be filled by reset()
        deck=jnp.zeros((n_games, 52), dtype=jnp.int32),
        deck_idx=jnp.zeros(n_games, dtype=jnp.int32),
        hole_cards=jnp.full((n_games, 2, 2), -1, dtype=jnp.int32),
        community=jnp.full((n_games, 5), -1, dtype=jnp.int32),
        # Player state
        chips=jnp.full((n_games, 2), starting_chips, dtype=jnp.int32),
        bets=jnp.zeros((n_games, 2), dtype=jnp.int32),
        total_invested=jnp.zeros((n_games, 2), dtype=jnp.int32),
        folded=jnp.zeros((n_games, 2), dtype=jnp.bool_),
        all_in=jnp.zeros((n_games, 2), dtype=jnp.bool_),
        # Pot
        pot=jnp.zeros(n_games, dtype=jnp.int32),
        # Game flow
        round=jnp.zeros(n_games, dtype=jnp.int32),
        current_player=jnp.zeros(n_games, dtype=jnp.int32),
        button=jnp.zeros(n_games, dtype=jnp.int32),
        actions_this_round=jnp.zeros(n_games, dtype=jnp.int32),
        last_action=jnp.zeros(n_games, dtype=jnp.int32),
        last_raise_amount=jnp.full(n_games, big_blind, dtype=jnp.int32),
        # Completion
        done=jnp.zeros(n_games, dtype=jnp.bool_),
        winner=jnp.full(n_games, -1, dtype=jnp.int32),
        # Blinds
        small_blind=jnp.full(n_games, small_blind, dtype=jnp.int32),
        big_blind=jnp.full(n_games, big_blind, dtype=jnp.int32),
        starting_chips=jnp.full(n_games, starting_chips, dtype=jnp.int32),
        # RNG
        rng_key=keys,
        # Action history (v3)
        action_history=jnp.zeros((n_games, MAX_HISTORY, 3), dtype=jnp.float32),
        history_len=jnp.zeros(n_games, dtype=jnp.int32),
    )


def get_valid_actions_mask(state: GameState) -> Array:
    """Get mask of valid actions for current player in each game.

    Returns:
        [N, NUM_ACTIONS] bool mask where True = action is valid
        Actions: 0=none, 1=fold, 2=check, 3=call, 4-7=raise sizes, 8=all_in
    """
    n_games = state.done.shape[0]
    mask = jnp.zeros((n_games, NUM_ACTIONS), dtype=jnp.bool_)

    # Get current player's state
    player_idx = state.current_player
    my_chips = jnp.take_along_axis(
        state.chips, player_idx[:, None], axis=1
    ).squeeze(-1)
    my_bet = jnp.take_along_axis(
        state.bets, player_idx[:, None], axis=1
    ).squeeze(-1)

    # Opponent's bet
    opp_idx = 1 - player_idx
    opp_bet = jnp.take_along_axis(
        state.bets, opp_idx[:, None], axis=1
    ).squeeze(-1)

    # Amount needed to call
    to_call = opp_bet - my_bet

    # Current pot for raise calculations
    pot = state.pot + state.bets[:, 0] + state.bets[:, 1]

    # Can always fold if game not done
    can_fold = ~state.done

    # Can check if no bet to call
    can_check = ~state.done & (to_call == 0)

    # Can call if there's a bet to call and we have chips
    can_call = ~state.done & (to_call > 0) & (my_chips >= to_call)

    # Min raise = last raise amount (or big blind)
    min_raise_total = opp_bet + state.last_raise_amount

    # For each raise size, check if player can afford it
    # Raise = opp_bet + (pot * fraction), but at least min_raise_total
    def can_make_raise(fraction):
        raise_amount = jnp.maximum(opp_bet + (pot * fraction).astype(jnp.int32), min_raise_total)
        chips_needed = raise_amount - my_bet
        # Can raise if: not done, have more than call amount, and can afford this size (but not all-in)
        return ~state.done & (my_chips > to_call) & (my_chips >= chips_needed) & (my_chips > chips_needed)

    can_raise_33 = can_make_raise(0.33)
    can_raise_66 = can_make_raise(0.66)
    can_raise_100 = can_make_raise(1.0)
    can_raise_150 = can_make_raise(1.5)

    # Can go all-in if we have chips
    can_all_in = ~state.done & (my_chips > 0)

    # Build mask
    mask = mask.at[:, ACTION_FOLD].set(can_fold)
    mask = mask.at[:, ACTION_CHECK].set(can_check)
    mask = mask.at[:, ACTION_CALL].set(can_call)
    mask = mask.at[:, ACTION_RAISE_33].set(can_raise_33)
    mask = mask.at[:, ACTION_RAISE_66].set(can_raise_66)
    mask = mask.at[:, ACTION_RAISE_100].set(can_raise_100)
    mask = mask.at[:, ACTION_RAISE_150].set(can_raise_150)
    mask = mask.at[:, ACTION_ALL_IN].set(can_all_in)

    return mask
