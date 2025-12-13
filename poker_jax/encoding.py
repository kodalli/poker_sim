"""State encoding for neural network input.

Encodes GameState into fixed-size observation vectors for the neural network.
"""

import jax
import jax.numpy as jnp
from jax import Array

from poker_jax.state import GameState, ROUND_PREFLOP, MAX_HISTORY, NUM_ACTIONS
from poker_jax.deck import cards_to_one_hot, NUM_CARDS
from poker_jax.hands import compute_hand_strength_batch

# Observation dimension breakdown (v5):
# - Hole cards: 52 * 2 = 104 (one-hot for each card)
# - Community cards: 52 * 5 = 260 (one-hot for each card)
# - Round: 4 (one-hot: preflop, flop, turn, river)
# - Position: 2 (one-hot: button or not)
# - Pot (normalized): 1
# - My chips (normalized): 1
# - Opponent chips (normalized): 1
# - My bet (normalized): 1
# - Opponent bet (normalized): 1
# - To call (normalized): 1
# - Valid actions: 9 (fold, check, call, raise_33, raise_66, raise_100, raise_150, all_in)
# - Hand strength: 18 (10 category one-hot + 1 strength + 4 draws + 3 texture)
# - Action history: 40 (10 actions * 4 features: player_id, action_type, bet_amount, position)
# Total: 104 + 260 + 4 + 2 + 6 + 9 + 18 + 40 = 443

HAND_STRENGTH_DIM = 18
ACTION_HISTORY_DIM = MAX_HISTORY * 4  # 10 * 4 = 40 (added position encoding)

OBS_DIM = 443


@jax.jit
def encode_state(state: GameState, player_id: int = 0) -> Array:
    """Encode game state as observation vector for one player.

    Args:
        state: Game state for N games
        player_id: Which player's perspective (0 or 1)

    Returns:
        [N, OBS_DIM] observation vectors
    """
    n_games = state.done.shape[0]
    game_idx = jnp.arange(n_games)

    # Opponent ID
    opp_id = 1 - player_id

    # === Hole cards (one-hot) ===
    # [N, 2] -> [N, 2, 52] -> [N, 104]
    my_hole = state.hole_cards[:, player_id, :]  # [N, 2]
    hole_one_hot = cards_to_one_hot(my_hole)  # [N, 2, 52]
    hole_flat = hole_one_hot.reshape(n_games, -1)  # [N, 104]

    # === Community cards (one-hot) ===
    # [N, 5] -> [N, 5, 52] -> [N, 260]
    community_one_hot = cards_to_one_hot(state.community)  # [N, 5, 52]
    community_flat = community_one_hot.reshape(n_games, -1)  # [N, 260]

    # === Round (one-hot) ===
    round_one_hot = jax.nn.one_hot(state.round, 4)  # [N, 4]

    # === Position (am I the button?) ===
    is_button = (state.button == player_id).astype(jnp.float32)  # [N]
    position = jnp.stack([is_button, 1 - is_button], axis=1)  # [N, 2]

    # === Chip/Bet information (normalized by starting chips) ===
    # All numeric features are normalized to approximately [0, 2] range
    # Values > 1.0 can occur when a player has won chips
    norm = state.starting_chips.astype(jnp.float32)

    # Current pot (including current round bets)
    total_pot = state.pot + state.bets[:, 0] + state.bets[:, 1]
    pot_norm = jnp.clip(total_pot / norm, 0.0, 4.0)[:, None]  # [N, 1], clamp to prevent extreme values

    # My chips
    my_chips = state.chips[:, player_id]
    my_chips_norm = jnp.clip(my_chips / norm, 0.0, 4.0)[:, None]  # [N, 1]

    # Opponent chips
    opp_chips = state.chips[:, opp_id]
    opp_chips_norm = jnp.clip(opp_chips / norm, 0.0, 4.0)[:, None]  # [N, 1]

    # My current bet
    my_bet = state.bets[:, player_id]
    my_bet_norm = jnp.clip(my_bet / norm, 0.0, 4.0)[:, None]  # [N, 1]

    # Opponent current bet
    opp_bet = state.bets[:, opp_id]
    opp_bet_norm = jnp.clip(opp_bet / norm, 0.0, 4.0)[:, None]  # [N, 1]

    # Amount to call
    to_call = jnp.maximum(opp_bet - my_bet, 0)
    to_call_norm = jnp.clip(to_call / norm, 0.0, 4.0)[:, None]  # [N, 1]

    # === Valid actions (9 actions in v3) ===
    # Compute valid action mask
    can_fold = jnp.ones(n_games, dtype=jnp.float32)

    # Can check if no bet to call
    can_check = (to_call == 0).astype(jnp.float32)

    # Can call if there's a bet and we have chips
    can_call = ((to_call > 0) & (my_chips >= to_call)).astype(jnp.float32)

    # Current pot for raise calculations
    pot = state.pot + state.bets[:, 0] + state.bets[:, 1]

    # Min raise floor
    min_raise_total = opp_bet + state.last_raise_amount

    # For each raise size, check if player can afford it (but not be all-in)
    def can_make_raise(fraction):
        raise_amount = jnp.maximum(opp_bet + (pot * fraction).astype(jnp.int32), min_raise_total)
        chips_needed = raise_amount - my_bet
        return ((my_chips > to_call) & (my_chips >= chips_needed) & (my_chips > chips_needed)).astype(jnp.float32)

    can_raise_33 = can_make_raise(0.33)
    can_raise_66 = can_make_raise(0.66)
    can_raise_100 = can_make_raise(1.0)
    can_raise_150 = can_make_raise(1.5)

    # Can always go all-in if we have chips
    can_all_in = (my_chips > 0).astype(jnp.float32)

    valid_actions = jnp.stack(
        [can_fold, can_check, can_call, can_raise_33, can_raise_66, can_raise_100, can_raise_150, can_all_in], axis=1
    )  # [N, 8] - note: index 0 (ACTION_NONE) not included, starts at ACTION_FOLD

    # Pad to NUM_ACTIONS (9) - first element is ACTION_NONE which is always invalid
    valid_actions_full = jnp.concatenate([
        jnp.zeros((n_games, 1), dtype=jnp.float32),  # ACTION_NONE = 0
        valid_actions,  # 8 actions
    ], axis=1)  # [N, 9]

    # === Hand strength features (18 dims) ===
    hand_strength = compute_hand_strength_batch(my_hole, state.community)  # [N, 18]

    # === Action history (40 dims) ===
    # Add positional encoding to each action slot for temporal structure
    # Original: [N, 10, 3] with features [player_id, action_type, bet_amount]
    # Enhanced: [N, 10, 4] with features [player_id, action_type, bet_amount, position]
    # Position = slot_index / 10.0 (0.0 = oldest, 0.9 = most recent)
    position_encoding = jnp.arange(MAX_HISTORY, dtype=jnp.float32) / MAX_HISTORY  # [10]
    position_encoding = jnp.broadcast_to(position_encoding[None, :, None], (n_games, MAX_HISTORY, 1))  # [N, 10, 1]
    action_history_with_pos = jnp.concatenate([state.action_history, position_encoding], axis=2)  # [N, 10, 4]
    action_history_flat = action_history_with_pos.reshape(n_games, -1)  # [N, 40]

    # === Concatenate all features ===
    obs = jnp.concatenate([
        hole_flat,            # 104
        community_flat,       # 260
        round_one_hot,        # 4
        position,             # 2
        pot_norm,             # 1
        my_chips_norm,        # 1
        opp_chips_norm,       # 1
        my_bet_norm,          # 1
        opp_bet_norm,         # 1
        to_call_norm,         # 1
        valid_actions_full,   # 9
        hand_strength,        # 18
        action_history_flat,  # 40 (was 30, now includes position encoding)
    ], axis=1)

    return obs


@jax.jit
def encode_state_for_current_player(state: GameState) -> Array:
    """Encode state from perspective of current player to act.

    Args:
        state: Game state for N games

    Returns:
        [N, OBS_DIM] observations from current player's perspective
    """
    n_games = state.done.shape[0]
    game_idx = jnp.arange(n_games)

    # Get observations for both players
    obs_p0 = encode_state(state, player_id=0)
    obs_p1 = encode_state(state, player_id=1)

    # Select based on current player
    is_p0 = (state.current_player == 0)[:, None]
    obs = jnp.where(is_p0, obs_p0, obs_p1)

    return obs


@jax.jit
def get_valid_actions_from_obs(obs: Array) -> Array:
    """Extract valid actions mask from observation.

    Args:
        obs: [N, OBS_DIM] observations

    Returns:
        [N, NUM_ACTIONS] valid action masks (9 actions)
    """
    # Valid actions are at position: after chips/bets (377), before hand_strength and action_history
    # Layout: ... chips/bets (6) | valid_actions (9) | hand_strength (18) | action_history (30)
    # So valid_actions start at index 377 and go to 386
    start_idx = 104 + 260 + 4 + 2 + 6  # = 376
    return obs[:, start_idx:start_idx + NUM_ACTIONS]


def describe_observation(obs: Array) -> dict:
    """Describe observation contents (for debugging).

    Args:
        obs: [OBS_DIM] single observation

    Returns:
        Dictionary with decoded observation
    """
    idx = 0

    # Hole cards
    hole_one_hot = obs[idx:idx + 104].reshape(2, 52)
    idx += 104

    # Community
    community_one_hot = obs[idx:idx + 260].reshape(5, 52)
    idx += 260

    # Round
    round_one_hot = obs[idx:idx + 4]
    round_idx = int(jnp.argmax(round_one_hot))
    round_names = ["preflop", "flop", "turn", "river"]
    idx += 4

    # Position
    position = obs[idx:idx + 2]
    is_button = bool(position[0] > 0.5)
    idx += 2

    # Numeric features
    pot = float(obs[idx])
    idx += 1
    my_chips = float(obs[idx])
    idx += 1
    opp_chips = float(obs[idx])
    idx += 1
    my_bet = float(obs[idx])
    idx += 1
    opp_bet = float(obs[idx])
    idx += 1
    to_call = float(obs[idx])
    idx += 1

    # Valid actions (v3: 9 actions including ACTION_NONE at index 0)
    valid = obs[idx:idx + NUM_ACTIONS]
    action_names = ["none", "fold", "check", "call", "raise_33", "raise_66", "raise_100", "raise_150", "all_in"]
    valid_actions = [name for name, v in zip(action_names, valid) if v > 0.5]
    idx += NUM_ACTIONS

    # Decode cards
    def decode_card(one_hot):
        if jnp.sum(one_hot) < 0.5:
            return None
        card_idx = int(jnp.argmax(one_hot))
        rank = card_idx // 4
        suit = card_idx % 4
        rank_names = "23456789TJQKA"
        suit_names = "cdhs"
        return f"{rank_names[rank]}{suit_names[suit]}"

    hole_cards = [decode_card(hole_one_hot[i]) for i in range(2)]
    community_cards = [decode_card(community_one_hot[i]) for i in range(5)]

    return {
        "hole_cards": hole_cards,
        "community_cards": community_cards,
        "round": round_names[round_idx],
        "is_button": is_button,
        "pot": pot,
        "my_chips": my_chips,
        "opp_chips": opp_chips,
        "my_bet": my_bet,
        "opp_bet": opp_bet,
        "to_call": to_call,
        "valid_actions": valid_actions,
    }


# === Opponent Action Encoding (v9) ===

OPPONENT_ACTION_DIM = 13  # 9 action types + bet_amount + round + pot_odds + position


@jax.jit
def encode_opponent_action(
    action_type: Array,
    bet_amount: Array,
    round_idx: Array,
    pot: Array,
    starting_chips: Array,
    is_button: Array,
) -> Array:
    """Encode opponent action for LSTM input.

    Args:
        action_type: [N] int action type (0-8)
        bet_amount: [N] int bet amount in chips
        round_idx: [N] int round (0=preflop, 1=flop, 2=turn, 3=river)
        pot: [N] int current pot size
        starting_chips: [N] int starting chips for normalization
        is_button: [N] bool whether opponent is button

    Returns:
        [N, OPPONENT_ACTION_DIM] encoded opponent action features
    """
    n_games = action_type.shape[0]

    # One-hot encode action type (9 dims)
    action_one_hot = jax.nn.one_hot(action_type, NUM_ACTIONS)  # [N, 9]

    # Normalize bet amount
    bet_norm = (bet_amount / starting_chips.astype(jnp.float32)).clip(0.0, 4.0)[:, None]  # [N, 1]

    # Normalize round (0-1 range)
    round_norm = (round_idx / 3.0)[:, None]  # [N, 1]

    # Pot odds (bet / pot, clipped)
    pot_odds = jnp.where(
        pot > 0,
        (bet_amount / pot.astype(jnp.float32)).clip(0.0, 4.0),
        jnp.zeros_like(bet_amount, dtype=jnp.float32),
    )[:, None]  # [N, 1]

    # Position (button indicator)
    position = is_button.astype(jnp.float32)[:, None]  # [N, 1]

    # Concatenate: 9 + 1 + 1 + 1 + 1 = 13 dims
    opp_action_features = jnp.concatenate([
        action_one_hot,  # 9
        bet_norm,        # 1
        round_norm,      # 1
        pot_odds,        # 1
        position,        # 1
    ], axis=1)

    return opp_action_features


@jax.jit
def encode_opponent_action_from_state(
    state: GameState,
    action_type: Array,
    bet_amount: Array,
    current_player: Array,
) -> Array:
    """Encode opponent action using current game state.

    Convenience wrapper that extracts state info for encode_opponent_action.

    Args:
        state: Current game state
        action_type: [N] action taken by opponent
        bet_amount: [N] bet amount by opponent
        current_player: [N] current player (opponent is 1 - current_player)

    Returns:
        [N, OPPONENT_ACTION_DIM] encoded opponent action
    """
    opp_id = 1 - current_player
    pot = state.pot + state.bets[:, 0] + state.bets[:, 1]
    is_button = (state.button == opp_id)

    return encode_opponent_action(
        action_type=action_type,
        bet_amount=bet_amount,
        round_idx=state.round,
        pot=pot,
        starting_chips=state.starting_chips,
        is_button=is_button,
    )
