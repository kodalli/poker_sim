"""State encoding for neural network input.

Encodes GameState into fixed-size observation vectors for the neural network.
"""

import jax
import jax.numpy as jnp
from jax import Array

from poker_jax.state import GameState, ROUND_PREFLOP
from poker_jax.deck import cards_to_one_hot, NUM_CARDS

# Observation dimension breakdown:
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
# - Valid actions: 5 (fold, check, call, raise, all_in)
# Total: 104 + 260 + 4 + 2 + 6 + 5 = 381

OBS_DIM = 381


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

    # === Valid actions ===
    # Compute valid action mask
    can_fold = jnp.ones(n_games, dtype=jnp.float32)

    # Can check if no bet to call
    can_check = (to_call == 0).astype(jnp.float32)

    # Can call if there's a bet and we have chips
    can_call = ((to_call > 0) & (my_chips >= to_call)).astype(jnp.float32)

    # Can raise if we have more than call amount
    min_raise_total = opp_bet + state.last_raise_amount
    can_raise = ((my_chips > to_call) & (my_chips + my_bet >= min_raise_total)).astype(jnp.float32)

    # Can always go all-in if we have chips
    can_all_in = (my_chips > 0).astype(jnp.float32)

    valid_actions = jnp.stack(
        [can_fold, can_check, can_call, can_raise, can_all_in], axis=1
    )  # [N, 5]

    # === Concatenate all features ===
    obs = jnp.concatenate([
        hole_flat,        # 104
        community_flat,   # 260
        round_one_hot,    # 4
        position,         # 2
        pot_norm,         # 1
        my_chips_norm,    # 1
        opp_chips_norm,   # 1
        my_bet_norm,      # 1
        opp_bet_norm,     # 1
        to_call_norm,     # 1
        valid_actions,    # 5
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
        [N, 5] valid action masks
    """
    # Valid actions are the last 5 elements
    return obs[:, -5:]


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

    # Valid actions
    valid = obs[idx:idx + 5]
    action_names = ["fold", "check", "call", "raise", "all_in"]
    valid_actions = [name for name, v in zip(action_names, valid) if v > 0.5]

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
