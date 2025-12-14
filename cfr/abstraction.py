"""Hand and betting history abstraction for CFR.

This module provides JAX-jittable functions for:
- Mapping hole cards to canonical hand buckets
- Mapping postflop hands to equity-based buckets (finer granularity)
- Encoding betting history with pot-relative sizing
- Computing information set keys for strategy lookup

V2: Improved abstraction with finer buckets and better history encoding.
"""

import jax
import jax.numpy as jnp
from jax import Array

from poker_jax.state import (
    GameState,
    NUM_ACTIONS,
    ROUND_PREFLOP,
    ROUND_FLOP,
    ROUND_TURN,
    ROUND_RIVER,
    ACTION_FOLD,
    ACTION_CHECK,
    ACTION_CALL,
    ACTION_RAISE_33,
    ACTION_RAISE_66,
    ACTION_RAISE_100,
    ACTION_RAISE_150,
    ACTION_ALL_IN,
)
from poker_jax.hands import evaluate_hand, get_hand_category

# ============================================================
# Abstraction Constants (V3 - Reduced for better convergence)
# ============================================================

# Hand buckets per street (V3 - reduced for more visits per bucket)
NUM_BUCKETS_PREFLOP = 169  # Canonical starting hands (unchanged)
NUM_BUCKETS_FLOP = 15      # Reduced from 50 for better convergence
NUM_BUCKETS_TURN = 10      # Reduced from 30
NUM_BUCKETS_RIVER = 10     # Reduced from 20

# V2 constants (kept for reference)
NUM_BUCKETS_FLOP_V2 = 50
NUM_BUCKETS_TURN_V2 = 30
NUM_BUCKETS_RIVER_V2 = 20

# Legacy constant for backwards compatibility
NUM_BUCKETS_POSTFLOP = 10

# History encoding (V3 - simplified)
POT_RATIO_BUCKETS = 3      # small(<0.5), medium(0.5-1.5), large(>1.5)
MAX_BETS_PER_STREET = 3    # 0, 1, 2+
BET_SIZE_BUCKETS = 3       # none(0), small(<0.75pot), large(>=0.75pot)

# Total history keys = pot_buckets * bets * bet_size = 3 * 3 * 3 = 27
NUM_HISTORY_KEYS_V2 = POT_RATIO_BUCKETS * MAX_BETS_PER_STREET * BET_SIZE_BUCKETS

# Legacy
MAX_RAISES_ENCODED = 4
NUM_HISTORY_KEYS = MAX_RAISES_ENCODED * 2

# Info set capacity (V3)
# Preflop: 169 * 27 * 2 = 9,126
# Flop: 15 * 27 * 2 = 810
# Turn: 10 * 27 * 2 = 540
# River: 10 * 27 * 2 = 540
# Total theoretical max: ~11,016
# Use buffer for hash collisions
MAX_INFO_SETS = 50_000


# ============================================================
# Preflop Bucketing (unchanged)
# ============================================================

@jax.jit
def preflop_bucket(hole_cards: Array) -> Array:
    """Map hole cards to canonical preflop bucket (0-168).

    Canonical hand encoding:
    - Pairs: 0-12 (22 through AA)
    - Suited: 13-90 (78 combinations, high-low indexed)
    - Offsuit: 91-168 (78 combinations, high-low indexed)

    Args:
        hole_cards: [N, 2] hole cards as integers 0-51

    Returns:
        [N] bucket indices (0-168)
    """
    ranks = hole_cards // 4
    r1, r2 = ranks[:, 0], ranks[:, 1]

    high_rank = jnp.maximum(r1, r2)
    low_rank = jnp.minimum(r1, r2)
    is_pair = r1 == r2

    suits = hole_cards % 4
    suited = suits[:, 0] == suits[:, 1]

    pair_bucket = high_rank
    triangle_idx = (high_rank * (high_rank - 1)) // 2 + low_rank
    triangle_idx = jnp.where(is_pair, 0, triangle_idx)

    suited_bucket = 13 + triangle_idx
    offsuit_bucket = 13 + 78 + triangle_idx

    bucket = jnp.where(
        is_pair, pair_bucket,
        jnp.where(suited, suited_bucket, offsuit_bucket)
    )

    return bucket.astype(jnp.int32)


# ============================================================
# Postflop Bucketing (V2 - Finer)
# ============================================================

@jax.jit
def postflop_bucket_v2(normalized_strength: Array, street: Array) -> Array:
    """Map postflop hand strength to bucket with street-specific granularity.

    V2: Uses different bucket counts per street for finer abstraction.

    Args:
        normalized_strength: [N] normalized hand strength (0-1) from obs
        street: [N] current street (1=flop, 2=turn, 3=river)

    Returns:
        [N] bucket indices
    """
    # Street-specific bucket counts
    flop_bucket = (normalized_strength * NUM_BUCKETS_FLOP).astype(jnp.int32)
    flop_bucket = jnp.clip(flop_bucket, 0, NUM_BUCKETS_FLOP - 1)

    turn_bucket = (normalized_strength * NUM_BUCKETS_TURN).astype(jnp.int32)
    turn_bucket = jnp.clip(turn_bucket, 0, NUM_BUCKETS_TURN - 1)

    river_bucket = (normalized_strength * NUM_BUCKETS_RIVER).astype(jnp.int32)
    river_bucket = jnp.clip(river_bucket, 0, NUM_BUCKETS_RIVER - 1)

    # Select based on street
    bucket = jnp.where(
        street == ROUND_FLOP, flop_bucket,
        jnp.where(street == ROUND_TURN, turn_bucket, river_bucket)
    )

    return bucket


@jax.jit
def river_bucket_by_hand_type(hole_cards: Array, community: Array) -> Array:
    """River bucket based on made hand category + relative strength.

    Uses actual hand evaluation for more accurate river bucketing.

    Args:
        hole_cards: [N, 2] hole cards
        community: [N, 5] community cards

    Returns:
        [N] bucket indices (0-19, 2 per hand category)
    """
    # Evaluate hands
    def eval_one(hole, comm):
        cards = jnp.concatenate([hole, comm])
        hand_value = evaluate_hand(cards)
        category = get_hand_category(hand_value)
        # Extract relative strength within category (kicker quality)
        # Higher bits after category = better within category
        within_strength = (hand_value >> 12) & 0xFF
        sub_bucket = jnp.where(within_strength > 128, 1, 0)
        return category * 2 + sub_bucket

    buckets = jax.vmap(eval_one)(hole_cards, community)
    return jnp.clip(buckets, 0, NUM_BUCKETS_RIVER - 1).astype(jnp.int32)


# Legacy function for backwards compatibility
@jax.jit
def postflop_bucket(normalized_strength: Array, street: Array) -> Array:
    """Map postflop hand strength to bucket (legacy 10-bucket version)."""
    bucket = (normalized_strength * NUM_BUCKETS_POSTFLOP).astype(jnp.int32)
    bucket = jnp.clip(bucket, 0, NUM_BUCKETS_POSTFLOP - 1)
    return bucket


# ============================================================
# Hand Bucket Selection
# ============================================================

@jax.jit
def get_hand_bucket(
    hole_cards: Array,
    street: Array,
    normalized_strength: Array,
) -> Array:
    """Get hand bucket for current street (legacy version)."""
    preflop_buckets = preflop_bucket(hole_cards)
    postflop_buckets = postflop_bucket(normalized_strength, street)

    is_preflop = street == ROUND_PREFLOP
    return jnp.where(is_preflop, preflop_buckets, postflop_buckets)


@jax.jit
def get_hand_bucket_v2(
    hole_cards: Array,
    street: Array,
    normalized_strength: Array,
) -> Array:
    """Get hand bucket for current street (V2 with finer buckets)."""
    preflop_buckets = preflop_bucket(hole_cards)
    postflop_buckets = postflop_bucket_v2(normalized_strength, street)

    is_preflop = street == ROUND_PREFLOP
    return jnp.where(is_preflop, preflop_buckets, postflop_buckets)


# ============================================================
# History Encoding (V2 - Pot-relative)
# ============================================================

@jax.jit
def quantize_pot_ratio(pot_ratio: Array) -> Array:
    """Quantize pot ratio into buckets (V3 - 3 buckets).

    Buckets: small(<0.5), medium(0.5-1.5), large(>1.5)
    """
    bucket = jnp.where(
        pot_ratio < 0.5, 0,
        jnp.where(pot_ratio < 1.5, 1, 2)
    )
    return bucket.astype(jnp.int32)


@jax.jit
def quantize_bet_size(bet_ratio: Array) -> Array:
    """Quantize bet size relative to pot (V3 - 3 buckets).

    Buckets: none(0), small(<0.75pot), large(>=0.75pot)
    """
    bucket = jnp.where(
        bet_ratio <= 0, 0,
        jnp.where(bet_ratio < 0.75, 1, 2)
    )
    return bucket.astype(jnp.int32)


@jax.jit
def encode_history_v2(
    pot: Array,
    bets: Array,
    current_player: Array,
    starting_chips: Array,
    action_history: Array,
    history_len: Array,
) -> Array:
    """Encode betting history with pot-relative sizing (V2).

    Args:
        pot: [N] current pot
        bets: [N, 2] current bets per player
        current_player: [N] current player index
        starting_chips: [N] starting chip count
        action_history: [N, MAX_HISTORY, 3] action history
        history_len: [N] number of actions

    Returns:
        [N] history keys
    """
    n_games = pot.shape[0]
    game_idx = jnp.arange(n_games)

    # 1. Pot ratio bucket
    total_pot = pot + bets[:, 0] + bets[:, 1]
    pot_ratio = total_pot / starting_chips
    pot_bucket = quantize_pot_ratio(pot_ratio)

    # 2. Count raises this hand (V3: 0, 1, 2+)
    actions_normalized = action_history[:, :, 1]
    is_valid = jnp.arange(action_history.shape[1])[None, :] < history_len[:, None]
    is_raise = (actions_normalized >= 0.5) & is_valid
    num_raises = jnp.sum(is_raise, axis=1)
    num_raises = jnp.minimum(num_raises, MAX_BETS_PER_STREET - 1)  # 0, 1, 2

    # 3. Facing bet size bucket
    my_bet = bets[game_idx, current_player]
    opp_bet = bets[game_idx, 1 - current_player]
    to_call = jnp.maximum(opp_bet - my_bet, 0)

    # Bet size relative to pot
    safe_pot = jnp.maximum(total_pot, 1)
    bet_ratio = to_call / safe_pot
    bet_bucket = quantize_bet_size(bet_ratio)

    # Combine: pot_bucket * (MAX_BETS * BET_SIZE) + num_raises * BET_SIZE + bet_bucket
    history_key = (
        pot_bucket * (MAX_BETS_PER_STREET * BET_SIZE_BUCKETS) +
        num_raises * BET_SIZE_BUCKETS +
        bet_bucket
    )

    return history_key.astype(jnp.int32)


# Legacy history encoding
@jax.jit
def encode_history(action_history: Array, history_len: Array) -> Array:
    """Encode betting history into compact key (legacy version)."""
    n_games = action_history.shape[0]

    actions_normalized = action_history[:, :, 1]
    is_action_valid = jnp.arange(action_history.shape[1])[None, :] < history_len[:, None]

    is_raise = (actions_normalized >= 0.5) & is_action_valid
    num_raises = jnp.sum(is_raise, axis=1)
    num_raises = jnp.minimum(num_raises, MAX_RAISES_ENCODED - 1)

    last_idx = jnp.maximum(history_len - 1, 0)
    batch_idx = jnp.arange(n_games)
    last_action = action_history[batch_idx, last_idx, 1]
    facing_bet = ((last_action >= 0.5) & (history_len > 0)).astype(jnp.int32)

    history_key = num_raises * 2 + facing_bet

    return history_key.astype(jnp.int32)


# ============================================================
# Position
# ============================================================

@jax.jit
def get_position(current_player: Array, street: Array, button: Array) -> Array:
    """Determine position (in position = 1, out of position = 0)."""
    is_preflop = street == ROUND_PREFLOP
    is_button = current_player == button
    position = jnp.where(is_preflop, ~is_button, is_button)
    return position.astype(jnp.int32)


# ============================================================
# Info Key Computation
# ============================================================

@jax.jit
def make_info_key(
    street: Array,
    hand_bucket: Array,
    history_key: Array,
    position: Array,
) -> Array:
    """Combine components into information set lookup key (legacy)."""
    key = (
        (street.astype(jnp.int32) << 24) |
        (hand_bucket.astype(jnp.int32) << 16) |
        (history_key.astype(jnp.int32) << 8) |
        position.astype(jnp.int32)
    )
    return key


@jax.jit
def make_info_key_v2(
    street: Array,
    hand_bucket: Array,
    history_key: Array,
    position: Array,
) -> Array:
    """Combine components into information set lookup key (V2).

    Encoding (more bits for larger buckets):
    - Bits 28-29: street (0-3)
    - Bits 20-27: hand_bucket (0-168 preflop, 0-49 postflop)
    - Bits 8-19: history_key (0-79)
    - Bits 0-7: position (0-1)
    """
    key = (
        (street.astype(jnp.int32) << 28) |
        (hand_bucket.astype(jnp.int32) << 20) |
        (history_key.astype(jnp.int32) << 8) |
        position.astype(jnp.int32)
    )
    return key


@jax.jit
def compute_info_key_batch(
    hole_cards: Array,
    street: Array,
    normalized_strength: Array,
    action_history: Array,
    history_len: Array,
    current_player: Array,
    button: Array,
) -> Array:
    """Compute info set keys for a batch of game states (legacy)."""
    hand_bucket = get_hand_bucket(hole_cards, street, normalized_strength)
    history_key = encode_history(action_history, history_len)
    position = get_position(current_player, street, button)

    return make_info_key(street, hand_bucket, history_key, position)


@jax.jit
def compute_info_key_batch_v2(
    hole_cards: Array,
    street: Array,
    normalized_strength: Array,
    pot: Array,
    bets: Array,
    current_player: Array,
    button: Array,
    starting_chips: Array,
    action_history: Array,
    history_len: Array,
) -> Array:
    """Compute info set keys for a batch of game states (V2 with finer abstraction)."""
    hand_bucket = get_hand_bucket_v2(hole_cards, street, normalized_strength)
    history_key = encode_history_v2(
        pot, bets, current_player, starting_chips,
        action_history, history_len
    )
    position = get_position(current_player, street, button)

    return make_info_key_v2(street, hand_bucket, history_key, position)


@jax.jit
def info_key_to_index(info_key: Array) -> Array:
    """Convert info key to array index for strategy lookup."""
    return info_key % MAX_INFO_SETS
