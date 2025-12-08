"""Vectorized poker hand evaluation in JAX.

Hand ranking (encoded in upper bits for comparison):
    9 - Royal Flush
    8 - Straight Flush
    7 - Four of a Kind
    6 - Full House
    5 - Flush
    4 - Straight
    3 - Three of a Kind
    2 - Two Pair
    1 - One Pair
    0 - High Card

Hand value encoding (32-bit int):
    Bits 28-31: Hand category (0-9)
    Bits 20-27: Primary rank (e.g., quad rank, pair rank)
    Bits 12-19: Secondary rank (e.g., full house kicker pair)
    Bits 0-11: Kicker bits (for tiebreakers)
"""

import jax
import jax.numpy as jnp
from jax import Array

from poker_jax.deck import card_to_rank, card_to_suit, NUM_RANKS

# Hand category constants
HIGH_CARD = 0
ONE_PAIR = 1
TWO_PAIR = 2
THREE_OF_A_KIND = 3
STRAIGHT = 4
FLUSH = 5
FULL_HOUSE = 6
FOUR_OF_A_KIND = 7
STRAIGHT_FLUSH = 8
ROYAL_FLUSH = 9


def _encode_hand_value(category: Array, primary: Array, secondary: Array, kickers: Array) -> Array:
    """Encode hand value as a single comparable integer.

    Uses uint32 to avoid overflow for high categories (8=straight flush, 9=royal flush)
    since category 8 << 28 = 2147483648 > max int32.
    """
    return (
        (category.astype(jnp.uint32) << 28)
        | (primary.astype(jnp.uint32) << 20)
        | (secondary.astype(jnp.uint32) << 12)
        | kickers.astype(jnp.uint32)
    )


@jax.jit
def _count_ranks(cards: Array) -> Array:
    """Count occurrences of each rank in the hand.

    Args:
        cards: [7] card indices

    Returns:
        [13] count of each rank
    """
    ranks = card_to_rank(cards)
    # Handle -1 (undealt) cards by clamping
    valid = cards >= 0
    ranks = jnp.where(valid, ranks, 0)

    counts = jnp.zeros(NUM_RANKS, dtype=jnp.int32)
    for i in range(7):
        counts = jnp.where(
            valid[i],
            counts.at[ranks[i]].add(1),
            counts
        )
    return counts


@jax.jit
def _count_suits(cards: Array) -> Array:
    """Count occurrences of each suit in the hand.

    Args:
        cards: [7] card indices

    Returns:
        [4] count of each suit
    """
    suits = card_to_suit(cards)
    valid = cards >= 0
    suits = jnp.where(valid, suits, 0)

    counts = jnp.zeros(4, dtype=jnp.int32)
    for i in range(7):
        counts = jnp.where(
            valid[i],
            counts.at[suits[i]].add(1),
            counts
        )
    return counts


@jax.jit
def _check_straight(rank_counts: Array) -> tuple[Array, Array]:
    """Check for a straight and return its high card.

    Args:
        rank_counts: [13] count of each rank

    Returns:
        Tuple of (is_straight, high_card_rank)
    """
    # Convert counts to binary (has rank or not)
    has_rank = rank_counts > 0

    # Check each possible straight (5 consecutive ranks)
    # Straights: A-5, 2-6, 3-7, ..., T-A
    # Handle wheel (A-2-3-4-5) specially

    # Regular straights: check 5 consecutive
    is_straight = jnp.zeros((), dtype=jnp.bool_)
    high_card = jnp.zeros((), dtype=jnp.int32)

    # Check from highest to lowest
    for high in range(12, 3, -1):  # A down to 5 (high card of straight)
        low = high - 4
        consecutive = jnp.all(has_rank[low:high + 1])
        is_straight = jnp.where(consecutive & ~is_straight, True, is_straight)
        high_card = jnp.where(consecutive & (high_card == 0), high, high_card)

    # Check wheel (A-2-3-4-5) - Ace is rank 12, wheel high is 3 (5)
    wheel = has_rank[12] & has_rank[0] & has_rank[1] & has_rank[2] & has_rank[3]
    is_straight = is_straight | wheel
    high_card = jnp.where(wheel & (high_card == 0), 3, high_card)  # 5-high

    return is_straight, high_card


@jax.jit
def _get_flush_suit(suit_counts: Array) -> tuple[Array, Array]:
    """Find if there's a flush and which suit.

    Args:
        suit_counts: [4] count of each suit

    Returns:
        Tuple of (is_flush, flush_suit)
    """
    is_flush = jnp.any(suit_counts >= 5)
    flush_suit = jnp.argmax(suit_counts)  # Will be the flush suit if exists
    return is_flush, flush_suit


@jax.jit
def _get_flush_ranks(cards: Array, flush_suit: Array) -> Array:
    """Get rank counts for cards of flush suit only.

    Args:
        cards: [7] card indices
        flush_suit: The flush suit

    Returns:
        [13] count of each rank (only flush suit cards)
    """
    ranks = card_to_rank(cards)
    suits = card_to_suit(cards)
    valid = (cards >= 0) & (suits == flush_suit)

    counts = jnp.zeros(NUM_RANKS, dtype=jnp.int32)
    for i in range(7):
        counts = jnp.where(
            valid[i],
            counts.at[ranks[i]].add(1),
            counts
        )
    return counts


@jax.jit
def _get_kicker_bits(rank_counts: Array, exclude_ranks: Array, num_kickers: int) -> Array:
    """Get kicker bits from remaining cards.

    Args:
        rank_counts: [13] count of each rank
        exclude_ranks: [13] bool mask of ranks to exclude
        num_kickers: Number of kickers to include

    Returns:
        Kicker bits (each kicker is 3 bits)
    """
    # Get available ranks
    available = (rank_counts > 0) & ~exclude_ranks

    # Take highest num_kickers ranks
    bits = jnp.zeros((), dtype=jnp.int32)
    count = 0
    for rank in range(12, -1, -1):  # High to low
        should_add = available[rank] & (count < num_kickers)
        bits = jnp.where(should_add, bits | (1 << rank), bits)
        count = jnp.where(should_add, count + 1, count)

    return bits


@jax.jit
def evaluate_hand(cards: Array) -> Array:
    """Evaluate a 7-card poker hand.

    Args:
        cards: [7] card indices (can include -1 for undealt)

    Returns:
        Hand value as a single comparable integer (higher = better)
    """
    rank_counts = _count_ranks(cards)
    suit_counts = _count_suits(cards)

    # Count pairs, trips, quads
    num_quads = jnp.sum(rank_counts == 4)
    num_trips = jnp.sum(rank_counts == 3)
    num_pairs = jnp.sum(rank_counts == 2)

    # Find specific ranks
    quad_rank = jnp.argmax(jnp.where(rank_counts == 4, jnp.arange(13), -1))
    trip_rank = jnp.argmax(jnp.where(rank_counts == 3, jnp.arange(13), -1))

    # For two trips case, also need second-highest trip
    trips_mask = rank_counts == 3
    second_trip_ranks = jnp.where(trips_mask & (jnp.arange(13) != trip_rank), jnp.arange(13), -1)
    second_trip_rank = jnp.max(second_trip_ranks)

    # For pairs, need highest and second highest
    pair_mask = rank_counts == 2
    pair_ranks = jnp.where(pair_mask, jnp.arange(13), -1)
    high_pair = jnp.max(pair_ranks)
    # Second pair: max of pairs excluding high pair
    second_pair_ranks = jnp.where(pair_mask & (jnp.arange(13) != high_pair), jnp.arange(13), -1)
    low_pair = jnp.max(second_pair_ranks)

    # Check for straights and flushes
    is_straight, straight_high = _check_straight(rank_counts)
    is_flush, flush_suit = _get_flush_suit(suit_counts)

    # Check for straight flush
    flush_rank_counts = _get_flush_ranks(cards, flush_suit)
    is_sf, sf_high = _check_straight(flush_rank_counts)
    is_straight_flush = is_flush & is_sf
    is_royal_flush = is_straight_flush & (sf_high == 12)  # A-high straight flush

    # Build hand value based on category
    # Check from highest to lowest

    # Default: high card
    exclude = jnp.zeros(13, dtype=jnp.bool_)
    kickers = _get_kicker_bits(rank_counts, exclude, 5)
    high_card_rank = jnp.argmax(jnp.where(rank_counts > 0, jnp.arange(13), -1))
    value = _encode_hand_value(
        jnp.array(HIGH_CARD), high_card_rank, jnp.array(0), kickers
    )

    # One pair
    is_one_pair = (num_pairs >= 1) & (num_trips == 0) & (num_quads == 0)
    pair_exclude = jnp.arange(13) == high_pair
    pair_kickers = _get_kicker_bits(rank_counts, pair_exclude, 3)
    pair_value = _encode_hand_value(
        jnp.array(ONE_PAIR), high_pair, jnp.array(0), pair_kickers
    )
    value = jnp.where(is_one_pair, pair_value, value)

    # Two pair
    is_two_pair = (num_pairs >= 2) & (num_quads == 0)
    two_pair_exclude = (jnp.arange(13) == high_pair) | (jnp.arange(13) == low_pair)
    two_pair_kickers = _get_kicker_bits(rank_counts, two_pair_exclude, 1)
    two_pair_value = _encode_hand_value(
        jnp.array(TWO_PAIR), high_pair, low_pair, two_pair_kickers
    )
    value = jnp.where(is_two_pair, two_pair_value, value)

    # Three of a kind
    is_three = (num_trips >= 1) & (num_quads == 0) & ~((num_trips >= 1) & (num_pairs >= 1))
    trips_exclude = jnp.arange(13) == trip_rank
    trips_kickers = _get_kicker_bits(rank_counts, trips_exclude, 2)
    trips_value = _encode_hand_value(
        jnp.array(THREE_OF_A_KIND), trip_rank, jnp.array(0), trips_kickers
    )
    value = jnp.where(is_three, trips_value, value)

    # Straight
    straight_value = _encode_hand_value(
        jnp.array(STRAIGHT), straight_high, jnp.array(0), jnp.array(0)
    )
    value = jnp.where(is_straight & ~is_flush, straight_value, value)

    # Flush
    flush_kickers = _get_kicker_bits(flush_rank_counts, jnp.zeros(13, dtype=jnp.bool_), 5)
    flush_high = jnp.argmax(jnp.where(flush_rank_counts > 0, jnp.arange(13), -1))
    flush_value = _encode_hand_value(
        jnp.array(FLUSH), flush_high, jnp.array(0), flush_kickers
    )
    value = jnp.where(is_flush & ~is_straight_flush, flush_value, value)

    # Full house
    # Can also occur with two trips (use higher as trips, lower as pair)
    is_full_house = ((num_trips >= 1) & (num_pairs >= 1)) | (num_trips >= 2)
    # If two trips, higher is the trips, lower is the pair
    fh_trips = jnp.where(num_trips >= 2, trip_rank, trip_rank)
    fh_pair = jnp.where(
        num_trips >= 2,
        second_trip_rank,  # Use second trip as pair when we have two trips
        jnp.maximum(high_pair, low_pair)
    )
    full_house_value = _encode_hand_value(
        jnp.array(FULL_HOUSE), fh_trips, fh_pair, jnp.array(0)
    )
    value = jnp.where(is_full_house, full_house_value, value)

    # Four of a kind
    is_quads = num_quads >= 1
    quads_exclude = jnp.arange(13) == quad_rank
    quads_kickers = _get_kicker_bits(rank_counts, quads_exclude, 1)
    quads_value = _encode_hand_value(
        jnp.array(FOUR_OF_A_KIND), quad_rank, jnp.array(0), quads_kickers
    )
    value = jnp.where(is_quads, quads_value, value)

    # Straight flush
    sf_value = _encode_hand_value(
        jnp.array(STRAIGHT_FLUSH), sf_high, jnp.array(0), jnp.array(0)
    )
    value = jnp.where(is_straight_flush & ~is_royal_flush, sf_value, value)

    # Royal flush
    rf_value = _encode_hand_value(
        jnp.array(ROYAL_FLUSH), jnp.array(12), jnp.array(0), jnp.array(0)
    )
    value = jnp.where(is_royal_flush, rf_value, value)

    return value


@jax.jit
def evaluate_hands_batch(hole_cards: Array, community: Array) -> Array:
    """Evaluate hands for multiple games.

    Args:
        hole_cards: [N, 2, 2] hole cards (games, players, cards)
        community: [N, 5] community cards

    Returns:
        [N, 2] hand values for each player in each game
    """
    n_games = hole_cards.shape[0]

    def eval_one_game(hole, comm):
        # Combine hole cards with community for each player
        p0_cards = jnp.concatenate([hole[0], comm])  # [7]
        p1_cards = jnp.concatenate([hole[1], comm])  # [7]
        return jnp.array([evaluate_hand(p0_cards), evaluate_hand(p1_cards)])

    return jax.vmap(eval_one_game)(hole_cards, community)


@jax.jit
def determine_winner(hand_values: Array) -> Array:
    """Determine winner from hand values.

    Args:
        hand_values: [N, 2] hand values for each player

    Returns:
        [N] winner index (0, 1, or -1 for tie)
    """
    p0_wins = hand_values[:, 0] > hand_values[:, 1]
    p1_wins = hand_values[:, 1] > hand_values[:, 0]
    tie = hand_values[:, 0] == hand_values[:, 1]

    winner = jnp.where(p0_wins, 0, jnp.where(p1_wins, 1, -1))
    return winner


@jax.jit
def get_hand_category(hand_value: Array) -> Array:
    """Extract hand category (0-9) from encoded hand value.

    Args:
        hand_value: Encoded hand value from evaluate_hand()

    Returns:
        Hand category: 0=high card, 1=pair, ..., 9=royal flush
    """
    return (hand_value.astype(jnp.uint32) >> 28) & 0xF


@jax.jit
def has_flush_draw(cards: Array) -> Array:
    """Check if hand has a flush draw (4 cards of same suit).

    Args:
        cards: [7] card indices

    Returns:
        Boolean indicating flush draw
    """
    suit_counts = _count_suits(cards)
    return jnp.max(suit_counts) == 4


@jax.jit
def has_backdoor_flush_draw(cards: Array) -> Array:
    """Check if hand has a backdoor flush draw (3 cards of same suit).

    Args:
        cards: [7] card indices

    Returns:
        Boolean indicating backdoor flush draw
    """
    suit_counts = _count_suits(cards)
    return jnp.max(suit_counts) == 3


@jax.jit
def count_straight_outs(cards: Array) -> Array:
    """Count outs to make a straight.

    Args:
        cards: [7] card indices

    Returns:
        Number of straight outs (0, 4 for gutshot, 8 for OESD)
    """
    rank_counts = _count_ranks(cards)
    has_rank = rank_counts > 0

    # Check for open-ended straight draw (4 consecutive + gaps on both ends)
    # OESD patterns: e.g., 5-6-7-8 with 4 and 9 available = 8 outs
    is_oesd = jnp.zeros((), dtype=jnp.bool_)
    is_gutshot = jnp.zeros((), dtype=jnp.bool_)

    # Check all 4-card sequences
    for start in range(10):  # Can start from 0 (2) to 9 (J)
        end = start + 4
        if end <= 13:
            # Count how many of these 4 consecutive ranks we have
            consecutive_count = jnp.sum(has_rank[start:end])

            # 4 consecutive = open-ended straight draw
            is_4_consecutive = consecutive_count == 4

            # Check if both ends are open (not at edges)
            can_complete_low = (start > 0) | ((start == 0) & has_rank[12])  # A-low straight
            can_complete_high = end < 13

            oesd_here = is_4_consecutive & can_complete_low & can_complete_high
            is_oesd = is_oesd | oesd_here

            # Gutshot: 4 cards in 5-card span with one gap
            if end < 13:
                span_5 = jnp.sum(has_rank[start:end + 1])
                gutshot_here = (span_5 == 4) & ~is_4_consecutive
                is_gutshot = is_gutshot | gutshot_here

    # Also check A-high gutshots (T-J-Q-K-A missing one)
    broadway = has_rank[8:13]  # T, J, Q, K, A
    broadway_count = jnp.sum(broadway)
    broadway_gutshot = broadway_count == 4
    is_gutshot = is_gutshot | broadway_gutshot

    # OESD = 8 outs, gutshot = 4 outs
    outs = jnp.where(is_oesd, 8, jnp.where(is_gutshot, 4, 0))
    return outs


@jax.jit
def get_board_texture(community: Array) -> tuple[Array, Array, Array]:
    """Analyze board texture.

    Args:
        community: [5] community cards (-1 for undealt)

    Returns:
        Tuple of (is_paired, is_monotone, is_connected)
    """
    valid = community >= 0
    n_cards = jnp.sum(valid)

    # Only analyze if we have cards
    rank_counts = _count_ranks(community)
    suit_counts = _count_suits(community)

    # Board is paired if any rank appears 2+ times
    is_paired = jnp.any(rank_counts >= 2)

    # Board is monotone if 3+ cards of same suit
    is_monotone = jnp.any(suit_counts >= 3)

    # Board is connected if 3+ cards within 4-rank span
    has_rank = rank_counts > 0
    is_connected = jnp.zeros((), dtype=jnp.bool_)
    for start in range(10):  # Check 4-rank spans
        span_count = jnp.sum(has_rank[start:start + 4])
        is_connected = is_connected | (span_count >= 3)

    return is_paired, is_monotone, is_connected


@jax.jit
def compute_hand_strength_features(hole_cards: Array, community: Array) -> Array:
    """Compute hand strength features for neural network input.

    Args:
        hole_cards: [2] hole cards for one player
        community: [5] community cards

    Returns:
        [18] feature vector:
            - [0:10] hand category one-hot
            - [10] normalized hand strength (category / 9)
            - [11] has flush draw
            - [12] has OESD (8 outs)
            - [13] has gutshot (4 outs)
            - [14] has backdoor flush draw
            - [15] board is paired
            - [16] board is monotone
            - [17] board is connected
    """
    # Combine cards
    cards = jnp.concatenate([hole_cards, community])

    # Evaluate hand
    hand_value = evaluate_hand(cards)
    category = get_hand_category(hand_value)

    # Hand category one-hot
    category_one_hot = jax.nn.one_hot(category, 10)

    # Normalized strength
    normalized_strength = category.astype(jnp.float32) / 9.0

    # Draw detection
    flush_draw = has_flush_draw(cards).astype(jnp.float32)
    straight_outs = count_straight_outs(cards)
    has_oesd = (straight_outs >= 8).astype(jnp.float32)
    has_gutshot = ((straight_outs >= 4) & (straight_outs < 8)).astype(jnp.float32)
    backdoor_flush = has_backdoor_flush_draw(cards).astype(jnp.float32)

    # Board texture
    is_paired, is_monotone, is_connected = get_board_texture(community)

    # Combine all features
    features = jnp.concatenate([
        category_one_hot,                          # 10 dims
        jnp.array([normalized_strength]),          # 1 dim
        jnp.array([flush_draw]),                   # 1 dim
        jnp.array([has_oesd]),                     # 1 dim
        jnp.array([has_gutshot]),                  # 1 dim
        jnp.array([backdoor_flush]),               # 1 dim
        jnp.array([is_paired.astype(jnp.float32)]),     # 1 dim
        jnp.array([is_monotone.astype(jnp.float32)]),   # 1 dim
        jnp.array([is_connected.astype(jnp.float32)]),  # 1 dim
    ])

    return features


@jax.jit
def compute_hand_strength_batch(hole_cards: Array, community: Array) -> Array:
    """Compute hand strength features for batch of games.

    Args:
        hole_cards: [N, 2] hole cards
        community: [N, 5] community cards

    Returns:
        [N, 18] hand strength features
    """
    return jax.vmap(compute_hand_strength_features)(hole_cards, community)


def hand_value_to_string(value: int) -> str:
    """Convert hand value to human-readable string."""
    # Handle both int32 and uint32 representations
    unsigned_val = int(value) & 0xFFFFFFFF
    category = (unsigned_val >> 28) & 0xF
    primary = (unsigned_val >> 20) & 0xFF
    secondary = (unsigned_val >> 12) & 0xFF

    rank_names = ["2", "3", "4", "5", "6", "7", "8", "9", "T", "J", "Q", "K", "A"]
    cat_names = [
        "High Card", "One Pair", "Two Pair", "Three of a Kind",
        "Straight", "Flush", "Full House", "Four of a Kind",
        "Straight Flush", "Royal Flush"
    ]

    cat_name = cat_names[category] if category < len(cat_names) else f"Unknown({category})"
    primary_rank = rank_names[primary] if primary < len(rank_names) else f"?{primary}"

    if category == ONE_PAIR:
        return f"{cat_name} ({primary_rank}s)"
    elif category == TWO_PAIR:
        secondary_rank = rank_names[secondary] if secondary < len(rank_names) else f"?{secondary}"
        return f"{cat_name} ({primary_rank}s and {secondary_rank}s)"
    elif category == THREE_OF_A_KIND:
        return f"{cat_name} ({primary_rank}s)"
    elif category in (STRAIGHT, FLUSH, STRAIGHT_FLUSH):
        return f"{cat_name} ({primary_rank}-high)"
    elif category == FULL_HOUSE:
        secondary_rank = rank_names[secondary] if secondary < len(rank_names) else f"?{secondary}"
        return f"{cat_name} ({primary_rank}s full of {secondary_rank}s)"
    elif category == FOUR_OF_A_KIND:
        return f"{cat_name} ({primary_rank}s)"
    elif category == ROYAL_FLUSH:
        return "Royal Flush"
    else:
        return f"{cat_name} ({primary_rank}-high)"
