"""Equity estimation for reward shaping in RL training.

Uses a hybrid approach:
1. Fast heuristics for base estimation
2. Blend with learned value function as training progresses
"""

from typing import Sequence

import torch

from poker.cards import Card, Rank, Suit
from poker.hand_evaluator import HandEvaluator, HandRank


# Pre-computed preflop equity vs random opponent (169 canonical hands)
# Values from standard poker equity tables
PREFLOP_EQUITY: dict[str, float] = {
    # Pairs (high to low)
    "AA": 0.852, "KK": 0.824, "QQ": 0.799, "JJ": 0.775,
    "TT": 0.750, "99": 0.720, "88": 0.691, "77": 0.663,
    "66": 0.634, "55": 0.605, "44": 0.577, "33": 0.549, "22": 0.502,
    # Suited aces
    "AKs": 0.670, "AQs": 0.660, "AJs": 0.650, "ATs": 0.640,
    "A9s": 0.610, "A8s": 0.600, "A7s": 0.590, "A6s": 0.580,
    "A5s": 0.590, "A4s": 0.580, "A3s": 0.570, "A2s": 0.560,
    # Offsuit aces
    "AKo": 0.650, "AQo": 0.640, "AJo": 0.630, "ATo": 0.610,
    "A9o": 0.580, "A8o": 0.570, "A7o": 0.560, "A6o": 0.550,
    "A5o": 0.560, "A4o": 0.550, "A3o": 0.540, "A2o": 0.530,
    # Suited kings
    "KQs": 0.630, "KJs": 0.620, "KTs": 0.610, "K9s": 0.580,
    "K8s": 0.560, "K7s": 0.550, "K6s": 0.540, "K5s": 0.530,
    "K4s": 0.520, "K3s": 0.510, "K2s": 0.500,
    # Offsuit kings
    "KQo": 0.610, "KJo": 0.600, "KTo": 0.590, "K9o": 0.550,
    "K8o": 0.530, "K7o": 0.520, "K6o": 0.510, "K5o": 0.500,
    "K4o": 0.490, "K3o": 0.480, "K2o": 0.470,
    # Suited queens
    "QJs": 0.600, "QTs": 0.590, "Q9s": 0.560, "Q8s": 0.540,
    "Q7s": 0.520, "Q6s": 0.510, "Q5s": 0.500, "Q4s": 0.490,
    "Q3s": 0.480, "Q2s": 0.470,
    # Offsuit queens
    "QJo": 0.580, "QTo": 0.570, "Q9o": 0.530, "Q8o": 0.510,
    "Q7o": 0.490, "Q6o": 0.480, "Q5o": 0.470, "Q4o": 0.460,
    "Q3o": 0.450, "Q2o": 0.440,
    # Suited jacks
    "JTs": 0.580, "J9s": 0.550, "J8s": 0.530, "J7s": 0.510,
    "J6s": 0.490, "J5s": 0.480, "J4s": 0.470, "J3s": 0.460,
    "J2s": 0.450,
    # Offsuit jacks
    "JTo": 0.560, "J9o": 0.520, "J8o": 0.500, "J7o": 0.470,
    "J6o": 0.450, "J5o": 0.440, "J4o": 0.430, "J3o": 0.420,
    "J2o": 0.410,
    # Suited tens
    "T9s": 0.540, "T8s": 0.520, "T7s": 0.500, "T6s": 0.470,
    "T5s": 0.450, "T4s": 0.440, "T3s": 0.430, "T2s": 0.420,
    # Offsuit tens
    "T9o": 0.510, "T8o": 0.490, "T7o": 0.460, "T6o": 0.430,
    "T5o": 0.410, "T4o": 0.400, "T3o": 0.390, "T2o": 0.380,
    # Suited connectors and others
    "98s": 0.510, "97s": 0.490, "96s": 0.460, "95s": 0.440,
    "94s": 0.420, "93s": 0.400, "92s": 0.390,
    "98o": 0.480, "97o": 0.450, "96o": 0.420, "95o": 0.400,
    "94o": 0.370, "93o": 0.360, "92o": 0.350,
    "87s": 0.490, "86s": 0.470, "85s": 0.440, "84s": 0.410,
    "83s": 0.390, "82s": 0.380,
    "87o": 0.460, "86o": 0.430, "85o": 0.400, "84o": 0.370,
    "83o": 0.350, "82o": 0.340,
    "76s": 0.470, "75s": 0.450, "74s": 0.420, "73s": 0.390,
    "72s": 0.370,
    "76o": 0.440, "75o": 0.410, "74o": 0.380, "73o": 0.350,
    "72o": 0.330,
    "65s": 0.450, "64s": 0.430, "63s": 0.400, "62s": 0.380,
    "65o": 0.420, "64o": 0.390, "63o": 0.360, "62o": 0.340,
    "54s": 0.440, "53s": 0.410, "52s": 0.390,
    "54o": 0.400, "53o": 0.370, "52o": 0.350,
    "43s": 0.400, "42s": 0.370,
    "43o": 0.360, "42o": 0.340,
    "32s": 0.360,
    "32o": 0.320,
}


def _rank_to_char(rank: Rank) -> str:
    """Convert rank to canonical character."""
    return {
        Rank.TWO: "2", Rank.THREE: "3", Rank.FOUR: "4", Rank.FIVE: "5",
        Rank.SIX: "6", Rank.SEVEN: "7", Rank.EIGHT: "8", Rank.NINE: "9",
        Rank.TEN: "T", Rank.JACK: "J", Rank.QUEEN: "Q", Rank.KING: "K",
        Rank.ACE: "A",
    }[rank]


def canonicalize_hand(card1: Card, card2: Card) -> str:
    """Convert two hole cards to canonical string (e.g., 'AKs', 'QQo')."""
    # Order by rank (high first)
    if card1.rank < card2.rank:
        card1, card2 = card2, card1

    r1 = _rank_to_char(card1.rank)
    r2 = _rank_to_char(card2.rank)

    if card1.rank == card2.rank:
        return f"{r1}{r2}"  # Pair
    elif card1.suit == card2.suit:
        return f"{r1}{r2}s"  # Suited
    else:
        return f"{r1}{r2}o"  # Offsuit


def preflop_equity(card1: Card, card2: Card) -> float:
    """Get preflop equity from lookup table."""
    hand = canonicalize_hand(card1, card2)
    return PREFLOP_EQUITY.get(hand, 0.40)  # Default to ~40% for unknown


def _count_flush_draws(hole_cards: tuple[Card, Card], community: Sequence[Card]) -> int:
    """Count cards toward a flush draw."""
    all_cards = list(hole_cards) + list(community)
    suit_counts = {}
    for card in all_cards:
        suit_counts[card.suit] = suit_counts.get(card.suit, 0) + 1
    return max(suit_counts.values()) if suit_counts else 0


def _count_straight_draws(hole_cards: tuple[Card, Card], community: Sequence[Card]) -> int:
    """Count consecutive ranks (rough straight draw indicator)."""
    all_cards = list(hole_cards) + list(community)
    ranks = sorted(set(c.rank for c in all_cards))

    if len(ranks) < 4:
        return 0

    # Count max consecutive ranks
    max_consecutive = 1
    current = 1
    for i in range(1, len(ranks)):
        if ranks[i] - ranks[i - 1] == 1:
            current += 1
            max_consecutive = max(max_consecutive, current)
        elif ranks[i] - ranks[i - 1] > 2:
            current = 1
        # Gap of 2 is still a gutshot potential

    return max_consecutive


def estimate_draw_potential(
    hole_cards: tuple[Card, Card],
    community: Sequence[Card],
) -> float:
    """Estimate potential to improve (0.0 to 0.3 bonus)."""
    if len(community) >= 5:
        return 0.0  # No more cards coming

    bonus = 0.0

    # Flush draw (4 to a flush)
    flush_count = _count_flush_draws(hole_cards, community)
    if flush_count == 4:
        bonus += 0.15  # ~35% to hit flush
    elif flush_count == 3 and len(community) <= 3:
        bonus += 0.05  # Backdoor flush draw

    # Straight draw
    straight_count = _count_straight_draws(hole_cards, community)
    if straight_count >= 4:
        bonus += 0.12  # Open-ended or gutshot
    elif straight_count == 3 and len(community) <= 3:
        bonus += 0.03  # Backdoor straight draw

    return min(bonus, 0.25)  # Cap at 0.25


def heuristic_equity(
    hole_cards: tuple[Card, Card],
    community: Sequence[Card],
) -> float:
    """Fast rule-based hand strength estimation (0.0 to 1.0).

    Args:
        hole_cards: Player's two hole cards
        community: Community cards (0-5 cards)

    Returns:
        Estimated equity between 0.0 and 1.0
    """
    # Preflop: use lookup table
    if len(community) == 0:
        return preflop_equity(hole_cards[0], hole_cards[1])

    # Postflop: evaluate current hand + estimate improvement
    all_cards = list(hole_cards) + list(community)

    if len(all_cards) < 5:
        # Not enough cards to evaluate, use adjusted preflop
        base = preflop_equity(hole_cards[0], hole_cards[1])
        draw_bonus = estimate_draw_potential(hole_cards, community)
        return min(base + draw_bonus, 0.95)

    # Evaluate made hand
    hand = HandEvaluator.evaluate(hole_cards, community)

    # Convert hand rank to base equity
    # Higher ranks = higher equity
    rank_equity = {
        HandRank.HIGH_CARD: 0.20,
        HandRank.PAIR: 0.40,
        HandRank.TWO_PAIR: 0.60,
        HandRank.THREE_OF_A_KIND: 0.70,
        HandRank.STRAIGHT: 0.80,
        HandRank.FLUSH: 0.85,
        HandRank.FULL_HOUSE: 0.90,
        HandRank.FOUR_OF_A_KIND: 0.95,
        HandRank.STRAIGHT_FLUSH: 0.99,
    }

    base_equity = rank_equity.get(hand.rank, 0.30)

    # Adjust within rank based on kickers/values
    # Higher values within a rank = slightly higher equity
    if hand.values:
        # Normalize primary value (2-14) to small bonus
        primary = hand.values[0]
        value_bonus = (primary - 2) / 12 * 0.05  # 0 to 0.05
        base_equity = min(base_equity + value_bonus, 0.99)

    # Add draw potential (less valuable with more community cards)
    if len(community) < 5:
        draw_bonus = estimate_draw_potential(hole_cards, community)
        # Reduce draw value as we get closer to river
        draw_multiplier = (5 - len(community)) / 5
        base_equity = min(base_equity + draw_bonus * draw_multiplier, 0.99)

    return base_equity


def estimate_equity(
    hole_cards: tuple[Card, Card],
    community: Sequence[Card],
    value_network: torch.nn.Module | None = None,
    state_tensor: torch.Tensor | None = None,
    training_progress: float = 0.0,
) -> float:
    """Hybrid equity estimation blending heuristics with learned value.

    Args:
        hole_cards: Player's two hole cards
        community: Community cards (0-5 cards)
        value_network: Optional trained value network for blending
        state_tensor: Encoded state tensor for value network
        training_progress: Training progress (0.0 to 1.0), affects blend ratio

    Returns:
        Estimated equity between 0.0 and 1.0
    """
    # Base: fast heuristic
    base = heuristic_equity(hole_cards, community)

    if value_network is None or state_tensor is None:
        return base

    # Get learned value estimate
    with torch.no_grad():
        value_network.eval()
        # Assume value network has a get_value method or forward returns value
        if hasattr(value_network, "get_value"):
            learned_value = value_network.get_value(state_tensor)
        else:
            # For ActorCriticMLP, forward returns (action_logits, bet_frac, value)
            _, _, learned_value = value_network(state_tensor)

        # Convert value to equity scale (assuming value is normalized reward)
        # Clip to reasonable range
        learned = torch.sigmoid(learned_value).item()

    # Blend: trust learned more as training progresses
    # At 0% training: 100% heuristic
    # At 100% training: 70% learned, 30% heuristic (keep some grounding)
    alpha = 0.7 * training_progress
    blended = (1 - alpha) * base + alpha * learned

    return float(blended)


def hand_strength_category(
    hole_cards: tuple[Card, Card],
    community: Sequence[Card],
) -> str:
    """Get human-readable hand strength category for debugging."""
    equity = heuristic_equity(hole_cards, community)

    if equity >= 0.85:
        return "monster"
    elif equity >= 0.70:
        return "strong"
    elif equity >= 0.50:
        return "medium"
    elif equity >= 0.35:
        return "marginal"
    else:
        return "weak"
