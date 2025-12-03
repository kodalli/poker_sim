"""Hand evaluation for Texas Hold'em poker."""

from collections import Counter
from dataclasses import dataclass
from enum import IntEnum
from itertools import combinations
from typing import Sequence

from poker.cards import Card, Rank


class HandRank(IntEnum):
    """Poker hand rankings from lowest to highest."""

    HIGH_CARD = 1
    PAIR = 2
    TWO_PAIR = 3
    THREE_OF_A_KIND = 4
    STRAIGHT = 5
    FLUSH = 6
    FULL_HOUSE = 7
    FOUR_OF_A_KIND = 8
    STRAIGHT_FLUSH = 9

    def __str__(self) -> str:
        names = {
            1: "High Card",
            2: "Pair",
            3: "Two Pair",
            4: "Three of a Kind",
            5: "Straight",
            6: "Flush",
            7: "Full House",
            8: "Four of a Kind",
            9: "Straight Flush",
        }
        return names[self.value]


@dataclass(frozen=True, slots=True)
class EvaluatedHand:
    """Result of evaluating a poker hand."""

    rank: HandRank
    values: tuple[int, ...]  # Primary values for comparison (high to low)
    cards: tuple[Card, ...]  # The 5 cards that make up the hand

    def __lt__(self, other: "EvaluatedHand") -> bool:
        if self.rank != other.rank:
            return self.rank < other.rank
        return self.values < other.values

    def __le__(self, other: "EvaluatedHand") -> bool:
        return self == other or self < other

    def __gt__(self, other: "EvaluatedHand") -> bool:
        return other < self

    def __ge__(self, other: "EvaluatedHand") -> bool:
        return self == other or self > other

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, EvaluatedHand):
            return NotImplemented
        return self.rank == other.rank and self.values == other.values

    def __str__(self) -> str:
        cards_str = " ".join(str(c) for c in self.cards)
        return f"{self.rank}: {cards_str}"


class HandEvaluator:
    """Evaluate poker hands."""

    @staticmethod
    def evaluate_five(cards: Sequence[Card]) -> EvaluatedHand:
        """Evaluate exactly 5 cards."""
        if len(cards) != 5:
            raise ValueError(f"Expected 5 cards, got {len(cards)}")

        cards = tuple(sorted(cards, key=lambda c: c.rank, reverse=True))
        ranks = [c.rank for c in cards]
        suits = [c.suit for c in cards]
        rank_counts = Counter(ranks)

        is_flush = len(set(suits)) == 1
        is_straight, straight_high = HandEvaluator._check_straight(ranks)

        # Straight flush (includes royal flush)
        if is_flush and is_straight:
            return EvaluatedHand(
                rank=HandRank.STRAIGHT_FLUSH, values=(straight_high,), cards=cards
            )

        # Four of a kind
        if 4 in rank_counts.values():
            quad_rank = [r for r, c in rank_counts.items() if c == 4][0]
            kicker = [r for r, c in rank_counts.items() if c == 1][0]
            return EvaluatedHand(
                rank=HandRank.FOUR_OF_A_KIND, values=(quad_rank, kicker), cards=cards
            )

        # Full house
        if 3 in rank_counts.values() and 2 in rank_counts.values():
            trips_rank = [r for r, c in rank_counts.items() if c == 3][0]
            pair_rank = [r for r, c in rank_counts.items() if c == 2][0]
            return EvaluatedHand(
                rank=HandRank.FULL_HOUSE, values=(trips_rank, pair_rank), cards=cards
            )

        # Flush
        if is_flush:
            return EvaluatedHand(rank=HandRank.FLUSH, values=tuple(ranks), cards=cards)

        # Straight
        if is_straight:
            return EvaluatedHand(
                rank=HandRank.STRAIGHT, values=(straight_high,), cards=cards
            )

        # Three of a kind
        if 3 in rank_counts.values():
            trips_rank = [r for r, c in rank_counts.items() if c == 3][0]
            kickers = sorted([r for r, c in rank_counts.items() if c == 1], reverse=True)
            return EvaluatedHand(
                rank=HandRank.THREE_OF_A_KIND,
                values=(trips_rank, *kickers),
                cards=cards,
            )

        # Two pair
        pairs = [r for r, c in rank_counts.items() if c == 2]
        if len(pairs) == 2:
            pairs_sorted = tuple(sorted(pairs, reverse=True))
            kicker = [r for r, c in rank_counts.items() if c == 1][0]
            return EvaluatedHand(
                rank=HandRank.TWO_PAIR, values=(*pairs_sorted, kicker), cards=cards
            )

        # One pair
        if len(pairs) == 1:
            pair_rank = pairs[0]
            kickers = tuple(sorted([r for r, c in rank_counts.items() if c == 1], reverse=True))
            return EvaluatedHand(
                rank=HandRank.PAIR, values=(pair_rank, *kickers), cards=cards
            )

        # High card
        return EvaluatedHand(rank=HandRank.HIGH_CARD, values=tuple(ranks), cards=cards)

    @staticmethod
    def _check_straight(ranks: list[Rank]) -> tuple[bool, int]:
        """Check if ranks form a straight. Returns (is_straight, high_card)."""
        unique_ranks = sorted(set(ranks), reverse=True)
        if len(unique_ranks) < 5:
            return False, 0

        # Check regular straight
        for i in range(len(unique_ranks) - 4):
            window = unique_ranks[i : i + 5]
            if window[0] - window[4] == 4:
                return True, window[0]

        # Check wheel (A-2-3-4-5)
        wheel = {Rank.ACE, Rank.TWO, Rank.THREE, Rank.FOUR, Rank.FIVE}
        if wheel.issubset(set(ranks)):
            return True, 5  # 5-high straight

        return False, 0

    @staticmethod
    def evaluate_seven(cards: Sequence[Card]) -> EvaluatedHand:
        """Evaluate best 5-card hand from 7 cards."""
        if len(cards) != 7:
            raise ValueError(f"Expected 7 cards, got {len(cards)}")

        best_hand: EvaluatedHand | None = None
        for combo in combinations(cards, 5):
            hand = HandEvaluator.evaluate_five(combo)
            if best_hand is None or hand > best_hand:
                best_hand = hand

        assert best_hand is not None
        return best_hand

    @staticmethod
    def evaluate(hole_cards: tuple[Card, Card], community: Sequence[Card]) -> EvaluatedHand:
        """Evaluate a player's best hand from hole cards + community cards."""
        all_cards = list(hole_cards) + list(community)
        if len(all_cards) < 5:
            raise ValueError(f"Need at least 5 cards, got {len(all_cards)}")
        if len(all_cards) == 5:
            return HandEvaluator.evaluate_five(all_cards)
        if len(all_cards) == 6:
            # Evaluate all 6 choose 5 = 6 combinations
            best: EvaluatedHand | None = None
            for combo in combinations(all_cards, 5):
                hand = HandEvaluator.evaluate_five(combo)
                if best is None or hand > best:
                    best = hand
            assert best is not None
            return best
        return HandEvaluator.evaluate_seven(all_cards)

    @staticmethod
    def compare_hands(hands: list[EvaluatedHand]) -> list[int]:
        """Compare hands and return indices of winners (handles ties)."""
        if not hands:
            return []

        best_hand = max(hands)
        return [i for i, h in enumerate(hands) if h == best_hand]
