"""Tests for poker hand evaluation (poker_jax/hands.py)."""

import pytest
import jax.numpy as jnp

from poker_jax.hands import (
    evaluate_hand,
    evaluate_hands_batch,
    determine_winner,
    hand_value_to_string,
    HIGH_CARD,
    ONE_PAIR,
    TWO_PAIR,
    THREE_OF_A_KIND,
    STRAIGHT,
    FLUSH,
    FULL_HOUSE,
    FOUR_OF_A_KIND,
    STRAIGHT_FLUSH,
    ROYAL_FLUSH,
)
from tests.helpers.card_utils import make_cards_from_strings


def get_category(value: int) -> int:
    """Extract category from hand value.

    Note: Hand values are stored as int32, but high categories (8,9)
    result in values > 2^31-1, causing negative representation.
    We use unsigned interpretation to extract correctly.
    """
    # Convert to unsigned 32-bit, then extract
    unsigned_val = int(value) & 0xFFFFFFFF
    return (unsigned_val >> 28) & 0xF


class TestHandCategories:
    """Test each hand category is correctly identified."""

    @pytest.mark.parametrize(
        "cards,expected_category",
        [
            # High Card
            (["2c", "5d", "7h", "9s", "Jc", "3d", "Kh"], HIGH_CARD),
            (["2c", "4d", "6h", "8s", "Tc", "Qd", "Ah"], HIGH_CARD),
            # One Pair
            (["2c", "2d", "5h", "7s", "9c", "Jd", "Kh"], ONE_PAIR),
            (["Ac", "Ad", "2h", "3s", "5c", "7d", "9h"], ONE_PAIR),
            # Two Pair
            (["2c", "2d", "5h", "5s", "9c", "Jd", "Kh"], TWO_PAIR),
            (["Ac", "Ad", "Kh", "Ks", "2c", "3d", "4h"], TWO_PAIR),
            # Three of a Kind
            (["5c", "5d", "5h", "2s", "7c", "9d", "Jh"], THREE_OF_A_KIND),
            (["Ac", "Ad", "Ah", "2s", "3c", "4d", "6h"], THREE_OF_A_KIND),
            # Straight
            (["5c", "6d", "7h", "8s", "9c", "2d", "3h"], STRAIGHT),
            (["Tc", "Jd", "Qh", "Ks", "Ac", "2d", "3h"], STRAIGHT),  # Broadway
            (["Ac", "2d", "3h", "4s", "5c", "8d", "Kh"], STRAIGHT),  # Wheel
            # Flush
            (["2c", "5c", "7c", "9c", "Jc", "3d", "Kh"], FLUSH),
            (["As", "Ks", "Qs", "Js", "2s", "3d", "4h"], FLUSH),
            # Full House
            (["5c", "5d", "5h", "2s", "2c", "9d", "Jh"], FULL_HOUSE),
            (["Ac", "Ad", "Ah", "Ks", "Kc", "2d", "3h"], FULL_HOUSE),
            # Four of a Kind
            (["5c", "5d", "5h", "5s", "2c", "9d", "Jh"], FOUR_OF_A_KIND),
            (["Ac", "Ad", "Ah", "As", "2c", "3d", "4h"], FOUR_OF_A_KIND),
            # Straight Flush
            (["5c", "6c", "7c", "8c", "9c", "2d", "Kh"], STRAIGHT_FLUSH),
            (["Ac", "2c", "3c", "4c", "5c", "8d", "Kh"], STRAIGHT_FLUSH),  # Wheel SF
            # Royal Flush
            (["Tc", "Jc", "Qc", "Kc", "Ac", "2d", "3h"], ROYAL_FLUSH),
            (["Th", "Jh", "Qh", "Kh", "Ah", "2d", "3c"], ROYAL_FLUSH),
        ],
    )
    def test_hand_category_identification(self, cards, expected_category):
        """Test that each hand category is correctly identified."""
        card_array = make_cards_from_strings(cards)
        value = evaluate_hand(card_array)
        category = get_category(value)
        assert category == expected_category, f"Expected {expected_category}, got {category}"


class TestHandRanking:
    """Test that hands are ranked correctly relative to each other."""

    @pytest.mark.parametrize(
        "better_cards,worse_cards",
        [
            # Royal flush beats straight flush
            (
                ["Tc", "Jc", "Qc", "Kc", "Ac", "2d", "3h"],
                ["9c", "Tc", "Jc", "Qc", "Kc", "2d", "3h"],
            ),
            # Straight flush beats four of a kind
            (
                ["5c", "6c", "7c", "8c", "9c", "2d", "3h"],
                ["Ac", "Ad", "Ah", "As", "Kc", "2d", "3h"],
            ),
            # Four of a kind beats full house
            (
                ["5c", "5d", "5h", "5s", "2c", "3d", "4h"],
                ["Ac", "Ad", "Ah", "Ks", "Kc", "2d", "3h"],
            ),
            # Full house beats flush
            (
                ["5c", "5d", "5h", "2s", "2c", "9d", "Jh"],
                ["2c", "5c", "7c", "9c", "Jc", "3d", "Kh"],
            ),
            # Flush beats straight
            (
                ["2c", "5c", "7c", "9c", "Jc", "3d", "Kh"],
                ["5c", "6d", "7h", "8s", "9c", "2d", "3h"],
            ),
            # Straight beats three of a kind
            (
                ["5c", "6d", "7h", "8s", "9c", "2d", "3h"],
                ["Ac", "Ad", "Ah", "2s", "3c", "4d", "6h"],
            ),
            # Three of a kind beats two pair
            (
                ["5c", "5d", "5h", "2s", "7c", "9d", "Jh"],
                ["Ac", "Ad", "Kh", "Ks", "2c", "3d", "4h"],
            ),
            # Two pair beats one pair
            (
                ["2c", "2d", "5h", "5s", "9c", "Jd", "Kh"],
                ["Ac", "Ad", "2h", "3s", "5c", "7d", "9h"],
            ),
            # One pair beats high card
            (
                ["2c", "2d", "5h", "7s", "9c", "Jd", "Kh"],
                ["2c", "5d", "7h", "9s", "Jc", "Kd", "3h"],
            ),
        ],
    )
    def test_category_ordering(self, better_cards, worse_cards):
        """Test that higher categories beat lower ones."""
        better = evaluate_hand(make_cards_from_strings(better_cards))
        worse = evaluate_hand(make_cards_from_strings(worse_cards))
        assert better > worse


class TestKickers:
    """Test tiebreaker handling with kickers."""

    @pytest.mark.parametrize(
        "better_cards,worse_cards,description",
        [
            # Higher pair wins
            (
                ["Ac", "Ad", "2h", "3s", "5c", "7d", "9h"],
                ["Kc", "Kd", "2h", "3s", "5c", "7d", "9h"],
                "Higher pair wins",
            ),
            # Same pair, higher kicker wins
            (
                ["Ac", "Ad", "Kh", "3s", "5c", "7d", "9h"],
                ["Ac", "Ad", "Qh", "3s", "5c", "7d", "9h"],
                "Same pair, higher kicker",
            ),
            # Two pair: higher top pair wins
            (
                ["Ac", "Ad", "2h", "2s", "5c", "7d", "9h"],
                ["Kc", "Kd", "Qh", "Qs", "5c", "7d", "9h"],
                "Two pair, higher top pair",
            ),
            # Two pair: same top pair, higher second pair wins
            (
                ["Ac", "Ad", "Kh", "Ks", "5c", "7d", "9h"],
                ["Ac", "Ad", "Qh", "Qs", "5c", "7d", "9h"],
                "Two pair, same top, higher second",
            ),
            # Trips: higher trips wins
            (
                ["Ac", "Ad", "Ah", "2s", "5c", "7d", "9h"],
                ["Kc", "Kd", "Kh", "2s", "5c", "7d", "9h"],
                "Higher trips",
            ),
            # Straight: higher straight wins
            (
                ["Tc", "Jd", "Qh", "Ks", "Ac", "2d", "3h"],
                ["9c", "Td", "Jh", "Qs", "Kc", "2d", "3h"],
                "Higher straight",
            ),
            # Full house: higher trips wins
            (
                ["Ac", "Ad", "Ah", "2s", "2c", "7d", "9h"],
                ["Kc", "Kd", "Kh", "As", "Ac", "7d", "9h"],
                "Full house, higher trips",
            ),
            # Quads: higher quads win
            (
                ["Ac", "Ad", "Ah", "As", "2c", "3d", "4h"],
                ["Kc", "Kd", "Kh", "Ks", "Ac", "3d", "4h"],
                "Higher quads",
            ),
            # Straight flush: higher wins
            (
                ["9h", "Th", "Jh", "Qh", "Kh", "2d", "3c"],
                ["8h", "9h", "Th", "Jh", "Qh", "2d", "3c"],
                "Higher straight flush",
            ),
        ],
    )
    def test_kicker_comparison(self, better_cards, worse_cards, description):
        """Test kicker-based tiebreakers."""
        better = evaluate_hand(make_cards_from_strings(better_cards))
        worse = evaluate_hand(make_cards_from_strings(worse_cards))
        assert better > worse, f"Failed: {description}"


class TestEdgeCases:
    """Test edge cases in hand evaluation."""

    def test_wheel_straight(self):
        """A-2-3-4-5 is the lowest straight (5-high, not ace-high)."""
        wheel = make_cards_from_strings(["Ac", "2d", "3h", "4s", "5c", "8d", "Kh"])
        six_high = make_cards_from_strings(["2c", "3d", "4h", "5s", "6c", "9d", "Kh"])

        wheel_val = evaluate_hand(wheel)
        six_val = evaluate_hand(six_high)

        # Both are straights
        assert get_category(wheel_val) == STRAIGHT
        assert get_category(six_val) == STRAIGHT
        # 6-high beats wheel
        assert six_val > wheel_val

    def test_ace_high_straight(self):
        """T-J-Q-K-A is the highest straight (Broadway)."""
        broadway = make_cards_from_strings(["Tc", "Jd", "Qh", "Ks", "Ac", "2d", "3h"])
        king_high = make_cards_from_strings(["9c", "Td", "Jh", "Qs", "Kc", "2d", "3h"])

        broadway_val = evaluate_hand(broadway)
        king_val = evaluate_hand(king_high)

        assert get_category(broadway_val) == STRAIGHT
        assert broadway_val > king_val

    def test_two_trips_makes_full_house(self):
        """Two sets of trips uses higher as trips, lower as pair."""
        two_trips = make_cards_from_strings(["Ac", "Ad", "Ah", "Ks", "Kc", "Kd", "2h"])
        value = evaluate_hand(two_trips)

        # Should be full house, not trips
        assert get_category(value) == FULL_HOUSE
        # Primary should be Aces (rank 12)
        primary = (int(value) >> 20) & 0xFF
        assert primary == 12  # Ace

    def test_three_pairs_uses_best_two(self):
        """Three pairs picks the two highest."""
        three_pairs = make_cards_from_strings(
            ["Ac", "Ad", "Kh", "Ks", "Qc", "Qd", "2h"]
        )
        value = evaluate_hand(three_pairs)

        assert get_category(value) == TWO_PAIR
        # Should be AA KK, not AA QQ or KK QQ
        primary = (int(value) >> 20) & 0xFF
        secondary = (int(value) >> 12) & 0xFF
        assert primary == 12  # Aces
        assert secondary == 11  # Kings

    def test_wheel_straight_flush(self):
        """A-2-3-4-5 suited is a straight flush (steel wheel)."""
        steel_wheel = make_cards_from_strings(["Ac", "2c", "3c", "4c", "5c", "8d", "Kh"])
        value = evaluate_hand(steel_wheel)

        assert get_category(value) == STRAIGHT_FLUSH
        # High card of wheel is 5 (rank 3)
        primary = (int(value) >> 20) & 0xFF
        assert primary == 3  # 5

    def test_ties_are_equal(self):
        """Identical hands should have equal values."""
        hand1 = make_cards_from_strings(["Ac", "Ad", "Kh", "Qs", "Jc", "2d", "3h"])
        hand2 = make_cards_from_strings(["As", "Ah", "Kd", "Qc", "Js", "2h", "3c"])

        assert evaluate_hand(hand1) == evaluate_hand(hand2)

    def test_flush_beats_straight_when_both_possible(self):
        """When hand has both straight and flush, flush wins (unless straight flush)."""
        # 6c 7c 8c 9c Ts - this is a straight and almost a flush
        # Add Qc to make it a flush but not straight flush
        cards = make_cards_from_strings(["6c", "7c", "8c", "9c", "Qc", "Ts", "2d"])
        value = evaluate_hand(cards)

        # Should be flush, not straight
        assert get_category(value) == FLUSH


class TestBatchEvaluation:
    """Test vectorized batch evaluation."""

    def test_evaluate_hands_batch_shape(self, batch_game_state):
        """Test batch evaluation returns correct shape."""
        values = evaluate_hands_batch(
            batch_game_state.hole_cards, batch_game_state.community
        )
        assert values.shape == (100, 2)  # N games, 2 players

    def test_determine_winner_batch(self):
        """Test winner determination for multiple games."""
        # Create hand values where player 0 wins, player 1 wins, and tie
        hand_values = jnp.array(
            [
                [1000, 500],  # P0 wins
                [500, 1000],  # P1 wins
                [750, 750],  # Tie
            ]
        )

        winners = determine_winner(hand_values)
        assert jnp.array_equal(winners, jnp.array([0, 1, -1]))

    def test_batch_consistency(self, rng_key):
        """Test that batch evaluation matches individual evaluation."""
        import jax.random as jrandom
        from poker_jax import reset

        state = reset(rng_key, n_games=10)

        # Batch evaluation
        batch_values = evaluate_hands_batch(state.hole_cards, state.community)

        # Individual evaluation
        for i in range(10):
            p0_cards = jnp.concatenate([state.hole_cards[i, 0], state.community[i]])
            p1_cards = jnp.concatenate([state.hole_cards[i, 1], state.community[i]])

            p0_val = evaluate_hand(p0_cards)
            p1_val = evaluate_hand(p1_cards)

            assert batch_values[i, 0] == p0_val
            assert batch_values[i, 1] == p1_val


class TestHandValueToString:
    """Test human-readable hand descriptions."""

    def test_high_card_string(self):
        """Test high card description."""
        cards = make_cards_from_strings(["2c", "5d", "7h", "9s", "Jc", "3d", "Kh"])
        value = int(evaluate_hand(cards))
        result = hand_value_to_string(value)
        assert "High Card" in result

    def test_pair_string(self):
        """Test pair description."""
        cards = make_cards_from_strings(["Ac", "Ad", "2h", "3s", "5c", "7d", "9h"])
        value = int(evaluate_hand(cards))
        result = hand_value_to_string(value)
        assert "Pair" in result
        assert "A" in result

    def test_full_house_string(self):
        """Test full house description."""
        cards = make_cards_from_strings(["Ac", "Ad", "Ah", "Ks", "Kc", "2d", "3h"])
        value = int(evaluate_hand(cards))
        result = hand_value_to_string(value)
        assert "Full House" in result


class TestAllHandCategories:
    """Comprehensive test for all 10 hand categories with varied examples."""

    @pytest.mark.parametrize(
        "cards",
        [
            ["2c", "4d", "6h", "8s", "Tc", "Qd", "Ah"],  # High card A
            ["3c", "5d", "7h", "9s", "Jc", "Kd", "2h"],  # High card K
        ],
    )
    def test_high_card_variations(self, cards):
        """Test various high card hands."""
        value = evaluate_hand(make_cards_from_strings(cards))
        assert get_category(value) == HIGH_CARD

    @pytest.mark.parametrize(
        "cards,pair_rank",
        [
            (["2c", "2d", "4h", "6s", "8c", "Td", "Qh"], 0),  # Pair of 2s
            (["5c", "5d", "2h", "4s", "7c", "9d", "Jh"], 3),  # Pair of 5s
            (["Ac", "Ad", "3h", "5s", "7c", "9d", "Jh"], 12),  # Pair of As
        ],
    )
    def test_pair_variations(self, cards, pair_rank):
        """Test various pair hands."""
        value = evaluate_hand(make_cards_from_strings(cards))
        assert get_category(value) == ONE_PAIR
        primary = (int(value) >> 20) & 0xFF
        assert primary == pair_rank

    @pytest.mark.parametrize(
        "cards",
        [
            ["2c", "2d", "3h", "3s", "5c", "7d", "9h"],  # 22 33
            ["Tc", "Td", "Jh", "Js", "2c", "4d", "6h"],  # TT JJ
            ["Ac", "Ad", "Kh", "Ks", "2c", "4d", "6h"],  # AA KK
        ],
    )
    def test_two_pair_variations(self, cards):
        """Test various two pair hands."""
        value = evaluate_hand(make_cards_from_strings(cards))
        assert get_category(value) == TWO_PAIR

    @pytest.mark.parametrize(
        "cards",
        [
            ["2c", "2d", "2h", "4s", "6c", "8d", "Th"],  # Trip 2s
            ["Kc", "Kd", "Kh", "2s", "4c", "6d", "8h"],  # Trip Ks
        ],
    )
    def test_three_of_kind_variations(self, cards):
        """Test various three of a kind hands."""
        value = evaluate_hand(make_cards_from_strings(cards))
        assert get_category(value) == THREE_OF_A_KIND

    @pytest.mark.parametrize(
        "cards,high_rank",
        [
            (["Ac", "2d", "3h", "4s", "5c", "7d", "9h"], 3),  # Wheel (5-high)
            (["2c", "3d", "4h", "5s", "6c", "8d", "Th"], 4),  # 6-high
            (["6c", "7d", "8h", "9s", "Tc", "2d", "4h"], 8),  # T-high
            (["Tc", "Jd", "Qh", "Ks", "Ac", "2d", "4h"], 12),  # A-high (Broadway)
        ],
    )
    def test_straight_variations(self, cards, high_rank):
        """Test various straight hands."""
        value = evaluate_hand(make_cards_from_strings(cards))
        assert get_category(value) == STRAIGHT
        primary = (int(value) >> 20) & 0xFF
        assert primary == high_rank

    @pytest.mark.parametrize(
        "cards",
        [
            ["2c", "4c", "6c", "8c", "Tc", "3d", "5h"],  # Club flush
            ["3h", "5h", "7h", "9h", "Jh", "2d", "4c"],  # Heart flush
            ["As", "Ks", "Qs", "Js", "2s", "3d", "4h"],  # Spade flush (high)
        ],
    )
    def test_flush_variations(self, cards):
        """Test various flush hands."""
        value = evaluate_hand(make_cards_from_strings(cards))
        assert get_category(value) == FLUSH

    @pytest.mark.parametrize(
        "cards",
        [
            ["2c", "2d", "2h", "3s", "3c", "5d", "7h"],  # 222 33
            ["Ac", "Ad", "Ah", "Ks", "Kc", "2d", "4h"],  # AAA KK
            ["Tc", "Td", "Th", "9s", "9c", "2d", "4h"],  # TTT 99
        ],
    )
    def test_full_house_variations(self, cards):
        """Test various full house hands."""
        value = evaluate_hand(make_cards_from_strings(cards))
        assert get_category(value) == FULL_HOUSE

    @pytest.mark.parametrize(
        "cards",
        [
            ["2c", "2d", "2h", "2s", "4c", "6d", "8h"],  # Quad 2s
            ["Ac", "Ad", "Ah", "As", "2c", "4d", "6h"],  # Quad As
        ],
    )
    def test_four_of_kind_variations(self, cards):
        """Test various four of a kind hands."""
        value = evaluate_hand(make_cards_from_strings(cards))
        assert get_category(value) == FOUR_OF_A_KIND

    @pytest.mark.parametrize(
        "cards,high_rank",
        [
            (["Ac", "2c", "3c", "4c", "5c", "7d", "9h"], 3),  # Steel wheel
            (["5d", "6d", "7d", "8d", "9d", "2c", "4h"], 7),  # 9-high SF
            (["9h", "Th", "Jh", "Qh", "Kh", "2c", "4d"], 11),  # K-high SF
        ],
    )
    def test_straight_flush_variations(self, cards, high_rank):
        """Test various straight flush hands."""
        value = evaluate_hand(make_cards_from_strings(cards))
        assert get_category(value) == STRAIGHT_FLUSH
        primary = (int(value) >> 20) & 0xFF
        assert primary == high_rank

    @pytest.mark.parametrize(
        "cards",
        [
            ["Tc", "Jc", "Qc", "Kc", "Ac", "2d", "4h"],  # Club royal
            ["Ts", "Js", "Qs", "Ks", "As", "2d", "4h"],  # Spade royal
            ["Th", "Jh", "Qh", "Kh", "Ah", "2d", "4c"],  # Heart royal
            ["Td", "Jd", "Qd", "Kd", "Ad", "2c", "4h"],  # Diamond royal
        ],
    )
    def test_royal_flush_all_suits(self, cards):
        """Test royal flush in all four suits."""
        value = evaluate_hand(make_cards_from_strings(cards))
        assert get_category(value) == ROYAL_FLUSH
