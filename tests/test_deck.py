"""Tests for card dealing (poker_jax/deck.py)."""

import pytest
import jax.numpy as jnp
import jax.random as jrandom

from poker_jax.deck import (
    shuffle_deck,
    shuffle_decks,
    deal_hole_cards_batch,
    deal_flop_batch,
    deal_turn_batch,
    deal_river_batch,
    card_to_rank,
    card_to_suit,
    make_card,
    card_to_string,
    NUM_CARDS,
    NUM_RANKS,
    NUM_SUITS,
)


class TestCardEncoding:
    """Test card representation functions."""

    @pytest.mark.parametrize(
        "rank,suit,expected_id",
        [
            (0, 0, 0),  # 2c
            (0, 3, 3),  # 2s
            (12, 0, 48),  # Ac
            (12, 3, 51),  # As
            (8, 2, 34),  # Th
            (11, 1, 45),  # Kd
        ],
    )
    def test_make_card(self, rank, suit, expected_id):
        """Test card ID creation."""
        assert make_card(rank, suit) == expected_id

    def test_card_to_rank_all_cards(self):
        """Test rank extraction for all 52 cards."""
        for card_id in range(NUM_CARDS):
            rank = card_to_rank(jnp.array(card_id))
            assert 0 <= rank < NUM_RANKS
            assert rank == card_id // NUM_SUITS

    def test_card_to_suit_all_cards(self):
        """Test suit extraction for all 52 cards."""
        for card_id in range(NUM_CARDS):
            suit = card_to_suit(jnp.array(card_id))
            assert 0 <= suit < NUM_SUITS
            assert suit == card_id % NUM_SUITS

    def test_card_to_string_known_cards(self):
        """Test string representation."""
        assert card_to_string(0) == "2c"
        assert card_to_string(51) == "As"
        assert card_to_string(48) == "Ac"
        assert card_to_string(3) == "2s"
        assert card_to_string(-1) == "??"

    def test_round_trip_encoding(self):
        """Test rank/suit extraction matches make_card."""
        for rank in range(NUM_RANKS):
            for suit in range(NUM_SUITS):
                card_id = make_card(rank, suit)
                assert card_to_rank(jnp.array(card_id)) == rank
                assert card_to_suit(jnp.array(card_id)) == suit


class TestDeckShuffle:
    """Test deck shuffling."""

    def test_shuffle_deck_contains_all_cards(self, rng_key):
        """Shuffled deck should contain all 52 unique cards."""
        deck = shuffle_deck(rng_key)

        assert deck.shape == (NUM_CARDS,)
        assert len(jnp.unique(deck)) == NUM_CARDS
        assert jnp.min(deck) == 0
        assert jnp.max(deck) == NUM_CARDS - 1

    def test_shuffle_deck_different_seeds_different_order(self):
        """Different seeds should produce different shuffles."""
        deck1 = shuffle_deck(jrandom.PRNGKey(1))
        deck2 = shuffle_deck(jrandom.PRNGKey(2))

        # Extremely unlikely to be equal with different seeds
        assert not jnp.array_equal(deck1, deck2)

    def test_shuffle_deck_same_seed_same_order(self):
        """Same seed should produce same shuffle."""
        deck1 = shuffle_deck(jrandom.PRNGKey(42))
        deck2 = shuffle_deck(jrandom.PRNGKey(42))

        assert jnp.array_equal(deck1, deck2)

    def test_shuffle_decks_batch(self, rng_key):
        """Test batch shuffle produces N valid decks."""
        n_games = 100
        keys = jrandom.split(rng_key, n_games)
        decks = shuffle_decks(keys)

        assert decks.shape == (n_games, NUM_CARDS)

        # Each deck should have all unique cards
        for i in range(n_games):
            assert len(jnp.unique(decks[i])) == NUM_CARDS

    def test_shuffle_decks_different_between_games(self, rng_key):
        """Different games should have different shuffles."""
        n_games = 10
        keys = jrandom.split(rng_key, n_games)
        decks = shuffle_decks(keys)

        # Check that decks are different from each other
        for i in range(n_games):
            for j in range(i + 1, n_games):
                assert not jnp.array_equal(decks[i], decks[j])


class TestCardDealing:
    """Test card dealing functions."""

    def test_deal_hole_cards_shape(self, rng_key):
        """Test hole card dealing produces correct shape."""
        n_games = 50
        keys = jrandom.split(rng_key, n_games)
        decks = shuffle_decks(keys)
        deck_indices = jnp.zeros(n_games, dtype=jnp.int32)

        hole_cards, new_indices = deal_hole_cards_batch(decks, deck_indices)

        assert hole_cards.shape == (n_games, 2, 2)  # games, players, cards
        assert new_indices.shape == (n_games,)
        assert jnp.all(new_indices == 4)  # 4 cards dealt

    def test_deal_hole_cards_no_duplicates(self, rng_key):
        """Hole cards should have no duplicates within a game."""
        n_games = 100
        keys = jrandom.split(rng_key, n_games)
        decks = shuffle_decks(keys)
        deck_indices = jnp.zeros(n_games, dtype=jnp.int32)

        hole_cards, _ = deal_hole_cards_batch(decks, deck_indices)

        for i in range(n_games):
            all_holes = hole_cards[i].flatten()
            assert len(jnp.unique(all_holes)) == 4

    def test_deal_flop_adds_three_cards(self, single_game_state):
        """Flop should add exactly 3 community cards."""
        state = single_game_state
        initial_community = state.community.copy()

        # Community starts as [-1, -1, -1, -1, -1]
        assert jnp.all(initial_community == -1)

        new_community, new_idx = deal_flop_batch(
            state.deck, state.deck_idx, state.community
        )

        # First 3 should be dealt (valid card IDs >= 0)
        assert jnp.all(new_community[:, :3] >= 0)
        # Last 2 still undealt
        assert jnp.all(new_community[:, 3:] == -1)

    def test_deal_turn_adds_one_card(self, single_game_state):
        """Turn should add exactly 1 community card."""
        state = single_game_state

        # Deal flop first
        community, deck_idx = deal_flop_batch(
            state.deck, state.deck_idx, state.community
        )

        # Deal turn
        community, deck_idx = deal_turn_batch(state.deck, deck_idx, community)

        # First 4 should be dealt
        assert jnp.all(community[:, :4] >= 0)
        # Last 1 still undealt
        assert jnp.all(community[:, 4] == -1)

    def test_deal_river_adds_final_card(self, single_game_state):
        """River should complete the 5 community cards."""
        state = single_game_state

        # Deal flop, turn, river
        community, deck_idx = deal_flop_batch(
            state.deck, state.deck_idx, state.community
        )
        community, deck_idx = deal_turn_batch(state.deck, deck_idx, community)
        community, deck_idx = deal_river_batch(state.deck, deck_idx, community)

        # All 5 should be dealt
        assert jnp.all(community >= 0)

    def test_no_duplicate_cards_in_game(self, single_game_state):
        """All dealt cards should be unique within a game."""
        state = single_game_state

        # Get all cards from a full game
        community, deck_idx = deal_flop_batch(
            state.deck, state.deck_idx, state.community
        )
        community, deck_idx = deal_turn_batch(state.deck, deck_idx, community)
        community, deck_idx = deal_river_batch(state.deck, deck_idx, community)

        # Collect all cards
        hole_cards = state.hole_cards[0].flatten()  # 4 cards
        comm_cards = community[0]  # 5 cards
        all_cards = jnp.concatenate([hole_cards, comm_cards])

        # Should have 9 unique cards
        assert len(jnp.unique(all_cards)) == 9

    def test_deck_index_advances_correctly(self, single_game_state):
        """Deck index should advance by correct amount each phase."""
        state = single_game_state

        assert state.deck_idx[0] == 4  # After hole cards

        _, idx1 = deal_flop_batch(state.deck, state.deck_idx, state.community)
        assert idx1[0] == 8  # +4 (burn + 3)

        _, idx2 = deal_turn_batch(state.deck, idx1, state.community)
        assert idx2[0] == 10  # +2 (burn + 1)

        _, idx3 = deal_river_batch(state.deck, idx2, state.community)
        assert idx3[0] == 12  # +2 (burn + 1)

    def test_cards_are_valid_ids(self, single_game_state):
        """All dealt cards should have valid IDs (0-51)."""
        state = single_game_state

        # Check hole cards
        assert jnp.all(state.hole_cards >= 0)
        assert jnp.all(state.hole_cards < NUM_CARDS)

        # Deal community
        community, _ = deal_flop_batch(
            state.deck, state.deck_idx, state.community
        )
        community, _ = deal_turn_batch(state.deck, state.deck_idx + 4, community)
        community, _ = deal_river_batch(state.deck, state.deck_idx + 6, community)

        # Check community cards (only dealt ones)
        dealt = community >= 0
        assert jnp.all(community[dealt] < NUM_CARDS)


class TestBatchDealing:
    """Test batch dealing operations."""

    @pytest.mark.parametrize("n_games", [1, 10, 100, 1000])
    def test_batch_sizes(self, rng_key, n_games):
        """Test dealing works for various batch sizes."""
        from poker_jax import reset

        state = reset(rng_key, n_games=n_games)

        assert state.hole_cards.shape == (n_games, 2, 2)
        assert state.community.shape == (n_games, 5)
        assert state.deck_idx.shape == (n_games,)

    def test_batch_independence(self, rng_key):
        """Games in a batch should be independent."""
        from poker_jax import reset

        state = reset(rng_key, n_games=100)

        # Check that hole cards differ between games
        first_game_holes = state.hole_cards[0]
        different_count = 0

        for i in range(1, 100):
            if not jnp.array_equal(state.hole_cards[i], first_game_holes):
                different_count += 1

        # Most games should have different hole cards
        assert different_count > 90  # At least 90% different
