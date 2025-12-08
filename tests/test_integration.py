"""Integration tests for complete poker games."""

import pytest
import jax.numpy as jnp
import jax.random as jrandom

from poker_jax import reset, step, get_rewards
from poker_jax.state import (
    ACTION_FOLD,
    ACTION_CHECK,
    ACTION_CALL,
    ACTION_RAISE,
    ACTION_ALL_IN,
    ROUND_PREFLOP,
    get_valid_actions_mask,
)
from poker_jax.hands import evaluate_hands_batch, determine_winner


class TestCompleteGames:
    """Test complete game scenarios."""

    def test_check_call_to_showdown(self, rng_key):
        """Play a complete game with only checks and calls."""
        state = reset(rng_key, n_games=1)

        action_count = 0
        while not state.done[0]:
            # Call preflop, check postflop
            action = ACTION_CALL if state.round[0] == ROUND_PREFLOP else ACTION_CHECK
            state = step(state, jnp.array([action]))
            action_count += 1

            # Safety check to prevent infinite loop
            assert action_count < 50, "Game took too many actions"

        assert state.done[0]
        assert state.winner[0] in [0, 1, -1]

    def test_all_in_preflop(self, rng_key):
        """Test all-in preflop scenario."""
        state = reset(rng_key, n_games=1, starting_chips=50)

        # SB goes all-in
        state = step(state, jnp.array([ACTION_ALL_IN]))
        assert not state.done[0]

        # BB calls all-in
        state = step(state, jnp.array([ACTION_CALL]))

        # Game should go to showdown immediately
        assert state.done[0]
        assert state.winner[0] in [0, 1, -1]

    def test_raise_fold_sequence(self, rng_key):
        """Test raise followed by fold."""
        state = reset(rng_key, n_games=1, starting_chips=200)

        # SB raises
        state = step(state, jnp.array([ACTION_RAISE]), raise_amounts=jnp.array([10]))
        assert not state.done[0]

        # BB folds
        state = step(state, jnp.array([ACTION_FOLD]))

        assert state.done[0]
        assert state.winner[0] == 0  # SB wins

    def test_raise_reraise_sequence(self, rng_key):
        """Test raise-reraise pattern."""
        state = reset(rng_key, n_games=1, starting_chips=200)

        # SB raises to 6
        state = step(state, jnp.array([ACTION_RAISE]), raise_amounts=jnp.array([6]))
        assert state.bets[0, 0] == 6

        # BB 3-bets to 18
        state = step(state, jnp.array([ACTION_RAISE]), raise_amounts=jnp.array([18]))
        assert state.bets[0, 1] == 18

        # SB 4-bets to 40
        state = step(state, jnp.array([ACTION_RAISE]), raise_amounts=jnp.array([40]))
        assert state.bets[0, 0] == 40

        # Game continues
        assert not state.done[0]

    def test_multi_street_betting(self, rng_key):
        """Test betting across multiple streets."""
        state = reset(rng_key, n_games=1, starting_chips=200)

        # Preflop: SB raises, BB calls
        state = step(state, jnp.array([ACTION_RAISE]), raise_amounts=jnp.array([6]))
        state = step(state, jnp.array([ACTION_CALL]))

        # Flop: both check
        state = step(state, jnp.array([ACTION_CHECK]))
        state = step(state, jnp.array([ACTION_CHECK]))

        # Turn: BB bets, SB calls
        state = step(state, jnp.array([ACTION_RAISE]), raise_amounts=jnp.array([10]))
        state = step(state, jnp.array([ACTION_CALL]))

        # River: both check
        state = step(state, jnp.array([ACTION_CHECK]))
        state = step(state, jnp.array([ACTION_CHECK]))

        assert state.done[0]


class TestBatchGames:
    """Test batched game execution."""

    @pytest.mark.parametrize("n_games", [10, 100, 1000])
    def test_batch_games_complete(self, rng_key, n_games):
        """Test that all games in batch complete."""
        state = reset(rng_key, n_games=n_games)

        # Play all games to completion
        max_steps = 50
        for _ in range(max_steps):
            if jnp.all(state.done):
                break

            actions = jnp.where(
                state.round == ROUND_PREFLOP, ACTION_CALL, ACTION_CHECK
            )
            state = step(state, actions)

        # All games should complete
        assert jnp.all(state.done)

    def test_batch_independence(self, rng_key):
        """Games in batch should be independent."""
        state = reset(rng_key, n_games=100)

        # Play to completion
        while not jnp.all(state.done):
            actions = jnp.where(
                state.round == ROUND_PREFLOP, ACTION_CALL, ACTION_CHECK
            )
            state = step(state, actions)

        # Not all games should have same winner
        winners = state.winner
        assert len(jnp.unique(winners)) > 1  # Multiple different outcomes


class TestPropertyBased:
    """Property-based tests for invariants."""

    @pytest.mark.parametrize("seed", range(20))
    def test_chip_conservation_random_games(self, seed):
        """Chips should always be conserved regardless of action sequence."""
        key = jrandom.PRNGKey(seed)
        state = reset(key, n_games=1, starting_chips=200)
        initial_total = (
            jnp.sum(state.chips) + jnp.sum(state.bets) + state.pot[0]
        )

        # Random actions
        for i in range(30):
            if state.done[0]:
                break

            key, subkey = jrandom.split(key)

            # Get valid actions
            mask = get_valid_actions_mask(state)

            # Choose a random valid action
            valid_actions = jnp.where(mask[0, 1:], jnp.arange(1, 6), 0)
            valid_actions = valid_actions[valid_actions > 0]

            if len(valid_actions) == 0:
                break

            action_idx = jrandom.randint(subkey, (), 0, len(valid_actions))
            action = valid_actions[action_idx]

            state = step(state, jnp.array([action]))

            current_total = (
                jnp.sum(state.chips) + jnp.sum(state.bets) + state.pot[0]
            )
            assert current_total == initial_total

    @pytest.mark.parametrize("seed", range(20))
    def test_game_always_terminates(self, seed):
        """Games should always terminate within reasonable steps."""
        key = jrandom.PRNGKey(seed)
        state = reset(key, n_games=1, starting_chips=200)

        max_steps = 100
        for step_num in range(max_steps):
            if state.done[0]:
                break

            # Always call/check to guarantee termination
            action = ACTION_CALL if state.round[0] == ROUND_PREFLOP else ACTION_CHECK
            state = step(state, jnp.array([action]))

        assert state.done[0], f"Game did not terminate after {max_steps} steps"

    @pytest.mark.parametrize("seed", range(10))
    def test_rewards_always_zero_sum(self, seed):
        """Total rewards should always sum to zero."""
        key = jrandom.PRNGKey(seed)
        state = reset(key, n_games=100)

        # Play to completion
        while not jnp.all(state.done):
            actions = jnp.where(
                state.round == ROUND_PREFLOP, ACTION_CALL, ACTION_CHECK
            )
            state = step(state, actions)

        rewards = get_rewards(state)
        sums = jnp.sum(rewards, axis=1)

        assert jnp.allclose(sums, 0.0, atol=1e-5)


class TestWinnerDetermination:
    """Test winner is correctly determined."""

    def test_better_hand_wins(self, rng_key):
        """Player with better hand should win at showdown."""
        n_games = 1000
        state = reset(rng_key, n_games=n_games)

        # Play to showdown
        while not jnp.all(state.done):
            actions = jnp.where(
                state.round == ROUND_PREFLOP, ACTION_CALL, ACTION_CHECK
            )
            state = step(state, actions)

        # Evaluate hands
        hand_values = evaluate_hands_batch(state.hole_cards, state.community)
        expected_winners = determine_winner(hand_values)

        # Winners should match expected
        assert jnp.array_equal(state.winner, expected_winners)

    def test_ties_handled_correctly(self, rng_key):
        """Ties should result in winner = -1."""
        # Run many games, verify any ties have winner = -1
        n_games = 1000
        state = reset(rng_key, n_games=n_games)

        while not jnp.all(state.done):
            actions = jnp.where(
                state.round == ROUND_PREFLOP, ACTION_CALL, ACTION_CHECK
            )
            state = step(state, actions)

        # Check hand values for ties
        hand_values = evaluate_hands_batch(state.hole_cards, state.community)
        ties = hand_values[:, 0] == hand_values[:, 1]

        # Where hands are equal, winner should be -1
        if jnp.any(ties):
            tie_winners = state.winner[ties]
            assert jnp.all(tie_winners == -1)


class TestEdgeCases:
    """Test edge cases and corner scenarios."""

    def test_heads_up_all_in_first_hand(self, rng_key):
        """Test immediate all-in and call scenario."""
        state = reset(rng_key, n_games=1, starting_chips=100)

        # Immediate all-in and call
        state = step(state, jnp.array([ACTION_ALL_IN]))
        state = step(state, jnp.array([ACTION_CALL]))

        assert state.done[0]
        rewards = get_rewards(state)

        # One player wins ~100, other loses ~100
        assert abs(rewards[0, 0]) > 90  # Roughly pot size
        assert abs(rewards[0, 1]) > 90

    def test_very_small_stacks(self, rng_key):
        """Test with very small stack sizes."""
        state = reset(rng_key, n_games=1, starting_chips=5, small_blind=1, big_blind=2)

        # SB all-in (4 chips)
        state = step(state, jnp.array([ACTION_ALL_IN]))

        # BB calls
        state = step(state, jnp.array([ACTION_CALL]))

        assert state.done[0]

    def test_very_large_batch(self, rng_key):
        """Test with large batch size."""
        n_games = 10000
        state = reset(rng_key, n_games=n_games)

        # Quick game completion
        for _ in range(20):
            if jnp.all(state.done):
                break
            actions = jnp.where(
                state.round == ROUND_PREFLOP, ACTION_CALL, ACTION_CHECK
            )
            state = step(state, actions)

        # Most games should complete
        completed = jnp.sum(state.done)
        assert completed > n_games * 0.99

    def test_repeated_raises(self, rng_key):
        """Test multiple raises in a row."""
        state = reset(rng_key, n_games=1, starting_chips=1000)

        # Multiple raises
        amounts = [6, 18, 50, 120, 250]
        for amount in amounts:
            if state.done[0]:
                break
            state = step(state, jnp.array([ACTION_RAISE]), raise_amounts=jnp.array([amount]))

        # Game should still be valid
        assert not state.done[0] or state.winner[0] in [0, 1, -1]


class TestGameReproducibility:
    """Test that games are reproducible with same seed."""

    def test_same_seed_same_result(self):
        """Same seed should produce identical games."""
        key = jrandom.PRNGKey(12345)

        # First run
        state1 = reset(key, n_games=10)
        while not jnp.all(state1.done):
            actions = jnp.where(
                state1.round == ROUND_PREFLOP, ACTION_CALL, ACTION_CHECK
            )
            state1 = step(state1, actions)

        # Second run with same key
        state2 = reset(key, n_games=10)
        while not jnp.all(state2.done):
            actions = jnp.where(
                state2.round == ROUND_PREFLOP, ACTION_CALL, ACTION_CHECK
            )
            state2 = step(state2, actions)

        # Results should be identical
        assert jnp.array_equal(state1.winner, state2.winner)
        assert jnp.array_equal(state1.hole_cards, state2.hole_cards)
        assert jnp.array_equal(state1.community, state2.community)

    def test_different_seeds_different_results(self):
        """Different seeds should produce different results."""
        state1 = reset(jrandom.PRNGKey(1), n_games=100)
        state2 = reset(jrandom.PRNGKey(2), n_games=100)

        # Hole cards should be different
        assert not jnp.array_equal(state1.hole_cards, state2.hole_cards)
