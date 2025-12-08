"""Tests for chip accounting and betting (poker_jax/game.py)."""

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
)


def get_total_chips(state):
    """Calculate total chips in the system (should be constant)."""
    # chips in stacks + current bets + pot
    return jnp.sum(state.chips, axis=1) + jnp.sum(state.bets, axis=1) + state.pot


class TestChipConservation:
    """Test that chips are always conserved."""

    def test_chip_conservation_after_reset(self, rng_key):
        """Total chips should equal 2 * starting_chips after reset."""
        starting = 200
        state = reset(rng_key, n_games=10, starting_chips=starting)

        total = get_total_chips(state)
        expected = 2 * starting  # Two players
        assert jnp.all(total == expected)

    def test_chip_conservation_after_call(self, single_game_state):
        """Calling should preserve total chips."""
        initial_total = get_total_chips(single_game_state)

        # Player 0 (SB) calls the BB
        state = step(single_game_state, jnp.array([ACTION_CALL]))

        assert jnp.all(get_total_chips(state) == initial_total)

    def test_chip_conservation_after_raise(self, single_game_state):
        """Raising should preserve total chips."""
        initial_total = get_total_chips(single_game_state)

        # Min raise preflop (to 4)
        state = step(
            single_game_state, jnp.array([ACTION_RAISE]), raise_amounts=jnp.array([4])
        )

        assert jnp.all(get_total_chips(state) == initial_total)

    def test_chip_conservation_after_fold(self, single_game_state):
        """Folding should preserve total chips."""
        initial_total = get_total_chips(single_game_state)

        state = step(single_game_state, jnp.array([ACTION_FOLD]))

        assert jnp.all(get_total_chips(state) == initial_total)

    def test_chip_conservation_after_check(self, rng_key):
        """Checking should preserve total chips."""
        state = reset(rng_key, n_games=1)

        # Complete preflop to get to a point where check is valid
        state = step(state, jnp.array([ACTION_CALL]))  # SB calls
        initial_total = get_total_chips(state)

        state = step(state, jnp.array([ACTION_CHECK]))  # BB checks

        assert jnp.all(get_total_chips(state) == initial_total)

    def test_chip_conservation_after_all_in(self, rng_key):
        """All-in should preserve total chips."""
        state = reset(rng_key, n_games=1, starting_chips=100)
        initial_total = get_total_chips(state)

        state = step(state, jnp.array([ACTION_ALL_IN]))

        assert jnp.all(get_total_chips(state) == initial_total)

    def test_chip_conservation_through_full_game(self, rng_key):
        """Chips should be conserved through entire hand."""
        state = reset(rng_key, n_games=1, starting_chips=200)
        initial_total = get_total_chips(state)

        # Play through to showdown with checks/calls
        # Preflop: SB calls, BB checks
        state = step(state, jnp.array([ACTION_CALL]))
        assert jnp.all(get_total_chips(state) == initial_total)

        state = step(state, jnp.array([ACTION_CHECK]))
        assert jnp.all(get_total_chips(state) == initial_total)

        # Post-flop rounds: both check
        for _ in range(3):  # flop, turn, river
            if state.done[0]:
                break
            state = step(state, jnp.array([ACTION_CHECK]))
            assert jnp.all(get_total_chips(state) == initial_total)
            if not state.done[0]:
                state = step(state, jnp.array([ACTION_CHECK]))
                assert jnp.all(get_total_chips(state) == initial_total)

    @pytest.mark.parametrize("n_games", [1, 10, 100])
    def test_chip_conservation_batch(self, rng_key, n_games):
        """Test chip conservation for batched games."""
        state = reset(rng_key, n_games=n_games, starting_chips=200)
        initial_total = get_total_chips(state)

        # Random sequence of calls
        for _ in range(10):
            if jnp.all(state.done):
                break
            state = step(state, jnp.full(n_games, ACTION_CALL))
            assert jnp.all(get_total_chips(state) == initial_total)


class TestBlindPosting:
    """Test correct blind amounts are posted."""

    @pytest.mark.parametrize("sb,bb", [(1, 2), (5, 10), (10, 20), (25, 50)])
    def test_blinds_posted_correctly(self, rng_key, sb, bb):
        """Test various blind levels are posted correctly."""
        state = reset(
            rng_key, n_games=1, starting_chips=200, small_blind=sb, big_blind=bb
        )

        # Player 0 posts SB, Player 1 posts BB
        assert state.chips[0, 0] == 200 - sb
        assert state.chips[0, 1] == 200 - bb
        assert state.bets[0, 0] == sb
        assert state.bets[0, 1] == bb
        assert state.pot[0] == 0  # Pot starts empty, blinds in bets
        assert state.total_invested[0, 0] == sb
        assert state.total_invested[0, 1] == bb

    def test_pot_starts_empty(self, rng_key):
        """Pot should start empty (blinds are in bets, moved to pot when round ends)."""
        state = reset(rng_key, n_games=1, small_blind=5, big_blind=10)
        assert state.pot[0] == 0
        # Blinds are in bets
        assert state.bets[0, 0] == 5
        assert state.bets[0, 1] == 10


class TestCallAmount:
    """Test call amount calculation."""

    def test_sb_calls_bb(self, rng_key):
        """SB calling BB should cost BB - SB."""
        sb, bb = 1, 2
        state = reset(
            rng_key, n_games=1, starting_chips=200, small_blind=sb, big_blind=bb
        )

        # Player 0 (SB) to act, should pay 1 chip to call
        initial_chips_p0 = state.chips[0, 0]
        state = step(state, jnp.array([ACTION_CALL]))

        assert state.chips[0, 0] == initial_chips_p0 - 1  # Paid 1 to call
        assert state.bets[0, 0] == bb  # Now matches BB

    def test_call_after_raise(self, rng_key):
        """Calling a raise should match the raise amount."""
        state = reset(
            rng_key, n_games=1, starting_chips=200, small_blind=1, big_blind=2
        )

        # SB raises to 6
        state = step(state, jnp.array([ACTION_RAISE]), raise_amounts=jnp.array([6]))

        # BB calls 6 (needs to add 4 more)
        initial_chips_p1 = state.chips[0, 1]
        state = step(state, jnp.array([ACTION_CALL]))

        # BB paid 4 more (had bet 2, now 6)
        assert state.chips[0, 1] == initial_chips_p1 - 4
        # After BB calls, round advances and bets reset to 0
        # Pot now contains the bets from preflop
        assert state.bets[0, 1] == 0
        assert state.pot[0] == 12  # 6 + 6 from preflop betting


class TestRaiseMinimum:
    """Test minimum raise enforcement."""

    def test_min_raise_is_bb_preflop(self, rng_key):
        """Minimum raise preflop should be the big blind."""
        state = reset(
            rng_key, n_games=1, starting_chips=200, small_blind=1, big_blind=2
        )

        # Min raise = BB's bet (2) + min raise amount (BB=2) = 4
        assert state.last_raise_amount[0] == 2

        # Try min raise
        state = step(state, jnp.array([ACTION_RAISE]), raise_amounts=jnp.array([4]))
        assert state.bets[0, 0] == 4  # Raised to 4

    def test_raise_increases_bet(self, rng_key):
        """Raising should increase the bet amount."""
        state = reset(rng_key, n_games=1, starting_chips=200)

        initial_bet = state.bets[0, 0]
        state = step(state, jnp.array([ACTION_RAISE]), raise_amounts=jnp.array([10]))

        assert state.bets[0, 0] > initial_bet
        assert state.bets[0, 0] == 10


class TestAllInDetection:
    """Test all-in scenarios."""

    def test_all_in_sets_flag(self, rng_key):
        """Going all-in should set the all_in flag."""
        state = reset(rng_key, n_games=1, starting_chips=50, small_blind=1, big_blind=2)

        # SB goes all-in
        state = step(state, jnp.array([ACTION_ALL_IN]))

        assert state.all_in[0, 0] == True
        assert state.chips[0, 0] == 0

    def test_all_in_commits_all_chips(self, rng_key):
        """All-in should put all remaining chips in."""
        starting = 50
        sb = 1
        state = reset(
            rng_key, n_games=1, starting_chips=starting, small_blind=sb, big_blind=2
        )

        # SB has 49 chips after posting SB
        state = step(state, jnp.array([ACTION_ALL_IN]))

        # All 49 remaining chips should be bet
        assert state.bets[0, 0] == starting  # 1 (SB) + 49 = 50
        assert state.chips[0, 0] == 0

    def test_call_becomes_all_in_when_short(self, rng_key):
        """Calling with insufficient chips should result in all-in."""
        state = reset(rng_key, n_games=1, starting_chips=30, small_blind=1, big_blind=2)

        # SB raises all-in (30 chips total)
        state = step(state, jnp.array([ACTION_ALL_IN]))

        # BB calls but only has 28 chips left (30 - 2 BB)
        state = step(state, jnp.array([ACTION_CALL]))

        # BB should be all-in too
        assert state.all_in[0, 1] == True
        assert state.chips[0, 1] == 0


class TestRewardCalculation:
    """Test reward computation."""

    def test_rewards_zero_sum(self, rng_key):
        """Rewards should sum to zero (zero-sum game)."""
        n_games = 100
        state = reset(rng_key, n_games=n_games)

        # Play to completion
        while not jnp.all(state.done):
            actions = jnp.where(
                state.round == ROUND_PREFLOP, ACTION_CALL, ACTION_CHECK
            )
            state = step(state, actions)

        rewards = get_rewards(state)

        # Sum across players should be ~0 (floating point)
        sums = jnp.sum(rewards, axis=1)
        assert jnp.allclose(sums, 0.0, atol=1e-5)

    def test_winner_gets_pot(self, rng_key):
        """Winner should receive entire pot."""
        # Use a deterministic scenario with fold
        state = reset(
            rng_key, n_games=1, starting_chips=200, small_blind=1, big_blind=2
        )

        # P0 folds, P1 wins
        state = step(state, jnp.array([ACTION_FOLD]))

        rewards = get_rewards(state)

        # P1 wins pot (3) minus their investment (2) = +1
        # P0 loses their investment (1) = -1
        assert rewards[0, 1] == 1  # BB wins SB
        assert rewards[0, 0] == -1  # SB loses SB

    def test_fold_rewards(self, rng_key):
        """Test rewards when one player folds after betting."""
        state = reset(rng_key, n_games=1, starting_chips=200, small_blind=5, big_blind=10)

        # P0 raises to 30
        state = step(state, jnp.array([ACTION_RAISE]), raise_amounts=jnp.array([30]))

        # P1 folds
        state = step(state, jnp.array([ACTION_FOLD]))

        rewards = get_rewards(state)

        # P0 wins pot (40) minus investment (30) = +10
        # P1 loses investment (10) = -10
        assert rewards[0, 0] == 10
        assert rewards[0, 1] == -10
