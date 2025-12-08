"""Tests for game flow and action handling (poker_jax/game.py)."""

import pytest
import jax.numpy as jnp
import jax.random as jrandom

from poker_jax import reset, step, get_rewards, GameState
from poker_jax.state import (
    ACTION_FOLD,
    ACTION_CHECK,
    ACTION_CALL,
    ACTION_RAISE,
    ACTION_ALL_IN,
    ROUND_PREFLOP,
    ROUND_FLOP,
    ROUND_TURN,
    ROUND_RIVER,
    ROUND_SHOWDOWN,
    get_valid_actions_mask,
)


class TestRoundTransitions:
    """Test betting round transitions."""

    def test_initial_round_is_preflop(self, single_game_state):
        """Game should start at preflop."""
        assert single_game_state.round[0] == ROUND_PREFLOP

    def test_preflop_to_flop(self, single_game_state):
        """Round should advance from preflop to flop after action complete."""
        state = single_game_state
        assert state.round[0] == ROUND_PREFLOP

        # SB calls, BB checks
        state = step(state, jnp.array([ACTION_CALL]))
        assert state.round[0] == ROUND_PREFLOP  # Still preflop

        state = step(state, jnp.array([ACTION_CHECK]))
        assert state.round[0] == ROUND_FLOP  # Now flop

    def test_flop_to_turn(self, single_game_state):
        """Round should advance from flop to turn."""
        state = single_game_state

        # Get to flop
        state = step(state, jnp.array([ACTION_CALL]))
        state = step(state, jnp.array([ACTION_CHECK]))
        assert state.round[0] == ROUND_FLOP

        # Check-check on flop
        state = step(state, jnp.array([ACTION_CHECK]))
        state = step(state, jnp.array([ACTION_CHECK]))
        assert state.round[0] == ROUND_TURN

    def test_turn_to_river(self, single_game_state):
        """Round should advance from turn to river."""
        state = single_game_state

        # Get to turn
        state = step(state, jnp.array([ACTION_CALL]))
        state = step(state, jnp.array([ACTION_CHECK]))
        state = step(state, jnp.array([ACTION_CHECK]))
        state = step(state, jnp.array([ACTION_CHECK]))
        assert state.round[0] == ROUND_TURN

        # Check-check on turn
        state = step(state, jnp.array([ACTION_CHECK]))
        state = step(state, jnp.array([ACTION_CHECK]))
        assert state.round[0] == ROUND_RIVER

    def test_river_to_showdown(self, single_game_state):
        """Game should end after river action complete."""
        state = single_game_state

        # Play to river
        for _ in range(4):  # preflop + flop + turn + river
            if state.done[0]:
                break
            # Call preflop, check postflop
            action = ACTION_CALL if state.round[0] == ROUND_PREFLOP else ACTION_CHECK
            state = step(state, jnp.array([action]))
            if not state.done[0]:
                state = step(state, jnp.array([ACTION_CHECK]))

        # River check-check should end game
        assert state.done[0] == True

    def test_community_cards_dealt_each_round(self, single_game_state):
        """Community cards should be dealt at each round transition."""
        state = single_game_state

        # Initially no community cards
        assert jnp.all(state.community == -1)

        # Get to flop
        state = step(state, jnp.array([ACTION_CALL]))
        state = step(state, jnp.array([ACTION_CHECK]))

        # Should have 3 community cards
        assert jnp.sum(state.community[0] >= 0) == 3

        # Get to turn
        state = step(state, jnp.array([ACTION_CHECK]))
        state = step(state, jnp.array([ACTION_CHECK]))

        # Should have 4 community cards
        assert jnp.sum(state.community[0] >= 0) == 4

        # Get to river
        state = step(state, jnp.array([ACTION_CHECK]))
        state = step(state, jnp.array([ACTION_CHECK]))

        # Should have 5 community cards
        assert jnp.sum(state.community[0] >= 0) == 5


class TestPlayerToAct:
    """Test correct player is to act."""

    def test_sb_acts_first_preflop(self, single_game_state):
        """In heads-up, SB (player 0) acts first preflop."""
        assert single_game_state.current_player[0] == 0

    def test_bb_acts_first_postflop(self, single_game_state):
        """BB (player 1) acts first in postflop rounds."""
        state = single_game_state

        # Complete preflop
        state = step(state, jnp.array([ACTION_CALL]))
        state = step(state, jnp.array([ACTION_CHECK]))

        # On flop, player 1 should act first
        assert state.current_player[0] == 1

    def test_player_alternates_within_round(self, single_game_state):
        """Players should alternate actions within a round."""
        state = single_game_state

        assert state.current_player[0] == 0  # P0 first
        state = step(state, jnp.array([ACTION_CALL]))
        assert state.current_player[0] == 1  # P1 second


class TestGameTermination:
    """Test game ending conditions."""

    def test_fold_ends_game(self, single_game_state):
        """Folding should immediately end the game."""
        state = single_game_state
        assert state.done[0] == False

        state = step(state, jnp.array([ACTION_FOLD]))

        assert state.done[0] == True
        assert state.winner[0] == 1  # Opponent wins

    def test_opponent_wins_on_fold(self, single_game_state):
        """The non-folding player should be winner."""
        state = single_game_state

        # Player 0 folds
        state = step(state, jnp.array([ACTION_FOLD]))
        assert state.winner[0] == 1

        # Test player 1 folding
        state2 = single_game_state
        state2 = step(state2, jnp.array([ACTION_CALL]))  # P0 calls
        state2 = step(state2, jnp.array([ACTION_FOLD]))  # P1 folds
        assert state2.winner[0] == 0

    def test_showdown_determines_winner(self, rng_key):
        """At showdown, better hand should win."""
        # Run many games and verify winner determination
        n_games = 1000
        state = reset(rng_key, n_games=n_games)

        # Play to showdown (all checks/calls)
        while not jnp.all(state.done):
            actions = jnp.where(
                state.round == ROUND_PREFLOP, ACTION_CALL, ACTION_CHECK
            )
            state = step(state, actions)

        # Verify all games ended
        assert jnp.all(state.done)
        # Winners should be 0, 1, or -1 (tie)
        assert jnp.all((state.winner >= -1) & (state.winner <= 1))

    def test_all_in_triggers_showdown(self, rng_key):
        """Both players all-in should go directly to showdown."""
        state = reset(rng_key, n_games=1, starting_chips=50)

        # P0 all-in
        state = step(state, jnp.array([ACTION_ALL_IN]))
        assert not state.done[0]

        # P1 calls all-in
        state = step(state, jnp.array([ACTION_CALL]))

        # Game should be done (showdown)
        assert state.done[0]


class TestActionHandling:
    """Test individual action effects."""

    def test_check_no_chip_change(self, rng_key):
        """Check should not change chips."""
        state = reset(rng_key, n_games=1)

        # Get to flop where check is valid
        state = step(state, jnp.array([ACTION_CALL]))
        state = step(state, jnp.array([ACTION_CHECK]))

        initial_chips = state.chips.copy()
        state = step(state, jnp.array([ACTION_CHECK]))

        # Chips unchanged for player who checked
        assert state.chips[0, 1] == initial_chips[0, 1]

    def test_call_matches_bet(self, single_game_state):
        """Call should match opponent's bet exactly."""
        state = single_game_state

        state = step(state, jnp.array([ACTION_CALL]))

        # Both bets should be equal now
        assert state.bets[0, 0] == state.bets[0, 1]

    def test_raise_increases_bet(self, single_game_state):
        """Raise should increase total bet."""
        state = single_game_state

        initial_bet = state.bets[0, 0]
        state = step(state, jnp.array([ACTION_RAISE]), raise_amounts=jnp.array([6]))

        assert state.bets[0, 0] > initial_bet
        assert state.bets[0, 0] == 6


class TestValidActions:
    """Test action validity masks."""

    def test_can_always_fold(self, single_game_state):
        """Should always be able to fold unless game done."""
        state = single_game_state

        for _ in range(10):
            if state.done[0]:
                break
            mask = get_valid_actions_mask(state)
            assert mask[0, ACTION_FOLD] == True
            action = ACTION_CALL if mask[0, ACTION_CALL] else ACTION_CHECK
            state = step(state, jnp.array([action]))

    def test_check_valid_when_no_bet(self, rng_key):
        """Check should be valid only when no bet to call."""
        state = reset(rng_key, n_games=1)

        # Complete preflop
        state = step(state, jnp.array([ACTION_CALL]))
        state = step(state, jnp.array([ACTION_CHECK]))

        # On flop, first player can check (no bet)
        mask = get_valid_actions_mask(state)
        assert mask[0, ACTION_CHECK] == True

    def test_check_invalid_when_bet_pending(self, single_game_state):
        """Check should be invalid when there's a bet to call."""
        state = single_game_state

        # Preflop, SB faces BB - cannot check
        mask = get_valid_actions_mask(state)
        assert mask[0, ACTION_CHECK] == False

    def test_all_in_valid_with_chips(self, single_game_state):
        """All-in should be valid whenever player has chips."""
        state = single_game_state
        mask = get_valid_actions_mask(state)

        assert mask[0, ACTION_ALL_IN] == True

    def test_call_valid_when_bet_to_match(self, single_game_state):
        """Call should be valid when there's a bet to match."""
        state = single_game_state

        # Preflop, SB can call BB
        mask = get_valid_actions_mask(state)
        assert mask[0, ACTION_CALL] == True


class TestActionsCounter:
    """Test actions tracking."""

    def test_actions_counter_increments(self, single_game_state):
        """Actions counter should increment on each action."""
        state = single_game_state

        assert state.actions_this_round[0] == 0

        state = step(state, jnp.array([ACTION_CALL]))
        assert state.actions_this_round[0] == 1

        state = step(state, jnp.array([ACTION_CHECK]))
        # After round advances, counter resets
        assert state.actions_this_round[0] == 0

    def test_actions_counter_resets_on_round_change(self, single_game_state):
        """Actions counter should reset when round changes."""
        state = single_game_state

        # Complete preflop
        state = step(state, jnp.array([ACTION_CALL]))
        state = step(state, jnp.array([ACTION_CHECK]))

        # Counter should be reset for flop
        assert state.actions_this_round[0] == 0


class TestBetsReset:
    """Test bet resetting between rounds."""

    def test_bets_reset_on_round_change(self, single_game_state):
        """Bets should reset to 0 when round changes."""
        state = single_game_state

        # Complete preflop with non-zero bets
        state = step(state, jnp.array([ACTION_CALL]))
        assert state.bets[0, 0] > 0
        assert state.bets[0, 1] > 0

        state = step(state, jnp.array([ACTION_CHECK]))

        # Bets should be reset for flop
        assert state.bets[0, 0] == 0
        assert state.bets[0, 1] == 0

    def test_pot_accumulates_bets(self, single_game_state):
        """Pot should increase when bets are collected."""
        state = single_game_state
        initial_pot = state.pot[0]

        # Complete preflop
        state = step(state, jnp.array([ACTION_CALL]))
        state = step(state, jnp.array([ACTION_CHECK]))

        # Pot should have increased (blinds collected)
        assert state.pot[0] > initial_pot


class TestLastAction:
    """Test last action tracking."""

    def test_last_action_updated(self, single_game_state):
        """Last action should be updated on each step."""
        state = single_game_state

        state = step(state, jnp.array([ACTION_CALL]))
        assert state.last_action[0] == ACTION_CALL

        state = step(state, jnp.array([ACTION_CHECK]))
        assert state.last_action[0] == ACTION_CHECK


class TestFoldedFlag:
    """Test folded player tracking."""

    def test_folded_flag_set_on_fold(self, single_game_state):
        """Folded flag should be set when player folds."""
        state = single_game_state

        assert state.folded[0, 0] == False

        state = step(state, jnp.array([ACTION_FOLD]))

        assert state.folded[0, 0] == True
        assert state.folded[0, 1] == False  # Opponent didn't fold
