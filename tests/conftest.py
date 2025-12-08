"""Shared pytest fixtures for JAX poker tests."""

import pytest
import jax.numpy as jnp
import jax.random as jrandom

from poker_jax import reset, step, GameState
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
)
from tests.helpers.card_utils import make_cards_from_strings


@pytest.fixture
def rng_key():
    """Provide a reproducible JAX PRNG key."""
    return jrandom.PRNGKey(42)


@pytest.fixture
def single_game_state(rng_key):
    """Create a single-game state for testing."""
    return reset(rng_key, n_games=1, starting_chips=200, small_blind=1, big_blind=2)


@pytest.fixture
def batch_game_state(rng_key):
    """Create a batch of 100 games for vectorized testing."""
    return reset(rng_key, n_games=100, starting_chips=200, small_blind=1, big_blind=2)


@pytest.fixture(params=[1, 10, 100])
def n_games(request):
    """Parametrize over different batch sizes."""
    return request.param
