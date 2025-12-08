"""JAX-compatible rule-based opponents for evaluation."""

import jax
import jax.numpy as jnp
import jax.random as jrandom
from jax import Array

from poker_jax.state import (
    GameState,
    ACTION_FOLD, ACTION_CHECK, ACTION_CALL,
    ACTION_RAISE_33, ACTION_RAISE_66, ACTION_RAISE_100, ACTION_RAISE_150,
    ACTION_ALL_IN, NUM_ACTIONS,
    ROUND_PREFLOP,
)

# Hand strength feature indices in observation
# Observation layout: hole(104) + community(260) + round(4) + position(2) + chips(6) + valid(9) + hand_strength(18)
HAND_STRENGTH_OFFSET = 104 + 260 + 4 + 2 + 6 + 9  # = 385
# Index 10 within hand strength is normalized strength (category / 9)
NORMALIZED_STRENGTH_IDX = HAND_STRENGTH_OFFSET + 10  # = 395


# ========== RANDOM OPPONENT ==========
@jax.jit
def random_opponent(
    state: GameState,
    valid_mask: Array,
    rng_key: Array,
) -> Array:
    """Random opponent - uniformly selects from valid actions.

    Args:
        state: GameState [N games]
        valid_mask: [N, NUM_ACTIONS] valid action mask
        rng_key: PRNG key

    Returns:
        [N] action indices
    """
    # Sample uniformly from valid actions using categorical with equal logits
    logits = jnp.where(valid_mask, 0.0, -1e9)
    actions = jax.random.categorical(rng_key, logits)
    return actions


# ========== CALL STATION ==========
@jax.jit
def call_station_opponent(
    state: GameState,
    valid_mask: Array,
    rng_key: Array,
) -> Array:
    """Call station - always calls or checks, never folds or raises.

    Priority: CHECK > CALL > ALL_IN (if forced) > FOLD (last resort)
    """
    can_check = valid_mask[:, ACTION_CHECK]
    can_call = valid_mask[:, ACTION_CALL]
    can_all_in = valid_mask[:, ACTION_ALL_IN]

    actions = jnp.where(
        can_check, ACTION_CHECK,
        jnp.where(
            can_call, ACTION_CALL,
            jnp.where(
                can_all_in, ACTION_ALL_IN,
                ACTION_FOLD  # Fallback
            )
        )
    )

    return actions


# ========== PREFLOP STRENGTH HEURISTIC ==========
@jax.jit
def _compute_preflop_strength(hole_cards: Array) -> Array:
    """Compute preflop hand strength heuristic.

    Args:
        hole_cards: [N, 2] hole cards (encoded as 0-51)

    Returns:
        [N] strength values in [0, 1]
    """
    # Extract ranks (0-12, where 0=2, 12=A)
    ranks = hole_cards // 4  # [N, 2]
    r1, r2 = ranks[:, 0], ranks[:, 1]

    high_card = jnp.maximum(r1, r2)
    low_card = jnp.minimum(r1, r2)
    is_pair = r1 == r2
    gap = high_card - low_card

    # Suited check (same suit = card % 4 equal)
    suits = hole_cards % 4
    suited = suits[:, 0] == suits[:, 1]

    # Base strength
    strength = jnp.where(
        is_pair,
        0.5 + (high_card.astype(jnp.float32)) / 24.0,  # Pairs: 0.5-1.0
        (high_card + low_card).astype(jnp.float32) / 24.0  # Non-pairs: 0.0-1.0
    )

    # Bonuses
    strength = jnp.where(suited, strength + 0.08, strength)
    strength = jnp.where(gap <= 2, strength + 0.04, strength)
    strength = jnp.where(gap <= 1, strength + 0.02, strength)

    # Premium hands override
    # AA, KK, QQ (pair with high >= 10, which is Q)
    premium_pair = is_pair & (high_card >= 10)
    strength = jnp.where(premium_pair, jnp.maximum(strength, 0.92), strength)

    # JJ, TT
    good_pair = is_pair & (high_card >= 8) & (high_card < 10)
    strength = jnp.where(good_pair, jnp.maximum(strength, 0.80), strength)

    # AK (A=12, K=11)
    is_ak = (high_card == 12) & (low_card == 11)
    strength = jnp.where(is_ak, jnp.maximum(strength, 0.85), strength)

    # AQ, AJ
    is_aq_aj = (high_card == 12) & (low_card >= 9)
    strength = jnp.where(is_aq_aj, jnp.maximum(strength, 0.75), strength)

    return jnp.clip(strength, 0.0, 1.0)


# ========== TIGHT-AGGRESSIVE (TAG) ==========
@jax.jit
def tag_opponent(
    state: GameState,
    valid_mask: Array,
    rng_key: Array,
    obs: Array,
) -> Array:
    """Tight-Aggressive opponent.

    Strategy:
    - Strong hands (>0.75): Raise
    - Medium hands (0.5-0.75): Call/Check
    - Weak hands (0.35-0.5): Check if free
    - Very weak (<0.35): Fold
    """
    n_games = state.done.shape[0]
    game_idx = jnp.arange(n_games)
    player_idx = state.current_player

    # Get hole cards for current player
    hole_cards = state.hole_cards[game_idx, player_idx, :]  # [N, 2]

    # Compute preflop strength
    preflop_strength = _compute_preflop_strength(hole_cards)

    # Get postflop strength from observation
    postflop_strength = obs[:, NORMALIZED_STRENGTH_IDX]

    # Use preflop strength for preflop, postflop otherwise
    is_preflop = state.round == ROUND_PREFLOP
    strength = jnp.where(is_preflop, preflop_strength, postflop_strength)

    # Add small noise for variety
    rng_key, noise_key = jrandom.split(rng_key)
    noise = jrandom.normal(noise_key, (n_games,)) * 0.08
    strength = jnp.clip(strength + noise, 0.0, 1.0)

    # Decision thresholds for TAG
    strong = strength > 0.75
    medium = (strength > 0.5) & ~strong
    weak = (strength > 0.35) & ~medium & ~strong

    # Get valid action masks
    can_check = valid_mask[:, ACTION_CHECK]
    can_call = valid_mask[:, ACTION_CALL]
    can_raise_66 = valid_mask[:, ACTION_RAISE_66]
    can_raise_100 = valid_mask[:, ACTION_RAISE_100]
    can_all_in = valid_mask[:, ACTION_ALL_IN]

    # Strong: prefer raise_66, then raise_100, then all_in
    strong_action = jnp.where(
        can_raise_66, ACTION_RAISE_66,
        jnp.where(can_raise_100, ACTION_RAISE_100,
        jnp.where(can_all_in, ACTION_ALL_IN,
        jnp.where(can_call, ACTION_CALL, ACTION_FOLD)))
    )

    # Medium: check or call
    medium_action = jnp.where(
        can_check, ACTION_CHECK,
        jnp.where(can_call, ACTION_CALL, ACTION_FOLD)
    )

    # Weak: only check
    weak_action = jnp.where(can_check, ACTION_CHECK, ACTION_FOLD)

    # Default: fold
    actions = jnp.where(
        strong, strong_action,
        jnp.where(medium, medium_action,
        jnp.where(weak, weak_action, ACTION_FOLD))
    )

    return actions


# ========== LOOSE-AGGRESSIVE (LAG) ==========
@jax.jit
def lag_opponent(
    state: GameState,
    valid_mask: Array,
    rng_key: Array,
    obs: Array,
) -> Array:
    """Loose-Aggressive opponent.

    Strategy:
    - Very aggressive (raises frequently)
    - Lower hand requirements than TAG
    - Bluffs often
    """
    n_games = state.done.shape[0]
    game_idx = jnp.arange(n_games)
    player_idx = state.current_player

    hole_cards = state.hole_cards[game_idx, player_idx, :]
    preflop_strength = _compute_preflop_strength(hole_cards)

    postflop_strength = obs[:, NORMALIZED_STRENGTH_IDX]

    is_preflop = state.round == ROUND_PREFLOP
    strength = jnp.where(is_preflop, preflop_strength, postflop_strength)

    # LAG adds aggression boost and more noise
    rng_key, noise_key = jrandom.split(rng_key)
    noise = jrandom.normal(noise_key, (n_games,)) * 0.12
    aggression_boost = 0.15  # Makes LAG more likely to raise
    adjusted_strength = jnp.clip(strength + noise + aggression_boost, 0.0, 1.0)

    # Lower thresholds than TAG
    strong = adjusted_strength > 0.55
    medium = (adjusted_strength > 0.35) & ~strong
    weak = (adjusted_strength > 0.2) & ~medium & ~strong

    can_check = valid_mask[:, ACTION_CHECK]
    can_call = valid_mask[:, ACTION_CALL]
    can_raise_66 = valid_mask[:, ACTION_RAISE_66]
    can_raise_100 = valid_mask[:, ACTION_RAISE_100]
    can_raise_150 = valid_mask[:, ACTION_RAISE_150]
    can_all_in = valid_mask[:, ACTION_ALL_IN]

    # Strong: prefer big raises (100% or 150% pot)
    strong_action = jnp.where(
        can_raise_100, ACTION_RAISE_100,
        jnp.where(can_raise_150, ACTION_RAISE_150,
        jnp.where(can_raise_66, ACTION_RAISE_66,
        jnp.where(can_all_in, ACTION_ALL_IN,
        jnp.where(can_call, ACTION_CALL, ACTION_FOLD))))
    )

    # Medium: still aggressive, prefer raising
    medium_action = jnp.where(
        can_raise_66, ACTION_RAISE_66,
        jnp.where(can_call, ACTION_CALL,
        jnp.where(can_check, ACTION_CHECK, ACTION_FOLD))
    )

    # Weak: check or fold
    weak_action = jnp.where(can_check, ACTION_CHECK, ACTION_FOLD)

    actions = jnp.where(
        strong, strong_action,
        jnp.where(medium, medium_action,
        jnp.where(weak, weak_action, ACTION_FOLD))
    )

    return actions


# ========== ROCK (Very Tight) ==========
@jax.jit
def rock_opponent(
    state: GameState,
    valid_mask: Array,
    rng_key: Array,
    obs: Array,
) -> Array:
    """Rock opponent - only plays premium hands.

    Strategy:
    - Only plays with very strong hands (>0.82)
    - Very tight preflop (only AA, KK, QQ, JJ, AK, AQs type hands)
    - Folds everything else
    """
    n_games = state.done.shape[0]
    game_idx = jnp.arange(n_games)
    player_idx = state.current_player

    hole_cards = state.hole_cards[game_idx, player_idx, :]
    preflop_strength = _compute_preflop_strength(hole_cards)

    postflop_strength = obs[:, NORMALIZED_STRENGTH_IDX]

    is_preflop = state.round == ROUND_PREFLOP
    strength = jnp.where(is_preflop, preflop_strength, postflop_strength)

    # Rock: very high thresholds
    premium = strength > 0.82
    good = (strength > 0.65) & ~premium

    can_check = valid_mask[:, ACTION_CHECK]
    can_call = valid_mask[:, ACTION_CALL]
    can_raise_66 = valid_mask[:, ACTION_RAISE_66]
    can_raise_100 = valid_mask[:, ACTION_RAISE_100]

    # Premium: raise
    premium_action = jnp.where(
        can_raise_100, ACTION_RAISE_100,
        jnp.where(can_raise_66, ACTION_RAISE_66,
        jnp.where(can_call, ACTION_CALL, ACTION_FOLD))
    )

    # Good: call if cheap, check if free
    good_action = jnp.where(
        can_check, ACTION_CHECK,
        jnp.where(can_call, ACTION_CALL, ACTION_FOLD)
    )

    # Everything else: check or fold
    default_action = jnp.where(can_check, ACTION_CHECK, ACTION_FOLD)

    actions = jnp.where(
        premium, premium_action,
        jnp.where(good, good_action, default_action)
    )

    return actions


# ========== OPPONENT REGISTRY ==========
OPPONENT_TYPES = {
    "random": random_opponent,
    "call_station": call_station_opponent,
    "tag": tag_opponent,
    "lag": lag_opponent,
    "rock": rock_opponent,
}

# Opponents that need obs parameter
NEEDS_OBS = {"tag", "lag", "rock"}
