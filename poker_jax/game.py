"""Core game logic for JAX-accelerated poker.

This module provides the step() and reset() functions for running
vectorized poker games on GPU.
"""

import jax
import jax.numpy as jnp
import jax.random as jrandom
from jax import Array

from poker_jax.state import (
    GameState,
    create_initial_state,
    ACTION_FOLD,
    ACTION_CHECK,
    ACTION_CALL,
    ACTION_RAISE_33,
    ACTION_RAISE_66,
    ACTION_RAISE_100,
    ACTION_RAISE_150,
    ACTION_ALL_IN,
    MAX_HISTORY,
    ROUND_PREFLOP,
    ROUND_FLOP,
    ROUND_TURN,
    ROUND_RIVER,
    ROUND_SHOWDOWN,
)
from poker_jax.deck import (
    shuffle_decks,
    deal_hole_cards_batch,
    deal_flop_batch,
    deal_turn_batch,
    deal_river_batch,
)
from poker_jax.hands import evaluate_hands_batch, determine_winner


from functools import partial

# Raise fractions for pot-relative betting
RAISE_FRACTIONS = {
    ACTION_RAISE_33: 0.33,
    ACTION_RAISE_66: 0.66,
    ACTION_RAISE_100: 1.0,
    ACTION_RAISE_150: 1.5,
}


@jax.jit
def _compute_raise_amount(state: GameState, action: Array) -> Array:
    """Compute raise amount based on action type (pot-relative).

    Args:
        state: Current game state
        action: [N] action indices

    Returns:
        [N] raise amounts (total bet size)
    """
    n_games = state.done.shape[0]
    game_idx = jnp.arange(n_games)

    player_idx = state.current_player
    my_bet = state.bets[game_idx, player_idx]
    opp_bet = state.bets[game_idx, 1 - player_idx]

    # Current pot
    pot = state.pot + state.bets[:, 0] + state.bets[:, 1]

    # Min raise floor
    min_raise_total = opp_bet + state.last_raise_amount

    # Compute raise amount for each fraction
    # raise_amount = opp_bet + pot * fraction
    raise_33 = jnp.maximum(opp_bet + (pot * 0.33).astype(jnp.int32), min_raise_total)
    raise_66 = jnp.maximum(opp_bet + (pot * 0.66).astype(jnp.int32), min_raise_total)
    raise_100 = jnp.maximum(opp_bet + (pot * 1.0).astype(jnp.int32), min_raise_total)
    raise_150 = jnp.maximum(opp_bet + (pot * 1.5).astype(jnp.int32), min_raise_total)

    # Select based on action
    raise_amount = jnp.where(action == ACTION_RAISE_33, raise_33,
                   jnp.where(action == ACTION_RAISE_66, raise_66,
                   jnp.where(action == ACTION_RAISE_100, raise_100,
                   jnp.where(action == ACTION_RAISE_150, raise_150,
                             min_raise_total))))  # Default fallback

    return raise_amount


@jax.jit
def _record_action(state: GameState, actions: Array, bet_amounts: Array) -> GameState:
    """Record action in history.

    Args:
        state: Current game state
        actions: [N] action indices taken
        bet_amounts: [N] bet amounts (for raises)

    Returns:
        Updated state with action recorded in history
    """
    n_games = state.done.shape[0]
    game_idx = jnp.arange(n_games)

    # Normalize bet amount by starting chips
    norm = state.starting_chips.astype(jnp.float32)
    bet_normalized = bet_amounts / norm

    # Current history position (clamped to MAX_HISTORY - 1 for safety)
    hist_idx = jnp.minimum(state.history_len, MAX_HISTORY - 1)

    # Create action record: [player_id, action_type_normalized, bet_amount_normalized]
    player_id = state.current_player.astype(jnp.float32)
    action_normalized = actions.astype(jnp.float32) / 8.0  # Normalize to ~[0, 1]

    # Update history at current position
    # Only update if we haven't filled history yet
    should_update = state.history_len < MAX_HISTORY

    new_history = state.action_history
    new_history = new_history.at[game_idx, hist_idx, 0].set(
        jnp.where(should_update, player_id, state.action_history[game_idx, hist_idx, 0])
    )
    new_history = new_history.at[game_idx, hist_idx, 1].set(
        jnp.where(should_update, action_normalized, state.action_history[game_idx, hist_idx, 1])
    )
    new_history = new_history.at[game_idx, hist_idx, 2].set(
        jnp.where(should_update, bet_normalized, state.action_history[game_idx, hist_idx, 2])
    )

    # Increment history length
    new_history_len = jnp.minimum(state.history_len + 1, MAX_HISTORY)

    return state._replace(action_history=new_history, history_len=new_history_len)


@partial(jax.jit, static_argnums=(1, 2, 3, 4))
def reset(
    rng_key: Array,
    n_games: int,
    starting_chips: int = 200,
    small_blind: int = 1,
    big_blind: int = 2,
) -> GameState:
    """Initialize N fresh poker games.

    Sets up the game state with:
    - Shuffled decks
    - Dealt hole cards
    - Posted blinds
    - First player to act (in heads-up: SB/button acts first preflop)

    Args:
        rng_key: JAX PRNG key
        n_games: Number of parallel games
        starting_chips: Starting chip count per player
        small_blind: Small blind amount
        big_blind: Big blind amount

    Returns:
        GameState ready for first action
    """
    # Create initial state
    state = create_initial_state(
        rng_key, n_games, starting_chips, small_blind, big_blind
    )

    # Shuffle decks
    key, subkey = jrandom.split(rng_key)
    keys = jrandom.split(subkey, n_games)
    decks = shuffle_decks(keys)
    state = state._replace(deck=decks, deck_idx=jnp.zeros(n_games, dtype=jnp.int32))

    # Deal hole cards
    hole_cards, deck_idx = deal_hole_cards_batch(state.deck, state.deck_idx)
    state = state._replace(hole_cards=hole_cards, deck_idx=deck_idx)

    # Post blinds
    # In heads-up: button/dealer posts SB, other player posts BB
    # Button is player 0, so:
    # - Player 0 (button) posts SB
    # - Player 1 posts BB

    # Post small blind (player 0)
    sb = state.small_blind
    chips_p0 = state.chips[:, 0] - sb
    bets_p0 = sb
    invested_p0 = sb

    # Post big blind (player 1)
    bb = state.big_blind
    chips_p1 = state.chips[:, 1] - bb
    bets_p1 = bb
    invested_p1 = bb

    # Update state
    chips = jnp.stack([chips_p0, chips_p1], axis=1)
    bets = jnp.stack([bets_p0, bets_p1], axis=1)
    total_invested = jnp.stack([invested_p0, invested_p1], axis=1)

    # Pot starts at 0 - blinds are in bets, will be moved to pot when round advances
    pot = jnp.zeros(n_games, dtype=jnp.int32)

    # In heads-up preflop, button (SB) acts first
    current_player = jnp.zeros(n_games, dtype=jnp.int32)

    state = state._replace(
        chips=chips,
        bets=bets,
        total_invested=total_invested,
        pot=pot,
        current_player=current_player,
        round=jnp.zeros(n_games, dtype=jnp.int32),  # PREFLOP
        last_raise_amount=bb,  # Min raise is BB
    )

    return state


@jax.jit
def _handle_fold(state: GameState, action_mask: Array) -> GameState:
    """Handle fold action for games where fold was chosen."""
    # Mark player as folded
    player_idx = state.current_player
    folded = state.folded.at[jnp.arange(state.done.shape[0]), player_idx].set(
        jnp.where(action_mask, True, state.folded[jnp.arange(state.done.shape[0]), player_idx])
    )

    # Other player wins
    other_player = 1 - player_idx
    winner = jnp.where(action_mask, other_player, state.winner)
    done = jnp.where(action_mask, True, state.done)

    return state._replace(folded=folded, winner=winner, done=done)


@jax.jit
def _handle_check(state: GameState, action_mask: Array) -> GameState:
    """Handle check action."""
    # Check is just passing - no chips change
    return state


@jax.jit
def _handle_call(state: GameState, action_mask: Array) -> GameState:
    """Handle call action."""
    n_games = state.done.shape[0]
    game_idx = jnp.arange(n_games)

    player_idx = state.current_player
    opp_idx = 1 - player_idx

    # Get current bets
    my_bet = state.bets[game_idx, player_idx]
    opp_bet = state.bets[game_idx, opp_idx]
    my_chips = state.chips[game_idx, player_idx]

    # Amount to call
    call_amount = opp_bet - my_bet
    call_amount = jnp.minimum(call_amount, my_chips)  # Can't call more than we have

    # Update chips and bets
    new_chips = jnp.where(action_mask, my_chips - call_amount, my_chips)
    new_bet = jnp.where(action_mask, my_bet + call_amount, my_bet)
    new_invested = jnp.where(
        action_mask,
        state.total_invested[game_idx, player_idx] + call_amount,
        state.total_invested[game_idx, player_idx]
    )

    # Update arrays
    chips = state.chips.at[game_idx, player_idx].set(new_chips)
    bets = state.bets.at[game_idx, player_idx].set(new_bet)
    total_invested = state.total_invested.at[game_idx, player_idx].set(new_invested)

    # Check if player is now all-in
    all_in = state.all_in.at[game_idx, player_idx].set(
        jnp.where(action_mask & (new_chips == 0), True, state.all_in[game_idx, player_idx])
    )

    return state._replace(
        chips=chips, bets=bets, total_invested=total_invested, all_in=all_in
    )


@jax.jit
def _handle_raise(state: GameState, action_mask: Array, raise_amounts: Array) -> GameState:
    """Handle raise action.

    Args:
        state: Current game state
        action_mask: [N] bool mask for games taking this action
        raise_amounts: [N] raise amounts (total bet, not additional)
    """
    n_games = state.done.shape[0]
    game_idx = jnp.arange(n_games)

    player_idx = state.current_player
    my_bet = state.bets[game_idx, player_idx]
    my_chips = state.chips[game_idx, player_idx]

    # Calculate additional chips needed
    additional = raise_amounts - my_bet
    additional = jnp.clip(additional, 0, my_chips)

    # Update chips and bets
    new_chips = jnp.where(action_mask, my_chips - additional, my_chips)
    new_bet = jnp.where(action_mask, my_bet + additional, my_bet)
    new_invested = jnp.where(
        action_mask,
        state.total_invested[game_idx, player_idx] + additional,
        state.total_invested[game_idx, player_idx]
    )

    # Track raise amount for min-raise calculation
    opp_bet = state.bets[game_idx, 1 - player_idx]
    raise_size = new_bet - opp_bet
    new_last_raise = jnp.where(action_mask, raise_size, state.last_raise_amount)

    # Update arrays
    chips = state.chips.at[game_idx, player_idx].set(new_chips)
    bets = state.bets.at[game_idx, player_idx].set(new_bet)
    total_invested = state.total_invested.at[game_idx, player_idx].set(new_invested)

    # Check if player is now all-in
    all_in = state.all_in.at[game_idx, player_idx].set(
        jnp.where(action_mask & (new_chips == 0), True, state.all_in[game_idx, player_idx])
    )

    return state._replace(
        chips=chips,
        bets=bets,
        total_invested=total_invested,
        all_in=all_in,
        last_raise_amount=new_last_raise,
    )


@jax.jit
def _handle_all_in(state: GameState, action_mask: Array) -> GameState:
    """Handle all-in action."""
    n_games = state.done.shape[0]
    game_idx = jnp.arange(n_games)

    player_idx = state.current_player
    my_bet = state.bets[game_idx, player_idx]
    my_chips = state.chips[game_idx, player_idx]

    # Put all chips in
    additional = my_chips
    new_bet = my_bet + additional
    new_chips = jnp.where(action_mask, jnp.zeros_like(my_chips), my_chips)
    new_bet_val = jnp.where(action_mask, new_bet, my_bet)
    new_invested = jnp.where(
        action_mask,
        state.total_invested[game_idx, player_idx] + additional,
        state.total_invested[game_idx, player_idx]
    )

    # Track raise amount if this is a raise
    opp_bet = state.bets[game_idx, 1 - player_idx]
    is_raise = new_bet_val > opp_bet
    raise_size = new_bet_val - opp_bet
    new_last_raise = jnp.where(
        action_mask & is_raise,
        jnp.maximum(raise_size, state.last_raise_amount),
        state.last_raise_amount
    )

    # Update arrays
    chips = state.chips.at[game_idx, player_idx].set(new_chips)
    bets = state.bets.at[game_idx, player_idx].set(new_bet_val)
    total_invested = state.total_invested.at[game_idx, player_idx].set(new_invested)
    all_in = state.all_in.at[game_idx, player_idx].set(
        jnp.where(action_mask, True, state.all_in[game_idx, player_idx])
    )

    return state._replace(
        chips=chips,
        bets=bets,
        total_invested=total_invested,
        all_in=all_in,
        last_raise_amount=new_last_raise,
    )


@jax.jit
def _advance_round(state: GameState) -> GameState:
    """Advance to next round if betting is complete."""
    n_games = state.done.shape[0]

    # Check if round is complete:
    # - Both players have acted at least once (actions_this_round >= 2)
    # - Bets are equal (or one is all-in)
    # - Neither has folded
    bets_equal = state.bets[:, 0] == state.bets[:, 1]
    someone_all_in = state.all_in[:, 0] | state.all_in[:, 1]
    both_acted = state.actions_this_round >= 2

    round_complete = both_acted & (bets_equal | someone_all_in) & ~state.done

    # Determine next round
    current_round = state.round
    next_round = jnp.minimum(current_round + 1, ROUND_SHOWDOWN)

    # Move bets to pot
    pot = jnp.where(
        round_complete,
        state.pot + state.bets[:, 0] + state.bets[:, 1],
        state.pot
    )

    # Reset bets for next round
    bets = jnp.where(
        round_complete[:, None],
        jnp.zeros_like(state.bets),
        state.bets
    )

    # Reset actions counter
    actions_this_round = jnp.where(round_complete, 0, state.actions_this_round)

    # Reset last raise amount to big blind
    last_raise_amount = jnp.where(round_complete, state.big_blind, state.last_raise_amount)

    # Deal community cards based on new round
    new_round = jnp.where(round_complete, next_round, current_round)

    # In postflop rounds, BB acts first (player 1 in heads-up)
    # Unless someone is all-in
    new_current_player = jnp.where(
        round_complete & ~someone_all_in,
        jnp.ones(n_games, dtype=jnp.int32),  # Player 1 (BB) acts first postflop
        state.current_player
    )

    state = state._replace(
        round=new_round,
        pot=pot,
        bets=bets,
        actions_this_round=actions_this_round,
        last_raise_amount=last_raise_amount,
        current_player=new_current_player,
    )

    # Deal cards for new round
    # We always call the deal functions and use jnp.where to conditionally apply
    # (JAX requires this pattern - no Python if with traced values)
    should_deal_flop = round_complete & (current_round == ROUND_PREFLOP)
    should_deal_turn = round_complete & (current_round == ROUND_FLOP)
    should_deal_river = round_complete & (current_round == ROUND_TURN)

    # Deal flop (always compute, conditionally apply)
    flop_community, flop_deck_idx = deal_flop_batch(
        state.deck, state.deck_idx, state.community
    )
    community = jnp.where(should_deal_flop[:, None], flop_community, state.community)
    deck_idx = jnp.where(should_deal_flop, flop_deck_idx, state.deck_idx)

    # Deal turn (always compute, conditionally apply)
    turn_community, turn_deck_idx = deal_turn_batch(
        state.deck, deck_idx, community
    )
    community = jnp.where(should_deal_turn[:, None], turn_community, community)
    deck_idx = jnp.where(should_deal_turn, turn_deck_idx, deck_idx)

    # Deal river (always compute, conditionally apply)
    river_community, river_deck_idx = deal_river_batch(
        state.deck, deck_idx, community
    )
    community = jnp.where(should_deal_river[:, None], river_community, community)
    deck_idx = jnp.where(should_deal_river, river_deck_idx, deck_idx)

    state = state._replace(community=community, deck_idx=deck_idx)

    # Check for showdown
    at_showdown = round_complete & (current_round == ROUND_RIVER)
    both_all_in = state.all_in[:, 0] & state.all_in[:, 1]
    go_to_showdown = at_showdown | (round_complete & both_all_in)

    # Evaluate hands and determine winner at showdown
    # Always compute, conditionally apply
    hand_values = evaluate_hands_batch(state.hole_cards, state.community)
    winners = determine_winner(hand_values)
    state = state._replace(
        winner=jnp.where(go_to_showdown, winners, state.winner),
        done=jnp.where(go_to_showdown, True, state.done),
    )

    return state


@jax.jit
def step(
    state: GameState,
    actions: Array,
    raise_amounts: Array | None = None,
) -> GameState:
    """Advance all games by one action.

    Args:
        state: Current game state for N games
        actions: [N] action indices (1=fold, 2=check, 3=call, 4-7=raise sizes, 8=all_in)
        raise_amounts: [N] raise amounts for raise actions (total bet, not additional)
                       If None, computed automatically from action type (pot-relative)

    Returns:
        Updated game state
    """
    n_games = state.done.shape[0]
    game_idx = jnp.arange(n_games)

    # Compute raise amounts based on action type (pot-relative)
    computed_raise_amounts = _compute_raise_amount(state, actions)
    if raise_amounts is None:
        raise_amounts = computed_raise_amounts
    else:
        # Use computed for pot-relative raises, provided for legacy
        is_pot_relative = (
            (actions == ACTION_RAISE_33) |
            (actions == ACTION_RAISE_66) |
            (actions == ACTION_RAISE_100) |
            (actions == ACTION_RAISE_150)
        )
        raise_amounts = jnp.where(is_pot_relative, computed_raise_amounts, raise_amounts)

    # Mask for games that aren't done
    active = ~state.done

    # Handle each action type
    is_fold = (actions == ACTION_FOLD) & active
    is_check = (actions == ACTION_CHECK) & active
    is_call = (actions == ACTION_CALL) & active
    is_raise = (
        (actions == ACTION_RAISE_33) |
        (actions == ACTION_RAISE_66) |
        (actions == ACTION_RAISE_100) |
        (actions == ACTION_RAISE_150)
    ) & active
    is_all_in = (actions == ACTION_ALL_IN) & active

    # Apply actions
    state = _handle_fold(state, is_fold)
    state = _handle_check(state, is_check)
    state = _handle_call(state, is_call)
    state = _handle_raise(state, is_raise, raise_amounts)
    state = _handle_all_in(state, is_all_in)

    # Track actions and switch player
    any_action = is_fold | is_check | is_call | is_raise | is_all_in

    # Record action in history (get bet amount for history)
    # For raises, use raise_amounts; for all-in, use player's chips; for others, use 0
    my_bet_after = state.bets[game_idx, state.current_player]
    bet_for_history = jnp.where(is_raise, raise_amounts,
                      jnp.where(is_all_in, my_bet_after,
                      jnp.where(is_call, my_bet_after,
                                jnp.zeros(n_games, dtype=jnp.int32))))
    state = _record_action(state, actions, bet_for_history.astype(jnp.float32))

    actions_this_round = jnp.where(
        any_action, state.actions_this_round + 1, state.actions_this_round
    )
    last_action = jnp.where(any_action, actions, state.last_action)

    # Switch to other player (if game not done)
    current_player = jnp.where(
        any_action & ~state.done,
        1 - state.current_player,
        state.current_player
    )

    state = state._replace(
        actions_this_round=actions_this_round,
        last_action=last_action,
        current_player=current_player,
    )

    # Check if round should advance
    state = _advance_round(state)

    return state


@jax.jit
def get_rewards(state: GameState) -> Array:
    """Calculate rewards for completed games.

    Rewards are scaled by pot size (sqrt) to emphasize big pot decisions.
    This helps the model learn to play well in large pots, improving BB/100.

    Args:
        state: Game state

    Returns:
        [N, 2] rewards for each player (pot-scaled chip delta)
    """
    n_games = state.done.shape[0]

    # Calculate final pot
    pot = state.pot + state.bets[:, 0] + state.bets[:, 1]

    # Pot importance scaling: sqrt to balance emphasis without extreme variance
    # Small pots (6 chips): 0.17x, Medium (50): 0.5x, Large (200): 1.0x, All-in (400): 1.4x
    pot_scale = jnp.sqrt(pot / state.starting_chips)

    # Rewards based on winner
    # Winner gets pot, loser gets nothing
    # Tie splits pot
    p0_reward = jnp.where(
        state.winner == 0,
        pot - state.total_invested[:, 0],  # Won
        jnp.where(
            state.winner == 1,
            -state.total_invested[:, 0],  # Lost
            pot / 2 - state.total_invested[:, 0]  # Tie
        )
    )

    p1_reward = jnp.where(
        state.winner == 1,
        pot - state.total_invested[:, 1],  # Won
        jnp.where(
            state.winner == 0,
            -state.total_invested[:, 1],  # Lost
            pot / 2 - state.total_invested[:, 1]  # Tie
        )
    )

    # Apply pot scaling to emphasize big pot decisions
    p0_reward = p0_reward * pot_scale
    p1_reward = p1_reward * pot_scale

    # Zero reward for incomplete games
    p0_reward = jnp.where(state.done, p0_reward, 0.0)
    p1_reward = jnp.where(state.done, p1_reward, 0.0)

    return jnp.stack([p0_reward, p1_reward], axis=1)
