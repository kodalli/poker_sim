"""External sampling MCCFR traversal for Single Deep CFR.

Implements the game tree traversal algorithm where:
- Traverser explores ALL actions at their decision points
- Opponent samples single action from current strategy
- Counterfactual advantages are computed and stored

This is the core algorithm that generates training data for the
advantage network.
"""

from dataclasses import dataclass
from typing import Tuple, List
from functools import partial

import jax
import jax.numpy as jnp
import jax.random as jrandom
from jax import Array
import numpy as np

from poker_jax.state import GameState, NUM_ACTIONS, get_valid_actions_mask
from poker_jax.game import reset, step, get_rewards
from poker_jax.encoding import encode_state_for_current_player, OBS_DIM

from deep_cfr.strategy import regret_match, sample_action_from_strategy


@dataclass
class AdvantageEntry:
    """Entry for advantage memory."""
    obs: np.ndarray       # [OBS_DIM] observation
    advantages: np.ndarray  # [NUM_ACTIONS] advantage values
    valid_mask: np.ndarray  # [NUM_ACTIONS] valid action mask


def traverse_single_game(
    state: GameState,
    traverser: int,
    network,
    params: dict,
    rng_key: Array,
) -> Tuple[float, List[AdvantageEntry]]:
    """Traverse a single game with external sampling.

    Non-vectorized recursive implementation for clarity.
    For production, use the vectorized batch traversal.

    Args:
        state: Current game state (batch size 1)
        traverser: Player we're computing advantages for (0 or 1)
        network: AdvantageNetwork module
        params: Network parameters
        rng_key: PRNG key

    Returns:
        utility: Expected utility for traverser from this state
        entries: List of (obs, advantages) for traverser's decision points
    """
    entries = []

    # Terminal state
    if state.done[0]:
        rewards = get_rewards(state)
        return float(rewards[0, traverser]), entries

    current_player = int(state.current_player[0])

    # Get valid actions and observation
    valid_mask = get_valid_actions_mask(state)[0]  # [NUM_ACTIONS]
    obs = encode_state_for_current_player(state)[0]  # [OBS_DIM]

    # Get current strategy from network
    obs_batch = obs[None, :]  # [1, OBS_DIM]
    advantages_pred = network.apply({"params": params}, obs_batch, training=False)
    strategy = regret_match(advantages_pred, valid_mask[None, :])[0]  # [NUM_ACTIONS]

    if current_player == traverser:
        # Traverser: explore ALL valid actions
        action_values = {}
        all_entries = []

        valid_indices = jnp.where(valid_mask)[0]

        for i, action in enumerate(valid_indices):
            action = int(action)
            rng_key, subkey = jrandom.split(rng_key)

            # Take action
            actions = jnp.array([action])
            next_state = step(state, actions)

            # Recursively traverse
            value, sub_entries = traverse_single_game(
                next_state, traverser, network, params, subkey
            )
            action_values[action] = value
            all_entries.extend(sub_entries)

        # Compute expected value under current strategy
        ev = sum(float(strategy[a]) * action_values[a] for a in action_values)

        # Compute advantages: value(a) - EV
        advantages = np.zeros(NUM_ACTIONS, dtype=np.float32)
        for a, v in action_values.items():
            advantages[a] = v - ev

        # Store entry
        entry = AdvantageEntry(
            obs=np.array(obs),
            advantages=advantages,
            valid_mask=np.array(valid_mask),
        )
        entries = [entry] + all_entries

        return ev, entries

    else:
        # Opponent: sample single action from strategy
        rng_key, action_key = jrandom.split(rng_key)

        # Sample action
        action = int(sample_action_from_strategy(action_key, strategy[None, :])[0])

        # Take action and continue
        actions = jnp.array([action])
        next_state = step(state, actions)

        return traverse_single_game(next_state, traverser, network, params, rng_key)


@partial(jax.jit, static_argnums=(1, 4))
def _play_opponent_step(
    carry: tuple,
    traverser: int,
    network,
    params: dict,
    max_steps: int,
) -> tuple:
    """Play steps until traverser's turn or terminal.

    Args:
        carry: (state, rng_key, step_count)
        traverser: Player we're computing advantages for
        network: AdvantageNetwork module
        params: Network parameters
        max_steps: Safety limit

    Returns:
        Updated carry with state at traverser's decision point
    """
    state, rng_key, step_count = carry

    def cond_fn(carry):
        state, _, step_count = carry
        # Continue while: not done, not traverser's turn, under step limit
        is_opponent_turn = state.current_player != traverser
        return jnp.logical_and(
            jnp.logical_and(~state.done.all(), is_opponent_turn.all()),
            step_count < max_steps
        )

    def body_fn(carry):
        state, rng_key, step_count = carry
        rng_key, action_key = jrandom.split(rng_key)

        # Get observation and strategy
        obs = encode_state_for_current_player(state)
        valid_mask = get_valid_actions_mask(state)

        advantages = network.apply({"params": params}, obs, training=False)
        strategy = regret_match(advantages, valid_mask)

        # Sample action
        actions = sample_action_from_strategy(action_key, strategy)

        # Step
        new_state = step(state, actions)

        return (new_state, rng_key, step_count + 1)

    return jax.lax.while_loop(cond_fn, body_fn, carry)


@partial(jax.jit, static_argnums=(2,))
def batch_traverse_step(
    state: GameState,
    params: dict,
    network,
    rng_key: Array,
    traverser: Array,
) -> Tuple[GameState, Array, Array, Array, Array]:
    """Perform one traversal step for a batch of games.

    At each traverser decision point, we need to explore all actions.
    This function handles a single decision point.

    Args:
        state: Batch game state [N]
        params: Network parameters
        network: AdvantageNetwork module
        rng_key: PRNG key
        traverser: [N] traverser player for each game

    Returns:
        next_state: State after taking sampled action
        obs: [N, OBS_DIM] observations at this step
        advantages: [N, NUM_ACTIONS] computed advantages
        valid_mask: [N, NUM_ACTIONS] valid action mask
        is_traverser: [N] bool mask for traverser decision points
    """
    n_games = state.done.shape[0]

    # Get observation and valid actions
    obs = encode_state_for_current_player(state)
    valid_mask = get_valid_actions_mask(state)

    # Get strategy from network
    advantage_pred = network.apply({"params": params}, obs, training=False)
    strategy = regret_match(advantage_pred, valid_mask)

    # Identify traverser decision points
    is_traverser = state.current_player == traverser

    # For traverser nodes, we need to compute action values
    # For now, we use the network's advantage prediction as a proxy
    # (Full implementation would explore all actions recursively)
    advantages = advantage_pred

    # Sample action for stepping (opponent uses strategy, traverser uses uniform for exploration)
    rng_key, action_key = jrandom.split(rng_key)

    # For exploration: mix strategy with uniform
    explore_prob = 0.3
    uniform = valid_mask.astype(jnp.float32) / jnp.maximum(valid_mask.sum(axis=-1, keepdims=True), 1)
    mixed_strategy = jnp.where(
        is_traverser[:, None],
        (1 - explore_prob) * strategy + explore_prob * uniform,
        strategy
    )

    actions = sample_action_from_strategy(action_key, mixed_strategy)

    # Step game
    next_state = step(state, actions)

    return next_state, obs, advantages, valid_mask, is_traverser


def batch_traverse(
    network,
    params: dict,
    rng_key: Array,
    n_games: int,
    max_steps: int = 50,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Perform batch traversal to collect advantage samples.

    Simplified implementation that runs games to completion and
    uses network predictions as advantage targets (bootstrapped).

    For each game:
    1. Play to completion with strategy sampling
    2. At each traverser decision point, record (obs, advantages)
    3. Advantages come from network prediction (will improve over training)

    Args:
        network: AdvantageNetwork module
        params: Network parameters
        rng_key: PRNG key
        n_games: Number of parallel games
        max_steps: Maximum steps per game

    Returns:
        observations: [M, OBS_DIM] collected observations
        advantages: [M, NUM_ACTIONS] collected advantages
        valid_masks: [M, NUM_ACTIONS] valid action masks
    """
    # Initialize games
    rng_key, reset_key = jrandom.split(rng_key)
    state = reset(reset_key, n_games)

    # Alternate traverser (half P0, half P1)
    traverser = jnp.concatenate([
        jnp.zeros(n_games // 2, dtype=jnp.int32),
        jnp.ones(n_games - n_games // 2, dtype=jnp.int32)
    ])

    # Storage for collected samples
    all_obs = []
    all_advantages = []
    all_valid_masks = []

    # Run game loop
    for step_idx in range(max_steps):
        # Check if all games done
        if state.done.all():
            break

        # Get observations and advantages for current state
        rng_key, step_key = jrandom.split(rng_key)
        next_state, obs, advantages, valid_mask, is_traverser = batch_traverse_step(
            state, params, network, step_key, traverser
        )

        # Only collect samples for non-done traverser decision points
        collect_mask = is_traverser & ~state.done

        if collect_mask.any():
            # Extract samples where mask is True
            indices = jnp.where(collect_mask)[0]
            all_obs.append(np.array(obs[indices]))
            all_advantages.append(np.array(advantages[indices]))
            all_valid_masks.append(np.array(valid_mask[indices]))

        state = next_state

    # Concatenate all samples
    if all_obs:
        observations = np.concatenate(all_obs, axis=0)
        advantages = np.concatenate(all_advantages, axis=0)
        valid_masks = np.concatenate(all_valid_masks, axis=0)
    else:
        observations = np.zeros((0, OBS_DIM), dtype=np.float32)
        advantages = np.zeros((0, NUM_ACTIONS), dtype=np.float32)
        valid_masks = np.zeros((0, NUM_ACTIONS), dtype=np.bool_)

    return observations, advantages, valid_masks


def traverse_with_outcome_values(
    network,
    params: dict,
    rng_key: Array,
    n_games: int,
    max_steps: int = 50,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Traverse games and compute advantages using outcome values.

    This is a more accurate (but slower) implementation that:
    1. Plays games to completion
    2. Uses actual outcomes to estimate action values
    3. Computes advantages as action_value - expected_value

    The key insight is that for outcome sampling, we weight the
    terminal utility by importance sampling ratios.

    Args:
        network: AdvantageNetwork module
        params: Network parameters
        rng_key: PRNG key
        n_games: Number of parallel games
        max_steps: Maximum steps per game

    Returns:
        observations: [M, OBS_DIM] collected observations
        advantages: [M, NUM_ACTIONS] target advantages
        valid_masks: [M, NUM_ACTIONS] valid action masks
    """
    # Initialize games
    rng_key, reset_key = jrandom.split(rng_key)
    state = reset(reset_key, n_games)

    # Alternate traverser
    traverser = jnp.concatenate([
        jnp.zeros(n_games // 2, dtype=jnp.int32),
        jnp.ones(n_games - n_games // 2, dtype=jnp.int32)
    ])

    # Trajectory storage
    traj_obs = []
    traj_actions = []
    traj_probs = []
    traj_players = []
    traj_valid_masks = []

    # Run game loop
    for step_idx in range(max_steps):
        if state.done.all():
            break

        # Get observation, strategy, actions
        obs = encode_state_for_current_player(state)
        valid_mask = get_valid_actions_mask(state)

        advantage_pred = network.apply({"params": params}, obs, training=False)
        strategy = regret_match(advantage_pred, valid_mask)

        rng_key, action_key = jrandom.split(rng_key)
        actions = sample_action_from_strategy(action_key, strategy)

        # Record trajectory data
        action_probs = strategy[jnp.arange(n_games), actions]

        traj_obs.append(np.array(obs))
        traj_actions.append(np.array(actions))
        traj_probs.append(np.array(action_probs))
        traj_players.append(np.array(state.current_player))
        traj_valid_masks.append(np.array(valid_mask))

        # Step
        state = step(state, actions)

    # Get terminal rewards
    rewards = np.array(get_rewards(state))  # [N, 2]

    # Convert trajectory to arrays
    T = len(traj_obs)
    if T == 0:
        return (
            np.zeros((0, OBS_DIM), dtype=np.float32),
            np.zeros((0, NUM_ACTIONS), dtype=np.float32),
            np.zeros((0, NUM_ACTIONS), dtype=np.bool_),
        )

    traj_obs = np.stack(traj_obs, axis=1)  # [N, T, OBS_DIM]
    traj_actions = np.stack(traj_actions, axis=1)  # [N, T]
    traj_probs = np.stack(traj_probs, axis=1)  # [N, T]
    traj_players = np.stack(traj_players, axis=1)  # [N, T]
    traj_valid_masks = np.stack(traj_valid_masks, axis=1)  # [N, T, NUM_ACTIONS]

    # Compute advantages using outcome sampling
    # For each traverser decision point:
    # advantage[a] = (utility_if_took_a - baseline) * importance_weight

    all_obs = []
    all_advantages = []
    all_valid_masks = []

    traverser_np = np.array(traverser)

    for game_idx in range(n_games):
        trav = traverser_np[game_idx]
        utility = rewards[game_idx, trav]

        # Find traverser decision points
        for t in range(T):
            if traj_players[game_idx, t] != trav:
                continue

            obs = traj_obs[game_idx, t]
            action_taken = traj_actions[game_idx, t]
            valid_mask = traj_valid_masks[game_idx, t]

            # Compute importance weight: product of opponent probs / product of all probs
            # For simplicity, use uniform weight here
            importance_weight = 1.0

            # Get current strategy for this observation
            obs_batch = obs[None, :]
            valid_batch = valid_mask[None, :]
            adv_pred = network.apply({"params": params}, obs_batch, training=False)
            strategy = np.array(regret_match(jnp.array(adv_pred), jnp.array(valid_batch))[0])

            # Baseline is expected value under strategy
            # Approximate with utility (since we sampled this trajectory)
            baseline = utility * np.sum(strategy[valid_mask])

            # Advantage for taken action
            advantages = np.zeros(NUM_ACTIONS, dtype=np.float32)

            # Simple advantage: utility - baseline for taken action
            # For other actions, use negative of strategy-weighted baseline
            for a in range(NUM_ACTIONS):
                if valid_mask[a]:
                    if a == action_taken:
                        advantages[a] = (utility - baseline) * importance_weight
                    else:
                        # Counterfactual: what if we took action a instead?
                        # Approximate with negative contribution
                        advantages[a] = -strategy[a] * utility * importance_weight

            all_obs.append(obs)
            all_advantages.append(advantages)
            all_valid_masks.append(valid_mask)

    if all_obs:
        observations = np.stack(all_obs)
        advantages = np.stack(all_advantages)
        valid_masks = np.stack(all_valid_masks)
    else:
        observations = np.zeros((0, OBS_DIM), dtype=np.float32)
        advantages = np.zeros((0, NUM_ACTIONS), dtype=np.float32)
        valid_masks = np.zeros((0, NUM_ACTIONS), dtype=np.bool_)

    return observations, advantages, valid_masks
