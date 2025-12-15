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


def _make_batch_traverse_jit(network, max_steps: int):
    """Create JIT-compiled batch traversal with proper MCCFR advantage computation."""

    @partial(jax.jit, static_argnums=())
    def _batch_traverse_loop(params, state, traverser, rng_key):
        """Fully JIT-compiled game loop with outcome sampling MCCFR.

        Key fix: Compute counterfactual advantages using terminal rewards
        and importance weighting, not network predictions.
        """
        n_games = state.done.shape[0]

        # Pre-allocate trajectory storage
        traj_obs = jnp.zeros((n_games, max_steps, OBS_DIM), dtype=jnp.float32)
        traj_actions = jnp.zeros((n_games, max_steps), dtype=jnp.int32)
        traj_probs = jnp.ones((n_games, max_steps), dtype=jnp.float32)  # Action probabilities
        traj_players = jnp.zeros((n_games, max_steps), dtype=jnp.int32)
        traj_valid = jnp.zeros((n_games, max_steps, NUM_ACTIONS), dtype=jnp.bool_)
        traj_active = jnp.zeros((n_games, max_steps), dtype=jnp.bool_)  # Was game active at this step

        def body_fn(step_idx, carry):
            state, traj_obs, traj_actions, traj_probs, traj_players, traj_valid, traj_active, rng_key = carry

            # Split key for this step
            rng_key, step_key = jrandom.split(rng_key)

            # Get observation and valid actions
            obs = encode_state_for_current_player(state)
            valid_mask = get_valid_actions_mask(state)

            # Get strategy from network
            advantage_pred = network.apply({"params": params}, obs, training=False)
            strategy = regret_match(advantage_pred, valid_mask)

            # Identify traverser decision points
            is_traverser = state.current_player == traverser

            # Sample action (with exploration for traverser)
            rng_key, action_key = jrandom.split(rng_key)
            explore_prob = 0.3
            n_valid = jnp.maximum(valid_mask.sum(axis=-1, keepdims=True), 1)
            uniform = valid_mask.astype(jnp.float32) / n_valid
            mixed_strategy = jnp.where(
                is_traverser[:, None],
                (1 - explore_prob) * strategy + explore_prob * uniform,
                strategy
            )
            actions = sample_action_from_strategy(action_key, mixed_strategy)

            # Get probability of chosen action (from mixed strategy, not pure strategy)
            game_idx = jnp.arange(n_games)
            action_probs = mixed_strategy[game_idx, actions]

            # Track if game was active at this step
            active = ~state.done

            # Step game
            next_state = step(state, actions)

            # Store in trajectory buffers
            traj_obs = traj_obs.at[:, step_idx].set(obs)
            traj_actions = traj_actions.at[:, step_idx].set(actions)
            traj_probs = traj_probs.at[:, step_idx].set(jnp.where(active, action_probs, 1.0))
            traj_players = traj_players.at[:, step_idx].set(state.current_player)
            traj_valid = traj_valid.at[:, step_idx].set(valid_mask)
            traj_active = traj_active.at[:, step_idx].set(active)

            return (next_state, traj_obs, traj_actions, traj_probs, traj_players, traj_valid, traj_active, rng_key)

        # Run fixed number of steps (JAX requires static loop bounds)
        init_carry = (state, traj_obs, traj_actions, traj_probs, traj_players, traj_valid, traj_active, rng_key)
        final_carry = jax.lax.fori_loop(0, max_steps, body_fn, init_carry)
        final_state, traj_obs, traj_actions, traj_probs, traj_players, traj_valid, traj_active, _ = final_carry

        # Get terminal rewards
        rewards = get_rewards(final_state)  # [N, 2]

        return final_state, rewards, traj_obs, traj_actions, traj_probs, traj_players, traj_valid, traj_active, traverser

    return _batch_traverse_loop


def _compute_cfr_advantages(
    rewards: np.ndarray,
    traj_obs: np.ndarray,
    traj_actions: np.ndarray,
    traj_probs: np.ndarray,
    traj_players: np.ndarray,
    traj_valid: np.ndarray,
    traj_active: np.ndarray,
    traverser: np.ndarray,
    network,
    params: dict,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute proper counterfactual advantages using outcome sampling MCCFR.

    Vectorized implementation for performance.

    Args:
        rewards: [N, 2] terminal rewards for each player
        traj_obs: [N, T, OBS_DIM] observations at each step
        traj_actions: [N, T] actions taken
        traj_probs: [N, T] probability of action taken
        traj_players: [N, T] which player acted
        traj_valid: [N, T, NUM_ACTIONS] valid action masks
        traj_active: [N, T] whether game was active
        traverser: [N] which player we're computing advantages for
        network: AdvantageNetwork module
        params: Network parameters

    Returns:
        observations: [M, OBS_DIM] collected observations
        advantages: [M, NUM_ACTIONS] counterfactual advantages
        valid_masks: [M, NUM_ACTIONS] valid action masks
    """
    n_games, max_steps = traj_actions.shape

    # Get traverser utility
    traverser_utility = rewards[np.arange(n_games), traverser]  # [N]

    # Compute cumulative log probabilities for importance weighting
    log_probs = np.log(traj_probs + 1e-9)  # [N, T]

    # Opponent mask: 1 where opponent acted, 0 where traverser acted
    is_opponent = (traj_players != traverser[:, None]).astype(np.float32)  # [N, T]

    # Cumulative opponent reach (product of opponent's action probs up to each step)
    opponent_log_reach = np.cumsum(log_probs * is_opponent, axis=1)  # [N, T]

    # Shift opponent reach by 1 (reach at step t uses probs up to t-1)
    opponent_log_reach_shifted = np.zeros_like(opponent_log_reach)
    opponent_log_reach_shifted[:, 1:] = opponent_log_reach[:, :-1]

    # Total sample probability (product of all probs)
    total_log_prob = np.sum(log_probs * traj_active.astype(np.float32), axis=1)  # [N]

    # Identify all traverser decision points: [N, T]
    is_traverser_turn = traj_players == traverser[:, None]
    collect_mask = is_traverser_turn & traj_active  # [N, T]

    # Flatten to get all decision points
    flat_mask = collect_mask.ravel()  # [N*T]
    n_samples = flat_mask.sum()

    if n_samples == 0:
        return (
            np.zeros((0, OBS_DIM), dtype=np.float32),
            np.zeros((0, NUM_ACTIONS), dtype=np.float32),
            np.zeros((0, NUM_ACTIONS), dtype=np.bool_),
        )

    # Extract data for all traverser decision points at once
    all_obs = traj_obs.reshape(n_games * max_steps, -1)[flat_mask]  # [M, OBS_DIM]
    all_actions = traj_actions.ravel()[flat_mask]  # [M]
    all_valid = traj_valid.reshape(n_games * max_steps, -1)[flat_mask]  # [M, NUM_ACTIONS]

    # Broadcast utility to all timesteps then filter
    utility_broadcast = np.repeat(traverser_utility, max_steps)  # [N*T]
    all_utility = utility_broadcast[flat_mask]  # [M]

    # Importance weights
    opp_reach_flat = opponent_log_reach_shifted.ravel()[flat_mask]  # [M]
    total_prob_broadcast = np.repeat(total_log_prob, max_steps)  # [N*T]
    total_prob_flat = total_prob_broadcast[flat_mask]  # [M]
    importance_weight = np.exp(opp_reach_flat - total_prob_flat)
    importance_weight = np.clip(importance_weight, 0, 100)  # Clip for stability

    # Get current strategy for ALL observations at once (single batch network call)
    all_obs_jax = jnp.array(all_obs)
    all_valid_jax = jnp.array(all_valid)
    adv_pred = network.apply({"params": params}, all_obs_jax, training=False)
    strategy = np.array(regret_match(adv_pred, all_valid_jax))  # [M, NUM_ACTIONS]

    # Counterfactual regret computation
    baseline = 0.0  # Pure outcome sampling
    regret_contribution = (all_utility - baseline) * importance_weight  # [M]

    # Create action one-hot
    action_onehot = np.eye(NUM_ACTIONS, dtype=np.float32)[all_actions]  # [M, NUM_ACTIONS]

    # Regret for taken action: positive contribution
    regret_for_taken = action_onehot * regret_contribution[:, None]  # [M, NUM_ACTIONS]

    # Regret for other actions: negative, weighted by strategy
    regret_for_others = -strategy * regret_contribution[:, None]  # [M, NUM_ACTIONS]

    # Net advantage: taken action gets positive, others get negative (except taken)
    advantages = regret_for_taken + regret_for_others * (1 - action_onehot)

    # Mask invalid actions
    advantages = advantages * all_valid.astype(np.float32)

    return all_obs, advantages.astype(np.float32), all_valid


# Cache compiled functions
_TRAVERSE_CACHE = {}


def batch_traverse(
    network,
    params: dict,
    rng_key: Array,
    n_games: int,
    max_steps: int = 50,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Perform batch traversal with proper outcome sampling MCCFR.

    Key fix: Computes counterfactual advantages using terminal rewards
    and importance weighting, matching the tabular CFR implementation.

    Args:
        network: AdvantageNetwork module
        params: Network parameters
        rng_key: PRNG key
        n_games: Number of parallel games
        max_steps: Maximum steps per game

    Returns:
        observations: [M, OBS_DIM] collected observations
        advantages: [M, NUM_ACTIONS] counterfactual advantages
        valid_masks: [M, NUM_ACTIONS] valid action masks
    """
    # Get or create JIT-compiled function
    cache_key = (id(network), max_steps)
    if cache_key not in _TRAVERSE_CACHE:
        _TRAVERSE_CACHE[cache_key] = _make_batch_traverse_jit(network, max_steps)
    traverse_fn = _TRAVERSE_CACHE[cache_key]

    # Initialize games
    rng_key, reset_key = jrandom.split(rng_key)
    state = reset(reset_key, n_games)

    # Alternate traverser (half P0, half P1)
    traverser = jnp.concatenate([
        jnp.zeros(n_games // 2, dtype=jnp.int32),
        jnp.ones(n_games - n_games // 2, dtype=jnp.int32)
    ])

    # Run JIT-compiled game loop
    result = traverse_fn(params, state, traverser, rng_key)
    final_state, rewards, traj_obs, traj_actions, traj_probs, traj_players, traj_valid, traj_active, _ = result

    # Transfer trajectory data to CPU
    rewards_np = np.array(rewards)
    traj_obs_np = np.array(traj_obs)
    traj_actions_np = np.array(traj_actions)
    traj_probs_np = np.array(traj_probs)
    traj_players_np = np.array(traj_players)
    traj_valid_np = np.array(traj_valid)
    traj_active_np = np.array(traj_active)
    traverser_np = np.array(traverser)

    # Compute proper CFR advantages using terminal rewards
    return _compute_cfr_advantages(
        rewards_np,
        traj_obs_np,
        traj_actions_np,
        traj_probs_np,
        traj_players_np,
        traj_valid_np,
        traj_active_np,
        traverser_np,
        network,
        params,
    )


def traverse_with_outcome_values(
    network,
    params: dict,
    rng_key: Array,
    n_games: int,
    max_steps: int = 50,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Traverse games and compute advantages using outcome values.

    Optimized: keeps data on GPU during loop, single transfer at end.

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

    # Pre-allocate trajectory storage ON GPU
    traj_obs = jnp.zeros((n_games, max_steps, OBS_DIM), dtype=jnp.float32)
    traj_actions = jnp.zeros((n_games, max_steps), dtype=jnp.int32)
    traj_players = jnp.zeros((n_games, max_steps), dtype=jnp.int32)
    traj_valid_masks = jnp.zeros((n_games, max_steps, NUM_ACTIONS), dtype=jnp.bool_)
    traj_valid_step = jnp.zeros((n_games, max_steps), dtype=jnp.bool_)

    # Run game loop - keep everything on GPU
    actual_steps = 0
    for step_idx in range(max_steps):
        if state.done.all():
            break
        actual_steps = step_idx + 1

        # Get observation, strategy, actions
        obs = encode_state_for_current_player(state)
        valid_mask = get_valid_actions_mask(state)

        advantage_pred = network.apply({"params": params}, obs, training=False)
        strategy = regret_match(advantage_pred, valid_mask)

        rng_key, action_key = jrandom.split(rng_key)
        actions = sample_action_from_strategy(action_key, strategy)

        # Record trajectory data ON GPU (no CPU transfer)
        not_done = ~state.done
        traj_obs = traj_obs.at[:, step_idx].set(obs)
        traj_actions = traj_actions.at[:, step_idx].set(actions)
        traj_players = traj_players.at[:, step_idx].set(state.current_player)
        traj_valid_masks = traj_valid_masks.at[:, step_idx].set(valid_mask)
        traj_valid_step = traj_valid_step.at[:, step_idx].set(not_done)

        # Step
        state = step(state, actions)

    # Get terminal rewards
    rewards = np.array(get_rewards(state))  # [N, 2]
    T = actual_steps

    if T == 0:
        return (
            np.zeros((0, OBS_DIM), dtype=np.float32),
            np.zeros((0, NUM_ACTIONS), dtype=np.float32),
            np.zeros((0, NUM_ACTIONS), dtype=np.bool_),
        )

    # Single GPU→CPU transfer for all trajectory data
    traj_obs_np = np.array(traj_obs[:, :T])  # [N, T, OBS_DIM]
    traj_actions_np = np.array(traj_actions[:, :T])  # [N, T]
    traj_players_np = np.array(traj_players[:, :T])  # [N, T]
    traj_valid_np = np.array(traj_valid_masks[:, :T])  # [N, T, NUM_ACTIONS]
    traj_valid_step_np = np.array(traj_valid_step[:, :T])  # [N, T]
    traverser_np = np.array(traverser)  # [N]

    # Vectorized advantage computation (all NumPy now)
    # Create mask for traverser decision points: [N, T]
    is_traverser_turn = traj_players_np == traverser_np[:, None]
    collect_mask = is_traverser_turn & traj_valid_step_np  # [N, T]

    # Get traverser utilities: [N]
    traverser_utility = rewards[np.arange(n_games), traverser_np]

    # Flatten to get all traverser decision points
    flat_mask = collect_mask.ravel()  # [N*T]
    n_samples = flat_mask.sum()

    if n_samples == 0:
        return (
            np.zeros((0, OBS_DIM), dtype=np.float32),
            np.zeros((0, NUM_ACTIONS), dtype=np.float32),
            np.zeros((0, NUM_ACTIONS), dtype=np.bool_),
        )

    # Extract data for traverser decision points
    # Reshape to [N*T, ...] then filter
    flat_obs = traj_obs_np.reshape(n_games * T, OBS_DIM)[flat_mask]  # [M, OBS_DIM]
    flat_actions = traj_actions_np.ravel()[flat_mask]  # [M]
    flat_valid = traj_valid_np.reshape(n_games * T, NUM_ACTIONS)[flat_mask]  # [M, NUM_ACTIONS]

    # Get utility for each sample (broadcast game utility to all timesteps)
    utility_per_step = np.repeat(traverser_utility, T)  # [N*T]
    flat_utility = utility_per_step[flat_mask]  # [M]

    # Compute strategy for all observations at once (batch network call)
    flat_obs_jax = jnp.array(flat_obs)
    flat_valid_jax = jnp.array(flat_valid)
    adv_pred = network.apply({"params": params}, flat_obs_jax, training=False)
    strategy = np.array(regret_match(adv_pred, flat_valid_jax))  # [M, NUM_ACTIONS]

    # Compute advantages vectorized
    # baseline = utility * sum(strategy[valid])
    valid_strategy_sum = (strategy * flat_valid).sum(axis=1)  # [M]
    baseline = flat_utility * valid_strategy_sum  # [M]

    # Create action one-hot for taken actions
    action_onehot = np.eye(NUM_ACTIONS, dtype=np.float32)[flat_actions]  # [M, NUM_ACTIONS]

    # Advantage for taken action: (utility - baseline)
    # Advantage for other valid actions: -strategy[a] * utility
    taken_advantage = (flat_utility - baseline)[:, None] * action_onehot  # [M, NUM_ACTIONS]
    other_advantage = -strategy * flat_utility[:, None]  # [M, NUM_ACTIONS]

    # Combine: use taken_advantage where action was taken, other_advantage otherwise
    advantages = np.where(action_onehot > 0, taken_advantage, other_advantage)

    # Mask invalid actions
    advantages = advantages * flat_valid

    return flat_obs, advantages.astype(np.float32), flat_valid
