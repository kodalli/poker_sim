"""JAX-optimized CFR+ training for poker.

This module implements Monte Carlo CFR (MCCFR) with outcome sampling,
fully vectorized for GPU acceleration.

The algorithm:
1. Sample batch of game trajectories
2. For each trajectory, compute counterfactual values
3. Update regret sums using CFR+ (clip negative regrets)
4. Accumulate strategy sums for averaging
"""

import argparse
import time
from dataclasses import dataclass
from functools import partial
from pathlib import Path

import jax
import jax.numpy as jnp
import jax.random as jrandom
from jax import Array

from poker_jax.state import (
    GameState,
    NUM_ACTIONS,
    ROUND_PREFLOP,
    ROUND_SHOWDOWN,
    ACTION_FOLD,
    ACTION_CHECK,
    ACTION_CALL,
    MAX_HISTORY,
)
from poker_jax.game import reset, step, get_rewards
from poker_jax.state import get_valid_actions_mask
from poker_jax.encoding import encode_state_for_current_player

from .abstraction import (
    compute_info_key_batch_v2,
    info_key_to_index,
    MAX_INFO_SETS,
    NUM_BUCKETS_PREFLOP,
)
from .exploitability import compute_exploitability, analyze_strategy_quality

# Hand strength index in observation
HAND_STRENGTH_OFFSET = 104 + 260 + 4 + 2 + 6 + 9  # = 385
NORMALIZED_STRENGTH_IDX = HAND_STRENGTH_OFFSET + 10  # = 395


@dataclass
class CFRState:
    """State for CFR training."""
    regret_sum: Array      # [MAX_INFO_SETS, NUM_ACTIONS]
    strategy_sum: Array    # [MAX_INFO_SETS, NUM_ACTIONS]
    iteration: int


def create_cfr_state() -> CFRState:
    """Create initial CFR state."""
    return CFRState(
        regret_sum=jnp.zeros((MAX_INFO_SETS, NUM_ACTIONS), dtype=jnp.float32),
        strategy_sum=jnp.zeros((MAX_INFO_SETS, NUM_ACTIONS), dtype=jnp.float32),
        iteration=0,
    )


@jax.jit
def regret_match(regrets: Array) -> Array:
    """Convert regrets to strategy using regret matching.

    Args:
        regrets: [N, NUM_ACTIONS] regret values

    Returns:
        [N, NUM_ACTIONS] strategy probabilities (normalized positive regrets)
    """
    positive = jnp.maximum(regrets, 0)
    total = positive.sum(axis=-1, keepdims=True)
    # If all regrets are non-positive, use uniform
    uniform = jnp.ones_like(positive) / NUM_ACTIONS
    strategy = jnp.where(total > 0, positive / (total + 1e-9), uniform)
    return strategy


@jax.jit
def get_strategy_from_regrets(regret_sum: Array, info_indices: Array) -> Array:
    """Get current strategy for given info sets.

    Args:
        regret_sum: [MAX_INFO_SETS, NUM_ACTIONS] cumulative regrets
        info_indices: [N] info set indices

    Returns:
        [N, NUM_ACTIONS] strategy probabilities
    """
    regrets = regret_sum[info_indices]  # [N, NUM_ACTIONS]
    return regret_match(regrets)


@jax.jit
def sample_action_from_strategy(
    rng_key: Array,
    strategy: Array,
    valid_mask: Array,
) -> Array:
    """Sample actions from strategy, respecting valid action mask.

    Args:
        rng_key: PRNG key
        strategy: [N, NUM_ACTIONS] strategy probabilities
        valid_mask: [N, NUM_ACTIONS] valid action mask

    Returns:
        [N] sampled actions
    """
    # Mask invalid actions
    masked = jnp.where(valid_mask, strategy, 0.0)
    # Renormalize
    total = masked.sum(axis=-1, keepdims=True)
    probs = masked / (total + 1e-9)
    # Handle case where no valid actions (shouldn't happen in practice)
    probs = jnp.where(total > 0, probs, valid_mask.astype(jnp.float32) / (valid_mask.sum(axis=-1, keepdims=True) + 1e-9))

    # Sample from categorical
    return jax.random.categorical(rng_key, jnp.log(probs + 1e-9))


@jax.jit
def play_one_step(
    carry: tuple,
    _: None,
) -> tuple:
    """Play one step in the game for all trajectories.

    Args:
        carry: (state, regret_sum, rng_key, trajectory_data)
        _: unused (for scan)

    Returns:
        Updated carry and step data for trajectory
    """
    state, regret_sum, rng_key, traj_info_keys, traj_actions, traj_probs, traj_players, step_idx = carry

    # Split RNG
    rng_key, action_key = jrandom.split(rng_key)

    # Get valid actions
    valid_mask = get_valid_actions_mask(state)

    # Encode current state
    obs = encode_state_for_current_player(state)

    # Get hand strength from observation
    normalized_strength = obs[:, NORMALIZED_STRENGTH_IDX]

    # Get current player's hole cards
    n_games = state.done.shape[0]
    game_idx = jnp.arange(n_games)
    player_idx = state.current_player
    hole_cards = state.hole_cards[game_idx, player_idx, :]

    # Compute info set indices using V2 abstraction (finer buckets + pot-relative history)
    info_keys = compute_info_key_batch_v2(
        hole_cards=hole_cards,
        street=state.round,
        normalized_strength=normalized_strength,
        pot=state.pot,
        bets=state.bets,
        current_player=state.current_player,
        button=state.button,
        starting_chips=jnp.full(n_games, 200.0),  # Standard starting stack
        action_history=state.action_history,
        history_len=state.history_len,
    )
    info_indices = info_key_to_index(info_keys)

    # Get strategy and sample actions
    strategy = get_strategy_from_regrets(regret_sum, info_indices)
    actions = sample_action_from_strategy(action_key, strategy, valid_mask)

    # Get probability of chosen action
    action_probs = strategy[game_idx, actions]

    # Record trajectory data (only for non-done games)
    not_done = ~state.done
    traj_info_keys = traj_info_keys.at[:, step_idx].set(
        jnp.where(not_done, info_indices, traj_info_keys[:, step_idx])
    )
    traj_actions = traj_actions.at[:, step_idx].set(
        jnp.where(not_done, actions, traj_actions[:, step_idx])
    )
    traj_probs = traj_probs.at[:, step_idx].set(
        jnp.where(not_done, action_probs, traj_probs[:, step_idx])
    )
    traj_players = traj_players.at[:, step_idx].set(
        jnp.where(not_done, player_idx, traj_players[:, step_idx])
    )

    # Step the game
    new_state = step(state, actions)

    return (new_state, regret_sum, rng_key, traj_info_keys, traj_actions, traj_probs, traj_players, step_idx + 1), None


@partial(jax.jit, static_argnums=(2, 3))
def simulate_trajectories(
    rng_key: Array,
    regret_sum: Array,
    n_games: int,
    max_steps: int = 50,
) -> tuple:
    """Simulate game trajectories using current strategy.

    Args:
        rng_key: PRNG key
        regret_sum: [MAX_INFO_SETS, NUM_ACTIONS] cumulative regrets
        n_games: Number of parallel games
        max_steps: Maximum steps per game

    Returns:
        Tuple of (final_state, trajectory_data)
    """
    # Initialize games
    rng_key, reset_key = jrandom.split(rng_key)
    state = reset(reset_key, n_games)

    # Trajectory storage
    traj_info_keys = jnp.zeros((n_games, max_steps), dtype=jnp.int32)
    traj_actions = jnp.zeros((n_games, max_steps), dtype=jnp.int32)
    traj_probs = jnp.ones((n_games, max_steps), dtype=jnp.float32)
    traj_players = jnp.zeros((n_games, max_steps), dtype=jnp.int32)

    # Run game loop
    init_carry = (state, regret_sum, rng_key, traj_info_keys, traj_actions, traj_probs, traj_players, 0)

    # Use fori_loop instead of scan for simpler state management
    def step_fn(i, carry):
        return play_one_step(carry, None)[0]

    final_carry = jax.lax.fori_loop(0, max_steps, step_fn, init_carry)
    final_state, _, _, traj_info_keys, traj_actions, traj_probs, traj_players, _ = final_carry

    return final_state, (traj_info_keys, traj_actions, traj_probs, traj_players)


@jax.jit
def compute_regret_updates(
    final_state: GameState,
    traj_info_keys: Array,
    traj_actions: Array,
    traj_probs: Array,
    traj_players: Array,
    traverser: int,
    regret_sum: Array,
    max_steps: int = 50,
) -> Array:
    """Compute regret updates for a batch of trajectories.

    Uses outcome sampling: weight by reach probability and
    counterfactual value estimation.

    Args:
        final_state: Final game state
        traj_info_keys: [N, max_steps] info set indices per step
        traj_actions: [N, max_steps] actions taken
        traj_probs: [N, max_steps] action probabilities
        traj_players: [N, max_steps] player who acted
        traverser: Player we're computing regrets for (0 or 1)
        regret_sum: Current regret sum table
        max_steps: Maximum steps in trajectory

    Returns:
        [MAX_INFO_SETS, NUM_ACTIONS] regret updates to add
    """
    n_games = final_state.done.shape[0]

    # Get terminal rewards
    rewards = get_rewards(final_state)
    traverser_reward = rewards[:, traverser]  # [N]

    # Initialize regret update accumulator
    regret_updates = jnp.zeros((MAX_INFO_SETS, NUM_ACTIONS), dtype=jnp.float32)

    # For outcome sampling, regret for action a at info set h is:
    # r(h, a) = (u(a) - v(h)) * pi_{-i}(h) / pi(h, a)
    # where:
    # - u(a) is the utility when taking action a (= terminal reward if on this path)
    # - v(h) is expected value at h (approximately terminal reward weighted by strategy)
    # - pi_{-i}(h) is opponent reach probability
    # - pi(h, a) is probability of reaching this (h, a) pair

    # Simplified outcome sampling:
    # For the sampled action path, update regrets based on terminal outcome
    # regret[h][a_taken] += (utility - baseline) / sample_prob

    # Compute reach probabilities
    # For traverser: product of traverser's action probs
    # For opponent: product of opponent's action probs
    is_traverser = (traj_players == traverser).astype(jnp.float32)
    is_opponent = (traj_players == (1 - traverser)).astype(jnp.float32)

    # Log probs for numerical stability
    log_probs = jnp.log(traj_probs + 1e-9)

    # Traverser reach (product of traverser's probs along path)
    traverser_log_reach = jnp.cumsum(log_probs * is_traverser, axis=1)
    # Opponent reach (product of opponent's probs along path)
    opponent_log_reach = jnp.cumsum(log_probs * is_opponent, axis=1)

    # Sample probability (product of all probs)
    total_log_prob = jnp.sum(log_probs, axis=1, keepdims=True)

    # For each step where traverser acted, compute regret contribution
    def update_step_regrets(step_idx, regret_acc):
        # Get data for this step
        info_keys = traj_info_keys[:, step_idx]
        actions = traj_actions[:, step_idx]
        players = traj_players[:, step_idx]

        # Only update for traverser's actions
        is_traverser_action = players == traverser

        # Importance weight: opponent_reach / sample_prob
        # In log space: opponent_log_reach - total_log_prob
        step_opponent_reach = jnp.where(
            step_idx > 0,
            opponent_log_reach[:, step_idx - 1],
            jnp.zeros(n_games)
        )
        importance_weight = jnp.exp(step_opponent_reach - total_log_prob.squeeze())
        importance_weight = jnp.clip(importance_weight, 0, 100)  # Clip for stability

        # Regret update for taken action: reward * importance_weight
        # This is a simplified version - full CFR would compute counterfactual values
        # for all actions, but outcome sampling approximates this

        # Get current strategy at this info set
        strategy = get_strategy_from_regrets(regret_sum, info_keys)

        # Baseline is expected value under current strategy (approximated by reward * sum of probs)
        # For simplicity, use 0 as baseline (pure outcome sampling)
        baseline = 0.0

        # Update for taken action
        regret_contribution = (traverser_reward - baseline) * importance_weight

        # Distribute regret across actions
        # For taken action: positive contribution
        # For other actions: negative contribution (weighted by strategy)
        action_onehot = jax.nn.one_hot(actions, NUM_ACTIONS)
        regret_for_taken = action_onehot * regret_contribution[:, None]
        regret_for_others = -strategy * regret_contribution[:, None]

        # Net regret: taken action gets positive, others get negative
        # regret[a] = regret_contribution if a == taken else -strategy[a] * regret_contribution
        net_regret = regret_for_taken + regret_for_others * (1 - action_onehot)

        # Mask to only update traverser's info sets
        update_mask = is_traverser_action[:, None]
        net_regret = net_regret * update_mask

        # Scatter-add to regret accumulator
        # This is a bit tricky in JAX - we need to handle duplicates
        regret_acc = regret_acc.at[info_keys].add(net_regret)

        return regret_acc

    # Apply updates for each step
    regret_updates = jax.lax.fori_loop(0, max_steps, update_step_regrets, regret_updates)

    return regret_updates


@partial(jax.jit, static_argnums=(2, 3))
def cfr_iteration(
    cfr_state: tuple,
    rng_key: Array,
    batch_size: int = 4096,
    max_steps: int = 50,
) -> tuple:
    """Run one CFR iteration.

    Args:
        cfr_state: Tuple of (regret_sum, strategy_sum, iteration)
        rng_key: PRNG key
        batch_size: Number of games per iteration
        max_steps: Maximum steps per game

    Returns:
        Updated CFR state
    """
    regret_sum, strategy_sum, iteration = cfr_state

    # Simulate trajectories
    rng_key, sim_key = jrandom.split(rng_key)
    final_state, traj_data = simulate_trajectories(sim_key, regret_sum, batch_size, max_steps)
    traj_info_keys, traj_actions, traj_probs, traj_players = traj_data

    # Compute regret updates for both players
    regret_update_p0 = compute_regret_updates(
        final_state, traj_info_keys, traj_actions, traj_probs, traj_players,
        traverser=0, regret_sum=regret_sum, max_steps=max_steps
    )
    regret_update_p1 = compute_regret_updates(
        final_state, traj_info_keys, traj_actions, traj_probs, traj_players,
        traverser=1, regret_sum=regret_sum, max_steps=max_steps
    )

    # Apply CFR+ update: clip negative regrets
    new_regret_sum = regret_sum + regret_update_p0 + regret_update_p1
    new_regret_sum = jnp.maximum(new_regret_sum, 0)  # CFR+ clipping

    # Update strategy sum (linear weighting for CFR+)
    # Weight by iteration number for better convergence
    weight = iteration + 1
    current_strategy = regret_match(new_regret_sum)
    new_strategy_sum = strategy_sum + current_strategy * weight

    return (new_regret_sum, new_strategy_sum, iteration + 1)


def compute_strategy_metrics(strategy_sum: Array) -> dict:
    """Compute metrics about the learned strategy.

    Args:
        strategy_sum: [MAX_INFO_SETS, NUM_ACTIONS] cumulative strategy

    Returns:
        Dictionary of metrics
    """
    avg_strategy = strategy_sum / (strategy_sum.sum(axis=-1, keepdims=True) + 1e-9)

    # Count active info sets (non-zero strategy sum)
    active_mask = strategy_sum.sum(axis=-1) > 0
    n_active = int(active_mask.sum())

    # Count non-uniform strategies (deviation from 1/9)
    uniform = 1.0 / NUM_ACTIONS
    deviation = jnp.abs(avg_strategy - uniform).sum(axis=-1)
    n_non_uniform = int((deviation > 0.01).sum())

    # Strategy quality metrics (for active info sets only)
    active_strategies = avg_strategy[active_mask]
    if len(active_strategies) > 0:
        # Pure vs mixed strategies
        n_pure = int((active_strategies.max(axis=-1) > 0.9).sum())
        n_mixed = n_active - n_pure

        # Action distribution
        avg_action_dist = active_strategies.mean(axis=0)

        # Aggression (raises + all-in)
        aggression = float(avg_action_dist[4:9].sum())

        # Passivity (check + call)
        passivity = float(avg_action_dist[2] + avg_action_dist[3])

        # Fold rate
        fold_rate = float(avg_action_dist[1])
    else:
        n_pure = n_mixed = 0
        aggression = passivity = fold_rate = 0.0
        avg_action_dist = jnp.zeros(NUM_ACTIONS)

    return {
        'active_infos': n_active,
        'non_uniform': n_non_uniform,
        'pure_strategies': n_pure,
        'mixed_strategies': n_mixed,
        'aggression': aggression,
        'passivity': passivity,
        'fold_rate': fold_rate,
    }


def train_cfr(
    iterations: int = 100_000,
    batch_size: int = 4096,
    max_steps: int = 50,
    checkpoint_every: int = 10_000,
    output_dir: str = "models/cfr",
    seed: int = 42,
    log_metrics_every: int = 5000,
    compute_exploitability_every: int = 0,
) -> Array:
    """Train CFR and return final average strategy.

    Args:
        iterations: Number of CFR iterations
        batch_size: Games per iteration
        max_steps: Max steps per game
        checkpoint_every: Save checkpoint every N iterations
        output_dir: Directory for outputs
        seed: Random seed
        log_metrics_every: Log detailed metrics every N iterations
        compute_exploitability_every: Compute exploitability every N iterations (0 to disable)

    Returns:
        [MAX_INFO_SETS, NUM_ACTIONS] average strategy
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Initialize
    rng_key = jrandom.PRNGKey(seed)
    regret_sum = jnp.zeros((MAX_INFO_SETS, NUM_ACTIONS), dtype=jnp.float32)
    strategy_sum = jnp.zeros((MAX_INFO_SETS, NUM_ACTIONS), dtype=jnp.float32)

    print(f"Starting CFR training: {iterations} iterations, batch_size={batch_size}")
    print(f"Total game samples: {iterations * batch_size:,}")

    start_time = time.time()
    metrics_history = []

    for i in range(iterations):
        rng_key, iter_key = jrandom.split(rng_key)

        # Run iteration
        regret_sum, strategy_sum, _ = cfr_iteration(
            (regret_sum, strategy_sum, i),
            iter_key,
            batch_size,
            max_steps,
        )

        # Basic logging
        if (i + 1) % 1000 == 0:
            current_time = time.time()
            elapsed = current_time - start_time
            iter_per_sec = (i + 1) / elapsed
            games_per_sec = iter_per_sec * batch_size

            metrics = compute_strategy_metrics(strategy_sum)

            print(
                f"Iter {i + 1:,}/{iterations:,} | "
                f"{elapsed:.1f}s | "
                f"{iter_per_sec:.1f} iter/s | "
                f"{games_per_sec:,.0f} games/s | "
                f"Active: {metrics['active_infos']:,} | "
                f"Non-uniform: {metrics['non_uniform']:,}"
            )

        # Detailed metrics logging
        if log_metrics_every > 0 and (i + 1) % log_metrics_every == 0:
            metrics = compute_strategy_metrics(strategy_sum)
            metrics['iteration'] = i + 1
            metrics['elapsed_sec'] = time.time() - start_time
            metrics_history.append(metrics)

            print(f"\n=== Detailed Metrics (Iter {i + 1:,}) ===")
            print(f"  Active info sets: {metrics['active_infos']:,}")
            print(f"  Non-uniform strategies: {metrics['non_uniform']:,}")
            print(f"  Pure strategies (>90%): {metrics['pure_strategies']:,}")
            print(f"  Mixed strategies: {metrics['mixed_strategies']:,}")
            print(f"  Aggression (raises): {metrics['aggression']:.1%}")
            print(f"  Passivity (check/call): {metrics['passivity']:.1%}")
            print(f"  Fold rate: {metrics['fold_rate']:.1%}")
            print()

        # Exploitability estimation
        if compute_exploitability_every > 0 and (i + 1) % compute_exploitability_every == 0:
            print(f"\n=== Exploitability Estimation (Iter {i + 1:,}) ===")
            avg_strategy = strategy_sum / (strategy_sum.sum(axis=-1, keepdims=True) + 1e-9)
            rng_key, exploit_key = jrandom.split(rng_key)
            exploit_result = compute_exploitability(
                avg_strategy, exploit_key,
                n_samples=5000,
                batch_size=512,
            )
            print(f"  Exploitability: {exploit_result['bb100']:.2f} BB/100")
            print(f"  Mean gap: {exploit_result['mean_chips']:.4f} chips")
            print()

            # Add to metrics history
            if metrics_history and metrics_history[-1]['iteration'] == i + 1:
                metrics_history[-1]['exploitability_bb100'] = exploit_result['bb100']
                metrics_history[-1]['exploitability_chips'] = exploit_result['mean_chips']

        # Checkpoint
        if checkpoint_every > 0 and (i + 1) % checkpoint_every == 0:
            checkpoint_path = output_path / f"strategy_iter{i + 1}.npy"
            avg_strategy = strategy_sum / (strategy_sum.sum(axis=-1, keepdims=True) + 1e-9)
            jnp.save(str(checkpoint_path), avg_strategy)
            print(f"Saved checkpoint to {checkpoint_path}")

    # Compute final average strategy
    avg_strategy = strategy_sum / (strategy_sum.sum(axis=-1, keepdims=True) + 1e-9)

    # Final metrics
    final_metrics = compute_strategy_metrics(strategy_sum)
    print("\n" + "=" * 50)
    print("=== FINAL CFR TRAINING RESULTS ===")
    print("=" * 50)
    print(f"Total iterations: {iterations:,}")
    print(f"Total game samples: {iterations * batch_size:,}")
    print(f"Training time: {time.time() - start_time:.1f}s")
    print(f"Active info sets: {final_metrics['active_infos']:,}")
    print(f"Non-uniform strategies: {final_metrics['non_uniform']:,}")
    print(f"Pure strategies: {final_metrics['pure_strategies']:,}")
    print(f"Mixed strategies: {final_metrics['mixed_strategies']:,}")
    print(f"Aggression: {final_metrics['aggression']:.1%}")
    print(f"Passivity: {final_metrics['passivity']:.1%}")
    print(f"Fold rate: {final_metrics['fold_rate']:.1%}")

    # Skip exploitability for now (can compute separately with exploitability.py)
    print("=" * 50)

    # Save final strategy
    final_path = output_path / "strategy_final.npy"
    jnp.save(str(final_path), avg_strategy)
    print(f"\nSaved final strategy to {final_path}")

    # Save metrics history
    if metrics_history:
        import json
        metrics_path = output_path / "metrics.json"
        with open(metrics_path, 'w') as f:
            json.dump(metrics_history, f, indent=2)
        print(f"Saved metrics to {metrics_path}")

    return avg_strategy


def main():
    parser = argparse.ArgumentParser(description="Train CFR for poker")
    parser.add_argument("--iterations", type=int, default=100_000, help="CFR iterations")
    parser.add_argument("--batch-size", type=int, default=4096, help="Games per iteration")
    parser.add_argument("--max-steps", type=int, default=50, help="Max steps per game")
    parser.add_argument("--checkpoint-every", type=int, default=25_000, help="Checkpoint interval")
    parser.add_argument("--output-dir", type=str, default="models/cfr_v3", help="Output directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--log-metrics-every", type=int, default=10_000, help="Log metrics interval")
    parser.add_argument("--exploitability-every", type=int, default=25_000, help="Compute exploitability interval (0 to disable)")

    args = parser.parse_args()

    train_cfr(
        iterations=args.iterations,
        batch_size=args.batch_size,
        max_steps=args.max_steps,
        checkpoint_every=args.checkpoint_every,
        output_dir=args.output_dir,
        seed=args.seed,
        log_metrics_every=args.log_metrics_every,
        compute_exploitability_every=args.exploitability_every,
    )


if __name__ == "__main__":
    main()
