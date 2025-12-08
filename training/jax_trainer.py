"""JAX-accelerated training loop for poker RL."""

import logging
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from typing import Callable
import time

import jax
import jax.numpy as jnp
import jax.random as jrandom
import optax
from jax import Array
from rich.console import Console
from rich.table import Table
from tqdm import tqdm

from poker_jax import (
    GameState,
    reset,
    step,
    get_rewards,
    encode_state_for_current_player,
    get_valid_actions_mask,
    ACTION_FOLD,
    ACTION_CHECK,
    ACTION_CALL,
    ACTION_RAISE,
    ACTION_ALL_IN,
)
from poker_jax.network import (
    ActorCriticMLP,
    create_network,
    init_network,
    sample_action,
    count_parameters,
)
from poker_jax.encoding import OBS_DIM, get_valid_actions_from_obs
from training.jax_ppo import (
    PPOConfig,
    PPOMetrics,
    Trajectory,
    create_optimizer,
    ppo_update,
)
from training.logging import MetricsLogger


@dataclass
class JAXTrainingConfig:
    """Configuration for JAX-accelerated training."""

    # Parallelism (optimized for RTX 4090 24GB VRAM)
    num_parallel_games: int = 1536  # Games to run in parallel
    steps_per_game: int = 200  # Max steps per game (auto-resets)

    # Training loop
    total_steps: int = 1_000_000  # Total environment steps
    steps_per_update: int = 3072  # Steps between PPO updates

    # Evaluation
    eval_every: int = 10_000  # Evaluate every N steps
    eval_games: int = 100  # Games for evaluation

    # Checkpointing (save ~20 checkpoints for long runs)
    checkpoint_every: int = 500_000_000  # Every 500M steps
    checkpoint_dir: str = "models/jax/checkpoints"

    # Logging (log_every is in updates, not steps)
    log_every: int = 100  # Log every N updates (reduces CSV/TB storage)
    tensorboard_dir: str | None = "logs/jax"

    # Game settings
    starting_chips: int = 200
    small_blind: int = 1
    big_blind: int = 2


@dataclass
class TrainingState:
    """Mutable training state."""

    params: dict
    opt_state: optax.OptState
    rng_key: Array
    total_steps: int = 0
    total_updates: int = 0
    total_games: int = 0

    # Rolling metrics
    win_rate: float = 0.0
    avg_reward: float = 0.0


@dataclass
class TrainingMetrics:
    """Aggregated training metrics."""

    steps: int = 0
    games_completed: int = 0
    avg_reward: float = 0.0
    avg_game_length: float = 0.0

    # PPO metrics
    policy_loss: float = 0.0
    value_loss: float = 0.0
    entropy: float = 0.0
    approx_kl: float = 0.0

    # Performance
    steps_per_second: float = 0.0
    games_per_second: float = 0.0

    # Poker behavior metrics
    wins: int = 0
    losses: int = 0  # Track losses for accurate win rate
    total_pot_won: float = 0.0
    action_counts: dict = field(default_factory=lambda: {
        "fold": 0, "check": 0, "call": 0,
        "raise_33": 0, "raise_66": 0, "raise_100": 0, "raise_150": 0,
        "all_in": 0
    })


class JAXTrainer:
    """JAX-accelerated RL trainer for poker."""

    def __init__(
        self,
        network: ActorCriticMLP | None = None,
        ppo_config: PPOConfig | None = None,
        training_config: JAXTrainingConfig | None = None,
        seed: int = 42,
        console: Console | None = None,
    ) -> None:
        self.console = console or Console()
        self.ppo_config = ppo_config or PPOConfig()
        self.training_config = training_config or JAXTrainingConfig()

        # Initialize RNG
        self.rng_key = jrandom.PRNGKey(seed)

        # Create network
        if network is None:
            network = create_network("mlp")
        self.network = network

        # Initialize parameters
        self.rng_key, init_key = jrandom.split(self.rng_key)
        self.params = init_network(network, init_key, OBS_DIM)

        # Create optimizer
        self.optimizer = create_optimizer(self.ppo_config)
        self.opt_state = self.optimizer.init(self.params)

        # Logger (TensorBoard + CSV for plotting)
        self.logger: MetricsLogger | None = None
        self.file_logger: logging.Logger | None = None
        if self.training_config.tensorboard_dir:
            log_dir = Path(self.training_config.tensorboard_dir)
            self.logger = MetricsLogger(
                log_dir,
                use_tensorboard=True,
                use_csv=True,  # Enable CSV for plotting
            )
            # File logger for training progress
            log_file = log_dir / "training.log"
            file_handler = logging.FileHandler(log_file, mode="w")
            file_handler.setFormatter(
                logging.Formatter("%(asctime)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
            )
            self.file_logger = logging.getLogger(f"jax_trainer_{id(self)}")
            self.file_logger.setLevel(logging.INFO)
            self.file_logger.addHandler(file_handler)

        # Print info
        num_params = count_parameters(self.params)
        self.console.print(f"\n[bold blue]JAX Poker Trainer[/bold blue]")
        self.console.print(f"Network parameters: {num_params:,}")
        self.console.print(f"Parallel games: {self.training_config.num_parallel_games:,}")
        self.console.print(f"Steps per update: {self.training_config.steps_per_update:,}")

    @staticmethod
    @partial(jax.jit, static_argnums=(0,))
    def _collect_step(
        network: ActorCriticMLP,
        params: dict,
        state: GameState,
        rng_key: Array,
    ) -> tuple[GameState, Array, Array, Array, Array, Array, Array]:
        """Collect one step of experience from all parallel games.

        Returns:
            Tuple of (new_state, obs, actions, log_probs, values, rewards, dones)
        """
        n_games = state.done.shape[0]

        # Encode state for current player
        obs = encode_state_for_current_player(state)  # [N, obs_dim]

        # Get valid actions
        valid_mask = get_valid_actions_from_obs(obs)  # [N, 5]
        # Pad to 6 actions (index 0 is unused)
        valid_mask_full = jnp.concatenate([
            jnp.zeros((n_games, 1)),  # Index 0 unused
            valid_mask,
        ], axis=1)

        # Forward pass
        action_logits, bet_frac, values = network.apply(
            {"params": params}, obs, training=False
        )
        values = values.squeeze(-1)

        # Sample actions
        rng_key, sample_key = jrandom.split(rng_key)
        actions, log_probs = sample_action(sample_key, action_logits, valid_mask)

        # Convert to game actions (add 1 for offset)
        game_actions = actions + 1  # 0->1 (fold), 1->2 (check), etc.

        # Compute raise amounts for raise actions
        # Default: min-raise
        opp_idx = 1 - state.current_player
        game_idx = jnp.arange(n_games)
        opp_bet = state.bets[game_idx, opp_idx]
        raise_amounts = opp_bet + state.last_raise_amount

        # Step environment
        new_state = step(state, game_actions, raise_amounts)

        # Get rewards
        rewards = get_rewards(new_state)
        # Get reward for current player (before step)
        player_rewards = rewards[game_idx, state.current_player]

        # Done flags
        dones = new_state.done.astype(jnp.float32)

        return new_state, obs, actions, log_probs, values, player_rewards, dones, valid_mask

    def collect_rollout(
        self,
        num_steps: int,
    ) -> tuple[Trajectory, TrainingMetrics]:
        """Collect rollout data from parallel games.

        Args:
            num_steps: Number of steps to collect

        Returns:
            Tuple of (trajectory, metrics)
        """
        config = self.training_config
        n_games = config.num_parallel_games

        # Initialize games
        self.rng_key, reset_key = jrandom.split(self.rng_key)
        state = reset(
            reset_key, n_games,
            config.starting_chips, config.small_blind, config.big_blind
        )

        # Storage
        all_obs = []
        all_actions = []
        all_log_probs = []
        all_values = []
        all_rewards = []
        all_dones = []
        all_valid_masks = []

        total_rewards = 0.0
        games_completed = 0
        total_game_steps = 0

        # Poker behavior tracking
        wins = 0
        losses = 0
        total_pot_won = 0.0
        # v3: 9 actions (indices 1-8, 0 is ACTION_NONE)
        action_counts = {
            "fold": 0, "check": 0, "call": 0,
            "raise_33": 0, "raise_66": 0, "raise_100": 0, "raise_150": 0,
            "all_in": 0
        }
        # Map action index to name (ACTION_FOLD=1, ..., ACTION_ALL_IN=8)
        action_idx_to_name = {
            1: "fold", 2: "check", 3: "call",
            4: "raise_33", 5: "raise_66", 6: "raise_100", 7: "raise_150",
            8: "all_in"
        }

        start_time = time.time()

        for step_idx in range(num_steps):
            # Collect step
            self.rng_key, step_key = jrandom.split(self.rng_key)

            new_state, obs, actions, log_probs, values, rewards, dones, valid_mask = \
                self._collect_step(self.network, self.params, state, step_key)

            # Store
            all_obs.append(obs)
            all_actions.append(actions)
            all_log_probs.append(log_probs)
            all_values.append(values)
            all_rewards.append(rewards)
            all_dones.append(dones)
            all_valid_masks.append(valid_mask)

            # Track metrics
            completed = dones.sum()
            games_completed += int(completed)
            total_rewards += float(rewards.sum())
            total_game_steps += int(n_games - completed)

            # Track action distribution (v3: actions are 1-8)
            actions_np = jnp.asarray(actions)
            for idx, name in action_idx_to_name.items():
                action_counts[name] += int((actions_np == idx).sum())

            # Track wins and losses (only when game ends on this player's action)
            done_mask = dones > 0.5
            won_games = (rewards > 0) & done_mask
            lost_games = (rewards < 0) & done_mask
            wins += int(won_games.sum())
            losses += int(lost_games.sum())
            # Sum positive rewards for pot won tracking
            total_pot_won += float(jnp.where(won_games, rewards, 0.0).sum())

            # Reset completed games
            if completed > 0:
                self.rng_key, reset_key = jrandom.split(self.rng_key)
                fresh_state = reset(
                    reset_key, n_games,
                    config.starting_chips, config.small_blind, config.big_blind
                )
                # Keep unfinished games, replace finished ones
                def select_state(fresh, old):
                    done_mask = new_state.done
                    # Reshape done mask to broadcast with array dimensions
                    for _ in range(old.ndim - 1):
                        done_mask = done_mask[:, None]
                    return jnp.where(done_mask, fresh, old)

                state = jax.tree_util.tree_map(select_state, fresh_state, new_state)
            else:
                state = new_state

        elapsed = time.time() - start_time

        # Stack trajectory
        trajectory = Trajectory(
            obs=jnp.stack(all_obs),  # [T, N, obs_dim]
            actions=jnp.stack(all_actions),  # [T, N]
            log_probs=jnp.stack(all_log_probs),  # [T, N]
            values=jnp.stack(all_values),  # [T, N]
            rewards=jnp.stack(all_rewards),  # [T, N]
            dones=jnp.stack(all_dones),  # [T, N]
            valid_masks=jnp.stack(all_valid_masks),  # [T, N, 5]
        )

        # Compute metrics
        total_steps = num_steps * n_games
        metrics = TrainingMetrics(
            steps=total_steps,
            games_completed=games_completed,
            avg_reward=total_rewards / max(games_completed, 1),
            avg_game_length=total_game_steps / max(games_completed, 1),
            steps_per_second=total_steps / max(elapsed, 0.001),
            games_per_second=games_completed / max(elapsed, 0.001),
            # Poker behavior
            wins=wins,
            losses=losses,
            total_pot_won=total_pot_won,
            action_counts=action_counts,
        )

        return trajectory, metrics

    def train(
        self,
        callback: Callable[[int, TrainingMetrics, PPOMetrics], None] | None = None,
        show_progress: bool = True,
    ) -> TrainingMetrics:
        """Run the full training loop.

        Args:
            callback: Optional callback(steps, metrics, ppo_metrics) after each update
            show_progress: Show progress bar

        Returns:
            Final training metrics
        """
        config = self.training_config
        total_updates = config.total_steps // config.steps_per_update

        checkpoint_dir = Path(config.checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.console.print(f"\n[bold]Starting training...[/bold]")
        self.console.print(f"Total steps: {config.total_steps:,}")
        self.console.print(f"Updates: {total_updates:,}")

        iterator = range(total_updates)
        if show_progress:
            # mininterval=5.0 reduces terminal I/O overhead (update display max every 5 seconds)
            iterator = tqdm(iterator, desc="Training", unit="updates", mininterval=5.0)

        global_step = 0
        total_games = 0
        start_time = time.time()

        try:
            for update_idx in iterator:
                # Collect rollout
                steps_per_game = config.steps_per_update // config.num_parallel_games
                trajectory, collect_metrics = self.collect_rollout(steps_per_game)

                global_step += collect_metrics.steps
                total_games += collect_metrics.games_completed

                # PPO update
                self.rng_key, update_key = jrandom.split(self.rng_key)
                self.params, self.opt_state, ppo_metrics = ppo_update(
                    self.network,
                    self.params,
                    self.opt_state,
                    self.optimizer,
                    trajectory,
                    self.ppo_config,
                    update_key,
                )

                # Logging - only update progress bar every 100 updates to reduce terminal I/O
                if show_progress and (update_idx + 1) % 100 == 0:
                    iterator.set_postfix(
                        rew=f"{collect_metrics.avg_reward:.2f}",
                        pol=f"{ppo_metrics.policy_loss:.4f}",
                        val=f"{ppo_metrics.value_loss:.4f}",
                        ent=f"{ppo_metrics.entropy:.3f}",
                        sps=f"{collect_metrics.steps_per_second:.0f}",
                    )

                if self.logger and (update_idx + 1) % config.log_every == 0:
                    # Compute derived metrics
                    ac = collect_metrics.action_counts
                    total_actions = sum(ac.values())
                    # Win rate: when I ended the game, how often did I win?
                    decided_games = collect_metrics.wins + collect_metrics.losses
                    win_rate = collect_metrics.wins / max(decided_games, 1)
                    fold_rate = ac["fold"] / max(total_actions, 1)
                    # v3: Sum all raise types for aggression
                    raise_count = ac["raise_33"] + ac["raise_66"] + ac["raise_100"] + ac["raise_150"] + ac["all_in"]
                    call_count = ac["call"]
                    aggression = raise_count / max(call_count, 1)

                    self.logger.log(global_step, {
                        # Rewards
                        "reward/avg": collect_metrics.avg_reward,
                        "reward/games_completed": collect_metrics.games_completed,
                        # Loss
                        "loss/policy": ppo_metrics.policy_loss,
                        "loss/value": ppo_metrics.value_loss,
                        "loss/entropy": ppo_metrics.entropy,
                        # PPO diagnostics
                        "ppo/approx_kl": ppo_metrics.approx_kl,
                        "ppo/clip_fraction": ppo_metrics.clip_fraction,
                        # RL diagnostics
                        "rl/explained_variance": ppo_metrics.explained_variance,
                        "rl/grad_norm": ppo_metrics.grad_norm,
                        "rl/value_pred_error": ppo_metrics.value_pred_error,
                        # Performance
                        "perf/steps_per_second": collect_metrics.steps_per_second,
                        "perf/games_per_second": collect_metrics.games_per_second,
                        # Poker behavior
                        "poker/win_rate": win_rate,
                        "poker/avg_pot_won": collect_metrics.total_pot_won / max(collect_metrics.wins, 1),
                        "poker/fold_rate": fold_rate,
                        "poker/aggression": aggression,
                        # v3: All 8 action types
                        "poker/action_fold": ac["fold"] / max(total_actions, 1),
                        "poker/action_check": ac["check"] / max(total_actions, 1),
                        "poker/action_call": ac["call"] / max(total_actions, 1),
                        "poker/action_raise_33": ac["raise_33"] / max(total_actions, 1),
                        "poker/action_raise_66": ac["raise_66"] / max(total_actions, 1),
                        "poker/action_raise_100": ac["raise_100"] / max(total_actions, 1),
                        "poker/action_raise_150": ac["raise_150"] / max(total_actions, 1),
                        "poker/action_allin": ac["all_in"] / max(total_actions, 1),
                    })

                # File logging (persist progress for long runs) - reduced frequency
                if self.file_logger and update_idx % 1000 == 0:
                    decided_games = collect_metrics.wins + collect_metrics.losses
                    win_rate = collect_metrics.wins / max(decided_games, 1)
                    total_actions = sum(collect_metrics.action_counts.values())
                    self.file_logger.info(
                        f"step={global_step:>8} | rew={collect_metrics.avg_reward:>7.3f} | "
                        f"win={win_rate:>5.1%} | pol={ppo_metrics.policy_loss:>7.4f} | "
                        f"ev={ppo_metrics.explained_variance:>5.2f} | sps={collect_metrics.steps_per_second:>5.0f}"
                    )
                    # Debug: action breakdown and game outcomes (v3: 8 action types)
                    ac = collect_metrics.action_counts
                    total_raises = ac['raise_33'] + ac['raise_66'] + ac['raise_100'] + ac['raise_150']
                    self.file_logger.info(
                        f"  actions: F={ac['fold']:>4} Ch={ac['check']:>4} Ca={ac['call']:>4} "
                        f"R33={ac['raise_33']:>3} R66={ac['raise_66']:>3} R100={ac['raise_100']:>3} R150={ac['raise_150']:>3} A={ac['all_in']:>4} | "
                        f"games: {collect_metrics.games_completed} done, {collect_metrics.wins}W/{collect_metrics.losses}L"
                    )

                # Checkpointing
                if global_step % config.checkpoint_every == 0:
                    ckpt_path = checkpoint_dir / f"step_{global_step:08d}.pkl"
                    self.save_checkpoint(ckpt_path)

                # Callback
                if callback:
                    callback(global_step, collect_metrics, ppo_metrics)

        except KeyboardInterrupt:
            self.console.print("\n[yellow]Training interrupted.[/yellow]")

        # Final checkpoint
        final_path = checkpoint_dir.parent / "final.pkl"
        self.save_checkpoint(final_path)

        elapsed = time.time() - start_time
        self.console.print(f"\n[green]Training complete![/green]")
        self.console.print(f"Total time: {elapsed / 60:.1f} minutes")
        self.console.print(f"Total steps: {global_step:,}")
        self.console.print(f"Total games: {total_games:,}")
        self.console.print(f"Avg steps/sec: {global_step / elapsed:.0f}")

        return TrainingMetrics(
            steps=global_step,
            games_completed=total_games,
            steps_per_second=global_step / elapsed,
        )

    def save_checkpoint(self, path: Path | str) -> None:
        """Save checkpoint."""
        import pickle

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        checkpoint = {
            "params": self.params,
            "opt_state": self.opt_state,
            "ppo_config": self.ppo_config,
            "training_config": self.training_config,
        }

        with open(path, "wb") as f:
            pickle.dump(checkpoint, f)

    def load_checkpoint(self, path: Path | str) -> None:
        """Load checkpoint."""
        import pickle

        with open(path, "rb") as f:
            checkpoint = pickle.load(f)

        self.params = checkpoint["params"]
        self.opt_state = checkpoint["opt_state"]


def create_jax_trainer(
    seed: int = 42,
    num_parallel_games: int = 1024,
    total_steps: int = 1_000_000,
    learning_rate: float = 3e-4,
    tensorboard_dir: str | None = "logs/jax",
) -> JAXTrainer:
    """Factory function to create a JAX trainer.

    Args:
        seed: Random seed
        num_parallel_games: Number of parallel games
        total_steps: Total training steps
        learning_rate: Learning rate
        tensorboard_dir: TensorBoard log directory

    Returns:
        Configured JAXTrainer
    """
    ppo_config = PPOConfig(learning_rate=learning_rate)
    training_config = JAXTrainingConfig(
        num_parallel_games=num_parallel_games,
        total_steps=total_steps,
        tensorboard_dir=tensorboard_dir,
    )

    return JAXTrainer(
        ppo_config=ppo_config,
        training_config=training_config,
        seed=seed,
    )
