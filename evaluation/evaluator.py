"""Evaluation engine for benchmarking trained models against opponents."""

import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable
import time

import jax
import jax.numpy as jnp
import jax.random as jrandom
from jax import Array
from rich.console import Console
from rich.table import Table

from poker_jax.state import GameState, NUM_ACTIONS
from poker_jax.game import reset, step, get_rewards
from poker_jax.encoding import (
    encode_state_for_current_player, get_valid_actions_from_obs, OBS_DIM,
    encode_opponent_action_from_state, OPPONENT_ACTION_DIM
)
from poker_jax.network import ActorCriticMLP, create_network, sample_action, OPPONENT_LSTM_HIDDEN

from evaluation.opponents import OPPONENT_TYPES, NEEDS_OBS
from evaluation.metrics import compute_bb_per_100, compute_confidence_interval, is_statistically_significant


@dataclass
class EvalConfig:
    """Configuration for evaluation."""
    num_games: int = 10_000
    starting_chips: int = 200
    small_blind: int = 1
    big_blind: int = 2

    # Position testing
    test_both_positions: bool = True  # Test as both player 0 and 1

    # Output
    output_csv: str | None = None


@dataclass
class EvalResults:
    """Results from evaluation run."""
    opponent_type: str
    total_games: int

    # Win statistics
    wins: int = 0
    losses: int = 0
    ties: int = 0

    # Chip statistics
    total_chips_won: float = 0.0
    total_chips_lost: float = 0.0

    # Action statistics
    action_counts: dict = field(default_factory=dict)

    # Per-position results (if tested)
    position_0_wins: int = 0
    position_0_games: int = 0
    position_1_wins: int = 0
    position_1_games: int = 0

    @property
    def win_rate(self) -> float:
        decided = self.wins + self.losses
        return self.wins / max(decided, 1)

    @property
    def bb_per_100(self) -> float:
        return compute_bb_per_100(
            self.total_chips_won - self.total_chips_lost,
            self.total_games,
            big_blind=2
        )

    @property
    def avg_chips_per_game(self) -> float:
        return (self.total_chips_won - self.total_chips_lost) / max(self.total_games, 1)


class ModelEvaluator:
    """Evaluates a trained model against benchmark opponents."""

    def __init__(
        self,
        model_path: str | Path,
        config: EvalConfig | None = None,
        seed: int = 42,
        console: Console | None = None,
    ) -> None:
        self.config = config or EvalConfig()
        self.console = console or Console()
        self.rng_key = jrandom.PRNGKey(seed)

        # Load model
        self._load_model(model_path)

    def _load_model(self, model_path: str | Path) -> None:
        """Load trained model checkpoint."""
        model_path = Path(model_path)

        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")

        with open(model_path, "rb") as f:
            checkpoint = pickle.load(f)

        self.params = checkpoint["params"]

        # Infer hidden dims from checkpoint params
        hidden_dims = []
        for i in range(10):  # Check up to 10 layers
            key = f"Dense_{i}"
            if key in self.params:
                # Get output dim from kernel shape [input, output]
                kernel_shape = self.params[key]["kernel"].shape
                hidden_dims.append(kernel_shape[1])
            else:
                break
        # Last few Dense layers are heads (policy, value), so use first 3 for backbone
        if len(hidden_dims) >= 3:
            hidden_dims = tuple(hidden_dims[:3])
        else:
            hidden_dims = (512, 256, 128)  # Default

        # Detect if this is an opponent model (v9) by checking for LSTM params
        self.use_opponent_model = "OpponentLSTM_0" in self.params

        if self.use_opponent_model:
            self.network = create_network("mlp_opponent", hidden_dims=hidden_dims)
        else:
            self.network = create_network("mlp", hidden_dims=hidden_dims)

        self.console.print(f"Loaded model from: [cyan]{model_path}[/cyan]")

    def _get_model_action(
        self,
        obs: Array,
        valid_mask: Array,
        rng_key: Array,
        opp_action: Array | None = None,
        lstm_carry: tuple | None = None,
    ) -> tuple[Array, tuple | None]:
        """Get action from trained model.

        Returns:
            Tuple of (actions, new_lstm_carry). new_lstm_carry is None for non-opponent models.
        """
        if self.use_opponent_model:
            action_logits, _, _, new_carry = self.network.apply(
                {"params": self.params}, obs, opp_action, lstm_carry, training=False
            )
        else:
            action_logits, _, _ = self.network.apply(
                {"params": self.params}, obs, training=False
            )
            new_carry = None

        # Use low temperature for more deterministic evaluation
        actions, _ = sample_action(rng_key, action_logits, valid_mask, temperature=0.1)
        return actions, new_carry

    def _run_evaluation_batch(
        self,
        opponent_fn: Callable,
        opponent_name: str,
        model_position: int,
        num_games: int,
    ) -> EvalResults:
        """Run evaluation batch with model in specified position.

        Args:
            opponent_fn: JAX-jittable opponent function
            opponent_name: Name for logging
            model_position: 0 or 1 (which player is the model)
            num_games: Number of games to run

        Returns:
            EvalResults from this batch
        """
        config = self.config
        n_games = num_games

        # Initialize games
        self.rng_key, reset_key = jrandom.split(self.rng_key)
        state = reset(
            reset_key, n_games,
            config.starting_chips, config.small_blind, config.big_blind
        )

        results = EvalResults(opponent_type=opponent_name, total_games=n_games)
        action_idx_to_name = {
            1: "fold", 2: "check", 3: "call",
            4: "raise_33", 5: "raise_66", 6: "raise_100", 7: "raise_150",
            8: "all_in"
        }
        results.action_counts = {name: 0 for name in action_idx_to_name.values()}

        max_steps = 200  # Safety limit per game
        needs_obs = opponent_name in NEEDS_OBS

        # Initialize LSTM state for opponent model
        if self.use_opponent_model:
            lstm_hidden = jnp.zeros((n_games, OPPONENT_LSTM_HIDDEN), dtype=jnp.float32)
            lstm_cell = jnp.zeros((n_games, OPPONENT_LSTM_HIDDEN), dtype=jnp.float32)
            last_opp_action = jnp.zeros((n_games, OPPONENT_ACTION_DIM), dtype=jnp.float32)
        else:
            lstm_hidden = lstm_cell = last_opp_action = None

        for step_idx in range(max_steps):
            if jnp.all(state.done):
                break

            # Get whose turn it is for each game
            current_player = state.current_player
            is_model_turn = current_player == model_position

            # Get observations for all games
            obs = encode_state_for_current_player(state)
            valid_mask = get_valid_actions_from_obs(obs)

            # Split RNG
            self.rng_key, model_key, opp_key = jrandom.split(self.rng_key, 3)

            # Get model actions for all games
            if self.use_opponent_model:
                lstm_carry = (lstm_hidden, lstm_cell)
                model_actions, new_carry = self._get_model_action(
                    obs, valid_mask, model_key, last_opp_action, lstm_carry
                )
            else:
                model_actions, _ = self._get_model_action(obs, valid_mask, model_key)

            # Get opponent actions for all games
            if needs_obs:
                opp_actions = opponent_fn(state, valid_mask, opp_key, obs)
            else:
                opp_actions = opponent_fn(state, valid_mask, opp_key)

            # Select actions based on whose turn it is
            actions = jnp.where(is_model_turn, model_actions, opp_actions)

            # Track model's actions (only when it's model's turn and game not done)
            active_model = is_model_turn & ~state.done
            for idx, name in action_idx_to_name.items():
                results.action_counts[name] += int(
                    ((actions == idx) & active_model).sum()
                )

            # Step environment
            state = step(state, actions)

            # Update LSTM state for opponent model
            if self.use_opponent_model:
                # Update LSTM carry where model acted
                lstm_hidden = jnp.where(is_model_turn[:, None], new_carry[0], lstm_hidden)
                lstm_cell = jnp.where(is_model_turn[:, None], new_carry[1], lstm_cell)

                # Encode opponent action where opponent acted
                opp_acted = ~is_model_turn & ~state.done
                opp_action_type = jnp.where(opp_acted, opp_actions + 1, jnp.zeros_like(opp_actions))
                game_idx = jnp.arange(n_games)
                opp_player_idx = 1 - model_position
                opp_bet = state.bets[game_idx, opp_player_idx]

                current_player_arr = jnp.full(n_games, model_position, dtype=jnp.int32)
                encoded_opp = encode_opponent_action_from_state(
                    state, opp_action_type, opp_bet, current_player_arr
                )
                last_opp_action = jnp.where(opp_acted[:, None], encoded_opp, last_opp_action)

        # Compute final results from rewards
        rewards = get_rewards(state)
        model_rewards = rewards[:, model_position]

        results.wins = int((model_rewards > 0).sum())
        results.losses = int((model_rewards < 0).sum())
        results.ties = int((model_rewards == 0).sum())
        results.total_chips_won = float(jnp.maximum(model_rewards, 0).sum())
        results.total_chips_lost = float(jnp.abs(jnp.minimum(model_rewards, 0)).sum())

        return results

    def evaluate_against(
        self,
        opponent_type: str,
    ) -> EvalResults:
        """Evaluate model against a specific opponent type.

        Args:
            opponent_type: One of 'random', 'call_station', 'tag', 'lag', 'rock'

        Returns:
            EvalResults with combined statistics
        """
        if opponent_type not in OPPONENT_TYPES:
            raise ValueError(
                f"Unknown opponent: {opponent_type}. "
                f"Available: {list(OPPONENT_TYPES.keys())}"
            )

        opponent_fn = OPPONENT_TYPES[opponent_type]
        config = self.config

        self.console.print(f"Evaluating vs [cyan]{opponent_type}[/cyan]...")

        if config.test_both_positions:
            # Split games between positions
            games_per_position = config.num_games // 2

            # Position 0
            results_p0 = self._run_evaluation_batch(
                opponent_fn, opponent_type,
                model_position=0,
                num_games=games_per_position
            )

            # Position 1
            results_p1 = self._run_evaluation_batch(
                opponent_fn, opponent_type,
                model_position=1,
                num_games=games_per_position
            )

            # Combine results
            combined = EvalResults(
                opponent_type=opponent_type,
                total_games=results_p0.total_games + results_p1.total_games,
                wins=results_p0.wins + results_p1.wins,
                losses=results_p0.losses + results_p1.losses,
                ties=results_p0.ties + results_p1.ties,
                total_chips_won=results_p0.total_chips_won + results_p1.total_chips_won,
                total_chips_lost=results_p0.total_chips_lost + results_p1.total_chips_lost,
                position_0_wins=results_p0.wins,
                position_0_games=results_p0.total_games,
                position_1_wins=results_p1.wins,
                position_1_games=results_p1.total_games,
            )

            # Merge action counts
            combined.action_counts = {}
            for k in results_p0.action_counts:
                combined.action_counts[k] = (
                    results_p0.action_counts[k] + results_p1.action_counts[k]
                )

            return combined
        else:
            return self._run_evaluation_batch(
                opponent_fn, opponent_type,
                model_position=0,
                num_games=config.num_games
            )

    def evaluate_all(
        self,
        opponents: list[str] | None = None,
    ) -> dict[str, EvalResults]:
        """Evaluate against all specified opponents.

        Args:
            opponents: List of opponent types. If None, tests all.

        Returns:
            Dict mapping opponent name to results
        """
        if opponents is None:
            opponents = list(OPPONENT_TYPES.keys())

        all_results = {}
        for opp in opponents:
            results = self.evaluate_against(opp)
            all_results[opp] = results

        return all_results

    def print_results(
        self,
        results: dict[str, EvalResults],
        show_actions: bool = True,
    ) -> None:
        """Print evaluation results as formatted table."""
        # Main results table
        table = Table(title="Evaluation Results")
        table.add_column("Opponent", style="cyan")
        table.add_column("Games", style="white")
        table.add_column("Win Rate", style="green")
        table.add_column("BB/100", style="yellow")
        table.add_column("Chips/Game", style="white")
        table.add_column("95% CI", style="dim")

        for opp_name, res in results.items():
            ci_low, ci_high = compute_confidence_interval(
                res.win_rate, res.wins + res.losses
            )

            # Color code BB/100
            bb_str = f"{res.bb_per_100:+.2f}"
            if res.bb_per_100 > 0:
                bb_str = f"[green]{bb_str}[/green]"
            elif res.bb_per_100 < 0:
                bb_str = f"[red]{bb_str}[/red]"

            table.add_row(
                opp_name,
                str(res.total_games),
                f"{res.win_rate:.1%}",
                bb_str,
                f"{res.avg_chips_per_game:+.2f}",
                f"({ci_low:.1%}, {ci_high:.1%})",
            )

        self.console.print(table)

        # Action distribution table
        if show_actions:
            self.console.print("\n")
            action_table = Table(title="Model Action Distribution")
            action_table.add_column("Opponent", style="cyan")

            action_names = ["fold", "check", "call", "raise_33", "raise_66",
                          "raise_100", "raise_150", "all_in"]
            for name in action_names:
                action_table.add_column(name[:6], style="white")  # Truncate for display

            for opp_name, res in results.items():
                total = sum(res.action_counts.values()) or 1
                row = [opp_name]
                for action in action_names:
                    pct = res.action_counts.get(action, 0) / total
                    row.append(f"{pct:.0%}")
                action_table.add_row(*row)

            self.console.print(action_table)

        # Statistical significance
        self.console.print("\n[bold]Statistical Significance (vs 50% baseline):[/bold]")
        for opp_name, res in results.items():
            decided = res.wins + res.losses
            sig = is_statistically_significant(res.win_rate, decided)
            status = "[green]Yes[/green]" if sig else "[yellow]No[/yellow]"
            self.console.print(f"  {opp_name}: {status}")

    def export_csv(self, results: dict[str, EvalResults], path: str) -> None:
        """Export results to CSV file."""
        import csv

        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "opponent", "games", "wins", "losses", "ties",
                "win_rate", "bb_per_100", "avg_chips_game",
                "chips_won", "chips_lost",
                "pos0_wins", "pos0_games", "pos1_wins", "pos1_games",
                "fold_pct", "check_pct", "call_pct",
                "raise_33_pct", "raise_66_pct", "raise_100_pct",
                "raise_150_pct", "all_in_pct"
            ])

            for opp_name, res in results.items():
                total_actions = sum(res.action_counts.values()) or 1
                writer.writerow([
                    opp_name,
                    res.total_games,
                    res.wins,
                    res.losses,
                    res.ties,
                    f"{res.win_rate:.4f}",
                    f"{res.bb_per_100:.4f}",
                    f"{res.avg_chips_per_game:.4f}",
                    res.total_chips_won,
                    res.total_chips_lost,
                    res.position_0_wins,
                    res.position_0_games,
                    res.position_1_wins,
                    res.position_1_games,
                    f"{res.action_counts.get('fold', 0) / total_actions:.4f}",
                    f"{res.action_counts.get('check', 0) / total_actions:.4f}",
                    f"{res.action_counts.get('call', 0) / total_actions:.4f}",
                    f"{res.action_counts.get('raise_33', 0) / total_actions:.4f}",
                    f"{res.action_counts.get('raise_66', 0) / total_actions:.4f}",
                    f"{res.action_counts.get('raise_100', 0) / total_actions:.4f}",
                    f"{res.action_counts.get('raise_150', 0) / total_actions:.4f}",
                    f"{res.action_counts.get('all_in', 0) / total_actions:.4f}",
                ])

        self.console.print(f"Results exported to: [cyan]{path}[/cyan]")
