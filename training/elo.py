"""ELO rating system and checkpoint pool for historical self-play."""

import copy
from dataclasses import dataclass, field
from typing import Optional

import jax.numpy as jnp
import jax.random as jrandom
from jax import Array


# Standard ELO parameters
INITIAL_ELO = 1500.0
K_FACTOR = 32.0  # How much ratings change per game


def compute_expected_score(rating_a: float, rating_b: float) -> float:
    """Compute expected score for player A against player B.

    Args:
        rating_a: ELO rating of player A
        rating_b: ELO rating of player B

    Returns:
        Expected score (probability of winning) for player A
    """
    return 1.0 / (1.0 + 10.0 ** ((rating_b - rating_a) / 400.0))


def update_elo_ratings(
    winner_elo: float,
    loser_elo: float,
    k: float = K_FACTOR,
) -> tuple[float, float]:
    """Update ELO ratings after a game.

    Args:
        winner_elo: Current ELO of winner
        loser_elo: Current ELO of loser
        k: K-factor (rating adjustment speed)

    Returns:
        Tuple of (new_winner_elo, new_loser_elo)
    """
    expected_winner = compute_expected_score(winner_elo, loser_elo)
    expected_loser = 1.0 - expected_winner

    new_winner_elo = winner_elo + k * (1.0 - expected_winner)
    new_loser_elo = loser_elo + k * (0.0 - expected_loser)

    return new_winner_elo, new_loser_elo


@dataclass
class HistoricalCheckpoint:
    """A saved checkpoint with its ELO rating."""
    params: dict
    elo: float
    step: int


@dataclass
class CheckpointPool:
    """Pool of historical checkpoints for self-play training.

    Maintains a fixed-size pool of past model checkpoints.
    New checkpoints are added periodically during training.
    When pool is full, oldest checkpoint is removed.
    """
    max_size: int = 20
    checkpoints: list[HistoricalCheckpoint] = field(default_factory=list)
    current_elo: float = INITIAL_ELO

    # Track games for ELO updates
    _pending_wins: int = 0
    _pending_losses: int = 0
    _opponent_elos: list[float] = field(default_factory=list)

    def add_checkpoint(self, params: dict, step: int) -> None:
        """Add a new checkpoint to the pool.

        Args:
            params: Model parameters (will be deep copied)
            step: Training step when checkpoint was saved
        """
        # Deep copy params to avoid mutation
        checkpoint = HistoricalCheckpoint(
            params=copy.deepcopy(params),
            elo=self.current_elo,  # New checkpoint inherits current ELO
            step=step,
        )

        self.checkpoints.append(checkpoint)

        # Remove oldest if over capacity
        if len(self.checkpoints) > self.max_size:
            self.checkpoints.pop(0)

    def sample_opponent(self, rng_key: Array) -> tuple[dict, int, float]:
        """Sample a random historical opponent.

        Args:
            rng_key: JAX PRNG key

        Returns:
            Tuple of (params, checkpoint_index, opponent_elo)
        """
        if not self.checkpoints:
            raise ValueError("No checkpoints in pool")

        # Uniform random selection
        idx = int(jrandom.randint(rng_key, (), 0, len(self.checkpoints)))
        checkpoint = self.checkpoints[idx]

        return checkpoint.params, idx, checkpoint.elo

    def has_checkpoints(self) -> bool:
        """Check if pool has any checkpoints."""
        return len(self.checkpoints) > 0

    def record_game_result(self, model_won: bool, opponent_elo: float) -> None:
        """Record a game result for batch ELO update.

        Args:
            model_won: True if current model won
            opponent_elo: ELO of the opponent
        """
        if model_won:
            self._pending_wins += 1
        else:
            self._pending_losses += 1
        self._opponent_elos.append(opponent_elo)

    def update_elo_batch(self) -> float:
        """Update current model's ELO based on accumulated results.

        Returns:
            New current ELO
        """
        if not self._opponent_elos:
            return self.current_elo

        # Compute average opponent ELO
        avg_opponent_elo = sum(self._opponent_elos) / len(self._opponent_elos)
        total_games = self._pending_wins + self._pending_losses

        if total_games == 0:
            return self.current_elo

        # Actual score (wins / total)
        actual_score = self._pending_wins / total_games

        # Expected score against average opponent
        expected_score = compute_expected_score(self.current_elo, avg_opponent_elo)

        # Update ELO (scale K by number of games, capped)
        effective_k = min(K_FACTOR * total_games / 100, K_FACTOR * 2)
        self.current_elo += effective_k * (actual_score - expected_score)

        # Reset pending results
        self._pending_wins = 0
        self._pending_losses = 0
        self._opponent_elos = []

        return self.current_elo

    def get_pool_stats(self) -> dict:
        """Get statistics about the checkpoint pool.

        Returns:
            Dict with pool statistics
        """
        if not self.checkpoints:
            return {
                "pool_size": 0,
                "min_elo": 0.0,
                "max_elo": 0.0,
                "avg_elo": 0.0,
                "oldest_step": 0,
                "newest_step": 0,
            }

        elos = [c.elo for c in self.checkpoints]
        steps = [c.step for c in self.checkpoints]

        return {
            "pool_size": len(self.checkpoints),
            "min_elo": min(elos),
            "max_elo": max(elos),
            "avg_elo": sum(elos) / len(elos),
            "oldest_step": min(steps),
            "newest_step": max(steps),
        }


# Fixed opponent ELO ratings (for mixed training)
OPPONENT_ELOS = {
    "random": 800.0,
    "call_station": 1000.0,
    "tag": 1200.0,
    "lag": 1300.0,
    "rock": 1100.0,
}
