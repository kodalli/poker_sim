"""Tournament system for model vs model competition."""

from dataclasses import dataclass, field
from itertools import combinations

import torch
from tqdm import tqdm

from agents.base import BaseAgent
from genetics.population import Population
from models.manager import get_final_checkpoint
from simulation.runner import GameRunner, SimulationConfig


@dataclass
class MatchupResult:
    """Result of a 1v1 matchup between two models."""

    model1: str
    model2: str
    games_played: int
    model1_wins: int
    model2_wins: int
    draws: int
    model1_chips: int  # Total chips won/lost
    model2_chips: int
    model1_avg_chips: float
    model2_avg_chips: float

    @property
    def model1_win_rate(self) -> float:
        """Win rate for model1."""
        return self.model1_wins / self.games_played if self.games_played > 0 else 0.0

    @property
    def model2_win_rate(self) -> float:
        """Win rate for model2."""
        return self.model2_wins / self.games_played if self.games_played > 0 else 0.0


@dataclass
class TournamentResult:
    """Result of a round-robin tournament."""

    models: list[str]
    matchups: list[MatchupResult]
    games_per_matchup: int

    # Aggregated stats per model
    wins: dict[str, int] = field(default_factory=dict)
    losses: dict[str, int] = field(default_factory=dict)
    draws: dict[str, int] = field(default_factory=dict)
    total_chips: dict[str, int] = field(default_factory=dict)
    games_played: dict[str, int] = field(default_factory=dict)

    def get_leaderboard(self) -> list[tuple[str, dict]]:
        """Get models sorted by win rate, then by total chips."""
        stats = []
        for model in self.models:
            total_games = self.games_played.get(model, 0)
            win_rate = self.wins.get(model, 0) / total_games if total_games > 0 else 0
            stats.append(
                (
                    model,
                    {
                        "wins": self.wins.get(model, 0),
                        "losses": self.losses.get(model, 0),
                        "draws": self.draws.get(model, 0),
                        "games": total_games,
                        "win_rate": win_rate,
                        "total_chips": self.total_chips.get(model, 0),
                        "avg_chips": self.total_chips.get(model, 0) / total_games if total_games > 0 else 0,
                    },
                )
            )

        # Sort by win rate (descending), then total chips (descending)
        stats.sort(key=lambda x: (x[1]["win_rate"], x[1]["total_chips"]), reverse=True)
        return stats


def load_model_agent(
    version: str,
    device: torch.device | None = None,
    temperature: float = 0.3,
) -> BaseAgent:
    """Load a model version as an agent."""
    checkpoint_path = get_final_checkpoint(version)

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"No checkpoint found for model {version}")

    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    pop = Population(size=1, device=device)
    pop.load(str(checkpoint_path))

    best = pop.get_best_individuals(1)[0]
    agent = pop.get_agent(best, temperature=temperature)
    agent.name = version

    return agent


def run_matchup(
    model1: str,
    model2: str,
    num_games: int = 100,
    device: torch.device | None = None,
    show_progress: bool = True,
) -> MatchupResult:
    """Run a 1v1 matchup between two model versions.

    Args:
        model1: First model version
        model2: Second model version
        num_games: Number of games to play
        device: Torch device to use
        show_progress: Show progress bar

    Returns:
        MatchupResult with statistics
    """
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load agents
    agent1 = load_model_agent(model1, device)
    agent2 = load_model_agent(model2, device)

    # Run games
    config = SimulationConfig(
        starting_chips=200,
        small_blind=1,
        big_blind=2,
        max_hands_per_game=100,
    )
    runner = GameRunner(config)

    model1_wins = 0
    model2_wins = 0
    draws = 0
    model1_chips = 0
    model2_chips = 0

    iterator = range(num_games)
    if show_progress:
        iterator = tqdm(iterator, desc=f"{model1} vs {model2}", unit="games")

    for _ in iterator:
        # Alternate who is player 0
        agents = [agent1, agent2]
        result = runner.run_game(agents)

        # Track results
        delta1 = result.chip_deltas[0]
        delta2 = result.chip_deltas[1]
        model1_chips += delta1
        model2_chips += delta2

        if result.winner_id == 0:
            model1_wins += 1
        elif result.winner_id == 1:
            model2_wins += 1
        else:
            # No clear winner (game ended by max hands)
            if delta1 > delta2:
                model1_wins += 1
            elif delta2 > delta1:
                model2_wins += 1
            else:
                draws += 1

    return MatchupResult(
        model1=model1,
        model2=model2,
        games_played=num_games,
        model1_wins=model1_wins,
        model2_wins=model2_wins,
        draws=draws,
        model1_chips=model1_chips,
        model2_chips=model2_chips,
        model1_avg_chips=model1_chips / num_games if num_games > 0 else 0,
        model2_avg_chips=model2_chips / num_games if num_games > 0 else 0,
    )


def run_tournament(
    models: list[str],
    games_per_matchup: int = 100,
    device: torch.device | None = None,
    show_progress: bool = True,
) -> TournamentResult:
    """Run a round-robin tournament between multiple models.

    Each pair of models plays games_per_matchup games against each other.

    Args:
        models: List of model versions to compete
        games_per_matchup: Number of games per matchup
        device: Torch device to use
        show_progress: Show progress bar

    Returns:
        TournamentResult with all matchup results and leaderboard
    """
    if len(models) < 2:
        raise ValueError("Need at least 2 models for a tournament")

    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize result tracking
    result = TournamentResult(
        models=models,
        matchups=[],
        games_per_matchup=games_per_matchup,
        wins={m: 0 for m in models},
        losses={m: 0 for m in models},
        draws={m: 0 for m in models},
        total_chips={m: 0 for m in models},
        games_played={m: 0 for m in models},
    )

    # Generate all pairs
    pairs = list(combinations(models, 2))

    if show_progress:
        print(f"Running tournament: {len(models)} models, {len(pairs)} matchups")
        print(f"Total games: {len(pairs) * games_per_matchup}")

    # Run each matchup
    for model1, model2 in pairs:
        matchup = run_matchup(
            model1, model2, games_per_matchup, device, show_progress
        )
        result.matchups.append(matchup)

        # Aggregate stats
        result.wins[model1] += matchup.model1_wins
        result.wins[model2] += matchup.model2_wins
        result.losses[model1] += matchup.model2_wins
        result.losses[model2] += matchup.model1_wins
        result.draws[model1] += matchup.draws
        result.draws[model2] += matchup.draws
        result.total_chips[model1] += matchup.model1_chips
        result.total_chips[model2] += matchup.model2_chips
        result.games_played[model1] += matchup.games_played
        result.games_played[model2] += matchup.games_played

    return result
