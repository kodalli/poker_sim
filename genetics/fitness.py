"""Fitness evaluation for genetic algorithm."""

from dataclasses import dataclass
from random import Random
from typing import TYPE_CHECKING

from tqdm import tqdm

from agents.neural.agent import NeuralAgent
from agents.neural.encoding import StateEncoder
from genetics.population import Individual, Population
from poker.game import TexasHoldemGame
from poker.player import Player

if TYPE_CHECKING:
    import torch


@dataclass
class FitnessConfig:
    """Configuration for fitness evaluation."""

    games_per_evaluation: int = 50
    table_size: int = 9
    small_blind: int = 1
    big_blind: int = 2
    starting_chips: int = 200
    max_hands_per_game: int = 100
    temperature: float = 1.0  # Agent temperature during evaluation


class FitnessEvaluator:
    """Evaluate fitness of individuals through gameplay."""

    def __init__(
        self,
        config: FitnessConfig | None = None,
        device: "torch.device | None" = None,
        seed: int | None = None,
    ) -> None:
        self.config = config or FitnessConfig()
        self.device = device
        self.rng = Random(seed)
        self.encoder = StateEncoder(device) if device else None

    def evaluate_population(
        self,
        population: Population,
        show_progress: bool = True,
    ) -> dict:
        """Evaluate all individuals in the population.

        Each individual plays games against random subsets of other
        individuals. Fitness is the average chip gain per game.

        Returns:
            Statistics about the evaluation
        """
        # Reset fitness for all individuals
        population.reset_all_fitness()

        # Get encoder from population if not set
        if self.encoder is None:
            self.encoder = StateEncoder(population.device)

        # Create agents for all individuals
        agents_map: dict[int, NeuralAgent] = {}
        for ind in population.individuals:
            agents_map[ind.id] = population.get_agent(ind, self.config.temperature)

        total_games = 0
        total_hands = 0

        # Run games
        iterator = range(self.config.games_per_evaluation)
        if show_progress:
            iterator = tqdm(iterator, desc="Evaluating fitness", unit="games")

        for _ in iterator:
            # Sample a table of players
            if len(population.individuals) < self.config.table_size:
                # Not enough players - use all
                table_individuals = population.individuals.copy()
            else:
                table_individuals = self.rng.sample(
                    population.individuals, self.config.table_size
                )

            # Run a game (multiple hands)
            results = self._run_game(table_individuals, agents_map)
            total_games += 1
            total_hands += results["hands_played"]

            # Update fitness for each player
            for ind in table_individuals:
                chip_delta = results["chip_deltas"].get(ind.id, 0)
                ind.total_chips_won += chip_delta
                ind.games_played += 1
                if ind.id in results["winners"]:
                    ind.hands_won += 1

        # Calculate final fitness (average chips won per game)
        for ind in population.individuals:
            if ind.games_played > 0:
                ind.fitness = ind.total_chips_won / ind.games_played
            else:
                ind.fitness = 0

        return {
            "total_games": total_games,
            "total_hands": total_hands,
            "avg_hands_per_game": total_hands / max(1, total_games),
        }

    def _run_game(
        self,
        individuals: list[Individual],
        agents_map: dict[int, NeuralAgent],
    ) -> dict:
        """Run a complete game (multiple hands) with the given individuals.

        Returns:
            Dict with chip_deltas, winners, hands_played
        """
        # Create players
        players = [
            Player(id=ind.id, chips=self.config.starting_chips, name=f"P{ind.id}")
            for ind in individuals
        ]

        # Create agent dict for game
        game_agents = {ind.id: agents_map[ind.id] for ind in individuals}

        initial_chips = {p.id: p.chips for p in players}
        dealer_position = 0
        hands_played = 0
        all_winners: set[int] = set()

        for _ in range(self.config.max_hands_per_game):
            # Check if game over (one player has all chips)
            active_players = [p for p in players if p.chips > 0]
            if len(active_players) <= 1:
                break

            # Create and play a hand
            game = TexasHoldemGame(
                players=active_players,
                dealer_position=dealer_position % len(active_players),
                small_blind=self.config.small_blind,
                big_blind=self.config.big_blind,
                seed=self.rng.randint(0, 2**31),
            )

            result = game.play(game_agents)
            hands_played += 1

            # Track winners
            all_winners.update(result.winners)

            # Rotate dealer
            dealer_position = (dealer_position + 1) % len(active_players)

        # Calculate chip changes
        chip_deltas = {p.id: p.chips - initial_chips[p.id] for p in players}

        return {
            "chip_deltas": chip_deltas,
            "winners": list(all_winners),
            "hands_played": hands_played,
        }

    def evaluate_individual(
        self,
        individual: Individual,
        opponents: list[Individual],
        population: Population,
        num_games: int = 10,
    ) -> float:
        """Evaluate a single individual against a set of opponents.

        Useful for quick evaluation or testing.

        Returns:
            Average chip delta per game
        """
        if self.encoder is None:
            self.encoder = StateEncoder(population.device)

        agent = population.get_agent(individual, self.config.temperature)
        opponent_agents = {
            opp.id: population.get_agent(opp, self.config.temperature)
            for opp in opponents
        }

        agents_map = {individual.id: agent, **opponent_agents}
        all_individuals = [individual] + opponents

        total_chips = 0
        for _ in range(num_games):
            results = self._run_game(all_individuals, agents_map)
            total_chips += results["chip_deltas"].get(individual.id, 0)

        return total_chips / num_games if num_games > 0 else 0
