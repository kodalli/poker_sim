"""Fitness evaluation for genetic algorithm."""

import time
from dataclasses import dataclass
from random import Random
from typing import TYPE_CHECKING

import numpy as np
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
        start_time = time.perf_counter()

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

        # Aggregated metrics
        all_hands_per_game: list[int] = []
        total_actions = 0
        total_showdowns = 0
        total_all_ins = 0
        rounds_reached: dict[str, int] = {}
        winning_hands: dict[str, int] = {}
        total_pot = 0

        # Agent behavior aggregation
        total_raises = 0
        total_calls = 0
        total_folds = 0
        total_checks = 0
        preflop_raises = 0
        preflop_voluntary = 0  # Calls or raises preflop (VPIP)
        total_preflop_opportunities = 0

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
            all_hands_per_game.append(results["hands_played"])

            # Aggregate metrics
            total_actions += results.get("total_actions", 0)
            total_showdowns += results.get("showdowns", 0)
            total_all_ins += results.get("all_ins", 0)
            total_pot += results.get("total_pot", 0)

            for rnd, count in results.get("rounds_reached", {}).items():
                rounds_reached[rnd] = rounds_reached.get(rnd, 0) + count

            for rank, count in results.get("winning_hands", {}).items():
                winning_hands[rank] = winning_hands.get(rank, 0) + count

            # Aggregate agent behavior stats
            behavior = results.get("behavior_stats", {})
            total_raises += behavior.get("raises", 0)
            total_calls += behavior.get("calls", 0)
            total_folds += behavior.get("folds", 0)
            total_checks += behavior.get("checks", 0)
            preflop_raises += behavior.get("preflop_raises", 0)
            preflop_voluntary += behavior.get("preflop_voluntary", 0)
            total_preflop_opportunities += behavior.get("preflop_opportunities", 0)

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

        # Calculate timing
        elapsed = time.perf_counter() - start_time
        actions_per_sec = total_actions / elapsed if elapsed > 0 else 0

        # Calculate game length stats
        hands_arr = np.array(all_hands_per_game) if all_hands_per_game else np.array([0])

        # Calculate agent behavior stats
        total_actions_behavior = total_raises + total_calls + total_folds + total_checks
        avg_vpip = preflop_voluntary / max(1, total_preflop_opportunities)
        avg_pfr = preflop_raises / max(1, total_preflop_opportunities)
        avg_aggression = total_raises / max(1, total_calls)  # AF = raises / calls
        avg_fold_rate = total_folds / max(1, total_actions_behavior)

        return {
            "total_games": total_games,
            "total_hands": total_hands,
            "avg_hands_per_game": total_hands / max(1, total_games),
            # New metrics
            "hands_per_game_min": int(hands_arr.min()),
            "hands_per_game_max": int(hands_arr.max()),
            "hands_per_game_std": float(hands_arr.std()),
            "total_actions": total_actions,
            "actions_per_hand": total_actions / max(1, total_hands),
            "showdown_rate": total_showdowns / max(1, total_hands),
            "all_in_rate": total_all_ins / max(1, total_hands),
            "rounds_reached": rounds_reached,
            "winning_hands": winning_hands,
            "avg_pot_size": total_pot / max(1, total_hands),
            "bluff_success_rate": 1.0 - (total_showdowns / max(1, total_hands)),
            # Agent behavior stats
            "avg_vpip": avg_vpip,
            "avg_pfr": avg_pfr,
            "avg_aggression_factor": avg_aggression,
            "avg_fold_rate": avg_fold_rate,
            # Performance
            "time_seconds": elapsed,
            "actions_per_second": actions_per_sec,
        }

    def _run_game(
        self,
        individuals: list[Individual],
        agents_map: dict[int, NeuralAgent],
    ) -> dict:
        """Run a complete game (multiple hands) with the given individuals.

        Returns:
            Dict with chip_deltas, winners, hands_played, and detailed metrics
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

        # Track aggregated metrics
        total_actions = 0
        showdowns = 0
        all_ins = 0
        total_pot = 0
        rounds_reached: dict[str, int] = {}
        winning_hands: dict[str, int] = {}

        # Behavior stats aggregation
        behavior_raises = 0
        behavior_calls = 0
        behavior_folds = 0
        behavior_checks = 0
        behavior_preflop_raises = 0
        behavior_preflop_voluntary = 0
        behavior_preflop_opportunities = 0

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

            # Aggregate hand metrics
            metrics = result.metrics
            total_actions += metrics.total_actions
            total_pot += metrics.pot_size

            if metrics.went_to_showdown:
                showdowns += 1
            if metrics.had_all_in:
                all_ins += 1

            # Track round reached
            rnd = metrics.round_reached
            rounds_reached[rnd] = rounds_reached.get(rnd, 0) + 1

            # Track winning hand type
            if metrics.winning_hand_rank:
                rank = metrics.winning_hand_rank
                winning_hands[rank] = winning_hands.get(rank, 0) + 1

            # Aggregate player behavior from this hand
            for player_id, actions in metrics.player_actions.items():
                behavior_raises += actions.get("raises", 0)
                behavior_calls += actions.get("calls", 0)
                behavior_folds += actions.get("folds", 0)
                behavior_checks += actions.get("checks", 0)

            # Track preflop behavior (from actions_by_round)
            preflop_actions = metrics.actions_by_round.get("pre_flop", 0)
            if preflop_actions > 0:
                # Count players who had opportunity to act preflop
                behavior_preflop_opportunities += len(active_players)
                # Count preflop raises and voluntary entries
                for player_id, actions in metrics.player_actions.items():
                    if actions.get("raises", 0) > 0:
                        behavior_preflop_raises += 1
                        behavior_preflop_voluntary += 1
                    elif actions.get("calls", 0) > 0:
                        behavior_preflop_voluntary += 1

            # Rotate dealer
            dealer_position = (dealer_position + 1) % len(active_players)

        # Calculate chip changes
        chip_deltas = {p.id: p.chips - initial_chips[p.id] for p in players}

        return {
            "chip_deltas": chip_deltas,
            "winners": list(all_winners),
            "hands_played": hands_played,
            # New metrics
            "total_actions": total_actions,
            "showdowns": showdowns,
            "all_ins": all_ins,
            "total_pot": total_pot,
            "rounds_reached": rounds_reached,
            "winning_hands": winning_hands,
            # Behavior stats
            "behavior_stats": {
                "raises": behavior_raises,
                "calls": behavior_calls,
                "folds": behavior_folds,
                "checks": behavior_checks,
                "preflop_raises": behavior_preflop_raises,
                "preflop_voluntary": behavior_preflop_voluntary,
                "preflop_opportunities": behavior_preflop_opportunities,
            },
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
