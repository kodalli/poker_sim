"""Evolution engine for genetic algorithm."""

from dataclasses import dataclass, field
from pathlib import Path
from random import Random
from typing import TYPE_CHECKING

import torch
from tqdm import tqdm

from genetics.crossover import BlendCrossover, CrossoverOperator
from genetics.fitness import FitnessConfig, FitnessEvaluator
from genetics.mutation import AdaptiveMutation, GaussianMutation, MutationOperator
from genetics.population import Individual, Population
from genetics.selection import SelectionStrategy, TournamentSelection

if TYPE_CHECKING:
    pass


@dataclass
class EvolutionConfig:
    """Configuration for evolution."""

    generations: int = 100
    elite_fraction: float = 0.1
    crossover_prob: float = 0.8
    mutation_rate: float = 0.1
    mutation_strength: float = 0.1
    tournament_size: int = 3
    checkpoint_every: int = 10
    checkpoint_dir: str = "checkpoints"


@dataclass
class GenerationStats:
    """Statistics for a single generation."""

    generation: int
    best_fitness: float
    avg_fitness: float
    worst_fitness: float
    fitness_std: float
    elite_fitness: float  # Average of top 10%
    games_played: int
    hands_played: int
    mutation_strength: float = 0.0


class EvolutionEngine:
    """Orchestrate the genetic algorithm evolution process."""

    def __init__(
        self,
        population: Population,
        fitness_config: FitnessConfig | None = None,
        evolution_config: EvolutionConfig | None = None,
        selection: SelectionStrategy | None = None,
        crossover: CrossoverOperator | None = None,
        mutation: MutationOperator | None = None,
        seed: int | None = None,
    ) -> None:
        self.population = population
        self.fitness_config = fitness_config or FitnessConfig()
        self.config = evolution_config or EvolutionConfig()

        # Initialize operators
        self.selection = selection or TournamentSelection(self.config.tournament_size)
        self.crossover = crossover or BlendCrossover(alpha=0.5)

        if mutation is None:
            self.mutation = AdaptiveMutation(
                initial_rate=self.config.mutation_rate,
                initial_strength=self.config.mutation_strength,
            )
        else:
            self.mutation = mutation

        self.fitness_evaluator = FitnessEvaluator(
            config=self.fitness_config,
            device=population.device,
            seed=seed,
        )

        self.rng = Random(seed)
        self.history: list[GenerationStats] = []

        # Create checkpoint directory
        self.checkpoint_dir = Path(self.config.checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def evolve_generation(self, show_progress: bool = True) -> GenerationStats:
        """Run one generation of evolution.

        1. Evaluate fitness
        2. Record statistics
        3. Select parents
        4. Create offspring via crossover and mutation
        5. Replace population

        Returns:
            Statistics for this generation
        """
        # Evaluate fitness
        eval_stats = self.fitness_evaluator.evaluate_population(
            self.population, show_progress=show_progress
        )

        # Calculate statistics
        pop_stats = self.population.get_statistics()
        elite_count = max(1, int(len(self.population.individuals) * self.config.elite_fraction))
        elite = self.population.get_best_individuals(elite_count)
        elite_fitness = sum(ind.fitness for ind in elite) / len(elite)

        # Get mutation strength if adaptive
        mutation_strength = self.config.mutation_strength
        if isinstance(self.mutation, AdaptiveMutation):
            mutation_strength = self.mutation.mutation_strength

        stats = GenerationStats(
            generation=self.population.generation,
            best_fitness=pop_stats["best_fitness"],
            avg_fitness=pop_stats["avg_fitness"],
            worst_fitness=pop_stats["worst_fitness"],
            fitness_std=pop_stats["fitness_std"],
            elite_fitness=elite_fitness,
            games_played=eval_stats["total_games"],
            hands_played=eval_stats["total_hands"],
            mutation_strength=mutation_strength,
        )
        self.history.append(stats)

        # Adapt mutation if using adaptive
        if isinstance(self.mutation, AdaptiveMutation):
            self.mutation.adapt(pop_stats["best_fitness"])

        # Create next generation
        new_individuals: list[Individual] = []

        # Elitism: keep best individuals unchanged
        for ind in elite:
            # Clone the elite individuals
            weights = ind.network.get_flat_weights()
            new_ind = self.population.create_individual_from_weights(
                weights, generation=self.population.generation + 1
            )
            new_individuals.append(new_ind)

        # Fill rest with offspring
        while len(new_individuals) < self.population.size:
            # Select parents
            parents = self.selection.select(self.population.individuals, 2, self.rng)

            # Get parent weights
            w1 = parents[0].network.get_flat_weights()
            w2 = parents[1].network.get_flat_weights()

            # Crossover
            if self.rng.random() < self.config.crossover_prob:
                c1, c2 = self.crossover.crossover(w1, w2)
            else:
                c1, c2 = w1.clone(), w2.clone()

            # Mutation
            c1 = self.mutation.mutate(c1)
            c2 = self.mutation.mutate(c2)

            # Create new individuals
            for weights in [c1, c2]:
                if len(new_individuals) < self.population.size:
                    new_ind = self.population.create_individual_from_weights(
                        weights, generation=self.population.generation + 1
                    )
                    new_individuals.append(new_ind)

        # Replace population
        self.population.replace_population(new_individuals)

        return stats

    def run(
        self,
        generations: int | None = None,
        show_progress: bool = True,
        callback: "callable | None" = None,
    ) -> list[GenerationStats]:
        """Run evolution for N generations.

        Args:
            generations: Number of generations (default from config)
            show_progress: Show progress bar
            callback: Optional function called after each generation
                      with (engine, generation, stats) arguments

        Returns:
            List of generation statistics
        """
        generations = generations or self.config.generations

        iterator = range(generations)
        if show_progress:
            iterator = tqdm(iterator, desc="Evolution", unit="gen")

        for gen in iterator:
            stats = self.evolve_generation(show_progress=False)

            if show_progress:
                # Update progress bar description
                iterator.set_postfix(
                    best=f"{stats.best_fitness:.1f}",
                    avg=f"{stats.avg_fitness:.1f}",
                )

            # Save checkpoint
            if (gen + 1) % self.config.checkpoint_every == 0:
                self.save_checkpoint(f"gen_{gen + 1:04d}.pt")

            # Callback
            if callback is not None:
                callback(self, gen, stats)

        # Final checkpoint
        self.save_checkpoint("final.pt")

        return self.history

    def save_checkpoint(self, filename: str) -> None:
        """Save evolution state to checkpoint file."""
        path = self.checkpoint_dir / filename
        state = {
            "population_state": {
                "generation": self.population.generation,
                "size": self.population.size,
                "architecture": self.population.architecture,
                "individuals": [
                    {
                        "id": ind.id,
                        "weights": ind.network.get_flat_weights().cpu(),
                        "fitness": ind.fitness,
                        "games_played": ind.games_played,
                        "generation": ind.generation,
                    }
                    for ind in self.population.individuals
                ],
            },
            "history": [
                {
                    "generation": s.generation,
                    "best_fitness": s.best_fitness,
                    "avg_fitness": s.avg_fitness,
                    "worst_fitness": s.worst_fitness,
                    "fitness_std": s.fitness_std,
                    "elite_fitness": s.elite_fitness,
                    "games_played": s.games_played,
                    "hands_played": s.hands_played,
                    "mutation_strength": s.mutation_strength,
                }
                for s in self.history
            ],
            "config": {
                "evolution": self.config,
                "fitness": self.fitness_config,
            },
        }
        torch.save(state, path)

    def load_checkpoint(self, filename: str) -> None:
        """Load evolution state from checkpoint file."""
        path = self.checkpoint_dir / filename
        state = torch.load(path, map_location=self.population.device)

        # Restore population
        pop_state = state["population_state"]
        self.population.generation = pop_state["generation"]
        self.population.individuals = []

        for ind_state in pop_state["individuals"]:
            ind = self.population.create_individual_from_weights(
                ind_state["weights"],
                generation=ind_state["generation"],
            )
            ind.id = ind_state["id"]
            ind.fitness = ind_state["fitness"]
            ind.games_played = ind_state["games_played"]
            self.population.individuals.append(ind)

        # Restore history
        self.history = [
            GenerationStats(
                generation=s["generation"],
                best_fitness=s["best_fitness"],
                avg_fitness=s["avg_fitness"],
                worst_fitness=s["worst_fitness"],
                fitness_std=s["fitness_std"],
                elite_fitness=s["elite_fitness"],
                games_played=s["games_played"],
                hands_played=s["hands_played"],
                mutation_strength=s.get("mutation_strength", 0),
            )
            for s in state["history"]
        ]

    def get_best_agent(self) -> "NeuralAgent":
        """Get the best agent from the current population."""
        best = self.population.get_best_individuals(1)[0]
        return self.population.get_agent(best)

    def print_summary(self) -> None:
        """Print evolution summary."""
        if not self.history:
            print("No evolution history")
            return

        print("\n" + "=" * 50)
        print("Evolution Summary")
        print("=" * 50)
        print(f"Generations: {len(self.history)}")
        print(f"Population size: {self.population.size}")

        first = self.history[0]
        last = self.history[-1]

        print(f"\nInitial best fitness: {first.best_fitness:.2f}")
        print(f"Final best fitness: {last.best_fitness:.2f}")
        print(f"Improvement: {last.best_fitness - first.best_fitness:.2f}")

        print(f"\nInitial avg fitness: {first.avg_fitness:.2f}")
        print(f"Final avg fitness: {last.avg_fitness:.2f}")

        # Find best generation
        best_gen = max(self.history, key=lambda s: s.best_fitness)
        print(f"\nBest generation: {best_gen.generation}")
        print(f"Best fitness ever: {best_gen.best_fitness:.2f}")
        print("=" * 50)
