"""Selection strategies for genetic algorithm."""

from abc import ABC, abstractmethod
from random import Random

from genetics.population import Individual


class SelectionStrategy(ABC):
    """Abstract base class for selection strategies."""

    @abstractmethod
    def select(
        self,
        population: list[Individual],
        n: int,
        rng: Random | None = None,
    ) -> list[Individual]:
        """Select n individuals from the population.

        Args:
            population: List of individuals to select from
            n: Number of individuals to select
            rng: Random number generator

        Returns:
            List of selected individuals (may contain duplicates)
        """
        ...


class TournamentSelection(SelectionStrategy):
    """Tournament selection - pick best from random subset."""

    def __init__(self, tournament_size: int = 3) -> None:
        self.tournament_size = tournament_size

    def select(
        self,
        population: list[Individual],
        n: int,
        rng: Random | None = None,
    ) -> list[Individual]:
        """Select n individuals via tournament selection."""
        if rng is None:
            rng = Random()

        selected = []
        for _ in range(n):
            # Sample tournament
            tournament_size = min(self.tournament_size, len(population))
            tournament = rng.sample(population, tournament_size)

            # Pick best from tournament
            winner = max(tournament, key=lambda x: x.fitness)
            selected.append(winner)

        return selected


class EliteSelection(SelectionStrategy):
    """Elite selection - select only the best individuals."""

    def __init__(self, elite_fraction: float = 0.1) -> None:
        self.elite_fraction = elite_fraction

    def select(
        self,
        population: list[Individual],
        n: int,
        rng: Random | None = None,
    ) -> list[Individual]:
        """Select top n individuals by fitness."""
        sorted_pop = sorted(population, key=lambda x: x.fitness, reverse=True)
        return sorted_pop[:n]


class RouletteWheelSelection(SelectionStrategy):
    """Roulette wheel (fitness proportionate) selection."""

    def __init__(self, temperature: float = 1.0) -> None:
        self.temperature = temperature

    def select(
        self,
        population: list[Individual],
        n: int,
        rng: Random | None = None,
    ) -> list[Individual]:
        """Select n individuals with probability proportional to fitness."""
        if rng is None:
            rng = Random()

        if not population:
            return []

        # Shift fitness to positive and apply temperature
        min_fit = min(ind.fitness for ind in population)
        adjusted = [
            (ind.fitness - min_fit + 1e-6) ** (1 / self.temperature)
            for ind in population
        ]
        total = sum(adjusted)

        if total == 0:
            # All equal fitness - uniform selection
            return rng.choices(population, k=n)

        probabilities = [f / total for f in adjusted]

        # Weighted selection
        selected = []
        for _ in range(n):
            r = rng.random()
            cumsum = 0
            for ind, prob in zip(population, probabilities):
                cumsum += prob
                if r <= cumsum:
                    selected.append(ind)
                    break
            else:
                selected.append(population[-1])

        return selected


class RankSelection(SelectionStrategy):
    """Rank-based selection - selection probability based on rank, not fitness."""

    def __init__(self, selection_pressure: float = 2.0) -> None:
        """
        Args:
            selection_pressure: Higher values favor top ranks more
        """
        self.selection_pressure = selection_pressure

    def select(
        self,
        population: list[Individual],
        n: int,
        rng: Random | None = None,
    ) -> list[Individual]:
        """Select based on rank in population."""
        if rng is None:
            rng = Random()

        if not population:
            return []

        # Sort by fitness
        sorted_pop = sorted(population, key=lambda x: x.fitness, reverse=True)
        pop_size = len(sorted_pop)

        # Calculate rank-based probabilities
        # Linear ranking: P(i) = (2-s)/N + 2*(s-1)*(N-i)/(N*(N-1))
        # where s is selection pressure, N is population size, i is rank (0-indexed)
        s = self.selection_pressure
        probabilities = []
        for i in range(pop_size):
            if pop_size > 1:
                prob = (2 - s) / pop_size + 2 * (s - 1) * (pop_size - 1 - i) / (pop_size * (pop_size - 1))
            else:
                prob = 1.0
            probabilities.append(max(0, prob))

        # Normalize
        total = sum(probabilities)
        if total > 0:
            probabilities = [p / total for p in probabilities]
        else:
            probabilities = [1 / pop_size] * pop_size

        # Select
        selected = []
        for _ in range(n):
            r = rng.random()
            cumsum = 0
            for ind, prob in zip(sorted_pop, probabilities):
                cumsum += prob
                if r <= cumsum:
                    selected.append(ind)
                    break
            else:
                selected.append(sorted_pop[-1])

        return selected


class StochasticUniversalSampling(SelectionStrategy):
    """Stochastic Universal Sampling - evenly spaced selection points."""

    def select(
        self,
        population: list[Individual],
        n: int,
        rng: Random | None = None,
    ) -> list[Individual]:
        """Select n individuals using stochastic universal sampling."""
        if rng is None:
            rng = Random()

        if not population or n == 0:
            return []

        # Calculate fitness sum (shifted to positive)
        min_fit = min(ind.fitness for ind in population)
        adjusted = [ind.fitness - min_fit + 1e-6 for ind in population]
        total = sum(adjusted)

        if total == 0:
            return rng.choices(population, k=n)

        # Calculate cumulative probabilities
        cumulative = []
        running = 0
        for f in adjusted:
            running += f / total
            cumulative.append(running)

        # Select with evenly spaced pointers
        distance = 1.0 / n
        start = rng.random() * distance

        selected = []
        idx = 0
        for i in range(n):
            pointer = start + i * distance
            while pointer > cumulative[idx]:
                idx += 1
                if idx >= len(population):
                    idx = len(population) - 1
                    break
            selected.append(population[idx])

        return selected
