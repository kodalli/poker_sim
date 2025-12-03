"""Mutation operators for genetic algorithm."""

from abc import ABC, abstractmethod

import torch


class MutationOperator(ABC):
    """Abstract base class for mutation operators."""

    @abstractmethod
    def mutate(self, weights: torch.Tensor) -> torch.Tensor:
        """Apply mutation to weights.

        Args:
            weights: Flattened weight tensor

        Returns:
            Mutated weight tensor
        """
        ...


class GaussianMutation(MutationOperator):
    """Add Gaussian noise to weights."""

    def __init__(
        self,
        mutation_rate: float = 0.1,
        mutation_strength: float = 0.1,
    ) -> None:
        """
        Args:
            mutation_rate: Probability of mutating each weight
            mutation_strength: Standard deviation of Gaussian noise
        """
        self.mutation_rate = mutation_rate
        self.mutation_strength = mutation_strength

    def mutate(self, weights: torch.Tensor) -> torch.Tensor:
        """Add Gaussian noise with probability mutation_rate."""
        mask = torch.rand_like(weights) < self.mutation_rate
        noise = torch.randn_like(weights) * self.mutation_strength
        return weights + mask * noise


class UniformMutation(MutationOperator):
    """Replace weights with uniform random values."""

    def __init__(
        self,
        mutation_rate: float = 0.1,
        low: float = -1.0,
        high: float = 1.0,
    ) -> None:
        self.mutation_rate = mutation_rate
        self.low = low
        self.high = high

    def mutate(self, weights: torch.Tensor) -> torch.Tensor:
        """Replace with uniform random with probability mutation_rate."""
        mask = torch.rand_like(weights) < self.mutation_rate
        replacement = torch.rand_like(weights) * (self.high - self.low) + self.low
        return torch.where(mask, replacement, weights)


class AdaptiveMutation(MutationOperator):
    """Mutation with adaptive strength based on fitness improvement."""

    def __init__(
        self,
        initial_rate: float = 0.1,
        initial_strength: float = 0.1,
        min_strength: float = 0.01,
        max_strength: float = 0.5,
        adaptation_rate: float = 0.1,
    ) -> None:
        self.mutation_rate = initial_rate
        self.mutation_strength = initial_strength
        self.min_strength = min_strength
        self.max_strength = max_strength
        self.adaptation_rate = adaptation_rate
        self._prev_best_fitness: float | None = None

    def adapt(self, current_best_fitness: float) -> None:
        """Adjust mutation parameters based on fitness improvement."""
        if self._prev_best_fitness is not None:
            improvement = current_best_fitness - self._prev_best_fitness

            if improvement > 0:
                # Things are improving - reduce mutation (exploitation)
                self.mutation_strength *= 1 - self.adaptation_rate
                self.mutation_strength = max(self.min_strength, self.mutation_strength)
            else:
                # Stagnation - increase mutation (exploration)
                self.mutation_strength *= 1 + self.adaptation_rate
                self.mutation_strength = min(self.max_strength, self.mutation_strength)

        self._prev_best_fitness = current_best_fitness

    def mutate(self, weights: torch.Tensor) -> torch.Tensor:
        """Apply Gaussian mutation with current strength."""
        mask = torch.rand_like(weights) < self.mutation_rate
        noise = torch.randn_like(weights) * self.mutation_strength
        return weights + mask * noise

    def get_state(self) -> dict:
        """Get current state for logging."""
        return {
            "mutation_rate": self.mutation_rate,
            "mutation_strength": self.mutation_strength,
        }


class PolynomialMutation(MutationOperator):
    """Polynomial mutation - commonly used with SBX crossover."""

    def __init__(
        self,
        mutation_rate: float = 0.1,
        eta: float = 20.0,
        low: float = -1.0,
        high: float = 1.0,
    ) -> None:
        """
        Args:
            mutation_rate: Probability of mutating each weight
            eta: Distribution index. Higher = smaller mutations
            low: Lower bound for weights
            high: Upper bound for weights
        """
        self.mutation_rate = mutation_rate
        self.eta = eta
        self.low = low
        self.high = high

    def mutate(self, weights: torch.Tensor) -> torch.Tensor:
        """Apply polynomial mutation."""
        mask = torch.rand_like(weights) < self.mutation_rate

        u = torch.rand_like(weights)
        delta = torch.where(
            u < 0.5,
            (2 * u) ** (1 / (self.eta + 1)) - 1,
            1 - (2 * (1 - u)) ** (1 / (self.eta + 1)),
        )

        mutated = weights + delta * (self.high - self.low)
        mutated = torch.clamp(mutated, self.low, self.high)

        return torch.where(mask, mutated, weights)


class CreepMutation(MutationOperator):
    """Creep mutation - small incremental changes."""

    def __init__(
        self,
        mutation_rate: float = 0.2,
        creep_size: float = 0.05,
    ) -> None:
        self.mutation_rate = mutation_rate
        self.creep_size = creep_size

    def mutate(self, weights: torch.Tensor) -> torch.Tensor:
        """Add small creep to weights."""
        mask = torch.rand_like(weights) < self.mutation_rate
        # Uniform creep in [-creep_size, creep_size]
        creep = (torch.rand_like(weights) * 2 - 1) * self.creep_size
        return weights + mask * creep


class CompoundMutation(MutationOperator):
    """Combine multiple mutation operators."""

    def __init__(self, operators: list[tuple[MutationOperator, float]]) -> None:
        """
        Args:
            operators: List of (operator, probability) tuples
        """
        self.operators = operators

    def mutate(self, weights: torch.Tensor) -> torch.Tensor:
        """Apply each operator with its probability."""
        result = weights.clone()
        for operator, prob in self.operators:
            if torch.rand(1).item() < prob:
                result = operator.mutate(result)
        return result


class LayerWiseMutation(MutationOperator):
    """Different mutation rates for different layers.

    Useful when some layers (e.g., output layers) should be
    mutated more or less than others.
    """

    def __init__(
        self,
        layer_sizes: list[int],
        layer_rates: list[float],
        mutation_strength: float = 0.1,
    ) -> None:
        """
        Args:
            layer_sizes: Number of parameters in each layer
            layer_rates: Mutation rate for each layer
            mutation_strength: Standard deviation of Gaussian noise
        """
        self.layer_sizes = layer_sizes
        self.layer_rates = layer_rates
        self.mutation_strength = mutation_strength

    def mutate(self, weights: torch.Tensor) -> torch.Tensor:
        """Apply layer-specific mutation rates."""
        result = weights.clone()
        offset = 0

        for size, rate in zip(self.layer_sizes, self.layer_rates):
            layer_weights = weights[offset : offset + size]
            mask = torch.rand_like(layer_weights) < rate
            noise = torch.randn_like(layer_weights) * self.mutation_strength
            result[offset : offset + size] = layer_weights + mask * noise
            offset += size

        return result
