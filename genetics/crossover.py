"""Crossover operators for genetic algorithm."""

from abc import ABC, abstractmethod

import torch


class CrossoverOperator(ABC):
    """Abstract base class for crossover operators."""

    @abstractmethod
    def crossover(
        self,
        parent1: torch.Tensor,
        parent2: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Produce two offspring from two parents.

        Args:
            parent1: Flattened weight tensor of first parent
            parent2: Flattened weight tensor of second parent

        Returns:
            Tuple of two offspring weight tensors
        """
        ...


class UniformCrossover(CrossoverOperator):
    """Uniform crossover - randomly swap each weight."""

    def __init__(self, swap_prob: float = 0.5) -> None:
        self.swap_prob = swap_prob

    def crossover(
        self,
        parent1: torch.Tensor,
        parent2: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Swap each weight with probability swap_prob."""
        mask = torch.rand_like(parent1) < self.swap_prob
        child1 = torch.where(mask, parent2, parent1)
        child2 = torch.where(mask, parent1, parent2)
        return child1.clone(), child2.clone()


class SinglePointCrossover(CrossoverOperator):
    """Single point crossover - swap at one random point."""

    def crossover(
        self,
        parent1: torch.Tensor,
        parent2: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Swap weights after a random crossover point."""
        length = parent1.shape[0]
        point = torch.randint(0, length, (1,)).item()

        child1 = torch.cat([parent1[:point], parent2[point:]])
        child2 = torch.cat([parent2[:point], parent1[point:]])

        return child1.clone(), child2.clone()


class TwoPointCrossover(CrossoverOperator):
    """Two point crossover - swap between two random points."""

    def crossover(
        self,
        parent1: torch.Tensor,
        parent2: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Swap weights between two crossover points."""
        length = parent1.shape[0]
        points = sorted(torch.randint(0, length, (2,)).tolist())

        child1 = parent1.clone()
        child2 = parent2.clone()

        child1[points[0] : points[1]] = parent2[points[0] : points[1]]
        child2[points[0] : points[1]] = parent1[points[0] : points[1]]

        return child1, child2


class BlendCrossover(CrossoverOperator):
    """BLX-alpha crossover for real-valued weights.

    Creates offspring by sampling uniformly from an extended range
    around the parents' values.
    """

    def __init__(self, alpha: float = 0.5) -> None:
        """
        Args:
            alpha: Extension factor. 0 = interpolate only,
                   0.5 = blend, >0.5 = more exploration
        """
        self.alpha = alpha

    def crossover(
        self,
        parent1: torch.Tensor,
        parent2: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Create offspring by blending parent weights."""
        diff = torch.abs(parent1 - parent2)
        low = torch.min(parent1, parent2) - self.alpha * diff
        high = torch.max(parent1, parent2) + self.alpha * diff

        child1 = low + torch.rand_like(parent1) * (high - low)
        child2 = low + torch.rand_like(parent2) * (high - low)

        return child1, child2


class ArithmeticCrossover(CrossoverOperator):
    """Arithmetic crossover - weighted average of parents."""

    def __init__(self, weight: float = 0.5) -> None:
        """
        Args:
            weight: Weight for parent1 (1-weight for parent2)
        """
        self.weight = weight

    def crossover(
        self,
        parent1: torch.Tensor,
        parent2: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Create offspring as weighted average."""
        child1 = self.weight * parent1 + (1 - self.weight) * parent2
        child2 = (1 - self.weight) * parent1 + self.weight * parent2
        return child1.clone(), child2.clone()


class SimulatedBinaryCrossover(CrossoverOperator):
    """Simulated Binary Crossover (SBX).

    Produces offspring distribution similar to single-point crossover
    for binary strings, but for real-valued weights.
    """

    def __init__(self, eta: float = 2.0) -> None:
        """
        Args:
            eta: Distribution index. Higher = children closer to parents
        """
        self.eta = eta

    def crossover(
        self,
        parent1: torch.Tensor,
        parent2: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """SBX crossover."""
        u = torch.rand_like(parent1)

        # Calculate beta
        beta = torch.where(
            u <= 0.5,
            (2 * u) ** (1 / (self.eta + 1)),
            (1 / (2 * (1 - u))) ** (1 / (self.eta + 1)),
        )

        child1 = 0.5 * ((1 + beta) * parent1 + (1 - beta) * parent2)
        child2 = 0.5 * ((1 - beta) * parent1 + (1 + beta) * parent2)

        return child1, child2


class LayerWiseCrossover(CrossoverOperator):
    """Crossover that respects network layer boundaries.

    Swaps entire layers rather than individual weights, which may
    preserve learned features better.
    """

    def __init__(self, layer_sizes: list[int] | None = None) -> None:
        """
        Args:
            layer_sizes: Size of each layer in weights. If None,
                        falls back to uniform crossover.
        """
        self.layer_sizes = layer_sizes

    def crossover(
        self,
        parent1: torch.Tensor,
        parent2: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Swap entire layers between parents."""
        if self.layer_sizes is None:
            # Fallback to uniform
            mask = torch.rand_like(parent1) < 0.5
            child1 = torch.where(mask, parent2, parent1)
            child2 = torch.where(mask, parent1, parent2)
            return child1.clone(), child2.clone()

        child1 = parent1.clone()
        child2 = parent2.clone()

        offset = 0
        for size in self.layer_sizes:
            if torch.rand(1).item() < 0.5:
                # Swap this layer
                child1[offset : offset + size] = parent2[offset : offset + size]
                child2[offset : offset + size] = parent1[offset : offset + size]
            offset += size

        return child1, child2
