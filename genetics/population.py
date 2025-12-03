"""Population management for genetic algorithm."""

from dataclasses import dataclass, field

import torch

from agents.neural.agent import NeuralAgent
from agents.neural.encoding import StateEncoder
from agents.neural.network import NetworkConfig, PokerNetwork, create_network


@dataclass
class Individual:
    """A single member of the population."""

    id: int
    network: PokerNetwork
    fitness: float = 0.0
    games_played: int = 0
    hands_won: int = 0
    total_chips_won: int = 0
    generation: int = 0

    def reset_fitness(self) -> None:
        """Reset fitness for new evaluation."""
        self.fitness = 0.0
        self.games_played = 0
        self.hands_won = 0
        self.total_chips_won = 0


class Population:
    """Manage a population of neural network agents."""

    def __init__(
        self,
        size: int,
        network_config: NetworkConfig | None = None,
        architecture: str = "mlp",
        device: torch.device | None = None,
    ) -> None:
        self.size = size
        self.network_config = network_config or NetworkConfig()
        self.architecture = architecture
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.individuals: list[Individual] = []
        self.generation = 0
        self._encoder = StateEncoder(self.device)
        self._next_id = 0

    def initialize_random(self) -> None:
        """Create initial random population."""
        self.individuals = []
        for _ in range(self.size):
            network = create_network(self.architecture, self.network_config)
            network.to(self.device)
            individual = Individual(
                id=self._next_id,
                network=network,
                generation=0,
            )
            self.individuals.append(individual)
            self._next_id += 1

    def get_agent(self, individual: Individual, temperature: float = 1.0) -> NeuralAgent:
        """Create an agent from an individual."""
        return NeuralAgent(
            network=individual.network,
            encoder=self._encoder,
            device=self.device,
            temperature=temperature,
            name=f"Agent_{individual.id}",
        )

    def get_all_agents(self, temperature: float = 1.0) -> list[NeuralAgent]:
        """Get agents for all individuals."""
        return [self.get_agent(ind, temperature) for ind in self.individuals]

    def get_best_individuals(self, n: int) -> list[Individual]:
        """Get the n best individuals by fitness."""
        sorted_pop = sorted(self.individuals, key=lambda x: x.fitness, reverse=True)
        return sorted_pop[:n]

    def reset_all_fitness(self) -> None:
        """Reset fitness for all individuals."""
        for ind in self.individuals:
            ind.reset_fitness()

    def get_statistics(self) -> dict:
        """Get population statistics."""
        fitnesses = [ind.fitness for ind in self.individuals]
        return {
            "generation": self.generation,
            "population_size": self.size,
            "best_fitness": max(fitnesses) if fitnesses else 0,
            "worst_fitness": min(fitnesses) if fitnesses else 0,
            "avg_fitness": sum(fitnesses) / len(fitnesses) if fitnesses else 0,
            "fitness_std": (
                (sum((f - sum(fitnesses) / len(fitnesses)) ** 2 for f in fitnesses) / len(fitnesses)) ** 0.5
                if fitnesses
                else 0
            ),
            "total_games": sum(ind.games_played for ind in self.individuals),
            "total_hands_won": sum(ind.hands_won for ind in self.individuals),
        }

    def create_individual_from_weights(
        self,
        weights: torch.Tensor,
        generation: int | None = None,
    ) -> Individual:
        """Create a new individual from weight tensor."""
        network = create_network(self.architecture, self.network_config)
        network.to(self.device)
        network.set_flat_weights(weights.to(self.device))

        individual = Individual(
            id=self._next_id,
            network=network,
            generation=generation if generation is not None else self.generation,
        )
        self._next_id += 1
        return individual

    def add_individual(self, individual: Individual) -> None:
        """Add an individual to the population."""
        self.individuals.append(individual)

    def remove_worst(self, n: int) -> list[Individual]:
        """Remove and return the n worst individuals."""
        sorted_pop = sorted(self.individuals, key=lambda x: x.fitness)
        removed = sorted_pop[:n]
        self.individuals = sorted_pop[n:]
        return removed

    def replace_population(self, new_individuals: list[Individual]) -> None:
        """Replace population with new individuals."""
        self.individuals = new_individuals
        self.generation += 1

    def save(self, path: str) -> None:
        """Save population state to file."""
        state = {
            "generation": self.generation,
            "size": self.size,
            "architecture": self.architecture,
            "network_config": self.network_config,
            "individuals": [
                {
                    "id": ind.id,
                    "weights": ind.network.get_flat_weights().cpu(),
                    "fitness": ind.fitness,
                    "games_played": ind.games_played,
                    "generation": ind.generation,
                }
                for ind in self.individuals
            ],
        }
        torch.save(state, path)

    def load(self, path: str) -> None:
        """Load population state from file."""
        state = torch.load(path, map_location=self.device)

        self.generation = state["generation"]
        self.size = state["size"]
        self.architecture = state["architecture"]

        self.individuals = []
        for ind_state in state["individuals"]:
            individual = self.create_individual_from_weights(
                ind_state["weights"],
                generation=ind_state["generation"],
            )
            individual.id = ind_state["id"]
            individual.fitness = ind_state["fitness"]
            individual.games_played = ind_state["games_played"]
            self.individuals.append(individual)

        self._next_id = max(ind.id for ind in self.individuals) + 1 if self.individuals else 0
