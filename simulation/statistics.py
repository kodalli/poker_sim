"""Statistics tracking and visualization for evolution."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np

if TYPE_CHECKING:
    from genetics.evolution import GenerationStats


@dataclass
class StatisticsTracker:
    """Track and visualize evolution statistics."""

    history: list["GenerationStats"] = field(default_factory=list)
    output_dir: str = "plots"

    def __post_init__(self) -> None:
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)

    def add_generation(self, stats: "GenerationStats") -> None:
        """Add statistics for a generation."""
        self.history.append(stats)

    def plot_fitness_evolution(
        self,
        save_path: str | None = None,
        show: bool = False,
    ) -> None:
        """Plot fitness evolution over generations."""
        if not self.history:
            print("No history to plot")
            return

        generations = [s.generation for s in self.history]
        best_fitness = [s.best_fitness for s in self.history]
        avg_fitness = [s.avg_fitness for s in self.history]
        worst_fitness = [s.worst_fitness for s in self.history]

        fig, ax = plt.subplots(figsize=(12, 6))

        ax.plot(generations, best_fitness, "g-", label="Best", linewidth=2)
        ax.plot(generations, avg_fitness, "b-", label="Average", linewidth=2)
        ax.plot(generations, worst_fitness, "r-", label="Worst", linewidth=1, alpha=0.5)

        # Add std deviation band around average
        std = [s.fitness_std for s in self.history]
        avg_np = np.array(avg_fitness)
        std_np = np.array(std)
        ax.fill_between(
            generations,
            avg_np - std_np,
            avg_np + std_np,
            alpha=0.2,
            color="blue",
            label="±1 std",
        )

        ax.set_xlabel("Generation")
        ax.set_ylabel("Fitness (avg chips won)")
        ax.set_title("Fitness Evolution Over Generations")
        ax.legend(loc="upper left")
        ax.grid(True, alpha=0.3)

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
        elif show:
            plt.show()
        else:
            path = Path(self.output_dir) / "fitness_evolution.png"
            plt.savefig(path, dpi=150, bbox_inches="tight")

        plt.close(fig)

    def plot_mutation_adaptation(
        self,
        save_path: str | None = None,
        show: bool = False,
    ) -> None:
        """Plot mutation strength adaptation over generations."""
        if not self.history:
            return

        generations = [s.generation for s in self.history]
        mutation_strength = [s.mutation_strength for s in self.history]

        fig, ax = plt.subplots(figsize=(10, 4))

        ax.plot(generations, mutation_strength, "m-", linewidth=2)
        ax.set_xlabel("Generation")
        ax.set_ylabel("Mutation Strength")
        ax.set_title("Adaptive Mutation Strength")
        ax.grid(True, alpha=0.3)

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
        elif show:
            plt.show()
        else:
            path = Path(self.output_dir) / "mutation_adaptation.png"
            plt.savefig(path, dpi=150, bbox_inches="tight")

        plt.close(fig)

    def plot_diversity(
        self,
        save_path: str | None = None,
        show: bool = False,
    ) -> None:
        """Plot population diversity (std) over generations."""
        if not self.history:
            return

        generations = [s.generation for s in self.history]
        std = [s.fitness_std for s in self.history]

        fig, ax = plt.subplots(figsize=(10, 4))

        ax.plot(generations, std, "c-", linewidth=2)
        ax.set_xlabel("Generation")
        ax.set_ylabel("Fitness Std Dev")
        ax.set_title("Population Diversity")
        ax.grid(True, alpha=0.3)

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
        elif show:
            plt.show()
        else:
            path = Path(self.output_dir) / "diversity.png"
            plt.savefig(path, dpi=150, bbox_inches="tight")

        plt.close(fig)

    def plot_all(self, show: bool = False) -> None:
        """Generate all plots."""
        self.plot_fitness_evolution(show=show)
        self.plot_mutation_adaptation(show=show)
        self.plot_diversity(show=show)

    def get_summary(self) -> dict:
        """Get summary statistics."""
        if not self.history:
            return {}

        first = self.history[0]
        last = self.history[-1]
        best_gen = max(self.history, key=lambda s: s.best_fitness)

        return {
            "generations": len(self.history),
            "initial_best_fitness": first.best_fitness,
            "final_best_fitness": last.best_fitness,
            "best_fitness_ever": best_gen.best_fitness,
            "best_generation": best_gen.generation,
            "improvement": last.best_fitness - first.best_fitness,
            "total_games": sum(s.games_played for s in self.history),
            "total_hands": sum(s.hands_played for s in self.history),
        }

    def print_summary(self) -> None:
        """Print summary to console."""
        summary = self.get_summary()
        if not summary:
            print("No statistics to summarize")
            return

        print("\n" + "=" * 50)
        print("Evolution Summary")
        print("=" * 50)
        print(f"Generations completed: {summary['generations']}")
        print(f"Total games played: {summary['total_games']:,}")
        print(f"Total hands played: {summary['total_hands']:,}")
        print()
        print(f"Initial best fitness: {summary['initial_best_fitness']:.2f}")
        print(f"Final best fitness: {summary['final_best_fitness']:.2f}")
        print(f"Best fitness ever: {summary['best_fitness_ever']:.2f} (gen {summary['best_generation']})")
        print(f"Total improvement: {summary['improvement']:.2f}")
        print("=" * 50)


def plot_agent_comparison(
    results: dict[str, dict],
    save_path: str | None = None,
    show: bool = True,
) -> None:
    """Plot comparison of multiple agents.

    Args:
        results: Dict mapping agent name to results dict
                (from GameRunner.run_tournament)
        save_path: Path to save plot
        show: Whether to show plot
    """
    agent_names = list(results.keys())

    # Extract metrics
    win_rates = [results[name]["win_rate"][0] for name in agent_names]
    avg_chips = [results[name]["avg_chips_per_game"][0] for name in agent_names]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Win rate bar chart
    axes[0].bar(agent_names, win_rates, color="steelblue")
    axes[0].set_ylabel("Win Rate")
    axes[0].set_title("Win Rate by Agent")
    axes[0].set_ylim(0, max(win_rates) * 1.2 if win_rates else 1)

    # Average chips bar chart
    colors = ["green" if c > 0 else "red" for c in avg_chips]
    axes[1].bar(agent_names, avg_chips, color=colors)
    axes[1].set_ylabel("Avg Chips per Game")
    axes[1].set_title("Average Chips Won per Game")
    axes[1].axhline(y=0, color="black", linestyle="-", linewidth=0.5)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()

    plt.close(fig)
