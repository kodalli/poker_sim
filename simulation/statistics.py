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

    def plot_game_length_evolution(
        self,
        save_path: str | None = None,
        show: bool = False,
    ) -> None:
        """Plot hands per game over generations."""
        if not self.history:
            return

        generations = [s.generation for s in self.history]
        avg_hands = [s.hands_per_game_avg for s in self.history]
        min_hands = [s.hands_per_game_min for s in self.history]
        max_hands = [s.hands_per_game_max for s in self.history]

        fig, ax = plt.subplots(figsize=(12, 5))

        ax.plot(generations, avg_hands, "b-", label="Average", linewidth=2)
        ax.fill_between(generations, min_hands, max_hands, alpha=0.2, color="blue", label="Min-Max range")

        ax.set_xlabel("Generation")
        ax.set_ylabel("Hands per Game")
        ax.set_title("Game Length Evolution")
        ax.legend(loc="upper left")
        ax.grid(True, alpha=0.3)

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
        elif show:
            plt.show()
        else:
            path = Path(self.output_dir) / "game_length.png"
            plt.savefig(path, dpi=150, bbox_inches="tight")

        plt.close(fig)

    def plot_showdown_rates(
        self,
        save_path: str | None = None,
        show: bool = False,
    ) -> None:
        """Plot showdown % and all-in % over generations."""
        if not self.history:
            return

        generations = [s.generation for s in self.history]
        showdown_rates = [s.showdown_rate * 100 for s in self.history]
        all_in_rates = [s.all_in_rate * 100 for s in self.history]

        fig, ax = plt.subplots(figsize=(12, 5))

        ax.plot(generations, showdown_rates, "g-", label="Showdown %", linewidth=2)
        ax.plot(generations, all_in_rates, "r-", label="All-in %", linewidth=2)

        ax.set_xlabel("Generation")
        ax.set_ylabel("Rate (%)")
        ax.set_title("Showdown and All-in Rates")
        ax.legend(loc="upper right")
        ax.grid(True, alpha=0.3)

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
        elif show:
            plt.show()
        else:
            path = Path(self.output_dir) / "showdown_rates.png"
            plt.savefig(path, dpi=150, bbox_inches="tight")

        plt.close(fig)

    def plot_actions_per_hand(
        self,
        save_path: str | None = None,
        show: bool = False,
    ) -> None:
        """Plot actions per hand over generations."""
        if not self.history:
            return

        generations = [s.generation for s in self.history]
        actions = [s.actions_per_hand for s in self.history]

        fig, ax = plt.subplots(figsize=(12, 5))

        ax.plot(generations, actions, "purple", linewidth=2)

        ax.set_xlabel("Generation")
        ax.set_ylabel("Actions per Hand")
        ax.set_title("Decision Complexity Over Generations")
        ax.grid(True, alpha=0.3)

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
        elif show:
            plt.show()
        else:
            path = Path(self.output_dir) / "actions_per_hand.png"
            plt.savefig(path, dpi=150, bbox_inches="tight")

        plt.close(fig)

    def plot_winning_hands(
        self,
        save_path: str | None = None,
        show: bool = False,
    ) -> None:
        """Stacked area chart of winning hand types."""
        if not self.history:
            return

        # Collect all hand types
        hand_types = ["high_card", "pair", "two_pair", "three_of_a_kind",
                     "straight", "flush", "full_house", "four_of_a_kind",
                     "straight_flush", "royal_flush"]

        generations = [s.generation for s in self.history]

        # Build data for each hand type
        data = {ht: [] for ht in hand_types}
        for s in self.history:
            dist = s.winning_hand_distribution or {}
            for ht in hand_types:
                data[ht].append(dist.get(ht, 0) * 100)

        fig, ax = plt.subplots(figsize=(14, 6))

        # Stack area chart
        bottom = np.zeros(len(generations))
        colors = plt.cm.viridis(np.linspace(0, 1, len(hand_types)))

        for i, ht in enumerate(hand_types):
            values = np.array(data[ht])
            if np.any(values > 0):
                ax.fill_between(generations, bottom, bottom + values,
                              label=ht.replace("_", " ").title(),
                              alpha=0.7, color=colors[i])
                bottom += values

        ax.set_xlabel("Generation")
        ax.set_ylabel("Distribution (%)")
        ax.set_title("Winning Hand Distribution Over Generations")
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(True, alpha=0.3)

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
        elif show:
            plt.show()
        else:
            path = Path(self.output_dir) / "winning_hands.png"
            plt.savefig(path, dpi=150, bbox_inches="tight")

        plt.close(fig)

    def plot_performance(
        self,
        save_path: str | None = None,
        show: bool = False,
    ) -> None:
        """Plot time per generation and actions/sec."""
        if not self.history:
            return

        generations = [s.generation for s in self.history]
        time_secs = [s.time_seconds for s in self.history]
        actions_per_sec = [s.actions_per_second for s in self.history]

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

        ax1.plot(generations, time_secs, "b-", linewidth=2)
        ax1.set_ylabel("Time (seconds)")
        ax1.set_title("Generation Time")
        ax1.grid(True, alpha=0.3)

        ax2.plot(generations, actions_per_sec, "g-", linewidth=2)
        ax2.set_xlabel("Generation")
        ax2.set_ylabel("Actions per Second")
        ax2.set_title("Throughput")
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
        elif show:
            plt.show()
        else:
            path = Path(self.output_dir) / "performance.png"
            plt.savefig(path, dpi=150, bbox_inches="tight")

        plt.close(fig)

    def plot_agent_behavior(
        self,
        save_path: str | None = None,
        show: bool = False,
    ) -> None:
        """Plot VPIP, PFR, aggression factor over generations."""
        if not self.history:
            return

        generations = [s.generation for s in self.history]
        vpip = [s.avg_vpip * 100 for s in self.history]
        pfr = [s.avg_pfr * 100 for s in self.history]
        aggression = [s.avg_aggression_factor for s in self.history]
        fold_rate = [s.avg_fold_rate * 100 for s in self.history]

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

        # Top: VPIP and PFR
        ax1.plot(generations, vpip, "b-", label="VPIP %", linewidth=2)
        ax1.plot(generations, pfr, "g-", label="PFR %", linewidth=2)
        ax1.plot(generations, fold_rate, "r-", label="Fold %", linewidth=2, alpha=0.7)
        ax1.set_ylabel("Rate (%)")
        ax1.set_title("Agent Behavior: VPIP, PFR, Fold Rate")
        ax1.legend(loc="upper right")
        ax1.grid(True, alpha=0.3)

        # Bottom: Aggression factor
        ax2.plot(generations, aggression, "purple", linewidth=2)
        ax2.set_xlabel("Generation")
        ax2.set_ylabel("Aggression Factor")
        ax2.set_title("Aggression Factor (Raises / Calls)")
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
        elif show:
            plt.show()
        else:
            path = Path(self.output_dir) / "agent_behavior.png"
            plt.savefig(path, dpi=150, bbox_inches="tight")

        plt.close(fig)

    def plot_rounds_reached(
        self,
        save_path: str | None = None,
        show: bool = False,
    ) -> None:
        """Stacked area chart of round distribution."""
        if not self.history:
            return

        rounds = ["preflop", "flop", "turn", "river", "showdown"]
        generations = [s.generation for s in self.history]

        # Build data for each round
        data = {r: [] for r in rounds}
        for s in self.history:
            dist = s.rounds_reached or {}
            for r in rounds:
                data[r].append(dist.get(r, 0) * 100)

        fig, ax = plt.subplots(figsize=(12, 6))

        # Stack area chart
        bottom = np.zeros(len(generations))
        colors = ["#ff6b6b", "#feca57", "#48dbfb", "#1dd1a1", "#5f27cd"]

        for i, r in enumerate(rounds):
            values = np.array(data[r])
            ax.fill_between(generations, bottom, bottom + values,
                          label=r.title(), alpha=0.7, color=colors[i])
            bottom += values

        ax.set_xlabel("Generation")
        ax.set_ylabel("Distribution (%)")
        ax.set_title("Final Round Reached Distribution")
        ax.legend(loc="upper right")
        ax.grid(True, alpha=0.3)

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
        elif show:
            plt.show()
        else:
            path = Path(self.output_dir) / "rounds_reached.png"
            plt.savefig(path, dpi=150, bbox_inches="tight")

        plt.close(fig)

    def plot_all(self, show: bool = False) -> None:
        """Generate all plots."""
        self.plot_fitness_evolution(show=show)
        self.plot_mutation_adaptation(show=show)
        self.plot_diversity(show=show)
        # New plots
        self.plot_game_length_evolution(show=show)
        self.plot_showdown_rates(show=show)
        self.plot_actions_per_hand(show=show)
        self.plot_winning_hands(show=show)
        self.plot_performance(show=show)
        self.plot_rounds_reached(show=show)
        self.plot_agent_behavior(show=show)

    def get_summary(self) -> dict:
        """Get summary statistics."""
        if not self.history:
            return {}

        first = self.history[0]
        last = self.history[-1]
        best_gen = max(self.history, key=lambda s: s.best_fitness)

        # Calculate totals and averages
        total_time = sum(s.time_seconds for s in self.history)
        total_actions = sum(s.hands_played * s.actions_per_hand for s in self.history)

        return {
            "generations": len(self.history),
            "initial_best_fitness": first.best_fitness,
            "final_best_fitness": last.best_fitness,
            "best_fitness_ever": best_gen.best_fitness,
            "best_generation": best_gen.generation,
            "improvement": last.best_fitness - first.best_fitness,
            "total_games": sum(s.games_played for s in self.history),
            "total_hands": sum(s.hands_played for s in self.history),
            # New stats
            "avg_hands_per_game": last.hands_per_game_avg,
            "min_hands_per_game": min(s.hands_per_game_min for s in self.history),
            "max_hands_per_game": max(s.hands_per_game_max for s in self.history),
            "final_actions_per_hand": last.actions_per_hand,
            "final_showdown_rate": last.showdown_rate,
            "final_all_in_rate": last.all_in_rate,
            "total_time_seconds": total_time,
            "avg_time_per_gen": total_time / len(self.history) if self.history else 0,
            "avg_actions_per_sec": total_actions / total_time if total_time > 0 else 0,
            # Agent behavior
            "final_vpip": last.avg_vpip,
            "final_pfr": last.avg_pfr,
            "final_aggression": last.avg_aggression_factor,
            "final_fold_rate": last.avg_fold_rate,
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
        print()
        print("Game Dynamics:")
        print(f"  Avg hands/game: {summary['avg_hands_per_game']:.1f} "
              f"(min: {summary['min_hands_per_game']}, max: {summary['max_hands_per_game']})")
        print(f"  Avg actions/hand: {summary['final_actions_per_hand']:.1f}")
        print(f"  Showdown rate: {summary['final_showdown_rate']*100:.1f}%")
        print(f"  All-in rate: {summary['final_all_in_rate']*100:.1f}%")
        print()
        print("Agent Behavior (final gen):")
        print(f"  VPIP: {summary['final_vpip']*100:.1f}% | PFR: {summary['final_pfr']*100:.1f}%")
        print(f"  Aggression: {summary['final_aggression']:.2f} | Fold rate: {summary['final_fold_rate']*100:.1f}%")
        print()
        print("Performance:")
        total_mins = summary['total_time_seconds'] / 60
        print(f"  Total time: {total_mins:.1f}m")
        print(f"  Avg: {summary['avg_time_per_gen']:.1f}s/gen | "
              f"{summary['avg_actions_per_sec']:.0f} actions/sec")
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
