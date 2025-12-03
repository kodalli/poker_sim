"""Poker simulation with genetic algorithm AI evolution."""

from pathlib import Path
from typing import Optional

import torch
import typer
from rich.console import Console
from rich.table import Table

from agents.neural.encoding import StateEncoder
from agents.neural.network import NetworkConfig, create_network
from agents.random_agent import RandomAgent, TightAggressiveAgent
from config.settings import Config, load_config, save_config
from genetics.crossover import BlendCrossover
from genetics.evolution import EvolutionConfig, EvolutionEngine
from genetics.fitness import FitnessConfig
from genetics.mutation import AdaptiveMutation
from genetics.population import Population
from genetics.selection import TournamentSelection
from simulation.runner import GameRunner, SimulationConfig
from simulation.statistics import StatisticsTracker
from utils.device import get_device, print_device_info

app = typer.Typer(
    name="poker-sim",
    help="Texas Hold'em poker simulation with genetic algorithm AI evolution.",
)
console = Console()


@app.command()
def train(
    generations: int = typer.Option(100, "--generations", "-g", help="Number of generations"),
    population: int = typer.Option(100, "--population", "-p", help="Population size"),
    architecture: str = typer.Option("mlp", "--arch", "-a", help="Network architecture (mlp, deep_mlp, transformer)"),
    games_per_eval: int = typer.Option(50, "--games", help="Games per fitness evaluation"),
    table_size: int = typer.Option(9, "--table-size", "-t", help="Players per table (2-10)"),
    checkpoint_dir: str = typer.Option("checkpoints", "--checkpoint-dir", help="Checkpoint directory"),
    resume: Optional[str] = typer.Option(None, "--resume", "-r", help="Resume from checkpoint"),
    seed: Optional[int] = typer.Option(None, "--seed", "-s", help="Random seed"),
    cpu: bool = typer.Option(False, "--cpu", help="Force CPU (disable CUDA)"),
) -> None:
    """Train poker AI agents using genetic algorithm evolution."""
    console.print("\n[bold blue]Poker AI Training[/bold blue]")
    console.print("=" * 50)

    # Setup device
    device = get_device(prefer_cuda=not cpu)
    console.print(f"Device: [green]{device}[/green]")

    if device.type == "cuda":
        props = torch.cuda.get_device_properties(0)
        console.print(f"GPU: [green]{props.name}[/green] ({props.total_memory / 1024**3:.1f} GB)")

    # Create configurations
    network_config = NetworkConfig(
        hidden_dims=(256, 128, 64) if architecture != "transformer" else (128,),
    )

    fitness_config = FitnessConfig(
        games_per_evaluation=games_per_eval,
        table_size=table_size,
        small_blind=1,
        big_blind=2,
        starting_chips=200,
    )

    evolution_config = EvolutionConfig(
        generations=generations,
        elite_fraction=0.1,
        crossover_prob=0.8,
        checkpoint_every=10,
        checkpoint_dir=checkpoint_dir,
    )

    # Create population
    console.print(f"\nInitializing population of {population} {architecture} agents...")
    pop = Population(
        size=population,
        network_config=network_config,
        architecture=architecture,
        device=device,
    )

    if resume:
        console.print(f"Resuming from checkpoint: {resume}")
        pop.load(resume)
    else:
        pop.initialize_random()

    # Get network parameter count
    sample_net = pop.individuals[0].network
    param_count = sample_net.num_parameters()
    console.print(f"Network parameters: [cyan]{param_count:,}[/cyan]")

    # Create evolution engine
    engine = EvolutionEngine(
        population=pop,
        fitness_config=fitness_config,
        evolution_config=evolution_config,
        selection=TournamentSelection(tournament_size=3),
        crossover=BlendCrossover(alpha=0.5),
        mutation=AdaptiveMutation(initial_rate=0.1, initial_strength=0.1),
        seed=seed,
    )

    # Setup statistics tracker
    tracker = StatisticsTracker(output_dir="plots")

    def callback(eng: EvolutionEngine, gen: int, stats) -> None:
        tracker.add_generation(stats)
        if (gen + 1) % 10 == 0:
            console.print(
                f"Gen {gen + 1}: "
                f"best=[green]{stats.best_fitness:.1f}[/green] "
                f"avg=[blue]{stats.avg_fitness:.1f}[/blue] "
                f"mut=[yellow]{stats.mutation_strength:.3f}[/yellow]"
            )

    # Run evolution
    console.print(f"\n[bold]Starting evolution for {generations} generations...[/bold]\n")

    try:
        engine.run(generations=generations, show_progress=True, callback=callback)
    except KeyboardInterrupt:
        console.print("\n[yellow]Training interrupted. Saving checkpoint...[/yellow]")
        engine.save_checkpoint("interrupted.pt")

    # Generate plots
    console.print("\nGenerating plots...")
    tracker.plot_all()

    # Print summary
    tracker.print_summary()

    console.print(f"\n[green]Training complete![/green]")
    console.print(f"Checkpoints saved to: {checkpoint_dir}/")
    console.print(f"Plots saved to: plots/")


@app.command()
def evaluate(
    checkpoint: str = typer.Argument(..., help="Path to checkpoint file"),
    num_games: int = typer.Option(100, "--games", "-g", help="Number of games to play"),
    opponents: str = typer.Option("random", "--opponents", "-o", help="Opponent type (random, tag, mixed)"),
    table_size: int = typer.Option(9, "--table-size", "-t", help="Players per table"),
    cpu: bool = typer.Option(False, "--cpu", help="Force CPU"),
) -> None:
    """Evaluate a trained agent against baseline opponents."""
    console.print("\n[bold blue]Agent Evaluation[/bold blue]")
    console.print("=" * 50)

    device = get_device(prefer_cuda=not cpu)
    console.print(f"Device: {device}")

    # Load population and get best agent
    console.print(f"Loading checkpoint: {checkpoint}")
    pop = Population(size=1, device=device)
    pop.load(checkpoint)

    best_individual = pop.get_best_individuals(1)[0]
    best_agent = pop.get_agent(best_individual, temperature=0.5)
    console.print(f"Best agent fitness: {best_individual.fitness:.2f}")

    # Create opponents
    opponent_agents = []
    if opponents == "random":
        opponent_agents = [RandomAgent(name=f"Random_{i}") for i in range(table_size - 1)]
    elif opponents == "tag":
        opponent_agents = [TightAggressiveAgent(name=f"TAG_{i}") for i in range(table_size - 1)]
    else:  # mixed
        for i in range(table_size - 1):
            if i % 2 == 0:
                opponent_agents.append(RandomAgent(name=f"Random_{i}"))
            else:
                opponent_agents.append(TightAggressiveAgent(name=f"TAG_{i}"))

    # Run evaluation
    console.print(f"\nRunning {num_games} games against {opponents} opponents...")

    runner = GameRunner(SimulationConfig(num_games=num_games))
    results = runner.run_evaluation(
        agent=best_agent,
        opponents=opponent_agents,
        num_games=num_games,
        show_progress=True,
    )

    # Print results
    table = Table(title="Evaluation Results")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")

    table.add_row("Games Played", str(num_games))
    table.add_row("Games Won", str(results["games_won"]))
    table.add_row("Win Rate", f"{results['win_rate']:.1%}")
    table.add_row("Avg Chips/Game", f"{results['avg_chips_per_game']:.1f}")
    table.add_row("Total Chips Won", f"{results['total_chips']:+,}")

    console.print(table)


@app.command()
def play(
    checkpoint: Optional[str] = typer.Option(None, "--checkpoint", "-c", help="Load trained agent"),
    num_players: int = typer.Option(6, "--players", "-p", help="Number of players (2-10)"),
    hands: int = typer.Option(10, "--hands", "-n", help="Number of hands to play"),
    cpu: bool = typer.Option(False, "--cpu", help="Force CPU"),
) -> None:
    """Watch AI agents play poker."""
    console.print("\n[bold blue]Poker Game[/bold blue]")
    console.print("=" * 50)

    device = get_device(prefer_cuda=not cpu)

    # Create agents
    agents = []
    if checkpoint:
        pop = Population(size=1, device=device)
        pop.load(checkpoint)
        best = pop.get_best_individuals(1)[0]
        agents.append(pop.get_agent(best, temperature=0.5))
        agents[0].name = "Neural_AI"

        # Fill rest with opponents
        for i in range(num_players - 1):
            if i % 2 == 0:
                agents.append(TightAggressiveAgent(name=f"TAG_{i+1}"))
            else:
                agents.append(RandomAgent(name=f"Random_{i+1}"))
    else:
        # All random agents
        agents = [RandomAgent(name=f"Random_{i}") for i in range(num_players)]

    # Run game
    console.print(f"Playing {hands} hands with {num_players} players...\n")

    runner = GameRunner(SimulationConfig(max_hands_per_game=hands))
    result = runner.run_game(agents)

    # Show results
    table = Table(title=f"Game Results ({result.hands_played} hands)")
    table.add_column("Player", style="cyan")
    table.add_column("Final Chips", style="white")
    table.add_column("Change", style="green")

    for i, agent in enumerate(agents):
        chips = result.final_chips[i]
        delta = result.chip_deltas[i]
        delta_str = f"+{delta}" if delta > 0 else str(delta)
        delta_style = "green" if delta > 0 else "red" if delta < 0 else "white"
        table.add_row(agent.name, str(chips), f"[{delta_style}]{delta_str}[/{delta_style}]")

    console.print(table)

    if result.winner_id is not None:
        console.print(f"\n[bold green]Winner: {agents[result.winner_id].name}[/bold green]")


@app.command()
def info() -> None:
    """Show system information and configuration."""
    console.print("\n[bold blue]System Information[/bold blue]")
    console.print("=" * 50)

    print_device_info()

    console.print("\n[bold]Default Configuration:[/bold]")
    config = Config()

    table = Table()
    table.add_column("Category", style="cyan")
    table.add_column("Setting", style="white")
    table.add_column("Value", style="green")

    table.add_row("Game", "Table size", str(config.game.num_players))
    table.add_row("Game", "Small blind", str(config.game.small_blind))
    table.add_row("Game", "Big blind", str(config.game.big_blind))
    table.add_row("Game", "Starting chips", str(config.game.starting_chips))

    table.add_row("Network", "Architecture", config.network.architecture)
    table.add_row("Network", "Hidden dims", str(config.network.hidden_dims))
    table.add_row("Network", "Dropout", str(config.network.dropout))

    table.add_row("Genetic", "Population", str(config.genetic.population_size))
    table.add_row("Genetic", "Generations", str(config.genetic.generations))
    table.add_row("Genetic", "Elite fraction", str(config.genetic.elite_fraction))
    table.add_row("Genetic", "Mutation rate", str(config.genetic.mutation_rate))

    console.print(table)


@app.command()
def benchmark(
    games: int = typer.Option(100, "--games", "-g", help="Number of games"),
    table_size: int = typer.Option(9, "--table-size", "-t", help="Players per table"),
) -> None:
    """Benchmark game simulation speed."""
    import time

    console.print("\n[bold blue]Benchmark[/bold blue]")
    console.print("=" * 50)

    # Create random agents
    agents = [RandomAgent(name=f"Agent_{i}") for i in range(table_size)]

    console.print(f"Running {games} games with {table_size} players...")

    runner = GameRunner(SimulationConfig(num_games=games))

    start = time.time()
    results = runner.run_tournament(agents, num_games=games, show_progress=True)
    elapsed = time.time() - start

    console.print(f"\n[green]Completed in {elapsed:.2f}s[/green]")
    console.print(f"Games per second: {games / elapsed:.1f}")
    console.print(f"Hands per second: {results['total_hands'] / elapsed:.1f}")
    console.print(f"Average hands per game: {results['avg_hands_per_game']:.1f}")


def main() -> None:
    """Entry point."""
    app()


if __name__ == "__main__":
    main()
