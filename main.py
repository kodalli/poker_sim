"""Poker simulation with genetic algorithm AI evolution."""

from pathlib import Path
from typing import TYPE_CHECKING, Optional

import torch
import typer
from rich.console import Console
from rich.table import Table

from agents.base import BaseAgent
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
from poker.player import ActionType, Player, PlayerAction
from poker.table import TableState
from simulation.runner import GameRunner, SimulationConfig
from simulation.statistics import StatisticsTracker
from utils.device import get_device, print_device_info
from models.manager import (
    create_metadata,
    create_version,
    get_checkpoint_dir,
    get_final_checkpoint,
    get_latest_version,
    get_model_summary,
    get_next_version,
    list_models,
    save_metadata,
    version_exists,
    copy_final_checkpoint,
)

if TYPE_CHECKING:
    from agents.neural.agent import NeuralAgent

app = typer.Typer(
    name="poker-sim",
    help="Texas Hold'em poker simulation with genetic algorithm AI evolution.",
)
console = Console()


@app.command()
def train(
    model: str = typer.Option("v1", "--model", "-m", help="Model version (e.g., v1, v2)"),
    generations: int = typer.Option(100, "--generations", "-g", help="Number of generations"),
    population: int = typer.Option(100, "--population", "-p", help="Population size"),
    architecture: str = typer.Option("mlp", "--arch", "-a", help="Network architecture (mlp, deep_mlp, transformer)"),
    games_per_eval: int = typer.Option(50, "--games", help="Games per fitness evaluation"),
    table_size: int = typer.Option(9, "--table-size", "-t", help="Players per table (2-10)"),
    resume: Optional[str] = typer.Option(None, "--resume", "-r", help="Resume from checkpoint"),
    description: str = typer.Option("", "--desc", "-d", help="Model description"),
    seed: Optional[int] = typer.Option(None, "--seed", "-s", help="Random seed"),
    cpu: bool = typer.Option(False, "--cpu", help="Force CPU (disable CUDA)"),
) -> None:
    """Train poker AI agents using genetic algorithm evolution."""
    console.print("\n[bold blue]Poker AI Training[/bold blue]")
    console.print("=" * 50)

    # Handle model versioning
    if version_exists(model):
        console.print(f"\n[yellow]Model {model} already exists![/yellow]")
        console.print("Options:")
        console.print("  1. Overwrite existing model")
        console.print(f"  2. Create new version ({get_next_version()})")
        console.print("  3. Cancel")

        choice = typer.prompt("Choose option", default="2")

        if choice == "1":
            console.print(f"[yellow]Will overwrite {model}[/yellow]")
        elif choice == "2":
            model = get_next_version()
            console.print(f"[green]Creating new version: {model}[/green]")
        else:
            console.print("[red]Cancelled.[/red]")
            raise typer.Exit(0)

    # Create model directory
    checkpoint_dir = str(get_checkpoint_dir(model))
    create_version(model, exist_ok=True)
    console.print(f"Model version: [cyan]{model}[/cyan]")

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
    plots_dir = f"models/{model}/plots"
    tracker = StatisticsTracker(output_dir=plots_dir)

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

    # Copy final checkpoint to model directory
    copy_final_checkpoint(model)

    # Get best fitness
    best = pop.get_best_individuals(1)[0]
    best_fitness = best.fitness

    # Save metadata
    metadata = create_metadata(
        version=model,
        architecture=architecture,
        generations=pop.generation,
        population_size=population,
        best_fitness=best_fitness,
        description=description or f"Trained for {generations} generations",
        config={
            "hidden_dims": list(network_config.hidden_dims),
            "small_blind": fitness_config.small_blind,
            "big_blind": fitness_config.big_blind,
            "starting_chips": fitness_config.starting_chips,
            "games_per_eval": games_per_eval,
            "table_size": table_size,
        },
    )
    save_metadata(model, metadata)

    # Generate plots
    console.print("\nGenerating plots...")
    tracker.plot_all()

    # Print summary
    tracker.print_summary()

    console.print(f"\n[green]Training complete![/green]")
    console.print(f"Model saved to: models/{model}/")
    console.print(f"Plots saved to: {plots_dir}/")


@app.command(name="train-rl")
def train_rl(
    model: str = typer.Option("v2", "--model", "-m", help="Model version (e.g., v2)"),
    games: int = typer.Option(100_000, "--games", "-g", help="Total games to train"),
    games_per_update: int = typer.Option(100, "--batch", "-b", help="Games per PPO update"),
    lr: float = typer.Option(3e-4, "--lr", help="Learning rate"),
    opponent: str = typer.Option("mixed", "--opponent", "-o", help="Opponent type: random, tag, self, mixed"),
    description: str = typer.Option("", "--desc", "-d", help="Model description"),
    seed: Optional[int] = typer.Option(None, "--seed", "-s", help="Random seed"),
    cpu: bool = typer.Option(False, "--cpu", help="Force CPU (disable CUDA)"),
) -> None:
    """Train poker AI using PPO reinforcement learning."""
    from agents.neural.network import ActorCriticMLP, NetworkConfig
    from training.ppo import PPOConfig
    from training.trainer import RLTrainer, TrainingConfig

    console.print("\n[bold blue]RL Training (PPO)[/bold blue]")
    console.print("=" * 50)

    # Handle model versioning
    if version_exists(model):
        console.print(f"\n[yellow]Model {model} already exists![/yellow]")
        console.print("Options:")
        console.print("  1. Overwrite existing model")
        console.print(f"  2. Create new version ({get_next_version()})")
        console.print("  3. Cancel")

        choice = typer.prompt("Choose option", default="2")

        if choice == "1":
            console.print(f"[yellow]Will overwrite {model}[/yellow]")
        elif choice == "2":
            model = get_next_version()
            console.print(f"[green]Creating new version: {model}[/green]")
        else:
            console.print("[red]Cancelled.[/red]")
            raise typer.Exit(0)

    # Create model directory
    create_version(model, exist_ok=True)
    checkpoint_dir = str(get_checkpoint_dir(model))

    # Setup device
    device = get_device(prefer_cuda=not cpu)
    console.print(f"Model version: [cyan]{model}[/cyan]")
    console.print(f"Device: [green]{device}[/green]")

    if device.type == "cuda":
        props = torch.cuda.get_device_properties(0)
        console.print(f"GPU: [green]{props.name}[/green] ({props.total_memory / 1024**3:.1f} GB)")

    # Create configurations
    network_config = NetworkConfig(hidden_dims=(256, 128, 64))

    ppo_config = PPOConfig(
        learning_rate=lr,
        gamma=0.99,
        lambda_=0.95,
        epsilon=0.2,
        ppo_epochs=4,
        batch_size=64,
        entropy_coef=0.01,
        value_coef=0.5,
    )

    training_config = TrainingConfig(
        total_games=games,
        games_per_update=games_per_update,
        eval_every=1000,
        checkpoint_every=5000,
        log_every=100,
        opponent_type=opponent,
        model_version=model,
        checkpoint_dir=checkpoint_dir,
    )

    # Create network and trainer
    network = ActorCriticMLP(network_config)
    console.print(f"Network parameters: [cyan]{network.num_parameters():,}[/cyan]")

    trainer = RLTrainer(
        network=network,
        ppo_config=ppo_config,
        training_config=training_config,
        device=device,
        console=console,
        seed=seed,
    )

    console.print(f"\n[bold]Starting RL training for {games:,} games...[/bold]\n")

    try:
        stats = trainer.train(show_progress=True)
    except KeyboardInterrupt:
        console.print("\n[yellow]Training interrupted. Saving checkpoint...[/yellow]")
        trainer.save_checkpoint(f"{checkpoint_dir}/interrupted.pt")

    # Save metadata
    metadata = create_metadata(
        version=model,
        architecture="actor_critic",
        generations=stats.updates,  # Use updates as "generations"
        population_size=1,  # RL uses single agent
        best_fitness=stats.win_rate,  # Use win rate as fitness
        description=description or f"RL training with PPO - {games:,} games",
        config={
            "hidden_dims": list(network_config.hidden_dims),
            "learning_rate": lr,
            "games_trained": games,
            "opponent_type": opponent,
            "training_method": "ppo",
        },
    )
    save_metadata(model, metadata)

    console.print(f"\n[green]Training complete![/green]")
    console.print(f"Model saved to: models/{model}/")


@app.command(name="train-rl-jax")
def train_rl_jax(
    model: str = typer.Option("v2-jax", "--model", "-m", help="Model version"),
    steps: int = typer.Option(1_000_000, "--steps", "-s", help="Total training steps"),
    parallel_games: int = typer.Option(1536, "--parallel", "-p", help="Parallel games on GPU (1536 optimal for RTX 4090)"),
    lr: float = typer.Option(3e-4, "--lr", help="Learning rate"),
    tensorboard: Optional[str] = typer.Option("logs/jax", "--tensorboard", "-t", help="TensorBoard log directory"),
    description: str = typer.Option("", "--desc", "-d", help="Model description"),
    seed: Optional[int] = typer.Option(42, "--seed", help="Random seed"),
    background: bool = typer.Option(False, "--background", "-b", help="Run in background"),
    # Advanced PPO options
    minibatches: int = typer.Option(8, "--minibatches", help="PPO minibatches (8 optimal for large batches)"),
    steps_per_update: Optional[int] = typer.Option(None, "--steps-per-update", help="Steps between PPO updates (auto-computed if not set)"),
    ppo_epochs: int = typer.Option(4, "--ppo-epochs", help="PPO epochs per update"),
    # Mixed opponent training (v3.1)
    mixed_opponents: bool = typer.Option(False, "--mixed-opponents", help="Train against mixed opponents (50%% self, 15%% random, 15%% call_station, 10%% TAG, 10%% LAG)"),
    # Historical self-play (v3.2)
    historical_selfplay: bool = typer.Option(False, "--historical-selfplay", help="Train with historical self-play and ELO tracking (40%% self, 40%% historical, 10%% call_station, 5%% random, 5%% TAG)"),
) -> None:
    """Train poker AI using JAX-accelerated PPO (GPU-optimized)."""
    try:
        import jax
        from training.jax_trainer import JAXTrainer, JAXTrainingConfig
        from training.jax_ppo import PPOConfig as JAXPPOConfig
    except ImportError as e:
        console.print(f"[red]JAX not available: {e}[/red]")
        console.print("Install JAX with: pip install 'jax[cuda12]' flax optax")
        raise typer.Exit(1)

    console.print("\n[bold blue]JAX-Accelerated RL Training[/bold blue]")
    console.print("=" * 50)

    # Handle model versioning
    if version_exists(model):
        console.print(f"\n[yellow]Model {model} already exists![/yellow]")
        console.print("Options:")
        console.print("  1. Overwrite existing model")
        console.print(f"  2. Create new version ({get_next_version()})")
        console.print("  3. Cancel")

        choice = typer.prompt("Choose option", default="2")

        if choice == "1":
            console.print(f"[yellow]Will overwrite {model}[/yellow]")
        elif choice == "2":
            model = get_next_version()
            console.print(f"[green]Creating new version: {model}[/green]")
        else:
            console.print("[red]Cancelled.[/red]")
            raise typer.Exit(0)

    # Create model directory
    create_version(model, exist_ok=True)
    checkpoint_dir = str(get_checkpoint_dir(model))

    # Check JAX device
    devices = jax.devices()
    console.print(f"Model version: [cyan]{model}[/cyan]")
    console.print(f"JAX devices: [green]{devices}[/green]")
    console.print(f"Parallel games: [cyan]{parallel_games:,}[/cyan]")
    console.print(f"Total steps: [cyan]{steps:,}[/cyan]")

    # Create configurations
    ppo_config = JAXPPOConfig(
        learning_rate=lr,
        gamma=0.99,
        lambda_=0.95,
        epsilon=0.2,
        ppo_epochs=ppo_epochs,
        num_minibatches=minibatches,
        entropy_coef=0.01,
        value_coef=0.5,
    )

    # Compute steps_per_update if not specified
    if steps_per_update is None:
        # Default: 2x parallel games for good batch diversity
        target_steps = parallel_games * 2
        max_steps = max(steps // 4, parallel_games)  # Ensure at least 4 updates
        computed_steps_per_update = min(target_steps, max_steps)
    else:
        computed_steps_per_update = steps_per_update

    training_config = JAXTrainingConfig(
        num_parallel_games=parallel_games,
        total_steps=steps,
        steps_per_update=computed_steps_per_update,
        checkpoint_every=50_000,
        checkpoint_dir=checkpoint_dir,
        tensorboard_dir=tensorboard,
        use_mixed_opponents=mixed_opponents,
        use_historical_selfplay=historical_selfplay,
    )

    # Create trainer
    trainer = JAXTrainer(
        ppo_config=ppo_config,
        training_config=training_config,
        seed=seed or 42,
        console=console,
    )

    # Run training
    if background:
        console.print("\n[yellow]Background mode not yet implemented.[/yellow]")
        console.print("For now, use: nohup uv run poker-sim train-rl-jax ... &")
        raise typer.Exit(1)

    console.print(f"\n[bold]Starting JAX training for {steps:,} steps...[/bold]\n")

    try:
        metrics = trainer.train(show_progress=True)
    except KeyboardInterrupt:
        console.print("\n[yellow]Training interrupted.[/yellow]")

    # Save metadata
    metadata = create_metadata(
        version=model,
        architecture="actor_critic_jax",
        generations=int(metrics.steps // training_config.steps_per_update),
        population_size=parallel_games,
        best_fitness=metrics.avg_reward,
        description=description or f"JAX training - {steps:,} steps",
        config={
            "learning_rate": lr,
            "total_steps": steps,
            "parallel_games": parallel_games,
            "training_method": "ppo_jax",
            "steps_per_second": metrics.steps_per_second,
            "mixed_opponents": mixed_opponents,
            "historical_selfplay": historical_selfplay,
        },
    )
    save_metadata(model, metadata)

    # Generate training plots
    csv_path = Path(tensorboard) / "metrics.csv"
    if csv_path.exists():
        from training.logging import plot_ppo_training
        plots_dir = Path(f"models/{model}/plots")
        console.print("\nGenerating training plots...")
        plot_ppo_training(csv_path, plots_dir)
        console.print(f"Plots saved to: [cyan]{plots_dir}/[/cyan]")

    console.print(f"\n[green]Training complete![/green]")
    console.print(f"Model saved to: models/{model}/")
    console.print(f"Steps/sec: [cyan]{metrics.steps_per_second:.0f}[/cyan]")


@app.command(name="evaluate-jax")
def evaluate_jax(
    model: str = typer.Option("v3", "--model", "-m", help="Model version (e.g., v3)"),
    games: int = typer.Option(10_000, "--games", "-g", help="Number of games per opponent"),
    opponents: str = typer.Option(
        "all", "--opponents", "-o",
        help="Opponent types (comma-separated or 'all'): random,call_station,tag,lag,rock"
    ),
    csv: Optional[str] = typer.Option(None, "--csv", help="Export results to CSV"),
    seed: int = typer.Option(42, "--seed", "-s", help="Random seed"),
) -> None:
    """Evaluate JAX-trained model against benchmark opponents.

    Examples:
        uv run python main.py evaluate-jax --model v3 --games 10000
        uv run python main.py evaluate-jax -m v3 -g 50000 -o tag,lag --csv results.csv
    """
    from evaluation.evaluator import ModelEvaluator, EvalConfig
    from evaluation.opponents import OPPONENT_TYPES

    console.print("\n[bold blue]JAX Model Evaluation[/bold blue]")
    console.print("=" * 50)

    # Find model checkpoint
    model_path = Path(f"models/{model}/final.pkl")
    if not model_path.exists():
        # Try alternate paths
        alt_path = Path(f"models/{model}/checkpoints")
        if alt_path.exists():
            checkpoints = list(alt_path.glob("*.pkl"))
            if checkpoints:
                model_path = max(checkpoints, key=lambda p: p.stat().st_mtime)

        if not model_path.exists():
            console.print(f"[red]Model {model} not found![/red]")
            console.print(f"Looked for: models/{model}/final.pkl")
            raise typer.Exit(1)

    # Parse opponents
    if opponents == "all":
        opponent_list = list(OPPONENT_TYPES.keys())
    else:
        opponent_list = [o.strip() for o in opponents.split(",")]
        for o in opponent_list:
            if o not in OPPONENT_TYPES:
                console.print(f"[red]Unknown opponent: {o}[/red]")
                console.print(f"Available: {list(OPPONENT_TYPES.keys())}")
                raise typer.Exit(1)

    console.print(f"Model: [cyan]{model}[/cyan]")
    console.print(f"Games per opponent: [cyan]{games:,}[/cyan]")
    console.print(f"Opponents: [cyan]{', '.join(opponent_list)}[/cyan]")

    # Create evaluator
    config = EvalConfig(num_games=games, output_csv=csv)
    evaluator = ModelEvaluator(
        model_path=model_path,
        config=config,
        seed=seed,
        console=console,
    )

    # Run evaluation
    console.print("\n[bold]Running evaluation...[/bold]\n")
    import time
    start = time.time()
    results = evaluator.evaluate_all(opponent_list)
    elapsed = time.time() - start

    # Display results
    console.print(f"\nCompleted in {elapsed:.1f}s\n")
    evaluator.print_results(results)

    # Export if requested
    if csv:
        evaluator.export_csv(results, csv)


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


class DebugNeuralAgent(BaseAgent):
    """Wrapper that stores AI decision details for display."""

    def __init__(
        self,
        agent: "NeuralAgent",
        console_: Console,
    ) -> None:
        super().__init__(agent.name)
        self._agent = agent
        self._console = console_
        self._last_action: PlayerAction | None = None
        self._last_probs: dict[ActionType, float] | None = None
        self._last_bet_fraction: float | None = None
        self.last_decision_panel = None  # Store panel for display by HumanAgent

    def decide(self, table_state: TableState) -> PlayerAction:
        # Get action from underlying agent
        action = self._agent.decide(table_state)

        # Get probabilities for display
        probs = self._agent.get_action_distribution(table_state)

        # Calculate bet fraction if raising
        bet_fraction = None
        if action.action_type == ActionType.RAISE:
            bet_range = table_state.max_raise - table_state.min_raise
            if bet_range > 0:
                bet_fraction = (action.amount - table_state.min_raise) / bet_range

        # Store for display
        self._last_action = action
        self._last_probs = probs
        self._last_bet_fraction = bet_fraction

        # Create and store the panel (don't print - let HumanAgent display it)
        from ui.display import render_ai_decision
        self.last_decision_panel = render_ai_decision(action, probs, bet_fraction)

        return action

    def clear_decision(self) -> None:
        """Clear the stored decision panel."""
        self.last_decision_panel = None

    def notify_hand_result(self, chip_delta: int, won: bool) -> None:
        self._agent.notify_hand_result(chip_delta, won)

    def notify_game_end(self, final_chips: int) -> None:
        self._agent.notify_game_end(final_chips)


def _run_human_game(
    model_version: str,
    max_hands: int,
    cpu: bool,
) -> None:
    """Run interactive human vs AI game with TUI-style display."""
    from poker.game import TexasHoldemGame
    from agents.human_agent import HumanAgent
    from ui.display import print_divider, render_hand_result

    device = get_device(prefer_cuda=not cpu)

    # Load AI from model version
    checkpoint_path = get_final_checkpoint(model_version)
    if not checkpoint_path.exists():
        console.print(f"[red]Model {model_version} not found![/red]")
        available = list_models()
        if available:
            console.print(f"Available models: {', '.join(available)}")
        else:
            console.print("[yellow]Run 'uv run poker-sim train' first to train a model.[/yellow]")
        raise typer.Exit(1)

    console.print(f"Loading model: [cyan]{model_version}[/cyan]")
    pop = Population(size=1, device=device)
    pop.load(str(checkpoint_path))
    best = pop.get_best_individuals(1)[0]

    # Create agents
    ai_agent = pop.get_agent(best, temperature=0.3)
    ai_agent.name = model_version
    debug_ai = DebugNeuralAgent(ai_agent, console)

    human_agent = HumanAgent(name="You", console=console)
    human_agent.use_alternate_screen = True
    # Connect human agent to AI's decision panel
    human_agent.set_ai_decision_getter(lambda: debug_ai.last_decision_panel)

    # Game settings
    small_blind = 1
    big_blind = 2
    starting_chips = 200

    # Create players (human is player 0, AI is player 1)
    players = [
        Player(id=0, chips=starting_chips, name="You"),
        Player(id=1, chips=starting_chips, name="AI"),
    ]

    agents_dict = {0: human_agent, 1: debug_ai}
    agents_list = [human_agent, debug_ai]

    dealer_position = 0
    hands_played = 0
    game_interrupted = False

    console.print("\n[bold green]Starting game...[/bold green]")
    console.print(f"Starting chips: {starting_chips} | Blinds: {small_blind}/{big_blind}")
    console.print("[dim]Type 'q' to quit at any time. Press Enter to continue...[/dim]")
    input()

    # Enter alternate screen buffer for TUI-style display
    with console.screen():
        try:
            while hands_played < max_hands:
                # Check if game over
                active_players = [p for p in players if p.chips > 0]
                if len(active_players) <= 1:
                    break

                hands_played += 1

                # Update human agent context
                human_agent.hand_number = hands_played
                debug_ai.clear_decision()

                # Play a hand
                game = TexasHoldemGame(
                    players=active_players,
                    dealer_position=dealer_position % len(active_players),
                    small_blind=small_blind,
                    big_blind=big_blind,
                )

                result = game.play(agents_dict)

                # Show hand result (clear and display result screen)
                console.clear()
                human_delta = result.chip_changes.get(0, 0)
                human_won = 0 in result.winners

                # Get winning hand description if showdown
                winning_hand = None
                if result.showdown_hands and result.winning_hand:
                    winning_hand = str(result.winning_hand.rank.name.replace("_", " ").title())

                # Display result screen
                console.print(f"\n[bold blue]POKER - Hand #{hands_played} Result[/bold blue]\n")
                result_panel = render_hand_result(
                    won=human_won,
                    chip_delta=human_delta,
                    my_chips=players[0].chips,
                    opponent_chips=players[1].chips,
                    showdown=not result.all_folded,
                    winning_hand=winning_hand,
                )
                console.print(result_panel)

                # Notify agents
                for i, agent in enumerate(agents_list):
                    chip_delta = result.chip_changes.get(i, 0)
                    won = i in result.winners
                    agent.notify_hand_result(chip_delta, won)

                # Rotate dealer
                dealer_position = (dealer_position + 1) % 2

                # Pause between hands
                if hands_played < max_hands and len([p for p in players if p.chips > 0]) > 1:
                    console.print("\n[dim]Press Enter for next hand (or 'q' to quit)...[/dim]")
                    try:
                        inp = input()
                        if inp.lower() in ('q', 'quit', 'exit'):
                            break
                    except EOFError:
                        break

        except KeyboardInterrupt:
            game_interrupted = True

    # Back to normal screen - show final results
    if game_interrupted:
        console.print("\n[yellow]Game interrupted.[/yellow]")

    console.print("\n")
    print_divider(console, "=")
    console.print("[bold]Final Results[/bold]")
    print_divider(console)

    for i, agent in enumerate(agents_list):
        chips = players[i].chips
        delta = chips - starting_chips
        delta_str = f"+{delta}" if delta > 0 else str(delta)
        delta_style = "green" if delta > 0 else "red" if delta < 0 else "white"
        console.print(f"{agent.name}: {chips} chips ([{delta_style}]{delta_str}[/{delta_style}])")

    # Determine winner
    if players[0].chips > players[1].chips:
        console.print("\n[bold green]You win![/bold green]")
    elif players[1].chips > players[0].chips:
        console.print("\n[bold red]AI wins![/bold red]")
    else:
        console.print("\n[bold yellow]It's a tie![/bold yellow]")


@app.command()
def play(
    model: Optional[str] = typer.Option(None, "--model", "-m", help="Model version (default: latest)"),
    num_players: int = typer.Option(6, "--players", "-p", help="Number of players (2-10)"),
    hands: int = typer.Option(10, "--hands", "-n", help="Number of hands to play"),
    cpu: bool = typer.Option(False, "--cpu", help="Force CPU"),
    human: bool = typer.Option(False, "--human", "-H", help="Play as human against AI"),
) -> None:
    """Play poker - watch AI or play as human."""
    # Default to latest model
    if model is None:
        model = get_latest_version()
        if model is None:
            console.print("[red]No models found![/red]")
            console.print("[yellow]Run 'uv run poker-sim train' first to train a model.[/yellow]")
            raise typer.Exit(1)

    if human:
        # Human vs AI mode
        console.print("\n[bold blue]Human vs AI Poker[/bold blue]")
        console.print("=" * 50)
        _run_human_game(model, hands, cpu)
        return

    # Watch mode (existing behavior)
    console.print("\n[bold blue]Poker Game[/bold blue]")
    console.print("=" * 50)

    device = get_device(prefer_cuda=not cpu)

    # Load model
    checkpoint_path = get_final_checkpoint(model)
    if not checkpoint_path.exists():
        console.print(f"[red]Model {model} not found![/red]")
        raise typer.Exit(1)

    pop = Population(size=1, device=device)
    pop.load(str(checkpoint_path))
    best = pop.get_best_individuals(1)[0]

    agents = [pop.get_agent(best, temperature=0.5)]
    agents[0].name = model

    # Fill rest with opponents
    for i in range(num_players - 1):
        if i % 2 == 0:
            agents.append(TightAggressiveAgent(name=f"TAG_{i+1}"))
        else:
            agents.append(RandomAgent(name=f"Random_{i+1}"))

    # Run game
    console.print(f"Model: [cyan]{model}[/cyan]")
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


@app.command(name="models")
def list_models_cmd() -> None:
    """List all available model versions."""
    console.print("\n[bold blue]Available Models[/bold blue]")
    console.print("=" * 50)

    versions = list_models()
    if not versions:
        console.print("[yellow]No models found.[/yellow]")
        console.print("Run 'uv run poker-sim train' to train a model.")
        return

    table = Table()
    table.add_column("Version", style="cyan")
    table.add_column("Architecture", style="white")
    table.add_column("Generations", style="white")
    table.add_column("Best Fitness", style="green")
    table.add_column("Created", style="dim")
    table.add_column("Description", style="white")

    for version in versions:
        summary = get_model_summary(version)
        table.add_row(
            summary["version"],
            str(summary["architecture"]),
            str(summary["generations"]),
            f"{summary['best_fitness']:.2f}" if isinstance(summary["best_fitness"], float) else str(summary["best_fitness"]),
            summary["created"][:10] if summary["created"] != "unknown" else "?",
            summary["description"][:30] + "..." if len(summary.get("description", "")) > 30 else summary.get("description", ""),
        )

    console.print(table)
    console.print(f"\nTotal: {len(versions)} model(s)")


@app.command()
def tournament(
    model_versions: list[str] = typer.Argument(..., help="Model versions to compete (e.g., v1 v2 v3)"),
    games: int = typer.Option(100, "--games", "-g", help="Games per matchup"),
    mode: str = typer.Option("round-robin", "--mode", help="Tournament mode: 'matchup' (1v1) or 'round-robin'"),
    cpu: bool = typer.Option(False, "--cpu", help="Force CPU"),
) -> None:
    """Run a tournament between model versions."""
    from simulation.tournament import run_matchup, run_tournament as run_tourney

    console.print("\n[bold blue]Model Tournament[/bold blue]")
    console.print("=" * 50)

    # Validate models exist
    for v in model_versions:
        if not version_exists(v):
            console.print(f"[red]Model {v} not found![/red]")
            available = list_models()
            if available:
                console.print(f"Available models: {', '.join(available)}")
            raise typer.Exit(1)

    device = torch.device("cpu") if cpu else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    console.print(f"Device: [green]{device}[/green]")

    if mode == "matchup" and len(model_versions) == 2:
        # Simple 1v1 matchup
        m1, m2 = model_versions
        console.print(f"\n[bold]Matchup: {m1} vs {m2}[/bold]")
        console.print(f"Playing {games} games...\n")

        result = run_matchup(m1, m2, games, device)

        # Display results
        table = Table(title="Matchup Results")
        table.add_column("Model", style="cyan")
        table.add_column("Wins", style="green")
        table.add_column("Win Rate", style="white")
        table.add_column("Avg Chips", style="white")

        table.add_row(
            m1,
            str(result.model1_wins),
            f"{result.model1_win_rate:.1%}",
            f"{result.model1_avg_chips:+.1f}",
        )
        table.add_row(
            m2,
            str(result.model2_wins),
            f"{result.model2_win_rate:.1%}",
            f"{result.model2_avg_chips:+.1f}",
        )

        console.print(table)

        if result.model1_wins > result.model2_wins:
            console.print(f"\n[bold green]Winner: {m1}[/bold green]")
        elif result.model2_wins > result.model1_wins:
            console.print(f"\n[bold green]Winner: {m2}[/bold green]")
        else:
            console.print("\n[bold yellow]Draw![/bold yellow]")

    else:
        # Round-robin tournament
        console.print(f"\n[bold]Round-Robin Tournament[/bold]")
        console.print(f"Models: {', '.join(model_versions)}")
        console.print(f"Games per matchup: {games}\n")

        result = run_tourney(model_versions, games, device)

        # Display leaderboard
        table = Table(title="Tournament Leaderboard")
        table.add_column("Rank", style="bold")
        table.add_column("Model", style="cyan")
        table.add_column("W", style="green")
        table.add_column("L", style="red")
        table.add_column("D", style="yellow")
        table.add_column("Win Rate", style="white")
        table.add_column("Total Chips", style="white")
        table.add_column("Avg Chips", style="white")

        leaderboard = result.get_leaderboard()
        for rank, (model, stats) in enumerate(leaderboard, 1):
            table.add_row(
                str(rank),
                model,
                str(stats["wins"]),
                str(stats["losses"]),
                str(stats["draws"]),
                f"{stats['win_rate']:.1%}",
                f"{stats['total_chips']:+,}",
                f"{stats['avg_chips']:+.1f}",
            )

        console.print(table)

        # Show champion
        if leaderboard:
            champion = leaderboard[0][0]
            console.print(f"\n[bold green]Champion: {champion}[/bold green]")


def main() -> None:
    """Entry point."""
    app()


def play_human() -> None:
    """Entry point for human vs AI play mode."""
    # Get latest model
    model = get_latest_version()
    if model is None:
        console.print("[red]No models found![/red]")
        console.print("[yellow]Run 'uv run poker-sim train' first to train a model.[/yellow]")
        raise typer.Exit(1)

    console.print("\n[bold blue]Human vs AI Poker[/bold blue]")
    console.print("=" * 50)
    _run_human_game(
        model_version=model,
        max_hands=50,
        cpu=False,
    )


if __name__ == "__main__":
    main()
