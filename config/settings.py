"""Configuration settings for poker simulation."""

from dataclasses import dataclass, field
from pathlib import Path

import torch
import yaml


@dataclass
class GameConfig:
    """Game configuration."""

    num_players: int = 9
    small_blind: int = 1
    big_blind: int = 2
    starting_chips: int = 200
    max_hands_per_game: int = 100


@dataclass
class NetworkConfig:
    """Neural network configuration."""

    architecture: str = "mlp"  # mlp, deep_mlp, transformer
    hidden_dims: tuple[int, ...] = (256, 128, 64)
    dropout: float = 0.1
    # Transformer specific
    num_heads: int = 4
    num_layers: int = 2
    embed_dim: int = 128


@dataclass
class GeneticConfig:
    """Genetic algorithm configuration."""

    population_size: int = 100
    generations: int = 500
    elite_fraction: float = 0.1
    crossover_prob: float = 0.8
    mutation_rate: float = 0.1
    mutation_strength: float = 0.1
    tournament_size: int = 3
    games_per_evaluation: int = 50


@dataclass
class TrainingConfig:
    """Training configuration."""

    seed: int | None = None
    device: str = "cuda"  # cuda or cpu
    checkpoint_every: int = 10
    checkpoint_dir: str = "checkpoints"
    plots_dir: str = "plots"
    temperature: float = 1.0


@dataclass
class Config:
    """Complete configuration."""

    game: GameConfig = field(default_factory=GameConfig)
    network: NetworkConfig = field(default_factory=NetworkConfig)
    genetic: GeneticConfig = field(default_factory=GeneticConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)

    def get_device(self) -> torch.device:
        """Get torch device."""
        if self.training.device == "cuda" and torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")


def load_config(path: str | Path) -> Config:
    """Load configuration from YAML file."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path) as f:
        data = yaml.safe_load(f)

    config = Config()

    if "game" in data:
        config.game = GameConfig(**data["game"])
    if "network" in data:
        config.network = NetworkConfig(**data["network"])
    if "genetic" in data:
        config.genetic = GeneticConfig(**data["genetic"])
    if "training" in data:
        config.training = TrainingConfig(**data["training"])

    return config


def save_config(config: Config, path: str | Path) -> None:
    """Save configuration to YAML file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    data = {
        "game": {
            "num_players": config.game.num_players,
            "small_blind": config.game.small_blind,
            "big_blind": config.game.big_blind,
            "starting_chips": config.game.starting_chips,
            "max_hands_per_game": config.game.max_hands_per_game,
        },
        "network": {
            "architecture": config.network.architecture,
            "hidden_dims": list(config.network.hidden_dims),
            "dropout": config.network.dropout,
            "num_heads": config.network.num_heads,
            "num_layers": config.network.num_layers,
            "embed_dim": config.network.embed_dim,
        },
        "genetic": {
            "population_size": config.genetic.population_size,
            "generations": config.genetic.generations,
            "elite_fraction": config.genetic.elite_fraction,
            "crossover_prob": config.genetic.crossover_prob,
            "mutation_rate": config.genetic.mutation_rate,
            "mutation_strength": config.genetic.mutation_strength,
            "tournament_size": config.genetic.tournament_size,
            "games_per_evaluation": config.genetic.games_per_evaluation,
        },
        "training": {
            "seed": config.training.seed,
            "device": config.training.device,
            "checkpoint_every": config.training.checkpoint_every,
            "checkpoint_dir": config.training.checkpoint_dir,
            "plots_dir": config.training.plots_dir,
            "temperature": config.training.temperature,
        },
    }

    with open(path, "w") as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)


# Default configuration
DEFAULT_CONFIG = Config()
