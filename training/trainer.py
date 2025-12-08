"""RL training loop for PPO."""

from dataclasses import dataclass, field
from pathlib import Path
from random import Random
from typing import Callable

import torch
from rich.console import Console
from rich.table import Table
from tqdm import tqdm

from agents.base import BaseAgent
from agents.neural.encoding import StateEncoder
from agents.neural.network import ActorCriticMLP, NetworkConfig
from agents.random_agent import RandomAgent, TightAggressiveAgent
from agents.rl.agent import RLAgent
from poker.game import TexasHoldemGame
from poker.player import Player
from training.buffer import Experience, RolloutBuffer
from training.ppo import PPOConfig, PPOMetrics, PPOTrainer
from training.rewards import RewardConfig


@dataclass
class TrainingConfig:
    """Configuration for RL training."""

    # Game settings
    starting_chips: int = 200
    small_blind: int = 1
    big_blind: int = 2

    # Training loop
    total_games: int = 100_000
    games_per_update: int = 100  # Collect this many games before PPO update
    eval_every: int = 1000  # Evaluate every N games
    checkpoint_every: int = 5000  # Save checkpoint every N games
    log_every: int = 100  # Print log every N games

    # Opponent settings
    opponent_type: str = "mixed"  # random, tag, self, mixed

    # Model settings
    model_version: str = "v2"
    checkpoint_dir: str = "models/v2/checkpoints"


@dataclass
class TrainingStats:
    """Training statistics."""

    games_played: int = 0
    total_experiences: int = 0
    updates: int = 0

    # Rolling averages
    win_rate: float = 0.0
    avg_reward: float = 0.0
    avg_policy_loss: float = 0.0
    avg_value_loss: float = 0.0
    avg_entropy: float = 0.0

    # History for plotting
    win_rates: list[float] = field(default_factory=list)
    rewards: list[float] = field(default_factory=list)
    policy_losses: list[float] = field(default_factory=list)
    value_losses: list[float] = field(default_factory=list)


class RLTrainer:
    """Reinforcement learning trainer using PPO."""

    def __init__(
        self,
        network: ActorCriticMLP,
        ppo_config: PPOConfig | None = None,
        training_config: TrainingConfig | None = None,
        reward_config: RewardConfig | None = None,
        device: torch.device | None = None,
        console: Console | None = None,
        seed: int | None = None,
    ) -> None:
        self.device = device or torch.device("cpu")
        self.console = console or Console()
        self.rng = Random(seed)

        self.training_config = training_config or TrainingConfig()
        self.reward_config = reward_config or RewardConfig()
        self.ppo_config = ppo_config or PPOConfig()

        # Set buffer gamma/lambda from PPO config
        self.buffer = RolloutBuffer(
            gamma=self.ppo_config.gamma,
            lambda_=self.ppo_config.lambda_,
        )

        # Create network and trainer
        self.network = network.to(self.device)
        self.encoder = StateEncoder(self.device)
        self.ppo = PPOTrainer(network, self.ppo_config, self.device)

        # Create RL agent
        self.agent = RLAgent(
            network=self.network,
            encoder=self.encoder,
            device=self.device,
            reward_config=self.reward_config,
            name="RL_Agent",
        )

        # Stats
        self.stats = TrainingStats()

        # Opponent pool for self-play
        self.opponent_pool: list[BaseAgent] = []

    def _create_opponent(self, opponent_type: str | None = None) -> BaseAgent:
        """Create an opponent agent."""
        opponent_type = opponent_type or self.training_config.opponent_type

        if opponent_type == "random":
            return RandomAgent(name="Random")
        elif opponent_type == "tag":
            return TightAggressiveAgent(name="TAG")
        elif opponent_type == "self":
            # Clone current agent (frozen)
            clone = self.agent.clone()
            clone.eval_mode()
            clone.name = "Self"
            return clone
        elif opponent_type == "mixed":
            # Mix of opponents
            choice = self.rng.random()
            if choice < 0.4:
                return RandomAgent(name="Random")
            elif choice < 0.7:
                return TightAggressiveAgent(name="TAG")
            else:
                # Use clone from pool or current
                if self.opponent_pool and self.rng.random() < 0.5:
                    return self.rng.choice(self.opponent_pool)
                clone = self.agent.clone()
                clone.eval_mode()
                clone.name = "Self"
                return clone
        else:
            return RandomAgent(name="Random")

    def _run_game(self, opponent: BaseAgent) -> list[Experience]:
        """Run a single heads-up game and collect experiences.

        Args:
            opponent: Opponent agent

        Returns:
            List of experiences from all hands
        """
        config = self.training_config
        experiences = []

        # Create players (RL agent is player 0)
        players = [
            Player(id=0, chips=config.starting_chips, name=self.agent.name),
            Player(id=1, chips=config.starting_chips, name=opponent.name),
        ]

        agent_dict = {0: self.agent, 1: opponent}
        dealer_position = self.rng.randint(0, 1)

        max_hands = 100
        for hand_num in range(max_hands):
            # Check if game over
            active = [p for p in players if p.chips > 0]
            if len(active) <= 1:
                break

            # Get hole cards for RL agent before hand starts
            # We'll start the hand tracking in the agent
            initial_chips = players[0].chips

            # Run the hand
            game = TexasHoldemGame(
                players=active,
                dealer_position=dealer_position % len(active),
                small_blind=config.small_blind,
                big_blind=config.big_blind,
                seed=self.rng.randint(0, 2**31),
            )

            # Play the hand (this deals cards and runs betting)
            result = game.play(agent_dict)

            # Get hole cards from the player after the hand is done
            # The agent will have received them via TableState during decide()
            rl_player = next((p for p in active if p.id == 0), None)
            if rl_player and rl_player.hole_cards:
                # Note: start_hand should have been called via decide() already
                # We just need to make sure the hole_cards are set for reward computation
                if self.agent.collector.hole_cards is None:
                    self.agent.collector.hole_cards = rl_player.hole_cards

            # Collect experiences from RL agent
            won = 0 in result.winners
            pot_won = result.winnings.get(0, 0)
            folded = result.all_folded and 0 not in result.winners

            hand_experiences = self.agent.end_hand(
                won=won,
                pot_won=pot_won,
                showdown=result.showdown_hands is not None,
                folded=folded,
            )
            experiences.extend(hand_experiences)

            # Rotate dealer
            dealer_position = (dealer_position + 1) % 2

        return experiences

    def collect_rollout(self, num_games: int) -> None:
        """Collect experiences from multiple games into the buffer.

        Args:
            num_games: Number of games to play
        """
        self.agent.train_mode()

        total_wins = 0
        total_rewards = 0.0
        total_experiences = 0

        for _ in range(num_games):
            opponent = self._create_opponent()
            experiences = self._run_game(opponent)

            for exp in experiences:
                self.buffer.add(exp)
                total_rewards += exp.reward

            # Track wins (if we ended with more chips)
            if experiences and sum(e.reward for e in experiences) > 0:
                total_wins += 1

            total_experiences += len(experiences)

        # Update stats
        self.stats.games_played += num_games
        self.stats.total_experiences += total_experiences

        if num_games > 0:
            self.stats.win_rate = total_wins / num_games
        if total_experiences > 0:
            self.stats.avg_reward = total_rewards / total_experiences

    def update(self) -> PPOMetrics:
        """Run PPO update on collected experiences."""
        # Compute advantages
        self.buffer.compute_returns_and_advantages(
            normalize_advantages=self.ppo_config.normalize_advantages
        )

        # Run PPO update
        metrics = self.ppo.update(self.buffer)

        # Update stats
        self.stats.updates += 1
        self.stats.avg_policy_loss = metrics.policy_loss
        self.stats.avg_value_loss = metrics.value_loss
        self.stats.avg_entropy = metrics.entropy

        # Save to history
        self.stats.win_rates.append(self.stats.win_rate)
        self.stats.rewards.append(self.stats.avg_reward)
        self.stats.policy_losses.append(metrics.policy_loss)
        self.stats.value_losses.append(metrics.value_loss)

        # Clear buffer
        self.buffer.clear()

        return metrics

    def evaluate(self, num_games: int = 100) -> dict[str, float]:
        """Evaluate current agent against baselines.

        Args:
            num_games: Games per opponent type

        Returns:
            Dictionary of win rates against each opponent
        """
        self.agent.eval_mode()
        results = {}

        for opp_type in ["random", "tag"]:
            wins = 0
            for _ in range(num_games):
                opponent = self._create_opponent(opp_type)
                experiences = self._run_game(opponent)
                total_reward = sum(e.reward for e in experiences)
                if total_reward > 0:
                    wins += 1
            results[f"vs_{opp_type}"] = wins / num_games

        self.agent.train_mode()
        return results

    def save_checkpoint(self, path: str | Path) -> None:
        """Save training checkpoint."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        torch.save(
            {
                "network_state": self.network.state_dict(),
                "optimizer_state": self.ppo.optimizer.state_dict(),
                "stats": self.stats,
                "training_config": self.training_config,
                "ppo_config": self.ppo_config,
                "reward_config": self.reward_config,
            },
            path,
        )

    def load_checkpoint(self, path: str | Path) -> None:
        """Load training checkpoint."""
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.network.load_state_dict(checkpoint["network_state"])
        self.ppo.optimizer.load_state_dict(checkpoint["optimizer_state"])
        self.stats = checkpoint.get("stats", TrainingStats())

    def train(
        self,
        callback: Callable[[int, TrainingStats, PPOMetrics], None] | None = None,
        show_progress: bool = True,
    ) -> TrainingStats:
        """Run the full training loop.

        Args:
            callback: Optional callback(games, stats, metrics) after each update
            show_progress: Show progress bar

        Returns:
            Final training statistics
        """
        config = self.training_config

        total_updates = config.total_games // config.games_per_update
        checkpoint_dir = Path(config.checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.console.print(f"\n[bold blue]RL Training (PPO)[/bold blue]")
        self.console.print(f"Total games: {config.total_games:,}")
        self.console.print(f"Games per update: {config.games_per_update}")
        self.console.print(f"Opponent type: {config.opponent_type}")
        self.console.print(f"Device: {self.device}")
        self.console.print()

        iterator = range(total_updates)
        if show_progress:
            iterator = tqdm(iterator, desc="Training", unit="updates")

        try:
            for update_idx in iterator:
                games_so_far = (update_idx + 1) * config.games_per_update

                # Collect rollout
                self.collect_rollout(config.games_per_update)

                # PPO update
                metrics = self.update()

                # Update opponent pool periodically
                if update_idx > 0 and update_idx % 10 == 0:
                    clone = self.agent.clone()
                    clone.eval_mode()
                    clone.name = f"Self_{update_idx}"
                    self.opponent_pool.append(clone)
                    # Keep pool limited
                    if len(self.opponent_pool) > 10:
                        self.opponent_pool.pop(0)

                # Logging
                if games_so_far % config.log_every == 0:
                    if show_progress:
                        iterator.set_postfix(
                            win=f"{self.stats.win_rate:.1%}",
                            rew=f"{self.stats.avg_reward:.3f}",
                            pol=f"{metrics.policy_loss:.4f}",
                            val=f"{metrics.value_loss:.4f}",
                        )

                # Evaluation
                if games_so_far % config.eval_every == 0:
                    eval_results = self.evaluate(50)
                    if not show_progress:
                        self.console.print(
                            f"[{games_so_far:,}] "
                            f"Win rate: {self.stats.win_rate:.1%} | "
                            f"vs Random: {eval_results['vs_random']:.1%} | "
                            f"vs TAG: {eval_results['vs_tag']:.1%}"
                        )

                # Checkpointing
                if games_so_far % config.checkpoint_every == 0:
                    ckpt_path = checkpoint_dir / f"games_{games_so_far:06d}.pt"
                    self.save_checkpoint(ckpt_path)

                # Callback
                if callback:
                    callback(games_so_far, self.stats, metrics)

        except KeyboardInterrupt:
            self.console.print("\n[yellow]Training interrupted.[/yellow]")

        # Final checkpoint
        final_path = checkpoint_dir.parent / "final.pt"
        self.save_checkpoint(final_path)
        self.console.print(f"\n[green]Training complete![/green]")
        self.console.print(f"Final checkpoint: {final_path}")

        # Final evaluation
        self.console.print("\n[bold]Final Evaluation:[/bold]")
        eval_results = self.evaluate(100)
        table = Table()
        table.add_column("Opponent")
        table.add_column("Win Rate")
        for opp, rate in eval_results.items():
            table.add_row(opp, f"{rate:.1%}")
        self.console.print(table)

        return self.stats


def create_trainer(
    model_version: str = "v2",
    device: torch.device | None = None,
    network_config: NetworkConfig | None = None,
    ppo_config: PPOConfig | None = None,
    training_config: TrainingConfig | None = None,
    seed: int | None = None,
) -> RLTrainer:
    """Factory function to create an RL trainer.

    Args:
        model_version: Version string for checkpointing
        device: Torch device
        network_config: Network configuration
        ppo_config: PPO configuration
        training_config: Training configuration
        seed: Random seed

    Returns:
        Configured RLTrainer instance
    """
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    network_config = network_config or NetworkConfig()
    training_config = training_config or TrainingConfig(model_version=model_version)

    # Create network
    network = ActorCriticMLP(network_config)

    return RLTrainer(
        network=network,
        ppo_config=ppo_config,
        training_config=training_config,
        device=device,
        seed=seed,
    )
