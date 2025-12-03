"""Game runner for simulating poker games."""

from dataclasses import dataclass, field
from random import Random
from typing import TYPE_CHECKING

from tqdm import tqdm

from agents.base import BaseAgent
from poker.game import HandResult, TexasHoldemGame
from poker.player import Player

if TYPE_CHECKING:
    pass


@dataclass
class SimulationConfig:
    """Configuration for simulation."""

    small_blind: int = 1
    big_blind: int = 2
    starting_chips: int = 200
    max_hands_per_game: int = 100
    num_games: int = 100


@dataclass
class AgentBehaviorStats:
    """Behavior statistics for a single agent."""

    hands_played: int = 0
    vpip_count: int = 0  # Voluntarily put money in (raised or called preflop)
    pfr_count: int = 0  # Pre-flop raises
    total_raises: int = 0
    total_calls: int = 0
    total_folds: int = 0
    total_checks: int = 0
    total_3bets: int = 0  # Re-raises

    @property
    def vpip(self) -> float:
        """Voluntarily Put $ In Pot rate."""
        return self.vpip_count / self.hands_played if self.hands_played > 0 else 0.0

    @property
    def pfr(self) -> float:
        """Pre-Flop Raise rate."""
        return self.pfr_count / self.hands_played if self.hands_played > 0 else 0.0

    @property
    def aggression_factor(self) -> float:
        """(raises) / calls ratio."""
        return self.total_raises / self.total_calls if self.total_calls > 0 else float(self.total_raises)

    @property
    def fold_rate(self) -> float:
        """Fold rate."""
        total = self.total_raises + self.total_calls + self.total_folds + self.total_checks
        return self.total_folds / total if total > 0 else 0.0


@dataclass
class GameMetrics:
    """Aggregated metrics for a complete game."""

    hands_played: int = 0
    total_actions: int = 0
    showdowns: int = 0
    all_ins: int = 0
    actions_by_round: dict[str, int] = field(default_factory=dict)
    rounds_reached: dict[str, int] = field(default_factory=dict)
    winning_hands: dict[str, int] = field(default_factory=dict)
    total_pot_won: int = 0


@dataclass
class GameResult:
    """Result of a complete game (multiple hands)."""

    chip_deltas: dict[int, int]
    final_chips: dict[int, int]
    hands_played: int
    winner_id: int | None  # Player who won all chips, or None if max hands reached
    metrics: GameMetrics = field(default_factory=GameMetrics)


class GameRunner:
    """Run poker games between agents."""

    def __init__(
        self,
        config: SimulationConfig | None = None,
        seed: int | None = None,
    ) -> None:
        self.config = config or SimulationConfig()
        self.rng = Random(seed)

    def run_hand(
        self,
        players: list[Player],
        agents: dict[int, BaseAgent],
        dealer_position: int,
    ) -> HandResult:
        """Run a single hand of poker."""
        active_players = [p for p in players if p.chips > 0]

        if len(active_players) < 2:
            # Not enough players
            return HandResult(
                winners=[],
                winnings={},
                chip_changes={p.id: 0 for p in players},
                showdown_hands=None,
                winning_hand=None,
                all_folded=True,
            )

        game = TexasHoldemGame(
            players=active_players,
            dealer_position=dealer_position % len(active_players),
            small_blind=self.config.small_blind,
            big_blind=self.config.big_blind,
            seed=self.rng.randint(0, 2**31),
        )

        return game.play(agents)

    def run_game(
        self,
        agents: list[BaseAgent],
    ) -> GameResult:
        """Run a complete game (multiple hands) until one player wins all chips.

        Args:
            agents: List of agents to play

        Returns:
            GameResult with chip changes and outcome
        """
        num_players = len(agents)

        # Create players
        players = [
            Player(id=i, chips=self.config.starting_chips, name=agents[i].name)
            for i in range(num_players)
        ]

        # Create agent dict
        agent_dict = {i: agents[i] for i in range(num_players)}

        initial_chips = {p.id: p.chips for p in players}
        dealer_position = 0
        hands_played = 0

        # Initialize game metrics
        game_metrics = GameMetrics()

        for _ in range(self.config.max_hands_per_game):
            # Check if game over
            active_players = [p for p in players if p.chips > 0]
            if len(active_players) <= 1:
                break

            # Play a hand
            result = self.run_hand(players, agent_dict, dealer_position)
            hands_played += 1

            # Aggregate hand metrics into game metrics
            hand_metrics = result.metrics
            game_metrics.hands_played += 1
            game_metrics.total_actions += hand_metrics.total_actions
            game_metrics.total_pot_won += hand_metrics.pot_size

            if hand_metrics.went_to_showdown:
                game_metrics.showdowns += 1
            if hand_metrics.had_all_in:
                game_metrics.all_ins += 1

            # Track rounds reached
            round_name = hand_metrics.round_reached
            game_metrics.rounds_reached[round_name] = game_metrics.rounds_reached.get(round_name, 0) + 1

            # Track actions by round
            for rnd, count in hand_metrics.actions_by_round.items():
                game_metrics.actions_by_round[rnd] = game_metrics.actions_by_round.get(rnd, 0) + count

            # Track winning hands
            if hand_metrics.winning_hand_rank:
                rank = hand_metrics.winning_hand_rank
                game_metrics.winning_hands[rank] = game_metrics.winning_hands.get(rank, 0) + 1

            # Notify agents of result
            for i, agent in enumerate(agents):
                chip_delta = result.chip_changes.get(i, 0)
                won = i in result.winners
                agent.notify_hand_result(chip_delta, won)

            # Rotate dealer
            dealer_position = (dealer_position + 1) % num_players

        # Determine winner
        active_players = [p for p in players if p.chips > 0]
        winner_id = active_players[0].id if len(active_players) == 1 else None

        # Notify agents of game end
        for i, agent in enumerate(agents):
            agent.notify_game_end(players[i].chips)

        return GameResult(
            chip_deltas={p.id: p.chips - initial_chips[p.id] for p in players},
            final_chips={p.id: p.chips for p in players},
            hands_played=hands_played,
            winner_id=winner_id,
            metrics=game_metrics,
        )

    def run_tournament(
        self,
        agents: list[BaseAgent],
        num_games: int | None = None,
        show_progress: bool = True,
    ) -> dict:
        """Run multiple games and collect statistics.

        Args:
            agents: List of agents to play
            num_games: Number of games to run
            show_progress: Show progress bar

        Returns:
            Dictionary with tournament results
        """
        num_games = num_games or self.config.num_games

        total_chips_won: dict[int, int] = {i: 0 for i in range(len(agents))}
        games_won: dict[int, int] = {i: 0 for i in range(len(agents))}
        hands_won: dict[int, int] = {i: 0 for i in range(len(agents))}
        total_hands = 0

        # Aggregated metrics
        all_hands_per_game: list[int] = []
        total_actions = 0
        total_showdowns = 0
        total_all_ins = 0
        rounds_reached: dict[str, int] = {}
        winning_hands: dict[str, int] = {}
        total_pot_won = 0

        iterator = range(num_games)
        if show_progress:
            iterator = tqdm(iterator, desc="Running games", unit="games")

        for _ in iterator:
            result = self.run_game(agents)
            total_hands += result.hands_played
            all_hands_per_game.append(result.hands_played)

            # Aggregate game metrics
            gm = result.metrics
            total_actions += gm.total_actions
            total_showdowns += gm.showdowns
            total_all_ins += gm.all_ins
            total_pot_won += gm.total_pot_won

            for rnd, count in gm.rounds_reached.items():
                rounds_reached[rnd] = rounds_reached.get(rnd, 0) + count

            for rank, count in gm.winning_hands.items():
                winning_hands[rank] = winning_hands.get(rank, 0) + count

            for player_id, delta in result.chip_deltas.items():
                total_chips_won[player_id] += delta

            if result.winner_id is not None:
                games_won[result.winner_id] += 1

            for agent_id, agent in enumerate(agents):
                hands_won[agent_id] = agent.hands_won

        # Calculate derived stats
        import numpy as np
        hands_arr = np.array(all_hands_per_game) if all_hands_per_game else np.array([0])

        return {
            "num_games": num_games,
            "total_hands": total_hands,
            "avg_hands_per_game": total_hands / num_games if num_games > 0 else 0,
            "total_chips_won": total_chips_won,
            "games_won": games_won,
            "avg_chips_per_game": {
                i: total_chips_won[i] / num_games for i in range(len(agents))
            } if num_games > 0 else {},
            "win_rate": {
                i: games_won[i] / num_games for i in range(len(agents))
            } if num_games > 0 else {},
            # New metrics
            "hands_per_game_min": int(hands_arr.min()),
            "hands_per_game_max": int(hands_arr.max()),
            "hands_per_game_std": float(hands_arr.std()),
            "total_actions": total_actions,
            "actions_per_hand": total_actions / total_hands if total_hands > 0 else 0,
            "showdown_rate": total_showdowns / total_hands if total_hands > 0 else 0,
            "all_in_rate": total_all_ins / total_hands if total_hands > 0 else 0,
            "rounds_reached": rounds_reached,
            "winning_hands": winning_hands,
            "avg_pot_size": total_pot_won / total_hands if total_hands > 0 else 0,
            "bluff_success_rate": 1.0 - (total_showdowns / total_hands) if total_hands > 0 else 0,
        }

    def run_evaluation(
        self,
        agent: BaseAgent,
        opponents: list[BaseAgent],
        num_games: int = 100,
        show_progress: bool = True,
    ) -> dict:
        """Evaluate a single agent against a set of opponents.

        Args:
            agent: Agent to evaluate
            opponents: Opponent agents
            num_games: Number of games
            show_progress: Show progress bar

        Returns:
            Evaluation results
        """
        all_agents = [agent] + opponents
        results = self.run_tournament(all_agents, num_games, show_progress)

        return {
            "avg_chips_per_game": results["avg_chips_per_game"][0],
            "win_rate": results["win_rate"][0],
            "games_won": results["games_won"][0],
            "total_chips": results["total_chips_won"][0],
        }
