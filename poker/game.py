"""Texas Hold'em game orchestration."""

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Callable

from poker.betting import BettingRound
from poker.cards import Card, Deck
from poker.hand_evaluator import EvaluatedHand, HandEvaluator
from poker.player import ActionType, Player, PlayerAction, PlayerStatus
from poker.pot import PotManager
from poker.table import ActionInfo, GameRound, TableState, create_table_state

if TYPE_CHECKING:
    from agents.base import BaseAgent


@dataclass
class HandMetrics:
    """Metrics collected during a single hand."""

    total_actions: int = 0
    actions_by_round: dict[str, int] = field(default_factory=dict)
    round_reached: str = "preflop"
    went_to_showdown: bool = False
    had_all_in: bool = False
    pot_size: int = 0
    winning_hand_rank: str | None = None
    # Per-player action tracking for behavior stats
    player_actions: dict[int, dict[str, int]] = field(default_factory=dict)


@dataclass
class HandResult:
    """Result of a completed hand."""

    winners: list[int]  # Player IDs who won
    winnings: dict[int, int]  # Player ID -> amount won
    chip_changes: dict[int, int]  # Player ID -> net chip change
    showdown_hands: dict[int, EvaluatedHand] | None  # If showdown occurred
    winning_hand: EvaluatedHand | None
    all_folded: bool  # True if everyone folded to one player
    metrics: HandMetrics = field(default_factory=HandMetrics)


class TexasHoldemGame:
    """Orchestrate a single hand of Texas Hold'em."""

    def __init__(
        self,
        players: list[Player],
        dealer_position: int,
        small_blind: int = 1,
        big_blind: int = 2,
        seed: int | None = None,
    ) -> None:
        if len(players) < 2:
            raise ValueError("Need at least 2 players")
        if len(players) > 10:
            raise ValueError("Maximum 10 players")

        self.players = players
        self.dealer_position = dealer_position % len(players)
        self.small_blind = small_blind
        self.big_blind = big_blind

        self.deck = Deck(seed)
        self.pot_manager = PotManager()
        self.community_cards: list[Card] = []
        self.current_round = GameRound.PRE_FLOP
        self.action_history: list[ActionInfo] = []

        # Track initial chips for calculating changes
        self._initial_chips = {p.id: p.chips for p in players}

        # Metrics tracking
        self._metrics = HandMetrics()
        self._metrics.player_actions = {p.id: {"raises": 0, "calls": 0, "folds": 0, "checks": 0} for p in players}

    def _get_active_players(self) -> list[Player]:
        """Get players still active in the hand."""
        return [p for p in self.players if p.is_in_hand()]

    def _get_sb_position(self) -> int:
        """Get small blind position."""
        if len(self.players) == 2:
            # Heads-up: dealer is small blind
            return self.dealer_position
        return (self.dealer_position + 1) % len(self.players)

    def _get_bb_position(self) -> int:
        """Get big blind position."""
        if len(self.players) == 2:
            # Heads-up: non-dealer is big blind
            return (self.dealer_position + 1) % len(self.players)
        return (self.dealer_position + 2) % len(self.players)

    def _post_blinds(self) -> None:
        """Post small and big blinds."""
        sb_player = self.players[self._get_sb_position()]
        bb_player = self.players[self._get_bb_position()]

        sb_amount = sb_player.bet(self.small_blind)
        bb_amount = bb_player.bet(self.big_blind)

        # Record blind actions
        self.action_history.append(
            ActionInfo(
                player_id=sb_player.id,
                player_name=sb_player.name,
                action=PlayerAction(ActionType.RAISE, sb_amount),
                round=GameRound.PRE_FLOP,
            )
        )
        self.action_history.append(
            ActionInfo(
                player_id=bb_player.id,
                player_name=bb_player.name,
                action=PlayerAction(ActionType.RAISE, bb_amount),
                round=GameRound.PRE_FLOP,
            )
        )

    def _deal_hole_cards(self) -> None:
        """Deal 2 hole cards to each player."""
        self.deck.shuffle()
        for player in self.players:
            if player.chips > 0:
                cards = self.deck.deal(2)
                player.hole_cards = (cards[0], cards[1])
                player.status = PlayerStatus.ACTIVE
            else:
                player.status = PlayerStatus.WAITING

    def _deal_flop(self) -> None:
        """Deal the flop (3 community cards)."""
        self.deck.deal(1)  # Burn card
        self.community_cards.extend(self.deck.deal(3))
        self.current_round = GameRound.FLOP

    def _deal_turn(self) -> None:
        """Deal the turn (1 community card)."""
        self.deck.deal(1)  # Burn card
        self.community_cards.append(self.deck.deal_one())
        self.current_round = GameRound.TURN

    def _deal_river(self) -> None:
        """Deal the river (1 community card)."""
        self.deck.deal(1)  # Burn card
        self.community_cards.append(self.deck.deal_one())
        self.current_round = GameRound.RIVER

    def _get_starting_position(self) -> int:
        """Get starting position for betting round."""
        if self.current_round == GameRound.PRE_FLOP:
            # Pre-flop: action starts left of big blind
            if len(self.players) == 2:
                return self.dealer_position  # Heads-up: dealer acts first pre-flop
            return (self._get_bb_position() + 1) % len(self.players)
        else:
            # Post-flop: action starts left of dealer
            return (self.dealer_position + 1) % len(self.players)

    def _create_table_state(
        self,
        observer_id: int,
        valid_actions: list[ActionType] | None = None,
        call_amount: int = 0,
        min_raise: int = 0,
        max_raise: int = 0,
    ) -> TableState:
        """Create table state for a player."""
        # Calculate current bet from players
        current_bet = max((p.current_bet for p in self.players), default=0)

        return create_table_state(
            players=self.players,
            community_cards=self.community_cards,
            pot_total=self.pot_manager.total() + sum(p.current_bet for p in self.players),
            current_bet=current_bet,
            round=self.current_round,
            dealer_position=self.dealer_position,
            small_blind=self.small_blind,
            big_blind=self.big_blind,
            action_history=self.action_history,
            observer_id=observer_id,
            valid_actions=valid_actions,
            call_amount=call_amount,
            min_raise=min_raise,
            max_raise=max_raise,
        )

    def _run_betting_round(
        self,
        agents: dict[int, "BaseAgent"],
    ) -> bool:
        """Run a betting round. Returns False if only one player remains."""
        # Find first active player for starting position
        start_pos = self._get_starting_position()
        while not self.players[start_pos].can_act():
            start_pos = (start_pos + 1) % len(self.players)
            # Safety check
            if all(not p.can_act() for p in self.players):
                return True

        current_bet = max((p.current_bet for p in self.players), default=0)

        betting = BettingRound(
            players=self.players,
            starting_player_idx=start_pos,
            big_blind=self.big_blind,
            current_bet=current_bet,
        )

        while not betting.is_complete():
            player_idx = betting.get_next_player_idx()
            if player_idx is None:
                break

            player = self.players[player_idx]
            agent = agents[player.id]

            # Get valid actions and create state
            valid_actions = betting.get_valid_actions(player)
            call_amount = betting.get_call_amount(player)
            min_raise = betting.get_min_raise_amount()
            max_raise = betting.get_max_raise_amount(player)

            table_state = self._create_table_state(
                observer_id=player.id,
                valid_actions=valid_actions,
                call_amount=call_amount,
                min_raise=min_raise,
                max_raise=max_raise,
            )

            # Get action from agent
            action = agent.decide(table_state)

            # Validate and sanitize action
            action = self._validate_action(player, action, valid_actions, call_amount, min_raise, max_raise)

            # Apply action
            betting.apply_action(player_idx, action)

            # Record action
            self.action_history.append(
                ActionInfo(
                    player_id=player.id,
                    player_name=player.name,
                    action=action,
                    round=self.current_round,
                )
            )

            # Track metrics
            self._metrics.total_actions += 1
            round_name = self.current_round.name.lower()  # "pre_flop", "flop", etc.
            self._metrics.actions_by_round[round_name] = self._metrics.actions_by_round.get(round_name, 0) + 1

            # Track per-player action types
            if action.action_type == ActionType.RAISE:
                self._metrics.player_actions[player.id]["raises"] += 1
            elif action.action_type == ActionType.CALL:
                self._metrics.player_actions[player.id]["calls"] += 1
            elif action.action_type == ActionType.FOLD:
                self._metrics.player_actions[player.id]["folds"] += 1
            elif action.action_type == ActionType.CHECK:
                self._metrics.player_actions[player.id]["checks"] += 1

            # Check for all-in
            if player.chips == 0 and player.status == PlayerStatus.ALL_IN:
                self._metrics.had_all_in = True

            # Check if only one player remains
            if betting.only_one_player_remaining():
                return False

        # Collect bets into pot
        self.pot_manager.collect_bets(self.players)

        # Reset bets for next round
        for player in self.players:
            player.reset_for_round()

        return len(self._get_active_players()) > 0 or len([p for p in self.players if p.is_in_hand()]) > 1

    def _validate_action(
        self,
        player: Player,
        action: PlayerAction,
        valid_actions: list[ActionType],
        call_amount: int,
        min_raise: int,
        max_raise: int,
    ) -> PlayerAction:
        """Validate and sanitize an action."""
        # If action type not valid, default to fold
        if action.action_type not in valid_actions:
            # Try to find a sensible default
            if ActionType.CHECK in valid_actions:
                return PlayerAction(ActionType.CHECK)
            if ActionType.CALL in valid_actions:
                return PlayerAction(ActionType.CALL, call_amount)
            return PlayerAction(ActionType.FOLD)

        # Validate raise amount
        if action.action_type == ActionType.RAISE:
            amount = action.amount
            if amount < min_raise:
                amount = min_raise
            if amount > max_raise:
                amount = max_raise
            return PlayerAction(ActionType.RAISE, amount)

        return action

    def _determine_winners(self) -> HandResult:
        """Determine winners at showdown."""
        active_players = [p for p in self.players if p.is_in_hand()]

        # Record final pot size
        self._metrics.pot_size = self.pot_manager.total() + sum(p.current_bet for p in self.players)

        # If only one player, they win without showdown
        if len(active_players) == 1:
            winner = active_players[0]
            winnings = self.pot_manager.distribute([[winner.id]] * len(self.pot_manager.pots))
            winner.win(winnings.get(winner.id, 0))

            # Update metrics - no showdown
            self._metrics.went_to_showdown = False
            self._metrics.winning_hand_rank = None

            return HandResult(
                winners=[winner.id],
                winnings=winnings,
                chip_changes={
                    p.id: p.chips - self._initial_chips[p.id] for p in self.players
                },
                showdown_hands=None,
                winning_hand=None,
                all_folded=True,
                metrics=self._metrics,
            )

        # Evaluate all hands
        hands: dict[int, EvaluatedHand] = {}
        for player in active_players:
            if player.hole_cards:
                hands[player.id] = HandEvaluator.evaluate(
                    player.hole_cards, self.community_cards
                )

        # Determine winners for each pot
        winners_by_pot: list[list[int]] = []
        for pot in self.pot_manager.pots:
            eligible_hands = [
                (pid, hands[pid])
                for pid in pot.eligible_players
                if pid in hands
            ]
            if eligible_hands:
                best_hand = max(h for _, h in eligible_hands)
                pot_winners = [pid for pid, h in eligible_hands if h == best_hand]
                winners_by_pot.append(pot_winners)
            else:
                winners_by_pot.append([])

        # Distribute pots
        winnings = self.pot_manager.distribute(winners_by_pot)

        # Award winnings
        for player_id, amount in winnings.items():
            for p in self.players:
                if p.id == player_id:
                    p.win(amount)
                    break

        # Get overall winners
        all_winners = list(winnings.keys())
        best_overall = max(hands.values()) if hands else None

        # Update metrics - went to showdown
        self._metrics.went_to_showdown = True
        if best_overall:
            self._metrics.winning_hand_rank = best_overall.rank.name.lower()

        return HandResult(
            winners=all_winners,
            winnings=winnings,
            chip_changes={
                p.id: p.chips - self._initial_chips[p.id] for p in self.players
            },
            showdown_hands=hands,
            winning_hand=best_overall,
            all_folded=False,
            metrics=self._metrics,
        )

    def play(self, agents: dict[int, "BaseAgent"]) -> HandResult:
        """Play a complete hand.

        Args:
            agents: Dict mapping player ID to agent.

        Returns:
            HandResult with winners and chip changes.
        """
        # Reset players for new hand
        for player in self.players:
            player.reset_for_hand()

        self.pot_manager.reset()
        self.community_cards = []
        self.action_history = []

        # Deal hole cards
        self._deal_hole_cards()

        # Post blinds
        self._post_blinds()

        # Pre-flop betting
        self.current_round = GameRound.PRE_FLOP
        self._metrics.round_reached = "preflop"
        if not self._run_betting_round(agents):
            return self._determine_winners()

        # Check if all-in and no more betting needed
        active_players = [p for p in self.players if p.status == PlayerStatus.ACTIVE]

        # Flop
        self._deal_flop()
        self._metrics.round_reached = "flop"
        if len(active_players) > 1:
            if not self._run_betting_round(agents):
                return self._determine_winners()
            active_players = [p for p in self.players if p.status == PlayerStatus.ACTIVE]

        # Turn
        self._deal_turn()
        self._metrics.round_reached = "turn"
        if len(active_players) > 1:
            if not self._run_betting_round(agents):
                return self._determine_winners()
            active_players = [p for p in self.players if p.status == PlayerStatus.ACTIVE]

        # River
        self._deal_river()
        self._metrics.round_reached = "river"
        if len(active_players) > 1:
            if not self._run_betting_round(agents):
                return self._determine_winners()

        # Showdown
        self.current_round = GameRound.SHOWDOWN
        self._metrics.round_reached = "showdown"
        return self._determine_winners()
