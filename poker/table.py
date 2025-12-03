"""Table state representation for poker."""

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import TYPE_CHECKING

from poker.cards import Card
from poker.player import ActionType, PlayerAction, PlayerStatus

if TYPE_CHECKING:
    from poker.player import Player


class GameRound(Enum):
    """Current round of the poker hand."""

    PRE_FLOP = auto()
    FLOP = auto()
    TURN = auto()
    RIVER = auto()
    SHOWDOWN = auto()

    def __str__(self) -> str:
        return self.name.replace("_", " ").title()


@dataclass
class PlayerState:
    """State of a single player visible to all."""

    id: int
    name: str
    chips: int
    status: PlayerStatus
    current_bet: int
    total_invested: int
    is_dealer: bool
    is_small_blind: bool
    is_big_blind: bool
    position: int  # 0 = dealer, 1 = SB, 2 = BB, etc.


@dataclass
class ActionInfo:
    """Information about an action taken."""

    player_id: int
    player_name: str
    action: PlayerAction
    round: GameRound


@dataclass
class TableState:
    """Complete state of the table visible to a specific player."""

    # Public information
    community_cards: list[Card]
    pot_total: int
    current_bet: int
    round: GameRound
    dealer_position: int
    small_blind: int
    big_blind: int
    num_players: int

    # All players' visible states
    players: list[PlayerState]

    # Action history
    action_history: list[ActionInfo] = field(default_factory=list)

    # Private information (only for the observing player)
    my_player_id: int | None = None
    my_hole_cards: tuple[Card, Card] | None = None
    my_position: int = 0  # Position relative to dealer (0 = dealer)

    # Current action context
    valid_actions: list[ActionType] = field(default_factory=list)
    call_amount: int = 0
    min_raise: int = 0
    max_raise: int = 0

    @property
    def my_chips(self) -> int:
        """Get observing player's chip count."""
        if self.my_player_id is None:
            return 0
        for p in self.players:
            if p.id == self.my_player_id:
                return p.chips
        return 0

    @property
    def my_current_bet(self) -> int:
        """Get observing player's current bet."""
        if self.my_player_id is None:
            return 0
        for p in self.players:
            if p.id == self.my_player_id:
                return p.current_bet
        return 0

    @property
    def players_in_hand(self) -> int:
        """Count of players still in the hand."""
        return sum(
            1 for p in self.players
            if p.status in (PlayerStatus.ACTIVE, PlayerStatus.ALL_IN)
        )

    @property
    def active_players(self) -> int:
        """Count of players who can still act."""
        return sum(1 for p in self.players if p.status == PlayerStatus.ACTIVE)

    def get_player_state(self, player_id: int) -> PlayerState | None:
        """Get a specific player's state."""
        for p in self.players:
            if p.id == player_id:
                return p
        return None

    def get_position_name(self, position: int) -> str:
        """Get name for a position (BTN, SB, BB, UTG, etc.)."""
        if position == 0:
            return "BTN"
        elif position == 1:
            return "SB"
        elif position == 2:
            return "BB"
        elif self.num_players <= 6:
            # 6-max positions
            names = {3: "UTG", 4: "MP", 5: "CO"}
            return names.get(position, f"P{position}")
        else:
            # Full ring positions
            names = {3: "UTG", 4: "UTG+1", 5: "MP", 6: "MP+1", 7: "HJ", 8: "CO"}
            return names.get(position, f"P{position}")


def create_table_state(
    players: list["Player"],
    community_cards: list[Card],
    pot_total: int,
    current_bet: int,
    round: GameRound,
    dealer_position: int,
    small_blind: int,
    big_blind: int,
    action_history: list[ActionInfo],
    observer_id: int | None = None,
    valid_actions: list[ActionType] | None = None,
    call_amount: int = 0,
    min_raise: int = 0,
    max_raise: int = 0,
) -> TableState:
    """Create a TableState from game components."""
    num_players = len(players)

    # Create player states
    player_states = []
    observer_position = 0
    observer_cards = None

    for i, player in enumerate(players):
        # Calculate position relative to dealer
        position = (i - dealer_position) % num_players

        state = PlayerState(
            id=player.id,
            name=player.name,
            chips=player.chips,
            status=player.status,
            current_bet=player.current_bet,
            total_invested=player.total_invested,
            is_dealer=(position == 0),
            is_small_blind=(position == 1),
            is_big_blind=(position == 2),
            position=position,
        )
        player_states.append(state)

        if player.id == observer_id:
            observer_position = position
            observer_cards = player.hole_cards

    return TableState(
        community_cards=list(community_cards),
        pot_total=pot_total,
        current_bet=current_bet,
        round=round,
        dealer_position=dealer_position,
        small_blind=small_blind,
        big_blind=big_blind,
        num_players=num_players,
        players=player_states,
        action_history=list(action_history),
        my_player_id=observer_id,
        my_hole_cards=observer_cards,
        my_position=observer_position,
        valid_actions=valid_actions or [],
        call_amount=call_amount,
        min_raise=min_raise,
        max_raise=max_raise,
    )
