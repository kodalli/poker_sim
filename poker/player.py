"""Player state and actions for poker."""

from dataclasses import dataclass, field
from enum import Enum, auto

from poker.cards import Card


class PlayerStatus(Enum):
    """Player status during a hand."""

    ACTIVE = auto()  # Still in hand, can act
    FOLDED = auto()  # Folded this hand
    ALL_IN = auto()  # All-in, cannot act further
    WAITING = auto()  # Waiting for next hand (no chips or sitting out)


class ActionType(Enum):
    """Types of actions a player can take."""

    FOLD = auto()
    CHECK = auto()
    CALL = auto()
    RAISE = auto()
    ALL_IN = auto()

    def __str__(self) -> str:
        return self.name.replace("_", " ").title()


@dataclass
class PlayerAction:
    """An action taken by a player."""

    action_type: ActionType
    amount: int = 0  # Amount bet (for RAISE/CALL/ALL_IN)

    def __str__(self) -> str:
        if self.amount > 0:
            return f"{self.action_type} ({self.amount})"
        return str(self.action_type)


@dataclass
class Player:
    """A player at the poker table."""

    id: int
    chips: int
    name: str = ""
    hole_cards: tuple[Card, Card] | None = None
    status: PlayerStatus = PlayerStatus.ACTIVE
    current_bet: int = 0  # Amount bet in current betting round
    total_invested: int = 0  # Total chips in pot this hand

    def __post_init__(self) -> None:
        if not self.name:
            self.name = f"Player {self.id}"

    def reset_for_hand(self) -> None:
        """Reset player state for a new hand."""
        self.hole_cards = None
        self.current_bet = 0
        self.total_invested = 0
        if self.chips > 0:
            self.status = PlayerStatus.ACTIVE
        else:
            self.status = PlayerStatus.WAITING

    def reset_for_round(self) -> None:
        """Reset player state for a new betting round."""
        self.current_bet = 0

    def can_act(self) -> bool:
        """Check if player can take an action."""
        return self.status == PlayerStatus.ACTIVE

    def is_in_hand(self) -> bool:
        """Check if player is still in the hand (active or all-in)."""
        return self.status in (PlayerStatus.ACTIVE, PlayerStatus.ALL_IN)

    def bet(self, amount: int) -> int:
        """Place a bet. Returns actual amount bet (may be less if all-in)."""
        actual_amount = min(amount, self.chips)
        self.chips -= actual_amount
        self.current_bet += actual_amount
        self.total_invested += actual_amount
        if self.chips == 0:
            self.status = PlayerStatus.ALL_IN
        return actual_amount

    def fold(self) -> None:
        """Fold the hand."""
        self.status = PlayerStatus.FOLDED

    def win(self, amount: int) -> None:
        """Receive winnings."""
        self.chips += amount

    def __str__(self) -> str:
        cards_str = ""
        if self.hole_cards:
            cards_str = f" [{self.hole_cards[0]} {self.hole_cards[1]}]"
        return f"{self.name}: {self.chips} chips{cards_str} ({self.status.name})"


@dataclass
class PlayerView:
    """What a player can see about another player (hidden information hidden)."""

    id: int
    name: str
    chips: int
    status: PlayerStatus
    current_bet: int
    total_invested: int
    has_cards: bool  # Whether they have hole cards (but not what they are)

    @classmethod
    def from_player(cls, player: Player, reveal_cards: bool = False) -> "PlayerView":
        """Create view of a player, optionally hiding hole cards."""
        return cls(
            id=player.id,
            name=player.name,
            chips=player.chips,
            status=player.status,
            current_bet=player.current_bet,
            total_invested=player.total_invested,
            has_cards=player.hole_cards is not None,
        )
