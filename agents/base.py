"""Base agent class for poker AI."""

from abc import ABC, abstractmethod

from poker.player import PlayerAction
from poker.table import TableState


class BaseAgent(ABC):
    """Abstract base class for all poker agents."""

    def __init__(self, name: str = "Agent") -> None:
        self.name = name
        self.games_played = 0
        self.hands_won = 0
        self.total_winnings = 0

    @abstractmethod
    def decide(self, table_state: TableState) -> PlayerAction:
        """Decide what action to take given the current table state.

        Args:
            table_state: Current state of the table including:
                - Community cards
                - Hole cards (if observer)
                - Pot size
                - Player positions and chip stacks
                - Valid actions
                - Call/raise amounts

        Returns:
            PlayerAction to take.
        """
        ...

    def notify_hand_result(self, chip_delta: int, won: bool) -> None:
        """Called after a hand completes to notify the agent of the result.

        Args:
            chip_delta: Net change in chips from this hand.
            won: Whether this agent won the hand.
        """
        self.total_winnings += chip_delta
        if won:
            self.hands_won += 1

    def notify_game_end(self, final_chips: int) -> None:
        """Called when a game (session of multiple hands) ends.

        Args:
            final_chips: Final chip count.
        """
        self.games_played += 1

    def reset_stats(self) -> None:
        """Reset statistics."""
        self.games_played = 0
        self.hands_won = 0
        self.total_winnings = 0

    def clone(self) -> "BaseAgent":
        """Create a copy of this agent. Override for stateful agents."""
        return self.__class__(name=self.name)

    def __str__(self) -> str:
        return f"{self.__class__.__name__}({self.name})"
