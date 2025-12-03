"""Pot and side pot management for poker."""

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from poker.player import Player


@dataclass
class SidePot:
    """A single pot that certain players are eligible for."""

    amount: int
    eligible_players: set[int]  # Player IDs eligible for this pot

    def __str__(self) -> str:
        players_str = ", ".join(str(p) for p in sorted(self.eligible_players))
        return f"Pot({self.amount}, players=[{players_str}])"


class PotManager:
    """Manage the main pot and any side pots."""

    def __init__(self) -> None:
        self.pots: list[SidePot] = []
        self._collected_this_round: int = 0

    def reset(self) -> None:
        """Reset for a new hand."""
        self.pots = []
        self._collected_this_round = 0

    def collect_bets(self, players: list["Player"]) -> None:
        """Collect bets from all players and create side pots as needed.

        This should be called at the end of each betting round.
        Side pots are created when players are all-in for different amounts.
        """
        # Get all players with bets
        betting_players = [(p.id, p.current_bet, p.is_in_hand()) for p in players if p.current_bet > 0]

        if not betting_players:
            return

        # Sort by bet amount to handle side pots
        betting_players.sort(key=lambda x: x[1])

        prev_level = 0
        for player_id, bet_amount, is_in_hand in betting_players:
            if bet_amount > prev_level:
                # Create pot for this level
                level_amount = bet_amount - prev_level

                # All players who bet at least this much are eligible
                eligible = {
                    p_id
                    for p_id, p_bet, p_in_hand in betting_players
                    if p_bet >= bet_amount and p_in_hand
                }

                # Calculate pot amount (level_amount * number of players at this level or above)
                contributors = sum(
                    1 for _, p_bet, _ in betting_players if p_bet >= bet_amount
                )
                pot_amount = level_amount * contributors

                if pot_amount > 0 and eligible:
                    # Try to merge with existing pot if same eligible players
                    merged = False
                    for pot in self.pots:
                        if pot.eligible_players == eligible:
                            pot.amount += pot_amount
                            merged = True
                            break
                    if not merged:
                        self.pots.append(SidePot(amount=pot_amount, eligible_players=eligible))

                prev_level = bet_amount

        self._collected_this_round = sum(p.current_bet for p in players)

    def add_dead_money(self, amount: int, eligible_players: set[int]) -> None:
        """Add dead money (e.g., from players who folded after betting)."""
        if amount <= 0 or not eligible_players:
            return

        # Add to main pot or create new one
        if self.pots and self.pots[0].eligible_players == eligible_players:
            self.pots[0].amount += amount
        else:
            # Create a new pot for dead money
            if self.pots:
                self.pots[0].amount += amount
            else:
                self.pots.append(SidePot(amount=amount, eligible_players=eligible_players))

    def total(self) -> int:
        """Total chips in all pots."""
        return sum(pot.amount for pot in self.pots)

    def distribute(self, winners_by_pot: list[list[int]]) -> dict[int, int]:
        """Distribute pots to winners.

        Args:
            winners_by_pot: For each pot (in order), list of winning player IDs.
                           If multiple winners, pot is split.

        Returns:
            Dict mapping player_id to total winnings.
        """
        winnings: dict[int, int] = {}

        for pot, winners in zip(self.pots, winners_by_pot):
            if not winners:
                continue

            # Filter winners to only eligible players
            eligible_winners = [w for w in winners if w in pot.eligible_players]
            if not eligible_winners:
                # No eligible winners - this shouldn't happen in normal play
                # Give to first eligible player
                eligible_winners = [next(iter(pot.eligible_players))]

            # Split pot among winners
            share = pot.amount // len(eligible_winners)
            remainder = pot.amount % len(eligible_winners)

            for i, winner_id in enumerate(eligible_winners):
                amount = share + (1 if i < remainder else 0)
                winnings[winner_id] = winnings.get(winner_id, 0) + amount

        self.pots = []
        return winnings

    def get_eligible_players(self, pot_index: int = 0) -> set[int]:
        """Get eligible players for a specific pot."""
        if pot_index < len(self.pots):
            return self.pots[pot_index].eligible_players
        return set()

    def __len__(self) -> int:
        return len(self.pots)

    def __str__(self) -> str:
        if not self.pots:
            return "Pot(0)"
        return " | ".join(str(pot) for pot in self.pots)
