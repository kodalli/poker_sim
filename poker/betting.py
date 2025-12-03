"""Betting round logic for poker."""

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from poker.player import ActionType, Player, PlayerAction, PlayerStatus

if TYPE_CHECKING:
    from agents.base import BaseAgent


@dataclass
class BettingRound:
    """Manage a single betting round."""

    players: list[Player]
    starting_player_idx: int
    big_blind: int
    current_bet: int = 0
    min_raise: int = 0
    last_aggressor_idx: int | None = None
    actions_taken: list[tuple[int, PlayerAction]] = field(default_factory=list)
    _players_acted: set[int] = field(default_factory=set)
    _current_idx: int = 0
    _round_complete: bool = False

    def __post_init__(self) -> None:
        self.min_raise = self.big_blind
        self._current_idx = self.starting_player_idx

    def get_valid_actions(self, player: Player) -> list[ActionType]:
        """Return list of valid actions for the given player."""
        if not player.can_act():
            return []

        actions = [ActionType.FOLD]

        if player.current_bet >= self.current_bet:
            # Can check if no bet to call
            actions.append(ActionType.CHECK)
        else:
            # Must call or fold
            call_amount = self.current_bet - player.current_bet
            if call_amount >= player.chips:
                # Can only go all-in
                actions.append(ActionType.ALL_IN)
            else:
                actions.append(ActionType.CALL)

        # Can always go all-in
        if ActionType.ALL_IN not in actions:
            actions.append(ActionType.ALL_IN)

        # Can raise if we have enough chips and raise is allowed
        min_raise_total = self.current_bet + self.min_raise
        chips_needed_to_raise = min_raise_total - player.current_bet
        if player.chips > chips_needed_to_raise:
            actions.append(ActionType.RAISE)

        return actions

    def get_call_amount(self, player: Player) -> int:
        """Get amount needed to call."""
        return max(0, self.current_bet - player.current_bet)

    def get_min_raise_amount(self) -> int:
        """Get minimum raise amount (total bet, not additional)."""
        return self.current_bet + self.min_raise

    def get_max_raise_amount(self, player: Player) -> int:
        """Get maximum raise amount (total bet, not additional)."""
        return player.current_bet + player.chips

    def apply_action(self, player_idx: int, action: PlayerAction) -> None:
        """Apply an action from a player."""
        player = self.players[player_idx]

        if action.action_type == ActionType.FOLD:
            player.fold()

        elif action.action_type == ActionType.CHECK:
            if player.current_bet < self.current_bet:
                raise ValueError(f"Cannot check when bet is {self.current_bet}")

        elif action.action_type == ActionType.CALL:
            call_amount = self.current_bet - player.current_bet
            player.bet(call_amount)

        elif action.action_type == ActionType.RAISE:
            # action.amount is the total bet amount
            raise_to = action.amount
            if raise_to < self.get_min_raise_amount():
                raise_to = self.get_min_raise_amount()
            if raise_to > self.get_max_raise_amount(player):
                raise_to = self.get_max_raise_amount(player)

            additional = raise_to - player.current_bet
            raise_amount = raise_to - self.current_bet
            player.bet(additional)
            self.min_raise = max(self.min_raise, raise_amount)
            self.current_bet = raise_to
            self.last_aggressor_idx = player_idx
            self._players_acted.clear()

        elif action.action_type == ActionType.ALL_IN:
            all_in_amount = player.chips
            total_bet = player.current_bet + all_in_amount

            if total_bet > self.current_bet:
                # This is a raise
                raise_amount = total_bet - self.current_bet
                if raise_amount >= self.min_raise:
                    self.min_raise = raise_amount
                    self.last_aggressor_idx = player_idx
                    self._players_acted.clear()
                self.current_bet = total_bet

            player.bet(all_in_amount)

        self._players_acted.add(player_idx)
        self.actions_taken.append((player_idx, action))

    def _get_active_players(self) -> list[int]:
        """Get indices of players who can still act."""
        return [
            i for i, p in enumerate(self.players)
            if p.status == PlayerStatus.ACTIVE
        ]

    def _get_players_in_hand(self) -> list[int]:
        """Get indices of players still in the hand."""
        return [i for i, p in enumerate(self.players) if p.is_in_hand()]

    def get_next_player_idx(self) -> int | None:
        """Get the index of the next player to act, or None if round is complete."""
        active = self._get_active_players()
        if not active:
            return None

        # Check if everyone has acted since last raise
        if self.last_aggressor_idx is not None:
            # Round ends when action returns to last aggressor
            # and all active players have acted
            all_have_acted = all(i in self._players_acted for i in active)
            if all_have_acted:
                return None
        else:
            # No aggressor - round ends when everyone has checked/called
            if all(i in self._players_acted for i in active):
                # Check if bets are even
                bets = [self.players[i].current_bet for i in self._get_players_in_hand()]
                if len(set(bets)) <= 1:  # All equal or empty
                    return None

        # Find next active player
        num_players = len(self.players)
        idx = self._current_idx
        for _ in range(num_players):
            idx = (idx + 1) % num_players
            if self.players[idx].status == PlayerStatus.ACTIVE:
                if idx not in self._players_acted or (
                    self.last_aggressor_idx is not None
                    and self.players[idx].current_bet < self.current_bet
                ):
                    self._current_idx = idx
                    return idx

        return None

    def is_complete(self) -> bool:
        """Check if betting round is complete."""
        # Only one player in hand
        in_hand = self._get_players_in_hand()
        if len(in_hand) <= 1:
            return True

        # No more active players (all folded or all-in)
        active = self._get_active_players()
        if not active:
            return True

        # Check if all active players have acted and bets are even
        return self.get_next_player_idx() is None

    def only_one_player_remaining(self) -> bool:
        """Check if only one player remains in the hand."""
        return len(self._get_players_in_hand()) <= 1

    def get_winner_if_all_folded(self) -> int | None:
        """Get winner ID if all but one player folded."""
        in_hand = self._get_players_in_hand()
        if len(in_hand) == 1:
            return in_hand[0]
        return None
