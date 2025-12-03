"""Random agent for baseline comparison."""

from random import Random

from agents.base import BaseAgent
from poker.player import ActionType, PlayerAction
from poker.table import TableState


class RandomAgent(BaseAgent):
    """Agent that takes random actions."""

    def __init__(self, name: str = "Random", seed: int | None = None) -> None:
        super().__init__(name)
        self._rng = Random(seed)

    def decide(self, table_state: TableState) -> PlayerAction:
        """Take a random valid action."""
        valid_actions = table_state.valid_actions

        if not valid_actions:
            return PlayerAction(ActionType.FOLD)

        action_type = self._rng.choice(valid_actions)

        if action_type == ActionType.CALL:
            return PlayerAction(ActionType.CALL, table_state.call_amount)

        elif action_type == ActionType.RAISE:
            # Random raise between min and max
            min_raise = table_state.min_raise
            max_raise = table_state.max_raise
            if max_raise > min_raise:
                raise_amount = self._rng.randint(min_raise, max_raise)
            else:
                raise_amount = min_raise
            return PlayerAction(ActionType.RAISE, raise_amount)

        elif action_type == ActionType.ALL_IN:
            return PlayerAction(ActionType.ALL_IN, table_state.my_chips)

        return PlayerAction(action_type)

    def clone(self) -> "RandomAgent":
        """Create a copy with new RNG."""
        return RandomAgent(name=self.name)


class CallStationAgent(BaseAgent):
    """Agent that always calls (or checks if possible)."""

    def __init__(self, name: str = "CallStation") -> None:
        super().__init__(name)

    def decide(self, table_state: TableState) -> PlayerAction:
        """Always call or check."""
        valid_actions = table_state.valid_actions

        if ActionType.CHECK in valid_actions:
            return PlayerAction(ActionType.CHECK)

        if ActionType.CALL in valid_actions:
            return PlayerAction(ActionType.CALL, table_state.call_amount)

        if ActionType.ALL_IN in valid_actions:
            return PlayerAction(ActionType.ALL_IN, table_state.my_chips)

        return PlayerAction(ActionType.FOLD)

    def clone(self) -> "CallStationAgent":
        return CallStationAgent(name=self.name)


class TightAggressiveAgent(BaseAgent):
    """Simple tight-aggressive agent based on hand strength heuristics."""

    def __init__(self, name: str = "TAG", seed: int | None = None) -> None:
        super().__init__(name)
        self._rng = Random(seed)

    def _get_hand_strength(self, table_state: TableState) -> float:
        """Estimate hand strength from 0 to 1."""
        if not table_state.my_hole_cards:
            return 0.0

        hole_cards = table_state.my_hole_cards
        r1, r2 = hole_cards[0].rank, hole_cards[1].rank
        suited = hole_cards[0].suit == hole_cards[1].suit

        # Simple hand strength heuristic
        high_card = max(r1, r2)
        low_card = min(r1, r2)
        is_pair = r1 == r2
        gap = high_card - low_card

        strength = 0.0

        # Pairs
        if is_pair:
            strength = 0.5 + (high_card - 2) / 24  # 0.5 to ~1.0

        # High cards
        else:
            strength = (high_card + low_card - 4) / 24  # Normalize to 0-1

            # Suited bonus
            if suited:
                strength += 0.1

            # Connected bonus
            if gap <= 2:
                strength += 0.05

        # Premium hands
        if is_pair and high_card >= 10:  # TT+
            strength = max(strength, 0.85)
        if high_card == 14 and low_card >= 10:  # AT+
            strength = max(strength, 0.7)
        if high_card == 14 and low_card == 13:  # AK
            strength = 0.82

        return min(1.0, strength)

    def decide(self, table_state: TableState) -> PlayerAction:
        """Make decision based on hand strength."""
        valid_actions = table_state.valid_actions
        strength = self._get_hand_strength(table_state)

        # Add some randomness
        strength += self._rng.gauss(0, 0.1)
        strength = max(0.0, min(1.0, strength))

        # Thresholds
        if strength > 0.8:
            # Strong hand - raise
            if ActionType.RAISE in valid_actions:
                raise_amount = int(
                    table_state.min_raise
                    + (table_state.max_raise - table_state.min_raise) * 0.5
                )
                return PlayerAction(ActionType.RAISE, raise_amount)
            if ActionType.ALL_IN in valid_actions:
                return PlayerAction(ActionType.ALL_IN, table_state.my_chips)

        elif strength > 0.5:
            # Medium hand - call
            if ActionType.CHECK in valid_actions:
                return PlayerAction(ActionType.CHECK)
            if ActionType.CALL in valid_actions:
                # Only call reasonable bets
                pot_odds = table_state.call_amount / (table_state.pot_total + 1)
                if pot_odds < strength:
                    return PlayerAction(ActionType.CALL, table_state.call_amount)

        elif strength > 0.3:
            # Weak hand - check if free
            if ActionType.CHECK in valid_actions:
                return PlayerAction(ActionType.CHECK)

        # Default: fold
        return PlayerAction(ActionType.FOLD)

    def clone(self) -> "TightAggressiveAgent":
        return TightAggressiveAgent(name=self.name)
