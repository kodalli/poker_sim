"""Reward computation with shaping for RL training."""

from dataclasses import dataclass
from typing import Sequence

from poker.cards import Card
from poker.player import ActionType, PlayerAction
from poker.table import TableState
from training.equity import heuristic_equity


@dataclass
class RewardConfig:
    """Configuration for reward shaping."""

    # Sparse rewards (end of hand)
    win_scale: float = 1.0  # Scale factor for winning
    loss_scale: float = 1.0  # Scale factor for losing

    # Shaping rewards (per decision)
    enable_shaping: bool = True
    fold_equity_threshold: float = 0.30  # Bonus for folding when behind
    fold_bonus: float = 0.05  # Reward for good folds

    position_bonus: float = 0.02  # Bonus for aggressive play in position
    pot_odds_bonus: float = 0.03  # Bonus for correct pot odds decisions
    value_bet_bonus: float = 0.02  # Bonus for betting with strong hands
    hand_strength_threshold: float = 0.70  # Threshold for "strong" hands

    # Normalization
    starting_chips: int = 200  # For normalizing chip amounts


@dataclass
class HandResult:
    """Result of a completed hand."""

    won: bool
    pot_won: int  # Chips won (0 if lost)
    amount_invested: int  # Total chips put in pot
    showdown: bool  # Whether hand went to showdown
    folded: bool  # Whether we folded


def compute_sparse_reward(
    result: HandResult,
    config: RewardConfig | None = None,
) -> float:
    """Compute sparse reward at end of hand.

    Args:
        result: Hand result with win/loss information
        config: Reward configuration

    Returns:
        Normalized reward (positive for wins, negative for losses)
    """
    config = config or RewardConfig()

    if result.won:
        # Reward is net chips won, normalized
        net_won = result.pot_won - result.amount_invested
        reward = (net_won / config.starting_chips) * config.win_scale
    else:
        # Penalty is chips lost, normalized
        reward = (-result.amount_invested / config.starting_chips) * config.loss_scale

    return reward


def compute_shaping_reward(
    action: PlayerAction,
    table_state: TableState,
    hole_cards: tuple[Card, Card],
    config: RewardConfig | None = None,
) -> float:
    """Compute shaping reward for a single decision.

    Args:
        action: The action taken
        table_state: Current table state
        hole_cards: Player's hole cards
        config: Reward configuration

    Returns:
        Shaping reward (small positive/negative adjustments)
    """
    config = config or RewardConfig()

    if not config.enable_shaping:
        return 0.0

    reward = 0.0

    # Get current equity estimate
    community = list(table_state.community_cards) if table_state.community_cards else []
    equity = heuristic_equity(hole_cards, community)

    # === Good fold reward ===
    # Folding when we have low equity is correct
    if action.action_type == ActionType.FOLD:
        if equity < config.fold_equity_threshold:
            reward += config.fold_bonus
        else:
            # Penalize folding with decent equity (but less than fold bonus)
            reward -= config.fold_bonus * 0.5

    # === Position-aware aggression ===
    # Reward aggressive play when in position
    is_aggressive = action.action_type in (ActionType.RAISE, ActionType.ALL_IN)
    in_position = _is_in_position(table_state)

    if is_aggressive and in_position:
        reward += config.position_bonus

    # === Pot odds adherence ===
    if action.action_type == ActionType.CALL:
        call_amount = table_state.call_amount
        pot_total = table_state.pot_total

        if pot_total > 0 and call_amount > 0:
            # Pot odds = call / (pot + call)
            pot_odds = call_amount / (pot_total + call_amount)

            # If we're getting good odds relative to equity, it's correct
            if equity >= pot_odds:
                reward += config.pot_odds_bonus
            else:
                # Bad call - calling without proper odds
                reward -= config.pot_odds_bonus

    # === Value betting with strong hands ===
    if is_aggressive and equity >= config.hand_strength_threshold:
        reward += config.value_bet_bonus

    # === Passive with weak hands (slight penalty for missed aggression) ===
    # Being passive with a monster is usually wrong
    is_passive = action.action_type in (ActionType.CHECK, ActionType.CALL)
    if is_passive and equity >= 0.85:
        reward -= config.value_bet_bonus * 0.5

    return reward


def _is_in_position(table_state: TableState) -> bool:
    """Check if the current player is in position (acts last postflop).

    In heads-up, the dealer (button) acts last postflop.
    With more players, position is relative to remaining active players.
    """
    # Simple heuristic: if we're the button or close to it, we're in position
    # dealer_position in table_state indicates the button
    my_position = table_state.my_position if hasattr(table_state, "my_position") else 0
    dealer_position = table_state.dealer_position if hasattr(table_state, "dealer_position") else 0

    # In heads-up, dealer is in position postflop
    num_active = table_state.players_in_hand
    if num_active <= 2:
        return my_position == dealer_position

    # With more players, being on the button or cutoff is good position
    # This is a simplification
    positions_from_button = (my_position - dealer_position) % max(num_active, 1)
    return positions_from_button <= 2  # Button, cutoff, hijack


def compute_reward(
    action: PlayerAction,
    table_state: TableState,
    hole_cards: tuple[Card, Card],
    hand_result: HandResult | None = None,
    config: RewardConfig | None = None,
) -> float:
    """Compute total reward combining sparse and shaping components.

    Args:
        action: The action taken
        table_state: Current table state
        hole_cards: Player's hole cards
        hand_result: If hand is complete, the result
        config: Reward configuration

    Returns:
        Total reward for this decision
    """
    config = config or RewardConfig()
    reward = 0.0

    # Shaping reward (every decision)
    reward += compute_shaping_reward(action, table_state, hole_cards, config)

    # Sparse reward (end of hand only)
    if hand_result is not None:
        reward += compute_sparse_reward(hand_result, config)

    return reward


def compute_hand_rewards(
    actions: list[PlayerAction],
    states: list[TableState],
    hole_cards: tuple[Card, Card],
    hand_result: HandResult,
    config: RewardConfig | None = None,
) -> list[float]:
    """Compute rewards for all decisions in a hand.

    The sparse reward is only added to the final decision.
    Shaping rewards are added to each decision.

    Args:
        actions: List of actions taken during the hand
        states: Corresponding table states
        hole_cards: Player's hole cards
        hand_result: Final hand result
        config: Reward configuration

    Returns:
        List of rewards, one per decision
    """
    config = config or RewardConfig()
    rewards = []

    for i, (action, state) in enumerate(zip(actions, states)):
        is_last = (i == len(actions) - 1)

        if is_last:
            # Final decision gets sparse + shaping
            r = compute_reward(action, state, hole_cards, hand_result, config)
        else:
            # Intermediate decisions get only shaping
            r = compute_shaping_reward(action, state, hole_cards, config)

        rewards.append(r)

    return rewards
