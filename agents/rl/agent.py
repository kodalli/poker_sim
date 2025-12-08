"""Reinforcement learning agent that logs experiences for PPO training."""

from dataclasses import dataclass, field

import torch
import torch.nn.functional as F

from agents.base import BaseAgent
from agents.neural.encoding import StateEncoder
from agents.neural.network import ActorCriticMLP
from poker.cards import Card
from poker.player import ActionType, PlayerAction
from poker.table import TableState
from training.buffer import Experience
from training.rewards import HandResult, RewardConfig, compute_shaping_reward


@dataclass
class DecisionRecord:
    """Record of a single decision for computing rewards later."""

    state: torch.Tensor  # Encoded state
    action: int  # Action index
    action_log_prob: float  # Log probability
    bet_fraction: float  # Bet sizing
    value: float  # Value estimate
    table_state: TableState  # Original table state (for reward computation)
    hole_cards: tuple[Card, Card] | None = None


@dataclass
class EpisodeCollector:
    """Collects experiences during a hand/episode."""

    decisions: list[DecisionRecord] = field(default_factory=list)
    hole_cards: tuple[Card, Card] | None = None

    def clear(self) -> None:
        self.decisions.clear()
        self.hole_cards = None


class RLAgent(BaseAgent):
    """Agent for RL training that logs experiences.

    Uses ActorCriticMLP network and logs (state, action, prob, value)
    for each decision. After hand completion, rewards are computed
    and experiences are returned for PPO updates.
    """

    ACTION_MAP = {
        0: ActionType.FOLD,
        1: ActionType.CHECK,
        2: ActionType.CALL,
        3: ActionType.RAISE,
        4: ActionType.ALL_IN,
    }

    REVERSE_ACTION_MAP = {v: k for k, v in ACTION_MAP.items()}

    def __init__(
        self,
        network: ActorCriticMLP,
        encoder: StateEncoder | None = None,
        device: torch.device | None = None,
        reward_config: RewardConfig | None = None,
        name: str = "RLAgent",
    ) -> None:
        super().__init__(name)
        self.network = network
        self.device = device or torch.device("cpu")
        self.encoder = encoder or StateEncoder(self.device)
        self.reward_config = reward_config or RewardConfig()

        # Move network to device
        self.network.to(self.device)

        # Experience collection
        self.collector = EpisodeCollector()

        # Track cumulative investment for reward computation
        self._chips_invested = 0
        self._initial_chips = 0

    def _get_valid_action_mask(self, valid_actions: list[ActionType]) -> torch.Tensor:
        """Create binary mask for valid actions."""
        mask = torch.zeros(len(self.ACTION_MAP), device=self.device)
        for action in valid_actions:
            idx = self.REVERSE_ACTION_MAP.get(action)
            if idx is not None:
                mask[idx] = 1.0
        return mask

    def start_hand(
        self, hole_cards: tuple[Card, Card], initial_chips: int, blind_amount: int = 0
    ) -> None:
        """Called at the start of a hand to initialize tracking.

        Args:
            hole_cards: Player's hole cards for this hand
            initial_chips: Chips at start of hand
            blind_amount: Blind already posted (SB or BB amount)
        """
        self.collector.clear()
        self.collector.hole_cards = hole_cards
        # Initialize with blind amount already invested
        self._chips_invested = blind_amount
        self._initial_chips = initial_chips

    def decide(self, table_state: TableState) -> PlayerAction:
        """Decide action and log experience.

        This method:
        1. Encodes the state
        2. Gets action logits, bet fraction, and value from network
        3. Samples an action
        4. Logs the decision for later reward computation
        5. Returns the action
        """
        # Set hole cards from table state if not already set
        if self.collector.hole_cards is None and table_state.my_hole_cards:
            self.collector.hole_cards = table_state.my_hole_cards

        # Encode state
        state_tensor = self.encoder.encode_state(table_state)
        state_tensor = state_tensor.unsqueeze(0)  # Add batch dim

        # Forward pass
        self.network.train()  # Use train mode for exploration
        action_logits, bet_fraction, value = self.network(state_tensor)

        # Get valid action mask
        valid_mask = self._get_valid_action_mask(table_state.valid_actions)
        valid_mask = valid_mask.unsqueeze(0)

        # Mask invalid actions
        masked_logits = action_logits.masked_fill(valid_mask == 0, -1e9)

        # Get action distribution
        action_probs = F.softmax(masked_logits, dim=-1)
        action_dist = torch.distributions.Categorical(action_probs)

        # Sample action
        action_idx = action_dist.sample()
        action_log_prob = action_dist.log_prob(action_idx)

        action_type = self.ACTION_MAP[action_idx.item()]

        # Validate action - if sampled action is invalid, fall back to first valid
        if action_type not in table_state.valid_actions:
            action_type = table_state.valid_actions[0] if table_state.valid_actions else ActionType.FOLD
            action_idx = torch.tensor(self.REVERSE_ACTION_MAP[action_type], device=self.device)
            # Use uniform probability over valid actions for fallback
            # This is more principled than using the masked distribution
            num_valid = len(table_state.valid_actions)
            action_log_prob = torch.log(torch.tensor(1.0 / max(num_valid, 1), device=self.device))

        # Create action
        bet_frac = bet_fraction.item()
        if action_type == ActionType.RAISE:
            bet_range = table_state.max_raise - table_state.min_raise
            amount = table_state.min_raise + int(bet_frac * bet_range)
            action = PlayerAction(ActionType.RAISE, amount)
            self._chips_invested += amount - table_state.my_current_bet
        elif action_type == ActionType.CALL:
            action = PlayerAction(ActionType.CALL, table_state.call_amount)
            self._chips_invested += table_state.call_amount
        elif action_type == ActionType.ALL_IN:
            action = PlayerAction(ActionType.ALL_IN, table_state.my_chips)
            self._chips_invested += table_state.my_chips
        else:
            action = PlayerAction(action_type)

        # Log decision
        record = DecisionRecord(
            state=state_tensor.squeeze(0).cpu(),
            action=action_idx.item(),
            action_log_prob=action_log_prob.item(),
            bet_fraction=bet_frac,
            value=value.item(),
            table_state=table_state,
            hole_cards=self.collector.hole_cards,
        )
        self.collector.decisions.append(record)

        return action

    def end_hand(
        self,
        won: bool,
        pot_won: int,
        showdown: bool = False,
        folded: bool = False,
    ) -> list[Experience]:
        """Called at end of hand to compute rewards and return experiences.

        Args:
            won: Whether we won the hand
            pot_won: Chips won (0 if lost)
            showdown: Whether hand went to showdown
            folded: Whether we folded

        Returns:
            List of Experience objects for PPO training
        """
        experiences = []

        if not self.collector.decisions:
            return experiences

        # Create hand result
        hand_result = HandResult(
            won=won,
            pot_won=pot_won,
            amount_invested=self._chips_invested,
            showdown=showdown,
            folded=folded,
        )

        # Compute sparse reward (end of hand)
        from training.rewards import compute_sparse_reward
        sparse_reward = compute_sparse_reward(hand_result, self.reward_config)

        # Process each decision
        for i, record in enumerate(self.collector.decisions):
            is_last = (i == len(self.collector.decisions) - 1)

            # Compute shaping reward
            action = PlayerAction(
                self.ACTION_MAP[record.action],
                amount=int(record.bet_fraction * 100),  # Approximate
            )

            if record.hole_cards:
                shaping = compute_shaping_reward(
                    action,
                    record.table_state,
                    record.hole_cards,
                    self.reward_config,
                )
            else:
                shaping = 0.0

            # Total reward: shaping + sparse (only on last step)
            reward = shaping
            if is_last:
                reward += sparse_reward

            # Create experience
            exp = Experience(
                state=record.state.numpy(),
                action=record.action,
                action_log_prob=record.action_log_prob,
                bet_fraction=record.bet_fraction,
                reward=reward,
                value=record.value,
                done=is_last,
            )
            experiences.append(exp)

        # Clear collector
        self.collector.clear()

        return experiences

    def get_value(self, state: torch.Tensor) -> float:
        """Get value estimate for a state."""
        with torch.no_grad():
            self.network.eval()
            if state.dim() == 1:
                state = state.unsqueeze(0)
            _, _, value = self.network(state.to(self.device))
            return value.item()

    def get_action_distribution(
        self, table_state: TableState
    ) -> dict[ActionType, float]:
        """Get probability distribution over actions."""
        with torch.no_grad():
            self.network.eval()
            state_tensor = self.encoder.encode_state(table_state)
            state_tensor = state_tensor.unsqueeze(0)

            action_logits, _, _ = self.network(state_tensor)
            valid_mask = self._get_valid_action_mask(table_state.valid_actions)
            valid_mask = valid_mask.unsqueeze(0)

            masked_logits = action_logits.masked_fill(valid_mask == 0, -1e9)
            action_probs = F.softmax(masked_logits, dim=-1)

            return {
                self.ACTION_MAP[i]: prob.item()
                for i, prob in enumerate(action_probs[0])
            }

    def clone(self) -> "RLAgent":
        """Create a copy of this agent."""
        import copy
        new_network = copy.deepcopy(self.network)
        return RLAgent(
            network=new_network,
            encoder=self.encoder,
            device=self.device,
            reward_config=self.reward_config,
            name=self.name,
        )

    def train_mode(self) -> None:
        """Set network to training mode."""
        self.network.train()

    def eval_mode(self) -> None:
        """Set network to evaluation mode."""
        self.network.eval()
