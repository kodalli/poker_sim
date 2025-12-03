"""Neural network agent for poker."""

import torch

from agents.base import BaseAgent
from agents.neural.encoding import StateEncoder
from agents.neural.network import PokerNetwork
from poker.player import ActionType, PlayerAction
from poker.table import TableState


class NeuralAgent(BaseAgent):
    """Agent that uses a neural network for decision making."""

    ACTION_MAP = {
        0: ActionType.FOLD,
        1: ActionType.CHECK,
        2: ActionType.CALL,
        3: ActionType.RAISE,
        4: ActionType.ALL_IN,
    }

    def __init__(
        self,
        network: PokerNetwork,
        encoder: StateEncoder | None = None,
        device: torch.device | None = None,
        temperature: float = 1.0,
        name: str = "Neural",
    ) -> None:
        super().__init__(name)
        self.network = network
        self.device = device or torch.device("cpu")
        self.encoder = encoder or StateEncoder(self.device)
        self.temperature = temperature

        # Move network to device
        self.network.to(self.device)
        self.network.eval()

    def _get_valid_action_mask(self, valid_actions: list[ActionType]) -> torch.Tensor:
        """Create binary mask for valid actions."""
        mask = torch.zeros(len(self.ACTION_MAP), device=self.device)
        for action in valid_actions:
            for idx, act in self.ACTION_MAP.items():
                if act == action:
                    mask[idx] = 1.0
                    break
        return mask

    def decide(self, table_state: TableState) -> PlayerAction:
        """Use neural network to decide action."""
        with torch.no_grad():
            # Encode state
            state_tensor = self.encoder.encode_state(table_state)
            state_tensor = state_tensor.unsqueeze(0)  # Add batch dim

            # Get network output
            action_logits, bet_fraction = self.network(state_tensor)

            # Get valid action mask
            valid_mask = self._get_valid_action_mask(table_state.valid_actions)
            valid_mask = valid_mask.unsqueeze(0)  # Add batch dim

            # Apply mask to logits
            masked_logits = action_logits.masked_fill(valid_mask == 0, -1e9)

            # Apply temperature and get probabilities
            if self.temperature != 1.0:
                masked_logits = masked_logits / self.temperature

            action_probs = torch.softmax(masked_logits, dim=-1)

            # Sample action
            action_idx = torch.multinomial(action_probs, 1).item()
            action_type = self.ACTION_MAP[action_idx]

            # Validate action is actually valid
            if action_type not in table_state.valid_actions:
                # Fallback to first valid action
                action_type = table_state.valid_actions[0] if table_state.valid_actions else ActionType.FOLD

            # Calculate bet amount if raising
            if action_type == ActionType.RAISE:
                fraction = bet_fraction.item()
                bet_range = table_state.max_raise - table_state.min_raise
                amount = table_state.min_raise + int(fraction * bet_range)
                return PlayerAction(ActionType.RAISE, amount)

            elif action_type == ActionType.CALL:
                return PlayerAction(ActionType.CALL, table_state.call_amount)

            elif action_type == ActionType.ALL_IN:
                return PlayerAction(ActionType.ALL_IN, table_state.my_chips)

            return PlayerAction(action_type)

    def get_action_distribution(
        self, table_state: TableState
    ) -> dict[ActionType, float]:
        """Get probability distribution over actions."""
        with torch.no_grad():
            state_tensor = self.encoder.encode_state(table_state)
            state_tensor = state_tensor.unsqueeze(0)

            action_logits, _ = self.network(state_tensor)
            valid_mask = self._get_valid_action_mask(table_state.valid_actions)
            valid_mask = valid_mask.unsqueeze(0)

            masked_logits = action_logits.masked_fill(valid_mask == 0, -1e9)
            action_probs = torch.softmax(masked_logits, dim=-1)

            return {
                self.ACTION_MAP[i]: prob.item()
                for i, prob in enumerate(action_probs[0])
            }

    def clone(self) -> "NeuralAgent":
        """Create a copy of this agent with cloned network."""
        import copy

        new_network = copy.deepcopy(self.network)
        return NeuralAgent(
            network=new_network,
            encoder=self.encoder,
            device=self.device,
            temperature=self.temperature,
            name=self.name,
        )

    def set_weights(self, weights: torch.Tensor) -> None:
        """Set network weights from flattened tensor."""
        self.network.set_flat_weights(weights)

    def get_weights(self) -> torch.Tensor:
        """Get flattened network weights."""
        return self.network.get_flat_weights()


class BatchNeuralAgent:
    """Wrapper for batch inference of multiple agents."""

    def __init__(
        self,
        agents: list[NeuralAgent],
        device: torch.device | None = None,
    ) -> None:
        self.agents = agents
        self.device = device or torch.device("cpu")
        self.encoder = StateEncoder(self.device)

    def batch_decide(
        self, states: list[TableState]
    ) -> list[PlayerAction]:
        """Get actions for multiple states in a single batch.

        Note: This assumes all agents use the same network architecture
        and can be batched together. For genetic algorithms, each agent
        may have different weights, so we process them separately but
        can still benefit from GPU parallelism.
        """
        actions = []
        for agent, state in zip(self.agents, states):
            actions.append(agent.decide(state))
        return actions
