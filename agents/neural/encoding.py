"""Encode poker game state to tensors for neural networks."""

import torch
import torch.nn.functional as F

from poker.cards import Card
from poker.player import ActionType, PlayerStatus
from poker.table import GameRound, TableState


class StateEncoder:
    """Encode game state to tensor for neural network input."""

    # Feature dimensions
    CARD_DIM = 52  # One-hot encoding per card
    HOLE_CARDS_DIM = CARD_DIM * 2  # 104
    COMMUNITY_DIM = CARD_DIM * 5  # 260
    ROUND_DIM = 4  # One-hot for round
    POSITION_DIM = 10  # One-hot for position (max 10 players)
    PLAYER_FEATURES = 5  # chips_ratio, current_bet_ratio, pot_odds, is_dealer, is_blind
    MAX_PLAYERS = 10
    OPPONENT_FEATURES = PLAYER_FEATURES * (MAX_PLAYERS - 1)  # Features for other players
    ACTION_FEATURES = 5  # Valid action mask

    # Total input dimension
    TOTAL_DIM = (
        HOLE_CARDS_DIM
        + COMMUNITY_DIM
        + ROUND_DIM
        + POSITION_DIM
        + PLAYER_FEATURES
        + OPPONENT_FEATURES
        + ACTION_FEATURES
    )  # 104 + 260 + 4 + 10 + 5 + 45 + 5 = 433

    def __init__(self, device: torch.device | None = None) -> None:
        self.device = device or torch.device("cpu")

    def encode_card(self, card: Card | None) -> torch.Tensor:
        """One-hot encode a single card (52-dim)."""
        encoding = torch.zeros(self.CARD_DIM, device=self.device)
        if card is not None:
            encoding[card.to_index()] = 1.0
        return encoding

    def encode_cards(self, cards: list[Card] | None, max_cards: int) -> torch.Tensor:
        """Encode multiple cards, padding to max_cards."""
        result = torch.zeros(max_cards * self.CARD_DIM, device=self.device)
        if cards:
            for i, card in enumerate(cards[:max_cards]):
                start = i * self.CARD_DIM
                result[start : start + self.CARD_DIM] = self.encode_card(card)
        return result

    def encode_round(self, round: GameRound) -> torch.Tensor:
        """One-hot encode the current round."""
        encoding = torch.zeros(self.ROUND_DIM, device=self.device)
        round_map = {
            GameRound.PRE_FLOP: 0,
            GameRound.FLOP: 1,
            GameRound.TURN: 2,
            GameRound.RIVER: 3,
            GameRound.SHOWDOWN: 3,  # Same as river
        }
        encoding[round_map[round]] = 1.0
        return encoding

    def encode_position(self, position: int, num_players: int) -> torch.Tensor:
        """One-hot encode position (0 = dealer)."""
        encoding = torch.zeros(self.POSITION_DIM, device=self.device)
        if 0 <= position < self.POSITION_DIM:
            encoding[position] = 1.0
        return encoding

    def encode_valid_actions(self, valid_actions: list[ActionType]) -> torch.Tensor:
        """Binary mask for valid actions."""
        encoding = torch.zeros(self.ACTION_FEATURES, device=self.device)
        action_map = {
            ActionType.FOLD: 0,
            ActionType.CHECK: 1,
            ActionType.CALL: 2,
            ActionType.RAISE: 3,
            ActionType.ALL_IN: 4,
        }
        for action in valid_actions:
            if action in action_map:
                encoding[action_map[action]] = 1.0
        return encoding

    def encode_state(self, table_state: TableState) -> torch.Tensor:
        """Encode complete table state to tensor."""
        features = []

        # Hole cards (104)
        if table_state.my_hole_cards:
            hole_cards = list(table_state.my_hole_cards)
        else:
            hole_cards = []
        features.append(self.encode_cards(hole_cards, 2))

        # Community cards (260)
        features.append(self.encode_cards(table_state.community_cards, 5))

        # Round (4)
        features.append(self.encode_round(table_state.round))

        # Position (10)
        features.append(
            self.encode_position(table_state.my_position, table_state.num_players)
        )

        # Player features (5)
        total_chips = sum(p.chips for p in table_state.players) + table_state.pot_total
        if total_chips == 0:
            total_chips = 1

        my_player = table_state.get_player_state(table_state.my_player_id or 0)
        my_chips = my_player.chips if my_player else 0

        player_features = torch.tensor(
            [
                my_chips / total_chips,  # Chip ratio
                table_state.my_current_bet / max(1, table_state.pot_total),  # Bet ratio
                table_state.call_amount / max(1, table_state.pot_total),  # Pot odds
                1.0 if my_player and my_player.is_dealer else 0.0,  # Is dealer
                1.0 if my_player and (my_player.is_small_blind or my_player.is_big_blind) else 0.0,  # Is blind
            ],
            device=self.device,
        )
        features.append(player_features)

        # Opponent features (45 = 9 opponents * 5 features)
        opponent_features = torch.zeros(self.OPPONENT_FEATURES, device=self.device)
        opp_idx = 0
        for p in table_state.players:
            if p.id != table_state.my_player_id and opp_idx < self.MAX_PLAYERS - 1:
                start = opp_idx * self.PLAYER_FEATURES
                opponent_features[start] = p.chips / total_chips
                opponent_features[start + 1] = p.current_bet / max(1, table_state.pot_total)
                opponent_features[start + 2] = 1.0 if p.status == PlayerStatus.ACTIVE else 0.0
                opponent_features[start + 3] = 1.0 if p.status == PlayerStatus.ALL_IN else 0.0
                opponent_features[start + 4] = 1.0 if p.status == PlayerStatus.FOLDED else 0.0
                opp_idx += 1
        features.append(opponent_features)

        # Valid actions mask (5)
        features.append(self.encode_valid_actions(table_state.valid_actions))

        return torch.cat(features)

    def batch_encode(self, states: list[TableState]) -> torch.Tensor:
        """Batch encode multiple states."""
        if not states:
            return torch.zeros((0, self.TOTAL_DIM), device=self.device)
        return torch.stack([self.encode_state(s) for s in states])

    @classmethod
    def get_input_dim(cls) -> int:
        """Get the input dimension for the network."""
        return cls.TOTAL_DIM
