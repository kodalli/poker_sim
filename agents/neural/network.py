"""Neural network architectures for poker AI."""

from abc import ABC, abstractmethod
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from agents.neural.encoding import StateEncoder


@dataclass
class NetworkConfig:
    """Configuration for network architecture."""

    input_dim: int = StateEncoder.TOTAL_DIM
    hidden_dims: tuple[int, ...] = (256, 128, 64)
    num_actions: int = 5  # FOLD, CHECK, CALL, RAISE, ALL_IN
    dropout: float = 0.1
    # Transformer-specific
    num_heads: int = 4
    num_layers: int = 2
    embed_dim: int = 128


class PokerNetwork(nn.Module, ABC):
    """Abstract base class for poker neural networks."""

    def __init__(self, config: NetworkConfig | None = None) -> None:
        super().__init__()
        self.config = config or NetworkConfig()

    @abstractmethod
    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Args:
            x: Input tensor of shape (batch, input_dim)

        Returns:
            action_logits: (batch, num_actions) - raw logits for actions
            bet_fraction: (batch, 1) - bet size as fraction of max (0-1)
        """
        ...

    def get_flat_weights(self) -> torch.Tensor:
        """Flatten all weights into a single tensor for genetic operations."""
        return torch.cat([p.data.flatten() for p in self.parameters()])

    def set_flat_weights(self, weights: torch.Tensor) -> None:
        """Set weights from a flattened tensor."""
        offset = 0
        for p in self.parameters():
            numel = p.numel()
            p.data.copy_(weights[offset : offset + numel].view_as(p))
            offset += numel

    def num_parameters(self) -> int:
        """Get total number of parameters."""
        return sum(p.numel() for p in self.parameters())

    def get_action_probs(
        self, x: torch.Tensor, valid_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Get action probabilities, optionally masking invalid actions.

        Args:
            x: Input tensor
            valid_mask: Binary mask of valid actions (1 = valid)

        Returns:
            Probability distribution over actions
        """
        logits, _ = self(x)

        if valid_mask is not None:
            # Set invalid action logits to very negative
            logits = logits.masked_fill(valid_mask == 0, -1e9)

        return F.softmax(logits, dim=-1)


class MLPNetwork(PokerNetwork):
    """Multi-layer perceptron network for poker."""

    def __init__(self, config: NetworkConfig | None = None) -> None:
        super().__init__(config)

        # Build feature extractor
        layers: list[nn.Module] = []
        prev_dim = self.config.input_dim

        for hidden_dim in self.config.hidden_dims:
            layers.extend(
                [
                    nn.Linear(prev_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(self.config.dropout),
                ]
            )
            prev_dim = hidden_dim

        self.feature_extractor = nn.Sequential(*layers)

        # Action head - outputs logits for each action
        self.action_head = nn.Linear(prev_dim, self.config.num_actions)

        # Bet sizing head - outputs fraction of max raise
        self.bet_head = nn.Sequential(
            nn.Linear(prev_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize weights."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass."""
        features = self.feature_extractor(x)
        action_logits = self.action_head(features)
        bet_fraction = self.bet_head(features)
        return action_logits, bet_fraction


class TransformerNetwork(PokerNetwork):
    """Transformer-based network for poker.

    Uses self-attention to model relationships between different
    parts of the game state (cards, players, actions).
    """

    def __init__(self, config: NetworkConfig | None = None) -> None:
        super().__init__(config)

        embed_dim = self.config.embed_dim

        # Input projection
        self.input_projection = nn.Linear(self.config.input_dim, embed_dim)

        # Positional encoding (learnable)
        self.pos_embedding = nn.Parameter(torch.randn(1, 16, embed_dim) * 0.02)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=self.config.num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=self.config.dropout,
            activation="gelu",
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=self.config.num_layers
        )

        # Output heads
        self.action_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.ReLU(),
            nn.Linear(embed_dim // 2, self.config.num_actions),
        )

        self.bet_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.ReLU(),
            nn.Linear(embed_dim // 2, 1),
            nn.Sigmoid(),
        )

        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize weights."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        The input is reshaped into sequence tokens representing different
        aspects of the game state.
        """
        batch_size = x.shape[0]

        # Project input to embedding dimension
        # Shape: (batch, embed_dim)
        embedded = self.input_projection(x)

        # Reshape to sequence of tokens
        # Split into 16 tokens of embed_dim // 16 each, then project
        # For simplicity, we'll use the embedded vector as a single token
        # and add positional info
        embedded = embedded.unsqueeze(1)  # (batch, 1, embed_dim)

        # Expand with positional embeddings to create pseudo-sequence
        # This allows the transformer to learn different aspects
        seq_len = self.pos_embedding.shape[1]
        embedded = embedded.expand(-1, seq_len, -1) + self.pos_embedding

        # Apply transformer
        transformed = self.transformer(embedded)

        # Pool over sequence (use mean)
        pooled = transformed.mean(dim=1)  # (batch, embed_dim)

        # Output heads
        action_logits = self.action_head(pooled)
        bet_fraction = self.bet_head(pooled)

        return action_logits, bet_fraction


class DeepMLPNetwork(PokerNetwork):
    """Deeper MLP with residual connections."""

    def __init__(self, config: NetworkConfig | None = None) -> None:
        super().__init__(config)

        hidden_dim = self.config.hidden_dims[0] if self.config.hidden_dims else 256

        # Input projection
        self.input_proj = nn.Sequential(
            nn.Linear(self.config.input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        )

        # Residual blocks
        self.res_blocks = nn.ModuleList()
        for _ in range(4):
            self.res_blocks.append(
                ResidualBlock(hidden_dim, dropout=self.config.dropout)
            )

        # Output heads
        self.action_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, self.config.num_actions),
        )

        self.bet_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid(),
        )

        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize weights."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass with residual connections."""
        h = self.input_proj(x)

        for block in self.res_blocks:
            h = block(h)

        action_logits = self.action_head(h)
        bet_fraction = self.bet_head(h)

        return action_logits, bet_fraction


class ResidualBlock(nn.Module):
    """Residual block for deep networks."""

    def __init__(self, dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 2, dim),
            nn.Dropout(dropout),
        )
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(x + self.net(x))


class ActorCriticMLP(PokerNetwork):
    """MLP with separate actor (policy) and critic (value) heads for RL training.

    The actor outputs action logits and bet fraction (policy).
    The critic outputs a state value estimate (for advantage computation).
    """

    def __init__(self, config: NetworkConfig | None = None) -> None:
        super().__init__(config)

        # Build shared feature extractor (backbone)
        layers: list[nn.Module] = []
        prev_dim = self.config.input_dim

        for hidden_dim in self.config.hidden_dims:
            layers.extend(
                [
                    nn.Linear(prev_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(self.config.dropout),
                ]
            )
            prev_dim = hidden_dim

        self.feature_extractor = nn.Sequential(*layers)

        # Actor head - outputs action logits
        self.action_head = nn.Linear(prev_dim, self.config.num_actions)

        # Bet sizing head - outputs fraction of max raise (0-1)
        self.bet_head = nn.Sequential(
            nn.Linear(prev_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

        # Critic head - outputs state value estimate
        self.value_head = nn.Sequential(
            nn.Linear(prev_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize weights."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass.

        Args:
            x: Input tensor of shape (batch, input_dim)

        Returns:
            action_logits: (batch, num_actions) - raw logits for actions
            bet_fraction: (batch, 1) - bet size as fraction of max (0-1)
            value: (batch, 1) - state value estimate
        """
        features = self.feature_extractor(x)
        action_logits = self.action_head(features)
        bet_fraction = self.bet_head(features)
        value = self.value_head(features)
        return action_logits, bet_fraction, value

    def get_policy(
        self, x: torch.Tensor, valid_mask: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Get action probabilities and bet fraction (policy only).

        Args:
            x: Input tensor
            valid_mask: Binary mask of valid actions (1 = valid)

        Returns:
            action_probs: Probability distribution over actions
            bet_fraction: Bet sizing fraction
        """
        action_logits, bet_fraction, _ = self(x)

        if valid_mask is not None:
            action_logits = action_logits.masked_fill(valid_mask == 0, -1e9)

        action_probs = F.softmax(action_logits, dim=-1)
        return action_probs, bet_fraction

    def get_value(self, x: torch.Tensor) -> torch.Tensor:
        """Get state value estimate only.

        Args:
            x: Input tensor

        Returns:
            value: (batch, 1) - state value estimate
        """
        features = self.feature_extractor(x)
        return self.value_head(features)


def create_network(
    architecture: str = "mlp",
    config: NetworkConfig | None = None,
) -> PokerNetwork:
    """Factory function to create networks.

    Args:
        architecture: One of "mlp", "deep_mlp", "transformer", "actor_critic"
        config: Network configuration

    Returns:
        PokerNetwork instance
    """
    architectures = {
        "mlp": MLPNetwork,
        "deep_mlp": DeepMLPNetwork,
        "transformer": TransformerNetwork,
        "actor_critic": ActorCriticMLP,
    }

    if architecture not in architectures:
        raise ValueError(
            f"Unknown architecture: {architecture}. "
            f"Choose from: {list(architectures.keys())}"
        )

    return architectures[architecture](config)
