# v3: Attention-Based Architecture

## Prerequisites

This builds on v2 (RL training). The attention architecture will use the same PPO training loop but with a restructured network that properly leverages transformers.

## Problem with Current Encoding

The v1/v2 flat 433-dim vector destroys relational structure:

```
Current: [card1_onehot(52), card2_onehot(52), ..., player_features, ...]
         └─────────────────────────────────────────────────────────────┘
                              Flat, no structure
```

One-hot cards mean:
- Network must re-learn "A♠ and K♠ are suited" from raw bits
- No weight sharing between similar cards
- Position in vector matters, but relationships don't

## Proposed: Token-Based Encoding

### Token Types

| Token | Count | Embedding | Content |
|-------|-------|-----------|---------|
| Hole cards | 2 | Card embedding | Your private cards |
| Community | 5 | Card embedding | Board cards (masked if not dealt) |
| Self | 1 | Player embedding | Your chips, position, bets |
| Opponents | 9 | Player embedding | Their chips, position, bets, status |
| Context | 1 | Context embedding | Round, pot size, action history |

**Total: 18 tokens** (variable based on player count)

### Card Embedding

Instead of 52-dim one-hot, learn dense representations:

```python
class CardEmbedding(nn.Module):
    def __init__(self, embed_dim=64):
        self.rank_embed = nn.Embedding(13, embed_dim // 2)  # 2-A
        self.suit_embed = nn.Embedding(4, embed_dim // 2)   # ♠♥♦♣

    def forward(self, cards):
        # cards: (batch, num_cards, 2) -> rank, suit indices
        rank_emb = self.rank_embed(cards[:, :, 0])
        suit_emb = self.suit_embed(cards[:, :, 1])
        return torch.cat([rank_emb, suit_emb], dim=-1)
```

Benefits:
- A♠ and A♥ share rank embedding (learns "ace-ness")
- ♠ cards share suit embedding (learns "spade-ness")
- Much smaller: 64 dims vs 52 one-hot per card
- Combinatorial generalization

### Player Embedding

```python
class PlayerEmbedding(nn.Module):
    def __init__(self, embed_dim=64):
        self.position_embed = nn.Embedding(10, 16)
        self.feature_proj = nn.Linear(5, embed_dim - 16)

    def forward(self, position, features):
        # features: chip_ratio, bet_ratio, is_active, is_all_in, is_dealer
        pos_emb = self.position_embed(position)
        feat_emb = self.feature_proj(features)
        return torch.cat([pos_emb, feat_emb], dim=-1)
```

### Attention Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     Input Tokens                            │
│  [Card₁] [Card₂] [Comm₁..₅] [Self] [Opp₁..₉] [Context]     │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                   Token Type Embeddings                     │
│         + Learned embeddings for token types                │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│              Transformer Encoder (4 layers)                 │
│                                                             │
│   Layer 1: Self-attention + FFN                            │
│   Layer 2: Self-attention + FFN                            │
│   Layer 3: Self-attention + FFN                            │
│   Layer 4: Self-attention + FFN                            │
│                                                             │
│   Attention patterns learned:                               │
│   - Cards attend to cards (hand strength)                   │
│   - Self attends to opponents (relative position)           │
│   - Context attends to all (game state summary)             │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    Output Heads                             │
│                                                             │
│   [CLS] token → Policy head → action_logits, bet_fraction   │
│   [CLS] token → Value head  → state_value                   │
└─────────────────────────────────────────────────────────────┘
```

### What Attention Can Learn

1. **Hand strength patterns**
   - Hole cards attending to community cards
   - Detecting pairs, flushes, straights via attention weights

2. **Positional relationships**
   - Self attending to opponents based on relative position
   - Learning "player after me raised" vs "player before me raised"

3. **Opponent modeling**
   - Attending more to active/aggressive opponents
   - Learning betting patterns per position

4. **Board texture**
   - Community cards attending to each other
   - Detecting draw-heavy vs dry boards

### Implementation Plan

#### Phase 1: New Encoder
- [ ] Card embedding module (`agents/neural/embeddings.py`)
- [ ] Player embedding module
- [ ] Context embedding module
- [ ] Token-based state encoder (`agents/neural/encoding_v2.py`)

#### Phase 2: Transformer Network
- [ ] Proper transformer with token inputs (`network.py`)
- [ ] CLS token for aggregation
- [ ] Multi-head attention with interpretable heads

#### Phase 3: Integration
- [ ] Swap encoder in training pipeline
- [ ] Ensure backward compatibility with v2 training
- [ ] Add attention visualization for debugging

#### Phase 4: Evaluation
- [ ] Compare v3 vs v2 on same training budget
- [ ] Analyze attention patterns
- [ ] Test generalization to different game formats

### Files to Create/Modify

```
poker_sim/
├── agents/neural/
│   ├── embeddings.py         # Card, Player, Context embeddings
│   ├── encoding_v2.py        # Token-based encoder
│   ├── transformer.py        # Proper transformer architecture
│   └── network.py            # Add transformer_v2 option
└── visualization/
    └── attention.py          # Attention pattern visualization
```

### Architecture Comparison

| Aspect | v1/v2 MLP | v3 Transformer |
|--------|-----------|----------------|
| Input dim | 433 (flat) | 18 tokens × 64 dims |
| Card repr | One-hot (52) | Learned (64) |
| Parameters | ~157K | ~200K |
| Relationships | Implicit in weights | Explicit in attention |
| Interpretability | Black box | Attention visualization |
| Generalization | Poor | Better (compositional) |

### Attention Heads Design

Use 4 attention heads with different focuses:

| Head | Focus | What it might learn |
|------|-------|---------------------|
| 1 | Cards | Suit patterns, rank patterns |
| 2 | Position | Relative player positions |
| 3 | Betting | Who's aggressive, pot odds |
| 4 | Global | Overall game state |

### Success Metrics

| Metric | v2 (MLP+RL) | v3 Target |
|--------|-------------|-----------|
| Win rate vs TAG | 55% | >60% |
| Novel situation perf | Baseline | +10% |
| Training stability | Good | Same/better |
| Attention interpretable | N/A | Yes |
