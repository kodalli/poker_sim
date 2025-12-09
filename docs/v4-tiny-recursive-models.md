# v4: Tiny Recursive Models (TRM) Architecture

## Overview

This builds on v2 (RL training) and v3 (attention architecture). TRM represents a paradigm shift: instead of deeper networks, we use **iterative refinement** with a tiny network that recursively improves its decisions.

**Paper**: ["Less is More: Recursive Reasoning with Tiny Networks"](https://arxiv.org/abs/2510.04871)
**Reference Implementation**: [SamsungSAILMontreal/TinyRecursiveModels](https://github.com/SamsungSAILMontreal/TinyRecursiveModels)

## Why TRM for Poker?

Poker decisions naturally involve iterative reasoning:

1. **Initial read**: "I have top pair"
2. **Refine with position**: "But I'm out of position..."
3. **Consider opponent**: "Opponent is aggressive, likely bluffing"
4. **Evaluate pot odds**: "Getting 3:1, need 25% equity"
5. **Final decision**: "Call is +EV"

Current architectures (MLP, Transformer) make this decision in a single forward pass. TRM explicitly models this iterative refinement process.

### TRM vs Previous Approaches

| Aspect | v1 (GA+MLP) | v2 (PPO+MLP) | v3 (Transformer) | v4 (TRM) |
|--------|-------------|--------------|------------------|----------|
| Parameters | ~157K | ~160K | ~200K | ~50K |
| Forward passes | 1 | 1 | 1 | 8-16 |
| Reasoning | Implicit | Implicit | Attention patterns | Explicit recursion |
| Interpretability | None | None | Attention viz | Step-by-step refinement |
| Training | Evolution | PPO | PPO | Supervised + Deep supervision |

## TRM Background

### Core Idea

TRM maintains two representations that are iteratively refined:

- **y** (current solution): The model's current answer
- **z** (latent reasoning): Hidden state capturing reasoning process

At each step k:
```
z_{k+1} = LatentUpdate(input, y_k, z_k)   # Update reasoning
y_{k+1} = AnswerUpdate(z_{k+1})            # Refine answer
```

### Key Innovations

1. **Tiny network**: Just 2 layers, ~7M params (paper) - we'll use even smaller
2. **Deep supervision**: Loss computed on final K steps, not just last output
3. **No fixed-point constraints**: Unlike HRM, no complex convergence requirements
4. **Progressive refinement**: Each step improves on previous answer

### Why It Works

The paper shows TRM excels at tasks requiring multi-step reasoning:
- Sudoku-Extreme: 87.4% (vs 55% for HRM)
- Maze-Hard: 85.3% (vs 74.5%)
- ARC-AGI: 44.6% (vs 40.3%)

Poker decisions have similar structure: evaluate hand → consider position → model opponent → calculate odds → decide action.

## Poker-Adapted TRM Architecture

### State Representation

We keep the existing 433-dim encoding from v1/v2/v3:

| Component | Dims | Description |
|-----------|------|-------------|
| Hole cards | 104 | 2 × 52 one-hot |
| Community | 260 | 5 × 52 one-hot |
| Round | 4 | One-hot (preflop/flop/turn/river) |
| Position | 10 | One-hot seat position |
| Self features | 5 | chips, bet, pot_odds, dealer, blind |
| Opponent features | 45 | 9 × 5 (chips, bet, active, all_in, folded) |
| Valid actions | 5 | Binary mask |
| **Total** | **433** | |

### TRM Components

```
┌─────────────────────────────────────────────────────────────────┐
│                        Input x (433 dims)                       │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Initial Projection                           │
│              Linear(433 → 128) + LayerNorm + ReLU               │
└─────────────────────────────────────────────────────────────────┘
                                │
              ┌─────────────────┴─────────────────┐
              │                                   │
              ▼                                   ▼
    ┌──────────────────┐                ┌──────────────────┐
    │   y_0 (answer)   │                │  z_0 (latent)    │
    │   zeros(6)       │                │  zeros(64)       │
    │   [5 actions +   │                │  [reasoning      │
    │    1 bet_frac]   │                │   state]         │
    └──────────────────┘                └──────────────────┘
              │                                   │
              └─────────────────┬─────────────────┘
                                │
        ┌───────────────────────┼───────────────────────┐
        │                       │                       │
        │            ╔══════════╧══════════╗            │
        │            ║   Recursion Step k  ║            │
        │            ╚══════════╤══════════╝            │
        │                       │                       │
        │     ┌─────────────────┴─────────────────┐     │
        │     │                                   │     │
        │     ▼                                   │     │
        │  ┌─────────────────────────────────┐    │     │
        │  │         LatentUpdater           │    │     │
        │  │   concat(x_proj, y_k, z_k)      │    │     │
        │  │   → Linear(128+6+64, 128)       │    │     │
        │  │   → GELU                        │    │     │
        │  │   → Linear(128, 64)             │    │     │
        │  │   → LayerNorm                   │    │     │
        │  │   + z_k (residual)              │    │     │
        │  │   = z_{k+1}                     │    │     │
        │  └─────────────────────────────────┘    │     │
        │                   │                     │     │
        │                   ▼                     │     │
        │  ┌─────────────────────────────────┐    │     │
        │  │         AnswerUpdater           │    │     │
        │  │   Linear(64, 32) → GELU         │    │     │
        │  │   → Linear(32, 6)               │    │     │
        │  │   + y_k (residual)              │    │     │
        │  │   = y_{k+1}                     │    │     │
        │  └─────────────────────────────────┘    │     │
        │                   │                     │     │
        │                   ▼                     │     │
        │            [Repeat K times]             │     │
        │                                         │     │
        └─────────────────────────────────────────┘     │
                                                        │
                                ▼                       │
┌─────────────────────────────────────────────────────────────────┐
│                      Final Output y_K                           │
│                                                                 │
│   action_logits = y_K[:5]     # 5 action logits                 │
│   bet_fraction = sigmoid(y_K[5])  # continuous [0, 1]           │
└─────────────────────────────────────────────────────────────────┘
```

### Reasoning Interpretation

Each recursion step can be interpreted as a reasoning stage:

| Step | Latent z focuses on | Answer y refines |
|------|---------------------|------------------|
| 1-2 | Hand strength (cards → made hand) | Initial action tendency |
| 3-4 | Position awareness | Adjust for position |
| 5-6 | Opponent modeling | Consider opponent ranges |
| 7-8 | Pot odds / stack sizes | Final +EV decision |

### Network Dimensions

```python
@dataclass
class TRMConfig:
    input_dim: int = 433
    proj_dim: int = 128        # Input projection
    latent_dim: int = 64       # z dimension
    answer_dim: int = 6        # y dimension (5 actions + bet)
    num_steps: int = 8         # K recursion steps
    supervision_steps: int = 4  # Compute loss on last N steps
    dropout: float = 0.1
```

**Parameter count**: ~50K (vs 157K for MLP)
- Input projection: 433 × 128 = 55K
- LatentUpdater: (128+6+64) × 128 + 128 × 64 = 33K
- AnswerUpdater: 64 × 32 + 32 × 6 = 2K
- LayerNorms, biases: ~1K

## Implementation Plan

### Phase 1: Core TRM Network

#### Files to Create

```
agents/neural/
├── trm.py              # TRMNetwork implementation
└── trm_agent.py        # TRMAgent wrapper
```

#### `agents/neural/trm.py`

```python
class LatentUpdater(nn.Module):
    """Updates reasoning state z given input, current answer, and previous z."""

    def __init__(self, proj_dim: int, answer_dim: int, latent_dim: int, dropout: float):
        super().__init__()
        input_dim = proj_dim + answer_dim + latent_dim
        self.net = nn.Sequential(
            nn.Linear(input_dim, proj_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(proj_dim, latent_dim),
            nn.LayerNorm(latent_dim),
        )

    def forward(self, x_proj, y, z):
        combined = torch.cat([x_proj, y, z], dim=-1)
        return z + self.net(combined)  # Residual connection


class AnswerUpdater(nn.Module):
    """Updates answer y given reasoning state z."""

    def __init__(self, latent_dim: int, answer_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.GELU(),
            nn.Linear(32, answer_dim),
        )

    def forward(self, z, y_prev):
        return y_prev + self.net(z)  # Residual connection


class TRMNetwork(PokerNetwork):
    """Tiny Recursive Model for poker decisions."""

    def __init__(self, config: TRMConfig):
        super().__init__()
        self.config = config

        # Input projection
        self.input_proj = nn.Sequential(
            nn.Linear(config.input_dim, config.proj_dim),
            nn.LayerNorm(config.proj_dim),
            nn.ReLU(),
        )

        # Recursive components (shared across steps)
        self.latent_updater = LatentUpdater(
            config.proj_dim, config.answer_dim, config.latent_dim, config.dropout
        )
        self.answer_updater = AnswerUpdater(config.latent_dim, config.answer_dim)

    def forward(self, x: torch.Tensor, return_all_steps: bool = False):
        batch_size = x.shape[0]
        device = x.device

        # Project input
        x_proj = self.input_proj(x)

        # Initialize y and z
        y = torch.zeros(batch_size, self.config.answer_dim, device=device)
        z = torch.zeros(batch_size, self.config.latent_dim, device=device)

        all_y = []

        # Recursive refinement
        for k in range(self.config.num_steps):
            z = self.latent_updater(x_proj, y, z)
            y = self.answer_updater(z, y)
            all_y.append(y)

        # Final output
        action_logits = y[:, :5]
        bet_fraction = torch.sigmoid(y[:, 5:6])

        if return_all_steps:
            return action_logits, bet_fraction, all_y
        return action_logits, bet_fraction
```

### Phase 2: Training with Deep Supervision

#### Loss Function

```python
def trm_loss(
    model: TRMNetwork,
    states: torch.Tensor,
    target_actions: torch.Tensor,
    target_bets: torch.Tensor,
    valid_masks: torch.Tensor,
):
    """Compute loss with deep supervision on final K steps."""
    action_logits, bet_fraction, all_y = model(states, return_all_steps=True)

    total_loss = 0.0
    supervision_steps = model.config.supervision_steps

    # Only supervise last N steps
    for step_y in all_y[-supervision_steps:]:
        step_logits = step_y[:, :5]
        step_bet = torch.sigmoid(step_y[:, 5:6])

        # Mask invalid actions
        masked_logits = step_logits.masked_fill(~valid_masks.bool(), -1e9)

        # Action loss (cross-entropy)
        action_loss = F.cross_entropy(masked_logits, target_actions)

        # Bet loss (MSE, only for raise actions)
        raise_mask = (target_actions == 3)  # RAISE action
        if raise_mask.any():
            bet_loss = F.mse_loss(step_bet[raise_mask], target_bets[raise_mask])
        else:
            bet_loss = 0.0

        total_loss += action_loss + 0.5 * bet_loss

    return total_loss / supervision_steps
```

### Phase 3: Data Generation

Since TRM uses supervised learning, we need training data. Options:

#### Option A: Self-Play with Expert Labels
```python
def generate_training_data(num_games: int):
    """Play games and record state-action pairs."""
    data = []

    for game in play_games(num_games):
        for decision in game.decisions:
            # Record state and action taken
            data.append({
                'state': encode_state(decision.table_state),
                'action': decision.action.action_type.value,
                'bet_fraction': decision.bet_fraction,
                'result': game.chip_delta,  # For weighting
            })

    return data
```

#### Option B: Distillation from Larger Model
```python
def distill_from_teacher(teacher_model, num_samples: int):
    """Generate training data from a stronger model."""
    data = []

    for state in sample_states(num_samples):
        # Get teacher's decision
        action_probs = teacher_model.get_action_distribution(state)
        bet_fraction = teacher_model.get_bet_fraction(state)

        data.append({
            'state': encode_state(state),
            'action_probs': action_probs,  # Soft labels
            'bet_fraction': bet_fraction,
        })

    return data
```

#### Option C: Hybrid with RL Fine-tuning
1. Pre-train TRM on self-play data (supervised)
2. Fine-tune with PPO (from v2 infrastructure)

### Phase 4: Integration

#### Add to network factory

```python
# In agents/neural/network.py

def create_network(architecture: str, config=None):
    if architecture == "trm":
        from agents.neural.trm import TRMNetwork, TRMConfig
        trm_config = TRMConfig(**(config or {}))
        return TRMNetwork(trm_config)
    # ... existing architectures
```

#### CLI command

```python
# In main.py

@app.command()
def train_trm(
    model: str = typer.Option("v4", "--model", "-m"),
    epochs: int = typer.Option(100, "--epochs", "-e"),
    batch_size: int = typer.Option(64, "--batch-size", "-b"),
    num_steps: int = typer.Option(8, "--steps", "-k", help="TRM recursion steps"),
    data_source: str = typer.Option("self-play", "--data", help="self-play or distill"),
):
    """Train a TRM poker model with supervised learning."""
    ...
```

## Training Strategy

### Curriculum Learning

1. **Stage 1**: Train on simple scenarios (heads-up, single street)
2. **Stage 2**: Add complexity (multi-way pots, multiple streets)
3. **Stage 3**: Full game with all streets and player counts

### Hyperparameters

```python
# TRM specific
num_steps = 8              # Recursion depth
supervision_steps = 4      # Deep supervision window
latent_dim = 64            # z dimension

# Training
learning_rate = 1e-3
batch_size = 128
epochs = 100
weight_decay = 1e-4

# Data
games_per_epoch = 10000
self_play_temperature = 0.5  # Exploration in data generation
```

### Training Loop

```python
for epoch in range(epochs):
    # Generate fresh self-play data each epoch
    data = generate_training_data(games_per_epoch)

    for batch in dataloader(data, batch_size):
        loss = trm_loss(model, batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Evaluate against baselines
    if epoch % 10 == 0:
        eval_vs_random = evaluate(model, RandomAgent())
        eval_vs_tag = evaluate(model, TightAggressiveAgent())
```

## Files to Create/Modify

```
poker_sim/
├── agents/neural/
│   ├── trm.py                 # TRMNetwork, TRMConfig
│   └── trm_agent.py           # TRMAgent wrapper
├── training/
│   ├── __init__.py
│   ├── data_generator.py      # Self-play data generation
│   ├── trm_trainer.py         # Training loop with deep supervision
│   └── distillation.py        # (Optional) distillation from teacher
└── main.py                    # Add train-trm command
```

## Success Metrics

| Metric | v3 (Transformer) | v4 Target (TRM) |
|--------|------------------|-----------------|
| Parameters | ~200K | <50K |
| Inference time | 1× | 2-4× (K steps, but smaller) |
| Win rate vs Random | 70% | >70% |
| Win rate vs TAG | 55% | >55% |
| Training data efficiency | 100K games | <50K games |
| Interpretability | Attention maps | Step-by-step decisions |

## Advantages

1. **Explicit reasoning**: Each step has semantic meaning
2. **Tiny footprint**: 4× fewer parameters than MLP
3. **Adaptive compute**: Can adjust K based on decision complexity
4. **Debuggable**: Inspect y at each step to see decision evolution
5. **Data efficient**: Deep supervision provides dense gradients

## Potential Challenges

1. **Sequential steps**: Cannot parallelize K steps (latency trade-off)
2. **Training stability**: Recursive networks can be sensitive
3. **Data quality**: Supervised learning requires good training data

## Visualization

```python
def visualize_decision(model, state):
    """Show how decision evolves over recursion steps."""
    action_logits, bet_fraction, all_y = model(state, return_all_steps=True)

    print("Step | FOLD  CHECK  CALL  RAISE  ALL_IN | Bet")
    print("-" * 50)

    for k, y in enumerate(all_y):
        probs = F.softmax(y[:, :5], dim=-1).squeeze()
        bet = torch.sigmoid(y[:, 5]).item()
        print(f"  {k+1}  | {probs[0]:.2f}  {probs[1]:.2f}   {probs[2]:.2f}  {probs[3]:.2f}   {probs[4]:.2f}  | {bet:.2f}")
```

Example output:
```
Step | FOLD  CHECK  CALL  RAISE  ALL_IN | Bet
--------------------------------------------------
  1  | 0.20  0.20   0.20  0.20   0.20  | 0.50   # Uniform (no info yet)
  2  | 0.15  0.25   0.25  0.25   0.10  | 0.45   # Hand strength assessed
  3  | 0.10  0.20   0.30  0.30   0.10  | 0.40   # Position considered
  4  | 0.08  0.15   0.35  0.35   0.07  | 0.38   # Opponent modeled
  5  | 0.05  0.10   0.40  0.40   0.05  | 0.35   # Pot odds calculated
  6  | 0.03  0.07   0.45  0.42   0.03  | 0.32   # Converging...
  7  | 0.02  0.05   0.50  0.40   0.03  | 0.30   # Nearly decided
  8  | 0.01  0.04   0.55  0.38   0.02  | 0.28   # Final: CALL
```

---

# TRM + PPO: Reinforcement Learning Approach

## Motivation

The original TRM paper uses **supervised learning** with labeled data. However, this has limitations for poker:

1. **Labeled data constrains learning**: Model can only imitate existing play
2. **No novel strategies**: Can't discover strategies beyond training data
3. **Data generation overhead**: Need to create/curate labeled datasets

**Our insight**: TRM's iterative refinement is analogous to **tree search in latent space**:

```
MCTS/CFR:  Explicitly explores decision tree → Better strategy
TRM:       Implicitly refines in latent space → Better decision

Both use multiple "passes" to improve decisions.
```

## TRM + PPO Architecture

Replace supervised learning with PPO (Proximal Policy Optimization):

```
┌─────────────────────────────────────────────────────────────┐
│                     TRM + PPO Architecture                  │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Input: obs (433 dims)                                      │
│           │                                                 │
│           ▼                                                 │
│  ┌─────────────────┐                                        │
│  │ Input Projection │  Linear(433→128) + LayerNorm + ReLU   │
│  └─────────────────┘                                        │
│           │                                                 │
│     ┌─────┴─────┐                                           │
│     │           │                                           │
│     ▼           ▼                                           │
│   y_0=0       z_0=0                                         │
│     │           │                                           │
│     └─────┬─────┘                                           │
│           │                                                 │
│     ╔═════╧═════╗                                           │
│     ║ K steps   ║  ←── Iterative refinement (K=8)           │
│     ║ of refine ║      Each step: z = update(x, y, z)       │
│     ╚═════╤═════╝               y = refine(z, y)            │
│           │                                                 │
│     ┌─────┴─────┐                                           │
│     │           │                                           │
│     ▼           ▼                                           │
│  ┌──────┐   ┌──────┐                                        │
│  │Policy│   │Value │  ←── Two heads for actor-critic        │
│  │ Head │   │ Head │                                        │
│  └──────┘   └──────┘                                        │
│     │           │                                           │
│     ▼           ▼                                           │
│  action     value                                           │
│  logits    estimate                                         │
│     │           │                                           │
│     └─────┬─────┘                                           │
│           │                                                 │
│           ▼                                                 │
│     PPO Loss (no labels needed!)                            │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### Key Differences from Supervised TRM

| Aspect | Supervised TRM | TRM + PPO |
|--------|----------------|-----------|
| Training signal | Labeled (state, action) pairs | Rewards from environment |
| Data source | Pre-generated dataset | Online self-play |
| Novel strategies | Limited to training data | Can emerge through exploration |
| Value estimation | Not needed | Critic head required |
| Loss function | Cross-entropy on labels | PPO surrogate objective |

### JAX/Flax Implementation

```python
class TRMActorCritic(nn.Module):
    """TRM with actor-critic heads for PPO training."""

    proj_dim: int = 128
    latent_dim: int = 64
    answer_dim: int = 9      # 9 actions in v3
    num_steps: int = 8       # K recursion steps
    dropout_rate: float = 0.1

    @nn.compact
    def __call__(self, x: Array, training: bool = True) -> tuple[Array, Array]:
        """Forward pass returning policy logits and value.

        Args:
            x: [batch, 433] observations

        Returns:
            action_logits: [batch, 9] policy logits
            value: [batch, 1] state value estimate
        """
        batch_size = x.shape[0]

        # Input projection
        x_proj = nn.Dense(self.proj_dim)(x)
        x_proj = nn.LayerNorm()(x_proj)
        x_proj = nn.relu(x_proj)

        # Initialize y (answer) and z (latent reasoning state)
        y = jnp.zeros((batch_size, self.answer_dim))
        z = jnp.zeros((batch_size, self.latent_dim))

        # Shared updaters (parameters reused across steps)
        latent_updater = LatentUpdater(
            self.proj_dim, self.answer_dim, self.latent_dim, self.dropout_rate
        )
        answer_updater = AnswerUpdater(self.latent_dim, self.answer_dim)

        # Iterative refinement
        for k in range(self.num_steps):
            z = latent_updater(x_proj, y, z, training)
            y = answer_updater(z, y)

        # Policy head: action logits from final y
        action_logits = y  # y is already [batch, 9]

        # Value head: from final latent state z
        value_hidden = nn.Dense(64)(z)
        value_hidden = nn.relu(value_hidden)
        value = nn.Dense(1)(value_hidden)

        return action_logits, value


class LatentUpdater(nn.Module):
    """Updates reasoning state z."""
    proj_dim: int
    answer_dim: int
    latent_dim: int
    dropout_rate: float

    @nn.compact
    def __call__(self, x_proj, y, z, training):
        combined = jnp.concatenate([x_proj, y, z], axis=-1)
        h = nn.Dense(self.proj_dim)(combined)
        h = nn.gelu(h)
        h = nn.Dropout(self.dropout_rate, deterministic=not training)(h)
        h = nn.Dense(self.latent_dim)(h)
        h = nn.LayerNorm()(h)
        return z + h  # Residual connection


class AnswerUpdater(nn.Module):
    """Updates answer y from latent z."""
    latent_dim: int
    answer_dim: int

    @nn.compact
    def __call__(self, z, y_prev):
        h = nn.Dense(32)(z)
        h = nn.gelu(h)
        h = nn.Dense(self.answer_dim)(h)
        return y_prev + h  # Residual connection
```

## Connection to Game Theory

### TRM as Implicit Tree Search

Traditional poker AI uses explicit tree search (CFR, MCTS):

```
CFR/MCTS:
                    [Current State]
                    /      |      \
                 Fold    Call    Raise
                  /        |        \
            [eval]    [recurse]   [recurse]
                         ...        ...

Explicit tree: O(actions^depth) nodes
```

TRM does something analogous in **latent space**:

```
TRM Refinement:
Step 1: z₁ encodes "what are my options?"
Step 2: z₂ encodes "what if I fold? call? raise?"
Step 3: z₃ encodes "how might opponent respond?"
Step 4: z₄ encodes "what's my expected value?"
...
Step K: z_K has "searched" the decision space

Implicit search: O(K) forward passes, fixed compute
```

The key insight: **TRM learns WHAT to reason about**, while CFR/MCTS have hand-crafted search procedures.

### Emergent Multi-Step Reasoning

With PPO training, TRM can learn:

| Step | What Model Might Learn |
|------|------------------------|
| 1-2 | Hand strength evaluation |
| 3-4 | Position and pot odds |
| 5-6 | Opponent modeling (what would they do?) |
| 7-8 | Counterfactual reasoning (what if I...?) |

This is **not prescribed** - it emerges from optimizing reward.

## Integration with Anti-Exploitation

TRM + PPO naturally combines with anti-exploitation features (see `docs/anti-exploitation.md`):

```python
# TRM training config with anti-exploitation
trm_training_config = {
    # TRM architecture
    "num_steps": 8,
    "latent_dim": 64,
    "proj_dim": 128,

    # PPO settings
    "learning_rate": 3e-4,
    "entropy_coef": 0.05,  # Higher entropy for exploration

    # Anti-exploitation opponent mix
    "opponent_mix": {
        "self": 0.30,
        "historical": 0.25,
        "trapper": 0.15,
        "call_station": 0.10,
        "tag": 0.08,
        "rock": 0.07,
        "random": 0.05,
    },
}
```

## Expected Benefits

### vs Supervised TRM

1. **Novel strategies**: Not limited to imitation
2. **No labeling**: Train directly from self-play
3. **Continuous improvement**: Can exceed any teacher

### vs MLP + PPO (Current v3)

1. **Explicit reasoning steps**: Interpretable decision process
2. **Smaller model**: 50K params vs 160K
3. **Adaptive compute**: More passes for hard decisions (future)
4. **Better generalization**: Reasoning transfers across situations

## Implementation Plan

### Phase 1: TRM Network in JAX

1. Create `poker_jax/trm_network.py`
   - `TRMActorCritic` class
   - `LatentUpdater`, `AnswerUpdater` modules

2. Add to network factory
   - `create_network("trm")` option

### Phase 2: Training Integration

1. Modify `training/jax_trainer.py`
   - Support TRM network in `_collect_step_mixed`
   - No changes to PPO (same loss function!)

2. Add CLI option
   - `--architecture trm` flag

### Phase 3: Evaluation & Comparison

1. Train TRM with same opponent mix as v3.3
2. Compare metrics:
   - Win rate vs all opponents
   - Action diversity (entropy)
   - Training speed (steps/sec)
   - Interpretability (step-by-step visualization)

## Training Command (v4)

```bash
nohup uv run python main.py train-rl-jax \
  --model v4 \
  --architecture trm \
  --steps 1000000000 \
  --parallel 1536 \
  --historical-selfplay \
  --entropy-coef 0.05 \
  --tensorboard logs/v4 \
  --desc "TRM with PPO and anti-exploitation" \
  > nohup_v4.out 2>&1 &
```

## Open Questions

1. **Optimal K**: How many refinement steps? 8? 16? Adaptive?
2. **Latent supervision**: Should we add auxiliary losses on intermediate z?
3. **Opponent embedding**: Feed opponent type into TRM for adaptive play?
4. **Compute vs params**: Is 8 passes of 50K better than 1 pass of 160K?

## References

- [TRM Paper](https://arxiv.org/abs/2510.04871): "Less is More: Recursive Reasoning with Tiny Networks"
- [TRM Code](https://github.com/SamsungSAILMontreal/TinyRecursiveModels)
- [PPO Paper](https://arxiv.org/abs/1707.06347): "Proximal Policy Optimization Algorithms"
- [MuZero](https://arxiv.org/abs/1911.08265): Learned model for planning (related concept)
