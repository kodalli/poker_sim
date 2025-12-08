# Architecture Analysis & Improvement Roadmap

Current JAX poker training architecture and gaps vs state-of-the-art.

## Current Architecture

### Network (`poker_jax/network.py`)
```
Input: 381 features
    ↓
Dense(256) → LayerNorm → ReLU → Dropout(0.1)
    ↓
Dense(128) → LayerNorm → ReLU → Dropout(0.1)
    ↓
Dense(64) → LayerNorm → ReLU → Dropout(0.1)
    ↓
┌─────────────────┬──────────────────┐
│  Actor Head     │   Critic Head    │
│  Dense(5)       │   Dense(64)→ReLU │
│  (action logits)│   Dense(1)       │
└─────────────────┴──────────────────┘

Parameters: ~50k
```

### State Encoding (`poker_jax/encoding.py`)

| Feature | Dim | Description |
|---------|-----|-------------|
| Hole cards | 104 | 2 cards × 52 one-hot |
| Community | 260 | 5 cards × 52 one-hot |
| Round | 4 | preflop/flop/turn/river one-hot |
| Position | 2 | button or not |
| Pot | 1 | normalized by starting chips |
| My chips | 1 | normalized |
| Opp chips | 1 | normalized |
| My bet | 1 | normalized |
| Opp bet | 1 | normalized |
| To call | 1 | normalized |
| Valid actions | 5 | fold/check/call/raise/all-in mask |
| **Total** | **381** | |

### Actions
5 discrete: Fold, Check, Call, Raise (min-raise), All-in

---

## Critical Gaps vs State-of-the-Art

### 1. No Action History (CRITICAL)

**Problem**: Model sees only current snapshot, not how we got here.

Poker is about *reading opponents*:
- Did they check-raise? (strength signal)
- Did they limp pre then bet flop? (weak range)
- How big was their bet? (sizing tells)

**Current**: Can't distinguish "opponent bet pot" from "opponent min-bet"

**Fix**: Add action sequence encoding (last N actions with bet sizes)

```python
# Example action history encoding
# For each of last 10 actions:
# - player_id: 1 (who acted)
# - action_type: 5 (fold/check/call/raise/all-in one-hot)
# - bet_size: 1 (normalized amount)
# Total: 10 * 7 = 70 additional features
```

---

### 2. Crude Bet Sizing (CRITICAL)

**Problem**: Only 5 discrete actions. Real poker has continuous sizing.

Real poker bet sizing matters enormously:
- Small bet (25-33% pot) = probe/draw/blocking bet
- Medium bet (50-75% pot) = standard value/bluff
- Pot bet (100%) = polarized range
- Overbet (150%+ pot) = very polarized (nuts or air)

**Current**: "Raise" means min-raise only. No strategic sizing.

**Fix Options**:

**Option A: Multiple discrete sizes**
```python
# Expand from 5 to 9 actions:
ACTIONS = [
    FOLD,
    CHECK,
    CALL,
    RAISE_33,   # 33% of pot
    RAISE_66,   # 66% of pot
    RAISE_100,  # Pot-size bet
    RAISE_150,  # 1.5x pot
    RAISE_200,  # 2x pot (overbet)
    ALL_IN,
]
```

**Option B: Continuous sizing (hybrid)**
```python
# Use existing bet_fraction head (currently unused)
# Action selects category, fraction selects exact size
action_logits, bet_fraction, value = network(obs)
# If action == RAISE: actual_bet = pot * bet_fraction
```

---

### 3. No Hand Strength Features (MAJOR)

**Problem**: Network must learn hand evaluation from raw one-hots.

Learning "A♠K♠ on A♣7♣2♦ = top pair top kicker" from scratch is extremely hard.

**Professional poker AIs include**:
- Hand rank category (high card, pair, two pair, trips, straight, flush, etc.)
- Hand strength percentile (equity vs random hand)
- Draw potential (flush draw, straight draw, backdoor draws)
- Board texture (wet/dry, paired, monotone)

**Fix**: Add precomputed features

```python
# Additional hand strength features (~20 dims):
hand_rank = one_hot(get_hand_rank(hole, community), 10)  # 10 categories
equity = monte_carlo_equity(hole, community, 100)  # [0, 1]
flush_draw = has_flush_draw(hole, community)  # 0 or 1
straight_draw = has_straight_draw(hole, community)  # 0 or 1
nut_advantage = compute_nut_advantage(hole, community)  # [0, 1]
```

**Note**: Can precompute lookup tables for common situations.

---

### 4. No Opponent Modeling (MAJOR)

**Problem**: Single policy plays same way against everyone.

Good poker means *adapting*:
- Bluff more vs tight players who fold too much
- Value bet thin vs calling stations
- Trap vs loose-aggressive players

**Current**: Same strategy regardless of opponent tendencies

**Fix Options**:

**Option A: Population-based training**
- Train against pool of diverse opponents
- Model learns robust strategy

**Option B: Opponent embedding**
```python
# Track opponent stats over session:
opp_stats = [
    vpip,          # Voluntarily put in pot %
    pfr,           # Preflop raise %
    aggression,    # Bet+Raise / Call ratio
    fold_to_cbet,  # Fold to continuation bet %
]
# Concatenate to observation
```

**Option C: Meta-learning**
- Train model to quickly adapt to new opponents
- More complex but powerful

---

### 5. Small Network (MODERATE)

**Problem**: 256→128→64 (~50k params) may lack capacity.

**Comparison**:
| System | Parameters | Notes |
|--------|------------|-------|
| Our MLP | ~50k | Current |
| ReBeL (Facebook) | ~1M | Poker + general games |
| Pluribus | Millions | 6-player NLHE champion |

**Fix**: Scale up network

```python
# Option A: Wider MLP
hidden_dims = (512, 256, 128)  # ~200k params

# Option B: Residual MLP
class ResidualMLP(nn.Module):
    def __call__(self, x):
        for _ in range(4):
            residual = x
            x = Dense(256)(x)
            x = LayerNorm()(x)
            x = relu(x)
            x = Dense(256)(x)
            x = x + residual  # Skip connection
        return x
```

---

### 6. PPO vs CFR (FUNDAMENTAL)

**Problem**: PPO is designed for MDPs, not imperfect information games.

Poker is a *partially observable, multi-agent* game. Key distinction:

| Property | MDP (PPO designed for) | Poker |
|----------|------------------------|-------|
| Observability | Full | Partial (hidden cards) |
| Players | Single agent | Two+ adversarial agents |
| Equilibrium | Optimal policy | Nash equilibrium |
| Convergence | Guaranteed | Not guaranteed with self-play |

**State-of-the-art uses**:
- **CFR** (Counterfactual Regret Minimization): Provably converges to Nash
- **MCCFR**: Monte Carlo variant for large games
- **Deep CFR**: Neural network function approximation + CFR
- **ReBeL**: Combines RL with search, works for any game

**Current**: Pure PPO self-play. May not converge to Nash, could be exploitable.

**Fix**: Implement Deep CFR (significant undertaking)

```python
# Deep CFR overview:
# 1. Build game tree abstractions
# 2. For each information set:
#    - Compute counterfactual values
#    - Update regrets
#    - Network predicts regret-matching strategy
# 3. Use reservoir sampling for training data
```

---

### 7. Self-Play Instability (MODERATE)

**Problem**: Training against copy of self can cause cycling.

Rock-paper-scissors dynamics:
- Model learns to beat current opponent
- But loses to what it used to do
- Cycles without converging

**Current**: Both players share same weights (latest)

**Fix Options**:

**Option A: Frozen opponent pool**
```python
# Keep pool of past checkpoints
opponent_pool = load_checkpoints(last_n=10)
# Sample opponent from pool each game
opponent = random.choice(opponent_pool)
```

**Option B: League training (AlphaStar style)**
- Main agents that improve
- Exploiter agents that find weaknesses
- League of past versions

**Option C: Prioritized fictitious self-play (PSRO)**
- Game-theoretic approach
- Provably converges to Nash

---

### 8. Broken Transformer (MINOR)

**Problem**: `ActorCriticTransformer` in network.py is broken.

Currently operates on single token (entire 381-dim observation). Self-attention on 1 token is equivalent to a linear layer.

**Proper approach**: Tokenize inputs
```python
# Example tokenization:
# - 7 card tokens (2 hole + 5 community), each with:
#   - rank embedding (13 dims)
#   - suit embedding (4 dims)
#   - position embedding (7 dims)
# - Action history tokens
# - Game state token (pot, chips, etc.)

# Then apply transformer:
card_tokens = encode_cards(hole_cards, community)  # [7, 24]
action_tokens = encode_actions(history)            # [N, action_dim]
state_token = encode_state(pot, chips, etc.)       # [1, state_dim]
all_tokens = concat([card_tokens, action_tokens, state_token])
output = transformer(all_tokens)
```

---

## Priority Ranking

| Priority | Gap | Impact | Effort | ROI |
|----------|-----|--------|--------|-----|
| 1 | Bet sizing | High | Low | Highest |
| 2 | Hand strength features | High | Medium | High |
| 3 | Action history | High | Medium | High |
| 4 | Network scaling | Medium | Low | Good |
| 5 | Opponent pool | Medium | Medium | Medium |
| 6 | Deep CFR | Very High | Very High | Long-term |

---

## Recommended Implementation Phases

### Phase 1: Quick Wins (1-2 days)

**1.1 Multiple bet sizes**
- Files: `poker_jax/state.py`, `poker_jax/game.py`, `poker_jax/encoding.py`
- Change: Add RAISE_33, RAISE_66, RAISE_100, RAISE_150 actions
- Impact: Much richer strategy space

**1.2 Scale network**
- Files: `poker_jax/network.py`, `training/jax_trainer.py`
- Change: `hidden_dims = (512, 256, 128)`
- Impact: More capacity for complex strategies

### Phase 2: Better Features (3-5 days)

**2.1 Hand strength features**
- Files: `poker_jax/encoding.py`, `poker_jax/hands.py`
- Add: Hand rank, equity estimate, draw detection
- Impact: Much easier for network to learn

**2.2 Action history**
- Files: `poker_jax/state.py`, `poker_jax/encoding.py`
- Add: Last 10 actions with bet sizes
- Impact: Can read opponent patterns

### Phase 3: Training Improvements (1 week+)

**3.1 Opponent pool**
- Files: `training/jax_trainer.py`
- Add: Pool of past checkpoints, sample opponents
- Impact: More robust strategies

**3.2 Evaluate with baselines**
- Files: `scripts/evaluate_vs_baselines.py`
- Add: Test against known opponents
- Impact: Measure actual poker skill

### Phase 4: Algorithmic Upgrade (Major project)

**4.1 Deep CFR**
- New module: `training/deep_cfr.py`
- Major undertaking but provides Nash convergence
- Consider ReBeL paper as modern approach

---

## References

- [Libratus](https://science.sciencemag.org/content/359/6374/418) - Superhuman heads-up NLHE (2017)
- [Pluribus](https://science.sciencemag.org/content/365/6456/885) - Superhuman 6-player NLHE (2019)
- [Deep CFR](https://arxiv.org/abs/1811.00164) - Neural network + CFR (2019)
- [ReBeL](https://arxiv.org/abs/2007.13544) - RL + Search for imperfect info games (2020)
- [AlphaStar](https://www.nature.com/articles/s41586-019-1724-z) - League training for StarCraft (2019)
