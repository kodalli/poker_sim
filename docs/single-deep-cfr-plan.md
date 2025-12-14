# Single Deep CFR Implementation Plan

## Overview

Replace tabular CFR with neural network-based **Single Deep CFR (SD-CFR)** to eliminate abstraction limitations and achieve better generalization.

### Why SD-CFR over Deep CFR?

| Aspect | Deep CFR | Single Deep CFR |
|--------|----------|-----------------|
| Networks | 2 (advantage + strategy) | 1 (advantage only) |
| Memory | 2 replay buffers | 1 replay buffer |
| Averaging | Separate network | Single network with linear CFR averaging |
| Speed | Slower | ~2x faster |
| Performance | Good | Equal or better |

Reference: [EricSteinberger/Deep-CFR](https://github.com/EricSteinberger/Deep-CFR)

---

## Architecture

### Current (Tabular CFR)
```
Game State → Hand Bucket (169) → History Key (27) → Position (2)
                                        ↓
                            Info Set Index (0-50K)
                                        ↓
                            Lookup Table [50K × 9]
                                        ↓
                                   Strategy
```

### Proposed (Single Deep CFR)
```
Game State → Full Encoding (443 dims)
                    ↓
            Advantage Network
            [443 → 512 → 256 → 128 → 9]
                    ↓
            Advantage Values (per action)
                    ↓
            Regret Matching → Strategy
```

---

## Components

### 1. Advantage Network (`deep_cfr/network.py`)

```python
class AdvantageNetwork:
    """Predicts advantage values for each action given game state."""

    # Architecture (same as existing ActorCriticMLP backbone)
    hidden_dims: tuple = (512, 256, 128)
    input_dim: int = 443  # OBS_DIM
    output_dim: int = 9   # NUM_ACTIONS

    def __call__(self, obs: Array) -> Array:
        """
        Args:
            obs: [batch, 443] encoded game state
        Returns:
            advantages: [batch, 9] advantage estimate per action
        """
```

**Key differences from PPO network:**
- No value head (advantages are relative, not absolute)
- No bet_fraction output (discrete actions only)
- Output represents counterfactual advantages, not logits

### 2. Advantage Memory (`deep_cfr/memory.py`)

```python
@dataclass
class AdvantageMemory:
    """Reservoir sampling buffer for advantage training."""

    max_size: int = 2_000_000  # ~2M samples

    # Storage arrays
    observations: Array    # [max_size, 443]
    advantages: Array      # [max_size, 9]
    iterations: Array      # [max_size] for linear averaging

    def add(self, obs, advantages, iteration):
        """Add samples with reservoir sampling."""

    def sample(self, batch_size: int) -> Tuple[Array, Array, Array]:
        """Sample batch for training."""
```

**Reservoir Sampling**: Maintains uniform distribution over all iterations without storing everything.

### 3. CFR Traversal (`deep_cfr/traverse.py`)

```python
def traverse_game(
    state: GameState,
    player: int,
    network: AdvantageNetwork,
    params: PyTree,
    rng_key: Array,
) -> Tuple[float, List[AdvantageEntry]]:
    """
    External sampling MCCFR traversal.

    For traverser:
        - Explore all actions
        - Compute counterfactual values
        - Store (obs, advantages) in memory

    For opponent:
        - Sample single action from current strategy
        - Continue traversal

    Returns:
        utility: Expected utility for traverser
        entries: Advantage memory entries to store
    """
```

**Vectorized Implementation:**
- Run N=4096 parallel traversals
- Each traversal samples opponent actions
- Collect advantages for all traverser decision points

### 4. Training Loop (`deep_cfr/trainer.py`)

```python
def train_single_deep_cfr(
    iterations: int = 10_000,
    traversals_per_iter: int = 4096,
    train_steps_per_iter: int = 1000,
    batch_size: int = 2048,
    learning_rate: float = 1e-4,
):
    """
    Main SD-CFR training loop.

    For each iteration t:
        1. Traverse games, collecting (obs, advantage) samples
        2. Add samples to reservoir memory with weight t
        3. Train network on sampled batches
        4. Strategy = regret_match(network(obs))

    Linear CFR averaging:
        - Weight iteration t samples by t
        - Network learns weighted average strategy
    """
```

### 5. Strategy Computation (`deep_cfr/strategy.py`)

```python
@jax.jit
def get_strategy(
    params: PyTree,
    network: AdvantageNetwork,
    obs: Array,
    valid_mask: Array,
) -> Array:
    """
    Compute strategy from advantage network.

    1. advantages = network(obs)
    2. positive_adv = max(advantages, 0)
    3. strategy = positive_adv / sum(positive_adv)
    4. Apply valid action mask
    5. Fallback to uniform if all zero
    """
```

---

## Algorithm Details

### External Sampling MCCFR

For each traversal:

```
function traverse(state, player):
    if terminal:
        return utility(player)

    if state.current_player == player:
        # Traverser: explore all actions
        obs = encode(state)
        values = {}
        for action in valid_actions:
            next_state = step(state, action)
            values[action] = traverse(next_state, player)

        # Compute advantages
        ev = sum(strategy[a] * values[a] for a in valid_actions)
        advantages = {a: values[a] - ev for a in valid_actions}

        # Store in memory
        memory.add(obs, advantages, iteration)

        return ev
    else:
        # Opponent: sample single action
        action = sample(strategy)
        return traverse(step(state, action), player)
```

### Linear CFR Averaging

Instead of maintaining separate cumulative strategy:
- Weight samples by iteration number
- Network trained on weighted samples learns average strategy
- At iteration T, sample weight = t for sample from iteration t

```python
# During training
loss = weighted_mse(
    predicted_advantages,
    target_advantages,
    weights=iterations  # Linear weighting
)
```

### Regret Matching+

```python
def regret_match_plus(advantages: Array) -> Array:
    """CFR+ style regret matching with clipping."""
    positive = jnp.maximum(advantages, 0)
    total = positive.sum(axis=-1, keepdims=True)
    strategy = jnp.where(total > 0, positive / total, 1/9)
    return strategy
```

---

## Implementation Steps

### Phase 1: Core Infrastructure (2-3 hours)

1. **Create `deep_cfr/` module structure**
   ```
   deep_cfr/
   ├── __init__.py
   ├── network.py      # AdvantageNetwork
   ├── memory.py       # ReservoirMemory
   ├── traverse.py     # MCCFR traversal
   ├── trainer.py      # Training loop
   └── strategy.py     # Strategy computation
   ```

2. **Implement AdvantageNetwork**
   - Reuse existing MLP architecture from `poker_jax/network.py`
   - Remove value head and bet_fraction
   - Output: 9 advantage values

3. **Implement ReservoirMemory**
   - Fixed-size JAX arrays
   - Reservoir sampling for uniform iteration coverage
   - Efficient batch sampling

### Phase 2: Traversal & Advantage Computation (2-3 hours)

4. **Implement vectorized traversal**
   - Adapt existing `cfr_trainer.py` trajectory simulation
   - Compute counterfactual values at each decision point
   - Collect (obs, advantages, iteration) tuples

5. **Advantage computation**
   ```python
   # At each decision point
   action_values = {a: traverse(step(state, a)) for a in valid}
   ev = sum(strategy[a] * action_values[a] for a in valid)
   advantages = {a: action_values[a] - ev for a in valid}
   ```

### Phase 3: Training Loop (1-2 hours)

6. **Network training**
   - Huber loss (robust to outliers)
   - Adam optimizer with weight decay
   - Iteration-weighted samples

7. **Main training loop**
   ```python
   for iteration in range(num_iterations):
       # Traverse and collect samples
       samples = parallel_traverse(network, iteration)
       memory.add_batch(samples)

       # Train network
       for _ in range(train_steps):
           batch = memory.sample(batch_size)
           params = update(params, batch)
   ```

### Phase 4: Evaluation & Integration (1-2 hours)

8. **Create SD-CFR opponent**
   - Similar to existing `cfr_opponent.py`
   - Load trained network
   - Compute strategy via regret matching

9. **Evaluation**
   - Test against same opponents as tabular CFR
   - Compare convergence metrics
   - Measure exploitability if possible

---

## Key Hyperparameters

| Parameter | Value | Notes |
|-----------|-------|-------|
| `iterations` | 10,000 | CFR iterations |
| `traversals_per_iter` | 4,096 | Parallel games |
| `memory_size` | 2,000,000 | Reservoir capacity |
| `batch_size` | 2,048 | Training batch |
| `learning_rate` | 1e-4 | Adam LR |
| `weight_decay` | 1e-5 | L2 regularization |
| `train_steps_per_iter` | 100 | Network updates per CFR iter |
| `hidden_dims` | (512, 256, 128) | Network architecture |

---

## Expected Improvements over Tabular CFR

| Metric | Tabular V3 | Expected SD-CFR |
|--------|------------|-----------------|
| Info sets | 50,000 (fixed) | Continuous (neural) |
| Generalization | None | Cross-situation |
| Memory | O(info_sets × actions) | O(network_params) |
| Non-uniform strategies | 3.9% | ~100% |
| Training time | 36 min | ~2-4 hours |

---

## File Changes Summary

### New Files
- `deep_cfr/__init__.py`
- `deep_cfr/network.py` - AdvantageNetwork definition
- `deep_cfr/memory.py` - ReservoirMemory buffer
- `deep_cfr/traverse.py` - MCCFR traversal logic
- `deep_cfr/trainer.py` - Main training loop
- `deep_cfr/strategy.py` - Strategy computation utilities
- `deep_cfr/opponent.py` - SD-CFR opponent for evaluation

### Modified Files
- `evaluation/opponents.py` - Register SD-CFR opponent
- `docs/experiment-log.md` - Log results

### Unchanged
- `poker_jax/*` - Reuse existing game logic
- `cfr/*` - Keep for comparison

---

## Risk Mitigation

1. **Slow convergence**: Start with smaller network, scale up if needed
2. **Memory overflow**: Reservoir sampling keeps memory bounded
3. **Unstable training**: Use Huber loss, gradient clipping, warmup
4. **Poor performance**: Fallback to tabular CFR for comparison

---

## Success Criteria

1. **Convergence**: Non-uniform strategies > 50% (vs 3.9% tabular)
2. **Performance**: Positive BB/100 against TAG, rock, trapper
3. **Generalization**: Better than tabular on unseen situations
4. **Speed**: < 4 hours training time for competitive results

---

## References

- [Deep Counterfactual Regret Minimization](https://arxiv.org/abs/1811.00164) - Original Deep CFR paper
- [Single Deep CFR](https://github.com/EricSteinberger/Deep-CFR) - SD-CFR implementation
- [PokerRL Framework](https://github.com/EricSteinberger/PokerRL) - Reference implementation
- [DREAM](https://github.com/EricSteinberger/DREAM) - State-of-art extension
