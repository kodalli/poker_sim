# v2: Reinforcement Learning Architecture

## Problem with v1 (Genetic Algorithm)

The current GA approach has fundamental limitations:

1. **No credit assignment** - Fitness is computed per-game, not per-decision. The network can't learn "that bluff was good" vs "that call was bad."

2. **Sample inefficiency** - Evolution requires many generations to discover useful weight configurations. Each generation requires thousands of games.

3. **No gradient signal** - Unlike policy gradient methods, the network never learns from the reward structure of individual poker decisions.

4. **Sparse feedback** - Only end-of-game chip counts matter, ignoring the quality of individual plays.

## Proposed Architecture: PPO (Proximal Policy Optimization)

### Why PPO?

- Stable training with clipped objective (no catastrophic updates)
- Works well with discrete action spaces
- Handles partial observability (we don't see opponent cards)
- Battle-tested in game AI (OpenAI Five, AlphaStar used PPO variants)

### Network Changes

```
Current (v1):
  Input (433) -> MLP -> action_logits (5), bet_fraction (1)

Proposed (v2):
  Input (433) -> Shared MLP ->
    ├── Policy head  -> action_logits (5), bet_fraction (1)
    └── Value head   -> state_value (1)
```

The value head estimates expected future rewards from current state, enabling:
- Advantage estimation (was this action better than average?)
- Variance reduction in policy gradients

### Reward Structure

| Event | Reward | Rationale |
|-------|--------|-----------|
| Win pot | +pot_size | Direct reward for winning |
| Lose pot | -amount_invested | Penalize losses proportionally |
| Fold (would have lost) | +small_bonus | Reward good folds |
| Fold (would have won) | -pot_size * 0.5 | Penalize missed value |
| Showdown (best hand) | +pot_size | Standard win |
| Bluff success | +pot_size | Reward successful bluffs |

**Per-decision rewards** (shaped):
- Calling with bad odds: small negative
- Raising with strong hand: small positive
- Position-aware play: small bonus

### Training Loop

```python
for episode in training:
    # Collect experience
    states, actions, rewards, values = play_hand(agent)

    # Compute advantages using GAE (Generalized Advantage Estimation)
    advantages = compute_gae(rewards, values, gamma=0.99, lambda_=0.95)

    # PPO update
    for epoch in range(ppo_epochs):
        # Policy loss (clipped)
        ratio = new_prob / old_prob
        clipped = clip(ratio, 1-epsilon, 1+epsilon) * advantages
        policy_loss = -min(ratio * advantages, clipped)

        # Value loss
        value_loss = (value_pred - returns)**2

        # Entropy bonus (encourage exploration)
        entropy_loss = -entropy_coef * entropy(action_probs)

        loss = policy_loss + value_coef * value_loss + entropy_loss
        optimizer.step(loss)
```

### Self-Play Training

1. **Population of agents** - Maintain pool of past versions
2. **Opponent sampling** - Train against mix of:
   - Current self (50%)
   - Past versions (30%)
   - Baseline agents (20%)
3. **ELO tracking** - Monitor relative skill progression

### Implementation Plan

#### Phase 1: Core RL Infrastructure
- [ ] Add value head to network (`network.py`)
- [ ] Create `RLAgent` class with action logging (`agents/rl/agent.py`)
- [ ] Implement experience buffer (`training/buffer.py`)
- [ ] Define reward function (`training/rewards.py`)

#### Phase 2: PPO Training
- [ ] Implement GAE computation (`training/gae.py`)
- [ ] PPO loss function (`training/ppo.py`)
- [ ] Training loop with batching (`training/trainer.py`)
- [ ] Checkpointing and logging

#### Phase 3: Self-Play
- [ ] Agent pool management
- [ ] Opponent sampling strategy
- [ ] ELO rating system
- [ ] Curriculum learning (start simple, increase complexity)

#### Phase 4: Evaluation
- [ ] Compare v2 vs v1 on same evaluation set
- [ ] Measure sample efficiency (games to reach skill level)
- [ ] Analyze learned behaviors (aggression, position play, bluffing)

### Files to Create/Modify

```
poker_sim/
├── agents/
│   └── rl/
│       ├── __init__.py
│       ├── agent.py          # RLAgent with experience logging
│       └── policy.py         # Policy wrapper for action selection
├── training/
│   ├── __init__.py
│   ├── buffer.py             # Experience replay buffer
│   ├── rewards.py            # Reward shaping functions
│   ├── gae.py                # Advantage estimation
│   ├── ppo.py                # PPO algorithm
│   └── trainer.py            # Training orchestration
├── agents/neural/
│   └── network.py            # Add value head (modify existing)
└── main.py                   # Add `train-rl` command
```

### Hyperparameters (Starting Point)

```python
# PPO
learning_rate = 3e-4
gamma = 0.99           # Discount factor
lambda_ = 0.95         # GAE parameter
epsilon = 0.2          # Clipping parameter
ppo_epochs = 4         # Updates per batch
batch_size = 64
entropy_coef = 0.01    # Exploration bonus
value_coef = 0.5       # Value loss weight

# Training
total_timesteps = 1_000_000
eval_frequency = 10_000
checkpoint_frequency = 50_000
```

### Success Metrics

| Metric | v1 Baseline | v2 Target |
|--------|-------------|-----------|
| Win rate vs Random | ~55% | >70% |
| Win rate vs TAG | ~45% | >55% |
| Training games needed | 500K+ | <100K |
| Learned behaviors | Random-ish | Position-aware, pot odds |
