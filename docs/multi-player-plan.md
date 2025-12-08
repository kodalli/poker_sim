# Plan: Multi-Player Support for JAX Poker (2-10 Players)

Add support for 2-10 players in the JAX-accelerated poker training, starting with a 3-player MVP.

## Scope

- **Goal**: Extend JAX poker from 2-player heads-up to support 2-10 players
- **Approach**: Start with 3-player MVP, then generalize to N players
- **Policy**: Position-conditioned single network (learns position-specific strategies)
- **Side pots**: Full support for accurate poker rules

---

## Files to Modify

| File | Changes |
|------|---------|
| `poker_jax/state.py` | Add `num_players` field, change arrays from `[N,2]` to `[N,P]` |
| `poker_jax/game.py` | N-player betting, rotation, side pots, showdown |
| `poker_jax/deck.py` | Deal `2*num_players` hole cards |
| `poker_jax/hands.py` | Evaluate N players, multi-way winner determination |
| `poker_jax/encoding.py` | Dynamic OBS_DIM, position encoding, N opponent info |
| `poker_jax/network.py` | Accept dynamic input dimension |
| `training/jax_trainer.py` | Per-player trajectory collection, metrics |
| `training/jax_ppo.py` | Handle variable player trajectories |

---

## Phase 1: Core State Structure

### 1.1 Update GameState (state.py)

```python
@dataclass
class GameState:
    # Add num_players field
    num_players: Array  # [N] int - players per game (2-10)

    # Change all player arrays from [N, 2] to [N, MAX_PLAYERS]
    # Use MAX_PLAYERS=10 with masking for unused slots
    hole_cards: Array      # [N, 10, 2] (games, max_players, cards)
    chips: Array           # [N, 10]
    bets: Array            # [N, 10]
    total_invested: Array  # [N, 10]
    folded: Array          # [N, 10]
    all_in: Array          # [N, 10]

    # Position tracking
    button: Array          # [N] - button position (0 to num_players-1)
    current_player: Array  # [N] - who acts next

    # Side pot tracking (new)
    side_pots: Array       # [N, 10] - side pot amounts per level
    side_pot_eligible: Array  # [N, 10, 10] - which players eligible for each pot
```

### 1.2 Update create_initial_state()

- Accept `num_players` parameter
- Initialize arrays with proper masking
- Set up blinds based on position (SB = button+1, BB = button+2 for 3+ players)

### 1.3 Update get_valid_actions_mask()

- Replace `opp_idx = 1 - player_idx` with `max_bet = jnp.max(bets, axis=1)`
- Call amount = max_bet - my_bet
- Works for any N players

---

## Phase 2: Game Logic (game.py)

### 2.1 Player Rotation

Replace all `1 - player_idx` with proper rotation:

```python
def get_next_active_player(current: int, folded: Array, all_in: Array, num_players: int) -> int:
    """Find next player who can act (not folded, not all-in)."""
    for offset in range(1, num_players):
        next_player = (current + offset) % num_players
        if not folded[next_player] and not all_in[next_player]:
            return next_player
    return -1  # No one can act (all folded or all-in)
```

### 2.2 Round Completion Check

Current (2-player): `bets[0] == bets[1]`

N-player:
```python
active_mask = ~folded & ~all_in  # Players who can still act
all_acted = jnp.all(actions_this_round >= 1, where=active_mask)
max_bet = jnp.max(bets)
bets_equal = jnp.all(bets == max_bet, where=active_mask)
round_complete = all_acted & bets_equal
```

### 2.3 Side Pot Calculation

```python
def calculate_side_pots(bets: Array, folded: Array, all_in: Array) -> tuple[Array, Array]:
    """
    Calculate side pots for all-in situations.

    Returns:
        side_pots: [num_pots] pot amounts
        eligible: [num_pots, num_players] who can win each pot
    """
    # Sort players by total invested
    # Create pot for each all-in level
    # Track eligibility for each pot
```

### 2.4 Fold Winner Detection

```python
# Count non-folded players
active_count = jnp.sum(~folded)
one_remaining = active_count == 1
winner = jnp.argmax(~folded)  # First non-folded player
```

### 2.5 Showdown with Side Pots

```python
def resolve_showdown(state: GameState) -> Array:
    """
    Resolve showdown with side pots.

    Returns:
        rewards: [N, num_players] chips won/lost per player
    """
    hand_values = evaluate_hands_batch(state.hole_cards, state.community)

    rewards = jnp.zeros((N, num_players))
    for pot_idx, (pot_amount, eligible) in enumerate(side_pots):
        # Find best hand among eligible players
        eligible_values = jnp.where(eligible, hand_values, -1)
        winner = jnp.argmax(eligible_values)
        # Handle ties: split pot
        rewards[:, winner] += pot_amount

    # Subtract total invested
    rewards -= state.total_invested
    return rewards
```

### 2.6 Blinds Setup (3+ players)

```python
# For 3+ players:
sb_pos = (button + 1) % num_players
bb_pos = (button + 2) % num_players
first_to_act_preflop = (button + 3) % num_players  # UTG

# For 2 players (heads-up special case):
sb_pos = button  # Button posts SB
bb_pos = 1 - button  # Other player posts BB
first_to_act_preflop = button  # SB acts first
```

---

## Phase 3: Card Dealing (deck.py)

### 3.1 Update deal_hole_cards_batch()

```python
def deal_hole_cards_batch(decks, deck_indices, num_players):
    """Deal 2 cards to each of num_players players."""
    cards_needed = 2 * num_players

    def deal_one_game(deck, idx):
        cards, new_idx = deal_cards_n(deck, idx, cards_needed)
        # Reshape: [2*P] -> [P, 2]
        hole = cards.reshape(num_players, 2)
        return hole, new_idx

    return jax.vmap(deal_one_game)(decks, deck_indices)
```

---

## Phase 4: Hand Evaluation (hands.py)

### 4.1 Update evaluate_hands_batch()

```python
def evaluate_hands_batch(hole_cards: Array, community: Array) -> Array:
    """
    Args:
        hole_cards: [N, num_players, 2]
        community: [N, 5]
    Returns:
        [N, num_players] hand values
    """
    def eval_one_game(hole, comm):
        # hole: [num_players, 2], comm: [5]
        def eval_player(h):
            return evaluate_hand(jnp.concatenate([h, comm]))
        return jax.vmap(eval_player)(hole)

    return jax.vmap(eval_one_game)(hole_cards, community)
```

### 4.2 Update determine_winner()

```python
def determine_winner(hand_values: Array, folded: Array) -> tuple[Array, Array]:
    """
    Args:
        hand_values: [N, num_players]
        folded: [N, num_players]
    Returns:
        winner: [N] winning player index
        is_tie: [N, num_players] bool mask of tied winners
    """
    # Mask folded players
    active_values = jnp.where(~folded, hand_values, -1)
    max_value = jnp.max(active_values, axis=1, keepdims=True)
    is_tie = active_values == max_value
    winner = jnp.argmax(active_values, axis=1)
    return winner, is_tie
```

---

## Phase 5: Encoding (encoding.py)

### 5.1 Dynamic OBS_DIM

```python
def compute_obs_dim(num_players: int) -> int:
    """Compute observation dimension for given player count."""
    return (
        52 * 2 +              # Hole cards (2 cards one-hot)
        52 * 5 +              # Community cards
        4 +                   # Round (preflop/flop/turn/river)
        num_players +         # Position one-hot
        1 +                   # Pot (normalized)
        1 +                   # My chips
        (num_players - 1) +   # Other players' chips
        1 +                   # My bet
        (num_players - 1) +   # Other players' bets
        (num_players - 1) +   # Other players folded/active
        1 +                   # To call amount
        5                     # Valid actions
    )
    # = 375 + 4*(num_players - 1)
    # 3 players: 383
    # 6 players: 395
    # 10 players: 411
```

### 5.2 Update encode_state()

```python
def encode_state(state: GameState, player_id: int) -> Array:
    """Encode state from player_id's perspective."""
    num_players = state.num_players

    # Position: relative to button (0 = button, 1 = SB, 2 = BB, etc.)
    relative_pos = (player_id - state.button) % num_players
    position_one_hot = jax.nn.one_hot(relative_pos, num_players)

    # My info
    my_chips = state.chips[:, player_id] / CHIP_NORM
    my_bet = state.bets[:, player_id] / CHIP_NORM

    # Other players' info (in relative order from my left)
    other_indices = [(player_id + i) % num_players for i in range(1, num_players)]
    other_chips = state.chips[:, other_indices] / CHIP_NORM
    other_bets = state.bets[:, other_indices] / CHIP_NORM
    other_active = (~state.folded[:, other_indices]).astype(jnp.float32)

    # To call
    max_bet = jnp.max(state.bets, axis=1)
    to_call = jnp.maximum(max_bet - state.bets[:, player_id], 0) / CHIP_NORM

    # Concatenate all features
    return jnp.concatenate([
        hole_card_encoding,
        community_encoding,
        round_one_hot,
        position_one_hot,
        pot_normalized,
        my_chips, my_bet,
        other_chips, other_bets, other_active,
        to_call,
        valid_actions,
    ], axis=-1)
```

---

## Phase 6: Network (network.py)

### 6.1 Dynamic Input Dimension

```python
def init_network(
    network: nn.Module,
    rng_key: Array,
    num_players: int = 2,
) -> dict:
    obs_dim = compute_obs_dim(num_players)
    dummy_input = jnp.zeros((1, obs_dim))
    return network.init(rng_key, dummy_input)
```

No architecture changes needed - Dense layers handle variable input sizes.

---

## Phase 7: Trainer (jax_trainer.py)

### 7.1 Per-Player Trajectory Collection

```python
def collect_rollout(self, num_steps: int) -> tuple[Trajectory, TrainingMetrics]:
    """Collect experience from all players' perspectives."""

    # Storage per player
    player_trajectories = {p: [] for p in range(num_players)}

    for step_idx in range(num_steps):
        current_player = state.current_player

        # Encode state for current player
        obs = encode_state(state, current_player)

        # Sample action
        action, log_prob, value = self.network.apply(...)

        # Step environment
        new_state = step(state, action)

        # Store experience for this player
        player_trajectories[current_player].append({
            'obs': obs,
            'action': action,
            'log_prob': log_prob,
            'value': value,
        })

        # On game end, compute rewards for all players
        if new_state.done:
            rewards = get_rewards(new_state)
            for p in range(num_players):
                player_trajectories[p][-1]['reward'] = rewards[p]

    # Merge all player trajectories for PPO update
    return merge_trajectories(player_trajectories)
```

### 7.2 Updated Metrics

```python
# Track per-position statistics
position_wins = {pos: 0 for pos in range(num_players)}
position_games = {pos: 0 for pos in range(num_players)}

# Action distribution per position
position_actions = {pos: defaultdict(int) for pos in range(num_players)}
```

---

## Phase 8: PPO (jax_ppo.py)

### 8.1 Trajectory Structure Update

```python
class Trajectory(NamedTuple):
    obs: Array           # [T, N, obs_dim]
    actions: Array       # [T, N]
    log_probs: Array     # [T, N]
    values: Array        # [T, N]
    rewards: Array       # [T, N]
    dones: Array         # [T, N]
    valid_masks: Array   # [T, N, 5]
    player_ids: Array    # [T, N] - which player acted (for position conditioning)
```

GAE computation remains the same - operates on flattened experiences.

---

## Implementation Order

### MVP: 3-Player Support

1. **state.py**: Add `num_players`, change arrays to `[N, 3]` initially
2. **game.py**: Player rotation, round completion, basic showdown
3. **deck.py**: Deal 6 cards (3 players x 2)
4. **hands.py**: Evaluate 3 players
5. **encoding.py**: OBS_DIM for 3 players, position encoding
6. **network.py**: Accept dynamic input
7. **jax_trainer.py**: Basic 3-player rollout collection
8. **Test**: Run 3-player training, verify convergence

### Generalization: 2-10 Players

1. Change hardcoded `3` to `num_players` parameter
2. Add `MAX_PLAYERS = 10` constant with masking
3. Implement full side pot logic
4. Test with 2, 4, 6, 9, 10 players

---

## Testing Strategy

1. **Unit tests**: Each modified function with N=2,3,6,10
2. **Integration test**: Full game with known hands, verify pot distribution
3. **Side pot test**: All-in scenarios with unequal stacks
4. **Training test**: 50k steps, verify loss decreases and rewards improve

---

## Risk Mitigation

- **Backward compatibility**: Keep existing 2-player code path initially
- **Incremental testing**: Test each phase before proceeding
- **Side pots complexity**: Start with simplified version if needed
