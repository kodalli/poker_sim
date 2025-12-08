# Poker AI Evaluation System

How to measure if the poker AI is actually getting better at poker.

## The Problem

Training metrics (loss, reward, entropy) tell us the model is learning *something*, but not whether it's learning *good poker*. A model could:
- Minimize loss by memorizing patterns that don't generalize
- Achieve positive rewards by exploiting a flaw in the training setup
- Converge to a suboptimal local minimum

We need objective evaluation against external benchmarks.

---

## Evaluation Methods

### 1. Checkpoint Elo Rating

**How it works:**
1. Save model checkpoints periodically during training (e.g., every 500M steps)
2. After training, run tournaments between all checkpoints
3. Compute Elo ratings from win/loss records

**Elo calculation:**
```python
# Expected score for player A against player B
E_a = 1 / (1 + 10^((R_b - R_a) / 400))

# Updated rating after game
R_a_new = R_a + K * (S_a - E_a)
# where S_a = 1 (win), 0.5 (draw), 0 (loss)
# K = 32 (typical for new players)
```

**Implementation:**
```python
def evaluate_checkpoints(checkpoint_paths: list[Path], games_per_match: int = 1000):
    """Run round-robin tournament between checkpoints."""
    elo_ratings = {path: 1500 for path in checkpoint_paths}  # Start at 1500

    for i, ckpt_a in enumerate(checkpoint_paths):
        for ckpt_b in checkpoint_paths[i+1:]:
            wins_a, wins_b, draws = play_match(ckpt_a, ckpt_b, games_per_match)
            # Update Elo ratings
            update_elo(elo_ratings, ckpt_a, ckpt_b, wins_a, wins_b)

    return elo_ratings
```

**Expected result:** Elo should increase with training steps if the model is improving.

---

### 2. Baseline Opponent Strategies

Test against known strategies with predictable behavior:

| Opponent | Strategy | Expected Result |
|----------|----------|-----------------|
| **RandomAgent** | Picks random valid action | Should crush (>80% win rate) |
| **CallingStation** | Never folds, always calls | Should beat by value betting |
| **TightPassive** | Only plays premium hands, rarely raises | Should steal blinds, bluff |
| **LooseAggressive** | Plays many hands, raises frequently | Tests our defense |
| **OptimalPreflop** | Uses GTO preflop ranges | Tests postflop skill |

**Implementation:**
```python
class CallingStation:
    """Never folds, always calls or checks."""
    def act(self, state):
        if can_check(state):
            return CHECK
        else:
            return CALL

class TightPassive:
    """Only plays top 15% of hands, rarely raises."""
    def act(self, state):
        hand_strength = evaluate_preflop(state.hole_cards)
        if hand_strength < 0.85:
            return FOLD
        if state.to_call == 0:
            return CHECK
        return CALL

class LooseAggressive:
    """Plays 60% of hands, raises 40% of the time."""
    def act(self, state):
        if random() < 0.4:
            return FOLD
        if random() < 0.4:
            return RAISE
        return CALL
```

---

### 3. Win Rate Metrics

**bb/100 (Big Blinds per 100 hands)**

The standard poker metric for measuring skill:
- Professional winning players: +5 to +15 bb/100
- Break-even: 0 bb/100
- Losing players: negative bb/100

```python
def calculate_bb_per_100(total_chips_won: float, hands_played: int, big_blind: int) -> float:
    """Calculate big blinds won per 100 hands."""
    return (total_chips_won / big_blind) / (hands_played / 100)
```

**Variance considerations:**
- Poker has high variance
- Need 10,000+ hands for statistically significant results
- Use confidence intervals:
```python
# Standard deviation for heads-up NLHE is ~100 bb/100
std_dev = 100
hands = 10000
std_error = std_dev / sqrt(hands / 100)  # ~10 bb/100

# 95% confidence interval
ci_95 = 1.96 * std_error  # ~20 bb/100
```

So with 10k hands and measured +15 bb/100, true win rate is likely between -5 and +35 bb/100.

---

### 4. Exploitability (Advanced)

**What is exploitability?**

The maximum amount an optimal counter-strategy can win against our strategy.

- **GTO (Nash Equilibrium)**: Exploitability = 0 (unexploitable)
- **Exploitable strategy**: Positive exploitability (has leaks)

**Calculation (for small games):**

For full No-Limit Hold'em, computing exact exploitability is intractable. But for simplified games:

```python
def compute_exploitability(strategy):
    """
    For each decision point:
    1. Compute best response to our strategy
    2. Calculate EV of best response
    3. Exploitability = EV of best response - EV of Nash equilibrium
    """
    best_response = compute_best_response(strategy)
    exploitability = expected_value(best_response, strategy)
    return exploitability
```

**Practical alternative: Local exploitability**

Measure exploitability in specific situations:
- How much do we lose to a perfect bluff-catcher?
- How much do we lose to someone who never folds to our bluffs?

---

### 5. Decision Quality Analysis

Analyze specific decisions the model makes:

```python
def analyze_decisions(model, test_hands: list[Hand]):
    """Evaluate decision quality on curated test hands."""
    results = []
    for hand in test_hands:
        model_action = model.predict(hand.state)
        expected_action = hand.correct_action  # From poker theory
        ev_loss = hand.action_evs[expected_action] - hand.action_evs[model_action]
        results.append({
            'hand': hand,
            'model_action': model_action,
            'expected_action': expected_action,
            'ev_loss': ev_loss,
        })
    return results
```

**Test hand categories:**
1. Clear value bets (should bet)
2. Clear bluffs (should bluff with right frequency)
3. Clear folds (should fold weak hands to aggression)
4. Marginal spots (tests nuanced understanding)

---

### 6. Sample Size & Statistical Significance

Poker has extremely high variance. Understanding sample sizes is critical for meaningful evaluation.

**The Core Problem**

Win rate standard deviation in heads-up NLHE: ~100 bb/100 hands

This means a player winning +10 bb/100 could easily appear to be losing -20 bb/100 over a small sample due to variance.

---

#### For Win Rate / Elo Measurements

Win rate follows a binomial distribution. Standard error formula:

```python
SE_win_rate = sqrt(p * (1-p) / n)
# where p = true win rate, n = number of games
```

**For 50% win rate (evenly matched):**

| Games | Standard Error | 95% CI | Can Detect |
|-------|---------------|--------|------------|
| 100 | ±5.0% | ±10% | 60%+ difference |
| 1,000 | ±1.6% | ±3.1% | 55%+ difference |
| 10,000 | ±0.5% | ±1.0% | 52%+ difference |
| 100,000 | ±0.16% | ±0.3% | 50.5%+ difference |

**Recommendation: 10,000 games per matchup for Elo**
- Standard error: ±0.5%
- Can reliably distinguish 52% vs 48% win rates
- Takes ~30 seconds at 300k steps/sec

---

#### For bb/100 Measurements

bb/100 has much higher variance. Standard error formula:

```python
# Standard deviation in heads-up NLHE: ~100 bb/100
std_dev = 100  # bb/100
SE_bb100 = std_dev / sqrt(hands / 100)
```

**Sample size requirements:**

| Hands | Standard Error | 95% CI | Can Detect |
|-------|---------------|--------|------------|
| 1,000 | ±31.6 bb/100 | ±62 bb/100 | Huge differences only |
| 10,000 | ±10 bb/100 | ±20 bb/100 | 20+ bb/100 difference |
| 100,000 | ±3.2 bb/100 | ±6.3 bb/100 | 10+ bb/100 difference |
| 1,000,000 | ±1.0 bb/100 | ±2 bb/100 | 5+ bb/100 difference |

**Recommendation: 100,000+ hands for bb/100**
- With 100k hands, measured +15 bb/100 has true value between +8.7 and +21.3 (95% CI)
- Takes ~5 minutes at 300k steps/sec

---

#### Variance Reduction: Duplicate Poker

A powerful technique to reduce variance by ~10x:

```python
def play_duplicate_match(model_a, model_b, num_hands: int):
    """
    Play each hand twice with cards swapped.

    Game 1: A gets cards X, B gets cards Y
    Game 2: A gets cards Y, B gets cards X

    This controls for card luck - the better player should win
    both or split evenly.
    """
    total_diff = 0
    for _ in range(num_hands // 2):
        cards = deal_cards()

        # Game 1: A=cards[0], B=cards[1]
        result1 = play_hand(model_a, model_b, cards[0], cards[1])

        # Game 2: A=cards[1], B=cards[0] (swapped)
        result2 = play_hand(model_a, model_b, cards[1], cards[0])

        # Net result removes card luck
        total_diff += result1 + result2

    return total_diff
```

**Duplicate poker benefits:**
- Reduces standard deviation from ~100 bb/100 to ~30 bb/100
- 10k duplicate hands ≈ 100k regular hands in precision
- Used by professional poker AI researchers (Libratus, Pluribus)

---

#### Practical Recommendations

| Evaluation Type | Minimum Sample | Recommended | Purpose |
|-----------------|---------------|-------------|---------|
| Quick sanity check | 1,000 games | 1,000 | Verify model isn't broken |
| Elo rating | 10,000 games | 10,000 | Checkpoint comparison |
| bb/100 (rough) | 10,000 hands | 50,000 | Baseline opponent testing |
| bb/100 (precise) | 100,000 hands | 500,000 | Publication-quality results |
| Statistical proof | 1,000,000 hands | 1,000,000+ | Proving small edges |

**Is 1M games overkill?**
- For Elo: Yes, 10k is sufficient
- For bb/100: Gives ±1 bb/100 precision, useful for detecting small edges
- For duplicate poker: 100k duplicate hands (200k total) achieves similar precision

---

## Implementation Plan

### Phase 1: Checkpoint Evaluation (Easy)

1. **Modify training to save more checkpoints**
   - Save every 100M steps during 8hr run
   - Results in ~100 checkpoints

2. **Create evaluation script**
   ```python
   # scripts/evaluate_checkpoints.py
   def main():
       checkpoints = load_checkpoints("models/v1-8hr-4090/checkpoints/")
       elo_ratings = run_tournament(checkpoints, games_per_match=1000)
       plot_elo_progression(elo_ratings)
   ```

3. **Visualize Elo progression**
   - X-axis: Training steps
   - Y-axis: Elo rating
   - Should see upward trend

### Phase 2: Baseline Opponents (Medium)

1. **Implement baseline agents**
   - `agents/baselines/random_agent.py`
   - `agents/baselines/calling_station.py`
   - `agents/baselines/tight_passive.py`
   - `agents/baselines/loose_aggressive.py`

2. **Create evaluation harness**
   ```python
   # scripts/evaluate_vs_baselines.py
   def evaluate_model(model_path: str, num_hands: int = 10000):
       model = load_model(model_path)
       baselines = [RandomAgent(), CallingStation(), TightPassive(), LooseAggressive()]

       for opponent in baselines:
           wins, losses, total_chips = play_heads_up(model, opponent, num_hands)
           bb_per_100 = calculate_bb_per_100(total_chips, num_hands, BIG_BLIND)
           print(f"vs {opponent.name}: {bb_per_100:+.1f} bb/100")
   ```

3. **Expected results table**
   | Opponent | Untrained | After 1B steps | After 10B steps |
   |----------|-----------|----------------|-----------------|
   | Random | ~0 bb/100 | +50 bb/100 | +100 bb/100 |
   | CallingStation | ? | ? | +30 bb/100 |
   | TightPassive | ? | ? | +20 bb/100 |
   | LooseAggressive | ? | ? | +10 bb/100 |

### Phase 3: Advanced Metrics (Hard)

1. **Decision analysis on curated hands**
2. **Approximate exploitability in key spots**
3. **Compare to known GTO solutions (for simplified games)**

---

## How AlphaZero Validated Improvement

For reference, AlphaZero's evaluation approach:

1. **Internal Elo**: Tracked Elo from self-play matches between versions
2. **External validation**: Played against Stockfish (world's best chess engine)
3. **Match conditions**: Played thousands of games to reduce variance
4. **Time controls**: Used tournament time controls for fair comparison

Key insight: AlphaZero's Elo kept increasing during training, and this correlated with beating stronger external opponents.

---

## Recommended Evaluation Pipeline

```bash
# After training completes:

# 1. Run checkpoint tournament
python scripts/evaluate_checkpoints.py --model v1-8hr-4090 --games 1000

# 2. Evaluate against baselines
python scripts/evaluate_vs_baselines.py --model v1-8hr-4090 --hands 10000

# 3. Generate report
python scripts/generate_eval_report.py --model v1-8hr-4090
```

**Output:**
```
=== Evaluation Report: v1-8hr-4090 ===

Checkpoint Elo Progression:
  Step 500M:  1500 (baseline)
  Step 1B:    1623 (+123)
  Step 5B:    1891 (+391)
  Step 10B:   2047 (+547)

Baseline Performance (bb/100):
  vs Random:         +87.3 bb/100
  vs CallingStation: +24.1 bb/100
  vs TightPassive:   +18.7 bb/100
  vs LooseAggressive: +8.2 bb/100

Conclusion: Model shows consistent improvement and beats all baselines.
```

---

## Files to Create

| File | Purpose |
|------|---------|
| `scripts/evaluate_checkpoints.py` | Checkpoint tournament runner |
| `scripts/evaluate_vs_baselines.py` | Baseline opponent evaluation |
| `scripts/generate_eval_report.py` | Report generator |
| `agents/baselines/random_agent.py` | Random action baseline |
| `agents/baselines/calling_station.py` | Never-fold baseline |
| `agents/baselines/tight_passive.py` | Tight passive baseline |
| `agents/baselines/loose_aggressive.py` | LAG baseline |
| `training/elo.py` | Elo calculation utilities |
