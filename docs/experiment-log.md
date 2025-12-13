# Poker AI Experiment Log

Research log tracking model training experiments, results, and lessons learned.

---

## v10 - 2025-12-13

**Summary**: Mixed opponent training with LSTM opponent model

**Hypothesis**: Training the LSTM opponent model against diverse opponents (not just self-play) would enable the model to learn opponent-type discrimination and adapt its strategy accordingly.

**Configuration**:
- Architecture: ActorCriticMLPWithOpponentModel (same as v9)
  - OpponentLSTM: 64 hidden, 32 embedding dim
- Starting chips: 200 (100BB)
- Training: 100M steps
- **Mixed Opponent Training**: 50% self, 15% random, 15% call_station, 10% TAG, 10% LAG

**Results** (10,000 games each):
| Opponent     | Win%  | BB/100  | vs v9 Δ |
|--------------|-------|---------|---------|
| random       | 59.0% | +1,039  | -277    |
| call_station | 31.5% | -1,062  | **+690**|
| tag          | 82.0% | +382    | -23     |
| lag          | 55.0% | +331    | -243    |
| rock         | 82.7% | -241    | **+107**|
| trapper      | 72.3% | -350    | **+471**|
| value_bettor | 90.1% | -114    | **+189**|

**Aggregate Performance**:
- v9 total: -929 BB/100 (across all opponents)
- v10 total: **-15 BB/100** (nearly break-even!)

**Action Distribution**:
- Adapts per opponent: 12-16% folds, 5-17% checks
- More checking vs passive opponents (rock 14%, trapper 17%)
- Less calling vs call_station (1%) - learned not to chase

**Analysis**:
1. **Massive improvement vs passive/trapping opponents**: call_station +690, trapper +471, value_bettor +189
2. **Slight regression vs aggressive opponents**: random -277, lag -243
3. **Nearly break-even overall** (-15 BB/100) vs v9's -929 BB/100
4. Mixed training successfully taught the LSTM to distinguish opponent types
5. The model learned to reduce losses against opponents who trap/call rather than fold

**Lessons Learned**:
1. **Training diversity is crucial** - exposing LSTM to different playstyles works
2. Trade-off exists: better vs passive opponents, slightly worse vs aggressive
3. The 64-dim LSTM is sufficient for basic opponent-type discrimination
4. Next steps: try curriculum learning (easy opponents first), or larger LSTM for finer adaptation
5. Consider adding rock/trapper/value_bettor to training mix to further improve

---

## v9 - 2025-12-13

**Summary**: LSTM-based opponent modeling for cross-hand memory

**Hypothesis**: Adding an LSTM to encode opponent actions across hands would enable the model to adapt to different opponent playstyles dynamically, improving generalization.

**Configuration**:
- Architecture: ActorCriticMLPWithOpponentModel
  - OpponentLSTM: 64 hidden, 32 embedding dim
  - Opponent action encoding: 13 dims (9 action one-hot + bet_amount + round + pot_odds + position)
  - LSTM output concatenated with observations before policy backbone
- Starting chips: 200 (100BB)
- Training: 100M steps, pure self-play
- Opponent mix: Self-play only (mixed_opponents=False, historical_selfplay=False)

**Results** (10,000 games each):
| Opponent     | Win%  | BB/100  |
|--------------|-------|---------|
| random       | 62.1% | +1,316  |
| call_station | 26.8% | -1,752  |

**Training Checkpoints**:
| Checkpoint | vs Random | vs Call Station |
|------------|-----------|-----------------|
| 76.8M      | 47.4% (+712)  | 17.8% (-1,640) |
| 86.4M      | 60.4% (+1,016) | 25.1% (-2,342) |
| 100M       | 62.1% (+1,316) | 26.8% (-1,752) |

**Action Distribution**:
- vs random: 13% fold, 7% check, 19% call, 27% raise_150, 7% all_in
- vs call_station: 13% fold, 11% check, 2% call, 28% raise_150, 8% all_in

**Analysis**:
1. Strong performance vs random (+1316 BB/100) - best result so far
2. Still struggles vs call_station (-1752 BB/100) - can't exploit someone who never folds
3. Healthy action distribution with reasonable checking (7-11%) unlike v7/v8
4. Training was stable - no instability in policy loss or explained variance
5. **Critical limitation**: LSTM only ever saw self-play during training, so it learned to model "how my clone plays" rather than diverse opponent types

**Lessons Learned**:
1. Opponent modeling architecture works but needs diverse training opponents to be effective
2. Pure self-play limits the LSTM's ability to learn opponent-type discrimination
3. LSTM capacity (64 hidden) may be too small for encoding multiple distinct playstyles
4. Next experiment should enable `--mixed-opponents` to expose LSTM to call_station, TAG, LAG, etc.
5. Consider increasing LSTM hidden from 64 → 128 for more representational capacity

---

## v8 - 2025-12-13

**Summary**: v5-style opponent curriculum + deeper stacks (200BB)

**Hypothesis**: Combining v5's successful baseline-grounded opponent mix with moderate stack depth (200BB vs 100BB) will enable deeper strategic play while maintaining fundamental winning strategies.

**Configuration**:
- Starting chips: 400 (200BB) - moderate increase from v5's 100BB
- Opponent mix: Dynamic curriculum (use_dynamic_opponent_schedule=True)
  - Stage 0: value_bettor 20%, trapper 20%, call_station 15%, self 15%, historical 10%
  - Stage 100M: self 25%, historical 20%, trapper 15%, value_bettor 15%, call_station 10%
  - Stage 300M: self 40%, historical 30%, trapper 10%, value_bettor 10%, call_station 5%
- Rewards: Pure chip delta (no pot-scaling)
- Training: 150M steps (stopped at 97k generations)

**Results**:
| Opponent     | Win%  | BB/100  |
|--------------|-------|---------|
| random       | 51.9% | -720    |
| call_station | 23.2% | -3,092  |
| tag          | 81.6% | -132    |
| lag          | 47.7% | -1,455  |
| rock         | 76.8% | -1,081  |
| trapper      | 62.0% | -1,732  |
| value_bettor | 86.3% | -489    |

**Action Distribution**: 0% checks, 43-64% raise_150+all_in (hyper-aggressive)

**Analysis**: Catastrophic failure similar to v7. Model wins pots but hemorrhages chips. The 200BB stacks amplified losses - with deeper stacks, aggressive mistakes cost more. Key symptoms:
1. Never checks (0%) - missing free cards and pot control
2. Hyper-aggressive (43-64% large raises/all-ins)
3. High win% but massive negative BB/100 = winning small pots, losing big ones
4. Worse than v5 on every metric despite similar curriculum

**Lessons Learned**:
1. Deeper stacks (200BB vs 100BB) don't work without architecture changes
2. Stack depth amplifies errors - need better fundamentals first
3. Model can't learn proper pot odds with current architecture
4. Consider returning to 100BB, or need opponent modeling to handle deeper play

---

## v7 - 2025-12-12

**Summary**: Deep stacks (500BB) + pure self-play (no baseline opponents)

**Hypothesis**: Deeper stacks would force careful play, and pure self-play would lead to emergent balanced strategies without prescriptive reward shaping.

**Configuration**:
- Starting chips: 1000 (500BB) vs 200 (100BB) in previous versions
- Opponent mix: 70% historical + 30% exploiter, 0% baselines
- Rewards: Pure chip delta (reverted v6 pot-scaling)
- Training stopped early at ~180M/300M steps due to poor results

**Results** (mid-training @ 172.8M steps):
| Opponent     | Win%  | BB/100  |
|--------------|-------|---------|
| random       | 48.7% | -6,643  |
| call_station | 19.4% | -12,200 |

**Analysis**: Catastrophic failure. The model performed far worse than random play against basic opponents. Pure self-play without baseline opponents meant the model never learned fundamental winning strategies. Both sides of self-play learned correlated bad habits, creating a feedback loop of poor play. The deep stacks added complexity without the fundamentals being in place first.

**Lessons Learned**:
1. Pure self-play is not sufficient - baseline opponents provide essential grounding in basic winning strategies
2. Don't increase complexity (deep stacks) before fundamentals are solid
3. Self-play can create echo chambers where both sides learn the same bad patterns
4. Need a curriculum: first learn to beat simple opponents, then add self-play for refinement

---

## v6 - 2025-12-12

**Summary**: Added pot-scaling to reward function

**Results**:
| Opponent     | v5 Win% / BB  | v6 Win% / BB  |
|--------------|---------------|---------------|
| random       | 71.7% / +1495 | 65.9% / +1448 |
| call_station | 44.6% / -710  | 37.7% / -1030 |
| tag          | 95.8% / +497  | 92.6% / +229  |
| lag          | 62.6% / -420  | 55.1% / -271  |
| rock         | 90.1% / -557  | 88.3% / -700  |
| trapper      | 83.2% / -570  | 78.7% / -780  |

**Analysis**: The pot-scaling made things worse. v6 learned to always build big pots (71% pot-sized raises vs call_station) regardless of hand strength. The sqrt scaling rewarded big pots, so it builds them even when it shouldn't.

**Lessons Learned**: Reward shaping that encourages pot-building can backfire - model learns to build pots indiscriminately rather than value-betting appropriately.

---
