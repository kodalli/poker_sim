# Poker AI Experiment Log

Research log tracking model training experiments, results, and lessons learned.

---

## Single Deep CFR V1 - 2025-12-14

**Summary**: First neural network-based CFR implementation to replace tabular CFR

**Hypothesis**: A neural network can learn to predict counterfactual advantages directly from game states, eliminating the need for hand abstraction and achieving better generalization.

**Configuration**:
- Algorithm: Single Deep CFR (SD-CFR) with external sampling MCCFR
- Network: MLP [443 → 512 → 256 → 128 → 9] (394K params)
- Training: 500 iterations, 4096 traversals/iter, 100 train steps/iter
- Memory: 2M reservoir buffer with linear CFR averaging
- Total samples: ~4.86M traversal decisions
- Training time: 9.6 min on RTX 4090

**Results** (10,000 games each):
| Opponent     | Win%  | BB/100  | vs CFR V3 Δ |
|--------------|-------|---------|-------------|
| random       | 64.7% | +927    | +42         |
| call_station | 31.8% | -849    | **-1535**   |
| tag          | 89.5% | +439    | **+656**    |
| lag          | 60.5% | +652    | -154        |
| rock         | 82.4% | -888    | -383        |
| trapper      | 67.9% | -1169   | -1031       |
| value_bettor | 89.9% | -476    | -309        |
| **TOTAL**    |       | -1363   | -2715       |

**Analysis**:
1. **Beats aggressive opponents**: TAG +439 (huge gain!), random +927, lag +652
2. **Struggles against passive opponents**: call_station -849, rock -888, trapper -1169
3. **High win rates but negative BB/100** = winning small pots, losing big ones (same pattern as tabular CFR)
4. Only 500 iterations - Deep CFR typically needs 10,000+ for proper convergence
5. Bootstrap advantage targets (network predicting its own outputs) may cause feedback loops

**Key Difference from Tabular CFR**:
- SD-CFR uses neural network to predict advantages (no hand abstraction)
- Network generalizes across similar situations
- Much faster per-iteration but needs more iterations to converge

**Lessons Learned**:
1. **500 iterations insufficient** - Deep CFR needs 10K+ iterations for convergence
2. **Bootstrap targets problematic** - using network's own predictions as training targets creates feedback loops
3. **Should use outcome-based values** - traverse to terminal states and use actual utilities
4. **Need more training** - the loss (0.0005) is low but the network hasn't learned good strategies yet
5. **Potential improvements**:
   - Train for 10,000+ iterations
   - Use `--use-outcome-values` flag for more accurate advantage computation
   - Increase traversals per iteration for more diverse game situations
   - Consider DREAM or Double Neural CFR for better convergence

---

## CFR V3 - 2025-12-14

**Summary**: Reduced abstraction CFR for better convergence (50K info sets vs 200K)

**Hypothesis**: The V2 CFR had only 2% non-uniform strategies because info sets weren't visited enough. Reducing the abstraction space from 200K to 50K would increase visits per info set by 4x, leading to better strategy convergence.

**Configuration**:
- Algorithm: Monte Carlo CFR+ (MCCFR) with outcome sampling
- Abstraction V3 (reduced):
  - Hand buckets: Preflop 169, Flop 15, Turn 10, River 10
  - History encoding: 27 keys (3 pot ratios × 3 bet levels × 3 bet sizes)
  - Total: ~50,000 info set slots
- Training: 100K iterations, batch_size=4096
- Total samples: 409.6M games (~36 min on RTX 4090)

**Results** (10,000 games each):
| Opponent     | Win%  | BB/100  | vs V2 Δ |
|--------------|-------|---------|---------|
| random       | 61.1% | +885    | +7      |
| call_station | 45.1% | +686    | **+85** |
| tag          | 62.8% | -217    | **+52** |
| lag          | 50.8% | +806    | **+138**|
| rock         | 69.0% | -505    | -6      |
| trapper      | 58.2% | -138    | **+56** |
| value_bettor | 79.4% | -167    | **+28** |

**Convergence Metrics**:
- V2: 4,104 non-uniform / 200,000 active (2.05%)
- V3: 1,933 non-uniform / 50,000 active (3.87%)
- V3 achieved nearly 2x the proportion of non-uniform strategies

**Analysis**:
1. **Reduced abstraction improved performance** on 6 of 7 opponents
2. **Biggest gains vs aggressive opponents**: LAG +138, call_station +85
3. **Still loses to sophisticated opponents**: TAG, rock, trapper, value_bettor
4. High win rates but negative BB/100 = winning small pots, losing big ones
5. Strategy distribution stable: 55.4% aggression, 22.7% passivity, 10.7% fold

**Lessons Learned**:
1. **Fewer info sets = more visits = better convergence** - the hypothesis was confirmed
2. **CFR abstraction quality matters more than quantity** - 50K well-visited > 200K sparse
3. **CFR struggles with deep-stack play** - loses to opponents who trap/value bet
4. **Potential improvements**:
   - Train longer (500K+ iterations) with reduced abstraction
   - Better river bucketing (hand categories instead of equity)
   - Consider NFSP (Neural Fictitious Self-Play) to combine PPO + CFR

---

## CFR V2 - 2025-12-14

**Summary**: First improved CFR with finer abstraction (200K info sets)

**Configuration**:
- Algorithm: Monte Carlo CFR+ (MCCFR)
- Abstraction V2 (finer buckets):
  - Hand buckets: Preflop 169, Flop 50, Turn 30, River 20
  - History encoding: 80 keys (5 pot ratios × 4 bet levels × 4 bet sizes)
  - Total: ~200,000 info set slots
- Training: 100K iterations, batch_size=4096
- Total samples: 409.6M games

**Results** (10,000 games each):
| Opponent     | Win%  | BB/100  |
|--------------|-------|---------|
| random       | 60.3% | +878    |
| call_station | 45.0% | +601    |
| tag          | 61.8% | -269    |
| lag          | 51.0% | +668    |
| rock         | 69.5% | -499    |
| trapper      | 58.7% | -194    |
| value_bettor | 79.3% | -195    |

**Analysis**:
1. Only 4,104 of 200,000 info sets had non-uniform strategies (2%)
2. Most situations played near-uniform (essentially random)
3. Won against random/call_station/LAG but lost to sophisticated opponents
4. High win rates but negative BB/100 indicates poor value betting

**Lessons Learned**:
1. 200K info sets too sparse with 409M samples (~2K visits per slot)
2. Need more visits per info set for CFR to converge
3. Reduced abstraction would increase visit frequency

---

## v11 - 2025-12-13

**Summary**: Extended training (500M steps) with full opponent diversity (all 7 types)

**Hypothesis**: Longer training with exposure to ALL opponent types (including rock, trapper, value_bettor) would enable the model to develop more robust strategies and improve against passive/trapping opponents.

**Configuration**:
- Architecture: ActorCriticMLPWithOpponentModel (same as v9/v10)
  - OpponentLSTM: 64 hidden, 32 embedding dim
- Starting chips: 200 (100BB)
- Training: **500M steps** (5x longer than v10)
- **Full Mixed Opponent Training**: 40% self, 10% random, 10% call_station, 10% TAG, 10% LAG, 7% rock, 7% trapper, 6% value_bettor

**Results** (10,000 games each):
| Opponent     | Win%  | BB/100  | vs v10 Δ |
|--------------|-------|---------|----------|
| random       | 78.8% | +2,711  | **+1,672** |
| call_station | 38.4% | -1,396  | -334     |
| tag          | 96.4% | +939    | **+557** |
| lag          | 78.9% | +2,591  | **+2,260** |
| rock         | 86.3% | -330    | -89      |
| trapper      | 77.0% | -331    | +19      |
| value_bettor | 91.2% | -105    | +9       |

**Aggregate Performance**:
- v9 total: -929 BB/100
- v10 total: -15 BB/100
- **v11 total: +4,080 BB/100** 🚀

**Action Distribution**:
- Learned heavy checking: 22-40% check rate (vs 5-17% in v10)
- 40% check vs call_station - excellent pot control!
- Low fold rates (2-7%) - confident play
- Adapts aggression per opponent

**Training Observations**:
- Mid-training dip (100-250M): vs_random dropped to negative BB/100
- Recovery (300M+): Performance rebounded strongly
- Explained variance: Only reached ~12% (room for value function improvement)
- Policy loss: Converged to ~0.0005 (stable)

**Analysis**:
1. **Massive gains vs aggressive opponents**: random +1,672, lag +2,260, tag +557
2. **Still struggles vs call_station**: -1,396 BB/100 (worse than v10's -1,062)
3. **Learned pot control**: 40% checking vs call_station shows adaptation
4. **High win rates across board**: 77-96% vs all except call_station (38%)
5. **5x training paid off**: Total BB/100 went from -15 to +4,080

**Lessons Learned**:
1. **Longer training (500M) significantly helps** - model continued improving past 100M
2. **Full opponent diversity works** - adding rock/trapper/value_bettor to mix helped
3. **Trade-off shifted**: Now dominates aggressive opponents but regressed vs call_station
4. **Explained variance at 12%** suggests value function could still improve with more training or architecture changes
5. **Potential next steps**:
   - Try 1B steps to see if improvement continues
   - Increase LSTM capacity (64 → 128) for finer opponent modeling
   - Consider curriculum: heavy call_station first, then diversify

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
