# Poker AI Experiment Log

Research log tracking model training experiments, results, and lessons learned.

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
