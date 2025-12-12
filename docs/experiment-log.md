# Poker AI Experiment Log

Research log tracking model training experiments, results, and lessons learned.

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
