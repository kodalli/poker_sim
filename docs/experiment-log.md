# Poker AI Experiment Log

Research log tracking model training experiments, results, and lessons learned.

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
