"""Single Deep CFR implementation for poker.

This module implements Single Deep CFR (SD-CFR), a neural network-based
variant of Counterfactual Regret Minimization that eliminates the need
for hand abstraction by learning advantages directly from game states.

Key components:
- AdvantageNetwork: Neural network predicting counterfactual advantages
- ReservoirMemory: Memory buffer with reservoir sampling
- MCCFR traversal: External sampling Monte Carlo CFR
- Training loop: Main SD-CFR algorithm

Reference: https://github.com/EricSteinberger/Deep-CFR
"""

from deep_cfr.network import (
    AdvantageNetwork,
    create_advantage_network,
    init_advantage_network,
    count_parameters,
)
from deep_cfr.memory import ReservoirMemory
from deep_cfr.strategy import get_strategy, regret_match
from deep_cfr.opponent import (
    create_sdcfr_opponent,
    load_sdcfr_model,
    register_sdcfr_opponent,
)

__all__ = [
    # Network
    "AdvantageNetwork",
    "create_advantage_network",
    "init_advantage_network",
    "count_parameters",
    # Memory
    "ReservoirMemory",
    # Strategy
    "get_strategy",
    "regret_match",
    # Opponent
    "create_sdcfr_opponent",
    "load_sdcfr_model",
    "register_sdcfr_opponent",
]
