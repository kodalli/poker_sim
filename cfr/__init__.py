"""CFR (Counterfactual Regret Minimization) module for poker.

This module provides:
- JAX-optimized CFR+ training for computing Nash equilibrium strategies
- Hand and betting history abstraction for tractable game tree (V2 with finer buckets)
- CFR opponent function for use in training
- Exploitability measurement for Nash distance
"""

from .abstraction import (
    preflop_bucket,
    postflop_bucket,
    postflop_bucket_v2,
    encode_history,
    encode_history_v2,
    make_info_key,
    make_info_key_v2,
    compute_info_key_batch_v2,
    MAX_INFO_SETS,
    NUM_BUCKETS_PREFLOP,
    NUM_BUCKETS_POSTFLOP,
    NUM_BUCKETS_FLOP,
    NUM_BUCKETS_TURN,
    NUM_BUCKETS_RIVER,
)
from .cfr_opponent import cfr_opponent, load_cfr_strategy, create_cfr_opponent
from .exploitability import compute_exploitability, analyze_strategy_quality

__all__ = [
    # Abstraction (legacy)
    "preflop_bucket",
    "postflop_bucket",
    "encode_history",
    "make_info_key",
    # Abstraction (V2)
    "postflop_bucket_v2",
    "encode_history_v2",
    "make_info_key_v2",
    "compute_info_key_batch_v2",
    # Opponent
    "cfr_opponent",
    "load_cfr_strategy",
    "create_cfr_opponent",
    # Exploitability
    "compute_exploitability",
    "analyze_strategy_quality",
    # Constants
    "MAX_INFO_SETS",
    "NUM_BUCKETS_PREFLOP",
    "NUM_BUCKETS_POSTFLOP",
    "NUM_BUCKETS_FLOP",
    "NUM_BUCKETS_TURN",
    "NUM_BUCKETS_RIVER",
]
