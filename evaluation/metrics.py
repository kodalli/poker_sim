"""Statistical metrics for poker evaluation."""

import math
from typing import Tuple


def compute_bb_per_100(
    total_chips: float,
    num_games: int,
    big_blind: int = 2,
) -> float:
    """Compute BB/100 (big blinds won per 100 hands).

    BB/100 is the standard measure of poker win rate.
    Positive = winning, negative = losing.

    Args:
        total_chips: Net chips won/lost
        num_games: Number of games played
        big_blind: Big blind amount

    Returns:
        BB/100 rate
    """
    if num_games == 0:
        return 0.0

    bb_won = total_chips / big_blind
    return (bb_won / num_games) * 100


def compute_confidence_interval(
    win_rate: float,
    n_samples: int,
    confidence: float = 0.95,
) -> Tuple[float, float]:
    """Compute confidence interval for win rate using Wilson score interval.

    More accurate than normal approximation for win rates near 0 or 1.

    Args:
        win_rate: Observed win rate (0-1)
        n_samples: Number of samples
        confidence: Confidence level (default 95%)

    Returns:
        Tuple of (lower_bound, upper_bound)
    """
    if n_samples == 0:
        return (0.0, 1.0)

    # Z-score for confidence level
    z_scores = {0.90: 1.645, 0.95: 1.96, 0.99: 2.576}
    z = z_scores.get(confidence, 1.96)

    # Wilson score interval
    n = n_samples
    p = win_rate

    denominator = 1 + z**2 / n
    center = (p + z**2 / (2*n)) / denominator
    spread = z * math.sqrt((p*(1-p) + z**2/(4*n)) / n) / denominator

    lower = max(0, center - spread)
    upper = min(1, center + spread)

    return (lower, upper)


def required_games_for_significance(
    win_rate: float,
    margin: float = 0.02,
    confidence: float = 0.95,
) -> int:
    """Calculate games needed for desired confidence interval width.

    Args:
        win_rate: Expected win rate
        margin: Desired margin of error (half the CI width)
        confidence: Confidence level

    Returns:
        Number of games required
    """
    z_scores = {0.90: 1.645, 0.95: 1.96, 0.99: 2.576}
    z = z_scores.get(confidence, 1.96)

    p = win_rate
    # n = (z^2 * p * (1-p)) / margin^2
    n = (z**2 * p * (1 - p)) / (margin**2)

    return int(math.ceil(n))


def is_statistically_significant(
    win_rate: float,
    n_samples: int,
    baseline: float = 0.5,
    alpha: float = 0.05,
) -> bool:
    """Test if win rate is significantly different from baseline.

    Args:
        win_rate: Observed win rate
        n_samples: Number of samples
        baseline: Expected baseline win rate (default 0.5 for fair game)
        alpha: Significance level

    Returns:
        True if significantly different from baseline
    """
    lower, upper = compute_confidence_interval(win_rate, n_samples, 1 - alpha)
    return baseline < lower or baseline > upper
