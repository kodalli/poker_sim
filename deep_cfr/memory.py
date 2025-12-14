"""Reservoir memory buffer for Single Deep CFR.

Implements reservoir sampling to maintain a fixed-size buffer that
uniformly samples from all iterations. This allows the network to
learn from experiences across the entire training history without
storing everything in memory.

Linear CFR averaging is achieved by weighting samples by their
iteration number during training.
"""

from dataclasses import dataclass, field
from typing import Tuple

import jax
import jax.numpy as jnp
from jax import Array
import numpy as np

from poker_jax.encoding import OBS_DIM
from poker_jax.state import NUM_ACTIONS


@dataclass
class ReservoirMemory:
    """Reservoir sampling buffer for advantage training.

    Uses reservoir sampling to maintain a uniform distribution over
    all samples seen, regardless of total count. This is memory-efficient
    and ensures the network sees diverse game situations.

    Attributes:
        max_size: Maximum number of samples to store
        observations: [max_size, OBS_DIM] game state observations
        advantages: [max_size, NUM_ACTIONS] target advantages
        iterations: [max_size] iteration number for linear averaging
        size: Current number of valid samples
        total_seen: Total samples seen (for reservoir sampling)
    """

    max_size: int = 2_000_000

    # Storage arrays (initialized in __post_init__)
    observations: np.ndarray = field(init=False)
    advantages: np.ndarray = field(init=False)
    iterations: np.ndarray = field(init=False)

    # Counters
    size: int = field(default=0, init=False)
    total_seen: int = field(default=0, init=False)

    def __post_init__(self):
        """Initialize storage arrays."""
        self.observations = np.zeros((self.max_size, OBS_DIM), dtype=np.float32)
        self.advantages = np.zeros((self.max_size, NUM_ACTIONS), dtype=np.float32)
        self.iterations = np.zeros(self.max_size, dtype=np.int32)

    def add_batch(
        self,
        obs: np.ndarray,
        advantages: np.ndarray,
        iteration: int,
        rng: np.random.Generator | None = None,
    ) -> None:
        """Add a batch of samples using vectorized reservoir sampling.

        Optimized version that avoids per-sample loops.

        Args:
            obs: [batch, OBS_DIM] observations
            advantages: [batch, NUM_ACTIONS] advantage values
            iteration: Current CFR iteration
            rng: Random number generator
        """
        if rng is None:
            rng = np.random.default_rng()

        batch_size = obs.shape[0]
        if batch_size == 0:
            return

        # Case 1: Buffer can fit all samples directly
        if self.size + batch_size <= self.max_size:
            start_idx = self.size
            end_idx = self.size + batch_size
            self.observations[start_idx:end_idx] = obs
            self.advantages[start_idx:end_idx] = advantages
            self.iterations[start_idx:end_idx] = iteration
            self.size += batch_size
            self.total_seen += batch_size
            return

        # Case 2: Buffer partially full - fill remaining space first
        if self.size < self.max_size:
            remaining = self.max_size - self.size
            self.observations[self.size:self.max_size] = obs[:remaining]
            self.advantages[self.size:self.max_size] = advantages[:remaining]
            self.iterations[self.size:self.max_size] = iteration
            self.size = self.max_size
            self.total_seen += remaining

            # Continue with rest using reservoir sampling
            obs = obs[remaining:]
            advantages = advantages[remaining:]
            batch_size = obs.shape[0]
            if batch_size == 0:
                return

        # Case 3: Buffer full - vectorized reservoir sampling
        # For each sample i, generate random index in [0, total_seen + i + 1)
        # Keep sample if random index < max_size

        # Generate cumulative totals for each sample position
        sample_totals = self.total_seen + np.arange(1, batch_size + 1)

        # Generate random indices - one per sample
        # random_indices[i] is uniform in [0, total_seen + i + 1)
        random_floats = rng.random(batch_size)
        random_indices = (random_floats * sample_totals).astype(np.int64)

        # Samples to keep: where random index falls within buffer
        keep_mask = random_indices < self.max_size

        if keep_mask.any():
            # Get destination indices for kept samples
            dest_indices = random_indices[keep_mask]

            # Handle potential duplicate destination indices by processing in order
            # This is still vectorized for the common case (few duplicates)
            kept_obs = obs[keep_mask]
            kept_adv = advantages[keep_mask]

            # Use advanced indexing for batch update
            self.observations[dest_indices] = kept_obs
            self.advantages[dest_indices] = kept_adv
            self.iterations[dest_indices] = iteration

        self.total_seen += batch_size

    def sample(
        self,
        batch_size: int,
        rng: np.random.Generator | None = None,
    ) -> Tuple[Array, Array, Array]:
        """Sample a batch of experiences.

        Args:
            batch_size: Number of samples to return
            rng: Random number generator

        Returns:
            Tuple of (observations, advantages, iterations) as JAX arrays
        """
        if rng is None:
            rng = np.random.default_rng()

        if self.size == 0:
            raise ValueError("Cannot sample from empty buffer")

        # Sample with replacement
        indices = rng.integers(0, self.size, size=batch_size)

        obs = jnp.array(self.observations[indices])
        adv = jnp.array(self.advantages[indices])
        iters = jnp.array(self.iterations[indices])

        return obs, adv, iters

    def sample_weighted(
        self,
        batch_size: int,
        rng: np.random.Generator | None = None,
    ) -> Tuple[Array, Array, Array]:
        """Sample with iteration-based weights for linear CFR averaging.

        Samples are weighted by their iteration number, giving more weight
        to later iterations as per linear CFR averaging.

        Args:
            batch_size: Number of samples to return
            rng: Random number generator

        Returns:
            Tuple of (observations, advantages, weights) as JAX arrays
        """
        obs, adv, iters = self.sample(batch_size, rng)

        # Linear weights: iteration number (later = more important)
        weights = iters.astype(jnp.float32)
        # Normalize weights
        weights = weights / jnp.maximum(weights.sum(), 1.0)
        weights = weights * batch_size  # Scale to sum to batch_size

        return obs, adv, weights

    def clear(self) -> None:
        """Clear all samples from buffer."""
        self.size = 0
        self.total_seen = 0

    def __len__(self) -> int:
        """Return number of valid samples."""
        return self.size

    def stats(self) -> dict:
        """Return buffer statistics."""
        return {
            "size": self.size,
            "max_size": self.max_size,
            "total_seen": self.total_seen,
            "fill_ratio": self.size / self.max_size,
            "mean_iteration": float(np.mean(self.iterations[:self.size])) if self.size > 0 else 0,
        }
