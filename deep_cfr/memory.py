"""Hybrid reservoir memory buffer for Single Deep CFR.

Uses CPU storage (NumPy) to avoid JAX's immutable array copy overhead,
but GPU-accelerated sampling for fast training.

Linear CFR averaging is achieved by weighting samples by their
iteration number during training.
"""

from dataclasses import dataclass, field
from typing import Tuple

import jax
import jax.numpy as jnp
import jax.random as jrandom
from jax import Array
import numpy as np

from poker_jax.encoding import OBS_DIM
from poker_jax.state import NUM_ACTIONS


@jax.jit
def _compute_weights(iterations: Array, batch_size: int) -> Array:
    """Compute linear CFR weights from iteration numbers."""
    weights = iterations.astype(jnp.float32)
    weights = weights / jnp.maximum(weights.sum(), 1.0)
    return weights * batch_size


@jax.jit
def _compute_multi_batch_weights(
    iterations: Array,  # [n_batches, batch_size]
    batch_size: int,
) -> Array:
    """Compute weights for multiple batches."""
    weights = iterations.astype(jnp.float32)
    weights_sum = jnp.maximum(weights.sum(axis=1, keepdims=True), 1.0)
    return weights / weights_sum * batch_size


@dataclass
class ReservoirMemory:
    """Hybrid reservoir sampling buffer for advantage training.

    Uses NumPy for storage (efficient updates) and JAX for sampling
    (GPU-accelerated). This avoids the memory overhead of JAX's
    immutable array updates while maintaining fast training throughput.

    Attributes:
        max_size: Maximum number of samples to store
        observations: [max_size, OBS_DIM] game state observations (CPU)
        advantages: [max_size, NUM_ACTIONS] target advantages (CPU)
        iterations: [max_size] iteration number for linear averaging (CPU)
        size: Current number of valid samples
        total_seen: Total samples seen (for reservoir sampling)
    """

    max_size: int = 10_000_000  # 10M samples - uses ~18GB on GPU when sampled

    # CPU storage arrays (initialized in __post_init__)
    observations: np.ndarray = field(init=False)
    advantages: np.ndarray = field(init=False)
    iterations: np.ndarray = field(init=False)

    # Counters
    size: int = field(default=0, init=False)
    total_seen: int = field(default=0, init=False)

    # JAX RNG key for sampling
    _rng_key: Array = field(init=False)

    def __post_init__(self):
        """Initialize CPU storage arrays."""
        self.observations = np.zeros((self.max_size, OBS_DIM), dtype=np.float32)
        self.advantages = np.zeros((self.max_size, NUM_ACTIONS), dtype=np.float32)
        self.iterations = np.zeros(self.max_size, dtype=np.int32)
        self._rng_key = jrandom.PRNGKey(42)

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
            rng: Random number generator (uses internal JAX RNG if None)
        """
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
        # Use JAX RNG for GPU-friendly random number generation
        self._rng_key, subkey = jrandom.split(self._rng_key)

        # For each sample i, generate random index in [0, total_seen + i + 1)
        sample_totals = self.total_seen + np.arange(1, batch_size + 1)
        random_floats = np.array(jrandom.uniform(subkey, shape=(batch_size,)))
        random_indices = (random_floats * sample_totals).astype(np.int64)

        # Samples to keep: where random index falls within buffer
        keep_mask = random_indices < self.max_size

        if keep_mask.any():
            # Get destination indices for kept samples
            dest_indices = random_indices[keep_mask]

            # Use NumPy advanced indexing for efficient batch update
            self.observations[dest_indices] = obs[keep_mask]
            self.advantages[dest_indices] = advantages[keep_mask]
            self.iterations[dest_indices] = iteration

        self.total_seen += batch_size

    def sample(
        self,
        batch_size: int,
        rng: np.random.Generator | None = None,
    ) -> Tuple[Array, Array, Array]:
        """Sample a batch of experiences.

        Uses JAX RNG and transfers data to GPU for training.

        Args:
            batch_size: Number of samples to return
            rng: Unused (uses internal JAX RNG)

        Returns:
            Tuple of (observations, advantages, iterations) as JAX arrays on GPU
        """
        if self.size == 0:
            raise ValueError("Cannot sample from empty buffer")

        # Generate random indices using JAX (GPU)
        self._rng_key, subkey = jrandom.split(self._rng_key)
        indices = np.array(jrandom.randint(subkey, shape=(batch_size,), minval=0, maxval=self.size))

        # Sample from CPU and transfer to GPU
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
            rng: Unused (uses internal JAX RNG)

        Returns:
            Tuple of (observations, advantages, weights) as JAX arrays on GPU
        """
        obs, adv, iters = self.sample(batch_size, rng)
        weights = _compute_weights(iters, batch_size)
        return obs, adv, weights

    def sample_multi_batch(
        self,
        n_batches: int,
        batch_size: int,
        rng: np.random.Generator | None = None,
    ) -> Tuple[Array, Array, Array]:
        """Sample multiple batches at once for efficient training.

        Optimized for multi-step training - single CPU→GPU transfer.

        Args:
            n_batches: Number of batches to sample
            batch_size: Samples per batch
            rng: Unused (uses internal JAX RNG)

        Returns:
            Tuple of (observations, advantages, weights) as JAX arrays
            with shapes [n_batches, batch_size, ...]
        """
        if self.size == 0:
            raise ValueError("Cannot sample from empty buffer")

        # Generate all random indices at once using JAX
        total_samples = n_batches * batch_size
        self._rng_key, subkey = jrandom.split(self._rng_key)
        all_indices = np.array(jrandom.randint(
            subkey, shape=(total_samples,), minval=0, maxval=self.size
        ))

        # Get all data at once from CPU
        all_obs = self.observations[all_indices]
        all_adv = self.advantages[all_indices]
        all_iters = self.iterations[all_indices]

        # Reshape to [n_batches, batch_size, ...]
        obs = all_obs.reshape(n_batches, batch_size, -1)
        adv = all_adv.reshape(n_batches, batch_size, -1)
        iters = all_iters.reshape(n_batches, batch_size)

        # Transfer to GPU and compute weights
        obs_jax = jnp.array(obs)
        adv_jax = jnp.array(adv)
        iters_jax = jnp.array(iters)
        weights = _compute_multi_batch_weights(iters_jax, batch_size)

        return obs_jax, adv_jax, weights

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
