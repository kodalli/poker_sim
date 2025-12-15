"""Single Deep CFR training loop.

Main training algorithm:
1. For each iteration:
   a. Traverse games, collecting (obs, advantage) samples
   b. Add samples to reservoir memory with iteration weight
   c. Train network on sampled batches

Linear CFR averaging is achieved by weighting samples by iteration number.
"""

import argparse
import time
from pathlib import Path
from functools import partial

import jax
import jax.numpy as jnp
import jax.random as jrandom
from jax import Array
import numpy as np
import optax
import flax.linen as nn

from poker_jax.encoding import OBS_DIM
from poker_jax.state import NUM_ACTIONS

from deep_cfr.network import (
    AdvantageNetwork,
    create_advantage_network,
    init_advantage_network,
    count_parameters,
)
from deep_cfr.memory import ReservoirMemory
from deep_cfr.traverse import batch_traverse, traverse_with_outcome_values


@partial(jax.jit, static_argnums=(4, 5))
def train_step(
    params: dict,
    opt_state: optax.OptState,
    batch: tuple,
    rng_key: Array,
    network: AdvantageNetwork,
    optimizer: optax.GradientTransformation,
) -> tuple:
    """Single training step.

    Args:
        params: Network parameters
        opt_state: Optimizer state
        batch: (observations, target_advantages, weights)
        rng_key: PRNG key for dropout
        network: AdvantageNetwork module
        optimizer: Optax optimizer

    Returns:
        new_params: Updated parameters
        new_opt_state: Updated optimizer state
        loss: Training loss
    """
    obs, target_adv, weights = batch

    def loss_fn(params):
        # Forward pass
        pred_adv = network.apply(
            {"params": params}, obs, training=True,
            rngs={"dropout": rng_key}
        )

        # Huber loss (robust to outliers)
        diff = pred_adv - target_adv
        abs_diff = jnp.abs(diff)
        huber = jnp.where(abs_diff <= 1.0, 0.5 * diff**2, abs_diff - 0.5)

        # Weight by iteration (linear CFR averaging)
        weighted_loss = huber * weights[:, None]

        return weighted_loss.mean()

    loss, grads = jax.value_and_grad(loss_fn)(params)

    # Apply gradients
    updates, new_opt_state = optimizer.update(grads, opt_state, params)
    new_params = optax.apply_updates(params, updates)

    return new_params, new_opt_state, loss


def _make_multi_step_train(network, optimizer, n_steps: int):
    """Create JIT-compiled multi-step training function."""

    @jax.jit
    def multi_step_train(params, opt_state, batches, rng_key):
        """Train for n_steps on pre-sampled batches.

        Args:
            params: Network parameters
            opt_state: Optimizer state
            batches: (obs, adv, weights) each with shape [n_steps, batch_size, ...]
            rng_key: PRNG key

        Returns:
            new_params, new_opt_state, mean_loss
        """
        obs_batches, adv_batches, weight_batches = batches

        def body_fn(i, carry):
            params, opt_state, total_loss, rng_key = carry
            rng_key, step_key = jrandom.split(rng_key)

            # Get batch for this step
            batch = (obs_batches[i], adv_batches[i], weight_batches[i])

            # Train step
            params, opt_state, loss = train_step(
                params, opt_state, batch, step_key, network, optimizer
            )

            return (params, opt_state, total_loss + loss, rng_key)

        init_carry = (params, opt_state, 0.0, rng_key)
        params, opt_state, total_loss, _ = jax.lax.fori_loop(
            0, n_steps, body_fn, init_carry
        )

        return params, opt_state, total_loss / n_steps

    return multi_step_train


# Cache for multi-step train functions
_TRAIN_CACHE = {}


def train_single_deep_cfr(
    iterations: int = 10_000,
    traversals_per_iter: int = 4096,
    train_steps_per_iter: int = 100,
    batch_size: int = 2048,
    learning_rate: float = 1e-4,
    weight_decay: float = 1e-5,
    memory_size: int = 10_000_000,  # 10M samples (CPU storage, GPU sampling)
    max_game_steps: int = 50,
    hidden_dims: tuple = (512, 256, 128),
    checkpoint_every: int = 1000,
    log_every: int = 100,
    output_dir: str = "models/deep_cfr",
    seed: int = 42,
    use_outcome_values: bool = False,
    resume_from: str = None,
):
    """Train Single Deep CFR.

    Args:
        iterations: Number of CFR iterations
        traversals_per_iter: Number of game traversals per iteration
        train_steps_per_iter: Network training steps per iteration
        batch_size: Training batch size
        learning_rate: Adam learning rate
        weight_decay: L2 regularization
        memory_size: Reservoir memory capacity
        max_game_steps: Maximum steps per game
        hidden_dims: Network hidden layer dimensions
        checkpoint_every: Save checkpoint every N iterations
        log_every: Log metrics every N iterations
        output_dir: Output directory for checkpoints
        seed: Random seed
        use_outcome_values: Use outcome-based advantage computation (slower but more accurate)
        resume_from: Path to checkpoint file to resume from (e.g., 'models/deep_cfr/params_iter5000.pkl')
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Single Deep CFR Training")
    print("=" * 60)
    print(f"Iterations: {iterations:,}")
    print(f"Traversals per iter: {traversals_per_iter:,}")
    print(f"Train steps per iter: {train_steps_per_iter}")
    print(f"Batch size: {batch_size:,}")
    print(f"Learning rate: {learning_rate}")
    print(f"Memory size: {memory_size:,}")
    print(f"Hidden dims: {hidden_dims}")
    print(f"Use outcome values: {use_outcome_values}")
    print(f"Output: {output_dir}")
    print("=" * 60)

    # Initialize
    rng = np.random.default_rng(seed)
    rng_key = jrandom.PRNGKey(seed)

    # Create network
    network = create_advantage_network(hidden_dims=hidden_dims)
    rng_key, init_key = jrandom.split(rng_key)
    params = init_advantage_network(network, init_key)
    n_params = count_parameters(params)
    print(f"Network parameters: {n_params:,}")

    # Resume from checkpoint if specified
    start_iteration = 1
    if resume_from is not None:
        import pickle
        import re
        resume_path = Path(resume_from)
        if not resume_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {resume_from}")

        with open(resume_path, 'rb') as f:
            params = pickle.load(f)

        # Extract iteration number from filename (e.g., params_iter5000.pkl -> 5000)
        match = re.search(r'iter(\d+)', resume_path.name)
        if match:
            start_iteration = int(match.group(1)) + 1
            print(f"Resuming from iteration {start_iteration - 1}")
        else:
            print(f"Warning: Could not parse iteration from filename, starting from 1")

        # Advance RNG state to match where we left off
        for _ in range(start_iteration - 1):
            rng_key, _ = jrandom.split(rng_key)

        print(f"Loaded checkpoint: {resume_from}")

    # Create optimizer
    optimizer = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adamw(learning_rate, weight_decay=weight_decay),
    )
    opt_state = optimizer.init(params)

    # Create memory buffer
    memory = ReservoirMemory(max_size=memory_size)

    # Training metrics
    start_time = time.time()
    total_samples = 0
    train_losses = []

    print("\nStarting training...")

    for iteration in range(start_iteration, iterations + 1):
        iter_start = time.time()

        # === Phase 1: Traverse games and collect samples ===
        rng_key, traverse_key = jrandom.split(rng_key)

        if use_outcome_values:
            obs, advantages, valid_masks = traverse_with_outcome_values(
                network, params, traverse_key,
                n_games=traversals_per_iter,
                max_steps=max_game_steps,
            )
        else:
            obs, advantages, valid_masks = batch_traverse(
                network, params, traverse_key,
                n_games=traversals_per_iter,
                max_steps=max_game_steps,
            )

        n_samples = obs.shape[0]
        total_samples += n_samples

        # Add to memory with iteration weight
        if n_samples > 0:
            memory.add_batch(obs, advantages, iteration, rng)

        # === Phase 2: Train network on memory samples ===
        avg_loss = 0.0

        if len(memory) >= batch_size:
            # Get or create multi-step train function
            cache_key = (id(network), id(optimizer), train_steps_per_iter)
            if cache_key not in _TRAIN_CACHE:
                _TRAIN_CACHE[cache_key] = _make_multi_step_train(
                    network, optimizer, train_steps_per_iter
                )
            multi_train = _TRAIN_CACHE[cache_key]

            # Sample all batches at once (single memory access + single CPU→GPU transfer)
            batches = memory.sample_multi_batch(train_steps_per_iter, batch_size, rng)

            # Run all train steps in JIT-compiled loop
            rng_key, train_key = jrandom.split(rng_key)
            params, opt_state, avg_loss = multi_train(
                params, opt_state, batches, train_key
            )
            avg_loss = float(avg_loss)
        train_losses.append(avg_loss)

        # === Logging ===
        if iteration % log_every == 0:
            elapsed = time.time() - start_time
            iter_time = time.time() - iter_start
            samples_per_sec = total_samples / elapsed

            mem_stats = memory.stats()

            print(
                f"Iter {iteration:,}/{iterations:,} | "
                f"Loss: {avg_loss:.4f} | "
                f"Samples: {n_samples:,} ({total_samples:,} total) | "
                f"Memory: {mem_stats['size']:,}/{mem_stats['max_size']:,} | "
                f"Time: {elapsed:.1f}s ({samples_per_sec:,.0f} samples/s)"
            )

        # === Checkpointing ===
        if iteration % checkpoint_every == 0:
            import pickle
            pickle_path = output_path / f"params_iter{iteration}.pkl"
            with open(pickle_path, 'wb') as f:
                pickle.dump(params, f)

            print(f"Saved checkpoint to {pickle_path}")

    # === Final save ===
    elapsed = time.time() - start_time

    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)
    print(f"Total time: {elapsed:.1f}s ({elapsed/60:.1f} min)")
    print(f"Total samples: {total_samples:,}")
    print(f"Final memory size: {len(memory):,}")
    print(f"Final loss: {train_losses[-1]:.4f}")

    # Save final model
    import pickle
    final_path = output_path / "params_final.pkl"
    with open(final_path, 'wb') as f:
        pickle.dump(params, f)
    print(f"Saved final model to {final_path}")

    # Save training config
    import json
    config = {
        "iterations": iterations,
        "traversals_per_iter": traversals_per_iter,
        "train_steps_per_iter": train_steps_per_iter,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "memory_size": memory_size,
        "hidden_dims": list(hidden_dims),
        "use_outcome_values": use_outcome_values,
        "total_samples": total_samples,
        "training_time_sec": elapsed,
        "final_loss": float(train_losses[-1]) if train_losses else 0.0,
    }
    config_path = output_path / "config.json"
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    print(f"Saved config to {config_path}")

    # Save loss history
    loss_path = output_path / "loss_history.npy"
    np.save(str(loss_path), np.array(train_losses))

    return params


def main():
    parser = argparse.ArgumentParser(description="Train Single Deep CFR")
    parser.add_argument("--iterations", type=int, default=10_000, help="CFR iterations")
    parser.add_argument("--traversals", type=int, default=4096, help="Traversals per iteration")
    parser.add_argument("--train-steps", type=int, default=100, help="Train steps per iteration")
    parser.add_argument("--batch-size", type=int, default=2048, help="Training batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--memory-size", type=int, default=10_000_000, help="Reservoir memory size")
    parser.add_argument("--hidden-dims", type=str, default="512,256,128", help="Hidden layer dims")
    parser.add_argument("--checkpoint-every", type=int, default=1000, help="Checkpoint interval")
    parser.add_argument("--log-every", type=int, default=100, help="Log interval")
    parser.add_argument("--output-dir", type=str, default="models/deep_cfr", help="Output directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--use-outcome-values", action="store_true", help="Use outcome-based advantages")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")

    args = parser.parse_args()

    hidden_dims = tuple(int(x) for x in args.hidden_dims.split(","))

    train_single_deep_cfr(
        iterations=args.iterations,
        traversals_per_iter=args.traversals,
        train_steps_per_iter=args.train_steps,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        memory_size=args.memory_size,
        hidden_dims=hidden_dims,
        checkpoint_every=args.checkpoint_every,
        log_every=args.log_every,
        output_dir=args.output_dir,
        seed=args.seed,
        use_outcome_values=args.use_outcome_values,
        resume_from=args.resume,
    )


if __name__ == "__main__":
    main()
