"""JAX-accelerated training loop for poker RL."""

import logging
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from typing import Callable
import time

import jax
import jax.numpy as jnp
import jax.random as jrandom
import optax
from jax import Array
from rich.console import Console
from rich.table import Table
from tqdm import tqdm

from poker_jax import (
    GameState,
    reset,
    step,
    get_rewards,
    encode_state_for_current_player,
    get_valid_actions_mask,
    ACTION_FOLD,
    ACTION_CHECK,
    ACTION_CALL,
    ACTION_RAISE,
    ACTION_ALL_IN,
)
from evaluation.opponents import (
    random_opponent,
    call_station_opponent,
    tag_opponent,
    lag_opponent,
    rock_opponent,
    trapper_opponent,
    value_bettor_opponent,
    NEEDS_OBS,
)
from poker_jax.network import (
    ActorCriticMLP,
    create_network,
    init_network,
    sample_action,
    count_parameters,
)
from poker_jax.encoding import OBS_DIM, get_valid_actions_from_obs
from training.jax_ppo import (
    PPOConfig,
    PPOMetrics,
    Trajectory,
    create_optimizer,
    ppo_update,
)
from training.logging import MetricsLogger
from training.elo import CheckpointPool, OPPONENT_ELOS, INITIAL_ELO


@dataclass
class JAXTrainingConfig:
    """Configuration for JAX-accelerated training."""

    # Parallelism (optimized for RTX 4090 24GB VRAM)
    num_parallel_games: int = 1536  # Games to run in parallel
    steps_per_game: int = 200  # Max steps per game (auto-resets)

    # Training loop
    total_steps: int = 1_000_000  # Total environment steps
    steps_per_update: int = 3072  # Steps between PPO updates

    # Evaluation
    eval_every: int = 10_000  # Evaluate every N steps
    eval_games: int = 100  # Games for evaluation

    # Checkpointing (save ~20 checkpoints for long runs)
    checkpoint_every: int = 500_000_000  # Every 500M steps
    checkpoint_dir: str = "models/jax/checkpoints"

    # Logging (log_every is in updates, not steps)
    log_every: int = 100  # Log every N updates (reduces CSV/TB storage)
    tensorboard_dir: str | None = "logs/jax"

    # Game settings
    starting_chips: int = 200
    small_blind: int = 1
    big_blind: int = 2

    # Mixed opponent training (v3.1)
    use_mixed_opponents: bool = False
    opponent_mix: dict = field(default_factory=lambda: {
        "self": 0.50,       # 50% self-play
        "random": 0.15,     # 15% random
        "call_station": 0.15,  # 15% call station
        "tag": 0.10,        # 10% tight-aggressive
        "lag": 0.10,        # 10% loose-aggressive
    })

    # Historical self-play (v3.2) with anti-exploitation (v3.3)
    use_historical_selfplay: bool = False
    historical_pool_size: int = 20
    historical_save_every: int = 50_000_000  # Save checkpoint every 50M steps
    historical_opponent_mix: dict = field(default_factory=lambda: {
        "self": 0.30,           # 30% current self-play
        "historical": 0.25,     # 25% historical checkpoints
        "trapper": 0.15,        # 15% trapper (anti-aggro) - v3.3
        "call_station": 0.10,   # 10% call station
        "tag": 0.08,            # 8% TAG
        "rock": 0.07,           # 7% rock (very tight) - v3.3
        "random": 0.05,         # 5% random
    })

    # Dynamic opponent scheduling (v3.4) - curriculum learning
    # Shifts from exploitative opponents early to self-play late
    use_dynamic_opponent_schedule: bool = True
    opponent_schedule: list = field(default_factory=lambda: [
        # (step, mix_dict) - linearly interpolate between milestones
        # v3.5: Added value_bettor (only bets with premium hands) to force fold learning
        (0, {
            "self": 0.15, "historical": 0.10, "trapper": 0.20,
            "value_bettor": 0.20,  # Forces folding - only bets with premium hands
            "call_station": 0.15, "tag": 0.10, "rock": 0.05, "random": 0.05,
        }),
        (100_000_000, {  # 100M: Start increasing self-play
            "self": 0.25, "historical": 0.20, "trapper": 0.15,
            "value_bettor": 0.15,
            "call_station": 0.10, "tag": 0.07, "rock": 0.05, "random": 0.03,
        }),
        (300_000_000, {  # 300M: Heavy self-play
            "self": 0.40, "historical": 0.30, "trapper": 0.10,
            "value_bettor": 0.10,
            "call_station": 0.05, "tag": 0.03, "rock": 0.02, "random": 0.00,
        }),
        (600_000_000, {  # 600M+: Mostly self-play for refinement
            "self": 0.50, "historical": 0.40, "trapper": 0.05,
            "value_bettor": 0.05,
            "call_station": 0.00, "tag": 0.00, "rock": 0.00, "random": 0.00,
        }),
    ])


# Opponent type indices for mixed training
OPP_SELF = 0
OPP_HISTORICAL = 1  # v3.2: Historical self-play
OPP_RANDOM = 2
OPP_CALL_STATION = 3
OPP_TAG = 4
OPP_LAG = 5
OPP_ROCK = 6
OPP_TRAPPER = 7  # v3.3: Anti-aggro trapper
OPP_VALUE_BETTOR = 8  # v3.5: Only bets with premium hands (exploits never-fold)


def interpolate_opponent_mix(
    current_step: int,
    schedule: list[tuple[int, dict]],
) -> dict:
    """Interpolate opponent mix based on training progress.

    Args:
        current_step: Current training step
        schedule: List of (step, mix_dict) tuples defining milestones

    Returns:
        Interpolated mix dictionary for current step
    """
    if not schedule:
        return {}

    # Sort schedule by step
    schedule = sorted(schedule, key=lambda x: x[0])

    # Find surrounding milestones
    prev_step, prev_mix = schedule[0]
    next_step, next_mix = schedule[-1]

    for i, (step, mix) in enumerate(schedule):
        if step <= current_step:
            prev_step, prev_mix = step, mix
        if step > current_step:
            next_step, next_mix = step, mix
            break

    # If past last milestone, use final mix
    if current_step >= schedule[-1][0]:
        return schedule[-1][1].copy()

    # If before first milestone, use first mix
    if current_step <= schedule[0][0]:
        return schedule[0][1].copy()

    # Linear interpolation
    if next_step == prev_step:
        alpha = 0.0
    else:
        alpha = (current_step - prev_step) / (next_step - prev_step)

    # Interpolate each key
    all_keys = set(prev_mix.keys()) | set(next_mix.keys())
    result = {}
    for key in all_keys:
        prev_val = prev_mix.get(key, 0.0)
        next_val = next_mix.get(key, 0.0)
        result[key] = prev_val + alpha * (next_val - prev_val)

    return result


def sample_opponent_types(
    rng_key: Array,
    n_games: int,
    opponent_mix: dict,
) -> Array:
    """Sample opponent types for each game based on mix probabilities.

    Args:
        rng_key: PRNG key
        n_games: Number of games
        opponent_mix: Dict mapping opponent name to probability

    Returns:
        [N] array of opponent type indices (0=self, 1=historical, 2=random, etc.)
    """
    # Build probability array in order: self, historical, random, call_station, tag, lag, rock, trapper, value_bettor
    probs = jnp.array([
        opponent_mix.get("self", 0.0),
        opponent_mix.get("historical", 0.0),  # v3.2
        opponent_mix.get("random", 0.0),
        opponent_mix.get("call_station", 0.0),
        opponent_mix.get("tag", 0.0),
        opponent_mix.get("lag", 0.0),
        opponent_mix.get("rock", 0.0),  # v3.3
        opponent_mix.get("trapper", 0.0),  # v3.3: anti-aggro
        opponent_mix.get("value_bettor", 0.0),  # v3.5: exploits never-fold
    ])
    # Normalize
    probs = probs / probs.sum()

    # Sample using categorical
    logits = jnp.log(probs + 1e-10)  # Avoid log(0)
    logits = jnp.broadcast_to(logits, (n_games, 9))  # 9 opponent types (v3.5)
    opponent_types = jax.random.categorical(rng_key, logits)

    return opponent_types


@partial(jax.jit, static_argnums=(5, 8))
def get_mixed_opponent_actions(
    state: GameState,
    valid_mask: Array,
    obs: Array,
    opponent_types: Array,
    rng_key: Array,
    network: "ActorCriticMLP",
    params: dict,
    historical_params: dict,
    use_historical: bool = False,
) -> Array:
    """Get actions for mixed opponents based on assigned types.

    Args:
        state: Current game state [N games]
        valid_mask: Valid action mask [N, NUM_ACTIONS]
        obs: Observation for current player [N, obs_dim]
        opponent_types: Opponent type per game [N]
        rng_key: PRNG key
        network: Network for self-play
        params: Network params for self-play
        historical_params: Params from historical checkpoint (always a valid dict)
        use_historical: Static flag - if True, compute separate forward pass for historical

    Returns:
        [N] action indices
    """
    n_games = state.done.shape[0]

    # Split RNG for each opponent type (9 types in v3.5)
    keys = jrandom.split(rng_key, 9)
    key_self, key_hist, key_random, key_call, key_tag, key_lag, key_rock, key_trapper, key_value = keys

    # Compute actions for each opponent type
    # Self: use network with current params
    action_logits, _, _ = network.apply({"params": params}, obs, training=False)
    self_actions, _ = sample_action(key_self, action_logits, valid_mask)
    self_actions = self_actions + 1  # Convert to game actions (1-indexed)

    # Historical: only compute separate forward pass if we have actual historical params
    # This is a static branch - JAX will trace two versions (use_historical=True/False)
    if use_historical:
        hist_logits, _, _ = network.apply({"params": historical_params}, obs, training=False)
        hist_actions, _ = sample_action(key_hist, hist_logits, valid_mask)
        hist_actions = hist_actions + 1
    else:
        # No historical checkpoint yet - reuse self actions (saves one forward pass)
        hist_actions = self_actions

    # Random opponent
    random_actions = random_opponent(state, valid_mask, key_random)

    # Call station
    call_actions = call_station_opponent(state, valid_mask, key_call)

    # TAG (needs obs)
    tag_actions = tag_opponent(state, valid_mask, key_tag, obs)

    # LAG (needs obs)
    lag_actions = lag_opponent(state, valid_mask, key_lag, obs)

    # Rock (needs obs) - v3.3
    rock_actions = rock_opponent(state, valid_mask, key_rock, obs)

    # Trapper (needs obs) - v3.3: anti-aggro opponent
    trapper_actions = trapper_opponent(state, valid_mask, key_trapper, obs)

    # Value bettor (needs obs) - v3.5: only bets with premium hands (exploits never-fold)
    value_actions = value_bettor_opponent(state, valid_mask, key_value, obs)

    # Select based on opponent type
    # opponent_types: 0=self, 1=historical, 2=random, 3=call_station, 4=tag, 5=lag, 6=rock, 7=trapper, 8=value_bettor
    actions = jnp.where(
        opponent_types == OPP_SELF, self_actions,
        jnp.where(
            opponent_types == OPP_HISTORICAL, hist_actions,
            jnp.where(
                opponent_types == OPP_RANDOM, random_actions,
                jnp.where(
                    opponent_types == OPP_CALL_STATION, call_actions,
                    jnp.where(
                        opponent_types == OPP_TAG, tag_actions,
                        jnp.where(
                            opponent_types == OPP_LAG, lag_actions,
                            jnp.where(
                                opponent_types == OPP_ROCK, rock_actions,
                                jnp.where(
                                    opponent_types == OPP_TRAPPER, trapper_actions,
                                    value_actions  # Default to value_bettor
                                )
                            )
                        )
                    )
                )
            )
        )
    )

    return actions


@dataclass
class TrainingState:
    """Mutable training state."""

    params: dict
    opt_state: optax.OptState
    rng_key: Array
    total_steps: int = 0
    total_updates: int = 0
    total_games: int = 0

    # Rolling metrics
    win_rate: float = 0.0
    avg_reward: float = 0.0


@dataclass
class TrainingMetrics:
    """Aggregated training metrics."""

    steps: int = 0
    games_completed: int = 0
    avg_reward: float = 0.0
    avg_game_length: float = 0.0

    # PPO metrics
    policy_loss: float = 0.0
    value_loss: float = 0.0
    entropy: float = 0.0
    approx_kl: float = 0.0

    # Performance
    steps_per_second: float = 0.0
    games_per_second: float = 0.0

    # Poker behavior metrics
    wins: int = 0
    losses: int = 0  # Track losses for accurate win rate
    total_pot_won: float = 0.0
    action_counts: dict = field(default_factory=lambda: {
        "fold": 0, "check": 0, "call": 0,
        "raise_33": 0, "raise_66": 0, "raise_100": 0, "raise_150": 0,
        "all_in": 0
    })


class JAXTrainer:
    """JAX-accelerated RL trainer for poker."""

    def __init__(
        self,
        network: ActorCriticMLP | None = None,
        ppo_config: PPOConfig | None = None,
        training_config: JAXTrainingConfig | None = None,
        seed: int = 42,
        console: Console | None = None,
    ) -> None:
        self.console = console or Console()
        self.ppo_config = ppo_config or PPOConfig()
        self.training_config = training_config or JAXTrainingConfig()

        # Initialize RNG
        self.rng_key = jrandom.PRNGKey(seed)

        # Create network
        if network is None:
            network = create_network("mlp")
        self.network = network

        # Initialize parameters
        self.rng_key, init_key = jrandom.split(self.rng_key)
        self.params = init_network(network, init_key, OBS_DIM)

        # Create optimizer
        self.optimizer = create_optimizer(self.ppo_config)
        self.opt_state = self.optimizer.init(self.params)

        # Logger (TensorBoard + CSV for plotting)
        self.logger: MetricsLogger | None = None
        self.file_logger: logging.Logger | None = None
        if self.training_config.tensorboard_dir:
            log_dir = Path(self.training_config.tensorboard_dir)
            self.logger = MetricsLogger(
                log_dir,
                use_tensorboard=True,
                use_csv=True,  # Enable CSV for plotting
            )
            # File logger for training progress
            log_file = log_dir / "training.log"
            file_handler = logging.FileHandler(log_file, mode="w")
            file_handler.setFormatter(
                logging.Formatter("%(asctime)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
            )
            self.file_logger = logging.getLogger(f"jax_trainer_{id(self)}")
            self.file_logger.setLevel(logging.INFO)
            self.file_logger.addHandler(file_handler)

        # Print info
        num_params = count_parameters(self.params)
        self.console.print(f"\n[bold blue]JAX Poker Trainer[/bold blue]")
        self.console.print(f"Network parameters: {num_params:,}")
        self.console.print(f"Parallel games: {self.training_config.num_parallel_games:,}")
        self.console.print(f"Steps per update: {self.training_config.steps_per_update:,}")

        # Mixed opponents info
        if self.training_config.use_mixed_opponents:
            self.console.print(f"[bold green]Mixed Opponent Training (v3.1)[/bold green]")
            mix = self.training_config.opponent_mix
            self.console.print(f"  Opponent mix: {', '.join(f'{k}={v:.0%}' for k,v in mix.items())}")

        # Historical self-play (v3.2)
        self.checkpoint_pool: CheckpointPool | None = None
        if self.training_config.use_historical_selfplay:
            self.checkpoint_pool = CheckpointPool(
                max_size=self.training_config.historical_pool_size
            )
            self.console.print(f"[bold cyan]Historical Self-Play (v3.2)[/bold cyan]")

            # Dynamic scheduling info (v3.4)
            if self.training_config.use_dynamic_opponent_schedule and self.training_config.opponent_schedule:
                self.console.print(f"  [bold yellow]Dynamic Opponent Scheduling (v3.4)[/bold yellow]")
                for step, mix in self.training_config.opponent_schedule:
                    self.console.print(f"    @{step/1e6:.0f}M: self={mix.get('self',0):.0%}, hist={mix.get('historical',0):.0%}, trap={mix.get('trapper',0):.0%}")
            else:
                mix = self.training_config.historical_opponent_mix
                self.console.print(f"  Opponent mix: {', '.join(f'{k}={v:.0%}' for k,v in mix.items())}")

            self.console.print(f"  Pool size: {self.training_config.historical_pool_size}")
            self.console.print(f"  Save every: {self.training_config.historical_save_every:,} steps")

    @staticmethod
    @partial(jax.jit, static_argnums=(0,))
    def _collect_step(
        network: ActorCriticMLP,
        params: dict,
        state: GameState,
        rng_key: Array,
    ) -> tuple[GameState, Array, Array, Array, Array, Array, Array]:
        """Collect one step of experience from all parallel games.

        Returns:
            Tuple of (new_state, obs, actions, log_probs, values, rewards, dones)
        """
        n_games = state.done.shape[0]

        # Encode state for current player
        obs = encode_state_for_current_player(state)  # [N, obs_dim]

        # Get valid actions
        valid_mask = get_valid_actions_from_obs(obs)  # [N, 5]
        # Pad to 6 actions (index 0 is unused)
        valid_mask_full = jnp.concatenate([
            jnp.zeros((n_games, 1)),  # Index 0 unused
            valid_mask,
        ], axis=1)

        # Forward pass
        action_logits, bet_frac, values = network.apply(
            {"params": params}, obs, training=False
        )
        values = values.squeeze(-1)

        # Sample actions
        rng_key, sample_key = jrandom.split(rng_key)
        actions, log_probs = sample_action(sample_key, action_logits, valid_mask)

        # Convert to game actions (add 1 for offset)
        game_actions = actions + 1  # 0->1 (fold), 1->2 (check), etc.

        # Compute raise amounts for raise actions
        # Default: min-raise
        opp_idx = 1 - state.current_player
        game_idx = jnp.arange(n_games)
        opp_bet = state.bets[game_idx, opp_idx]
        raise_amounts = opp_bet + state.last_raise_amount

        # Step environment
        new_state = step(state, game_actions, raise_amounts)

        # Get rewards
        rewards = get_rewards(new_state)
        # Get reward for current player (before step)
        player_rewards = rewards[game_idx, state.current_player]

        # Done flags
        dones = new_state.done.astype(jnp.float32)

        return new_state, obs, actions, log_probs, values, player_rewards, dones, valid_mask

    @staticmethod
    @partial(jax.jit, static_argnums=(0, 6))
    def _collect_step_mixed(
        network: ActorCriticMLP,
        params: dict,
        state: GameState,
        opponent_types: Array,
        rng_key: Array,
        historical_params: dict,
        use_historical: bool = False,
    ) -> tuple[GameState, Array, Array, Array, Array, Array, Array, Array, Array]:
        """Collect one step with mixed opponents (JIT-compiled).

        Model is always player 0. Opponents are player 1.

        Args:
            network: Network architecture (static)
            params: Current model params
            state: Game state
            opponent_types: Opponent type per game
            rng_key: PRNG key
            historical_params: Historical checkpoint params (always a valid dict)
            use_historical: Static flag - if True, use separate historical forward pass

        Returns:
            Tuple of (new_state, obs, actions, log_probs, values, rewards, dones, valid_mask, model_turn_mask)
        """
        n_games = state.done.shape[0]

        # Encode state for current player
        obs = encode_state_for_current_player(state)
        valid_mask = get_valid_actions_from_obs(obs)

        # Whose turn is it?
        is_model_turn = state.current_player == 0

        # Model forward pass (always compute for value estimates)
        action_logits, _, values = network.apply(
            {"params": params}, obs, training=False
        )
        values = values.squeeze(-1)

        # Sample model actions
        rng_key, model_key, opp_key = jrandom.split(rng_key, 3)
        model_actions, log_probs = sample_action(model_key, action_logits, valid_mask)

        # Get opponent actions based on type
        # Note: valid_mask from encoding is the full 8-action mask
        # Opponents expect this format (already in correct format)
        opponent_actions = get_mixed_opponent_actions(
            state, valid_mask, obs, opponent_types,
            opp_key, network, params, historical_params, use_historical
        )

        # Select action based on whose turn
        # Model actions are 0-indexed (need +1 for game), opponent actions are already 1-indexed
        game_actions = jnp.where(
            is_model_turn,
            model_actions + 1,  # Model uses 0-indexed, convert to 1-indexed
            opponent_actions    # Opponents already return 1-indexed
        )

        # For training, we use the model's actions (0-indexed for trajectory)
        # When it's opponent's turn, we still store model's hypothetical action
        # This is a simplification - we'll mask these in PPO loss
        trajectory_actions = model_actions

        # Compute raise amounts
        opp_idx = 1 - state.current_player
        game_idx = jnp.arange(n_games)
        opp_bet = state.bets[game_idx, opp_idx]
        raise_amounts = opp_bet + state.last_raise_amount

        # Step environment
        new_state = step(state, game_actions, raise_amounts)

        # Get rewards
        rewards = get_rewards(new_state)
        # Model's reward (always player 0)
        model_rewards = rewards[:, 0]

        # Done flags
        dones = new_state.done.astype(jnp.float32)

        # Create mask for model's turns (used to filter training data)
        model_turn_mask = is_model_turn.astype(jnp.float32)

        return new_state, obs, trajectory_actions, log_probs, values, model_rewards, dones, valid_mask, model_turn_mask

    def collect_rollout(
        self,
        num_steps: int,
        current_step: int = 0,
    ) -> tuple[Trajectory, TrainingMetrics]:
        """Collect rollout data from parallel games.

        Args:
            num_steps: Number of steps to collect
            current_step: Current global training step (for dynamic opponent scheduling)

        Returns:
            Tuple of (trajectory, metrics)
        """
        config = self.training_config
        n_games = config.num_parallel_games

        # Initialize games
        self.rng_key, reset_key = jrandom.split(self.rng_key)
        state = reset(
            reset_key, n_games,
            config.starting_chips, config.small_blind, config.big_blind
        )

        # Storage
        all_obs = []
        all_actions = []
        all_log_probs = []
        all_values = []
        all_rewards = []
        all_dones = []
        all_valid_masks = []

        total_rewards = 0.0
        games_completed = 0
        total_game_steps = 0

        # Poker behavior tracking
        wins = 0
        losses = 0
        total_pot_won = 0.0
        # v3: 9 actions (indices 1-8, 0 is ACTION_NONE)
        action_counts = {
            "fold": 0, "check": 0, "call": 0,
            "raise_33": 0, "raise_66": 0, "raise_100": 0, "raise_150": 0,
            "all_in": 0
        }
        # Map action index to name (ACTION_FOLD=1, ..., ACTION_ALL_IN=8)
        action_idx_to_name = {
            1: "fold", 2: "check", 3: "call",
            4: "raise_33", 5: "raise_66", 6: "raise_100", 7: "raise_150",
            8: "all_in"
        }

        start_time = time.time()

        # Determine training mode
        use_historical = config.use_historical_selfplay
        use_mixed = config.use_mixed_opponents or use_historical

        # Setup opponent types and historical params
        # Always pass valid params dict to maintain pytree structure (prevents JAX retracing)
        historical_params = self.params  # Default to current params
        historical_elo = INITIAL_ELO
        use_different_historical = False  # Static flag: True when using actual historical checkpoint

        if use_historical:
            # Historical self-play mode (v3.2) with optional dynamic scheduling (v3.4)
            self.rng_key, opp_key = jrandom.split(self.rng_key)

            # Get opponent mix (dynamic or static)
            if config.use_dynamic_opponent_schedule and config.opponent_schedule:
                opp_mix = interpolate_opponent_mix(current_step, config.opponent_schedule)
            else:
                opp_mix = config.historical_opponent_mix

            opponent_types = sample_opponent_types(opp_key, n_games, opp_mix)

            # Sample historical opponent if pool has checkpoints
            if self.checkpoint_pool and self.checkpoint_pool.has_checkpoints():
                self.rng_key, hist_key = jrandom.split(self.rng_key)
                historical_params, _, historical_elo = self.checkpoint_pool.sample_opponent(hist_key)
                use_different_historical = True  # Now using actual historical checkpoint
        elif use_mixed:
            # Mixed opponent mode (v3.1)
            self.rng_key, opp_key = jrandom.split(self.rng_key)
            opp_mix = config.opponent_mix  # Static mix for non-historical mode
            opponent_types = sample_opponent_types(opp_key, n_games, opp_mix)

        for step_idx in range(num_steps):
            # Collect step
            self.rng_key, step_key = jrandom.split(self.rng_key)

            if use_mixed:
                # Mixed/historical opponent mode: model is player 0, opponents are player 1
                new_state, obs, actions, log_probs, values, rewards, dones, valid_mask, model_mask = \
                    self._collect_step_mixed(self.network, self.params, state, opponent_types, step_key, historical_params, use_different_historical)
            else:
                # Self-play mode: both players use network
                new_state, obs, actions, log_probs, values, rewards, dones, valid_mask = \
                    self._collect_step(self.network, self.params, state, step_key)

            # Store
            all_obs.append(obs)
            all_actions.append(actions)
            all_log_probs.append(log_probs)
            all_values.append(values)
            all_rewards.append(rewards)
            all_dones.append(dones)
            all_valid_masks.append(valid_mask)

            # Track metrics
            completed = dones.sum()
            games_completed += int(completed)
            total_rewards += float(rewards.sum())
            total_game_steps += int(n_games - completed)

            # Track action distribution (only for model's actions in mixed mode)
            if use_mixed:
                # In mixed mode, actions are 0-indexed from model perspective
                # Only count when it was model's turn
                model_turn = state.current_player == 0
                for idx, name in action_idx_to_name.items():
                    # Convert from 0-indexed to 1-indexed for comparison
                    action_counts[name] += int(((actions == (idx - 1)) & model_turn).sum())
            else:
                # Self-play: actions are 0-indexed
                actions_np = jnp.asarray(actions)
                for idx, name in action_idx_to_name.items():
                    action_counts[name] += int((actions_np == (idx - 1)).sum())

            # Track wins and losses
            done_mask = dones > 0.5
            won_games = (rewards > 0) & done_mask
            lost_games = (rewards < 0) & done_mask
            wins += int(won_games.sum())
            losses += int(lost_games.sum())
            # Sum positive rewards for pot won tracking
            total_pot_won += float(jnp.where(won_games, rewards, 0.0).sum())

            # Reset completed games
            if completed > 0:
                self.rng_key, reset_key = jrandom.split(self.rng_key)
                fresh_state = reset(
                    reset_key, n_games,
                    config.starting_chips, config.small_blind, config.big_blind
                )
                # Keep unfinished games, replace finished ones
                def select_state(fresh, old):
                    done_mask = new_state.done
                    # Reshape done mask to broadcast with array dimensions
                    for _ in range(old.ndim - 1):
                        done_mask = done_mask[:, None]
                    return jnp.where(done_mask, fresh, old)

                state = jax.tree_util.tree_map(select_state, fresh_state, new_state)

                # Resample opponent types for reset games (mixed/historical mode)
                if use_mixed:
                    self.rng_key, opp_key = jrandom.split(self.rng_key)
                    # Use same opp_mix as initial (already computed with dynamic scheduling)
                    new_opp_types = sample_opponent_types(opp_key, n_games, opp_mix)
                    opponent_types = jnp.where(new_state.done, new_opp_types, opponent_types)
            else:
                state = new_state

        elapsed = time.time() - start_time

        # Stack trajectory
        trajectory = Trajectory(
            obs=jnp.stack(all_obs),  # [T, N, obs_dim]
            actions=jnp.stack(all_actions),  # [T, N]
            log_probs=jnp.stack(all_log_probs),  # [T, N]
            values=jnp.stack(all_values),  # [T, N]
            rewards=jnp.stack(all_rewards),  # [T, N]
            dones=jnp.stack(all_dones),  # [T, N]
            valid_masks=jnp.stack(all_valid_masks),  # [T, N, 5]
        )

        # Compute metrics
        total_steps = num_steps * n_games
        metrics = TrainingMetrics(
            steps=total_steps,
            games_completed=games_completed,
            avg_reward=total_rewards / max(games_completed, 1),
            avg_game_length=total_game_steps / max(games_completed, 1),
            steps_per_second=total_steps / max(elapsed, 0.001),
            games_per_second=games_completed / max(elapsed, 0.001),
            # Poker behavior
            wins=wins,
            losses=losses,
            total_pot_won=total_pot_won,
            action_counts=action_counts,
        )

        return trajectory, metrics

    def train(
        self,
        callback: Callable[[int, TrainingMetrics, PPOMetrics], None] | None = None,
        show_progress: bool = True,
    ) -> TrainingMetrics:
        """Run the full training loop.

        Args:
            callback: Optional callback(steps, metrics, ppo_metrics) after each update
            show_progress: Show progress bar

        Returns:
            Final training metrics
        """
        config = self.training_config
        total_updates = config.total_steps // config.steps_per_update

        checkpoint_dir = Path(config.checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.console.print(f"\n[bold]Starting training...[/bold]")
        self.console.print(f"Total steps: {config.total_steps:,}")
        self.console.print(f"Updates: {total_updates:,}")

        iterator = range(total_updates)
        if show_progress:
            # mininterval=5.0 reduces terminal I/O overhead (update display max every 5 seconds)
            iterator = tqdm(iterator, desc="Training", unit="updates", mininterval=5.0)

        global_step = 0
        total_games = 0
        start_time = time.time()

        try:
            for update_idx in iterator:
                # Collect rollout (pass current step for dynamic opponent scheduling)
                steps_per_game = config.steps_per_update // config.num_parallel_games
                trajectory, collect_metrics = self.collect_rollout(steps_per_game, current_step=global_step)

                global_step += collect_metrics.steps
                total_games += collect_metrics.games_completed

                # PPO update
                self.rng_key, update_key = jrandom.split(self.rng_key)
                self.params, self.opt_state, ppo_metrics = ppo_update(
                    self.network,
                    self.params,
                    self.opt_state,
                    self.optimizer,
                    trajectory,
                    self.ppo_config,
                    update_key,
                )

                # Historical self-play: update checkpoint pool and ELO
                if config.use_historical_selfplay and self.checkpoint_pool:
                    # Add checkpoint to pool periodically
                    if global_step > 0 and global_step % config.historical_save_every < config.steps_per_update:
                        self.checkpoint_pool.add_checkpoint(self.params, global_step)
                        if self.file_logger:
                            pool_stats = self.checkpoint_pool.get_pool_stats()
                            self.file_logger.info(
                                f"  [POOL] Added checkpoint at step {global_step:,} | "
                                f"Pool size: {pool_stats['pool_size']} | "
                                f"ELO range: {pool_stats['min_elo']:.0f}-{pool_stats['max_elo']:.0f}"
                            )

                    # Update ELO based on game outcomes (simplified: use win rate vs expected)
                    # Record wins/losses for ELO update
                    decided_games = collect_metrics.wins + collect_metrics.losses
                    if decided_games > 0:
                        # Assume ~40% of games are vs historical opponents
                        hist_games = int(decided_games * config.historical_opponent_mix.get("historical", 0.4))
                        hist_wins = int(collect_metrics.wins * config.historical_opponent_mix.get("historical", 0.4))
                        for _ in range(hist_wins):
                            self.checkpoint_pool.record_game_result(True, self.checkpoint_pool.current_elo)
                        for _ in range(hist_games - hist_wins):
                            self.checkpoint_pool.record_game_result(False, self.checkpoint_pool.current_elo)

                    # Batch update ELO every 1000 updates
                    if (update_idx + 1) % 1000 == 0:
                        self.checkpoint_pool.update_elo_batch()

                # Logging - only update progress bar every 100 updates to reduce terminal I/O
                if show_progress and (update_idx + 1) % 100 == 0:
                    iterator.set_postfix(
                        rew=f"{collect_metrics.avg_reward:.2f}",
                        pol=f"{ppo_metrics.policy_loss:.4f}",
                        val=f"{ppo_metrics.value_loss:.4f}",
                        ent=f"{ppo_metrics.entropy:.3f}",
                        sps=f"{collect_metrics.steps_per_second:.0f}",
                    )

                if self.logger and (update_idx + 1) % config.log_every == 0:
                    # Compute derived metrics
                    ac = collect_metrics.action_counts
                    total_actions = sum(ac.values())
                    # Win rate: when I ended the game, how often did I win?
                    decided_games = collect_metrics.wins + collect_metrics.losses
                    win_rate = collect_metrics.wins / max(decided_games, 1)
                    fold_rate = ac["fold"] / max(total_actions, 1)
                    # v3: Sum all raise types for aggression
                    raise_count = ac["raise_33"] + ac["raise_66"] + ac["raise_100"] + ac["raise_150"] + ac["all_in"]
                    call_count = ac["call"]
                    aggression = raise_count / max(call_count, 1)

                    self.logger.log(global_step, {
                        # Rewards
                        "reward/avg": collect_metrics.avg_reward,
                        "reward/games_completed": collect_metrics.games_completed,
                        # Loss
                        "loss/policy": ppo_metrics.policy_loss,
                        "loss/value": ppo_metrics.value_loss,
                        "loss/entropy": ppo_metrics.entropy,
                        # PPO diagnostics
                        "ppo/approx_kl": ppo_metrics.approx_kl,
                        "ppo/clip_fraction": ppo_metrics.clip_fraction,
                        # RL diagnostics
                        "rl/explained_variance": ppo_metrics.explained_variance,
                        "rl/grad_norm": ppo_metrics.grad_norm,
                        "rl/value_pred_error": ppo_metrics.value_pred_error,
                        # Performance
                        "perf/steps_per_second": collect_metrics.steps_per_second,
                        "perf/games_per_second": collect_metrics.games_per_second,
                        # Poker behavior
                        "poker/win_rate": win_rate,
                        "poker/avg_pot_won": collect_metrics.total_pot_won / max(collect_metrics.wins, 1),
                        "poker/fold_rate": fold_rate,
                        "poker/aggression": aggression,
                        # v3: All 8 action types
                        "poker/action_fold": ac["fold"] / max(total_actions, 1),
                        "poker/action_check": ac["check"] / max(total_actions, 1),
                        "poker/action_call": ac["call"] / max(total_actions, 1),
                        "poker/action_raise_33": ac["raise_33"] / max(total_actions, 1),
                        "poker/action_raise_66": ac["raise_66"] / max(total_actions, 1),
                        "poker/action_raise_100": ac["raise_100"] / max(total_actions, 1),
                        "poker/action_raise_150": ac["raise_150"] / max(total_actions, 1),
                        "poker/action_allin": ac["all_in"] / max(total_actions, 1),
                    })

                    # Add ELO metrics for historical self-play (v3.2)
                    if config.use_historical_selfplay and self.checkpoint_pool:
                        pool_stats = self.checkpoint_pool.get_pool_stats()
                        self.logger.log(global_step, {
                            "elo/current": self.checkpoint_pool.current_elo,
                            "elo/gain": self.checkpoint_pool.current_elo - INITIAL_ELO,
                            "elo/pool_size": pool_stats["pool_size"],
                            "elo/pool_avg": pool_stats["avg_elo"],
                        })

                # File logging (persist progress for long runs) - reduced frequency
                if self.file_logger and update_idx % 1000 == 0:
                    decided_games = collect_metrics.wins + collect_metrics.losses
                    win_rate = collect_metrics.wins / max(decided_games, 1)
                    total_actions = sum(collect_metrics.action_counts.values())

                    # Add ELO to log line if historical self-play
                    elo_str = ""
                    if config.use_historical_selfplay and self.checkpoint_pool:
                        elo_str = f" | elo={self.checkpoint_pool.current_elo:>6.0f}"

                    self.file_logger.info(
                        f"step={global_step:>8} | rew={collect_metrics.avg_reward:>7.3f} | "
                        f"win={win_rate:>5.1%} | pol={ppo_metrics.policy_loss:>7.4f} | "
                        f"ev={ppo_metrics.explained_variance:>5.2f} | sps={collect_metrics.steps_per_second:>5.0f}{elo_str}"
                    )
                    # Debug: action breakdown and game outcomes (v3: 8 action types)
                    ac = collect_metrics.action_counts
                    total_raises = ac['raise_33'] + ac['raise_66'] + ac['raise_100'] + ac['raise_150']
                    self.file_logger.info(
                        f"  actions: F={ac['fold']:>4} Ch={ac['check']:>4} Ca={ac['call']:>4} "
                        f"R33={ac['raise_33']:>3} R66={ac['raise_66']:>3} R100={ac['raise_100']:>3} R150={ac['raise_150']:>3} A={ac['all_in']:>4} | "
                        f"games: {collect_metrics.games_completed} done, {collect_metrics.wins}W/{collect_metrics.losses}L"
                    )

                # Checkpointing
                if global_step % config.checkpoint_every == 0:
                    ckpt_path = checkpoint_dir / f"step_{global_step:08d}.pkl"
                    self.save_checkpoint(ckpt_path)

                # Callback
                if callback:
                    callback(global_step, collect_metrics, ppo_metrics)

        except KeyboardInterrupt:
            self.console.print("\n[yellow]Training interrupted.[/yellow]")

        # Final checkpoint
        final_path = checkpoint_dir.parent / "final.pkl"
        self.save_checkpoint(final_path)

        elapsed = time.time() - start_time
        self.console.print(f"\n[green]Training complete![/green]")
        self.console.print(f"Total time: {elapsed / 60:.1f} minutes")
        self.console.print(f"Total steps: {global_step:,}")
        self.console.print(f"Total games: {total_games:,}")
        self.console.print(f"Avg steps/sec: {global_step / elapsed:.0f}")

        return TrainingMetrics(
            steps=global_step,
            games_completed=total_games,
            steps_per_second=global_step / elapsed,
        )

    def save_checkpoint(self, path: Path | str) -> None:
        """Save checkpoint."""
        import pickle

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        checkpoint = {
            "params": self.params,
            "opt_state": self.opt_state,
            "ppo_config": self.ppo_config,
            "training_config": self.training_config,
        }

        with open(path, "wb") as f:
            pickle.dump(checkpoint, f)

    def load_checkpoint(self, path: Path | str) -> None:
        """Load checkpoint."""
        import pickle

        with open(path, "rb") as f:
            checkpoint = pickle.load(f)

        self.params = checkpoint["params"]
        self.opt_state = checkpoint["opt_state"]


def create_jax_trainer(
    seed: int = 42,
    num_parallel_games: int = 1024,
    total_steps: int = 1_000_000,
    learning_rate: float = 3e-4,
    tensorboard_dir: str | None = "logs/jax",
) -> JAXTrainer:
    """Factory function to create a JAX trainer.

    Args:
        seed: Random seed
        num_parallel_games: Number of parallel games
        total_steps: Total training steps
        learning_rate: Learning rate
        tensorboard_dir: TensorBoard log directory

    Returns:
        Configured JAXTrainer
    """
    ppo_config = PPOConfig(learning_rate=learning_rate)
    training_config = JAXTrainingConfig(
        num_parallel_games=num_parallel_games,
        total_steps=total_steps,
        tensorboard_dir=tensorboard_dir,
    )

    return JAXTrainer(
        ppo_config=ppo_config,
        training_config=training_config,
        seed=seed,
    )
