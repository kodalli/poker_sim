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
    ActorCriticMLPWithOpponentModel,
    OpponentLSTM,
    create_network,
    OPPONENT_ACTION_DIM,
    OPPONENT_LSTM_HIDDEN,
    init_network,
    sample_action,
    count_parameters,
)
from poker_jax.encoding import (
    OBS_DIM,
    get_valid_actions_from_obs,
    encode_opponent_action_from_state,
)
from training.jax_ppo import (
    PPOConfig,
    PPOMetrics,
    Trajectory,
    create_optimizer,
    ppo_update,
    ppo_update_sequential,
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
    starting_chips: int = 400  # 200BB for deeper stack play (v8)
    small_blind: int = 1
    big_blind: int = 2

    # Mixed opponent training (v3.1, updated v11)
    use_mixed_opponents: bool = False
    opponent_mix: dict = field(default_factory=lambda: {
        "self": 0.40,           # 40% self-play
        "random": 0.10,         # 10% random
        "call_station": 0.10,   # 10% call station
        "tag": 0.10,            # 10% tight-aggressive
        "lag": 0.10,            # 10% loose-aggressive
        "rock": 0.07,           # 7% rock (tight-passive)
        "trapper": 0.07,        # 7% trapper (anti-aggro)
        "value_bettor": 0.06,   # 6% value bettor (exploits never-fold)
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

    # Adversarial self-play (v4) - train against learned exploiter
    # Replaces all fixed opponents with a co-evolving exploiter model
    use_adversarial_training: bool = False
    exploiter_update_freq: int = 1  # Update exploiter every main update (symmetric co-evolution)
    adversarial_main_checkpoint: str | None = None  # Load main model from this checkpoint (exploiter starts random)
    adversarial_historical_mix: float = 0.5  # Fraction of games vs historical (0=pure adversarial, 1=pure historical)
    adversarial_baseline_mix: float = 0.3  # Fraction of games vs rule-based opponents (random, call_station, tag)
    fold_dropout_rate: float = 0.05  # Force fold action this % of time when facing a bet (exploration)

    # Eval checkpoints during training (early detection of degenerate strategies)
    eval_checkpoint_every: int = 0  # Steps between evaluations (0=disabled). Recommended: 10_000_000
    eval_checkpoint_games: int = 2000  # Games per opponent for checkpoint eval

    # === v9 Opponent Modeling ===
    use_opponent_model: bool = False  # Enable LSTM-based opponent modeling
    opponent_lstm_hidden: int = 64  # LSTM hidden dimension
    opponent_embed_dim: int = 32  # Opponent embedding dimension
    bptt_window_size: int = 32  # Truncated BPTT window size for sequential processing


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
            if self.training_config.use_opponent_model:
                # v9: Use opponent modeling network
                network = ActorCriticMLPWithOpponentModel(
                    opponent_lstm_hidden=self.training_config.opponent_lstm_hidden,
                    opponent_embed_dim=self.training_config.opponent_embed_dim,
                )
            else:
                network = create_network("mlp")
        self.network = network

        # Initialize parameters
        self.rng_key, init_key = jrandom.split(self.rng_key)
        if self.training_config.use_opponent_model:
            # v9: Initialize with opponent action and carry inputs
            n_games = self.training_config.num_parallel_games
            dummy_obs = jnp.zeros((1, OBS_DIM))
            dummy_opp_action = jnp.zeros((1, OPPONENT_ACTION_DIM))
            dummy_carry = OpponentLSTM.init_carry(1, self.training_config.opponent_lstm_hidden)
            variables = network.init(init_key, dummy_obs, dummy_opp_action, dummy_carry, training=False)
            self.params = variables["params"]
        else:
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

        # Adversarial training (v4) - exploiter model
        self.exploiter_params = None
        self.exploiter_opt_state = None
        if self.training_config.use_adversarial_training:
            # Load main model from checkpoint if provided (warm start)
            if self.training_config.adversarial_main_checkpoint:
                import pickle
                ckpt_path = Path(self.training_config.adversarial_main_checkpoint)
                if ckpt_path.exists():
                    with open(ckpt_path, "rb") as f:
                        checkpoint = pickle.load(f)
                    self.params = checkpoint["params"]
                    self.opt_state = self.optimizer.init(self.params)  # Fresh optimizer state
                    self.console.print(f"[bold green]Loaded main model from {ckpt_path}[/bold green]")
                else:
                    self.console.print(f"[yellow]Warning: Checkpoint {ckpt_path} not found, using random init[/yellow]")

            # Initialize exploiter with random weights (will learn to exploit main)
            self.rng_key, exp_init_key = jrandom.split(self.rng_key)
            self.exploiter_params = init_network(network, exp_init_key, OBS_DIM)
            self.exploiter_opt_state = self.optimizer.init(self.exploiter_params)

            # Initialize historical pool for hybrid training (v4 + historical)
            if self.training_config.adversarial_historical_mix > 0:
                self.checkpoint_pool = CheckpointPool(
                    max_size=self.training_config.historical_pool_size
                )
                # Seed pool with main model's initial state
                self.checkpoint_pool.add_checkpoint(self.params, 0)

            hist_mix = self.training_config.adversarial_historical_mix
            baseline_mix = self.training_config.adversarial_baseline_mix
            exploiter_mix = max(0.0, 1.0 - hist_mix - baseline_mix)
            self.console.print(f"[bold magenta]Adversarial Training (v5)[/bold magenta]")
            self.console.print(f"  Opponent mix: {exploiter_mix:.0%} exploiter, {hist_mix:.0%} historical, {baseline_mix:.0%} baseline")
            if baseline_mix > 0:
                self.console.print(f"  Baselines: random, call_station, tag (equal split)")
            self.console.print(f"  Exploiter update freq: every {self.training_config.exploiter_update_freq} main update(s)")
        elif self.training_config.use_historical_selfplay:
            # Only show opponent scheduling if NOT in adversarial mode
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
    @partial(jax.jit, static_argnums=(0, 5))
    def _collect_step(
        network: ActorCriticMLP | ActorCriticMLPWithOpponentModel,
        params: dict,
        state: GameState,
        rng_key: Array,
        opp_action: Array,
        use_opponent_model: bool = False,
    ) -> tuple[GameState, Array, Array, Array, Array, Array, Array, Array, tuple[Array, Array] | None]:
        """Collect one step of experience from all parallel games.

        Args:
            network: Network (ActorCriticMLP or ActorCriticMLPWithOpponentModel)
            params: Network parameters
            state: Current game state
            rng_key: PRNG key
            opp_action: [N, OPPONENT_ACTION_DIM] last opponent action encoding
            use_opponent_model: Static flag - if True, network expects opponent model inputs

        Returns:
            Tuple of (new_state, obs, actions, log_probs, values, rewards, dones, valid_mask, new_carry)
            new_carry is None if not using opponent model
        """
        n_games = state.done.shape[0]

        # Encode state for current player
        obs = encode_state_for_current_player(state)  # [N, obs_dim]

        # Get valid actions
        valid_mask = get_valid_actions_from_obs(obs)  # [N, 9] (includes ACTION_NONE at index 0)

        # Forward pass (conditional on use_opponent_model)
        if use_opponent_model:
            # v9: Use opponent modeling network
            lstm_carry = (state.opp_lstm_hidden, state.opp_lstm_cell)
            action_logits, bet_frac, values, new_carry = network.apply(
                {"params": params}, obs, opp_action, lstm_carry, training=False
            )
        else:
            # Standard network
            action_logits, bet_frac, values = network.apply(
                {"params": params}, obs, training=False
            )
            new_carry = None

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

        return new_state, obs, actions, log_probs, values, player_rewards, dones, valid_mask, new_carry

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

    @staticmethod
    @partial(jax.jit, static_argnums=(0,))
    def _adversarial_step_core(
        network: ActorCriticMLP,
        main_params: dict,
        exploiter_params: dict,
        state: GameState,
        rng_key: Array,
    ) -> tuple[GameState, Array, Array, Array, Array, Array, Array, Array, Array, Array]:
        """Core adversarial step - returns data for both players.

        Both players use neural network policies.
        - Player 0 = Main model
        - Player 1 = Exploiter

        Returns:
            Tuple of (new_state, obs, actions, log_probs, main_values, exp_values,
                     rewards_p0, rewards_p1, dones, valid_mask)
        """
        n_games = state.done.shape[0]

        # Encode state for current player
        obs = encode_state_for_current_player(state)
        valid_mask = get_valid_actions_from_obs(obs)

        # Whose turn is it?
        is_player0_turn = state.current_player == 0  # Main's turn

        # Main forward pass
        main_logits, _, main_values = network.apply(
            {"params": main_params}, obs, training=False
        )
        main_values = main_values.squeeze(-1)

        # Exploiter forward pass
        exp_logits, _, exp_values = network.apply(
            {"params": exploiter_params}, obs, training=False
        )
        exp_values = exp_values.squeeze(-1)

        # Select logits based on whose turn (but return both values for PPO)
        action_logits = jnp.where(
            is_player0_turn[:, None],
            main_logits,
            exp_logits
        )

        # Sample actions
        rng_key, action_key = jrandom.split(rng_key)
        actions, log_probs = sample_action(action_key, action_logits, valid_mask)

        # Game actions (convert 0-indexed to 1-indexed)
        game_actions = actions + 1

        # Compute raise amounts
        opp_idx = 1 - state.current_player
        game_idx = jnp.arange(n_games)
        opp_bet = state.bets[game_idx, opp_idx]
        raise_amounts = opp_bet + state.last_raise_amount

        # Step environment
        new_state = step(state, game_actions, raise_amounts)

        # Get rewards for both players
        rewards = get_rewards(new_state)
        rewards_p0 = rewards[:, 0]
        rewards_p1 = rewards[:, 1]

        # Done flags
        dones = new_state.done.astype(jnp.float32)

        return (new_state, obs, actions, log_probs, main_values, exp_values,
                rewards_p0, rewards_p1, dones, valid_mask)

    @staticmethod
    @partial(jax.jit, static_argnums=(0, 5))
    def _adversarial_step_core_hybrid(
        network: ActorCriticMLP,
        main_params: dict,
        exploiter_params: dict,
        historical_params: dict,
        opponent_types: Array,  # [N] int: 0=exploiter, 1=historical, 2=random, 3=call_station, 4=tag
        fold_dropout_rate: float,  # Force fold this % of time when facing bet
        state: GameState,
        rng_key: Array,
    ) -> tuple[GameState, Array, Array, Array, Array, Array, Array, Array, Array, Array]:
        """Core hybrid adversarial step - mixes historical, exploiter, and baseline opponents.

        - Player 0 = Main model (always)
        - Player 1 = Exploiter, Historical, or Baseline (random/call_station/tag)

        Returns:
            Tuple of (new_state, obs, actions, log_probs, main_values, exp_values,
                     rewards_p0, rewards_p1, dones, valid_mask)
        """
        n_games = state.done.shape[0]

        # Encode state for current player
        obs = encode_state_for_current_player(state)
        valid_mask = get_valid_actions_from_obs(obs)

        # Whose turn is it?
        is_player0_turn = state.current_player == 0  # Main's turn

        # Main forward pass
        main_logits, _, main_values = network.apply(
            {"params": main_params}, obs, training=False
        )
        main_values = main_values.squeeze(-1)

        # Exploiter forward pass
        exp_logits, _, exp_values = network.apply(
            {"params": exploiter_params}, obs, training=False
        )
        exp_values = exp_values.squeeze(-1)

        # Historical forward pass
        hist_logits, _, _ = network.apply(
            {"params": historical_params}, obs, training=False
        )

        # Baseline opponent actions (rule-based, computed for all games but only used for some)
        rng_key, baseline_key = jrandom.split(rng_key)
        random_actions = random_opponent(state, valid_mask, baseline_key)
        call_station_actions = call_station_opponent(state, valid_mask, baseline_key)
        tag_actions = tag_opponent(state, valid_mask, baseline_key, obs)

        # Select opponent logits/actions based on game type
        # Start with exploiter as default
        opponent_logits = exp_logits

        # Override with historical for those games
        opponent_logits = jnp.where(
            (opponent_types == OPP_HISTORICAL)[:, None],
            hist_logits,
            opponent_logits
        )

        # Select logits based on whose turn
        # Player 0 (main) uses main_logits, Player 1 (opponent) uses opponent_logits
        action_logits = jnp.where(
            is_player0_turn[:, None],
            main_logits,
            opponent_logits
        )

        # Fold dropout: occasionally force fold when facing a bet (exploration)
        # This helps the model learn fold values since fold/loss have same reward
        rng_key, dropout_key = jrandom.split(rng_key)
        can_call = valid_mask[:, ACTION_CALL] > 0.5  # Facing a bet
        apply_dropout = jrandom.uniform(dropout_key, (n_games,)) < fold_dropout_rate
        force_fold = can_call & apply_dropout & is_player0_turn  # Only for main model

        # When force_fold, only allow fold action
        fold_only_mask = jnp.zeros_like(valid_mask).at[:, ACTION_FOLD].set(1.0)
        effective_valid_mask = jnp.where(
            force_fold[:, None],
            fold_only_mask,
            valid_mask
        )

        # Sample actions from network logits
        rng_key, action_key = jrandom.split(rng_key)
        actions, log_probs = sample_action(action_key, action_logits, effective_valid_mask)

        # Override with baseline opponent actions when it's opponent's turn and using baseline
        # Baseline actions are already 1-indexed (ACTION_FOLD=1, etc.), need to convert to 0-indexed
        is_opponent_turn = ~is_player0_turn
        use_random = is_opponent_turn & (opponent_types == OPP_RANDOM)
        use_call_station = is_opponent_turn & (opponent_types == OPP_CALL_STATION)
        use_tag = is_opponent_turn & (opponent_types == OPP_TAG)

        # Convert baseline actions to 0-indexed (they return 1-indexed game actions)
        actions = jnp.where(use_random, random_actions - 1, actions)
        actions = jnp.where(use_call_station, call_station_actions - 1, actions)
        actions = jnp.where(use_tag, tag_actions - 1, actions)

        # Game actions (convert 0-indexed to 1-indexed)
        game_actions = actions + 1

        # Compute raise amounts
        opp_idx = 1 - state.current_player
        game_idx = jnp.arange(n_games)
        opp_bet = state.bets[game_idx, opp_idx]
        raise_amounts = opp_bet + state.last_raise_amount

        # Step environment
        new_state = step(state, game_actions, raise_amounts)

        # Get rewards for both players
        rewards = get_rewards(new_state)
        rewards_p0 = rewards[:, 0]
        rewards_p1 = rewards[:, 1]

        # Done flags
        dones = new_state.done.astype(jnp.float32)

        return (new_state, obs, actions, log_probs, main_values, exp_values,
                rewards_p0, rewards_p1, dones, valid_mask)

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
        all_model_masks = []  # v3.6: Track which steps are model's turns (for mixed opponent training)

        # v9: Opponent model storage (only used if use_opponent_model=True)
        # Note: LSTM states are NOT stored - they're recomputed during training to save GPU memory
        all_opp_actions = []

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
                new_carry = None  # v9 not supported for mixed opponents yet
            else:
                # Self-play mode: both players use network (all turns are "model" turns)
                if config.use_opponent_model:
                    # v9: Pass opponent action and LSTM carry
                    new_state, obs, actions, log_probs, values, rewards, dones, valid_mask, new_carry = \
                        self._collect_step(self.network, self.params, state, step_key, state.last_opp_action, use_opponent_model=True)
                else:
                    # Legacy: no opponent modeling
                    opp_action_dummy = jnp.zeros((n_games, OPPONENT_ACTION_DIM))
                    new_state, obs, actions, log_probs, values, rewards, dones, valid_mask, new_carry = \
                        self._collect_step(self.network, self.params, state, step_key, opp_action_dummy, use_opponent_model=False)
                model_mask = jnp.ones(n_games, dtype=jnp.float32)  # All turns count in self-play

            # Store
            all_obs.append(obs)
            all_actions.append(actions)
            all_log_probs.append(log_probs)
            all_values.append(values)
            all_rewards.append(rewards)
            all_dones.append(dones)
            all_valid_masks.append(valid_mask)
            all_model_masks.append(model_mask)  # v3.6: Store model turn mask

            # v9: Store opponent actions (LSTM states no longer stored - recomputed during training)
            if config.use_opponent_model and new_carry is not None:
                all_opp_actions.append(state.last_opp_action)

                # Update LSTM carry in state (new_carry from forward pass)
                # This happens when MODEL acts, so we update the carry
                state = state._replace(
                    opp_lstm_hidden=new_carry[0],
                    opp_lstm_cell=new_carry[1],
                )

            # v9: After step, if opponent acted, encode their action
            if config.use_opponent_model and not use_mixed:
                # Determine who just acted by comparing current_player before/after step
                # If current_player changed, opponent just acted
                opponent_acted = new_state.current_player != state.current_player

                # Get opponent's action (the action that was just taken)
                # game_actions is 1-indexed, convert to 0-indexed for encoding
                opp_action_type = jnp.where(
                    opponent_acted,
                    actions,  # actions is 0-indexed from sample_action
                    jnp.zeros_like(actions)
                )

                # Get opponent's bet amount
                # After step, new_state has the updated bets
                opp_idx = 1 - state.current_player
                game_idx = jnp.arange(n_games)
                opp_bet_amount = new_state.bets[game_idx, opp_idx]

                # Encode opponent action
                encoded_opp_action = encode_opponent_action_from_state(
                    new_state, opp_action_type, opp_bet_amount, state.current_player
                )

                # Update state.last_opp_action where opponent acted
                new_state = new_state._replace(
                    last_opp_action=jnp.where(
                        opponent_acted[:, None],
                        encoded_opp_action,
                        new_state.last_opp_action
                    )
                )

            # Track metrics
            completed = dones.sum()
            games_completed += int(completed)
            total_rewards += float(rewards.sum())
            total_game_steps += int(n_games - completed)

            # Track action distribution (only for model's actions in mixed mode)
            if use_mixed:
                # Actions are 1-indexed (ACTION_FOLD=1, ACTION_CHECK=2, etc.)
                # Only count when it was model's turn
                model_turn = state.current_player == 0
                for idx, name in action_idx_to_name.items():
                    action_counts[name] += int(((actions == idx) & model_turn).sum())
            else:
                # Self-play: actions are 1-indexed (ACTION_FOLD=1, etc.)
                actions_np = jnp.asarray(actions)
                for idx, name in action_idx_to_name.items():
                    action_counts[name] += int((actions_np == idx).sum())

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
        # v9: Include opponent model fields if enabled
        if config.use_opponent_model and len(all_opp_actions) > 0:
            trajectory = Trajectory(
                obs=jnp.stack(all_obs),  # [T, N, obs_dim]
                actions=jnp.stack(all_actions),  # [T, N]
                log_probs=jnp.stack(all_log_probs),  # [T, N]
                values=jnp.stack(all_values),  # [T, N]
                rewards=jnp.stack(all_rewards),  # [T, N]
                dones=jnp.stack(all_dones),  # [T, N]
                valid_masks=jnp.stack(all_valid_masks),  # [T, N, 9]
                model_masks=jnp.stack(all_model_masks),  # [T, N] - v3.6: 1.0 when model acted, 0.0 on opponent turns
                # v9 fields (LSTM states recomputed during training, not stored)
                opp_actions=jnp.stack(all_opp_actions),  # [T, N, OPPONENT_ACTION_DIM]
            )
        else:
            # Legacy trajectory (no opponent model)
            trajectory = Trajectory(
                obs=jnp.stack(all_obs),  # [T, N, obs_dim]
                actions=jnp.stack(all_actions),  # [T, N]
                log_probs=jnp.stack(all_log_probs),  # [T, N]
                values=jnp.stack(all_values),  # [T, N]
                rewards=jnp.stack(all_rewards),  # [T, N]
                dones=jnp.stack(all_dones),  # [T, N]
                valid_masks=jnp.stack(all_valid_masks),  # [T, N, 9]
                model_masks=jnp.stack(all_model_masks),  # [T, N] - v3.6: 1.0 when model acted, 0.0 on opponent turns
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

    def _create_opponent_model_scan_fn(self):
        """Create JIT-compiled scan function for opponent model rollout.

        Returns:
            JIT-compiled scan function for self-play with opponent modeling.
        """
        network = self.network
        config = self.training_config
        n_games = config.num_parallel_games
        starting_chips = config.starting_chips
        small_blind = config.small_blind
        big_blind = config.big_blind

        @jax.jit
        def scan_step(carry, _):
            """Single step in opponent model rollout (self-play).

            Inlined step logic to avoid nested JIT compilation issues.
            """
            state, rng_key, params = carry

            # Split RNG
            rng_key, step_key, reset_key = jrandom.split(rng_key, 3)

            # Get opponent action from state
            opp_action = state.last_opp_action

            # === INLINED STEP LOGIC (from _collect_step) ===
            # Encode state for current player
            obs = encode_state_for_current_player(state)

            # Get valid actions
            valid_mask = get_valid_actions_from_obs(obs)

            # Forward pass with opponent model
            lstm_carry = (state.opp_lstm_hidden, state.opp_lstm_cell)
            action_logits, bet_frac, values, new_carry = network.apply(
                {"params": params}, obs, opp_action, lstm_carry, training=False
            )
            values = values.squeeze(-1)

            # Sample actions
            step_key, sample_key = jrandom.split(step_key)
            actions, log_probs = sample_action(sample_key, action_logits, valid_mask)

            # Convert to game actions (add 1 for offset)
            game_actions = actions + 1  # 0->1 (fold), 1->2 (check), etc.

            # Compute raise amounts for raise actions
            game_idx = jnp.arange(n_games)
            opp_idx = 1 - state.current_player
            opp_bet = state.bets[game_idx, opp_idx]
            raise_amounts = opp_bet + state.last_raise_amount

            # Step environment
            new_state = step(state, game_actions, raise_amounts)

            # Get rewards for current player (before step)
            rewards = get_rewards(new_state)
            player_rewards = rewards[game_idx, state.current_player]

            # Done flags
            dones = new_state.done.astype(jnp.float32)
            # === END INLINED STEP LOGIC ===

            # Encode opponent action after step (for next iteration)
            # The action taken by "us" is the opponent's action from their perspective
            new_opp_bet = new_state.bets[game_idx, opp_idx]
            encoded_opp_action = encode_opponent_action_from_state(
                new_state, game_actions, new_opp_bet, state.current_player
            )

            # Update state with new LSTM carry and opponent action
            new_state = new_state._replace(
                opp_lstm_hidden=new_carry[0],
                opp_lstm_cell=new_carry[1],
                last_opp_action=encoded_opp_action,
            )

            # Reset completed games
            fresh_state = reset(
                reset_key, n_games, starting_chips, small_blind, big_blind
            )

            def select_state(fresh, old):
                done_mask = new_state.done
                for _ in range(old.ndim - 1):
                    done_mask = done_mask[:, None]
                return jnp.where(done_mask, fresh, old)

            next_state = jax.tree_util.tree_map(select_state, fresh_state, new_state)

            # Store LSTM state for trajectory (state BEFORE this step's update)
            lstm_hidden = state.opp_lstm_hidden
            lstm_cell = state.opp_lstm_cell

            # Output for this step
            step_data = (
                obs, actions, log_probs, values, player_rewards, dones, valid_mask,
                opp_action, lstm_hidden, lstm_cell
            )

            return (next_state, rng_key, params), step_data

        return scan_step

    def collect_rollout_opponent_model(
        self,
        num_steps: int,
    ) -> tuple[Trajectory, TrainingMetrics]:
        """Collect rollout for opponent model training (self-play with LSTM).

        Uses JIT-compiled jax.lax.scan for fast GPU execution.

        Args:
            num_steps: Number of steps to collect

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

        start_time = time.time()

        # Get or create cached scan function
        if not hasattr(self, "_opp_model_scan_cache"):
            self._opp_model_scan_cache = self._create_opponent_model_scan_fn()
        scan_fn = self._opp_model_scan_cache

        # Run scan (fully JIT-compiled loop on GPU)
        self.rng_key, loop_key = jrandom.split(self.rng_key)
        carry = (state, loop_key, self.params)
        _, step_data = jax.lax.scan(scan_fn, carry, None, length=num_steps)

        # Unpack step data
        (all_obs, all_actions, all_log_probs, all_values, all_rewards, all_dones,
         all_valid_masks, all_opp_actions, all_lstm_hidden, all_lstm_cell) = step_data

        elapsed = time.time() - start_time

        # Build trajectory with v9 fields
        trajectory = Trajectory(
            obs=all_obs,  # [T, N, obs_dim]
            actions=all_actions,  # [T, N]
            log_probs=all_log_probs,  # [T, N]
            values=all_values,  # [T, N]
            rewards=all_rewards,  # [T, N]
            dones=all_dones,  # [T, N]
            valid_masks=all_valid_masks,  # [T, N, 9]
            model_masks=None,  # Self-play: all steps are model's
            opp_actions=all_opp_actions,  # [T, N, opp_action_dim]
            lstm_hidden=all_lstm_hidden,  # [T, N, lstm_hidden_dim]
            lstm_cell=all_lstm_cell,  # [T, N, lstm_hidden_dim]
        )

        # Compute metrics (done on CPU after scan completes)
        total_steps = num_steps * n_games
        games_completed = int(all_dones.sum())
        total_rewards = float(all_rewards.sum())

        # Action counts
        action_idx_to_name = {
            1: "fold", 2: "check", 3: "call",
            4: "raise_33", 5: "raise_66", 6: "raise_100", 7: "raise_150",
            8: "all_in"
        }
        action_counts = {}
        for idx, name in action_idx_to_name.items():
            action_counts[name] = int((all_actions == idx).sum())

        # Win/loss tracking
        done_mask = all_dones > 0.5
        won_games = (all_rewards > 0) & done_mask
        lost_games = (all_rewards < 0) & done_mask
        wins = int(won_games.sum())
        losses = int(lost_games.sum())
        total_pot_won = float(jnp.where(won_games, all_rewards, 0.0).sum())

        metrics = TrainingMetrics(
            steps=total_steps,
            games_completed=games_completed,
            avg_reward=total_rewards / max(games_completed, 1),
            avg_game_length=total_steps / max(games_completed, 1),
            steps_per_second=total_steps / max(elapsed, 0.001),
            games_per_second=games_completed / max(elapsed, 0.001),
            wins=wins,
            losses=losses,
            total_pot_won=total_pot_won,
            action_counts=action_counts,
        )

        return trajectory, metrics

    def _create_adversarial_scan_fn(self, train_main: bool):
        """Create JIT-compiled scan function for adversarial rollout.

        Args:
            train_main: If True, collect from main's perspective; else exploiter's

        Returns:
            JIT-compiled scan function
        """
        network = self.network
        config = self.training_config

        @jax.jit
        def scan_step(carry, _):
            """Single step in adversarial rollout."""
            state, rng_key, main_params, exp_params = carry

            # Split RNG
            rng_key, step_key, reset_key = jrandom.split(rng_key, 3)

            # Run core step (both forward passes)
            (new_state, obs, actions, log_probs, main_values, exp_values,
             rewards_p0, rewards_p1, dones, valid_mask) = self._adversarial_step_core(
                network, main_params, exp_params, state, step_key
            )

            # Select values and rewards based on who is learning
            is_player0_turn = state.current_player == 0
            if train_main:
                # Main is learning: use main's values, p0's rewards, mask for p0's turns
                values = main_values
                learner_rewards = rewards_p0
                learner_mask = is_player0_turn.astype(jnp.float32)
            else:
                # Exploiter is learning: use exp's values, p1's rewards, mask for p1's turns
                values = exp_values
                learner_rewards = rewards_p1
                learner_mask = (state.current_player == 1).astype(jnp.float32)

            # Reset completed games
            fresh_state = reset(
                reset_key, config.num_parallel_games,
                config.starting_chips, config.small_blind, config.big_blind
            )

            def select_state(fresh, old):
                done_mask = new_state.done
                for _ in range(old.ndim - 1):
                    done_mask = done_mask[:, None]
                return jnp.where(done_mask, fresh, old)

            next_state = jax.tree_util.tree_map(select_state, fresh_state, new_state)

            # Output for this step
            step_data = (obs, actions, log_probs, values, learner_rewards, dones, valid_mask, learner_mask)

            return (next_state, rng_key, main_params, exp_params), step_data

        return scan_step

    def collect_rollout_adversarial(
        self,
        num_steps: int,
        train_main: bool = True,
    ) -> tuple[Trajectory, TrainingMetrics]:
        """Collect rollout for adversarial training (main vs exploiter).

        Uses JIT-compiled jax.lax.scan for fast GPU execution.

        Args:
            num_steps: Number of steps to collect
            train_main: If True, collect from main's perspective; else from exploiter's

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

        start_time = time.time()

        # Get or create cached scan function
        cache_key = ("adversarial", train_main)
        if not hasattr(self, "_adversarial_scan_cache"):
            self._adversarial_scan_cache = {}
        if cache_key not in self._adversarial_scan_cache:
            self._adversarial_scan_cache[cache_key] = self._create_adversarial_scan_fn(train_main)
        scan_fn = self._adversarial_scan_cache[cache_key]

        # Run scan (fully JIT-compiled loop on GPU)
        self.rng_key, loop_key = jrandom.split(self.rng_key)
        carry = (state, loop_key, self.params, self.exploiter_params)
        _, step_data = jax.lax.scan(scan_fn, carry, None, length=num_steps)

        # Unpack step data
        all_obs, all_actions, all_log_probs, all_values, all_rewards, all_dones, all_valid_masks, all_learner_masks = step_data

        elapsed = time.time() - start_time

        # Build trajectory
        trajectory = Trajectory(
            obs=all_obs,  # [T, N, obs_dim]
            actions=all_actions,  # [T, N]
            log_probs=all_log_probs,  # [T, N]
            values=all_values,  # [T, N]
            rewards=all_rewards,  # [T, N]
            dones=all_dones,  # [T, N]
            valid_masks=all_valid_masks,  # [T, N, 9]
            model_masks=all_learner_masks,  # [T, N]
        )

        # Compute metrics (done on CPU after scan completes)
        total_steps = num_steps * n_games
        games_completed = int(all_dones.sum())
        total_rewards = float(all_rewards.sum())

        # Action counts (computed once at end)
        action_idx_to_name = {
            1: "fold", 2: "check", 3: "call",
            4: "raise_33", 5: "raise_66", 6: "raise_100", 7: "raise_150",
            8: "all_in"
        }
        action_counts = {}
        learner_turns = all_learner_masks > 0.5
        for idx, name in action_idx_to_name.items():
            action_counts[name] = int(((all_actions == idx) & learner_turns).sum())

        # Win/loss tracking
        done_mask = all_dones > 0.5
        won_games = (all_rewards > 0) & done_mask
        lost_games = (all_rewards < 0) & done_mask
        wins = int(won_games.sum())
        losses = int(lost_games.sum())
        total_pot_won = float(jnp.where(won_games, all_rewards, 0.0).sum())

        metrics = TrainingMetrics(
            steps=total_steps,
            games_completed=games_completed,
            avg_reward=total_rewards / max(games_completed, 1),
            avg_game_length=total_steps / max(games_completed, 1),
            steps_per_second=total_steps / max(elapsed, 0.001),
            games_per_second=games_completed / max(elapsed, 0.001),
            wins=wins,
            losses=losses,
            total_pot_won=total_pot_won,
            action_counts=action_counts,
        )

        return trajectory, metrics

    def _create_adversarial_hybrid_scan_fn(self):
        """Create JIT-compiled scan function for hybrid adversarial rollout.

        Historical params are passed through the carry to enable caching.

        Returns:
            JIT-compiled scan function
        """
        network = self.network
        config = self.training_config
        hist_mix = config.adversarial_historical_mix
        baseline_mix = config.adversarial_baseline_mix
        n_games = config.num_parallel_games

        # Pre-compute opponent types for each game
        # Split: [exploiter games] [historical games] [random] [call_station] [tag]
        # Remaining after hist and baseline goes to exploiter
        exploiter_frac = max(0.0, 1.0 - hist_mix - baseline_mix)
        n_exploiter = int(n_games * exploiter_frac)
        n_historical = int(n_games * hist_mix)
        n_baseline = n_games - n_exploiter - n_historical

        # Split baseline equally among random, call_station, tag
        n_random = n_baseline // 3
        n_call_station = n_baseline // 3
        n_tag = n_baseline - n_random - n_call_station

        # Create opponent_types array: 0=exploiter, 1=historical, 2=random, 3=call_station, 4=tag
        opponent_types = jnp.concatenate([
            jnp.full(n_exploiter, OPP_SELF, dtype=jnp.int32),  # 0 = exploiter (same as self for this context)
            jnp.full(n_historical, OPP_HISTORICAL, dtype=jnp.int32),
            jnp.full(n_random, OPP_RANDOM, dtype=jnp.int32),
            jnp.full(n_call_station, OPP_CALL_STATION, dtype=jnp.int32),
            jnp.full(n_tag, OPP_TAG, dtype=jnp.int32),
        ])

        @jax.jit
        def scan_step(carry, _):
            """Single step in hybrid adversarial rollout."""
            state, rng_key, main_params, exp_params, hist_params = carry

            # Split RNG
            rng_key, step_key, reset_key = jrandom.split(rng_key, 3)

            # Run hybrid core step (three forward passes + baseline opponents)
            (new_state, obs, actions, log_probs, main_values, exp_values,
             rewards_p0, rewards_p1, dones, valid_mask) = self._adversarial_step_core_hybrid(
                network, main_params, exp_params, hist_params,
                opponent_types, config.fold_dropout_rate, state, step_key
            )

            # Main is always learning: use main's values, p0's rewards
            values = main_values
            learner_rewards = rewards_p0
            is_player0_turn = state.current_player == 0
            learner_mask = is_player0_turn.astype(jnp.float32)

            # Reset completed games
            fresh_state = reset(
                reset_key, config.num_parallel_games,
                config.starting_chips, config.small_blind, config.big_blind
            )

            def select_state(fresh, old):
                done_mask = new_state.done
                for _ in range(old.ndim - 1):
                    done_mask = done_mask[:, None]
                return jnp.where(done_mask, fresh, old)

            next_state = jax.tree_util.tree_map(select_state, fresh_state, new_state)

            # Output for this step
            step_data = (obs, actions, log_probs, values, learner_rewards, dones, valid_mask, learner_mask)

            return (next_state, rng_key, main_params, exp_params, hist_params), step_data

        return scan_step

    def collect_rollout_adversarial_hybrid(
        self,
        num_steps: int,
    ) -> tuple[Trajectory, TrainingMetrics]:
        """Collect rollout for hybrid adversarial training (main vs historical + exploiter).

        Uses JIT-compiled jax.lax.scan for fast GPU execution.
        50% of games play against historical checkpoints, 50% against exploiter.

        Args:
            num_steps: Number of steps to collect

        Returns:
            Tuple of (trajectory, metrics)
        """
        config = self.training_config
        n_games = config.num_parallel_games

        # Sample historical opponent from pool
        if hasattr(self, 'checkpoint_pool') and len(self.checkpoint_pool.checkpoints) > 0:
            self.rng_key, sample_key = jrandom.split(self.rng_key)
            historical_params, _, _ = self.checkpoint_pool.sample_opponent(sample_key)
        else:
            # Fallback to self-play if pool is empty
            historical_params = self.params

        # Initialize games
        self.rng_key, reset_key = jrandom.split(self.rng_key)
        state = reset(
            reset_key, n_games,
            config.starting_chips, config.small_blind, config.big_blind
        )

        start_time = time.time()

        # Get or create cached scan function (historical_params passed via carry for caching)
        if not hasattr(self, "_hybrid_scan_cache"):
            self._hybrid_scan_cache = self._create_adversarial_hybrid_scan_fn()
        scan_fn = self._hybrid_scan_cache

        # Run scan (fully JIT-compiled loop on GPU)
        # Historical params passed through carry to enable JIT caching
        self.rng_key, loop_key = jrandom.split(self.rng_key)
        carry = (state, loop_key, self.params, self.exploiter_params, historical_params)
        _, step_data = jax.lax.scan(scan_fn, carry, None, length=num_steps)

        # Unpack step data
        all_obs, all_actions, all_log_probs, all_values, all_rewards, all_dones, all_valid_masks, all_learner_masks = step_data

        elapsed = time.time() - start_time

        # Build trajectory
        trajectory = Trajectory(
            obs=all_obs,
            actions=all_actions,
            log_probs=all_log_probs,
            values=all_values,
            rewards=all_rewards,
            dones=all_dones,
            valid_masks=all_valid_masks,
            model_masks=all_learner_masks,
        )

        # Compute metrics
        total_steps = num_steps * n_games
        games_completed = int(all_dones.sum())
        total_rewards = float(all_rewards.sum())

        action_idx_to_name = {
            1: "fold", 2: "check", 3: "call",
            4: "raise_33", 5: "raise_66", 6: "raise_100", 7: "raise_150",
            8: "all_in"
        }
        action_counts = {}
        learner_turns = all_learner_masks > 0.5
        for idx, name in action_idx_to_name.items():
            action_counts[name] = int(((all_actions == idx) & learner_turns).sum())

        done_mask = all_dones > 0.5
        won_games = (all_rewards > 0) & done_mask
        lost_games = (all_rewards < 0) & done_mask
        wins = int(won_games.sum())
        losses = int(lost_games.sum())
        total_pot_won = float(jnp.where(won_games, all_rewards, 0.0).sum())

        metrics = TrainingMetrics(
            steps=total_steps,
            games_completed=games_completed,
            avg_reward=total_rewards / max(games_completed, 1),
            avg_game_length=total_steps / max(games_completed, 1),
            steps_per_second=total_steps / max(elapsed, 0.001),
            games_per_second=games_completed / max(elapsed, 0.001),
            wins=wins,
            losses=losses,
            total_pot_won=total_pot_won,
            action_counts=action_counts,
        )

        return trajectory, metrics

    def _run_eval_checkpoint(
        self,
        num_games: int = 2000,
    ) -> dict[str, dict[str, float]]:
        """Run lightweight evaluation against key opponents during training.

        Tests against random and call_station to detect degenerate strategies.

        Args:
            num_games: Games per opponent (split between positions)

        Returns:
            Dict mapping opponent name to {win_rate, bb_per_100}
        """
        config = self.training_config
        results = {}

        opponents = [
            ("random", random_opponent),
            ("call_station", call_station_opponent),
        ]

        for opp_name, opp_fn in opponents:
            # Run evaluation games
            games_per_position = num_games // 2
            total_wins = 0
            total_losses = 0
            total_chip_delta = 0.0

            for model_position in [0, 1]:
                # Initialize games
                self.rng_key, reset_key = jrandom.split(self.rng_key)
                state = reset(reset_key, games_per_position, config.starting_chips)

                # Initialize LSTM state for opponent model
                if config.use_opponent_model:
                    lstm_hidden = jnp.zeros((games_per_position, OPPONENT_LSTM_HIDDEN), dtype=jnp.float32)
                    lstm_cell = jnp.zeros((games_per_position, OPPONENT_LSTM_HIDDEN), dtype=jnp.float32)
                    last_opp_action = jnp.zeros((games_per_position, OPPONENT_ACTION_DIM), dtype=jnp.float32)

                # Run until all games done
                max_steps = 500  # Safety limit
                for _ in range(max_steps):
                    if state.done.all():
                        break

                    # Get observation
                    obs = encode_state_for_current_player(state)
                    valid_mask = get_valid_actions_from_obs(obs)

                    # Determine who acts
                    is_model_turn = state.current_player == model_position

                    # Model action
                    self.rng_key, action_key = jrandom.split(self.rng_key)
                    if config.use_opponent_model:
                        lstm_carry = (lstm_hidden, lstm_cell)
                        model_logits, _, _, new_carry = self.network.apply(
                            {"params": self.params}, obs, last_opp_action, lstm_carry, training=False
                        )
                    else:
                        model_logits, _, _ = self.network.apply(
                            {"params": self.params}, obs, training=False
                        )
                    model_actions, _ = sample_action(action_key, model_logits, valid_mask)
                    # Convert from 1-indexed to 0-indexed for step()
                    model_actions = model_actions - 1

                    # Opponent action
                    self.rng_key, opp_key = jrandom.split(self.rng_key)
                    if opp_name in NEEDS_OBS:
                        opp_actions = opp_fn(state, valid_mask, opp_key, obs) - 1
                    else:
                        opp_actions = opp_fn(state, valid_mask, opp_key) - 1

                    # Select action based on current player
                    actions = jnp.where(is_model_turn[:, None], model_actions[:, None], opp_actions[:, None]).squeeze(-1)

                    # Step
                    state = step(state, actions)

                    # Update LSTM state for opponent model
                    if config.use_opponent_model:
                        # Update LSTM carry where model acted
                        lstm_hidden = jnp.where(is_model_turn[:, None], new_carry[0], lstm_hidden)
                        lstm_cell = jnp.where(is_model_turn[:, None], new_carry[1], lstm_cell)

                        # Encode opponent action where opponent acted
                        opp_acted = ~is_model_turn
                        # opp_actions is 0-indexed, encode expects action type (also 0-indexed internally)
                        opp_action_type = jnp.where(opp_acted, opp_actions + 1, jnp.zeros_like(opp_actions))
                        game_idx = jnp.arange(games_per_position)
                        opp_player_idx = 1 - model_position  # Opponent is always the other player
                        opp_bet = state.bets[game_idx, opp_player_idx]

                        # Encode from model's perspective (model_position is current player)
                        current_player_arr = jnp.full(games_per_position, model_position, dtype=jnp.int32)
                        encoded_opp = encode_opponent_action_from_state(
                            state, opp_action_type, opp_bet, current_player_arr
                        )
                        last_opp_action = jnp.where(opp_acted[:, None], encoded_opp, last_opp_action)

                # Get results
                rewards = get_rewards(state)
                model_rewards = rewards[:, model_position]

                total_wins += int((model_rewards > 0).sum())
                total_losses += int((model_rewards < 0).sum())
                total_chip_delta += float(model_rewards.sum())

            # Compute metrics
            decided = total_wins + total_losses
            win_rate = total_wins / max(decided, 1)
            # BB/100 = (chips won / hands) * 100 / big_blind
            bb_per_100 = (total_chip_delta / num_games) * 100 / 2  # big_blind = 2

            results[opp_name] = {
                "win_rate": win_rate,
                "bb_per_100": bb_per_100,
                "wins": total_wins,
                "losses": total_losses,
            }

        return results

    def train(
        self,
        callback: Callable[[int, TrainingMetrics, PPOMetrics], None] | None = None,
        show_progress: bool = True,
        start_step: int = 0,
    ) -> TrainingMetrics:
        """Run the full training loop.

        Args:
            callback: Optional callback(steps, metrics, ppo_metrics) after each update
            show_progress: Show progress bar
            start_step: Starting step count (for resuming from checkpoint)

        Returns:
            Final training metrics
        """
        config = self.training_config
        total_updates = config.total_steps // config.steps_per_update
        start_update = start_step // config.steps_per_update

        checkpoint_dir = Path(config.checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.console.print(f"\n[bold]Starting training...[/bold]")
        if start_step > 0:
            self.console.print(f"[cyan]Resuming from step {start_step:,} (update {start_update:,})[/cyan]")
        self.console.print(f"Total steps: {config.total_steps:,}")
        self.console.print(f"Updates: {total_updates:,} ({total_updates - start_update:,} remaining)")

        iterator = range(start_update, total_updates)
        if show_progress:
            # mininterval=5.0 reduces terminal I/O overhead (update display max every 5 seconds)
            iterator = tqdm(iterator, desc="Training", unit="updates", mininterval=5.0,
                           initial=start_update, total=total_updates)

        global_step = start_step
        total_games = 0
        start_time = time.time()

        try:
            for update_idx in iterator:
                steps_per_game = config.steps_per_update // config.num_parallel_games

                if config.use_adversarial_training:
                    # === ADVERSARIAL TRAINING (v4) ===
                    # Alternating: train main, then train exploiter

                    # Phase A: Train main model vs opponents
                    if config.adversarial_historical_mix > 0:
                        # HYBRID MODE: Mix historical + exploiter opponents
                        trajectory_main, collect_metrics = self.collect_rollout_adversarial_hybrid(
                            steps_per_game
                        )
                    else:
                        # PURE ADVERSARIAL: Main vs exploiter only
                        trajectory_main, collect_metrics = self.collect_rollout_adversarial(
                            steps_per_game, train_main=True
                        )

                    self.rng_key, update_key = jrandom.split(self.rng_key)
                    self.params, self.opt_state, ppo_metrics = ppo_update(
                        self.network,
                        self.params,
                        self.opt_state,
                        self.optimizer,
                        trajectory_main,
                        self.ppo_config,
                        update_key,
                    )

                    # Phase B: Train exploiter vs frozen main (every N updates)
                    exp_collect_metrics = None
                    exp_ppo_metrics = None
                    if (update_idx + 1) % config.exploiter_update_freq == 0:
                        trajectory_exp, exp_collect_metrics = self.collect_rollout_adversarial(
                            steps_per_game, train_main=False
                        )

                        self.rng_key, exp_update_key = jrandom.split(self.rng_key)
                        self.exploiter_params, self.exploiter_opt_state, exp_ppo_metrics = ppo_update(
                            self.network,
                            self.exploiter_params,
                            self.exploiter_opt_state,
                            self.optimizer,
                            trajectory_exp,
                            self.ppo_config,
                            exp_update_key,
                        )

                    global_step += collect_metrics.steps
                    total_games += collect_metrics.games_completed

                else:
                    # === LEGACY TRAINING (v1-v3) / v9 OPPONENT MODEL ===
                    if config.use_opponent_model:
                        # v9: Use scan-optimized rollout collection for opponent model
                        trajectory, collect_metrics = self.collect_rollout_opponent_model(steps_per_game)
                    else:
                        # Legacy: Collect rollout (pass current step for dynamic opponent scheduling)
                        trajectory, collect_metrics = self.collect_rollout(steps_per_game, current_step=global_step)

                    global_step += collect_metrics.steps
                    total_games += collect_metrics.games_completed

                    # PPO update
                    self.rng_key, update_key = jrandom.split(self.rng_key)

                    # v9: Use sequential PPO if opponent modeling is enabled
                    if config.use_opponent_model:
                        self.params, self.opt_state, ppo_metrics = ppo_update_sequential(
                            self.network,
                            self.params,
                            self.opt_state,
                            self.optimizer,
                            trajectory,
                            self.ppo_config,
                            update_key,
                            window_size=config.bptt_window_size,
                        )
                    else:
                        # Standard PPO (no LSTM)
                        self.params, self.opt_state, ppo_metrics = ppo_update(
                            self.network,
                            self.params,
                            self.opt_state,
                            self.optimizer,
                            trajectory,
                            self.ppo_config,
                            update_key,
                        )

                # Hybrid adversarial: update checkpoint pool
                if config.use_adversarial_training and config.adversarial_historical_mix > 0:
                    if hasattr(self, 'checkpoint_pool') and self.checkpoint_pool:
                        # Add checkpoint to pool periodically
                        if global_step > 0 and global_step % config.historical_save_every < config.steps_per_update:
                            self.checkpoint_pool.add_checkpoint(self.params, global_step)
                            if self.file_logger:
                                self.file_logger.info(
                                    f"  [POOL] Added checkpoint at step {global_step:,} | "
                                    f"Pool size: {len(self.checkpoint_pool.checkpoints)}"
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

                    # Add exploiter metrics for adversarial training (v4)
                    if config.use_adversarial_training and exp_collect_metrics is not None and exp_ppo_metrics is not None:
                        exp_ac = exp_collect_metrics.action_counts
                        exp_total_actions = sum(exp_ac.values())
                        exp_decided = exp_collect_metrics.wins + exp_collect_metrics.losses
                        exp_win_rate = exp_collect_metrics.wins / max(exp_decided, 1)
                        exp_fold_rate = exp_ac["fold"] / max(exp_total_actions, 1)
                        exp_raise_count = exp_ac["raise_33"] + exp_ac["raise_66"] + exp_ac["raise_100"] + exp_ac["raise_150"] + exp_ac["all_in"]
                        exp_call_count = exp_ac["call"]
                        exp_aggression = exp_raise_count / max(exp_call_count, 1)

                        self.logger.log(global_step, {
                            # Exploiter rewards
                            "exploiter/reward_avg": exp_collect_metrics.avg_reward,
                            # Exploiter loss
                            "exploiter/loss_policy": exp_ppo_metrics.policy_loss,
                            "exploiter/loss_value": exp_ppo_metrics.value_loss,
                            "exploiter/loss_entropy": exp_ppo_metrics.entropy,
                            # Exploiter PPO diagnostics
                            "exploiter/ppo_approx_kl": exp_ppo_metrics.approx_kl,
                            "exploiter/ppo_clip_fraction": exp_ppo_metrics.clip_fraction,
                            # Exploiter RL diagnostics
                            "exploiter/explained_variance": exp_ppo_metrics.explained_variance,
                            "exploiter/grad_norm": exp_ppo_metrics.grad_norm,
                            # Exploiter poker behavior
                            "exploiter/win_rate": exp_win_rate,
                            "exploiter/fold_rate": exp_fold_rate,
                            "exploiter/aggression": exp_aggression,
                            # Exploiter action distribution
                            "exploiter/action_fold": exp_ac["fold"] / max(exp_total_actions, 1),
                            "exploiter/action_check": exp_ac["check"] / max(exp_total_actions, 1),
                            "exploiter/action_call": exp_ac["call"] / max(exp_total_actions, 1),
                            "exploiter/action_raise_33": exp_ac["raise_33"] / max(exp_total_actions, 1),
                            "exploiter/action_raise_66": exp_ac["raise_66"] / max(exp_total_actions, 1),
                            "exploiter/action_raise_100": exp_ac["raise_100"] / max(exp_total_actions, 1),
                            "exploiter/action_raise_150": exp_ac["raise_150"] / max(exp_total_actions, 1),
                            "exploiter/action_allin": exp_ac["all_in"] / max(exp_total_actions, 1),
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
                        f"[MAIN] step={global_step:>8} | rew={collect_metrics.avg_reward:>7.3f} | "
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

                    # Exploiter logging for adversarial training (v4)
                    if config.use_adversarial_training and exp_collect_metrics is not None and exp_ppo_metrics is not None:
                        exp_decided = exp_collect_metrics.wins + exp_collect_metrics.losses
                        exp_win_rate = exp_collect_metrics.wins / max(exp_decided, 1)
                        self.file_logger.info(
                            f"[EXPL] step={global_step:>8} | rew={exp_collect_metrics.avg_reward:>7.3f} | "
                            f"win={exp_win_rate:>5.1%} | pol={exp_ppo_metrics.policy_loss:>7.4f} | "
                            f"ev={exp_ppo_metrics.explained_variance:>5.2f}"
                        )
                        exp_ac = exp_collect_metrics.action_counts
                        self.file_logger.info(
                            f"  actions: F={exp_ac['fold']:>4} Ch={exp_ac['check']:>4} Ca={exp_ac['call']:>4} "
                            f"R33={exp_ac['raise_33']:>3} R66={exp_ac['raise_66']:>3} R100={exp_ac['raise_100']:>3} R150={exp_ac['raise_150']:>3} A={exp_ac['all_in']:>4} | "
                            f"games: {exp_collect_metrics.games_completed} done, {exp_collect_metrics.wins}W/{exp_collect_metrics.losses}L"
                        )

                # Eval checkpoint (early detection of degenerate strategies)
                if config.eval_checkpoint_every > 0 and global_step > 0:
                    if global_step % config.eval_checkpoint_every < config.steps_per_update:
                        eval_results = self._run_eval_checkpoint(config.eval_checkpoint_games)

                        # Log to file
                        if self.file_logger:
                            self.file_logger.info(
                                f"[EVAL] step={global_step:>8} | "
                                f"vs_random: {eval_results['random']['win_rate']:.1%} ({eval_results['random']['bb_per_100']:+.1f} BB/100) | "
                                f"vs_call_station: {eval_results['call_station']['win_rate']:.1%} ({eval_results['call_station']['bb_per_100']:+.1f} BB/100)"
                            )

                        # Log to TensorBoard
                        if self.logger:
                            self.logger.log(global_step, {
                                "eval/random_win_rate": eval_results["random"]["win_rate"],
                                "eval/random_bb_per_100": eval_results["random"]["bb_per_100"],
                                "eval/call_station_win_rate": eval_results["call_station"]["win_rate"],
                                "eval/call_station_bb_per_100": eval_results["call_station"]["bb_per_100"],
                            })

                        # Warn if losing to random
                        if eval_results["random"]["win_rate"] < 0.5:
                            self.console.print(
                                f"[bold red]WARNING[/bold red]: Model losing to random play "
                                f"({eval_results['random']['win_rate']:.1%} win rate) at step {global_step:,}"
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

    def load_checkpoint(self, path: Path | str) -> int:
        """Load checkpoint and return step count.

        Args:
            path: Path to checkpoint file

        Returns:
            Step count extracted from filename (e.g., step_09600000.pkl -> 9600000)
        """
        import pickle
        import re

        path = Path(path)
        with open(path, "rb") as f:
            checkpoint = pickle.load(f)

        self.params = checkpoint["params"]
        self.opt_state = checkpoint["opt_state"]

        # Extract step count from filename
        match = re.search(r"step_(\d+)", path.stem)
        if match:
            return int(match.group(1))
        return 0


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
