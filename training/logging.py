"""Metrics logging for training (TensorBoard + CSV)."""

import csv
from pathlib import Path
from typing import Any


class MetricsLogger:
    """Logger that writes to TensorBoard and/or CSV.

    Usage:
        logger = MetricsLogger(Path("logs/run1"), use_tensorboard=True)
        logger.log(step=100, {"loss": 0.5, "accuracy": 0.9})
        logger.close()
    """

    def __init__(
        self,
        log_dir: Path,
        use_tensorboard: bool = True,
        use_csv: bool = True,
    ) -> None:
        """Initialize logger.

        Args:
            log_dir: Directory for log files
            use_tensorboard: Enable TensorBoard logging
            use_csv: Enable CSV logging
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.use_tensorboard = use_tensorboard
        self.use_csv = use_csv

        # TensorBoard writer
        self.writer = None
        if use_tensorboard:
            try:
                from torch.utils.tensorboard import SummaryWriter
                self.writer = SummaryWriter(str(self.log_dir / "tensorboard"))
            except ImportError:
                print("Warning: tensorboard not available, disabling TensorBoard logging")
                self.use_tensorboard = False

        # CSV file
        self.csv_file = None
        self.csv_writer = None
        self.csv_columns: list[str] | None = None
        if use_csv:
            self.csv_path = self.log_dir / "metrics.csv"

    def log(self, step: int, metrics: dict[str, float]) -> None:
        """Log metrics at a given step.

        Args:
            step: Training step/iteration
            metrics: Dictionary of metric name -> value
        """
        # TensorBoard
        if self.use_tensorboard and self.writer is not None:
            for name, value in metrics.items():
                self.writer.add_scalar(name, value, step)

        # CSV
        if self.use_csv:
            self._log_csv(step, metrics)

    def _log_csv(self, step: int, metrics: dict[str, float]) -> None:
        """Write metrics to CSV file."""
        # Initialize CSV if needed
        if self.csv_file is None:
            self.csv_columns = ["step"] + sorted(metrics.keys())
            self.csv_file = open(self.csv_path, "w", newline="")
            self.csv_writer = csv.DictWriter(self.csv_file, fieldnames=self.csv_columns)
            self.csv_writer.writeheader()

        # Add any new columns
        new_cols = set(metrics.keys()) - set(self.csv_columns[1:])
        if new_cols:
            # Reopen file with new columns
            self.csv_file.close()
            self.csv_columns = ["step"] + sorted(set(self.csv_columns[1:]) | set(metrics.keys()))
            self.csv_file = open(self.csv_path, "a", newline="")
            self.csv_writer = csv.DictWriter(self.csv_file, fieldnames=self.csv_columns)

        # Write row
        row = {"step": step, **metrics}
        self.csv_writer.writerow(row)
        self.csv_file.flush()

    def log_histogram(self, step: int, name: str, values: Any) -> None:
        """Log histogram data (TensorBoard only).

        Args:
            step: Training step
            name: Histogram name
            values: Array of values
        """
        if self.use_tensorboard and self.writer is not None:
            self.writer.add_histogram(name, values, step)

    def log_text(self, step: int, name: str, text: str) -> None:
        """Log text (TensorBoard only).

        Args:
            step: Training step
            name: Text name
            text: Text content
        """
        if self.use_tensorboard and self.writer is not None:
            self.writer.add_text(name, text, step)

    def flush(self) -> None:
        """Flush all writers."""
        if self.writer is not None:
            self.writer.flush()
        if self.csv_file is not None:
            self.csv_file.flush()

    def close(self) -> None:
        """Close all writers."""
        if self.writer is not None:
            self.writer.close()
        if self.csv_file is not None:
            self.csv_file.close()

    def __enter__(self) -> "MetricsLogger":
        return self

    def __exit__(self, *args) -> None:
        self.close()


def load_csv_metrics(csv_path: Path | str) -> dict[str, list[float]]:
    """Load metrics from CSV file.

    Args:
        csv_path: Path to CSV file

    Returns:
        Dictionary mapping metric name to list of values
    """
    metrics: dict[str, list[float]] = {}

    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            for key, value in row.items():
                if key not in metrics:
                    metrics[key] = []
                try:
                    metrics[key].append(float(value))
                except (ValueError, TypeError):
                    pass

    return metrics


def plot_training_curves(
    csv_path: Path | str,
    output_path: Path | str | None = None,
    metrics_to_plot: list[str] | None = None,
) -> None:
    """Plot training curves from CSV file.

    Args:
        csv_path: Path to CSV file
        output_path: Path to save plot (optional)
        metrics_to_plot: List of metrics to plot (default: all)
    """
    import matplotlib.pyplot as plt

    metrics = load_csv_metrics(csv_path)

    if metrics_to_plot is None:
        metrics_to_plot = [k for k in metrics.keys() if k != "step"]

    steps = metrics.get("step", list(range(len(next(iter(metrics.values()))))))

    # Create subplots
    n_metrics = len(metrics_to_plot)
    n_cols = min(3, n_metrics)
    n_rows = (n_metrics + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
    axes = axes.flatten() if n_metrics > 1 else [axes]

    for ax, metric_name in zip(axes, metrics_to_plot):
        if metric_name in metrics:
            ax.plot(steps, metrics[metric_name])
            ax.set_xlabel("Step")
            ax.set_ylabel(metric_name)
            ax.set_title(metric_name)
            ax.grid(True, alpha=0.3)

    # Hide unused subplots
    for ax in axes[len(metrics_to_plot):]:
        ax.set_visible(False)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
    else:
        plt.show()

    plt.close()


def plot_ppo_training(
    csv_path: Path | str,
    output_dir: Path | str,
) -> None:
    """Generate PPO training plots from CSV metrics.

    Creates 4 focused plots:
    1. losses.png - Policy loss, value loss, entropy
    2. rewards.png - Average reward over training
    3. ppo_diagnostics.png - Approximate KL and clip fraction
    4. performance.png - Steps/sec and games/sec

    Args:
        csv_path: Path to metrics.csv file
        output_dir: Directory to save plots
    """
    import matplotlib.pyplot as plt

    metrics = load_csv_metrics(csv_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    steps = metrics.get("step", [])
    if not steps:
        print("Warning: No data found in CSV file")
        return

    # Plot style
    plt.style.use("seaborn-v0_8-darkgrid")

    # 1. Loss curves
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    loss_metrics = [
        ("loss/policy", "Policy Loss", "tab:blue"),
        ("loss/value", "Value Loss", "tab:orange"),
        ("loss/entropy", "Entropy", "tab:green"),
    ]
    for ax, (key, title, color) in zip(axes, loss_metrics):
        if key in metrics:
            ax.plot(steps, metrics[key], color=color, linewidth=1.5)
            ax.set_xlabel("Step")
            ax.set_ylabel(title)
            ax.set_title(title)
            ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "losses.png", dpi=150, bbox_inches="tight")
    plt.close()

    # 2. Reward curve
    fig, ax = plt.subplots(figsize=(10, 5))
    if "reward/avg" in metrics:
        ax.plot(steps, metrics["reward/avg"], color="tab:green", linewidth=1.5)
        ax.set_xlabel("Step")
        ax.set_ylabel("Average Reward")
        ax.set_title("Training Reward")
        ax.grid(True, alpha=0.3)
        # Add smoothed line if enough data
        if len(steps) > 20:
            import numpy as np
            window = min(20, len(steps) // 5)
            smoothed = np.convolve(metrics["reward/avg"], np.ones(window)/window, mode="valid")
            ax.plot(steps[window-1:], smoothed, color="darkgreen", linewidth=2, label="Smoothed")
            ax.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "rewards.png", dpi=150, bbox_inches="tight")
    plt.close()

    # 3. PPO diagnostics
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    diag_metrics = [
        ("ppo/approx_kl", "Approximate KL", "tab:purple"),
        ("ppo/clip_fraction", "Clip Fraction", "tab:red"),
    ]
    for ax, (key, title, color) in zip(axes, diag_metrics):
        if key in metrics:
            ax.plot(steps, metrics[key], color=color, linewidth=1.5)
            ax.set_xlabel("Step")
            ax.set_ylabel(title)
            ax.set_title(title)
            ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "ppo_diagnostics.png", dpi=150, bbox_inches="tight")
    plt.close()

    # 4. Performance
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    perf_metrics = [
        ("perf/steps_per_second", "Steps/Second", "tab:blue"),
        ("perf/games_per_second", "Games/Second", "tab:orange"),
    ]
    for ax, (key, title, color) in zip(axes, perf_metrics):
        if key in metrics:
            ax.plot(steps, metrics[key], color=color, linewidth=1.5)
            ax.set_xlabel("Step")
            ax.set_ylabel(title)
            ax.set_title(title)
            ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "performance.png", dpi=150, bbox_inches="tight")
    plt.close()

    print(f"Saved 4 plots to {output_dir}/")
