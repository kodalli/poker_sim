"""Model version management utilities."""

import json
import re
import shutil
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


# Base directory for all models
MODELS_DIR = Path("models")


@dataclass
class ModelMetadata:
    """Metadata for a trained model version."""

    version: str
    architecture: str
    generations: int
    population_size: int
    best_fitness: float
    created: str  # ISO format datetime
    description: str = ""
    config: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ModelMetadata":
        """Create from dictionary."""
        return cls(**data)


def get_model_path(version: str) -> Path:
    """Get the path to a model version directory."""
    return MODELS_DIR / version


def get_checkpoint_dir(version: str) -> Path:
    """Get the checkpoints directory for a model version."""
    return get_model_path(version) / "checkpoints"


def get_final_checkpoint(version: str) -> Path:
    """Get the path to the final checkpoint for a model version."""
    return get_model_path(version) / "final.pt"


def get_metadata_path(version: str) -> Path:
    """Get the path to the metadata file for a model version."""
    return get_model_path(version) / "metadata.json"


def version_exists(version: str) -> bool:
    """Check if a model version exists."""
    return get_model_path(version).exists()


def _parse_version_number(version: str) -> int | None:
    """Extract version number from version string (e.g., 'v1' -> 1)."""
    match = re.match(r"v(\d+)", version)
    if match:
        return int(match.group(1))
    return None


def list_models() -> list[str]:
    """List all available model versions, sorted by version number."""
    if not MODELS_DIR.exists():
        return []

    versions = []
    for path in MODELS_DIR.iterdir():
        if path.is_dir() and path.name.startswith("v"):
            # Check if it has a final.pt or metadata.json
            if (path / "final.pt").exists() or (path / "metadata.json").exists():
                versions.append(path.name)

    # Sort by version number
    versions.sort(key=lambda v: _parse_version_number(v) or 0)
    return versions


def get_latest_version() -> str | None:
    """Get the latest (highest numbered) model version."""
    versions = list_models()
    if not versions:
        return None
    return versions[-1]


def get_next_version() -> str:
    """Get the next available version number."""
    versions = list_models()
    if not versions:
        return "v1"

    # Find highest version number
    max_num = 0
    for v in versions:
        num = _parse_version_number(v)
        if num and num > max_num:
            max_num = num

    return f"v{max_num + 1}"


def create_version(version: str, exist_ok: bool = False) -> Path:
    """Create a new model version directory structure.

    Args:
        version: Version string (e.g., 'v1', 'v2')
        exist_ok: If True, don't raise error if version exists

    Returns:
        Path to the created version directory

    Raises:
        FileExistsError: If version exists and exist_ok is False
    """
    model_path = get_model_path(version)

    if model_path.exists() and not exist_ok:
        raise FileExistsError(f"Model version {version} already exists")

    # Create directory structure
    model_path.mkdir(parents=True, exist_ok=True)
    get_checkpoint_dir(version).mkdir(exist_ok=True)

    return model_path


def delete_version(version: str) -> None:
    """Delete a model version and all its files."""
    model_path = get_model_path(version)
    if model_path.exists():
        shutil.rmtree(model_path)


def load_metadata(version: str) -> ModelMetadata | None:
    """Load metadata for a model version."""
    metadata_path = get_metadata_path(version)
    if not metadata_path.exists():
        return None

    with open(metadata_path) as f:
        data = json.load(f)

    return ModelMetadata.from_dict(data)


def save_metadata(version: str, metadata: ModelMetadata) -> None:
    """Save metadata for a model version."""
    metadata_path = get_metadata_path(version)

    # Ensure directory exists
    metadata_path.parent.mkdir(parents=True, exist_ok=True)

    with open(metadata_path, "w") as f:
        json.dump(metadata.to_dict(), f, indent=2)


def create_metadata(
    version: str,
    architecture: str,
    generations: int,
    population_size: int,
    best_fitness: float,
    description: str = "",
    config: dict[str, Any] | None = None,
) -> ModelMetadata:
    """Create a new metadata object with current timestamp."""
    return ModelMetadata(
        version=version,
        architecture=architecture,
        generations=generations,
        population_size=population_size,
        best_fitness=best_fitness,
        created=datetime.now().isoformat(),
        description=description,
        config=config,
    )


def copy_final_checkpoint(version: str) -> None:
    """Copy the latest checkpoint to final.pt in the model directory."""
    checkpoint_dir = get_checkpoint_dir(version)
    final_path = get_final_checkpoint(version)

    # Find the latest checkpoint in the checkpoints directory
    checkpoints = list(checkpoint_dir.glob("*.pt"))
    if not checkpoints:
        return

    # Sort by modification time and get the latest
    latest = max(checkpoints, key=lambda p: p.stat().st_mtime)

    # Copy to final.pt
    shutil.copy2(latest, final_path)


def get_model_summary(version: str) -> dict[str, Any]:
    """Get a summary of a model version for display."""
    metadata = load_metadata(version)
    final_exists = get_final_checkpoint(version).exists()

    if metadata:
        return {
            "version": version,
            "architecture": metadata.architecture,
            "generations": metadata.generations,
            "population_size": metadata.population_size,
            "best_fitness": metadata.best_fitness,
            "created": metadata.created,
            "description": metadata.description,
            "has_checkpoint": final_exists,
        }
    else:
        # No metadata, but checkpoint exists
        return {
            "version": version,
            "architecture": "unknown",
            "generations": "?",
            "population_size": "?",
            "best_fitness": "?",
            "created": "unknown",
            "description": "",
            "has_checkpoint": final_exists,
        }
