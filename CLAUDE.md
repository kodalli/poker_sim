# poker-sim

Poker simulation project.

## Package Management

This project uses **uv** for Python package management.

```bash
# Install dependencies
uv sync

# Add a new dependency
uv add <package>

# Add a dev dependency
uv add --dev <package>

# Run the project
uv run python main.py

# Run any command in the virtual environment
uv run <command>
```

## Python Version

Python 3.13 (managed via `.python-version`)

## Project Structure

```
poker_sim/
├── main.py          # Entry point
├── pyproject.toml   # Project configuration and dependencies
├── .python-version  # Python version specification
└── .gitignore
```
