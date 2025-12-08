"""Card creation and conversion utilities for testing."""

import jax.numpy as jnp

# Card encoding: card_id = rank * 4 + suit
# rank: 0=2, 1=3, ..., 12=A
# suit: 0=c, 1=d, 2=h, 3=s

RANK_MAP = {
    "2": 0,
    "3": 1,
    "4": 2,
    "5": 3,
    "6": 4,
    "7": 5,
    "8": 6,
    "9": 7,
    "T": 8,
    "J": 9,
    "Q": 10,
    "K": 11,
    "A": 12,
}

SUIT_MAP = {"c": 0, "d": 1, "h": 2, "s": 3}

RANK_NAMES = ["2", "3", "4", "5", "6", "7", "8", "9", "T", "J", "Q", "K", "A"]
SUIT_NAMES = ["c", "d", "h", "s"]


def make_cards_from_strings(card_strings: list[str]) -> jnp.ndarray:
    """Create JAX card array from strings like ['As', 'Kh', 'Qc'].

    Card encoding: card_id = rank * 4 + suit
    - rank: 0=2, 1=3, ..., 12=A
    - suit: 0=c, 1=d, 2=h, 3=s

    Args:
        card_strings: List of card strings (e.g., ['As', 'Kh'])

    Returns:
        JAX array of card IDs
    """
    cards = []
    for s in card_strings:
        rank = RANK_MAP[s[0].upper()]
        suit = SUIT_MAP[s[1].lower()]
        cards.append(rank * 4 + suit)
    return jnp.array(cards, dtype=jnp.int32)


def card_to_string(card_id: int) -> str:
    """Convert card ID to string representation.

    Args:
        card_id: Card ID (0-51) or -1 for undealt

    Returns:
        String like 'As' or '??' for undealt
    """
    if card_id < 0:
        return "??"
    rank = card_id // 4
    suit = card_id % 4
    return f"{RANK_NAMES[rank]}{SUIT_NAMES[suit]}"


def make_hand(
    hole1: str, hole2: str, community: list[str]
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Create hole cards and community cards for testing.

    Args:
        hole1: First hole card (e.g., 'As')
        hole2: Second hole card (e.g., 'Kh')
        community: List of community cards (0-5 cards)

    Returns:
        Tuple of (hole_cards array [2], community array [5])
    """
    hole = make_cards_from_strings([hole1, hole2])
    comm = make_cards_from_strings(community) if community else jnp.array(
        [], dtype=jnp.int32
    )
    # Pad community to 5 cards with -1 if needed
    if len(comm) < 5:
        padding = jnp.full(5 - len(comm), -1, dtype=jnp.int32)
        comm = jnp.concatenate([comm, padding])
    return hole, comm
