"""Card operations for JAX-accelerated poker.

Card representation:
    card_id = rank * 4 + suit
    rank: 0=2, 1=3, 2=4, ..., 8=T, 9=J, 10=Q, 11=K, 12=A
    suit: 0=clubs, 1=diamonds, 2=hearts, 3=spades
"""

import jax
import jax.numpy as jnp
import jax.random as jrandom
from jax import Array

# Card constants
NUM_CARDS = 52
NUM_RANKS = 13
NUM_SUITS = 4

# Rank names (for display)
RANK_NAMES = ["2", "3", "4", "5", "6", "7", "8", "9", "T", "J", "Q", "K", "A"]
SUIT_NAMES = ["c", "d", "h", "s"]  # clubs, diamonds, hearts, spades


def card_to_rank(card: Array) -> Array:
    """Extract rank from card ID."""
    return card // NUM_SUITS


def card_to_suit(card: Array) -> Array:
    """Extract suit from card ID."""
    return card % NUM_SUITS


def make_card(rank: int, suit: int) -> int:
    """Create card ID from rank and suit."""
    return rank * NUM_SUITS + suit


def card_to_string(card: int) -> str:
    """Convert card ID to string representation."""
    if card < 0:
        return "??"
    rank = card // NUM_SUITS
    suit = card % NUM_SUITS
    return f"{RANK_NAMES[rank]}{SUIT_NAMES[suit]}"


@jax.jit
def shuffle_deck(key: Array) -> Array:
    """Create a shuffled deck of card indices.

    Args:
        key: PRNG key

    Returns:
        [52] array of shuffled card indices 0-51
    """
    return jrandom.permutation(key, NUM_CARDS)


@jax.jit
def shuffle_decks(keys: Array) -> Array:
    """Shuffle multiple decks in parallel.

    Args:
        keys: [N, 2] PRNG keys

    Returns:
        [N, 52] shuffled decks
    """
    return jax.vmap(shuffle_deck)(keys)


def deal_cards_4(deck: Array, deck_idx: Array) -> tuple[Array, Array]:
    """Deal 4 cards from the deck (for hole cards).

    Args:
        deck: [52] shuffled deck
        deck_idx: Current position in deck

    Returns:
        Tuple of (dealt_cards [4], new_deck_idx)
    """
    cards = jax.lax.dynamic_slice(deck, (deck_idx,), (4,))
    new_idx = deck_idx + 4
    return cards, new_idx


def deal_cards_2(deck: Array, deck_idx: Array) -> tuple[Array, Array]:
    """Deal 2 cards from the deck (burn + 1).

    Args:
        deck: [52] shuffled deck
        deck_idx: Current position in deck

    Returns:
        Tuple of (dealt_cards [2], new_deck_idx)
    """
    cards = jax.lax.dynamic_slice(deck, (deck_idx,), (2,))
    new_idx = deck_idx + 2
    return cards, new_idx


def deal_cards_4_flop(deck: Array, deck_idx: Array) -> tuple[Array, Array]:
    """Deal 4 cards from the deck (burn + 3 for flop).

    Args:
        deck: [52] shuffled deck
        deck_idx: Current position in deck

    Returns:
        Tuple of (dealt_cards [4], new_deck_idx)
    """
    cards = jax.lax.dynamic_slice(deck, (deck_idx,), (4,))
    new_idx = deck_idx + 4
    return cards, new_idx


@jax.jit
def deal_hole_cards_batch(
    decks: Array, deck_indices: Array
) -> tuple[Array, Array]:
    """Deal hole cards to both players for N games.

    Args:
        decks: [N, 52] shuffled decks
        deck_indices: [N] current deck positions

    Returns:
        Tuple of:
            hole_cards: [N, 2, 2] hole cards (games, players, cards)
            new_indices: [N] updated deck positions
    """
    # Deal 4 cards total (2 per player, alternating)
    # Player 0 gets cards at idx 0, 2
    # Player 1 gets cards at idx 1, 3
    def deal_one_game(deck, idx):
        cards, new_idx = deal_cards_4(deck, idx)
        # Reshape to [2, 2] - player 0 gets cards 0,2; player 1 gets 1,3
        hole = jnp.stack([
            jnp.stack([cards[0], cards[2]]),  # Player 0
            jnp.stack([cards[1], cards[3]]),  # Player 1
        ])
        return hole, new_idx

    hole_cards, new_indices = jax.vmap(deal_one_game)(decks, deck_indices)
    return hole_cards, new_indices


@jax.jit
def deal_flop_batch(
    decks: Array, deck_indices: Array, community: Array
) -> tuple[Array, Array]:
    """Deal flop (3 community cards) for N games.

    Args:
        decks: [N, 52] shuffled decks
        deck_indices: [N] current deck positions
        community: [N, 5] existing community cards (-1 = not dealt)

    Returns:
        Tuple of:
            community: [N, 5] updated community cards
            new_indices: [N] updated deck positions
    """
    def deal_one_flop(deck, idx, comm):
        # Burn one card, deal 3
        cards, new_idx = deal_cards_4_flop(deck, idx)  # 1 burn + 3 flop
        new_comm = comm.at[0].set(cards[1])
        new_comm = new_comm.at[1].set(cards[2])
        new_comm = new_comm.at[2].set(cards[3])
        return new_comm, new_idx

    new_community, new_indices = jax.vmap(deal_one_flop)(
        decks, deck_indices, community
    )
    return new_community, new_indices


@jax.jit
def deal_turn_batch(
    decks: Array, deck_indices: Array, community: Array
) -> tuple[Array, Array]:
    """Deal turn (1 community card) for N games.

    Args:
        decks: [N, 52] shuffled decks
        deck_indices: [N] current deck positions
        community: [N, 5] existing community cards

    Returns:
        Tuple of:
            community: [N, 5] updated community cards
            new_indices: [N] updated deck positions
    """
    def deal_one_turn(deck, idx, comm):
        # Burn one card, deal 1
        cards, new_idx = deal_cards_2(deck, idx)  # 1 burn + 1 turn
        new_comm = comm.at[3].set(cards[1])  # Skip burn card
        return new_comm, new_idx

    new_community, new_indices = jax.vmap(deal_one_turn)(
        decks, deck_indices, community
    )
    return new_community, new_indices


@jax.jit
def deal_river_batch(
    decks: Array, deck_indices: Array, community: Array
) -> tuple[Array, Array]:
    """Deal river (1 community card) for N games.

    Args:
        decks: [N, 52] shuffled decks
        deck_indices: [N] current deck positions
        community: [N, 5] existing community cards

    Returns:
        Tuple of:
            community: [N, 5] updated community cards
            new_indices: [N] updated deck positions
    """
    def deal_one_river(deck, idx, comm):
        # Burn one card, deal 1
        cards, new_idx = deal_cards_2(deck, idx)  # 1 burn + 1 river
        new_comm = comm.at[4].set(cards[1])  # Skip burn card
        return new_comm, new_idx

    new_community, new_indices = jax.vmap(deal_one_river)(
        decks, deck_indices, community
    )
    return new_community, new_indices


@jax.jit
def cards_to_one_hot(cards: Array) -> Array:
    """Convert card indices to one-hot representation.

    Args:
        cards: [...] card indices (0-51, -1 for no card)

    Returns:
        [..., 52] one-hot representation
    """
    # Handle -1 (no card) by clamping to 0, then zeroing out
    valid_mask = cards >= 0
    safe_cards = jnp.maximum(cards, 0)
    one_hot = jax.nn.one_hot(safe_cards, NUM_CARDS)
    # Zero out invalid cards
    return one_hot * valid_mask[..., None]


@jax.jit
def cards_to_rank_suit(cards: Array) -> tuple[Array, Array]:
    """Split cards into rank and suit arrays.

    Args:
        cards: [...] card indices

    Returns:
        Tuple of (ranks, suits) with same shape as input
    """
    ranks = card_to_rank(cards)
    suits = card_to_suit(cards)
    return ranks, suits
