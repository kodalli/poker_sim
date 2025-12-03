"""Card, Deck, Suit, and Rank definitions for poker."""

from dataclasses import dataclass
from enum import IntEnum
from random import Random
from typing import Iterator


class Suit(IntEnum):
    """Card suits."""

    CLUBS = 0
    DIAMONDS = 1
    HEARTS = 2
    SPADES = 3

    def __str__(self) -> str:
        symbols = {0: "♣", 1: "♦", 2: "♥", 3: "♠"}
        return symbols[self.value]


class Rank(IntEnum):
    """Card ranks (2-14, where 14 is Ace)."""

    TWO = 2
    THREE = 3
    FOUR = 4
    FIVE = 5
    SIX = 6
    SEVEN = 7
    EIGHT = 8
    NINE = 9
    TEN = 10
    JACK = 11
    QUEEN = 12
    KING = 13
    ACE = 14

    def __str__(self) -> str:
        if self.value <= 10:
            return str(self.value)
        return {11: "J", 12: "Q", 13: "K", 14: "A"}[self.value]


@dataclass(frozen=True, slots=True)
class Card:
    """A single playing card."""

    rank: Rank
    suit: Suit

    def __str__(self) -> str:
        return f"{self.rank}{self.suit}"

    def __repr__(self) -> str:
        return f"Card({self.rank.name}, {self.suit.name})"

    def to_index(self) -> int:
        """Convert to 0-51 index for encoding.

        Index = suit * 13 + (rank - 2)
        """
        return self.suit * 13 + (self.rank - 2)

    @classmethod
    def from_index(cls, index: int) -> "Card":
        """Create card from 0-51 index."""
        suit = Suit(index // 13)
        rank = Rank((index % 13) + 2)
        return cls(rank=rank, suit=suit)

    @classmethod
    def from_string(cls, s: str) -> "Card":
        """Parse card from string like 'As', 'Kh', '2c', 'Td'."""
        rank_map = {
            "2": Rank.TWO,
            "3": Rank.THREE,
            "4": Rank.FOUR,
            "5": Rank.FIVE,
            "6": Rank.SIX,
            "7": Rank.SEVEN,
            "8": Rank.EIGHT,
            "9": Rank.NINE,
            "T": Rank.TEN,
            "J": Rank.JACK,
            "Q": Rank.QUEEN,
            "K": Rank.KING,
            "A": Rank.ACE,
        }
        suit_map = {
            "c": Suit.CLUBS,
            "d": Suit.DIAMONDS,
            "h": Suit.HEARTS,
            "s": Suit.SPADES,
        }
        s = s.strip()
        if len(s) != 2:
            raise ValueError(f"Invalid card string: {s}")
        rank_char = s[0].upper()
        suit_char = s[1].lower()
        if rank_char not in rank_map:
            raise ValueError(f"Invalid rank: {rank_char}")
        if suit_char not in suit_map:
            raise ValueError(f"Invalid suit: {suit_char}")
        return cls(rank=rank_map[rank_char], suit=suit_map[suit_char])


class Deck:
    """A standard 52-card deck."""

    def __init__(self, seed: int | None = None) -> None:
        self._rng = Random(seed)
        self._cards: list[Card] = []
        self.reset()

    def reset(self) -> None:
        """Reset deck to full 52 cards."""
        self._cards = [Card.from_index(i) for i in range(52)]

    def shuffle(self) -> None:
        """Shuffle the deck."""
        self._rng.shuffle(self._cards)

    def deal(self, n: int = 1) -> list[Card]:
        """Deal n cards from the top of the deck."""
        if n > len(self._cards):
            raise ValueError(f"Cannot deal {n} cards, only {len(self._cards)} remaining")
        dealt = self._cards[:n]
        self._cards = self._cards[n:]
        return dealt

    def deal_one(self) -> Card:
        """Deal a single card."""
        return self.deal(1)[0]

    def remaining(self) -> int:
        """Number of cards remaining in deck."""
        return len(self._cards)

    def __len__(self) -> int:
        return len(self._cards)

    def __iter__(self) -> Iterator[Card]:
        return iter(self._cards)
