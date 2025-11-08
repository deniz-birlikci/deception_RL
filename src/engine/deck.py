import random
from src.models import PolicyCard


class Deck:
    def __init__(self, multiplier: int = 1) -> None:
        self.multiplier = multiplier
        self.cards: list[PolicyCard] = []
        self.discard_pile: list[PolicyCard] = []
        self.total_fascist_cards = 11 * multiplier
        self.total_liberal_cards = 6 * multiplier
        self._initialize_deck()

    def draw(self, count: int = 1) -> list[PolicyCard]:
        drawn_cards: list[PolicyCard] = []

        for _ in range(count):
            if len(self.cards) == 0:
                self._reshuffle_discard()

            drawn_cards.append(self.cards.pop())

        return drawn_cards

    def cards_remaining(self) -> int:
        return len(self.cards)

    def add_to_discard(self, card: PolicyCard) -> None:
        self.discard_pile.append(card)

    def _reshuffle_discard(self) -> None:
        if len(self.discard_pile) == 0:
            raise ValueError("Cannot reshuffle: both draw pile and discard pile are empty")
        
        self.cards = self.discard_pile
        self.discard_pile = []
        random.shuffle(self.cards)

    def _initialize_deck(self) -> None:
        self.cards = [PolicyCard.FASCIST] * self.total_fascist_cards + [
            PolicyCard.LIBERAL
        ] * self.total_liberal_cards
        random.shuffle(self.cards)
