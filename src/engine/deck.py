import random
from src.models import PolicyCard


class Deck:
    def __init__(self, multiplier: int = 1) -> None:
        self.multiplier = multiplier
        self.cards: list[PolicyCard] = []
        self._initialize_deck()

    def draw(self, count: int = 1) -> list[PolicyCard]:
        drawn_cards: list[PolicyCard] = []

        for _ in range(count):
            if len(self.cards) == 0:
                self._initialize_deck()

            drawn_cards.append(self.cards.pop())

        return drawn_cards

    def cards_remaining(self) -> int:
        return len(self.cards)

    def _initialize_deck(self) -> None:
        fascist_count = 6 * self.multiplier
        liberal_count = 11 * self.multiplier

        self.cards = [PolicyCard.FASCIST] * fascist_count + [
            PolicyCard.LIBERAL
        ] * liberal_count
        random.shuffle(self.cards)
