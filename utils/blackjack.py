"""
Blackjack environment implemented from scratch — no gym required.

Standard casino rules:
  - Player hits or stands
  - Dealer stands on 17+
  - Natural blackjack (2-card 21) pays +1.5 (if dealer doesn't also have it)
  - Bust = -1

State:  (player_sum: 12–21,  dealer_showing: 1–10,  usable_ace: bool)
Actions: 0 = stand,  1 = hit
Rewards: +1 win,  0 draw,  −1 loss,  +1.5 natural
"""

import numpy as np

# Card ranks 1–13: Ace=1, 2–10 face value, J/Q/K=10
# In a real deck, ranks 10/J/Q/K each appear 4× out of 13 cards
CARD_POOL = list(range(1, 10)) + [10, 10, 10, 10]   # 13-element pool for correct distribution
CARD_LABEL = {1: "A", **{i: str(i) for i in range(2, 10)}, 10: "10/J/Q/K"}


def card_value(rank: int) -> int:
    """Map rank 1–13 → point value (Ace=1, face=10)."""
    return min(rank, 10)


def hand_total(hand: list[int]) -> tuple[int, bool]:
    """
    Returns (total, usable_ace) for a hand of card ranks.
    Ace is counted as 11 if it doesn't cause a bust; otherwise 1.
    """
    total = sum(card_value(c) for c in hand)
    has_ace = any(c == 1 for c in hand)
    usable = has_ace and total + 10 <= 21
    return total + (10 if usable else 0), usable


def apply_card(player_sum: int, usable_ace: bool, rank: int) -> tuple[int, bool]:
    """
    Update (player_sum, usable_ace) after drawing a card of the given rank.
    Handles ace-as-1 / ace-as-11 logic without tracking the full hand.
    """
    val = card_value(rank)
    if val == 1:                                   # drew an Ace
        if player_sum + 11 <= 21:
            return player_sum + 11, True
        return player_sum + 1, usable_ace          # ace must be 1
    new_sum = player_sum + val
    if new_sum > 21 and usable_ace:                # flip existing ace
        return new_sum - 10, False
    return new_sum, usable_ace


def simulate_dealer(upcard_rank: int, rng: np.random.Generator) -> int:
    """
    Simulate dealer play from a known upcard rank.
    Dealer hits until total >= 17. Returns final total.
    """
    hand = [upcard_rank, rng.choice(CARD_POOL)]
    while hand_total(hand)[0] < 17:
        hand.append(rng.choice(CARD_POOL))
    return hand_total(hand)[0]


# ── Environment ────────────────────────────────────────────────────────────

class BlackjackEnv:
    """
    Minimal Blackjack environment with reset() / step() API.

    Usage:
        env = BlackjackEnv()
        state = env.reset()                  # (player_sum, dealer_showing, usable_ace)
        state, reward, done = env.step(1)    # hit
        state, reward, done = env.step(0)    # stand
    """

    def __init__(self, natural: bool = True):
        self.natural = natural
        self.rng = np.random.default_rng()

    def seed(self, s: int):
        self.rng = np.random.default_rng(s)

    def reset(self) -> tuple:
        self.player = [self._draw(), self._draw()]
        self.dealer = [self._draw(), self._draw()]
        self.done   = False

        # Auto-hit below 12 — no real decision to be made
        while hand_total(self.player)[0] < 12:
            self.player.append(self._draw())

        return self._state()

    def step(self, action: int) -> tuple[tuple, float, bool]:
        assert not self.done, "Call reset() before step()"

        if action == 1:                            # hit
            self.player.append(self._draw())
            total, _ = hand_total(self.player)
            if total > 21:
                self.done = True
                return self._state(), -1.0, True
            return self._state(), 0.0, False

        else:                                      # stand — dealer plays
            while hand_total(self.dealer)[0] < 17:
                self.dealer.append(self._draw())

            p = hand_total(self.player)[0]
            d = hand_total(self.dealer)[0]

            if d > 21 or p > d:
                reward = 1.0
            elif p == d:
                reward = 0.0
            else:
                reward = -1.0

            if self.natural and reward == 1.0:
                p_nat = len(self.player) == 2 and p == 21
                d_nat = len(self.dealer) == 2 and d == 21
                if p_nat and not d_nat:
                    reward = 1.5

            self.done = True
            return self._state(), reward, True

    # ── Internals ──────────────────────────────────────────────────────────

    def _draw(self) -> int:
        return int(self.rng.choice(CARD_POOL))

    def _state(self) -> tuple:
        p_total, ua = hand_total(self.player)
        dealer_up   = card_value(self.dealer[0])   # Ace shows as 1
        return (int(p_total), int(dealer_up), bool(ua))

    @property
    def player_cards(self) -> list[int]:
        return self.player.copy()

    @property
    def dealer_upcard(self) -> int:
        return card_value(self.dealer[0])
