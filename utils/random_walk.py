import numpy as np


class RandomWalkEnv:
    """
    5-state random walk (Sutton & Barto, Chapters 5–6).
    States 0 and 6 are terminal; states 1–5 (A–E) are non-terminal.
    Start state is always 3 (C, centre). Moves left or right with equal probability.
    Reward: +1 exiting right, 0 exiting left, 0 otherwise.
    """

    N_STATES = 7
    START    = 3
    LEFT_T   = 0
    RIGHT_T  = 6
    LABELS   = ["A", "B", "C", "D", "E"]

    def __init__(self, seed: int | None = None):
        self.rng    = np.random.default_rng(seed)
        self._state = self.START

    def reset(self) -> int:
        self._state = self.START
        return self._state

    def step(self) -> tuple[int, float, bool]:
        """Returns (next_state, reward, done)."""
        move        = 1 if self.rng.random() < 0.5 else -1
        self._state = self._state + move
        if self._state == self.RIGHT_T:
            return self._state, 1.0, True
        if self._state == self.LEFT_T:
            return self._state, 0.0, True
        return self._state, 0.0, False

    @property
    def state(self) -> int:
        return self._state


def compute_true_values(gamma: float = 1.0) -> np.ndarray:
    """
    Iterative policy evaluation for the equiprobable random policy.
    Returns V for non-terminal states 1–5 (A–E) as a length-5 array.
    """
    V = np.zeros(7)
    for _ in range(100_000):
        delta = 0.0
        for s in range(1, 6):
            r_right = 1.0 if s == 5 else 0.0
            v_new   = 0.5 * (gamma * V[s - 1]) + 0.5 * (r_right + gamma * V[s + 1])
            delta   = max(delta, abs(V[s] - v_new))
            V[s]    = v_new
        if delta < 1e-12:
            break
    return V[1:6].copy()
