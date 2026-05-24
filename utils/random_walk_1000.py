import numpy as np


class RandomWalk1000:
    """
    1000-state random walk (Sutton & Barto, Chapter 9, Example 9.1).
    States 1–1000 are non-terminal.  State 0 is the left terminal (reward –1).
    State 1001 is the right terminal (reward +1).
    At each step the agent moves left or right with equal probability by a
    uniformly-chosen step size in [1, 100].  Start state is 500 (centre).
    """

    N_STATES  = 1000
    START     = 500
    LEFT_T    = 0
    RIGHT_T   = 1001
    MAX_STEP  = 100

    def __init__(self, seed: int | None = None):
        self.rng    = np.random.default_rng(seed)
        self._state = self.START

    def reset(self) -> int:
        self._state = self.START
        return self._state

    def step(self) -> tuple[int, float, bool]:
        """Returns (next_state, reward, done)."""
        direction = 1 if self.rng.random() < 0.5 else -1
        size      = int(self.rng.integers(1, self.MAX_STEP + 1))
        ns        = self._state + direction * size

        if ns <= self.LEFT_T:
            self._state = self.LEFT_T
            return self.LEFT_T, -1.0, True
        if ns >= self.RIGHT_T:
            self._state = self.RIGHT_T
            return self.RIGHT_T, 1.0, True

        self._state = ns
        return ns, 0.0, False

    @property
    def state(self) -> int:
        return self._state


def compute_true_values() -> np.ndarray:
    """
    Iterative policy evaluation for the 1000-state random walk with γ=1.
    Returns V for non-terminal states 1–1000 as a length-1000 array.
    By symmetry V*(s) = (2s – 1001) / 1001, but we compute it numerically
    to be exact.
    """
    N, MAX = 1000, 100
    V = np.zeros(N + 2)   # indices 0..1001; 0 = left terminal, 1001 = right terminal
    V[0]    = -1.0
    V[1001] =  1.0

    for _ in range(10_000):
        delta = 0.0
        for s in range(1, N + 1):
            v_new = 0.0
            # left moves
            for k in range(1, MAX + 1):
                ns = max(s - k, 0)
                v_new += 0.5 / MAX * V[ns]
                if ns == 0:
                    break         # remaining probability already at terminal
            # right moves
            for k in range(1, MAX + 1):
                ns = min(s + k, N + 1)
                v_new += 0.5 / MAX * V[ns]
                if ns == N + 1:
                    break
            delta   = max(delta, abs(V[s] - v_new))
            V[s]    = v_new
        if delta < 1e-6:
            break

    return V[1 : N + 1].copy()
