"""
4×4 GridWorld — shared constants, transition function, and cached solvers.

Layout (Sutton & Barto Example 4.1):
  Terminal states: 0 (top-left) and 15 (bottom-right)
  Actions: 0=up  1=down  2=left  3=right
  Reward:  -1 on every non-terminal step
  Gamma:   1.0 (undiscounted)
"""

import numpy as np
import streamlit as st

GRID_SIZE = 4
N_STATES = GRID_SIZE * GRID_SIZE
TERMINAL_STATES = frozenset({0, 15})
GAMMA = 1.0
REWARD = -1.0
N_ACTIONS = 4

ACTION_DELTAS = [(-1, 0), (1, 0), (0, -1), (0, 1)]   # up, down, left, right
ACTION_ARROWS = ["↑", "↓", "←", "→"]


def state_to_rc(s: int) -> tuple[int, int]:
    return s // GRID_SIZE, s % GRID_SIZE


def rc_to_state(r: int, c: int) -> int:
    return r * GRID_SIZE + c


def next_state_reward(s: int, a: int) -> tuple[int, float]:
    """Return (next_state, reward) for a deterministic step."""
    if s in TERMINAL_STATES:
        return s, 0.0
    r, c = state_to_rc(s)
    dr, dc = ACTION_DELTAS[a]
    nr, nc = r + dr, c + dc
    if not (0 <= nr < GRID_SIZE and 0 <= nc < GRID_SIZE):
        nr, nc = r, c          # bounce off wall
    return rc_to_state(nr, nc), REWARD


def initial_policy() -> np.ndarray:
    """
    DOWN-then-RIGHT seed policy: guaranteed to reach a terminal from every
    state, so iterative evaluation under γ=1 converges for every state.
    """
    policy = np.zeros(N_STATES, dtype=int)
    for s in range(N_STATES):
        if s in TERMINAL_STATES:
            continue
        r, _ = state_to_rc(s)
        policy[s] = 3 if r == GRID_SIZE - 1 else 1   # RIGHT or DOWN
    return policy


# ── Cached solvers ─────────────────────────────────────────────────────────

@st.cache_data
def solve_random_policy(theta: float) -> tuple[np.ndarray, list[float]]:
    """
    Iterative policy evaluation for the equiprobable random policy.
    Returns (V, deltas_per_sweep).  Cached by theta.
    """
    V = np.zeros(N_STATES)
    deltas: list[float] = []
    while True:
        delta = 0.0
        for s in range(N_STATES):
            if s in TERMINAL_STATES:
                continue
            v_old = V[s]
            v_new = sum(
                (1.0 / N_ACTIONS) * (r + GAMMA * V[ns])
                for a in range(N_ACTIONS)
                for ns, r in [next_state_reward(s, a)]
            )
            V[s] = v_new
            delta = max(delta, abs(v_old - v_new))
        deltas.append(delta)
        if delta < theta:
            break
    return V, deltas


@st.cache_data
def solve_policy_iteration(theta: float = 1e-6) -> list[dict]:
    """
    Full policy iteration from the DOWN-RIGHT seed policy.
    Returns a list of snapshots (one per iteration) with keys:
      iteration, policy, V, stable
    Cached by theta.
    """
    policy = initial_policy()
    history: list[dict] = []

    for iteration in range(100):
        # --- evaluate ---
        V = np.zeros(N_STATES)
        while True:
            delta = 0.0
            for s in range(N_STATES):
                if s in TERMINAL_STATES:
                    continue
                ns, r = next_state_reward(s, policy[s])
                v_new = r + GAMMA * V[ns]
                delta = max(delta, abs(V[s] - v_new))
                V[s] = v_new
            if delta < theta:
                break

        # --- improve ---
        new_policy = np.zeros(N_STATES, dtype=int)
        for s in range(N_STATES):
            if s in TERMINAL_STATES:
                continue
            q = [next_state_reward(s, a)[1] + GAMMA * V[next_state_reward(s, a)[0]]
                 for a in range(N_ACTIONS)]
            new_policy[s] = int(np.argmax(q))

        stable = bool(np.array_equal(policy, new_policy))
        history.append({
            "iteration": iteration + 1,
            "policy": policy.copy(),
            "V": V.copy(),
            "stable": stable,
        })
        if stable:
            break
        policy = new_policy

    return history


@st.cache_data
def solve_value_iteration(theta: float = 1e-6) -> tuple[np.ndarray, np.ndarray, list[float]]:
    """
    Value iteration.  Returns (V*, optimal_policy, deltas_per_sweep).
    Cached by theta.
    """
    V = np.zeros(N_STATES)
    deltas: list[float] = []

    while True:
        delta = 0.0
        for s in range(N_STATES):
            if s in TERMINAL_STATES:
                continue
            v_old = V[s]
            V[s] = max(
                next_state_reward(s, a)[1] + GAMMA * V[next_state_reward(s, a)[0]]
                for a in range(N_ACTIONS)
            )
            delta = max(delta, abs(v_old - V[s]))
        deltas.append(delta)
        if delta < theta:
            break

    policy = np.zeros(N_STATES, dtype=int)
    for s in range(N_STATES):
        if s in TERMINAL_STATES:
            continue
        q = [next_state_reward(s, a)[1] + GAMMA * V[next_state_reward(s, a)[0]]
             for a in range(N_ACTIONS)]
        policy[s] = int(np.argmax(q))

    return V, policy, deltas


def policy_to_arrow_grid(policy: np.ndarray) -> list[list[str]]:
    grid = []
    for r in range(GRID_SIZE):
        row = []
        for c in range(GRID_SIZE):
            s = rc_to_state(r, c)
            row.append("T" if s in TERMINAL_STATES else ACTION_ARROWS[policy[s]])
        grid.append(row)
    return grid


def make_value_heatmap(V: np.ndarray, title: str, annotations: list | None = None):
    """Return a Plotly heatmap figure for a value function."""
    import plotly.graph_objects as go
    grid = V.reshape(GRID_SIZE, GRID_SIZE)
    auto_ann = annotations or [
        dict(x=c, y=r, text=f"{grid[r][c]:.1f}", showarrow=False,
             font=dict(size=16, color="black"), xanchor="center", yanchor="middle")
        for r in range(GRID_SIZE) for c in range(GRID_SIZE)
    ]
    fig = go.Figure(data=go.Heatmap(
        z=grid, colorscale="RdYlGn", showscale=True,
        colorbar=dict(title="V(s)"), zmin=float(V.min()), zmax=0,
    ))
    fig.update_layout(
        title=title,
        xaxis=dict(tickvals=list(range(GRID_SIZE)),
                   ticktext=[f"col {c}" for c in range(GRID_SIZE)]),
        yaxis=dict(tickvals=list(range(GRID_SIZE)),
                   ticktext=[f"row {r}" for r in range(GRID_SIZE)],
                   autorange="reversed"),
        annotations=auto_ann,
        height=420,
        template="plotly_white",
    )
    return fig
