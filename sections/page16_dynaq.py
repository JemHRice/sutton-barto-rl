import numpy as np
import plotly.graph_objects as go
import streamlit as st


# ── Maze environment ───────────────────────────────────────────────────────
#
#  S = start, G = goal, # = wall
#
#  . . . . . . . . . #
#  . # # # # # # . . #
#  . . . . . . # . . .
#  . . . . . . # . . .
#  . . . . . . . . . .
#  . . . . . . . . . G
#  S . . . . . . . . .

MAZE_H, MAZE_W = 7, 10
START = (6, 0)
GOAL = (5, 9)  # adjusted to a reachable spot

_WALLS = {
    (0, 9),
    (1, 2), (1, 3), (1, 4), (1, 5), (1, 6), (1, 7), (1, 9),
    (2, 6), (3, 6),
}

N_STATES = MAZE_H * MAZE_W
N_ACTIONS = 4  # 0=up, 1=right, 2=down, 3=left
_DELTAS = [(-1, 0), (0, 1), (1, 0), (0, -1)]
_ACTION_LABELS = ["↑", "→", "↓", "←"]


def _rc_to_s(r: int, c: int) -> int:
    return r * MAZE_W + c


def _s_to_rc(s: int) -> tuple[int, int]:
    return divmod(s, MAZE_W)


def _is_wall(r: int, c: int) -> bool:
    return (r, c) in _WALLS


def _maze_step(state: int, action: int) -> tuple[int, float, bool]:
    r, c = _s_to_rc(state)
    dr, dc = _DELTAS[action]
    nr, nc = r + dr, c + dc
    if not (0 <= nr < MAZE_H and 0 <= nc < MAZE_W) or _is_wall(nr, nc):
        nr, nc = r, c
    next_state = _rc_to_s(nr, nc)
    done = (nr, nc) == GOAL
    reward = 1.0 if done else 0.0
    return next_state, reward, done


# ── Dyna-Q ─────────────────────────────────────────────────────────────────

@st.cache_data
def run_dynaq(
    n_planning: int,
    alpha: float,
    gamma: float,
    epsilon: float,
    total_steps: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Dyna-Q on the maze.
    Returns:
      cumulative_rewards [total_steps]
      steps_per_episode  [up to total_steps episodes]
      Q                  [N_STATES, N_ACTIONS]
    """
    rng = np.random.default_rng(42)
    Q = np.zeros((N_STATES, N_ACTIONS))
    model: dict[tuple[int, int], tuple[int, float]] = {}
    seen_sa: list[tuple[int, int]] = []

    start_s = _rc_to_s(*START)
    cumulative = np.zeros(total_steps)
    steps_list = []

    step = 0
    ep_steps = 0

    while step < total_steps:
        if ep_steps == 0:
            state = start_s

        # ε-greedy action — random tie-breaking prevents getting stuck when Q is all zeros
        if rng.random() < epsilon:
            action = int(rng.integers(0, N_ACTIONS))
        else:
            q_s = Q[state]
            action = int(rng.choice(np.flatnonzero(q_s == q_s.max())))

        next_state, reward, done = _maze_step(state, action)

        # Direct RL update
        Q[state, action] += alpha * (
            reward + gamma * np.max(Q[next_state]) - Q[state, action]
        )

        # Model update
        if (state, action) not in model:
            seen_sa.append((state, action))
        model[(state, action)] = (next_state, reward)

        # Planning
        for _ in range(n_planning):
            idx = rng.integers(0, len(seen_sa))
            s_p, a_p = seen_sa[idx]
            ns_p, r_p = model[(s_p, a_p)]
            Q[s_p, a_p] += alpha * (
                r_p + gamma * np.max(Q[ns_p]) - Q[s_p, a_p]
            )

        cumulative[step] = reward if step == 0 else cumulative[step - 1] + reward
        step += 1
        ep_steps += 1

        if done or ep_steps > 2000:
            steps_list.append(ep_steps)
            ep_steps = 0
        else:
            state = next_state

    return cumulative, np.array(steps_list), Q


@st.cache_data
def run_dynaq_sweep(
    n_values: tuple[int, ...],
    alpha: float,
    gamma: float,
    epsilon: float,
    total_steps: int,
) -> dict[int, np.ndarray]:
    return {n: run_dynaq(n, alpha, gamma, epsilon, total_steps)[0] for n in n_values}


# ── Maze figure ────────────────────────────────────────────────────────────

def _maze_figure(Q: np.ndarray | None = None, title: str = "Maze") -> go.Figure:
    cell_colors = []
    cell_texts = []
    for r in range(MAZE_H):
        row_colors = []
        row_texts = []
        for c in range(MAZE_W):
            if _is_wall(r, c):
                row_colors.append(0.0)
                row_texts.append("■")
            elif (r, c) == GOAL:
                row_colors.append(1.0)
                row_texts.append("G")
            elif (r, c) == START:
                row_colors.append(0.3)
                row_texts.append("S")
            else:
                if Q is not None:
                    s = _rc_to_s(r, c)
                    best_a = int(np.argmax(Q[s]))
                    row_colors.append(0.5 + 0.4 * (np.max(Q[s]) / (np.max(Q) + 1e-9)))
                    row_texts.append(_ACTION_LABELS[best_a])
                else:
                    row_colors.append(0.3)
                    row_texts.append("·")
        cell_colors.append(row_colors)
        cell_texts.append(row_texts)

    fig = go.Figure(
        go.Heatmap(
            z=cell_colors,
            text=cell_texts,
            texttemplate="%{text}",
            colorscale="Blues",
            showscale=False,
            zmin=0, zmax=1,
        )
    )
    fig.update_layout(
        title=title,
        height=280,
        margin=dict(l=5, r=5, t=40, b=5),
        template="plotly_white",
        xaxis=dict(showticklabels=False),
        yaxis=dict(showticklabels=False, autorange="reversed"),
    )
    return fig


# ── Page ───────────────────────────────────────────────────────────────────

def show():
    st.title("Dyna-Q: Learning with a Model")
    st.markdown("**Section 6 — Planning and Learning · Chapter 8**")

    # ── Concept ───────────────────────────────────────────────────────────
    st.header("Planning vs Learning")
    st.markdown(
        """
Every RL algorithm we have built so far learns directly from experience — the agent takes a
real action in the environment and updates from the real outcome. This is **model-free** learning.

**Dyna-Q** does something different: it simultaneously *learns* from real experience AND
*plans* using a learned model of the environment.

After each real environment step, Dyna-Q:
1. Updates Q directly from the real (s, a, r, s') tuple — just like Q-learning
2. Updates the model: records that state s + action a led to reward r and next state s'
3. **Plans n times**: randomly samples past (s, a) pairs from the model, and runs additional
   Q-learning updates on those *simulated* transitions

The model is just a dictionary: `model[(s, a)] → (r, s')`. Each planning step is a free
simulated experience — and with enough planning steps, the agent can learn a good policy
from very few real environment interactions.
"""
    )

    st.latex(
        r"\text{Planning update: } "
        r"Q(s, a) \leftarrow Q(s, a) + \alpha\bigl[r + \gamma \max_{a'} Q(s', a') - Q(s, a)\bigr]"
    )

    st.info(
        "**n = 0 is pure Q-learning** — no planning. Each additional planning step is a free "
        "simulated experience. With n = 50, the agent can effectively squeeze 50× more learning "
        "out of each real environment step. This is the core idea behind model-based RL."
    )

    with st.expander("Deep Dive — When Does Planning Help?"):
        st.markdown(
            """
Planning helps most when:
- **Real interactions are expensive** (robotics, clinical trials, anything that takes real time
  or has real costs). A learned model lets you simulate thousands of experiences cheaply.
- **The model is accurate** — if the model is wrong, planning propagates errors. In simple
  deterministic environments like this maze, the model is perfect after one visit.
- **The task has sparse rewards** — planning lets reward information propagate backward through
  the model quickly, rather than waiting for the agent to stumble across it again.

Planning helps less when:
- The environment is **stochastic or non-stationary** — the model may be outdated or wrong.
- **Real interactions are cheap** — if you can collect millions of samples, model-free methods
  often catch up.
"""
        )

    st.divider()

    # ── Controls ──────────────────────────────────────────────────────────
    st.header("Interactive Simulation")
    st.markdown(
        "The agent navigates a maze from **S** (start, bottom-left) to **G** (goal). "
        "Compare how many real environment steps are needed to find the goal with different "
        "amounts of planning."
    )

    col1, col2 = st.columns(2)
    with col1:
        total_steps = st.slider("Total environment steps", 500, 5000, 3000, 500)
        alpha = st.slider("α (step size)", 0.1, 1.0, 0.5, 0.05)
        gamma = st.slider("γ (discount)", 0.9, 1.0, 0.95, 0.01)
    with col2:
        epsilon = st.slider("ε (exploration)", 0.01, 0.3, 0.1, 0.01)
        n_options = [0, 1, 5, 10, 25, 50]
        selected_ns = st.multiselect(
            "Planning steps (n) to compare",
            n_options,
            default=[0, 5, 50],
        )

    if not selected_ns:
        st.warning("Select at least one value for n.")
        return

    if st.button("Run Simulation", type="primary"):
        with st.spinner("Running Dyna-Q agents..."):
            sweep = run_dynaq_sweep(
                tuple(sorted(selected_ns)), alpha, gamma, epsilon, total_steps
            )

        st.header("Results")

        COLOURS = ["#636EFA", "#EF553B", "#00CC96", "#AB63FA", "#FFA15A", "#19D3F3"]

        # ── Cumulative reward ──────────────────────────────────────────
        fig = go.Figure()
        for i, n in enumerate(sorted(selected_ns)):
            fig.add_trace(go.Scatter(
                x=list(range(1, total_steps + 1)),
                y=sweep[n],
                mode="lines",
                name=f"n = {n}",
                line=dict(color=COLOURS[i % len(COLOURS)], width=2),
            ))
        fig.update_layout(
            title="Cumulative Reward vs Environment Steps",
            xaxis_title="Environment Steps",
            yaxis_title="Cumulative Reward",
            template="plotly_white",
            height=400,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        st.plotly_chart(fig, use_container_width=True)

        # ── Maze with learned policy ───────────────────────────────────
        st.subheader("Learned Policy (largest n selected)")
        best_n = max(selected_ns)
        _, _, Q = run_dynaq(best_n, alpha, gamma, epsilon, total_steps)
        st.plotly_chart(_maze_figure(Q, title=f"Learned Policy (n={best_n})"), use_container_width=True)

        # ── Summary ────────────────────────────────────────────────────
        st.header("Key Takeaways")
        metric_cols = st.columns(len(selected_ns))
        for i, n in enumerate(sorted(selected_ns)):
            with metric_cols[i]:
                total_goals = int(sweep[n][-1])
                st.metric(f"n = {n}", f"{total_goals} goals reached",
                          help="Total number of times the agent reached the goal")

        st.info(
            "More planning steps → more goals reached with the same number of real environment "
            "steps. The cumulative reward curve rises faster because each real step generates n "
            "additional simulated updates that propagate value information backward through "
            "the maze more quickly."
        )
