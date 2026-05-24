import numpy as np
import plotly.graph_objects as go
import streamlit as st

# ── Maze definitions ───────────────────────────────────────────────────────

MAZE_H, MAZE_W = 6, 9
N_ACTIONS = 4
_DELTAS = [(-1, 0), (0, 1), (1, 0), (0, -1)]
START = (5, 3)
GOAL = (0, 8)


def _rc_to_s(r, c):
    return r * MAZE_W + c


def _s_to_rc(s):
    return divmod(s, MAZE_W)


# Blocking maze: wall blocks the only path at step CHANGE_STEP
BLOCKING_WALLS_BEFORE = {(_rc_to_s(3, c),) for c in range(0, 8)}  # row 3, cols 0-7
BLOCKING_WALLS_BEFORE = {(3, c) for c in range(0, 8)}
BLOCKING_WALLS_AFTER = {
    (3, c) for c in range(1, 9)
}  # wall shifts right, now blocks old path

# Shortcut maze: shortcut opens at step CHANGE_STEP
SHORTCUT_WALLS_BEFORE = {(3, c) for c in range(1, 9)}
SHORTCUT_WALLS_AFTER = {(3, c) for c in range(1, 8)}  # shortcut opens at col 8

CHANGE_STEP = 3000


def _make_step_fn(walls: set) -> callable:
    def step(state: int, action: int) -> tuple[int, float, bool]:
        r, c = _s_to_rc(state)
        dr, dc = _DELTAS[action]
        nr, nc = r + dr, c + dc
        if not (0 <= nr < MAZE_H and 0 <= nc < MAZE_W) or (nr, nc) in walls:
            nr, nc = r, c
        next_state = _rc_to_s(nr, nc)
        done = (nr, nc) == GOAL
        return next_state, 1.0 if done else 0.0, done

    return step


# ── Dyna-Q ─────────────────────────────────────────────────────────────────


def _run_dynaq_changing(
    walls_before: set,
    walls_after: set,
    n_planning: int,
    alpha: float,
    gamma: float,
    epsilon: float,
    total_steps: int,
    kappa: float = 0.0,
) -> np.ndarray:
    """Dyna-Q (kappa=0) or Dyna-Q+ (kappa>0) on a changing maze."""
    rng = np.random.default_rng(42)
    N_STATES = MAZE_H * MAZE_W
    Q = np.zeros((N_STATES, N_ACTIONS))
    model: dict[tuple[int, int], tuple[int, float]] = {}
    seen_sa: list[tuple[int, int]] = []
    time_since: dict[tuple[int, int], int] = {}  # for Dyna-Q+ exploration bonus

    start_s = _rc_to_s(*START)
    cumulative = np.zeros(total_steps)

    step_fn_before = _make_step_fn(walls_before)
    step_fn_after = _make_step_fn(walls_after)

    state = start_s
    ep_steps = 0

    for step in range(total_steps):
        step_fn = step_fn_before if step < CHANGE_STEP else step_fn_after

        # ε-greedy
        if rng.random() < epsilon:
            action = int(rng.integers(0, N_ACTIONS))
        else:
            action = int(np.argmax(Q[state]))

        next_state, reward, done = step_fn(state, action)

        # Direct Q-learning update
        Q[state, action] += alpha * (
            reward + gamma * np.max(Q[next_state]) - Q[state, action]
        )

        # Model update
        sa = (state, action)
        if sa not in model:
            seen_sa.append(sa)
        model[sa] = (next_state, reward)
        time_since[sa] = 0

        # Increment time for all other (s,a) pairs
        for k in time_since:
            if k != sa:
                time_since[k] += 1

        # Planning
        for _ in range(n_planning):
            idx = rng.integers(0, len(seen_sa))
            s_p, a_p = seen_sa[idx]
            ns_p, r_p = model[(s_p, a_p)]
            if kappa > 0:
                tau = time_since.get((s_p, a_p), 0)
                r_p = r_p + kappa * np.sqrt(tau)
            Q[s_p, a_p] += alpha * (r_p + gamma * np.max(Q[ns_p]) - Q[s_p, a_p])

        cumulative[step] = reward if step == 0 else cumulative[step - 1] + reward

        ep_steps += 1
        if done or ep_steps > 5000:
            state = start_s
            ep_steps = 0
        else:
            state = next_state

    return cumulative


@st.cache_data
def run_comparison(
    maze_type: str,
    n_planning: int,
    alpha: float,
    gamma: float,
    epsilon: float,
    kappa: float,
    total_steps: int,
) -> tuple[np.ndarray, np.ndarray]:
    if maze_type == "Blocking Maze":
        walls_before = BLOCKING_WALLS_BEFORE
        walls_after = BLOCKING_WALLS_AFTER
    else:
        walls_before = SHORTCUT_WALLS_BEFORE
        walls_after = SHORTCUT_WALLS_AFTER

    dq = _run_dynaq_changing(
        walls_before,
        walls_after,
        n_planning,
        alpha,
        gamma,
        epsilon,
        total_steps,
        kappa=0.0,
    )
    dq_plus = _run_dynaq_changing(
        walls_before,
        walls_after,
        n_planning,
        alpha,
        gamma,
        epsilon,
        total_steps,
        kappa=kappa,
    )
    return dq, dq_plus


def _walls_figure(walls: set, title: str) -> go.Figure:
    z = [[0.3] * MAZE_W for _ in range(MAZE_H)]
    text = [["·"] * MAZE_W for _ in range(MAZE_H)]
    for r, c in walls:
        z[r][c] = 0.0
        text[r][c] = "■"
    gr, gc = GOAL
    sr, sc = START
    z[gr][gc] = 1.0
    text[gr][gc] = "G"
    z[sr][sc] = 0.5
    text[sr][sc] = "S"

    fig = go.Figure(
        go.Heatmap(
            z=z,
            text=text,
            texttemplate="%{text}",
            colorscale="Blues",
            showscale=False,
            zmin=0,
            zmax=1,
        )
    )
    fig.update_layout(
        title=title,
        height=200,
        margin=dict(l=5, r=5, t=35, b=5),
        template="plotly_white",
        xaxis=dict(showticklabels=False),
        yaxis=dict(showticklabels=False, autorange="reversed"),
    )
    return fig


# ── Page ───────────────────────────────────────────────────────────────────


def show():
    st.title("Dyna-Q vs Dyna-Q+: Changing Environments")
    st.markdown("**Section 6 — Planning and Learning · Chapter 8**")

    # ── Concept ───────────────────────────────────────────────────────────
    st.header("When the World Changes")
    st.markdown("""
Dyna-Q assumes the environment is **stationary** — the model it builds is always correct.
But real environments change. If a maze is restructured mid-training, Dyna-Q's model becomes
stale and it may keep planning using the old, incorrect transitions.

**Dyna-Q+** addresses this with an **exploration bonus**: state-action pairs that have
*not been tried recently* get a bonus reward added to their simulated experience. This
encourages the agent to re-explore parts of the environment it hasn't visited in a while,
allowing it to detect and adapt to changes.
""")

    st.latex(r"r^+ = r + \kappa \sqrt{\tau(s, a)}")
    st.markdown(
        r"where $\tau(s, a)$ is the number of steps since state-action pair $(s, a)$ was last "
        r"tried, and $\kappa$ is a small bonus coefficient. Rarely-visited pairs get a larger "
        r"bonus, nudging the agent to go check on them."
    )

    st.warning(
        "**Dyna-Q fails to recover quickly** when the environment changes — its model "
        "confidently replays stale transitions. Dyna-Q+ is slower to converge in stationary "
        "environments (the bonus adds noise) but far more robust when things shift."
    )

    with st.expander("Deep Dive — Two Types of Change"):
        st.markdown("""
**Blocking maze:** an initially open path becomes blocked mid-training. Dyna-Q may keep
trying the now-blocked path because its model says it leads somewhere good. Dyna-Q+ will
notice that it hasn't visited those states recently and explore, discovering the block.

**Shortcut maze:** a new shorter path opens mid-training. Dyna-Q will find the new shortcut
only if random exploration happens to stumble across it. Dyna-Q+ actively seeks out
unexplored regions, making it much more likely to discover — and then exploit — the shortcut.

This is why exploration bonuses are a form of **intrinsic motivation**: the agent is
rewarded not just for external task rewards but for novelty and revisiting uncertain areas.
""")

    st.divider()

    # ── Controls ──────────────────────────────────────────────────────────
    st.header("Interactive Simulation")

    col1, col2 = st.columns(2)
    with col1:
        maze_type = st.radio(
            "Maze type",
            ["Blocking Maze", "Shortcut Maze"],
            help="Blocking: open path becomes blocked at step 3000. "
            "Shortcut: new shorter path opens at step 3000.",
        )
        total_steps = st.slider("Total steps", 2000, 10000, 6000, 500)
        n_planning = st.slider("Planning steps (n)", 1, 50, 10)
    with col2:
        alpha = st.slider("α (step size)", 0.1, 1.0, 0.5, 0.05)
        gamma = st.slider("γ (discount)", 0.9, 1.0, 0.95, 0.01)
        epsilon = st.slider("ε (exploration)", 0.01, 0.3, 0.1, 0.01)
        kappa = st.slider(
            "κ (exploration bonus coefficient)",
            0.001,
            0.1,
            0.01,
            0.001,
            format="%.3f",
            help="Scales the √τ bonus. Larger κ → more re-exploration.",
        )

    # ── Maze previews ──────────────────────────────────────────────────────
    st.subheader("Environment Layout")
    if maze_type == "Blocking Maze":
        walls_b, walls_a = BLOCKING_WALLS_BEFORE, BLOCKING_WALLS_AFTER
        change_desc = "A path is **blocked** at step 3,000, forcing the agent to find a new route."
    else:
        walls_b, walls_a = SHORTCUT_WALLS_BEFORE, SHORTCUT_WALLS_AFTER
        change_desc = (
            "A **shortcut** opens at step 3,000 — Dyna-Q+ should discover it faster."
        )

    st.caption(change_desc)
    map_col1, map_col2 = st.columns(2)
    with map_col1:
        st.plotly_chart(
            _walls_figure(walls_b, "Before change (steps 0–3000)"),
            use_container_width=True,
        )
    with map_col2:
        st.plotly_chart(
            _walls_figure(walls_a, "After change (steps 3000+)"),
            use_container_width=True,
        )

    if st.button("Run Comparison", type="primary"):
        with st.spinner("Running Dyna-Q and Dyna-Q+..."):
            dq, dq_plus = run_comparison(
                maze_type, n_planning, alpha, gamma, epsilon, kappa, total_steps
            )

        st.header("Results")

        x = list(range(1, total_steps + 1))
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=x,
                y=dq,
                mode="lines",
                name="Dyna-Q",
                line=dict(color="#636EFA", width=2),
            )
        )
        fig.add_trace(
            go.Scatter(
                x=x,
                y=dq_plus,
                mode="lines",
                name="Dyna-Q+",
                line=dict(color="#EF553B", width=2),
            )
        )
        fig.add_vline(
            x=CHANGE_STEP,
            line_dash="dash",
            line_color="gray",
            annotation_text="Environment changes",
            annotation_position="top right",
        )
        fig.update_layout(
            title=f"Cumulative Reward — {maze_type}",
            xaxis_title="Environment Steps",
            yaxis_title="Cumulative Reward",
            template="plotly_white",
            height=420,
            legend=dict(
                orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1
            ),
        )
        st.plotly_chart(fig, use_container_width=True)

        st.header("Key Takeaways")
        col_a, col_b = st.columns(2)
        with col_a:
            st.metric("Dyna-Q total reward", int(dq[-1]))
        with col_b:
            st.metric("Dyna-Q+ total reward", int(dq_plus[-1]))

        post_change = slice(CHANGE_STEP, total_steps)
        dq_post = dq[-1] - dq[CHANGE_STEP]
        dqp_post = dq_plus[-1] - dq_plus[CHANGE_STEP]

        if maze_type == "Shortcut Maze":
            st.success(
                f"After the shortcut opened, Dyna-Q+ accumulated {dqp_post:.0f} reward vs "
                f"Dyna-Q's {dq_post:.0f}. The exploration bonus drives faster shortcut discovery."
            )
        else:
            st.success(
                f"After the path was blocked, Dyna-Q+ accumulated {dqp_post:.0f} reward vs "
                f"Dyna-Q's {dq_post:.0f}. The exploration bonus helps detect and route around "
                f"the block faster."
            )

        st.info(
            f"Try increasing κ to see more aggressive re-exploration after the change, "
            f"or reduce it to see Dyna-Q+ behave more like plain Dyna-Q."
        )
