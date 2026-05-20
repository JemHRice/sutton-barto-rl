import heapq
import numpy as np
import plotly.graph_objects as go
import streamlit as st


# ── Reuse the same maze from Page 16 ──────────────────────────────────────

MAZE_H, MAZE_W = 7, 10
START = (6, 0)
GOAL = (5, 9)
_WALLS = {
    (0, 9),
    (1, 2), (1, 3), (1, 4), (1, 5), (1, 6), (1, 7), (1, 9),
    (2, 6), (3, 6),
}
N_STATES = MAZE_H * MAZE_W
N_ACTIONS = 4
_DELTAS = [(-1, 0), (0, 1), (1, 0), (0, -1)]
_ACTION_LABELS = ["↑", "→", "↓", "←"]


def _rc_to_s(r, c):
    return r * MAZE_W + c


def _s_to_rc(s):
    return divmod(s, MAZE_W)


def _is_wall(r, c):
    return (r, c) in _WALLS


def _maze_step(state: int, action: int) -> tuple[int, float, bool]:
    r, c = _s_to_rc(state)
    dr, dc = _DELTAS[action]
    nr, nc = r + dr, c + dc
    if not (0 <= nr < MAZE_H and 0 <= nc < MAZE_W) or _is_wall(nr, nc):
        nr, nc = r, c
    next_state = _rc_to_s(nr, nc)
    done = (nr, nc) == GOAL
    return next_state, 1.0 if done else 0.0, done


# ── Prioritised Sweeping ───────────────────────────────────────────────────

@st.cache_data
def run_prioritized_sweeping(
    theta: float,
    n_planning: int,
    alpha: float,
    gamma: float,
    epsilon: float,
    total_steps: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Prioritised sweeping: queue (s,a) pairs by |TD error|, update highest first.
    Returns:
      cumulative_rewards [total_steps]
      priority_snapshots [MAZE_H, MAZE_W] — max priority per state at end
      Q                  [N_STATES, N_ACTIONS]
    """
    rng = np.random.default_rng(42)
    Q = np.zeros((N_STATES, N_ACTIONS))
    model: dict[tuple[int, int], tuple[int, float]] = {}
    predecessors: dict[int, set] = {s: set() for s in range(N_STATES)}

    # Priority queue: heapq is a min-heap, so store negative priority
    pq: list[tuple[float, int, int]] = []
    in_queue: dict[tuple[int, int], float] = {}

    start_s = _rc_to_s(*START)
    cumulative = np.zeros(total_steps)
    state = start_s
    ep_steps = 0

    for step in range(total_steps):
        # ε-greedy — random tie-breaking prevents getting stuck when Q is all zeros
        if rng.random() < epsilon:
            action = int(rng.integers(0, N_ACTIONS))
        else:
            q_s = Q[state]
            action = int(rng.choice(np.flatnonzero(q_s == q_s.max())))

        next_state, reward, done = _maze_step(state, action)
        model[(state, action)] = (next_state, reward)
        predecessors[next_state].add((state, action))

        # Compute priority for (s, a)
        td_error = abs(reward + gamma * np.max(Q[next_state]) - Q[state, action])
        if td_error > theta:
            heapq.heappush(pq, (-td_error, state, action))
            in_queue[(state, action)] = td_error

        # Sweeping
        for _ in range(n_planning):
            if not pq:
                break
            neg_p, s_p, a_p = heapq.heappop(pq)
            in_queue.pop((s_p, a_p), None)

            ns_p, r_p = model[(s_p, a_p)]
            Q[s_p, a_p] += alpha * (r_p + gamma * np.max(Q[ns_p]) - Q[s_p, a_p])

            # Check predecessors of s_p
            for (pred_s, pred_a) in predecessors[s_p]:
                if (pred_s, pred_a) in model:
                    pred_ns, pred_r = model[(pred_s, pred_a)]
                    pred_td = abs(pred_r + gamma * np.max(Q[pred_ns]) - Q[pred_s, pred_a])
                    if pred_td > theta:
                        heapq.heappush(pq, (-pred_td, pred_s, pred_a))
                        in_queue[(pred_s, pred_a)] = pred_td

        cumulative[step] = reward if step == 0 else cumulative[step - 1] + reward
        ep_steps += 1

        if done or ep_steps > 2000:
            state = start_s
            ep_steps = 0
        else:
            state = next_state

    # Build priority snapshot (max Q range per state as proxy for priority)
    priority_map = np.zeros(N_STATES)
    for (s, a), p in in_queue.items():
        priority_map[s] = max(priority_map[s], p)

    return cumulative, priority_map.reshape(MAZE_H, MAZE_W), Q


@st.cache_data
def run_random_dyna(
    n_planning: int,
    alpha: float,
    gamma: float,
    epsilon: float,
    total_steps: int,
) -> np.ndarray:
    """Plain Dyna-Q with random planning (no prioritization) for comparison."""
    rng = np.random.default_rng(42)
    Q = np.zeros((N_STATES, N_ACTIONS))
    model: dict[tuple[int, int], tuple[int, float]] = {}
    seen_sa: list[tuple[int, int]] = []

    start_s = _rc_to_s(*START)
    cumulative = np.zeros(total_steps)
    state = start_s
    ep_steps = 0

    for step in range(total_steps):
        # ε-greedy — random tie-breaking prevents getting stuck when Q is all zeros
        if rng.random() < epsilon:
            action = int(rng.integers(0, N_ACTIONS))
        else:
            q_s = Q[state]
            action = int(rng.choice(np.flatnonzero(q_s == q_s.max())))

        next_state, reward, done = _maze_step(state, action)

        Q[state, action] += alpha * (
            reward + gamma * np.max(Q[next_state]) - Q[state, action]
        )

        sa = (state, action)
        if sa not in model:
            seen_sa.append(sa)
        model[sa] = (next_state, reward)

        for _ in range(n_planning):
            if seen_sa:
                idx = rng.integers(0, len(seen_sa))
                s_p, a_p = seen_sa[idx]
                ns_p, r_p = model[(s_p, a_p)]
                Q[s_p, a_p] += alpha * (r_p + gamma * np.max(Q[ns_p]) - Q[s_p, a_p])

        cumulative[step] = reward if step == 0 else cumulative[step - 1] + reward
        ep_steps += 1

        if done or ep_steps > 2000:
            state = start_s
            ep_steps = 0
        else:
            state = next_state

    return cumulative


# ── Page ───────────────────────────────────────────────────────────────────

def show():
    st.title("Prioritised Sweeping")
    st.markdown("**Section 6 — Planning and Learning · Chapter 8**")

    # ── Concept ───────────────────────────────────────────────────────────
    st.header("Smarter Planning")
    st.markdown(
        """
Plain Dyna-Q plans by picking **random** previously-seen state-action pairs and updating them.
This is wasteful — most updates are on states where the value is already accurate and the
TD error is near zero. Updating them again changes almost nothing.

**Prioritised Sweeping** directs planning effort where it matters most: towards the
state-action pairs with the largest **TD error** — the ones where the current Q estimate
is most wrong.
"""
    )

    st.markdown("After each real experience, compute the TD error for the observed (s, a):")
    st.latex(
        r"P(s,a) = \left| r + \gamma \max_{a'} Q(s', a') - Q(s, a) \right|"
    )
    st.markdown(
        "If this priority exceeds a threshold θ, insert (s, a) into a priority queue. "
        "At each planning step, pop the highest-priority pair, update it, then propagate "
        "backward to its **predecessors** — states that could lead to s — and add those to "
        "the queue if their priorities are also high."
    )

    st.success(
        "**The key insight:** when the agent first discovers the goal and receives reward 1, "
        "that state-action pair has a huge TD error. Prioritized sweeping immediately propagates "
        "this backward through the maze, updating all predecessor states in order of importance — "
        "rather than waiting for random planning to stumble across them."
    )

    with st.expander("Deep Dive — The Predecessor Propagation Loop"):
        st.markdown(
            r"""
After updating $Q(s, a)$, the values of all predecessor state-action pairs $(s^-, a^-)$ that
lead to $s$ may now be stale. We recompute their priorities:

$$P(s^-, a^-) = \left| r^- + \gamma \max_{a'} Q(s, a') - Q(s^-, a^-) \right|$$

If this exceeds $\theta$, we insert $(s^-, a^-)$ into the queue. This creates a cascade: a
single real experience can trigger a wave of updates rippling backward through the state space,
efficiently propagating value information to all states that eventually lead to the discovered
reward.

This is why prioritised sweeping can reach the goal in far fewer real environment steps than
random Dyna-Q — especially in large mazes with sparse rewards.
"""
        )

    st.divider()

    # ── Controls ──────────────────────────────────────────────────────────
    st.header("Interactive Comparison")

    col1, col2 = st.columns(2)
    with col1:
        total_steps = st.slider("Total environment steps", 500, 5000, 3000, 500)
        n_planning = st.slider("Planning steps per real step (n)", 1, 50, 10)
        alpha = st.slider("α (step size)", 0.1, 1.0, 0.5, 0.05)
    with col2:
        gamma = st.slider("γ (discount)", 0.9, 1.0, 0.95, 0.01)
        epsilon = st.slider("ε (exploration)", 0.01, 0.3, 0.1, 0.01)
        theta = st.slider(
            "θ (priority threshold)", 0.0001, 0.1, 0.001, 0.0001,
            format="%.4f",
            help="Minimum TD error to add to the priority queue. Smaller θ → more updates.",
        )

    if st.button("Run Comparison", type="primary"):
        with st.spinner("Running Prioritised Sweeping and random Dyna-Q..."):
            ps_cum, priority_map, Q = run_prioritized_sweeping(
                theta, n_planning, alpha, gamma, epsilon, total_steps
            )
            dq_cum = run_random_dyna(n_planning, alpha, gamma, epsilon, total_steps)

        st.header("Results")

        x = list(range(1, total_steps + 1))

        # ── Learning curves ────────────────────────────────────────────
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=x, y=dq_cum, mode="lines", name="Random Dyna-Q",
            line=dict(color="#636EFA", width=2),
        ))
        fig.add_trace(go.Scatter(
            x=x, y=ps_cum, mode="lines", name="Prioritised Sweeping",
            line=dict(color="#EF553B", width=2),
        ))
        fig.update_layout(
            title="Cumulative Reward vs Environment Steps",
            xaxis_title="Environment Steps",
            yaxis_title="Cumulative Reward",
            template="plotly_white",
            height=380,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        st.plotly_chart(fig, use_container_width=True)

        # ── Priority heatmap ───────────────────────────────────────────
        st.subheader("Priority Queue Heatmap")
        st.caption(
            "Colour intensity shows the remaining priority for each state. "
            "High priority = Q estimate still needs updating."
        )

        text_grid = [[""] * MAZE_W for _ in range(MAZE_H)]
        for r in range(MAZE_H):
            for c in range(MAZE_W):
                if _is_wall(r, c):
                    text_grid[r][c] = "■"
                elif (r, c) == GOAL:
                    text_grid[r][c] = "G"
                elif (r, c) == START:
                    text_grid[r][c] = "S"

        fig2 = go.Figure(go.Heatmap(
            z=priority_map,
            text=text_grid,
            texttemplate="%{text}",
            colorscale="Reds",
            showscale=True,
            zmin=0,
            colorbar=dict(title="Priority"),
        ))
        for r in range(MAZE_H):
            for c in range(MAZE_W):
                if _is_wall(r, c):
                    fig2.add_shape(
                        type="rect",
                        x0=c - 0.5, x1=c + 0.5,
                        y0=r - 0.5, y1=r + 0.5,
                        fillcolor="rgba(50,50,50,0.7)",
                        line=dict(width=0),
                    )
        fig2.update_layout(
            height=300,
            margin=dict(l=5, r=5, t=10, b=5),
            template="plotly_white",
            xaxis=dict(showticklabels=False),
            yaxis=dict(showticklabels=False, autorange="reversed"),
        )
        st.plotly_chart(fig2, use_container_width=True)

        # ── Summary ────────────────────────────────────────────────────
        st.header("Key Takeaways")
        col_a, col_b = st.columns(2)
        with col_a:
            st.metric("Random Dyna-Q total reward", int(dq_cum[-1]))
        with col_b:
            st.metric("Prioritised Sweeping total reward", int(ps_cum[-1]))

        improvement = (ps_cum[-1] - dq_cum[-1]) / max(dq_cum[-1], 1) * 100
        st.success(
            f"Prioritised Sweeping found the goal and accumulated reward {improvement:.0f}% "
            f"{'faster' if improvement > 0 else 'similarly'} than random Dyna-Q. "
            "The priority queue focuses computation on states where value estimates are most wrong, "
            "propagating reward information efficiently backward through the maze."
        )
        st.info(
            "Lower θ allows more state-action pairs into the queue — more thorough but slower "
            "per step. Higher θ is more selective. Try θ = 0 (everything goes in the queue) "
            "to see maximum propagation, or θ = 0.1 (very selective) for minimum updates."
        )
