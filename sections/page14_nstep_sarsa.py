import numpy as np
import plotly.graph_objects as go
import streamlit as st


# ── Environment: 4×4 GridWorld ─────────────────────────────────────────────

GRID_H, GRID_W = 4, 4
N_STATES = GRID_H * GRID_W
N_ACTIONS = 4  # 0=up, 1=right, 2=down, 3=left
GOAL = N_STATES - 1  # bottom-right corner

_DELTAS = [(-1, 0), (0, 1), (1, 0), (0, -1)]
_ACTION_LABELS = ["↑", "→", "↓", "←"]


def _step(state: int, action: int) -> tuple[int, float, bool]:
    if state == GOAL:
        return state, 0.0, True
    r, c = divmod(state, GRID_W)
    dr, dc = _DELTAS[action]
    nr, nc = r + dr, c + dc
    if not (0 <= nr < GRID_H and 0 <= nc < GRID_W):
        nr, nc = r, c  # bounce off wall
    next_state = nr * GRID_W + nc
    done = next_state == GOAL
    return next_state, -1.0, done  # -1 per step until goal


# ── Cached simulation ──────────────────────────────────────────────────────

@st.cache_data
def run_nstep_sarsa(
    n: int,
    alpha: float,
    gamma: float,
    epsilon: float,
    n_episodes: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    n-step SARSA on 4x4 GridWorld.
    Returns (episode_returns [n_episodes], Q [N_STATES, N_ACTIONS]).
    """
    rng = np.random.default_rng(42)
    Q = np.zeros((N_STATES, N_ACTIONS))
    episode_returns = np.zeros(n_episodes)

    for ep in range(n_episodes):
        state = 0
        # ε-greedy action selection
        if rng.random() < epsilon:
            action = rng.integers(0, N_ACTIONS)
        else:
            action = int(np.argmax(Q[state]))

        states = [state]
        actions = [action]
        rewards = [0.0]
        T = float("inf")
        t = 0
        total_reward = 0.0

        while True:
            if t < T:
                next_state, reward, done = _step(states[t], actions[t])
                rewards.append(reward)
                states.append(next_state)
                total_reward += reward
                if done:
                    T = t + 1
                else:
                    if rng.random() < epsilon:
                        next_action = rng.integers(0, N_ACTIONS)
                    else:
                        next_action = int(np.argmax(Q[next_state]))
                    actions.append(next_action)

            tau = t - n + 1
            if tau >= 0:
                end = int(min(tau + n, T))
                G = sum(
                    gamma ** (i - tau - 1) * rewards[i]
                    for i in range(tau + 1, end + 1)
                )
                if tau + n < T:
                    G += gamma ** n * Q[states[tau + n], actions[tau + n]]
                s_tau, a_tau = states[tau], actions[tau]
                Q[s_tau, a_tau] += alpha * (G - Q[s_tau, a_tau])

            t += 1
            if tau == T - 1:
                break

        episode_returns[ep] = total_reward

    return episode_returns, Q


@st.cache_data
def run_sarsa_sweep(
    n_values: tuple[int, ...],
    alpha: float,
    gamma: float,
    epsilon: float,
    n_episodes: int,
) -> dict[int, np.ndarray]:
    return {n: run_nstep_sarsa(n, alpha, gamma, epsilon, n_episodes)[0] for n in n_values}


# ── Page ───────────────────────────────────────────────────────────────────

def show():
    st.title("n-step SARSA Control")
    st.markdown("**Section 5 — n-step Bootstrapping · Chapter 7**")

    # ── Concept ───────────────────────────────────────────────────────────
    st.header("From Prediction to Control")
    st.markdown(
        """
Page 13 used n-step TD to *evaluate* a fixed policy (predict state values). Here we extend
n-step bootstrapping to *control* — learning the optimal policy by updating **action values**
Q(s, a) rather than state values V(s).

The algorithm is **n-step SARSA**: collect n real steps of (state, action, reward) tuples, then
compute the n-step return and use it to update Q for the first state-action pair in that window.
"""
    )

    st.latex(
        r"G_t^{(n)} = R_{t+1} + \gamma R_{t+2} + \cdots + \gamma^{n-1} R_{t+n}"
        r"+ \gamma^n Q(S_{t+n}, A_{t+n})"
    )
    st.latex(
        r"Q(S_t, A_t) \;\leftarrow\; Q(S_t, A_t)"
        r"+ \alpha\bigl[G_t^{(n)} - Q(S_t, A_t)\bigr]"
    )

    st.success(
        "**When does larger n help?** In tasks with delayed rewards — where many steps must pass "
        "before a meaningful reward arrives — small n is slow because each update only propagates "
        "credit one step at a time. Larger n lets a single reward update many earlier decisions "
        "in one shot, speeding up learning considerably."
    )

    with st.expander("Deep Dive — Why SARSA, Not Q-Learning?"):
        st.markdown(
            """
n-step SARSA uses the *actual actions taken* in its bootstrapped return. This makes it
an **on-policy** algorithm — it evaluates and improves the same ε-greedy policy it uses
to collect data.

n-step Q-learning (called n-step Expected SARSA or Tree Backup in off-policy form) would
instead use the *best* action at the end of the n-step window, making it off-policy.
On-policy SARSA is more stable on tasks where the exploration policy matters for safety
(e.g. near cliffs), as you saw in Cliff Walking on Page 12.
"""
        )

    st.divider()

    # ── Controls ──────────────────────────────────────────────────────────
    st.header("Interactive Simulation")
    st.markdown(
        "The agent navigates a **4×4 GridWorld** from the top-left to the bottom-right corner. "
        "Every step costs −1. Compare how quickly different n values find the optimal path."
    )

    col1, col2 = st.columns(2)
    with col1:
        n_episodes = st.slider("Episodes", 100, 2000, 500, 100)
        alpha = st.slider("α (step size)", 0.01, 0.5, 0.1, 0.01)
        gamma = st.slider("γ (discount)", 0.9, 1.0, 1.0, 0.01)
    with col2:
        epsilon = st.slider("ε (exploration)", 0.01, 0.3, 0.1, 0.01)
        n_options = [1, 2, 4, 8, 16]
        selected_ns = st.multiselect(
            "n values to compare", n_options, default=[1, 4, 16],
        )
        smooth_window = st.slider("Smoothing window (episodes)", 1, 50, 20)

    if not selected_ns:
        st.warning("Select at least one n value.")
        return

    if st.button("Run Simulation", type="primary"):
        with st.spinner("Training n-step SARSA agents..."):
            returns = run_sarsa_sweep(
                tuple(sorted(selected_ns)), alpha, gamma, epsilon, n_episodes
            )

        st.header("Results")

        COLOURS = ["#636EFA", "#EF553B", "#00CC96", "#AB63FA", "#FFA15A"]

        # ── Learning curves ────────────────────────────────────────────
        fig = go.Figure()
        x = list(range(1, n_episodes + 1))
        for i, n in enumerate(sorted(selected_ns)):
            raw = returns[n]
            if smooth_window > 1:
                kernel = np.ones(smooth_window) / smooth_window
                smoothed = np.convolve(raw, kernel, mode="same")
            else:
                smoothed = raw
            fig.add_trace(
                go.Scatter(
                    x=x, y=smoothed,
                    mode="lines",
                    name=f"n = {n}",
                    line=dict(color=COLOURS[i % len(COLOURS)], width=2),
                )
            )
        fig.update_layout(
            title="Episode Return vs Episodes (smoothed)",
            xaxis_title="Episode",
            yaxis_title="Total Return",
            template="plotly_white",
            height=400,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        st.plotly_chart(fig, use_container_width=True)

        # ── Policy heatmap ─────────────────────────────────────────────
        st.subheader("Learned Policies")
        n_cols = min(len(selected_ns), 3)
        policy_cols = st.columns(n_cols)

        for i, n in enumerate(sorted(selected_ns)):
            _, Q = run_nstep_sarsa(n, alpha, gamma, epsilon, n_episodes)
            policy = np.argmax(Q, axis=1).reshape(GRID_H, GRID_W)
            arrows = np.vectorize(lambda a: _ACTION_LABELS[a])(policy)
            arrows[GRID_H - 1, GRID_W - 1] = "★"  # goal

            z = np.max(Q, axis=1).reshape(GRID_H, GRID_W)
            text = [[f"{arrows[r,c]}<br>{z[r,c]:.1f}" for c in range(GRID_W)] for r in range(GRID_H)]

            fig_p = go.Figure(
                go.Heatmap(
                    z=z,
                    text=text,
                    texttemplate="%{text}",
                    colorscale="Blues",
                    showscale=False,
                )
            )
            fig_p.update_layout(
                title=f"n = {n}  (arrow = best action, value = max Q)",
                height=280,
                margin=dict(l=10, r=10, t=40, b=10),
                template="plotly_white",
                xaxis=dict(showticklabels=False),
                yaxis=dict(showticklabels=False, autorange="reversed"),
            )
            with policy_cols[i % n_cols]:
                st.plotly_chart(fig_p, use_container_width=True)

        # ── Summary metrics ────────────────────────────────────────────
        st.header("Key Takeaways")
        metric_cols = st.columns(len(selected_ns))
        for i, n in enumerate(sorted(selected_ns)):
            last_100 = returns[n][-100:].mean()
            with metric_cols[i]:
                st.metric(f"n = {n}", f"{last_100:.1f}", help="Mean return over last 100 episodes")

        best_n = max(sorted(selected_ns), key=lambda n: returns[n][-100:].mean())
        st.info(
            f"**n = {best_n}** achieved the highest average return in the last 100 episodes "
            f"with these settings. The optimal path in a 4×4 grid is −6 (6 steps to goal)."
        )
