import numpy as np
import plotly.graph_objects as go
import streamlit as st

try:
    import gymnasium as gym

    _GYM_OK = True
except ImportError:
    _GYM_OK = False

from utils.tile_coding import TileCoder, POS_MIN, POS_MAX, VEL_MIN, VEL_MAX

N_ACTIONS = 3
ACTION_LABELS = ["Push Left", "Neutral", "Push Right"]


# ── Training ───────────────────────────────────────────────────────────────────


@st.cache_data
def run_semi_gradient_sarsa(
    n_tilings: int,
    n_tiles: int,
    alpha: float,
    epsilon: float,
    gamma: float,
    n_episodes: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Semi-gradient SARSA on Mountain Car.
    Returns (steps_per_episode [n_episodes], weights [n_actions, n_features]).
    """
    tc = TileCoder(n_tilings=n_tilings, n_tiles=n_tiles)
    n_feat = tc.n_features
    w = np.zeros((N_ACTIONS, n_feat))
    rng = np.random.default_rng(seed)
    env = gym.make("MountainCar-v0")
    steps_ep = np.zeros(n_episodes)

    def q(state: np.ndarray, action: int) -> float:
        return tc.value(w[action], state)

    def eps_greedy(state: np.ndarray) -> int:
        if rng.random() < epsilon:
            return int(rng.integers(N_ACTIONS))
        vals = np.array([q(state, a) for a in range(N_ACTIONS)])
        ties = np.flatnonzero(vals == vals.max())
        return int(rng.choice(ties))

    for ep in range(n_episodes):
        obs, _ = env.reset(seed=int(seed + ep))
        state = np.array(obs, dtype=float)
        action = eps_greedy(state)
        steps = 0

        while True:
            obs2, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            state2 = np.array(obs2, dtype=float)
            steps += 1

            if done:
                target = float(reward)
                tiles = tc.active_tiles(state)
                error = target - w[action][tiles].sum()
                w[action][tiles] += (alpha / n_tilings) * error
                break

            action2 = eps_greedy(state2)
            target = float(reward) + gamma * q(state2, action2)
            tiles = tc.active_tiles(state)
            error = target - w[action][tiles].sum()
            w[action][tiles] += (alpha / n_tilings) * error

            state = state2
            action = action2

        steps_ep[ep] = steps

    env.close()
    return steps_ep, w


# ── Visualisations ─────────────────────────────────────────────────────────────


def _steps_fig(steps_ep: np.ndarray, smooth: int) -> go.Figure:
    eps = list(range(1, len(steps_ep) + 1))
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=eps,
            y=steps_ep.tolist(),
            mode="lines",
            name="Steps / episode",
            line=dict(color="#636EFA", width=1),
            opacity=0.3,
        )
    )
    if smooth > 1 and len(steps_ep) >= smooth:
        kernel = np.ones(smooth) / smooth
        smoothed = np.convolve(steps_ep, kernel, mode="valid")
        x_smooth = list(range(smooth, len(steps_ep) + 1))
        fig.add_trace(
            go.Scatter(
                x=x_smooth,
                y=smoothed.tolist(),
                mode="lines",
                name=f"Rolling mean ({smooth})",
                line=dict(color="#EF553B", width=2),
            )
        )
    fig.update_layout(
        title="Steps per Episode",
        xaxis_title="Episode",
        yaxis_title="Steps",
        template="plotly_white",
        height=350,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


def _policy_heatmap(w: np.ndarray, tc: TileCoder) -> go.Figure:
    n_pos, n_vel = 80, 60
    pos_grid = np.linspace(POS_MIN, POS_MAX, n_pos)
    vel_grid = np.linspace(VEL_MIN, VEL_MAX, n_vel)
    policy = np.zeros((n_vel, n_pos), dtype=int)
    for i, vel in enumerate(vel_grid):
        for j, pos in enumerate(pos_grid):
            vals = np.array(
                [tc.value(w[a], np.array([pos, vel])) for a in range(N_ACTIONS)]
            )
            policy[i, j] = int(np.argmax(vals))
    fig = go.Figure(
        go.Heatmap(
            z=policy,
            x=pos_grid.tolist(),
            y=vel_grid.tolist(),
            colorscale=[[0, "#636EFA"], [0.5, "#555"], [1, "#EF553B"]],
            zmin=0,
            zmax=2,
            colorbar=dict(title="Action", tickvals=[0, 1, 2], ticktext=ACTION_LABELS),
        )
    )
    fig.update_layout(
        title="Greedy Policy",
        xaxis_title="Position",
        yaxis_title="Velocity",
        template="plotly_white",
        height=380,
    )
    return fig


def _cost_to_go_fig(w: np.ndarray, tc: TileCoder) -> go.Figure:
    n_pos, n_vel = 50, 40
    pos_grid = np.linspace(POS_MIN, POS_MAX, n_pos)
    vel_grid = np.linspace(VEL_MIN, VEL_MAX, n_vel)
    ctg = np.zeros((n_vel, n_pos))
    for i, vel in enumerate(vel_grid):
        for j, pos in enumerate(pos_grid):
            vals = np.array(
                [tc.value(w[a], np.array([pos, vel])) for a in range(N_ACTIONS)]
            )
            ctg[i, j] = -float(np.max(vals))
    fig = go.Figure(
        go.Surface(
            z=ctg,
            x=pos_grid.tolist(),
            y=vel_grid.tolist(),
            colorscale="Viridis",
            colorbar=dict(title="Cost"),
        )
    )
    fig.update_layout(
        title="Cost-to-go: −max_a Q(s,a)",
        scene=dict(
            xaxis_title="Position",
            yaxis_title="Velocity",
            zaxis_title="Cost",
        ),
        template="plotly_white",
        height=480,
    )
    return fig


# ── Page ───────────────────────────────────────────────────────────────────────


def show():
    st.title("Semi-gradient SARSA — Mountain Car")
    st.markdown(
        '<div style="background:#7B2D8B;height:3px;'
        'border-radius:2px;margin-bottom:1rem;"></div>',
        unsafe_allow_html=True,
    )
    st.markdown("**Section 8 — Function Approximation: Control (Chapter 10)**")

    if not _GYM_OK:
        st.error(
            "Gymnasium is not installed.  Install it with `pip install gymnasium` and restart the app."
        )
        return

    # ── Concept ───────────────────────────────────────────────────────────────
    st.header("From Prediction to Control")
    st.markdown(r"""
Chapter 10 extends function approximation to **control** (finding the optimal policy).
The key move is to approximate the **action-value function** $\hat{q}(s, a, \mathbf{w})$
rather than the state-value $\hat{v}$.

With **semi-gradient SARSA**, the update uses the next *chosen* action $A_{t+1}$
(on-policy):

$$\mathbf{w} \leftarrow \mathbf{w} + \alpha\bigl[R_{t+1} + \gamma\hat{q}(S_{t+1}, A_{t+1}, \mathbf{w})
- \hat{q}(S_t, A_t, \mathbf{w})\bigr]\nabla\hat{q}(S_t, A_t, \mathbf{w})$$

For a **linear tile-coded** approximator, separate weight vectors $\mathbf{w}_a$ are
kept per action: $\hat{q}(s,a,\mathbf{w}) = \mathbf{w}_a^\top\mathbf{1}_\text{active}(s)$.
The gradient is the binary active-tile indicator vector, and only the weights for
action $a$ are updated.
""")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Mountain Car task**")
        st.markdown(
            "A car starts in a valley with insufficient engine power to climb directly.  "
            "It must rock back and forth to build kinetic energy.  "
            "Reward is −1 per step; episode ends on reaching position ≥ 0.5 or at 200 steps."
        )
    with col2:
        st.markdown("**Tile coding**")
        st.markdown(
            "Overlapping tilings discretise the continuous 2-D state.  "
            "Exactly $n\\_tilings$ tiles fire for any state, giving sparse binary features.  "
            "Nearby states share tiles, providing natural generalisation."
        )
        st.latex(r"n_\text{features} = n_\text{tilings} \times (n_\text{tiles}+1)^2")

    st.info(
        "**Why SARSA and not Q-learning here?**  Semi-gradient Q-learning (off-policy TD "
        "with function approximation) can diverge even on simple tasks — it hits the "
        "*deadly triad*.  SARSA is on-policy, so the update distribution matches the "
        "behaviour, which avoids this instability for linear approximators."
    )

    with st.expander("Deep Dive — Semi-gradient vs True Gradient for Control"):
        st.markdown(r"""
For episodic control, the target for the MSVE objective would require the **true**
$q_\pi(s,a)$.  We don't have that, so we substitute a bootstrapped estimate
$R_{t+1} + \gamma\hat{q}(S_{t+1},A_{t+1},\mathbf{w})$.

Since this target depends on $\mathbf{w}$ itself, the full gradient of the loss
would include a term $\nabla\hat{q}(S_{t+1},A_{t+1},\mathbf{w})$.  We drop it —
treating the target as a constant — giving the *semi*-gradient update.

For **linear** approximators + on-policy methods, this still converges.  For
non-linear approximators or off-policy training, convergence is not guaranteed
(the deadly triad again).
""")

    st.divider()

    # ── Simulation ─────────────────────────────────────────────────────────────
    st.header("Interactive Simulation")

    col1, col2 = st.columns(2)
    with col1:
        n_tilings = st.select_slider("Number of tilings", [2, 4, 8, 16], value=8)
        n_tiles = st.select_slider("Tiles per dimension", [4, 8, 16], value=8)
        alpha = st.slider("α (step size)", 0.01, 1.0, 0.5, 0.01)
        n_feat_per = n_tilings * (n_tiles + 1) ** 2
        st.caption(
            f"Features per action: **{n_feat_per}** — total parameters: **{N_ACTIONS * n_feat_per}**"
        )
    with col2:
        epsilon = st.slider("ε (exploration)", 0.0, 0.3, 0.05, 0.01)
        gamma = st.slider("γ (discount)", 0.9, 1.0, 1.0, 0.01)
        n_episodes = st.slider("Episodes", 50, 500, 200, 50)
        seed = st.number_input("Random seed", value=42, step=1)
        smooth = st.slider("Rolling mean window", 1, 50, 20)

    if st.button("Train Agent", type="primary"):
        with st.spinner("Training semi-gradient SARSA on Mountain Car…"):
            steps_ep, w = run_semi_gradient_sarsa(
                n_tilings,
                n_tiles,
                alpha,
                epsilon,
                float(gamma),
                n_episodes,
                int(seed),
            )
            tc = TileCoder(n_tilings=n_tilings, n_tiles=n_tiles)

        st.header("Results")
        st.plotly_chart(_steps_fig(steps_ep, smooth), use_container_width=True)

        col_a, col_b, col_c = st.columns(3)
        early = max(1, n_episodes // 5)
        with col_a:
            st.metric("Mean steps (first 20%)", f"{steps_ep[:early].mean():.0f}")
        with col_b:
            st.metric("Mean steps (last 20%)", f"{steps_ep[-early:].mean():.0f}")
        with col_c:
            pct = 100 * float((steps_ep < 200).mean())
            st.metric("Episodes solved (<200 steps)", f"{pct:.0f}%")

        tab1, tab2 = st.tabs(["Greedy Policy", "Cost-to-go Surface"])
        with tab1:
            st.plotly_chart(_policy_heatmap(w, tc), use_container_width=True)
            st.caption(
                "Blue = push left, red = push right.  "
                "An effective policy alternates based on velocity to build momentum toward the goal."
            )
        with tab2:
            st.plotly_chart(_cost_to_go_fig(w, tc), use_container_width=True)
            st.caption(
                "Cost-to-go = −max_a Q(s,a).  Lower = closer to goal.  "
                "The valley bottom (low position, zero velocity) has highest cost."
            )

        st.header("Key Takeaways")
        final_mean = steps_ep[-early:].mean()
        if final_mean < 150:
            st.success(
                f"Agent consistently solves Mountain Car ({final_mean:.0f} mean steps in last 20%). "
                "The tile-coded Q-function has captured the momentum-building strategy."
            )
        else:
            st.warning(
                f"Agent is still learning ({final_mean:.0f} mean steps in last 20%). "
                "Try more episodes, or increase tilings for finer resolution."
            )
        st.info(
            "**Try:** α=0.5, 8 tilings, 8 tiles, ε=0.05, 300 episodes — "
            "this is close to the S&B Example 10.1 setup.  "
            "Reducing ε to 0 after training shows whether the greedy policy is stable."
        )
