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
def run_full_training(
    n_tilings: int,
    n_tiles: int,
    alpha: float,
    epsilon_start: float,
    epsilon_end: float,
    n_episodes: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, list[list[tuple[float, float, int]]]]:
    """
    Semi-gradient SARSA with epsilon decay.
    Returns (steps_per_episode, weights, episode_trajectories[0,25,50,75,100%]).
    """
    tc = TileCoder(n_tilings=n_tilings, n_tiles=n_tiles)
    n_feat = tc.n_features
    w = np.zeros((N_ACTIONS, n_feat))
    rng = np.random.default_rng(seed)
    env = gym.make("MountainCar-v0")
    steps_ep = np.zeros(n_episodes)
    snap_eps = {
        int(n_episodes * f): f"Episode {int(n_episodes * f)}"
        for f in [0.0, 0.25, 0.5, 0.75, 1.0]
    }
    trajectories: list[list[tuple[float, float, int]]] = []

    def q(state: np.ndarray, action: int) -> float:
        return tc.value(w[action], state)

    for ep in range(n_episodes):
        eps = epsilon_start + (epsilon_end - epsilon_start) * ep / max(
            1, n_episodes - 1
        )
        obs, _ = env.reset(seed=int(seed + ep))
        state = np.array(obs, dtype=float)
        vals = np.array([q(state, a) for a in range(N_ACTIONS)])
        ties = np.flatnonzero(vals == vals.max())
        action = (
            int(rng.choice(ties))
            if rng.random() >= eps
            else int(rng.integers(N_ACTIONS))
        )
        steps = 0
        traj: list[tuple[float, float, int]] = []

        while True:
            obs2, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            state2 = np.array(obs2, dtype=float)
            traj.append((float(state[0]), float(state[1]), action))
            steps += 1

            if done:
                target = float(reward)
                tiles = tc.active_tiles(state)
                error = target - w[action][tiles].sum()
                w[action][tiles] += (alpha / n_tilings) * error
                break

            vals2 = np.array([q(state2, a) for a in range(N_ACTIONS)])
            ties2 = np.flatnonzero(vals2 == vals2.max())
            if rng.random() >= eps:
                action2 = int(rng.choice(ties2))
            else:
                action2 = int(rng.integers(N_ACTIONS))

            target = float(reward) + q(state2, action2)
            tiles = tc.active_tiles(state)
            error = target - w[action][tiles].sum()
            w[action][tiles] += (alpha / n_tilings) * error

            state = state2
            action = action2

        steps_ep[ep] = steps
        if ep in snap_eps:
            trajectories.append(traj)

    env.close()
    return steps_ep, w, trajectories


# ── Figures ────────────────────────────────────────────────────────────────────


def _trajectory_fig(
    trajectories: list[list[tuple[float, float, int]]],
    n_episodes: int,
) -> go.Figure:
    fracs = [0.0, 0.25, 0.5, 0.75, 1.0]
    labels = [f"Ep {int(n_episodes * f) or 1}" for f in fracs]
    colours = ["#636EFA", "#AB63FA", "#00CC96", "#FFA15A", "#EF553B"]
    fig = go.Figure()
    for i, (traj, label, col) in enumerate(zip(trajectories, labels, colours)):
        if not traj:
            continue
        positions = [t[0] for t in traj]
        steps_x = list(range(len(positions)))
        fig.add_trace(
            go.Scatter(
                x=steps_x,
                y=positions,
                mode="lines",
                name=label,
                line=dict(color=col, width=2),
            )
        )
    fig.add_hline(
        y=0.5, line_dash="dash", line_color="#00CC96", annotation_text="Goal (pos=0.5)"
    )
    fig.update_layout(
        title="Position Trajectory at Training Snapshots",
        xaxis_title="Step within episode",
        yaxis_title="Position",
        template="plotly_white",
        height=380,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


def _learning_curve_fig(steps_ep: np.ndarray, smooth: int) -> go.Figure:
    eps = list(range(1, len(steps_ep) + 1))
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=eps,
            y=steps_ep.tolist(),
            mode="lines",
            name="Raw steps",
            line=dict(color="#636EFA", width=1),
            opacity=0.25,
            showlegend=False,
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
        title="Learning Curve",
        xaxis_title="Episode",
        yaxis_title="Steps",
        template="plotly_white",
        height=340,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


def _cost_surface(w: np.ndarray, tc: TileCoder) -> go.Figure:
    n_pos, n_vel = 60, 45
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
            colorscale="Plasma",
            colorbar=dict(title="Cost"),
        )
    )
    fig.update_layout(
        title="Cost-to-go Surface (−max_a Q(s,a))",
        scene=dict(
            xaxis_title="Position",
            yaxis_title="Velocity",
            zaxis_title="Cost-to-go",
        ),
        template="plotly_white",
        height=500,
    )
    return fig


def _value_contour(w: np.ndarray, tc: TileCoder) -> go.Figure:
    n_pos, n_vel = 80, 60
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
        go.Contour(
            z=ctg,
            x=pos_grid.tolist(),
            y=vel_grid.tolist(),
            colorscale="Plasma",
            colorbar=dict(title="Cost"),
            contours=dict(showlabels=True),
        )
    )
    # Mark start region
    fig.add_trace(
        go.Scatter(
            x=[-0.5],
            y=[0.0],
            mode="markers",
            name="Typical start",
            marker=dict(symbol="star", size=14, color="yellow"),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[0.5],
            y=[0.0],
            mode="markers",
            name="Goal",
            marker=dict(symbol="diamond", size=12, color="lime"),
        )
    )
    fig.update_layout(
        title="Cost-to-go Contour",
        xaxis_title="Position",
        yaxis_title="Velocity",
        template="plotly_white",
        height=420,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


# ── Page ───────────────────────────────────────────────────────────────────────


def show():
    st.title("Mountain Car Solver")
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
    st.header("The Mountain Car Problem")
    st.markdown(r"""
Mountain Car is the canonical benchmark for **continuous-state control** with sparse rewards.
It challenges the agent in two ways:

1. **Sparse reward**: reward is −1 every step, +0 on reaching the goal.
   There is no gradient pointing toward the goal — the agent must explore to find it.
2. **Deceptive physics**: the car cannot drive straight up; it must first reverse to build
   kinetic energy.  The optimal strategy is counter-intuitive and requires the value function
   to capture velocity as well as position.

The state space is 2-D and continuous: $(x_t, \dot{x}_t)$ with
$x \in [-1.2, 0.5]$ and $\dot{x} \in [-0.07, 0.07]$.
""")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Physics**")
        st.latex(r"\dot{x}_{t+1} = \dot{x}_t + 0.001 A_t - 0.0025\cos(3 x_t)")
        st.latex(r"x_{t+1} = x_t + \dot{x}_{t+1}")
        st.caption(
            "The gravity term $-0.0025\\cos(3x)$ is always pulling the car back.  "
            "Maximum engine force is 0.001 — much weaker than gravity at the hill's base."
        )
    with col2:
        st.markdown("**Why the value function looks the way it does**")
        st.markdown(
            "High positive velocity **and** position near the hill → close to goal → low cost.  "
            "High negative velocity near the left wall → about to reverse and build energy → also low cost.  "
            "The worst state is near position −0.5, zero velocity: no momentum, must restart."
        )

    with st.expander("Connection to S&B Example 10.1"):
        st.markdown(r"""
Figure 10.1 in Sutton & Barto shows the cost-to-go surface after 1, 12, and 104 episodes
of semi-gradient SARSA with 8 tilings of 8×8 tiles.  The surface starts flat (random
initialisation at zero), develops a ridge near the goal, and eventually shows the full
saddle shape with the valley at $(\approx -0.5, 0)$.

This page reproduces that experiment interactively.  The **Position Trajectory** panel
shows episodes at 0%, 25%, 50%, 75%, and 100% of training — you can watch the agent
transition from random thrashing to coordinated rocking.
""")

    st.divider()

    # ── Simulation ─────────────────────────────────────────────────────────────
    st.header("Train and Visualise")

    col1, col2 = st.columns(2)
    with col1:
        n_tilings = st.select_slider("Number of tilings", [2, 4, 8, 16], value=8)
        n_tiles = st.select_slider("Tiles per dimension", [4, 8, 16], value=8)
        alpha = st.slider("α (step size)", 0.01, 1.0, 0.5, 0.01)
        n_feat_per = n_tilings * (n_tiles + 1) ** 2
        st.caption(
            f"Features per action: **{n_feat_per}** — total: **{N_ACTIONS * n_feat_per}**"
        )
    with col2:
        epsilon_start = st.slider("ε start (exploration)", 0.0, 0.5, 0.1, 0.01)
        epsilon_end = st.slider("ε end (at final episode)", 0.0, 0.2, 0.0, 0.01)
        n_episodes = st.slider("Episodes", 50, 1000, 400, 50)
        seed = st.number_input("Random seed", value=42, step=1)
        smooth = st.slider("Rolling mean window", 1, 50, 20)

    if st.button("Train Agent", type="primary"):
        with st.spinner("Training… this may take a moment for many episodes."):
            steps_ep, w, trajs = run_full_training(
                n_tilings,
                n_tiles,
                alpha,
                epsilon_start,
                epsilon_end,
                n_episodes,
                int(seed),
            )
            tc = TileCoder(n_tilings=n_tilings, n_tiles=n_tiles)

        st.header("Results")

        col_a, col_b, col_c = st.columns(3)
        early = max(1, n_episodes // 5)
        with col_a:
            st.metric("Mean steps (first 20%)", f"{steps_ep[:early].mean():.0f}")
        with col_b:
            st.metric("Mean steps (last 20%)", f"{steps_ep[-early:].mean():.0f}")
        with col_c:
            pct = 100 * float((steps_ep < 200).mean())
            st.metric("Episodes solved (<200 steps)", f"{pct:.0f}%")

        tab1, tab2, tab3 = st.tabs(
            [
                "Learning Curve",
                "Trajectory Snapshots",
                "Cost-to-go",
            ]
        )
        with tab1:
            st.plotly_chart(
                _learning_curve_fig(steps_ep, smooth), use_container_width=True
            )
        with tab2:
            st.plotly_chart(
                _trajectory_fig(trajs, n_episodes), use_container_width=True
            )
            st.caption(
                "Position over time for five training snapshots.  "
                "Early episodes hit the 200-step truncation; later ones reach the goal (pos=0.5)."
            )
        with tab3:
            inner_tab1, inner_tab2 = st.tabs(["3-D Surface", "Contour Map"])
            with inner_tab1:
                st.plotly_chart(_cost_surface(w, tc), use_container_width=True)
                st.caption(
                    "The cost-to-go surface — higher means farther from the goal.  "
                    "Compare with Figure 10.1 in Sutton & Barto."
                )
            with inner_tab2:
                st.plotly_chart(_value_contour(w, tc), use_container_width=True)
                st.caption(
                    "Same data as a 2-D contour.  Stars = typical start position; "
                    "diamond = goal."
                )

        st.header("Key Takeaways")
        final_mean = steps_ep[-early:].mean()
        if final_mean < 130:
            st.success(
                f"Agent reliably solves Mountain Car ({final_mean:.0f} mean steps in last 20%).  "
                "The cost-to-go surface should now show the characteristic saddle shape from S&B Fig 10.1."
            )
        elif final_mean < 180:
            st.warning(
                f"Agent partially learned ({final_mean:.0f} mean steps in last 20%).  "
                "More episodes or a slightly lower α may help convergence."
            )
        else:
            st.error(
                f"Agent struggling ({final_mean:.0f} mean steps in last 20%).  "
                "Try α=0.5, 8 tilings, ε decay from 0.1→0, 400 episodes."
            )
