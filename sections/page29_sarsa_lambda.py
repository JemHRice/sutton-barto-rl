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
def run_sarsa_lambda(
    n_tilings: int,
    n_tiles: int,
    alpha: float,
    lam: float,
    epsilon: float,
    n_episodes: int,
    seed: int,
    use_replace: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """
    SARSA(λ) with accumulating or replacing traces on Mountain Car.
    Returns (steps_per_episode [n_episodes], weights).
    """
    tc = TileCoder(n_tilings=n_tilings, n_tiles=n_tiles)
    n_feat = tc.n_features
    w = np.zeros((N_ACTIONS, n_feat))
    rng = np.random.default_rng(seed)
    env = gym.make("MountainCar-v0")
    gamma = 1.0

    steps_ep = np.zeros(n_episodes)

    def q(state: np.ndarray, action: int) -> float:
        return tc.value(w[action], state)

    for ep in range(n_episodes):
        obs, _ = env.reset(seed=int(seed + ep))
        state = np.array(obs, dtype=float)
        vals = np.array([q(state, a) for a in range(N_ACTIONS)])
        ties = np.flatnonzero(vals == vals.max())
        action = (
            int(rng.choice(ties))
            if rng.random() >= epsilon
            else int(rng.integers(N_ACTIONS))
        )

        # Per-action eligibility traces
        z = np.zeros((N_ACTIONS, n_feat))
        steps = 0

        while True:
            obs2, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            state2 = np.array(obs2, dtype=float)
            steps += 1

            tiles = tc.active_tiles(state)

            # Update traces (replace or accumulate)
            if use_replace:
                z[action][tiles] = 1.0
                # Decay other actions' traces (but leave chosen action alone after replace)
                for a in range(N_ACTIONS):
                    if a != action:
                        z[a] *= gamma * lam
                    else:
                        # Decay non-active tiles of chosen action too
                        mask = np.ones(n_feat, dtype=bool)
                        mask[tiles] = False
                        z[action][mask] *= gamma * lam
            else:
                for a in range(N_ACTIONS):
                    z[a] *= gamma * lam
                z[action][tiles] += 1.0

            if done:
                delta = float(reward) - q(state, action)
                w += (alpha / n_tilings) * delta * z
                break

            vals2 = np.array([q(state2, a) for a in range(N_ACTIONS)])
            ties2 = np.flatnonzero(vals2 == vals2.max())
            action2 = (
                int(rng.choice(ties2))
                if rng.random() >= epsilon
                else int(rng.integers(N_ACTIONS))
            )

            delta = float(reward) + gamma * q(state2, action2) - q(state, action)
            w += (alpha / n_tilings) * delta * z

            state = state2
            action = action2

        steps_ep[ep] = steps

    env.close()
    return steps_ep, w


@st.cache_data
def run_lambda_sweep(
    n_tilings: int,
    n_tiles: int,
    alpha: float,
    lambdas: tuple,
    epsilon: float,
    n_episodes: int,
    seed: int,
    use_replace: bool,
) -> dict:
    return {
        lam: run_sarsa_lambda(
            n_tilings, n_tiles, alpha, lam, epsilon, n_episodes, seed, use_replace
        )
        for lam in lambdas
    }


# ── Figures ────────────────────────────────────────────────────────────────────

_COLOURS = {
    0.0: "#636EFA",
    0.5: "#00CC96",
    0.9: "#EF553B",
    0.95: "#AB63FA",
    0.99: "#FFA15A",
}


def _learning_curve_fig(results: dict, smooth: int) -> go.Figure:
    fig = go.Figure()
    for lam, (steps_ep, _) in sorted(results.items()):
        colour = _COLOURS.get(lam, "#888888")
        eps = list(range(1, len(steps_ep) + 1))
        fig.add_trace(
            go.Scatter(
                x=eps,
                y=steps_ep.tolist(),
                mode="lines",
                name=f"λ = {lam} (raw)",
                line=dict(color=colour, width=1),
                opacity=0.2,
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
                    name=f"λ = {lam}",
                    line=dict(color=colour, width=2),
                )
            )
        else:
            fig.add_trace(
                go.Scatter(
                    x=eps,
                    y=steps_ep.tolist(),
                    mode="lines",
                    name=f"λ = {lam}",
                    line=dict(color=colour, width=2),
                )
            )
    fig.add_hline(
        y=200,
        line_dash="dash",
        line_color="black",
        annotation_text="Truncation (200 steps)",
    )
    fig.update_layout(
        title="SARSA(λ) — Steps per Episode",
        xaxis_title="Episode",
        yaxis_title="Steps",
        template="plotly_white",
        height=400,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


def _final_bar(results: dict, n_episodes: int) -> go.Figure:
    pct = int(max(1, n_episodes // 5))
    lams = sorted(results.keys())
    means = [float(results[lam][0][-pct:].mean()) for lam in lams]
    fig = go.Figure(
        go.Bar(
            x=[str(lam) for lam in lams],
            y=means,
            marker_color=[_COLOURS.get(lam, "#888888") for lam in lams],
        )
    )
    fig.update_layout(
        title=f"Mean Steps in Final 20% of Training by λ",
        xaxis_title="λ",
        yaxis_title="Mean steps",
        template="plotly_white",
        height=300,
    )
    return fig


def _cost_surface(w: np.ndarray, tc: TileCoder) -> go.Figure:
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
            colorscale="Plasma",
            colorbar=dict(title="Cost"),
        )
    )
    fig.update_layout(
        title="Cost-to-go Surface",
        scene=dict(
            xaxis_title="Position", yaxis_title="Velocity", zaxis_title="Cost-to-go"
        ),
        template="plotly_white",
        height=450,
    )
    return fig


# ── Page ───────────────────────────────────────────────────────────────────────


def show():
    st.title("SARSA(λ) — Action Eligibility Traces")
    st.markdown(
        '<div style="background:#7B2D8B;height:3px;'
        'border-radius:2px;margin-bottom:1rem;"></div>',
        unsafe_allow_html=True,
    )
    st.markdown("**Section 10 — Eligibility Traces (Chapter 12)**")

    if not _GYM_OK:
        st.error(
            "Gymnasium is not installed.  Install it with `pip install gymnasium` and restart the app."
        )
        return

    # ── Concept ───────────────────────────────────────────────────────────────
    st.header("Extending TD(λ) to Control")
    st.markdown(r"""
**SARSA(λ)** applies eligibility traces to action-value function approximation.
Instead of one trace per weight, we maintain one trace vector **per action**:

$$Q(s, a, \mathbf{w}) = \mathbf{w}_a^\top \boldsymbol{\phi}(s)$$

At each transition $(S_t, A_t, R_{t+1}, S_{t+1}, A_{t+1})$:

$$\delta_t = R_{t+1} + \gamma\,Q(S_{t+1}, A_{t+1}) - Q(S_t, A_t)$$

$$\mathbf{z}_{A_t} \leftarrow \gamma\lambda\,\mathbf{z}_{A_t} + \boldsymbol{\phi}(S_t)
\quad (\text{accumulating})$$

$$\mathbf{w}_a \leftarrow \mathbf{w}_a + \frac{\alpha}{n_\text{tilings}} \delta_t \mathbf{z}_a
\quad \forall a$$

For non-selected actions, their traces decay by $\gamma\lambda$ each step and their
weights are updated by the same $\delta_t$ multiplied by their (small) trace.
""")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Accumulating traces**")
        st.latex(
            r"\mathbf{z}_a \leftarrow \gamma\lambda\,\mathbf{z}_a + \boldsymbol{\phi}(S_t)"
        )
        st.caption(
            "Traces grow each time a tile is revisited within an episode.  "
            "Can lead to numerical issues over very long episodes.  "
            "Theoretically sound for on-policy prediction."
        )
    with col2:
        st.markdown("**Replacing traces** (preferred for tile coding)")
        st.latex(r"""z_i = \begin{cases}
1 & \text{tile } i \text{ active for action } a \\
\gamma\lambda\,z_i & \text{otherwise}
\end{cases}""")
        st.caption(
            "Active tiles are clamped to 1 instead of accumulating.  "
            "Prevents traces from exploding; preferred in S&B Chapter 12.7 "
            "and the original tile-coding experiments."
        )

    st.info(
        "**Why eligibility traces help on Mountain Car**: "
        "the car receives reward $-1$ at every step and $0$ at termination.  "
        "With $\\lambda = 0$, credit propagates back only one step at a time — very slow.  "
        "With $\\lambda > 0$, a single successful episode immediately credits all the "
        "states that were recently visited, dramatically speeding convergence."
    )

    with st.expander("Connection to n-step SARSA"):
        st.markdown(r"""
SARSA(λ) is the online, incremental equivalent of a weighted sum over all n-step SARSA
targets.  The λ-return for action values is:

$$G_t^{\lambda,Q} = (1-\lambda)\sum_{n=1}^\infty \lambda^{n-1} G_t^{(n)}$$

where each $G_t^{(n)}$ bootstraps from the $n$-step-ahead action-value estimate.
- $\lambda = 0$: one-step SARSA
- $\lambda = 1$: Monte Carlo (full episode returns, no bootstrapping)

The optimal $\lambda$ depends on how many steps ahead are needed to accurately estimate
returns.  For Mountain Car with sparse rewards, $\lambda = 0.9$ often converges fastest.
""")

    st.divider()

    # ── Simulation ─────────────────────────────────────────────────────────────
    st.header("Interactive Simulation — Mountain Car")

    col1, col2 = st.columns(2)
    with col1:
        n_tilings = st.select_slider("Number of tilings", [2, 4, 8, 16], value=8)
        n_tiles = st.select_slider("Tiles per dimension", [4, 8, 16], value=8)
        alpha = st.slider("α (step size)", 0.01, 1.0, 0.5, 0.01)
        epsilon = st.slider("ε (exploration)", 0.0, 0.3, 0.0, 0.01)
        use_replace = st.toggle("Use replacing traces", value=True)
    with col2:
        lambdas_sel = st.multiselect(
            "λ values to compare",
            [0.0, 0.5, 0.9, 0.95, 0.99],
            default=[0.0, 0.9],
        )
        n_episodes = st.slider("Episodes", 50, 500, 200, 25)
        seed = st.number_input("Random seed", value=42, step=1)
        smooth = st.slider("Rolling mean window", 1, 30, 10)
        n_feat_per = n_tilings * (n_tiles + 1) ** 2
        st.caption(
            f"Features per action: **{n_feat_per}** · Total: **{N_ACTIONS * n_feat_per}**"
        )

    if st.button("Train Agents", type="primary") and lambdas_sel:
        with st.spinner("Running SARSA(λ) for selected λ values…"):
            results = run_lambda_sweep(
                n_tilings,
                n_tiles,
                alpha,
                tuple(sorted(lambdas_sel)),
                epsilon,
                n_episodes,
                int(seed),
                use_replace,
            )

        st.header("Results")
        tab1, tab2, tab3 = st.tabs(
            ["Learning Curves", "Final Steps by λ", "Cost-to-go"]
        )

        with tab1:
            st.plotly_chart(
                _learning_curve_fig(results, smooth), use_container_width=True
            )

        with tab2:
            st.plotly_chart(_final_bar(results, n_episodes), use_container_width=True)
            st.caption("Mean steps per episode in the final 20% of training.")

        with tab3:
            best_lam = min(
                results, key=lambda l: results[l][0][-max(1, n_episodes // 5) :].mean()
            )
            tc = TileCoder(n_tilings=n_tilings, n_tiles=n_tiles)
            _, best_w = results[best_lam]
            st.plotly_chart(_cost_surface(best_w, tc), use_container_width=True)
            st.caption(f"Cost-to-go surface for best λ = {best_lam}.")

        st.header("Key Takeaways")
        best_lam = min(
            results, key=lambda l: results[l][0][-max(1, n_episodes // 5) :].mean()
        )
        final_mean = float(results[best_lam][0][-max(1, n_episodes // 5) :].mean())
        st.success(
            f"Best performance: **λ = {best_lam}** with {final_mean:.0f} mean steps "
            f"in the final 20% of training.  "
            "Higher λ tends to win on Mountain Car because it bridges the gap between "
            "the distant reward signal and the early exploration states."
        )
        st.info(
            "Compare replacing vs accumulating traces using the toggle above.  "
            "Replacing traces are usually more stable because they prevent large trace values "
            "when the same tile is activated many times in a single episode — common in "
            "Mountain Car when the car oscillates in a valley before finding the goal."
        )
