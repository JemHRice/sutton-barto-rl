import numpy as np
import plotly.graph_objects as go
import streamlit as st

from utils.random_walk_1000 import RandomWalk1000, compute_true_values
from utils.feature_bases import BASIS_REGISTRY

# ── Helpers ────────────────────────────────────────────────────────────────────


@st.cache_data
def _true_values() -> np.ndarray:
    return compute_true_values()


# ── Training ───────────────────────────────────────────────────────────────────


@st.cache_data
def run_td_lambda(
    basis_name: str,
    basis_param: int,
    alpha: float,
    lam: float,
    n_episodes: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    On-policy semi-gradient TD(λ) on the 1000-state random walk.
    Returns (rms_errors [n_episodes], V_hat [1000]).
    """
    info = BASIS_REGISTRY[basis_name]
    phi_fn = info["fn"]
    n_feat = info["n_feat"](basis_param)
    env = RandomWalk1000(seed=seed)
    w = np.zeros(n_feat)
    V_true = _true_values()
    gamma = 1.0
    rms = np.zeros(n_episodes)

    for ep in range(n_episodes):
        s = env.reset()
        z = np.zeros(n_feat)  # eligibility trace

        while True:
            ns, r, done = env.step()
            phi_s = phi_fn(s, RandomWalk1000.N_STATES, basis_param)
            phi_s2 = (
                np.zeros(n_feat)
                if done
                else phi_fn(ns, RandomWalk1000.N_STATES, basis_param)
            )

            v_s = float(w @ phi_s)
            v_s2 = 0.0 if done else float(w @ phi_s2)
            delta = r + gamma * v_s2 - v_s

            z = gamma * lam * z + phi_s  # accumulating traces
            w += alpha * delta * z

            if done:
                break
            s = ns

        V_hat = np.array(
            [
                float(w @ phi_fn(s + 1, RandomWalk1000.N_STATES, basis_param))
                for s in range(RandomWalk1000.N_STATES)
            ]
        )
        rms[ep] = float(np.sqrt(np.mean((V_hat - V_true) ** 2)))

    V_hat = np.array(
        [
            float(w @ phi_fn(s + 1, RandomWalk1000.N_STATES, basis_param))
            for s in range(RandomWalk1000.N_STATES)
        ]
    )
    return rms, V_hat


@st.cache_data
def run_lambda_sweep(
    basis_name: str,
    basis_param: int,
    alpha: float,
    lambdas: tuple,
    n_episodes: int,
    seed: int,
) -> dict:
    """Run TD(λ) for multiple λ values. Returns dict {λ: (rms, V_hat)}."""
    return {
        lam: run_td_lambda(basis_name, basis_param, alpha, lam, n_episodes, seed)
        for lam in lambdas
    }


# ── Figures ────────────────────────────────────────────────────────────────────

_LAMBDA_COLOURS = {
    0.0: "#636EFA",
    0.4: "#AB63FA",
    0.8: "#00CC96",
    0.9: "#FFA15A",
    0.95: "#EF553B",
    1.0: "#19D3F3",
}


def _rms_fig(results: dict, smooth: int) -> go.Figure:
    fig = go.Figure()
    for lam, (rms, _) in sorted(results.items()):
        colour = _LAMBDA_COLOURS.get(lam, "#888888")
        eps = list(range(1, len(rms) + 1))
        if smooth > 1 and len(rms) >= smooth:
            kernel = np.ones(smooth) / smooth
            sm = np.convolve(rms, kernel, mode="valid")
            x_sm = list(range(smooth, len(rms) + 1))
            fig.add_trace(
                go.Scatter(
                    x=x_sm,
                    y=sm.tolist(),
                    mode="lines",
                    name=f"λ = {lam}",
                    line=dict(color=colour, width=2),
                )
            )
        else:
            fig.add_trace(
                go.Scatter(
                    x=eps,
                    y=rms.tolist(),
                    mode="lines",
                    name=f"λ = {lam}",
                    line=dict(color=colour, width=2),
                )
            )
    fig.update_layout(
        title="RMS Error vs Episodes — TD(λ) for Various λ",
        xaxis_title="Episodes",
        yaxis_title="RMS Error",
        template="plotly_white",
        height=380,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


def _value_fig(V_true: np.ndarray, results: dict) -> go.Figure:
    states = list(range(1, RandomWalk1000.N_STATES + 1))
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=states,
            y=V_true.tolist(),
            mode="lines",
            name="True V(s)",
            line=dict(color="black", width=2, dash="dot"),
        )
    )
    for lam, (_, V_hat) in sorted(results.items()):
        colour = _LAMBDA_COLOURS.get(lam, "#888888")
        fig.add_trace(
            go.Scatter(
                x=states,
                y=V_hat.tolist(),
                mode="lines",
                name=f"λ = {lam}",
                line=dict(color=colour, width=2),
            )
        )
    fig.update_layout(
        title="Final Learned Value Functions",
        xaxis_title="State",
        yaxis_title="V(s)",
        template="plotly_white",
        height=380,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


def _final_rms_bar(results: dict) -> go.Figure:
    lams = sorted(results.keys())
    finals = [float(results[lam][0][-1]) for lam in lams]
    fig = go.Figure(
        go.Bar(
            x=[str(lam) for lam in lams],
            y=finals,
            marker_color=[_LAMBDA_COLOURS.get(lam, "#888888") for lam in lams],
        )
    )
    fig.update_layout(
        title="Final Episode RMS Error by λ",
        xaxis_title="λ",
        yaxis_title="RMS Error",
        template="plotly_white",
        height=300,
    )
    return fig


# ── Page ───────────────────────────────────────────────────────────────────────


def show():
    st.title("λ-Return and TD(λ) — Eligibility Traces")
    st.markdown(
        '<div style="background:#7B2D8B;height:3px;'
        'border-radius:2px;margin-bottom:1rem;"></div>',
        unsafe_allow_html=True,
    )
    st.markdown("**Section 10 — Eligibility Traces (Chapter 12)**")

    # ── Concept ───────────────────────────────────────────────────────────────
    st.header("Bridging n-step Returns")
    st.markdown(r"""
An **eligibility trace** is a short-term memory vector $\mathbf{z} \in \mathbb{R}^d$ that
accumulates the influence of recent feature vectors.  It provides an efficient, online
equivalent of the **λ-return** — the geometric mixture of all $n$-step returns:

$$G_t^\lambda = (1-\lambda) \sum_{n=1}^{\infty} \lambda^{n-1} G_t^{(n)}$$

**TD(λ)** computes this in a single forward pass using a trace updated at every step:

$$\mathbf{z}_t = \gamma\lambda\,\mathbf{z}_{t-1} + \boldsymbol{\phi}(S_t)$$

$$\mathbf{w}_{t+1} = \mathbf{w}_t + \alpha\,\delta_t\,\mathbf{z}_t$$

The parameter $\lambda \in [0, 1]$ controls the decay rate:
- $\lambda = 0$: TD(0) — only the most recent feature is credited (pure bootstrapping).
- $\lambda = 1$: MC-equivalent — all past features receive equal credit (pure Monte Carlo).
- $\lambda \in (0, 1)$: interpolates between them with exponential decay.
""")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Accumulating traces**")
        st.latex(
            r"\mathbf{z}_t = \gamma\lambda\,\mathbf{z}_{t-1} + \boldsymbol{\phi}(S_t)"
        )
        st.caption(
            "Each feature accumulates additively — if the same state is visited repeatedly, "
            "its trace grows.  Standard for prediction; can cause instability in control."
        )
    with col2:
        st.markdown("**Replacing traces** (for tile coding)")
        st.latex(
            r"z_i = \begin{cases} 1 & \phi_i(S_t) = 1 \\ \gamma\lambda\,z_i & \text{otherwise}\end{cases}"
        )
        st.caption(
            "Active components are reset to 1 rather than accumulated.  "
            "Prevents traces from growing arbitrarily large; standard for SARSA(λ) with tile coding."
        )

    st.info(
        "**Credit assignment**: the trace records which features were recently active.  "
        "When a reward arrives, all traces receive credit proportional to how recently "
        "their feature was active and how strongly it fired.  "
        "The effective 'horizon' is approximately $1/(1 - \\gamma\\lambda)$ steps."
    )

    with st.expander("Forward vs backward view"):
        st.markdown(r"""
**Forward view** (λ-return):  at time $t$, look forward and compute a weighted mixture of
$n$-step returns.  Intuitively appealing but requires the full future trajectory — only
usable offline.

**Backward view** (eligibility traces):  at each step, update weights using a trace that
summarises the past.  Mathematically equivalent to the forward view for on-policy linear TD,
but implementable *online* — one pass through experience.

The equivalence holds exactly only for on-policy, linear FA, and complete episodes (or
infinite episodes with discounting).  For off-policy and non-linear FA, the two views
diverge and separate corrections (like IS ratios in Retrace(λ)) are needed.
""")

    with st.expander("Computational complexity"):
        st.markdown(r"""
- Semi-gradient TD(0): $O(d)$ per step (update $\mathbf{w}$ at each transition).
- TD(λ) with accumulating traces: $O(d)$ per step (update $\mathbf{z}$ and then $\mathbf{w}$).
- Offline λ-return: $O(d \times T)$ per episode — must store full trajectory.

Eligibility traces give λ-return accuracy at TD(0) computational cost.  The only memory
overhead is a single extra vector $\mathbf{z}$ of the same dimension as $\mathbf{w}$.
""")

    st.divider()

    # ── Simulation ─────────────────────────────────────────────────────────────
    st.header("Interactive Simulation — 1000-State Random Walk")
    st.markdown(
        "On-policy TD(λ) with a selectable feature basis.  Compare convergence speed and "
        "final approximation quality across different values of λ."
    )

    col1, col2 = st.columns(2)
    with col1:
        basis_name = st.selectbox("Feature basis", list(BASIS_REGISTRY.keys()), index=2)
        info = BASIS_REGISTRY[basis_name]
        plabel, pdef, pmin, pmax, pstep = info["param"]
        basis_param = st.slider(plabel, pmin, pmax, pdef, pstep)
        alpha = st.slider("α (step size)", 0.0001, 0.05, 0.002, 0.0001, format="%.4f")
    with col2:
        lambdas_sel = st.multiselect(
            "λ values to compare",
            [0.0, 0.4, 0.8, 0.9, 0.95, 1.0],
            default=[0.0, 0.8, 0.95, 1.0],
        )
        n_episodes = st.slider("Episodes", 100, 3000, 1000, 100)
        seed = st.number_input("Random seed", value=42, step=1)
        smooth = st.slider("Rolling mean window", 1, 100, 20)

    if st.button("Run TD(λ)", type="primary") and lambdas_sel:
        with st.spinner("Running TD(λ) for selected λ values…"):
            results = run_lambda_sweep(
                basis_name,
                basis_param,
                alpha,
                tuple(sorted(lambdas_sel)),
                n_episodes,
                int(seed),
            )

        V_true = _true_values()

        st.header("Results")

        tab1, tab2, tab3 = st.tabs(
            ["Learning Curves", "Final Value Functions", "Final RMS by λ"]
        )
        with tab1:
            st.plotly_chart(_rms_fig(results, smooth), use_container_width=True)
        with tab2:
            st.plotly_chart(_value_fig(V_true, results), use_container_width=True)
        with tab3:
            st.plotly_chart(_final_rms_bar(results), use_container_width=True)

        best_lam = min(results, key=lambda l: results[l][0][-1])
        st.header("Key Takeaways")
        st.success(
            f"Best final RMS achieved with **λ = {best_lam}** "
            f"({results[best_lam][0][-1]:.4f}).  "
            "Intermediate λ values often outperform both extremes: "
            "λ=0 learns quickly but converges to a biased estimate, "
            "λ=1 is unbiased but has high variance (equivalent to Monte Carlo)."
        )
        st.info(
            "The optimal λ depends on the feature basis and step size.  "
            "With a good basis (e.g. Fourier Cosine), smaller λ often suffices "
            "because the basis already captures the function's shape — "
            "longer traces don't add much.  With a coarse basis (e.g. state aggregation), "
            "longer traces help bridge the gap."
        )
