import numpy as np
import plotly.graph_objects as go
import streamlit as st

from utils.random_walk_1000 import RandomWalk1000, compute_true_values
from utils.feature_bases import BASIS_REGISTRY, make_feature_matrix

# ── Training ───────────────────────────────────────────────────────────────────


@st.cache_data
def _true_values() -> np.ndarray:
    return compute_true_values()


@st.cache_data
def run_gradient_mc(
    basis_name: str,
    basis_param: int,
    alpha: float,
    n_episodes: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Gradient Monte Carlo prediction with linear function approximation.
    Returns (rms_errors [n_episodes], V_learned [1000]).
    """
    V_true = _true_values()
    basis = BASIS_REGISTRY[basis_name]
    basis_fn = basis["fn"]
    n_feat = basis["n_feat"](basis_param)
    w = np.zeros(n_feat)
    rng = np.random.default_rng(seed)
    env = RandomWalk1000(seed=seed)
    rms = np.zeros(n_episodes)

    for ep in range(n_episodes):
        # Generate full episode
        traj: list[tuple[int, float]] = []
        s = env.reset()
        while True:
            ns, r, done = env.step()
            traj.append((s, r))
            s = ns
            if done:
                break

        # Compute returns and gradient MC update (first-visit)
        G = 0.0
        visited: set[int] = set()
        for t in range(len(traj) - 1, -1, -1):
            sv, rv = traj[t]
            G = rv + G
            if sv not in visited:
                visited.add(sv)
                phi = basis_fn(sv, RandomWalk1000.N_STATES, basis_param)
                w += alpha * (G - float(w @ phi)) * phi

        # RMS error snapshot
        V_hat = np.array(
            [
                float(w @ basis_fn(s + 1, RandomWalk1000.N_STATES, basis_param))
                for s in range(RandomWalk1000.N_STATES)
            ]
        )
        rms[ep] = float(np.sqrt(np.mean((V_hat - V_true) ** 2)))

    V_hat = np.array(
        [
            float(w @ basis_fn(s + 1, RandomWalk1000.N_STATES, basis_param))
            for s in range(RandomWalk1000.N_STATES)
        ]
    )
    return rms, V_hat


@st.cache_data
def run_semi_gradient_td(
    basis_name: str,
    basis_param: int,
    alpha: float,
    n_episodes: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Semi-gradient TD(0) prediction with linear function approximation.
    Returns (rms_errors [n_episodes], V_learned [1000]).
    """
    V_true = _true_values()
    basis = BASIS_REGISTRY[basis_name]
    basis_fn = basis["fn"]
    n_feat = basis["n_feat"](basis_param)
    w = np.zeros(n_feat)
    rng = np.random.default_rng(seed)
    env = RandomWalk1000(seed=seed)
    rms = np.zeros(n_episodes)
    gamma = 1.0

    for ep in range(n_episodes):
        s = env.reset()
        while True:
            phi_s = basis_fn(s, RandomWalk1000.N_STATES, basis_param)
            ns, r, done = env.step()
            if done:
                target = r  # terminal: V(s') = 0
            else:
                phi_ns = basis_fn(ns, RandomWalk1000.N_STATES, basis_param)
                target = r + gamma * float(w @ phi_ns)
            w += alpha * (target - float(w @ phi_s)) * phi_s
            s = ns
            if done:
                break

        V_hat = np.array(
            [
                float(w @ basis_fn(s + 1, RandomWalk1000.N_STATES, basis_param))
                for s in range(RandomWalk1000.N_STATES)
            ]
        )
        rms[ep] = float(np.sqrt(np.mean((V_hat - V_true) ** 2)))

    V_hat = np.array(
        [
            float(w @ basis_fn(s + 1, RandomWalk1000.N_STATES, basis_param))
            for s in range(RandomWalk1000.N_STATES)
        ]
    )
    return rms, V_hat


# ── Visualisations ─────────────────────────────────────────────────────────────


def _rms_fig(rms_mc: np.ndarray, rms_td: np.ndarray) -> go.Figure:
    eps = list(range(1, len(rms_mc) + 1))
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=eps,
            y=rms_mc,
            mode="lines",
            name="Gradient MC",
            line=dict(color="#636EFA", width=2),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=eps,
            y=rms_td,
            mode="lines",
            name="Semi-gradient TD(0)",
            line=dict(color="#EF553B", width=2),
        )
    )
    fig.update_layout(
        title="RMS Error vs Episodes",
        xaxis_title="Episodes",
        yaxis_title="RMS Error",
        template="plotly_white",
        height=380,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


def _value_fig(V_true, V_mc, V_td) -> go.Figure:
    states = list(range(1, RandomWalk1000.N_STATES + 1))
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=states,
            y=V_true,
            mode="lines",
            name="True V(s)",
            line=dict(color="black", width=2, dash="dot"),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=states,
            y=V_mc,
            mode="lines",
            name="Gradient MC",
            line=dict(color="#636EFA", width=2),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=states,
            y=V_td,
            mode="lines",
            name="Semi-gradient TD(0)",
            line=dict(color="#EF553B", width=2),
        )
    )
    fig.update_layout(
        title="Learned Value Functions vs True",
        xaxis_title="State",
        yaxis_title="V(s)",
        template="plotly_white",
        height=380,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


# ── Page ───────────────────────────────────────────────────────────────────────


def show():
    st.title("Gradient MC vs Semi-gradient TD(0)")
    st.markdown(
        '<div style="background:#7B2D8B;height:3px;'
        'border-radius:2px;margin-bottom:1rem;"></div>',
        unsafe_allow_html=True,
    )
    st.markdown("**Section 7 — Function Approximation: Prediction (Chapter 9)**")

    # ── Concept ───────────────────────────────────────────────────────────────
    st.header("From Tables to Functions")
    st.markdown(r"""
Every algorithm in Sections 1–6 stored one value per state.  In small environments
like a 7×10 maze or a 5-state random walk, that is fine.  But consider a robot that
perceives a continuous-valued sensor reading — there is no finite table.

**Function approximation** replaces the table with a parameterised function:

$$\hat{v}(s, \mathbf{w}) \approx v_\pi(s)$$

where $\mathbf{w}$ is a weight vector.  We choose $\mathbf{w}$ to minimise the
**Mean Squared Value Error**:

$$\overline{VE}(\mathbf{w}) = \sum_s \mu(s)\bigl[v_\pi(s) - \hat{v}(s,\mathbf{w})\bigr]^2$$

with $\mu(s)$ weighting states by how often the agent visits them.
""")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Gradient Monte Carlo**")
        st.latex(
            r"\mathbf{w} \leftarrow \mathbf{w} + \alpha\bigl[G_t - \hat{v}(S_t,\mathbf{w})\bigr]"
            r"\nabla\hat{v}(S_t,\mathbf{w})"
        )
        st.caption(
            "Targets the true return $G_t$. Unbiased but must wait until episode ends. "
            "Converges to the true MSVE minimum for linear approximators."
        )
    with col2:
        st.markdown("**Semi-gradient TD(0)**")
        st.latex(
            r"\mathbf{w} \leftarrow \mathbf{w} + \alpha\bigl[R_{t+1} + \gamma\hat{v}(S_{t+1},\mathbf{w})"
            r" - \hat{v}(S_t,\mathbf{w})\bigr]\nabla\hat{v}(S_t,\mathbf{w})"
        )
        st.caption(
            "Bootstraps from $\hat{v}(S_{t+1}, \mathbf{w})$, which is itself an estimate — hence "
            "*semi*-gradient: only the gradient of $\hat{v}(S_t)$ is used, not of the target. "
            "Biased but online and often faster to converge early."
        )

    st.markdown(r"""
For a **linear** approximator $\hat{v}(s,\mathbf{w}) = \mathbf{w}^\top \boldsymbol{\phi}(s)$,
the gradient is simply $\boldsymbol{\phi}(s)$ — the feature vector.  This page uses the
**1000-state random walk** from S&B Example 9.1: 1000 states in a chain, random steps
of 1–100, left exits give reward –1, right exits give +1.
""")

    st.info(
        "**Why semi-gradient?**  In standard gradient descent the target is fixed — "
        "we compute the gradient of the loss and step toward the minimum.  In TD, the target "
        "$R + \\gamma\\hat{v}(S')$ depends on the same weights $\\mathbf{w}$, so it moves "
        "as we update.  Taking the gradient of the full target would give a different algorithm "
        "(residual gradient).  Treating the target as constant gives the simpler, more practical "
        "semi-gradient update — which still converges for linear approximators."
    )

    with st.expander("Deep Dive — Why Linear FA Converges to a TD Fixed Point"):
        st.markdown(r"""
For linear $\hat{v}(s,\mathbf{w}) = \mathbf{w}^\top\boldsymbol{\phi}(s)$, semi-gradient TD(0)
converges to the **TD fixed point** $\mathbf{w}_\text{TD}$ satisfying:

$$\mathbf{w}_\text{TD} = \mathbf{w}_\text{TD} + \alpha\bigl[\mathbf{b} - \mathbf{A}\mathbf{w}_\text{TD}\bigr] = 0
\implies \mathbf{A}\mathbf{w}_\text{TD} = \mathbf{b}$$

where $\mathbf{b} = \mathbb{E}[R_{t+1}\boldsymbol{\phi}(S_t)]$ and
$\mathbf{A} = \mathbb{E}[\boldsymbol{\phi}(S_t)(\boldsymbol{\phi}(S_t) - \gamma\boldsymbol{\phi}(S_{t+1}))^\top]$.

This fixed point is not the true MSVE minimiser, but it is guaranteed to be within a bounded
factor of it.  Gradient MC does converge to the true MSVE minimum — but may need more samples
to get there.
""")

    st.divider()

    # ── Simulation ─────────────────────────────────────────────────────────────
    st.header("Interactive Simulation")
    st.markdown(
        "Run both algorithms on the 1000-state random walk with your chosen feature basis. "
        "Compare how quickly each converges and how well the final value function matches the truth."
    )

    col1, col2 = st.columns(2)
    with col1:
        basis_name = st.selectbox("Feature basis", list(BASIS_REGISTRY.keys()), index=2)
        info = BASIS_REGISTRY[basis_name]
        plabel, pdef, pmin, pmax, pstep = info["param"]
        basis_param = st.slider(plabel, pmin, pmax, pdef, pstep)
        alpha_mc = st.slider(
            "α — Gradient MC", 0.0001, 0.05, 0.002, 0.0001, format="%.4f"
        )
    with col2:
        alpha_td = st.slider(
            "α — Semi-gradient TD(0)", 0.0001, 0.05, 0.002, 0.0001, format="%.4f"
        )
        n_episodes = st.slider("Episodes", 100, 2000, 500, 100)
        seed = st.number_input("Random seed", value=42, step=1)

    n_feat = info["n_feat"](basis_param)
    st.caption(f"Feature vector dimension: **{n_feat}**")

    if st.button("Run Comparison", type="primary"):
        with st.spinner("Running gradient MC and semi-gradient TD(0)…"):
            rms_mc, V_mc = run_gradient_mc(
                basis_name, basis_param, alpha_mc, n_episodes, int(seed)
            )
            rms_td, V_td = run_semi_gradient_td(
                basis_name, basis_param, alpha_td, n_episodes, int(seed)
            )
            V_true = _true_values()

        st.header("Results")
        st.plotly_chart(_rms_fig(rms_mc, rms_td), use_container_width=True)
        st.plotly_chart(_value_fig(V_true, V_mc, V_td), use_container_width=True)

        col_a, col_b = st.columns(2)
        with col_a:
            st.metric("Gradient MC — final RMS", f"{rms_mc[-1]:.4f}")
        with col_b:
            st.metric("Semi-gradient TD(0) — final RMS", f"{rms_td[-1]:.4f}")

        st.header("Key Takeaways")
        winner = "Gradient MC" if rms_mc[-1] < rms_td[-1] else "Semi-gradient TD(0)"
        st.success(
            f"**{winner}** achieved lower final RMS error with these settings. "
            "TD often converges faster early (online updates every step) while MC may "
            "reach a lower final error (unbiased target). The crossover depends on α and the basis."
        )
        st.info(
            "Try **State Aggregation** with a small number of groups to see the staircase "
            "approximation, then switch to **Fourier Cosine** (order ≥ 5) to see a smooth fit. "
            "Increasing the feature dimension generally reduces bias but needs a smaller α to stay stable."
        )
