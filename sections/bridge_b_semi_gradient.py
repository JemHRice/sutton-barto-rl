import numpy as np
import plotly.graph_objects as go
import streamlit as st

# ── Environment: 19-state Random Walk ─────────────────────────────────────

N_STATES = 19


def _true_values() -> np.ndarray:
    return np.linspace(-1, 1, N_STATES + 2)[1:-1]


def _random_walk_episode(rng: np.random.Generator) -> list[tuple[int, float]]:
    """Generate a full episode. Returns list of (state_idx, reward) pairs."""
    state = N_STATES // 2  # 0-indexed, centre state
    trajectory = []
    while True:
        move = rng.choice([-1, 1])
        next_state = state + move
        if next_state < 0:
            trajectory.append((state, -1.0))
            break
        elif next_state >= N_STATES:
            trajectory.append((state, 1.0))
            break
        else:
            trajectory.append((state, 0.0))
            state = next_state
    return trajectory


# ── Feature representations ────────────────────────────────────────────────


def _state_aggregation_features(state: int, n_groups: int = 5) -> np.ndarray:
    """Coarse coding: aggregate 19 states into n_groups groups."""
    group_size = N_STATES / n_groups
    group = int(state / group_size)
    group = min(group, n_groups - 1)
    phi = np.zeros(n_groups)
    phi[group] = 1.0
    return phi


def _polynomial_features(state: int, degree: int = 5) -> np.ndarray:
    """Polynomial basis: [1, x, x^2, ..., x^degree], x in [-1, 1]."""
    x = (state / (N_STATES - 1)) * 2 - 1  # normalise to [-1, 1]
    return np.array([x**k for k in range(degree + 1)])


def _get_features(state: int, feature_type: str, param: int) -> np.ndarray:
    if feature_type == "State Aggregation":
        return _state_aggregation_features(state, n_groups=param)
    else:
        return _polynomial_features(state, degree=param)


def _n_features(feature_type: str, param: int) -> int:
    if feature_type == "State Aggregation":
        return param
    else:
        return param + 1


# ── Gradient Monte Carlo ───────────────────────────────────────────────────


@st.cache_data
def run_gradient_mc(
    alpha: float,
    n_episodes: int,
    feature_type: str,
    feat_param: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Gradient Monte Carlo with linear function approximation.
    Returns (rms_errors [n_episodes], learned_V [N_STATES]).
    """
    rng = np.random.default_rng(42)
    true_V = _true_values()
    n_feat = _n_features(feature_type, feat_param)
    theta = np.zeros(n_feat)
    rms_errors = np.zeros(n_episodes)

    for ep in range(n_episodes):
        trajectory = _random_walk_episode(rng)
        # Compute returns (no discounting for random walk)
        G = trajectory[-1][1]  # terminal reward

        for state, _ in trajectory:
            phi = _get_features(state, feature_type, feat_param)
            v_hat = float(np.dot(theta, phi))
            theta += alpha * (G - v_hat) * phi

        learned_V = np.array(
            [
                float(np.dot(theta, _get_features(s, feature_type, feat_param)))
                for s in range(N_STATES)
            ]
        )
        rms_errors[ep] = float(np.sqrt(np.mean((learned_V - true_V) ** 2)))

    learned_V = np.array(
        [
            float(np.dot(theta, _get_features(s, feature_type, feat_param)))
            for s in range(N_STATES)
        ]
    )
    return rms_errors, learned_V


# ── Semi-gradient TD(0) ────────────────────────────────────────────────────


@st.cache_data
def run_semi_gradient_td(
    alpha: float,
    gamma: float,
    n_episodes: int,
    feature_type: str,
    feat_param: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Semi-gradient TD(0) with linear function approximation.
    Returns (rms_errors [n_episodes], learned_V [N_STATES]).
    """
    rng = np.random.default_rng(42)
    true_V = _true_values()
    n_feat = _n_features(feature_type, feat_param)
    theta = np.zeros(n_feat)
    rms_errors = np.zeros(n_episodes)

    for ep in range(n_episodes):
        state = N_STATES // 2
        while True:
            phi = _get_features(state, feature_type, feat_param)
            v_hat = float(np.dot(theta, phi))

            move = rng.choice([-1, 1])
            next_state = state + move

            if next_state < 0:
                reward = -1.0
                # Terminal: no bootstrap term
                theta += alpha * (reward - v_hat) * phi
                break
            elif next_state >= N_STATES:
                reward = 1.0
                theta += alpha * (reward - v_hat) * phi
                break
            else:
                reward = 0.0
                phi_next = _get_features(next_state, feature_type, feat_param)
                v_next = float(np.dot(theta, phi_next))
                # Semi-gradient update: treat bootstrap target as constant
                theta += alpha * (reward + gamma * v_next - v_hat) * phi
                state = next_state

        learned_V = np.array(
            [
                float(np.dot(theta, _get_features(s, feature_type, feat_param)))
                for s in range(N_STATES)
            ]
        )
        rms_errors[ep] = float(np.sqrt(np.mean((learned_V - true_V) ** 2)))

    learned_V = np.array(
        [
            float(np.dot(theta, _get_features(s, feature_type, feat_param)))
            for s in range(N_STATES)
        ]
    )
    return rms_errors, learned_V


# ── Page ───────────────────────────────────────────────────────────────────


def show():
    st.title("Semi-Gradient Methods")
    st.markdown(
        '<div style="background:#7B2D8B;height:3px;border-radius:2px;margin-bottom:1rem;"></div>',
        unsafe_allow_html=True,
    )
    st.markdown("**Neural Networks Bridge — Transition to Deep RL**")

    # ── Concept ───────────────────────────────────────────────────────────
    st.header("Learning the Weights")
    st.markdown("""
Bridge Page A introduced the MSVE objective and the gradient descent update. There is a
subtlety when we use **bootstrapping** (TD methods) with function approximation.

In pure gradient descent, the target is fixed — we compute the gradient of the error between
our estimate and a fixed target, and step in that direction. But in TD(0), the target
$R_{t+1} + \\gamma \\hat{v}(S_{t+1}, \\boldsymbol{\\theta})$ *depends on the current weights*.
When we change θ, the target moves too.

This makes the true gradient difficult to compute. Instead, we use a **semi-gradient** update:
we treat the bootstrap target as if it were a fixed constant, and only differentiate through
the estimate on the left side.
""")

    st.latex(
        r"\boldsymbol{\theta}_{t+1} = \boldsymbol{\theta}_t + \alpha"
        r"\underbrace{\bigl[R_{t+1} + \gamma \hat{v}(S_{t+1}, \boldsymbol{\theta}_t)"
        r"- \hat{v}(S_t, \boldsymbol{\theta}_t)\bigr]}_{\text{TD error}}"
        r"\nabla_{\boldsymbol{\theta}} \hat{v}(S_t, \boldsymbol{\theta}_t)"
    )

    st.markdown(
        "Note: the gradient $\\nabla_{\\boldsymbol{\\theta}} \\hat{v}$ is taken only with "
        "respect to $\\hat{v}(S_t, \\boldsymbol{\\theta}_t)$ — not through the bootstrap target. "
        "This is the *semi* in semi-gradient."
    )

    st.warning(
        "**The deadly triad:** semi-gradient methods can diverge when all three of these are "
        "combined: (1) function approximation, (2) bootstrapping, (3) off-policy learning. "
        "On-policy semi-gradient TD is stable. The divergence problem is covered fully in "
        "Section 9 — Off-Policy with Approximation."
    )

    with st.expander("Deep Dive — Why 'Semi' Gradient?"):
        st.markdown(r"""
True gradient descent on MSVE would require differentiating through the target, giving:

$$\nabla_{\boldsymbol{\theta}} \overline{VE} = \mathbb{E}\bigl[
    (v_\pi(S_t) - \hat{v}(S_t, \boldsymbol{\theta})) \nabla \hat{v}(S_t, \boldsymbol{\theta})
\bigr]$$

With Monte Carlo, the target $G_t$ is a sample of $v_\pi(S_t)$ — it does not depend on θ.
So gradient MC *is* a true stochastic gradient descent method and is guaranteed to converge
to a local minimum of MSVE.

With TD, the target $R + \gamma \hat{v}(S', \boldsymbol{\theta})$ depends on θ. Treating it as
fixed means we are not following the true gradient — hence "semi." Semi-gradient TD converges
to a different point than the MSVE minimum (the "TD fixed point"), but this point is usually
close and semi-gradient TD often learns much faster than gradient MC.
""")

    st.divider()

    # ── Feature types ──────────────────────────────────────────────────────
    st.header("Feature Representations")
    st.markdown("""
For a linear approximator, we need a **feature vector** $\\mathbf{x}(s)$ that represents
each state as a vector of numbers. The choice of features matters a lot.

**State aggregation:** group nearby states together. All states in the same group share the
same value estimate — coarse but simple.

**Polynomial basis:** represent state $s$ (normalised to $[-1, 1]$) as powers $[1, s, s^2, \\ldots, s^d]$.
Higher degree → richer representation → better fit, but more parameters.
""")

    st.divider()

    # ── Controls ──────────────────────────────────────────────────────────
    st.header("Interactive Simulation")

    col1, col2 = st.columns(2)
    with col1:
        feature_type = st.radio(
            "Feature representation",
            ["State Aggregation", "Polynomial Basis"],
            help="Which type of features to use for the linear approximator.",
        )
        if feature_type == "State Aggregation":
            feat_param = st.slider(
                "Number of groups",
                2,
                19,
                5,
                help="How many groups to aggregate the 19 states into.",
            )
        else:
            feat_param = st.slider(
                "Polynomial degree", 1, 9, 5, help="Degree of the polynomial basis."
            )
        n_episodes = st.slider("Episodes", 50, 1000, 300, 50)
    with col2:
        alpha_mc = st.slider("α — Gradient MC", 0.001, 0.1, 0.01, 0.001, format="%.3f")
        alpha_td = st.slider(
            "α — Semi-gradient TD", 0.001, 0.2, 0.05, 0.001, format="%.3f"
        )
        gamma = st.slider("γ (discount)", 0.9, 1.0, 1.0, 0.01)

    if st.button("Run Simulation", type="primary"):
        with st.spinner("Training both methods..."):
            mc_errors, mc_V = run_gradient_mc(
                alpha_mc, n_episodes, feature_type, feat_param
            )
            td_errors, td_V = run_semi_gradient_td(
                alpha_td, gamma, n_episodes, feature_type, feat_param
            )

        true_V = _true_values()

        st.header("Results")

        # ── RMS error curves ───────────────────────────────────────────
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=list(range(1, n_episodes + 1)),
                y=mc_errors,
                mode="lines",
                name="Gradient MC",
                line=dict(color="#636EFA", width=2),
            )
        )
        fig.add_trace(
            go.Scatter(
                x=list(range(1, n_episodes + 1)),
                y=td_errors,
                mode="lines",
                name="Semi-gradient TD(0)",
                line=dict(color="#EF553B", width=2),
            )
        )
        fig.update_layout(
            title="RMS Error vs Episodes",
            xaxis_title="Episode",
            yaxis_title="RMS Error",
            template="plotly_white",
            height=380,
            legend=dict(
                orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1
            ),
        )
        st.plotly_chart(fig, use_container_width=True)

        # ── Learned value functions ────────────────────────────────────
        states = list(range(1, N_STATES + 1))
        fig2 = go.Figure()
        fig2.add_trace(
            go.Scatter(
                x=states,
                y=true_V,
                mode="lines",
                name="True values",
                line=dict(color="black", dash="dash", width=2),
            )
        )
        fig2.add_trace(
            go.Scatter(
                x=states,
                y=mc_V,
                mode="lines+markers",
                name="Gradient MC",
                line=dict(color="#636EFA", width=1.5),
                marker=dict(size=4),
            )
        )
        fig2.add_trace(
            go.Scatter(
                x=states,
                y=td_V,
                mode="lines+markers",
                name="Semi-gradient TD(0)",
                line=dict(color="#EF553B", width=1.5),
                marker=dict(size=4),
            )
        )
        fig2.update_layout(
            title=f"Learned Value Functions ({feature_type})",
            xaxis_title="State",
            yaxis_title="Estimated Value",
            template="plotly_white",
            height=360,
            legend=dict(
                orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1
            ),
        )
        st.plotly_chart(fig2, use_container_width=True)

        # ── Summary ────────────────────────────────────────────────────
        st.header("Key Takeaways")
        col_a, col_b, col_c = st.columns(3)
        with col_a:
            st.metric("Gradient MC final RMS", f"{mc_errors[-1]:.4f}")
        with col_b:
            st.metric("Semi-grad TD final RMS", f"{td_errors[-1]:.4f}")
        with col_c:
            feat_desc = (
                f"{feat_param} groups"
                if feature_type == "State Aggregation"
                else f"degree {feat_param}"
            )
            st.metric("Feature capacity", feat_desc)

        st.info(
            "Semi-gradient TD often converges faster than gradient MC (lower RMS earlier) "
            "because it updates at every step rather than waiting until episode end. "
            "However, it converges to the TD fixed point, not the true MSVE minimum — "
            "which is why the final RMS may be slightly higher than gradient MC."
        )
        st.success(
            "**You are now ready for the Deep RL sections.** The update rule above is the "
            "foundation of every deep RL algorithm — the only change is that θ will be the "
            "weights of a neural network, and the gradient will be computed by PyTorch."
        )
