import numpy as np
import plotly.graph_objects as go
import streamlit as st

from utils.feature_bases import BASIS_REGISTRY

# ── 19-state random walk ───────────────────────────────────────────────────────

N_STATES = 19  # non-terminal; terminals at 0 and 20
START = 10


def _true_values_pi(p_right: float, gamma: float = 1.0) -> np.ndarray:
    """
    Iterative policy evaluation for a policy that moves right with p_right.
    Returns V^π for non-terminal states 1–19 as a length-19 array.
    """
    V = np.zeros(N_STATES + 2)  # indices 0..20; 0 and 20 are absorbing
    V[N_STATES + 1] = 1.0  # reward of +1 on exit right
    p_left = 1.0 - p_right
    for _ in range(200_000):
        delta = 0.0
        for s in range(1, N_STATES + 1):
            r_right = 1.0 if s == N_STATES else 0.0
            r_left = 0.0
            v_new = p_right * (r_right + gamma * V[s + 1]) + p_left * (
                r_left + gamma * V[s - 1]
            )
            delta = max(delta, abs(V[s] - v_new))
            V[s] = v_new
        if delta < 1e-12:
            break
    return V[1 : N_STATES + 1].copy()


def _phi_agg(s: int, n_groups: int) -> np.ndarray:
    """State aggregation: one-hot over n_groups groups."""
    phi = np.zeros(n_groups)
    group = min(int((s - 1) * n_groups / N_STATES), n_groups - 1)
    phi[group] = 1.0
    return phi


# ── Training ───────────────────────────────────────────────────────────────────


@st.cache_data
def run_comparison(
    n_groups: int,
    alpha: float,
    p_right_target: float,
    n_episodes: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Three algorithms on the 19-state random walk:
      1. On-policy TD under π
      2. Off-policy TD (no IS) under b
      3. Off-policy IS-corrected TD under b

    Returns (rms_on, rms_off_nois, rms_off_is, V_true) — each of length n_episodes.
    """
    rng = np.random.default_rng(seed)
    p_b = 0.5  # behaviour policy always 50/50
    V_true = _true_values_pi(p_right_target)

    rho_right = p_right_target / p_b
    rho_left = (1.0 - p_right_target) / p_b

    def _episode_on(w: np.ndarray) -> None:
        s = START
        while True:
            right = rng.random() < p_right_target
            s2 = s + 1 if right else s - 1
            r = 1.0 if s2 == N_STATES + 1 else 0.0
            done = s2 == 0 or s2 == N_STATES + 1
            phi_s = _phi_agg(s, n_groups)
            v_s = float(w @ phi_s)
            v_s2 = 0.0 if done else float(w @ _phi_agg(s2, n_groups))
            delta = r + v_s2 - v_s
            w += alpha * delta * phi_s
            if done:
                break
            s = s2

    def _episode_off(w: np.ndarray, use_is: bool) -> None:
        s = START
        while True:
            right = rng.random() < p_b
            rho = (rho_right if right else rho_left) if use_is else 1.0
            s2 = s + 1 if right else s - 1
            r = 1.0 if s2 == N_STATES + 1 else 0.0
            done = s2 == 0 or s2 == N_STATES + 1
            phi_s = _phi_agg(s, n_groups)
            v_s = float(w @ phi_s)
            v_s2 = 0.0 if done else float(w @ _phi_agg(s2, n_groups))
            delta = r + v_s2 - v_s
            w += alpha * rho * delta * phi_s
            if done:
                break
            s = s2

    def _rms(w: np.ndarray) -> float:
        V_hat = np.array(
            [float(w @ _phi_agg(s, n_groups)) for s in range(1, N_STATES + 1)]
        )
        return float(np.sqrt(np.mean((V_hat - V_true) ** 2)))

    w_on = np.zeros(n_groups)
    w_noi = np.zeros(n_groups)
    w_is = np.zeros(n_groups)

    rms_on = np.zeros(n_episodes)
    rms_noi = np.zeros(n_episodes)
    rms_is = np.zeros(n_episodes)

    for ep in range(n_episodes):
        _episode_on(w_on)
        _episode_off(w_noi, use_is=False)
        _episode_off(w_is, use_is=True)
        rms_on[ep] = _rms(w_on)
        rms_noi[ep] = _rms(w_noi)
        rms_is[ep] = _rms(w_is)

    return rms_on, rms_noi, rms_is, V_true


@st.cache_data
def final_value_functions(
    n_groups: int,
    alpha: float,
    p_right_target: float,
    n_episodes: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return final learned V̂ for each algorithm plus V_true."""
    rng = np.random.default_rng(seed)
    p_b = 0.5
    V_true = _true_values_pi(p_right_target)

    rho_right = p_right_target / p_b
    rho_left = (1.0 - p_right_target) / p_b

    w_on = np.zeros(n_groups)
    w_noi = np.zeros(n_groups)
    w_is = np.zeros(n_groups)

    for _ in range(n_episodes):
        for w, use_is, use_pi in [
            (w_on, False, True),
            (w_noi, False, False),
            (w_is, True, False),
        ]:
            s = START
            while True:
                right = rng.random() < (p_right_target if use_pi else p_b)
                rho = (rho_right if right else rho_left) if use_is else 1.0
                s2 = s + 1 if right else s - 1
                r = 1.0 if s2 == N_STATES + 1 else 0.0
                done = s2 == 0 or s2 == N_STATES + 1
                phi_s = _phi_agg(s, n_groups)
                v_s = float(w @ phi_s)
                v_s2 = 0.0 if done else float(w @ _phi_agg(s2, n_groups))
                delta = r + v_s2 - v_s
                w += alpha * rho * delta * phi_s
                if done:
                    break
                s = s2

    states = list(range(1, N_STATES + 1))
    V_on = np.array([float(w_on @ _phi_agg(s, n_groups)) for s in states])
    V_noi = np.array([float(w_noi @ _phi_agg(s, n_groups)) for s in states])
    V_is = np.array([float(w_is @ _phi_agg(s, n_groups)) for s in states])
    return V_true, V_on, V_noi, V_is


# ── Figures ────────────────────────────────────────────────────────────────────


def _rms_fig(rms_on: np.ndarray, rms_noi: np.ndarray, rms_is: np.ndarray) -> go.Figure:
    eps = list(range(1, len(rms_on) + 1))
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=eps,
            y=rms_on.tolist(),
            mode="lines",
            name="On-policy TD (π)",
            line=dict(color="#636EFA", width=2),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=eps,
            y=rms_noi.tolist(),
            mode="lines",
            name="Off-policy TD (no IS)",
            line=dict(color="#EF553B", width=2, dash="dash"),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=eps,
            y=rms_is.tolist(),
            mode="lines",
            name="Off-policy IS TD",
            line=dict(color="#00CC96", width=2),
        )
    )
    fig.update_layout(
        title="RMS Error vs Episodes",
        xaxis_title="Episodes",
        yaxis_title="RMS Error",
        template="plotly_white",
        height=370,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


def _value_fig(
    V_true: np.ndarray, V_on: np.ndarray, V_noi: np.ndarray, V_is: np.ndarray
) -> go.Figure:
    states = list(range(1, N_STATES + 1))
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=states,
            y=V_true.tolist(),
            mode="lines",
            name="True V^π",
            line=dict(color="black", width=2, dash="dot"),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=states,
            y=V_on.tolist(),
            mode="lines",
            name="On-policy TD",
            line=dict(color="#636EFA", width=2),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=states,
            y=V_noi.tolist(),
            mode="lines",
            name="Off-policy (no IS)",
            line=dict(color="#EF553B", width=2, dash="dash"),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=states,
            y=V_is.tolist(),
            mode="lines",
            name="Off-policy IS TD",
            line=dict(color="#00CC96", width=2),
        )
    )
    fig.update_layout(
        title="Learned Value Functions vs True V^π",
        xaxis_title="State",
        yaxis_title="V(s)",
        template="plotly_white",
        height=380,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


# ── Page ───────────────────────────────────────────────────────────────────────


def show():
    st.title("Importance Sampling with Function Approximation")
    st.markdown(
        '<div style="background:#7B2D8B;height:3px;'
        'border-radius:2px;margin-bottom:1rem;"></div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        "**Section 9 — Off-Policy Methods with Function Approximation (Chapter 11)**"
    )

    # ── Concept ───────────────────────────────────────────────────────────────
    st.header("Off-Policy Learning with Function Approximation")
    st.markdown(r"""
Off-policy learning requires data generated by a **behaviour policy** $b$ to evaluate or
improve a **target policy** $\pi$.  With tabular methods, importance sampling (IS) corrects
for the mismatch.  With function approximation the situation is more delicate.

The semi-gradient TD update becomes:

$$\mathbf{w}_{t+1} = \mathbf{w}_t + \alpha \,\rho_t\, \delta_t\, \nabla\hat{v}(S_t, \mathbf{w}_t)$$

where the per-step IS ratio is:

$$\rho_t = \frac{\pi(A_t \mid S_t)}{b(A_t \mid S_t)}$$

Without the IS ratio the update targets the wrong value function (the one for $b$, not $\pi$).
""")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**On-policy semi-gradient TD**")
        st.latex(
            r"\mathbf{w} \leftarrow \mathbf{w} + \alpha\,\delta_t\,\boldsymbol{\phi}(S_t)"
        )
        st.caption(
            "Follows the gradient of MSVE under the on-policy distribution $\\mu_\\pi$.  "
            "Semi-gradient because the target $R + \\gamma\\hat{v}(S')$ also depends on $\\mathbf{w}$."
        )
    with col2:
        st.markdown("**IS-corrected off-policy semi-gradient TD**")
        st.latex(
            r"\mathbf{w} \leftarrow \mathbf{w} + \alpha\,\rho_t\,\delta_t\,\boldsymbol{\phi}(S_t)"
        )
        st.caption(
            "The IS ratio $\\rho_t$ re-weights each update to approximate the on-policy gradient.  "
            "Unbiased in expectation, but IS introduces variance — especially when $\\pi$ and $b$ differ greatly."
        )

    st.info(
        "**Is this enough?**  For linear function approximation and gradient MC, IS corrects the "
        "update fully.  For semi-gradient TD, IS in the per-step ratio does not make the overall "
        "algorithm a true gradient method — the *deadly triad* (FA + bootstrapping + off-policy) "
        "can still cause instability, as explored on the next page."
    )

    with st.expander("Variance of importance sampling"):
        st.markdown(r"""
The IS ratio $\rho_t = \pi(a) / b(a)$ can be large when $b$ assigns low probability to actions
that $\pi$ favours.  Over a trajectory of length $T$, the product $\prod_{t} \rho_t$ can
have exponentially high variance, leading to unstable learning.

**Mitigations**:
- **Per-decision IS** (used here): apply $\rho_t$ only at the step where the action was taken,
  rather than accumulating products.  This is sufficient for TD methods.
- **Truncated IS / weighted IS**: cap the ratio at some maximum value $c$ to reduce variance
  at the cost of bias.
- **Gradient TD methods** (TDC, GTD2): avoid IS altogether by directly minimising the
  projected Bellman error — stable under the deadly triad.
""")

    st.divider()

    # ── Simulation ─────────────────────────────────────────────────────────────
    st.header("Interactive Simulation — 19-State Random Walk")
    st.markdown(
        "A 19-state random walk with terminal states at each end.  Reward +1 on exiting right, 0 on exiting left.  "
        "The **target policy** $\\pi$ moves right with $p_\\pi$; the **behaviour policy** $b$ is always "
        "50/50 (random walk).  Features are state aggregation into groups."
    )

    col1, col2 = st.columns(2)
    with col1:
        p_right = st.slider("Target policy p(right) — π", 0.51, 0.95, 0.7, 0.01)
        n_groups = st.slider("Number of aggregation groups", 2, 19, 10, 1)
        alpha = st.slider("α (step size)", 0.0001, 0.1, 0.005, 0.0001, format="%.4f")
    with col2:
        n_episodes = st.slider("Episodes", 100, 5000, 1000, 100)
        seed = st.number_input("Random seed", value=42, step=1)
        rho_r = p_right / 0.5
        rho_l = (1.0 - p_right) / 0.5
        st.caption(
            f"IS ratio for right step: **{rho_r:.2f}**  \n"
            f"IS ratio for left step: **{rho_l:.2f}**  \n"
            f"Behaviour policy $b$: **50% left, 50% right**"
        )

    if st.button("Run Comparison", type="primary"):
        with st.spinner("Training three TD variants…"):
            rms_on, rms_noi, rms_is, V_true = run_comparison(
                n_groups, alpha, p_right, n_episodes, int(seed)
            )
            _, V_on, V_noi, V_is = final_value_functions(
                n_groups, alpha, p_right, n_episodes, int(seed)
            )

        st.header("Results")
        st.plotly_chart(_rms_fig(rms_on, rms_noi, rms_is), use_container_width=True)
        st.plotly_chart(_value_fig(V_true, V_on, V_noi, V_is), use_container_width=True)

        col_a, col_b, col_c = st.columns(3)
        with col_a:
            st.metric("On-policy TD — final RMS", f"{rms_on[-1]:.4f}")
        with col_b:
            st.metric("Off-policy (no IS) — final RMS", f"{rms_noi[-1]:.4f}")
        with col_c:
            st.metric("Off-policy IS — final RMS", f"{rms_is[-1]:.4f}")

        st.header("Key Takeaways")
        st.success(
            "The **IS-corrected** off-policy TD should converge to a value function closer to "
            f"$V^\\pi$ (the target policy's values, with $p_{{\\rm right}} = {p_right}$) than "
            "the uncorrected version, which drifts towards the behaviour policy's value function "
            "(the symmetric random walk, $p_{\\rm right} = 0.5$)."
        )
        st.info(
            "Notice that IS correction adds **variance** — the IS curve is often noisier than "
            "the uncorrected one.  Increasing $\\alpha$ amplifies this effect, while lower $\\alpha$ "
            "reduces noise but slows learning."
        )
