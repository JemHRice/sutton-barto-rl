import numpy as np
import plotly.graph_objects as go
import streamlit as st

from utils.random_walk_1000 import RandomWalk1000
from utils.feature_bases import BASIS_REGISTRY

# ── Helpers ────────────────────────────────────────────────────────────────────


@st.cache_data
def _true_values_pi(p_right: float) -> np.ndarray:
    """
    True V^π for the ±1 biased random walk under policy π.
    States 1..N_STATES; reward +1 on right exit, 0 on left exit.
    Closed-form: V(s) = (1 - r^s) / (1 - r^{N+1}) where r = (1-p)/p, or s/(N+1) for p=0.5.
    """
    N = RandomWalk1000.N_STATES
    s = np.arange(1, N + 1, dtype=float)
    if abs(p_right - 0.5) < 1e-12:
        return s / (N + 1)
    r = (1.0 - p_right) / p_right
    return (1.0 - r**s) / (1.0 - r ** (N + 1))


# ── Training ───────────────────────────────────────────────────────────────────


@st.cache_data
def run_gradient_td(
    basis_name: str,
    basis_param: int,
    alpha: float,
    beta_ratio: float,
    p_right_target: float,
    n_episodes: int,
    seed: int,
    algorithm: str,  # "semi_td", "tdc", or "gtd2"
) -> tuple[np.ndarray, np.ndarray]:
    """
    Run off-policy semi-gradient TD, TDC, or GTD2 on the 1000-state random walk.
    Target π: p_right_target.  Behaviour b: 50% uniform.
    Returns (rms_errors [n_episodes], V_hat [1000]).
    """
    rng = np.random.default_rng(seed)
    V_true = _true_values_pi(p_right_target)
    info = BASIS_REGISTRY[basis_name]
    phi_fn = info["fn"]
    n_feat = info["n_feat"](basis_param)
    env = RandomWalk1000(seed=seed)
    w = np.zeros(n_feat)
    h = np.zeros(n_feat)
    beta = alpha * beta_ratio
    gamma = 1.0
    p_b = 0.5

    rho_right = p_right_target / p_b
    rho_left = (1.0 - p_right_target) / p_b

    rms = np.zeros(n_episodes)

    for ep in range(n_episodes):
        s = env.reset()
        while True:
            right = rng.random() < p_b
            rho = rho_right if right else rho_left
            step = 1 if right else -1
            ns = s + step
            if ns > RandomWalk1000.N_STATES:
                r, done = 1.0, True
                ns = RandomWalk1000.N_STATES + 1
            elif ns < 1:
                r, done = 0.0, True
                ns = 0
            else:
                r, done = 0.0, False

            phi_s = phi_fn(s, RandomWalk1000.N_STATES, basis_param)
            phi_s2 = (
                np.zeros(n_feat)
                if done
                else phi_fn(ns, RandomWalk1000.N_STATES, basis_param)
            )

            v_s = float(w @ phi_s)
            v_s2 = 0.0 if done else float(w @ phi_s2)
            delta = r + gamma * v_s2 - v_s

            if algorithm == "semi_td":
                w += alpha * rho * delta * phi_s

            elif algorithm == "tdc":
                w += alpha * rho * (delta * phi_s - gamma * float(h @ phi_s) * phi_s2)
                h += beta * rho * (delta - float(h @ phi_s)) * phi_s

            elif algorithm == "gtd2":
                w += alpha * rho * phi_s * float(h @ (phi_s - gamma * phi_s2))
                h += beta * rho * (delta - float(h @ phi_s)) * phi_s

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


# ── Figures ────────────────────────────────────────────────────────────────────


def _rms_fig(
    results: list[tuple[str, np.ndarray, str]],
) -> go.Figure:
    fig = go.Figure()
    for name, rms, colour in results:
        eps = list(range(1, len(rms) + 1))
        fig.add_trace(
            go.Scatter(
                x=eps,
                y=rms.tolist(),
                mode="lines",
                name=name,
                line=dict(color=colour, width=2),
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
    V_true: np.ndarray,
    results: list[tuple[str, np.ndarray, str]],
) -> go.Figure:
    states = list(range(1, RandomWalk1000.N_STATES + 1))
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
    for name, V_hat, colour in results:
        fig.add_trace(
            go.Scatter(
                x=states,
                y=V_hat.tolist(),
                mode="lines",
                name=name,
                line=dict(color=colour, width=2),
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
    st.title("Gradient TD Methods — TDC and GTD2")
    st.markdown(
        '<div style="background:#7B2D8B;height:3px;'
        'border-radius:2px;margin-bottom:1rem;"></div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        "**Section 9 — Off-Policy Methods with Function Approximation (Chapter 11)**"
    )

    # ── Concept ───────────────────────────────────────────────────────────────
    st.header("True Gradient Methods for Off-Policy TD")
    st.markdown(r"""
Semi-gradient TD applies the gradient of only part of the squared TD error — it does not
follow the gradient of any objective function.  This is fine on-policy, but off-policy the
updates can diverge under the deadly triad.

**Gradient TD methods** instead minimise the **Mean Squared Projected Bellman Error (MSPBE)**:

$$\overline{\text{MSPBE}}(\mathbf{w}) = \bigl\|\hat{v}_\mathbf{w} - \Pi T^\pi \hat{v}_\mathbf{w}\bigr\|^2_\mu$$

where $\Pi$ projects onto the representable value functions and $\mu$ is the behaviour
distribution.  Crucially, this *is* a well-defined scalar objective, so its gradient is
well-defined too.
""")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**TDC (TD with Gradient Correction)**")
        st.latex(
            r"""\mathbf{w}_{t+1} = \mathbf{w}_t
+ \alpha\rho_t\Bigl(\delta_t\boldsymbol{\phi}_t
  - \gamma\bigl(\mathbf{h}_t^\top\boldsymbol{\phi}_t\bigr)\boldsymbol{\phi}_{t+1}\Bigr)"""
        )
        st.latex(
            r"""\mathbf{h}_{t+1} = \mathbf{h}_t
+ \beta\rho_t\Bigl(\delta_t - \mathbf{h}_t^\top\boldsymbol{\phi}_t\Bigr)\boldsymbol{\phi}_t"""
        )
        st.caption(
            "TDC is the semi-gradient update **plus a correction term** "
            "$-\\alpha\\gamma\\rho(\\mathbf{h}^\\top\\boldsymbol{\\phi})\\boldsymbol{\\phi}'$ "
            "that removes the bias from bootstrapping.  "
            "$\\mathbf{h}$ is a secondary weight vector converging to $\\mathbf{A}^{-1}\\mathbf{b}$."
        )
    with col2:
        st.markdown("**GTD2 (Gradient TD version 2)**")
        st.latex(r"""\mathbf{w}_{t+1} = \mathbf{w}_t
+ \alpha\rho_t\boldsymbol{\phi}_t
  \bigl[\mathbf{h}_t^\top(\boldsymbol{\phi}_t - \gamma\boldsymbol{\phi}_{t+1})\bigr]""")
        st.latex(
            r"""\mathbf{h}_{t+1} = \mathbf{h}_t
+ \beta\rho_t\Bigl(\delta_t - \mathbf{h}_t^\top\boldsymbol{\phi}_t\Bigr)\boldsymbol{\phi}_t"""
        )
        st.caption(
            "GTD2 uses the same secondary vector $\\mathbf{h}$ with a slightly different "
            "primary update.  Both TDC and GTD2 converge off-policy with linear FA.  "
            "TDC typically has lower variance and is preferred in practice."
        )

    st.info(
        "Both methods have **two step sizes** ($\\alpha$ and $\\beta$).  "
        "Setting $\\beta \\ll \\alpha$ ensures the secondary vector tracks the primary "
        "weights slowly enough to remain a valid correction.  A common heuristic: $\\beta = c\\alpha$ "
        "with $c \\in [0.1, 0.5]$."
    )

    with st.expander("Convergence guarantee and the two-timescale argument"):
        st.markdown(r"""
TDC and GTD2 converge to the **MSPBE minimum** (not the TD fixed point) under the condition
that $\beta / \alpha \to 0$ as $\alpha, \beta \to 0$.  The **two-timescale** argument treats
$\mathbf{h}$ as operating on a fast timescale (relative to $\mathbf{w}$) so that it
effectively equilibrates at each step, providing a clean gradient direction for $\mathbf{w}$.

In practice, a fixed ratio $\beta = c\alpha$ works well.  Both methods:
- Converge with linear function approximation off-policy
- Are $O(d)$ per step (same complexity as semi-gradient TD)
- Do *not* require storing or inverting any matrix

The MSPBE minimum differs from the TD fixed point; on-policy, both are close to the MSVE
minimum, but off-policy they can differ significantly.
""")

    st.divider()

    # ── Simulation ─────────────────────────────────────────────────────────────
    st.header("Interactive Simulation — 1000-State Random Walk")
    st.markdown(
        "Off-policy prediction on the 1000-state random walk.  "
        "Target policy $\\pi$ moves right with probability $p_\\pi$.  "
        "Behaviour policy $b$ is always 50/50 (symmetric random walk)."
    )

    col1, col2 = st.columns(2)
    with col1:
        basis_name = st.selectbox("Feature basis", list(BASIS_REGISTRY.keys()), index=2)
        info = BASIS_REGISTRY[basis_name]
        plabel, pdef, pmin, pmax, pstep = info["param"]
        basis_param = st.slider(plabel, pmin, pmax, pdef, pstep)
        p_right = st.slider("Target policy p(right) — π", 0.51, 0.9, 0.7, 0.01)
    with col2:
        alpha = st.slider(
            "α (primary step size)", 0.0001, 0.05, 0.002, 0.0001, format="%.4f"
        )
        beta_ratio = st.slider("β / α ratio (secondary step)", 0.05, 1.0, 0.2, 0.05)
        n_episodes = st.slider("Episodes", 100, 3000, 1000, 100)
        seed = st.number_input("Random seed", value=42, step=1)
        st.caption(
            f"β = {alpha * beta_ratio:.5f}  |  feature dim: {info['n_feat'](basis_param)}"
        )

    algorithms = st.multiselect(
        "Algorithms to run",
        ["Semi-gradient TD", "TDC", "GTD2"],
        default=["Semi-gradient TD", "TDC"],
    )

    if st.button("Run Comparison", type="primary") and algorithms:
        alg_map = {"Semi-gradient TD": "semi_td", "TDC": "tdc", "GTD2": "gtd2"}
        colours = {"Semi-gradient TD": "#EF553B", "TDC": "#636EFA", "GTD2": "#00CC96"}

        with st.spinner("Running gradient TD methods…"):
            rms_results = []
            vf_results = []
            for alg in algorithms:
                rms, V_hat = run_gradient_td(
                    basis_name,
                    basis_param,
                    alpha,
                    beta_ratio,
                    p_right,
                    n_episodes,
                    int(seed),
                    alg_map[alg],
                )
                rms_results.append((alg, rms, colours[alg]))
                vf_results.append((alg, V_hat, colours[alg]))

        V_true = _true_values_pi(p_right)

        st.header("Results")
        st.plotly_chart(_rms_fig(rms_results), use_container_width=True)
        st.plotly_chart(_value_fig(V_true, vf_results), use_container_width=True)

        cols = st.columns(len(algorithms))
        for col, (alg, rms, _) in zip(cols, rms_results):
            with col:
                st.metric(f"{alg} — final RMS", f"{rms[-1]:.4f}")

        st.header("Key Takeaways")
        st.success(
            "**TDC and GTD2** should converge to a stable value function that approximates "
            f"$V^\\pi$ for the target policy ($p_\\pi = {p_right}$).  "
            "Semi-gradient TD may diverge or converge to the wrong fixed point when the "
            "off-policy distribution differs significantly from $\\pi$."
        )
        st.info(
            "Increase $p_\\pi$ (more off-policy) and observe the divergence risk.  "
            "With a Fourier Cosine basis, semi-gradient TD is often stable due to the "
            "near-on-policy structure of the 1000-state walk — try a state aggregation basis "
            "for a sharper contrast."
        )
