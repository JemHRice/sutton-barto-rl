import numpy as np
import plotly.graph_objects as go
import streamlit as st

from utils.random_walk_1000 import RandomWalk1000, compute_true_values
from utils.feature_bases import BASIS_REGISTRY

# ── Helpers ────────────────────────────────────────────────────────────────────


@st.cache_data
def _true_values() -> np.ndarray:
    return compute_true_values()


# ── Algorithms ─────────────────────────────────────────────────────────────────


@st.cache_data
def run_nstep_td(
    basis_name: str,
    basis_param: int,
    alpha: float,
    n: int,
    n_episodes: int,
    seed: int,
) -> np.ndarray:
    """On-policy n-step semi-gradient TD. Returns rms_errors [n_episodes]."""
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
        states: list[int] = [s]
        rewards: list[float] = [0.0]  # 1-indexed: rewards[i] = R_i
        T = float("inf")
        t = 0

        while True:
            if t < T:
                ns, r, done = env.step()
                rewards.append(r)
                if done:
                    T = t + 1
                else:
                    states.append(ns)

            tau = t - n + 1
            if tau >= 0:
                # rewards[i] = R_i (1-indexed with dummy rewards[0] = 0)
                G = sum(
                    gamma ** (i - tau - 1) * rewards[i]
                    for i in range(tau + 1, min(tau + n, int(T)) + 1)
                )
                if tau + n < T:
                    phi_tau_n = phi_fn(
                        states[tau + n], RandomWalk1000.N_STATES, basis_param
                    )
                    G += gamma**n * float(w @ phi_tau_n)
                phi_tau = phi_fn(states[tau], RandomWalk1000.N_STATES, basis_param)
                delta = G - float(w @ phi_tau)
                w += alpha * delta * phi_tau

            if tau == T - 1:
                break
            t += 1

        V_hat = np.array(
            [
                float(w @ phi_fn(s + 1, RandomWalk1000.N_STATES, basis_param))
                for s in range(RandomWalk1000.N_STATES)
            ]
        )
        rms[ep] = float(np.sqrt(np.mean((V_hat - V_true) ** 2)))

    return rms


@st.cache_data
def run_td_lambda(
    basis_name: str,
    basis_param: int,
    alpha: float,
    lam: float,
    n_episodes: int,
    seed: int,
) -> np.ndarray:
    """On-policy TD(λ) (accumulating traces). Returns rms_errors [n_episodes]."""
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
        z = np.zeros(n_feat)

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
            z = gamma * lam * z + phi_s
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

    return rms


# ── Figures ────────────────────────────────────────────────────────────────────

_NSTEP_COLOURS = {
    1: "#636EFA",
    2: "#AB63FA",
    5: "#00CC96",
    10: "#FFA15A",
    20: "#EF553B",
    50: "#19D3F3",
}
_LAMBDA_COLOURS = {
    0.0: "#636EFA",
    0.5: "#AB63FA",
    0.9: "#00CC96",
    0.95: "#FFA15A",
    0.99: "#EF553B",
}


def _comparison_fig(
    nstep_results: dict,
    lambda_results: dict,
    smooth: int,
) -> go.Figure:
    fig = go.Figure()

    def _add(y, name, colour, dash="solid"):
        if smooth > 1 and len(y) >= smooth:
            kernel = np.ones(smooth) / smooth
            sm = np.convolve(y, kernel, mode="valid")
            x = list(range(smooth, len(y) + 1))
        else:
            sm = y
            x = list(range(1, len(y) + 1))
        fig.add_trace(
            go.Scatter(
                x=x,
                y=sm.tolist(),
                mode="lines",
                name=name,
                line=dict(color=colour, width=2, dash=dash),
            )
        )

    for n, rms in sorted(nstep_results.items()):
        _add(rms, f"n-step n={n}", _NSTEP_COLOURS.get(n, "#888888"), dash="dash")
    for lam, rms in sorted(lambda_results.items()):
        _add(rms, f"TD(λ) λ={lam}", _LAMBDA_COLOURS.get(lam, "#888888"), dash="solid")

    fig.update_layout(
        title="n-step TD vs TD(λ) — RMS Error",
        xaxis_title="Episodes",
        yaxis_title="RMS Error",
        template="plotly_white",
        height=420,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


def _equivalence_fig() -> go.Figure:
    """Show the n ≈ 1/(1-λγ) relationship."""
    lams = np.linspace(0.0, 0.99, 200)
    gamma = 1.0
    n_eff = 1.0 / (1.0 - lams * gamma + 1e-12)
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=lams.tolist(),
            y=n_eff.tolist(),
            mode="lines",
            line=dict(color="#636EFA", width=2),
        )
    )
    # Mark key λ values
    key_lams = [0.0, 0.5, 0.9, 0.95, 0.99]
    key_ns = [1.0 / (1.0 - l + 1e-12) for l in key_lams]
    fig.add_trace(
        go.Scatter(
            x=key_lams,
            y=key_ns,
            mode="markers+text",
            text=[f"n≈{n:.0f}" for n in key_ns],
            textposition="top center",
            marker=dict(size=8, color="#EF553B"),
            showlegend=False,
        )
    )
    fig.update_layout(
        title="Effective Horizon: n ≈ 1/(1−λ)",
        xaxis_title="λ",
        yaxis_title="Effective n (steps)",
        template="plotly_white",
        height=340,
        yaxis_range=[0, 120],
    )
    return fig


def _final_rms_scatter(nstep_results: dict, lambda_results: dict) -> go.Figure:
    """Scatter of final RMS vs effective horizon for both methods."""
    fig = go.Figure()

    ns = sorted(nstep_results.keys())
    yrms = [float(nstep_results[n][-1]) for n in ns]
    fig.add_trace(
        go.Scatter(
            x=ns,
            y=yrms,
            mode="markers+lines",
            name="n-step TD",
            marker=dict(size=10, color="#636EFA"),
            line=dict(color="#636EFA", dash="dash"),
        )
    )

    lams = sorted(lambda_results.keys())
    n_effs = [1.0 / (1.0 - l + 1e-12) for l in lams]
    yrms_l = [float(lambda_results[l][-1]) for l in lams]
    fig.add_trace(
        go.Scatter(
            x=n_effs,
            y=yrms_l,
            mode="markers+lines",
            name="TD(λ)",
            marker=dict(size=10, color="#EF553B"),
            line=dict(color="#EF553B"),
        )
    )

    fig.update_layout(
        title="Final RMS Error vs Effective Horizon",
        xaxis_title="Effective n (steps)",
        yaxis_title="Final RMS Error",
        template="plotly_white",
        height=340,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


# ── Page ───────────────────────────────────────────────────────────────────────


def show():
    st.title("Unifying n-step Methods and Eligibility Traces")
    st.markdown(
        '<div style="background:#7B2D8B;height:3px;'
        'border-radius:2px;margin-bottom:1rem;"></div>',
        unsafe_allow_html=True,
    )
    st.markdown("**Section 10 — Eligibility Traces (Chapter 12)**")

    # ── Concept ───────────────────────────────────────────────────────────────
    st.header("Two Ways to Bridge TD and Monte Carlo")
    st.markdown(r"""
Both **n-step TD** and **TD(λ)** interpolate between the one-step TD target ($n=1$,
$\lambda=0$) and the Monte Carlo return ($n=T$, $\lambda=1$).  They do so in fundamentally
different ways:

| Property | n-step TD | TD(λ) |
|---|---|---|
| **Mechanism** | Waits $n$ steps, then bootstraps | Traces weighted geometric mixtures online |
| **Memory** | $O(n \times d)$ — stores last $n$ transitions | $O(d)$ — one extra trace vector |
| **Update** | Batch: at step $\tau + n$ | Online: at every step |
| **Horizon** | Fixed integer $n$ | Soft: $\approx 1/(1-\gamma\lambda)$ |
| **Off-policy** | Requires $n$-step IS ratio $\prod \rho_t$ | Requires per-step IS (or Retrace) |

The key relationship is the **effective horizon**:

$$n_{\text{eff}} \approx \frac{1}{1 - \gamma\lambda}$$

A TD(λ) agent with parameter $\lambda$ uses approximately the same depth of credit
assignment as an $n$-step agent with $n \approx 1/(1-\lambda)$ (for $\gamma = 1$).
""")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**n-step return**")
        st.latex(r"""G_t^{(n)} = \sum_{k=0}^{n-1} \gamma^k R_{t+k+1}
+ \gamma^n \hat{v}(S_{t+n}, \mathbf{w})""")
        st.caption(
            "Uses a bootstrap from state $S_{t+n}$.  "
            "Large $n$ reduces bias but increases variance.  "
            "Requires storing the last $n$ steps."
        )
    with col2:
        st.markdown("**λ-return**")
        st.latex(r"""G_t^\lambda = (1-\lambda)\sum_{n=1}^{T-t-1} \lambda^{n-1} G_t^{(n)}
+ \lambda^{T-t-1} G_t""")
        st.caption(
            "Geometric mixture of all $n$-step returns.  "
            "Each additional step is discounted by $\\lambda$.  "
            "Equivalent to TD(λ) with accumulating traces (backward view)."
        )

    st.info(
        "**Why prefer TD(λ) over n-step TD?**  Both achieve similar performance at their "
        "respective optimal parameters, but TD(λ) is strictly more memory-efficient ($O(d)$ vs "
        "$O(nd)$) and updates on every step rather than waiting $n$ steps.  "
        "For online learning, eligibility traces are the clear winner."
    )

    with st.expander("The compound update and Retrace(λ)"):
        st.markdown(r"""
**Off-policy traces** are more subtle.  Per-step IS ratios $\rho_t = \pi(a)/b(a)$ must be
folded into the trace update.  One principled approach is **Retrace(λ)**:

$$\mathbf{z}_t = \gamma\lambda\,\min(c, \rho_t)\,\mathbf{z}_{t-1} + \boldsymbol{\phi}(S_t, A_t)$$

where $c$ (often 1) clips the IS ratio to prevent trace explosion.  Retrace(λ) is
safe (converges off-policy) and retains the efficiency of traces — it is the backbone
of DQN successors like IMPALA and AGENT57.

For on-policy methods ($\rho_t = 1$ always), Retrace reduces to standard SARSA(λ).
""")

    with st.expander("Truncated IS and the bias-variance trade-off"):
        st.markdown(r"""
Both n-step IS and TD(λ) with IS face a bias-variance trade-off in off-policy settings.

- **Large $n$ or $\lambda$** → low bias (more of the true return) but high variance
  (product of many IS ratios $\prod_t \rho_t$ can be huge).
- **Small $n$ or $\lambda$** → high bias (heavy bootstrapping) but low variance.

Truncated IS methods (V-trace, used in IMPALA) clip each $\rho_t$ at 1, introducing
a small bias but keeping variance finite.  This trade-off is one of the key design
decisions in large-scale distributed RL systems.
""")

    st.plotly_chart(_equivalence_fig(), use_container_width=True)
    st.caption(
        "The effective horizon $n \\approx 1/(1-\\lambda)$ grows rapidly near $\\lambda = 1$.  "
        "Matching $\\lambda$ to a given $n$ provides a principled starting point when "
        "switching between the two families of algorithms."
    )

    st.divider()

    # ── Simulation ─────────────────────────────────────────────────────────────
    st.header("Side-by-Side Comparison — 1000-State Random Walk")
    st.markdown(
        "Compare n-step TD and TD(λ) directly on the same task.  "
        "Use the equivalence chart above to choose matching parameters."
    )

    col1, col2 = st.columns(2)
    with col1:
        basis_name = st.selectbox("Feature basis", list(BASIS_REGISTRY.keys()), index=2)
        info = BASIS_REGISTRY[basis_name]
        plabel, pdef, pmin, pmax, pstep = info["param"]
        basis_param = st.slider(plabel, pmin, pmax, pdef, pstep)
        alpha = st.slider("α (step size)", 0.0001, 0.05, 0.002, 0.0001, format="%.4f")
    with col2:
        nsteps_sel = st.multiselect(
            "n values for n-step TD",
            [1, 2, 5, 10, 20, 50],
            default=[1, 5, 20],
        )
        lambdas_sel = st.multiselect(
            "λ values for TD(λ)",
            [0.0, 0.5, 0.9, 0.95, 0.99],
            default=[0.0, 0.9, 0.99],
        )
        n_episodes = st.slider("Episodes", 100, 3000, 1000, 100)
        seed = st.number_input("Random seed", value=42, step=1)
        smooth = st.slider("Rolling mean window", 1, 100, 20)

    if st.button("Run Comparison", type="primary") and (nsteps_sel or lambdas_sel):
        nstep_results = {}
        lambda_results = {}

        with st.spinner("Running n-step TD and TD(λ)…"):
            for n in nsteps_sel:
                nstep_results[n] = run_nstep_td(
                    basis_name, basis_param, alpha, n, n_episodes, int(seed)
                )
            for lam in lambdas_sel:
                lambda_results[lam] = run_td_lambda(
                    basis_name, basis_param, alpha, lam, n_episodes, int(seed)
                )

        st.header("Results")
        tab1, tab2 = st.tabs(["Learning Curves", "Final RMS vs Horizon"])
        with tab1:
            st.plotly_chart(
                _comparison_fig(nstep_results, lambda_results, smooth),
                use_container_width=True,
            )
            st.caption(
                "Dashed lines: n-step TD.  Solid lines: TD(λ).  "
                "Methods with similar effective horizons should show similar convergence."
            )
        with tab2:
            st.plotly_chart(
                _final_rms_scatter(nstep_results, lambda_results),
                use_container_width=True,
            )
            st.caption(
                "Both methods trace out a U-shaped or monotone curve as their horizon increases.  "
                "The minimum (optimal horizon) is typically around $n = 5$–$20$ or $\\lambda = 0.9$–$0.95$."
            )

        st.header("Key Takeaways")
        st.success(
            "n-step TD and TD(λ) achieve **similar performance** when their effective horizons "
            "are matched ($n \\approx 1/(1-\\lambda)$).  TD(λ) is preferred in practice because "
            "it is $O(d)$ memory and updates online."
        )
        st.info(
            "The 1000-state random walk has a nearly linear true value function, "
            "so a good feature basis (e.g. Fourier Cosine) matters more than the "
            "choice of $n$ or $\\lambda$.  On problems with longer reward delays, "
            "the optimal horizon shifts to larger values of $n$ or $\\lambda$."
        )
