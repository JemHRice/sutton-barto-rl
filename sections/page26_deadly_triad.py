import numpy as np
import plotly.graph_objects as go
import streamlit as st

# ── Baird's counterexample ─────────────────────────────────────────────────────
# 7 non-terminal states: 0–5 are "dashed", 6 is "solid".
# Behaviour b: from any state, go to dashed (uniform over 0-5) w.p. 6/7, solid w.p. 1/7.
# Target π: from any state, always go to solid (state 6).
# IS ratios: ρ(→dashed) = 0/1 = 0, ρ(→solid) = 1 / (1/7) = 7.
# All rewards = 0.  γ = 0.99.

N_STATES = 7
N_FEATURES = 8
GAMMA = 0.99
SOLID = 6
DASHED = list(range(6))


def _phi(s: int) -> np.ndarray:
    phi = np.zeros(N_FEATURES)
    if s < 6:
        phi[s] = 2.0
        phi[7] = 1.0
    else:
        phi[6] = 1.0
        phi[7] = 2.0
    return phi


def _v(s: int, w: np.ndarray) -> float:
    return float(_phi(s) @ w)


# ── Training ───────────────────────────────────────────────────────────────────


@st.cache_data
def run_baird(
    alpha: float,
    n_steps: int,
    seed: int,
    use_tdc: bool,
    beta_ratio: float,
) -> np.ndarray:
    """
    Simulate IS semi-gradient TD (or TDC) on Baird's counterexample.
    Returns weight matrix of shape (n_steps, N_FEATURES).
    Behaviour: 6/7 dashed, 1/7 solid.  Target: always solid.
    """
    rng = np.random.default_rng(seed)
    w = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 10.0])
    h = np.zeros(N_FEATURES)
    beta = alpha * beta_ratio

    log = np.zeros((n_steps, N_FEATURES))

    s = int(rng.integers(N_STATES))

    for t in range(n_steps):
        log[t] = w.copy()

        # Sample from behaviour policy
        if rng.random() < 6 / 7:
            s2 = int(rng.integers(6))  # one of the dashed states
            rho = 0.0  # π(→dashed) = 0
        else:
            s2 = SOLID
            rho = 7.0  # π(→solid)/b(→solid) = 1/(1/7) = 7

        phi_s = _phi(s)
        phi_s2 = _phi(s2)
        delta = 0.0 + GAMMA * _v(s2, w) - _v(s, w)  # reward = 0

        if use_tdc:
            w += alpha * rho * (delta * phi_s - GAMMA * float(h @ phi_s) * phi_s2)
            h += beta * rho * (delta - float(h @ phi_s)) * phi_s
        else:
            w += alpha * rho * delta * phi_s

        s = s2

    return log


# ── Figures ────────────────────────────────────────────────────────────────────

_COLOURS = [
    "#636EFA",
    "#EF553B",
    "#00CC96",
    "#AB63FA",
    "#FFA15A",
    "#19D3F3",
    "#FF6692",
    "#B6E880",
]


def _weights_fig(log: np.ndarray, title: str) -> go.Figure:
    steps = list(range(log.shape[0]))
    fig = go.Figure()
    for i in range(N_FEATURES):
        label = f"w[{i}]" if i < 7 else "w[7]"
        fig.add_trace(
            go.Scatter(
                x=steps,
                y=log[:, i].tolist(),
                mode="lines",
                name=label,
                line=dict(color=_COLOURS[i % len(_COLOURS)], width=1.5),
            )
        )
    fig.update_layout(
        title=title,
        xaxis_title="Step",
        yaxis_title="Weight value",
        template="plotly_white",
        height=400,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


def _diverge_detect(log: np.ndarray, threshold: float = 1e6) -> int | None:
    """Return first step where any weight exceeds threshold, or None."""
    mask = np.abs(log) > threshold
    steps = np.where(mask.any(axis=1))[0]
    return int(steps[0]) if len(steps) else None


# ── Page ───────────────────────────────────────────────────────────────────────


def show():
    st.title("The Deadly Triad — Baird's Counterexample")
    st.markdown(
        '<div style="background:#7B2D8B;height:3px;'
        'border-radius:2px;margin-bottom:1rem;"></div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        "**Section 9 — Off-Policy Methods with Function Approximation (Chapter 11)**"
    )

    # ── Concept ───────────────────────────────────────────────────────────────
    st.header("When All Three Combine: Divergence")
    st.markdown(r"""
The **deadly triad** is the conjunction of three properties that individually are harmless,
but together can cause catastrophic divergence of the weight vector:

1. **Function approximation** — the value function is parameterised (e.g. linear features)
2. **Bootstrapping** — the TD target depends on the current estimate of $\hat{v}$
3. **Off-policy training** — the update distribution (induced by $b$) does not match $\pi$

**Baird's counterexample** (S&B §11.2) is the simplest known MDP that triggers this.
""")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("**States & transitions**")
        st.markdown(
            "- 7 non-terminal states: states 0–5 (*dashed*), state 6 (*solid*).\n"
            "- Behaviour $b$: from any state, go to a random dashed state (prob 6/7) or the solid state (prob 1/7).\n"
            "- Target $\\pi$: always transition to the solid state.\n"
            "- All rewards = 0, $\\gamma = 0.99$."
        )
    with col2:
        st.markdown("**Feature vectors** (8-dimensional)")
        st.latex(r"""\boldsymbol{\phi}(s) = \begin{cases}
2\mathbf{e}_s + \mathbf{e}_7 & s \in \{0,\ldots,5\} \\
\mathbf{e}_6 + 2\mathbf{e}_7 & s = 6
\end{cases}""")
        st.caption("$\\mathbf{e}_i$ is the $i$-th standard basis vector (0-indexed).")
    with col3:
        st.markdown("**IS ratios**")
        st.latex(r"\rho(\to\text{dashed}) = \frac{\pi}{b} = \frac{0}{6/7} = 0")
        st.latex(r"\rho(\to\text{solid})  = \frac{\pi}{b} = \frac{1}{1/7} = 7")
        st.caption(
            "Large $\\rho = 7$ when moving to solid creates a destabilising positive feedback loop."
        )

    st.info(
        "**Why does it diverge?**  When the behaviour policy moves to the solid state (prob 1/7), "
        "$\\rho = 7$ amplifies the update sevenfold.  Because the solid state's feature vector "
        "overlaps with $w[7]$, which also appears in all dashed state features, large updates "
        "propagate through all weights — creating a runaway feedback loop."
    )

    with st.expander("Mathematical intuition for the divergence"):
        st.markdown(r"""
Starting from $\mathbf{w} = [1,1,1,1,1,1,1,10]$, the value estimates are:

$$\hat{v}(\text{dashed}_i) = 2w_i + w_7 \approx 12$$
$$\hat{v}(\text{solid}) = w_6 + 2w_7 \approx 21$$

When we transition to the solid state under $b$ (which happens with prob 1/7):

$$\delta = 0 + \gamma \cdot 21 - 12 \approx 8.8 > 0$$

The update $\mathbf{w} \leftarrow \mathbf{w} + \alpha \cdot 7 \cdot \delta \cdot \boldsymbol{\phi}(s)$ increases the weights.
This raises both $\hat{v}(\text{dashed})$ and $\hat{v}(\text{solid})$, which then produces
an even larger $\delta$ on the next visit — a positive feedback loop leading to divergence.

With any two of the three triad conditions, convergence can be guaranteed.  With all three,
even simple 7-state linear problems diverge.
""")

    with st.expander("Contrast: Gradient TD (TDC) — stable under the deadly triad"):
        st.markdown(r"""
**TDC** (TD with gradient correction) modifies the semi-gradient update to make it a true
stochastic gradient descent on the **Mean Squared Projected Bellman Error (MSPBE)**:

$$\mathbf{w}_{t+1} = \mathbf{w}_t + \alpha\,\rho_t\Bigl(\delta_t \boldsymbol{\phi}_t
    - \gamma\bigl(\mathbf{h}_t^\top\boldsymbol{\phi}_t\bigr)\boldsymbol{\phi}_{t+1}\Bigr)$$

$$\mathbf{h}_{t+1} = \mathbf{h}_t + \beta\,\rho_t\Bigl(\delta_t - \mathbf{h}_t^\top\boldsymbol{\phi}_t\Bigr)\boldsymbol{\phi}_t$$

The secondary weight vector $\mathbf{h}$ provides an estimate of the gradient correction.
TDC converges for any linear FA even with all three triad conditions active — at the cost
of tracking an additional parameter vector and tuning $\beta$.
""")

    st.divider()

    # ── Simulation ─────────────────────────────────────────────────────────────
    st.header("Interactive Simulation")
    st.markdown(
        "Observe the weight trajectory for **IS semi-gradient TD** vs **TDC** on Baird's counterexample.  "
        "Both start from $\\mathbf{w} = [1,1,1,1,1,1,1,10]$."
    )

    col1, col2 = st.columns(2)
    with col1:
        alpha = st.slider("α (step size)", 0.001, 0.1, 0.01, 0.001, format="%.3f")
        n_steps = st.slider("Steps", 500, 20_000, 5_000, 500, format="%d")
    with col2:
        seed = st.number_input("Random seed", value=0, step=1)
        beta_ratio = st.slider(
            "β / α ratio (TDC secondary step)",
            0.1,
            1.0,
            0.5,
            0.05,
            help="TDC uses β = β_ratio × α for the secondary weight h.",
        )

    if st.button("Run Comparison", type="primary"):
        with st.spinner("Simulating IS TD and TDC on Baird's counterexample…"):
            log_td = run_baird(
                alpha, n_steps, int(seed), use_tdc=False, beta_ratio=beta_ratio
            )
            log_tdc = run_baird(
                alpha, n_steps, int(seed), use_tdc=True, beta_ratio=beta_ratio
            )

        st.header("Results")

        div_step = _diverge_detect(log_td)

        tab1, tab2 = st.tabs(["IS Semi-gradient TD (diverges)", "TDC (stable)"])
        with tab1:
            if div_step is not None:
                st.warning(
                    f"Divergence detected at step {div_step:,} (|w| > 10⁶).  "
                    "Plot truncated to the last stable window."
                )
                safe = min(div_step + 100, n_steps)
                st.plotly_chart(
                    _weights_fig(
                        log_td[:safe], "IS Semi-gradient TD — weight trajectories"
                    ),
                    use_container_width=True,
                )
            else:
                st.info(
                    "No divergence detected in this run — try more steps or a larger α."
                )
                st.plotly_chart(
                    _weights_fig(log_td, "IS Semi-gradient TD — weight trajectories"),
                    use_container_width=True,
                )

        with tab2:
            st.plotly_chart(
                _weights_fig(log_tdc, "TDC — weight trajectories"),
                use_container_width=True,
            )
            final_norm = float(np.linalg.norm(log_tdc[-1]))
            st.metric(
                "Final ‖w‖ (TDC)",
                f"{final_norm:.4f}",
                delta=f"{final_norm - float(np.linalg.norm(log_tdc[0])):.4f}",
            )

        st.header("Key Takeaways")
        if div_step is not None:
            st.error(
                f"IS semi-gradient TD diverged at step {div_step:,}.  "
                "All three deadly-triad conditions are active: linear FA, TD bootstrapping, "
                "and off-policy IS updates.  The weights grow without bound."
            )
        st.success(
            "**TDC** remains bounded throughout.  It converges to the MSPBE minimum by "
            "correcting the semi-gradient with a secondary weight vector $\\mathbf{h}$.  "
            "This makes it a true gradient method even under the deadly triad."
        )
        st.info(
            "Increase $\\alpha$ or the number of steps to trigger divergence more quickly.  "
            "Note that TDC requires tuning $\\beta$ (the secondary learning rate) — "
            "too small and $\\mathbf{h}$ converges slowly; too large and it overshoots."
        )
