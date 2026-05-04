import numpy as np
import plotly.graph_objects as go
import streamlit as st

from utils.random_walk import RandomWalkEnv, compute_true_values


# ── Training ───────────────────────────────────────────────────────────────────

def _log_checkpoints(n_episodes: int) -> list[int]:
    exponents   = np.linspace(0, np.log10(n_episodes), 30)
    raw         = sorted(set(int(round(10 ** e)) for e in exponents))
    checkpoints = sorted({1} | {c for c in raw if 1 <= c <= n_episodes} | {n_episodes})
    return checkpoints


@st.cache_data
def run_td0_vs_mc(n_episodes: int, alpha: float, gamma: float, seed: int) -> dict:
    """
    Run TD(0) and first-visit MC prediction in parallel on the 5-state random walk.
    Both share the same sequence of episodes (same underlying rng seed) for a fair comparison.
    Returns snapshots and RMS errors at log-spaced checkpoints for both methods.
    """
    V_true      = compute_true_values(gamma)
    checkpoints = _log_checkpoints(n_episodes)
    ck_set      = set(checkpoints)

    # TD(0) state
    V_td  = np.zeros(7)
    rms_td: list[float]             = []
    snaps_td: dict[int, np.ndarray] = {}

    # MC state
    V_mc          = np.zeros(7)
    returns_sum   = np.zeros(7)
    returns_count = np.zeros(7, dtype=int)
    rms_mc: list[float]             = []
    snaps_mc: dict[int, np.ndarray] = {}

    env_td = RandomWalkEnv(seed=seed)
    env_mc = RandomWalkEnv(seed=seed)   # same seed → identical episode sequences

    for ep in range(1, n_episodes + 1):
        # ── TD(0) episode ─────────────────────────────────────────────────────
        s = env_td.reset()
        while True:
            ns, r, done = env_td.step()
            V_td[s]    += alpha * (r + gamma * V_td[ns] - V_td[s])
            s           = ns
            if done:
                break

        # ── First-visit MC episode ────────────────────────────────────────────
        s_mc    = env_mc.reset()
        episode = []
        while True:
            ns_mc, r_mc, done_mc = env_mc.step()
            episode.append((s_mc, r_mc))
            if done_mc:
                break
            s_mc = ns_mc

        G           = 0.0
        first_visit: dict[int, float] = {}
        for t in range(len(episode) - 1, -1, -1):
            sv, rv       = episode[t]
            G            = rv + gamma * G
            first_visit[sv] = G

        for sv, Gv in first_visit.items():
            returns_sum[sv]   += Gv
            returns_count[sv] += 1
            V_mc[sv]           = returns_sum[sv] / returns_count[sv]

        # ── Snapshots ─────────────────────────────────────────────────────────
        if ep in ck_set:
            snaps_td[ep] = V_td[1:6].copy()
            snaps_mc[ep] = V_mc[1:6].copy()
            rms_td.append(float(np.sqrt(np.mean((V_td[1:6] - V_true) ** 2))))
            rms_mc.append(float(np.sqrt(np.mean((V_mc[1:6] - V_true) ** 2))))

    checked = [c for c in checkpoints if c in snaps_td]
    return {
        "V_true":      V_true,
        "V_td_final":  V_td[1:6].copy(),
        "V_mc_final":  V_mc[1:6].copy(),
        "checkpoints": checked,
        "snaps_td":    snaps_td,
        "snaps_mc":    snaps_mc,
        "rms_td":      rms_td,
        "rms_mc":      rms_mc,
    }


# ── Visualisations ─────────────────────────────────────────────────────────────

_LABELS = RandomWalkEnv.LABELS


def _make_rms_fig(checkpoints: list, rms_td: list, rms_mc: list) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=checkpoints, y=rms_td, mode="lines+markers", name="TD(0)",
        line=dict(color="#636EFA", width=2.5), marker=dict(size=5),
    ))
    fig.add_trace(go.Scatter(
        x=checkpoints, y=rms_mc, mode="lines+markers", name="MC",
        line=dict(color="#EF553B", width=2.5, dash="dash"), marker=dict(size=5),
    ))
    fig.update_layout(
        title="RMS Error vs Episodes — TD(0) vs First-Visit MC",
        xaxis_title="Episodes (log scale)", yaxis_title="RMS Error",
        xaxis_type="log", template="plotly_white", height=400,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


def _make_values_fig(V_true, V_td, V_mc) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=_LABELS, y=V_true, name="True V(s)",
        marker_color="black", opacity=0.25,
    ))
    fig.add_trace(go.Bar(
        x=_LABELS, y=V_td, name="TD(0)",
        marker_color="#636EFA", opacity=0.85,
    ))
    fig.add_trace(go.Bar(
        x=_LABELS, y=V_mc, name="MC",
        marker_color="#EF553B", opacity=0.85,
    ))
    fig.update_layout(
        title="Final Value Estimates — TD(0) vs MC vs True",
        xaxis_title="State", yaxis_title="V(s)",
        barmode="group", template="plotly_white",
        height=380, yaxis=dict(range=[0, 1.05]),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


# ── Page ───────────────────────────────────────────────────────────────────────

def show():
    st.title("TD(0) Prediction")
    st.markdown("**Section 4 — Temporal Difference Learning**")

    # ── Concept ───────────────────────────────────────────────────────────────
    st.header("Bootstrapping Without Complete Episodes")
    st.markdown(
        r"""
**Temporal Difference (TD) learning** is the central idea in modern RL. Like Monte Carlo it
learns from experience, without a model. Unlike Monte Carlo it does **not** wait until the
end of an episode — it updates the value estimate after every single step.

TD(0) — the simplest TD method — uses the observed reward and the *current estimate* of the
next state's value to update the current state. This is called **bootstrapping**: using an
estimate to update another estimate.
"""
    )

    # Side-by-side update rules
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**TD(0) Update Rule**")
        st.latex(
            r"V(S_t) \leftarrow V(S_t) + \alpha\!\left["
            r"R_{t+1} + \gamma V(S_{t+1}) - V(S_t)\right]"
        )
        st.markdown(
            r"The term in brackets is the **TD error** $\delta_t$: "
            r"$R_{t+1} + \gamma V(S_{t+1})$ is the TD *target* — a one-step return "
            r"using the bootstrap estimate $V(S_{t+1})$."
        )
    with col2:
        st.markdown("**MC Update Rule**")
        st.latex(
            r"V(S_t) \leftarrow V(S_t) + \alpha\!\left[G_t - V(S_t)\right]"
        )
        st.markdown(
            r"$G_t = \sum_{k=0}^{T-t-1} \gamma^k R_{t+k+1}$ is the actual return "
            r"observed over the *rest of the episode*. MC must wait until the episode "
            r"ends before updating."
        )

    st.markdown(
        r"""
### What Bootstrapping Means

In MC, the target $G_t$ is the *true* discounted return — there is no approximation in the
target itself. In TD(0), the target $R_{t+1} + \gamma V(S_{t+1})$ uses our **current guess**
for $V(S_{t+1})$. If that guess is wrong, the target is wrong too — this introduces bias.

But bootstrapping has a major advantage: we can update **online, mid-episode**. TD methods
work in continuing (non-episodic) tasks where there is no terminal state, which MC cannot
handle at all.
"""
    )

    st.warning(
        "**Bias–Variance Tradeoff:** MC has *zero bias* — it targets the actual return. "
        "But actual returns are noisy: a single episode is one random sample. MC has *high variance*. "
        "TD(0) introduces *some bias* (from bootstrapping a possibly-wrong $V$) but has *much lower variance* "
        "because the one-step target $R_{t+1} + \\gamma V(S_{t+1})$ is far less noisy than a full return. "
        "In practice TD often converges faster, especially in long-horizon tasks."
    )

    st.success(
        "**Why TD(0) Matters** — TD is the bridge between Monte Carlo and full Dynamic Programming. "
        "MC uses full returns (no bootstrapping, no model). DP uses full bootstrapping (model required). "
        "TD does partial bootstrapping with no model — the best of both worlds. "
        "It is the backbone of SARSA, Q-Learning, and the methods in the rest of this section."
    )

    with st.expander("Deep Dive — The TD Target and the Bellman Equation"):
        st.markdown(
            r"""
The Bellman expectation equation for policy $\pi$ is:
$$V^\pi(s) = \mathbb{E}_\pi\!\left[R_{t+1} + \gamma V^\pi(S_{t+1}) \mid S_t = s\right]$$

The TD target $R_{t+1} + \gamma V(S_{t+1})$ is a **sample estimate** of the right-hand side.
Because the expectation is over all possible next states and rewards, averaging many such
samples converges to the true Bellman value — which is why TD converges.

MC can be seen as the same equation but replacing $V(S_{t+1})$ with the full future return
$G_{t+1}$. This is unbiased but high-variance. TD uses a one-step lookahead and a bootstrap;
it is biased but low-variance. $n$-step TD methods interpolate between the two.
"""
        )

    st.divider()

    # ── Environment ───────────────────────────────────────────────────────────
    st.header("The Environment — 5-State Random Walk")
    st.markdown(
        r"""
The same 5-state chain used in Section 3 (Page 7). From every state the agent moves left
or right with equal probability. Right exit: $+1$. Left exit: $0$. Start at centre (**C**).
True values (with $\gamma=1$) are the right-exit probabilities: $\tfrac{1}{6}$ through $\tfrac{5}{6}$.

This is purely a **prediction** problem — the policy is fixed (random walk). We just want to
estimate how good each state is under that policy.
"""
    )

    chain_fig = go.Figure()
    positions = [-1, 0, 1, 2, 3, 4, 5]
    labels    = ["L\n(0)", "A", "B", "C", "D", "E", "R\n(+1)"]
    colors    = ["#EF553B", "#636EFA", "#636EFA", "#00CC96",
                 "#636EFA", "#636EFA", "#00CC96"]
    chain_fig.add_trace(go.Scatter(
        x=positions, y=[0] * 7, mode="markers+text",
        marker=dict(size=38, color=colors),
        text=labels, textposition="middle center",
        textfont=dict(size=13, color="white"),
    ))
    chain_fig.add_shape(type="line", x0=-1, x1=5, y0=0, y1=0,
                        line=dict(color="gray", width=2, dash="dot"))
    chain_fig.update_layout(
        title="5-State Random Walk",
        xaxis=dict(visible=False), yaxis=dict(visible=False),
        height=150, margin=dict(l=20, r=20, t=40, b=10),
        template="plotly_white", showlegend=False,
    )
    st.plotly_chart(chain_fig, use_container_width=True)

    st.divider()

    # ── Simulation ─────────────────────────────────────────────────────────────
    st.header("Interactive Simulation")
    st.markdown(
        "Run TD(0) and first-visit MC on the **same sequence of episodes** (same random seed). "
        "Both methods start with $V=0$ for all states. Watch how quickly each converges "
        "and how the RMS error compares."
    )

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        n_episodes = st.slider("Episodes", 100, 5000, 1000, 100)
    with col2:
        alpha = st.slider("α (TD learning rate)", 0.01, 0.5, 0.1, 0.01,
                          help="Only TD(0) uses α; MC averages returns regardless of α.")
    with col3:
        gamma = st.slider("γ (discount)", 0.5, 1.0, 1.0, 0.05)
    with col4:
        seed = st.number_input("Random seed", value=42, step=1)

    if st.button("Run TD(0) vs MC", type="primary"):
        with st.spinner("Running TD(0) and MC prediction…"):
            result = run_td0_vs_mc(n_episodes, alpha, gamma, int(seed))

        st.header("Results")

        # RMS convergence
        fig_rms = _make_rms_fig(
            result["checkpoints"], result["rms_td"], result["rms_mc"]
        )
        st.plotly_chart(fig_rms, use_container_width=True)

        # Final value estimates
        fig_vals = _make_values_fig(
            result["V_true"], result["V_td_final"], result["V_mc_final"]
        )
        st.plotly_chart(fig_vals, use_container_width=True)

        # Metrics
        st.header("Key Takeaways")
        col_a, col_b, col_c = st.columns(3)
        with col_a:
            st.metric("TD(0) final RMS error", f"{result['rms_td'][-1]:.4f}")
        with col_b:
            st.metric("MC final RMS error", f"{result['rms_mc'][-1]:.4f}")
        with col_c:
            winner = "TD(0)" if result["rms_td"][-1] < result["rms_mc"][-1] else "MC"
            st.metric("Lower final error", winner)

        rms_td_arr = np.array(result["rms_td"])
        rms_mc_arr = np.array(result["rms_mc"])
        td_wins_early = int(np.sum(rms_td_arr < rms_mc_arr))
        mc_wins_early = len(rms_td_arr) - td_wins_early

        st.success(
            f"TD(0) had lower RMS error at **{td_wins_early}** of {len(rms_td_arr)} checkpoints; "
            f"MC was lower at **{mc_wins_early}**. "
            "TD's lower variance typically makes it converge faster in early episodes. "
            "With enough episodes both methods reach similar accuracy — the bias in TD "
            "diminishes as $V$ improves."
        )
        st.info(
            f"With α = {alpha}: a larger α makes TD(0) learn faster initially but "
            "leaves more residual noise. Try α = 0.01 vs 0.4 to see the tradeoff."
            if alpha > 0.05
            else f"With α = {alpha}: TD(0) is updating very slowly. Try α = 0.1–0.2 "
                 "for the typical convergence behaviour shown in Sutton & Barto Figure 6.2."
        )

        import pandas as pd
        st.dataframe(pd.DataFrame({
            "State":       _LABELS,
            "True V(s)":   [f"{v:.4f}" for v in result["V_true"]],
            "TD(0) est.":  [f"{v:.4f}" for v in result["V_td_final"]],
            "MC est.":     [f"{v:.4f}" for v in result["V_mc_final"]],
            "TD error":    [f"{v-t:+.4f}" for v, t in zip(result["V_td_final"], result["V_true"])],
            "MC error":    [f"{v-t:+.4f}" for v, t in zip(result["V_mc_final"], result["V_true"])],
        }), use_container_width=True)
