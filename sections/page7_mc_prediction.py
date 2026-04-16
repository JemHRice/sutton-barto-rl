import numpy as np
import plotly.graph_objects as go
import streamlit as st


# ── True values & MC simulation ────────────────────────────────────────────

@st.cache_data
def compute_true_values(gamma: float) -> np.ndarray:
    """
    Iterative policy evaluation for the 5-state random walk.
    States: 0 (left terminal) … 6 (right terminal).
    Returns V[1:6] — true values for non-terminal states.
    """
    V = np.zeros(7)
    for _ in range(100_000):
        delta = 0.0
        for s in range(1, 6):
            r_right = 1.0 if s == 5 else 0.0
            v_new = 0.5 * (gamma * V[s - 1]) + 0.5 * (r_right + gamma * V[s + 1])
            delta = max(delta, abs(V[s] - v_new))
            V[s] = v_new
        if delta < 1e-12:
            break
    return V[1:6].copy()


@st.cache_data
def run_mc_prediction(n_episodes: int, gamma: float, seed: int) -> dict:
    """
    First-visit MC prediction on the 5-state random walk.
    Always starts from the centre (state 3).
    Returns snapshots at log-spaced episode checkpoints + final RMS errors.
    """
    rng = np.random.default_rng(seed)
    V_true = compute_true_values(gamma)

    # Build log-spaced checkpoints
    exponents = np.linspace(0, np.log10(n_episodes), 30)
    raw = sorted(set(int(round(10 ** e)) for e in exponents))
    checkpoints = sorted({1} | {c for c in raw if 1 <= c <= n_episodes} | {n_episodes})

    returns_sum   = np.zeros(7)
    returns_count = np.zeros(7, dtype=int)
    V             = np.zeros(7)

    V_snapshots: dict[int, np.ndarray] = {}
    rms_errors:  list[float]           = []
    checked: set[int]                  = set()

    for ep in range(1, n_episodes + 1):
        # ── Generate episode ──────────────────────────────────────────────
        state   = 3
        episode = []          # list of (state, reward_received_after)

        while True:
            move       = 1 if rng.random() < 0.5 else -1
            next_state = state + move
            if next_state == 6:
                episode.append((state, 1.0))
                break
            elif next_state == 0:
                episode.append((state, 0.0))
                break
            else:
                episode.append((state, 0.0))
                state = next_state

        # ── First-visit MC update ─────────────────────────────────────────
        G            = 0.0
        first_visit  = {}          # state → return at first occurrence

        for t in range(len(episode) - 1, -1, -1):
            s, r   = episode[t]
            G      = r + gamma * G
            first_visit[s] = G    # overwrites → keeps earliest (lowest t)

        for s, G_s in first_visit.items():
            returns_sum[s]   += G_s
            returns_count[s] += 1
            V[s] = returns_sum[s] / returns_count[s]

        # ── Record checkpoint ─────────────────────────────────────────────
        if ep in checkpoints and ep not in checked:
            checked.add(ep)
            V_est = V[1:6].copy()
            V_snapshots[ep] = V_est
            rms_errors.append(float(np.sqrt(np.mean((V_est - V_true) ** 2))))

    return {
        "V_final":    V[1:6].copy(),
        "V_true":     V_true,
        "checkpoints": [c for c in checkpoints if c in V_snapshots],
        "V_snapshots": V_snapshots,
        "rms_errors":  rms_errors,
    }


# ── Page ───────────────────────────────────────────────────────────────────

_STATE_LABELS = ["A", "B", "C", "D", "E"]


def show():
    st.title("Monte Carlo Prediction")
    st.markdown("**Section 3 — Monte Carlo Methods**")

    # ── Concept ───────────────────────────────────────────────────────────
    st.header("Learning Without a Map")
    st.markdown(
        """
In the Dynamic Programming section, I had a complete model of the environment.
I knew exactly where every action would take me and what reward I'd collect.
**Monte Carlo methods** work without that. I don't know the rules — I just play the game,
all the way to the end, and look at what actually happened.

The key shift is from *planning* to *experiencing*. Instead of asking
"what would happen if I took this action?", I ask "what *did* happen when I was in this state?"

### What Is a Return?

A **reward** is the immediate signal I get after one step. A **return** is the total reward
I collect from a point in time onward — the thing I actually care about maximising.

If I'm at step $t$ and the episode ends at step $T$:
"""
    )
    st.latex(r"G_t = R_{t+1} + \gamma R_{t+2} + \gamma^2 R_{t+3} + \cdots = \sum_{k=0}^{T-t-1} \gamma^k R_{t+k+1}")
    st.markdown(
        r"""
$\gamma$ (gamma) is the **discount factor**. When $\gamma = 1$, future rewards count equally
to immediate ones. When $\gamma < 1$, rewards further in the future count less — I prefer
getting rewards sooner.

### How MC Prediction Works

To estimate the value of a state, I run many full episodes. Every time I visit a state,
I record the return I collected from that point to the end of the episode. After many episodes,
I average those returns across all visits. That average converges to the true value $V^\pi(s)$.

I don't need to know *why* the environment gave me that return. I just need to observe it.
That's what "model-free" means.
"""
    )

    col1, col2 = st.columns(2)
    with col1:
        st.info(
            "**First-visit MC:** For each episode, I only use the return from the "
            "*first* time I visit a state. If I visit the same state twice, the second "
            "visit is ignored for that episode."
        )
    with col2:
        st.success(
            "**Every-visit MC:** Uses the return from *every* visit to a state in an episode. "
            "Both converge to the true value with enough episodes. First-visit has "
            "slightly better theoretical properties; every-visit can be more data-efficient."
        )
    st.warning(
        "**MC requires complete episodes.** I can't update until I reach the end of an episode "
        "and know the final return. This makes MC unsuitable for continuing (non-episodic) tasks "
        "— we'll see how TD methods fix this in a future section."
    )

    with st.expander("Deep Dive — Why MC Doesn't Need a Model"):
        st.markdown(
            r"""
Policy evaluation in DP required knowing $p(s', r \mid s, a)$ — the probability of every
transition. The Bellman equation averaged over all possible next states weighted by their probability.

MC skips this entirely. Instead of computing an **expected** return, I observe an **actual** return
and average over many observations. By the law of large numbers, the average of many samples
converges to the expectation — no model needed.

This is the foundational insight that separates model-free from model-based RL.
The trade-off: DP can be exact with a model, but MC needs more samples to achieve the same accuracy.
High variance is the cost of not having a model.
"""
        )

    st.divider()

    # ── Environment ───────────────────────────────────────────────────────
    st.header("The Environment — 5-State Random Walk")
    st.markdown(
        """
I'm walking along a 1D chain of 5 states (A through E, with C in the centre).
At each step I move one position left or right with equal probability.
The chain has two terminal ends:

- **Walk off the left** → episode ends, reward = **0**
- **Walk off the right** → episode ends, reward = **+1**

There are no intermediate rewards — only the final one at the right exit.
Every episode starts at the centre state **C**.

The true value of each state (with $\\gamma = 1$) is simply the probability of eventually
walking off the right end before the left — which is just $s/6$ for states $A=1$ through $E=5$.
"""
    )

    # Draw the chain
    chain_fig = go.Figure()
    positions = [-1, 0, 1, 2, 3, 4, 5]
    labels = ["L\n(0)", "A", "B", "C", "D", "E", "R\n(+1)"]
    colors = ["#EF553B", "#636EFA", "#636EFA", "#00CC96", "#636EFA", "#636EFA", "#00CC96"]
    chain_fig.add_trace(go.Scatter(
        x=positions, y=[0] * 7,
        mode="markers+text",
        marker=dict(size=38, color=colors),
        text=labels,
        textposition="middle center",
        textfont=dict(size=13, color="white"),
    ))
    chain_fig.add_shape(type="line", x0=-1, x1=5, y0=0, y1=0,
                        line=dict(color="gray", width=2, dash="dot"))
    chain_fig.update_layout(
        title="5-State Random Walk",
        xaxis=dict(visible=False), yaxis=dict(visible=False),
        height=160, margin=dict(l=20, r=20, t=40, b=10),
        template="plotly_white", showlegend=False,
    )
    st.plotly_chart(chain_fig, use_container_width=True)

    st.divider()

    # ── Simulation ────────────────────────────────────────────────────────
    st.header("Interactive Simulation")
    st.markdown(
        "Adjust the parameters and run first-visit MC prediction on the random walk. "
        "**Learning outcome:** the estimated values start noisy and converge toward the true "
        "values (1/6 through 5/6 when γ = 1). Most learning happens in the first few hundred "
        "episodes — try 100 vs 5,000 to see how the RMS error curve flattens."
    )

    col1, col2, col3 = st.columns(3)
    with col1:
        n_episodes = st.slider("Number of episodes", 100, 10_000, 1_000, 100,
                               help="How many full random walks to run. More episodes → estimates average out toward the true values.")
    with col2:
        gamma = st.slider("γ (discount factor)", 0.5, 1.0, 1.0, 0.05,
                          help="γ = 1 means all future rewards count equally — the true values are simply right-exit probabilities. Lower γ down-weights distant rewards.")
    with col3:
        seed = st.number_input("Random seed", value=42, step=1)

    if st.button("Run MC Prediction", type="primary"):
        with st.spinner("Running first-visit MC prediction…"):
            result = run_mc_prediction(n_episodes, gamma, int(seed))

        V_true = result["V_true"]
        V_final = result["V_final"]
        checkpoints = result["checkpoints"]
        V_snapshots = result["V_snapshots"]
        rms_errors = result["rms_errors"]

        st.header("Results")

        # ── Chart 1: Value function snapshots ─────────────────────────────
        # Show 4 representative snapshots + true values
        snap_count = len(checkpoints)
        show_indices = [0, snap_count // 3, 2 * snap_count // 3, snap_count - 1]
        show_indices = sorted(set(show_indices))
        show_eps = [checkpoints[i] for i in show_indices]

        colors_snap = ["#AAAAAA", "#EF553B", "#636EFA", "#00CC96"]

        fig_v = go.Figure()
        fig_v.add_trace(go.Bar(
            x=_STATE_LABELS, y=V_true,
            name="True V(s)", marker_color="black", opacity=0.25,
        ))
        for color, ep in zip(colors_snap, show_eps):
            fig_v.add_trace(go.Bar(
                x=_STATE_LABELS,
                y=V_snapshots[ep].tolist(),
                name=f"Episode {ep:,}",
                marker_color=color, opacity=0.85,
            ))
        fig_v.update_layout(
            title="Estimated Value Function at Episode Checkpoints",
            xaxis_title="State",
            yaxis_title="V(s)",
            barmode="group",
            template="plotly_white",
            height=400,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            yaxis=dict(range=[0, 1.05]),
        )
        st.plotly_chart(fig_v, use_container_width=True)

        # ── Chart 2: RMS error curve ───────────────────────────────────────
        fig_rms = go.Figure()
        fig_rms.add_trace(go.Scatter(
            x=checkpoints,
            y=rms_errors,
            mode="lines+markers",
            name="RMS Error",
            line=dict(color="#636EFA", width=2),
            marker=dict(size=4),
        ))
        fig_rms.update_layout(
            title="RMS Error vs Number of Episodes",
            xaxis_title="Episodes",
            yaxis_title="RMS Error",
            xaxis_type="log",
            template="plotly_white",
            height=350,
        )
        st.plotly_chart(fig_rms, use_container_width=True)

        # ── Metrics ────────────────────────────────────────────────────────
        st.header("Key Takeaways")
        col_a, col_b, col_c = st.columns(3)
        with col_a:
            st.metric("Final RMS error", f"{rms_errors[-1]:.4f}")
        with col_b:
            worst_state = int(np.argmax(np.abs(V_final - V_true)))
            st.metric("Worst-estimated state",
                      _STATE_LABELS[worst_state],
                      f"Δ = {V_final[worst_state] - V_true[worst_state]:+.3f}")
        with col_c:
            center_err = abs(V_final[2] - V_true[2])
            st.metric("Error at centre (C)", f"{center_err:.4f}")

        st.success(
            "The estimates converge toward the true values as episodes increase. "
            "The RMS error drops fast early on and then flattens — most of the learning "
            "happens in the first few hundred episodes."
        )
        st.info(
            f"With γ = {gamma}, the true values "
            + ("are just the right-exit probabilities: 1/6, 2/6, 3/6, 4/6, 5/6."
               if gamma == 1.0
               else "shift lower than the γ=1 case — future rewards are discounted, "
                    "so distant right exits are worth less.")
        )

        # ── Final value table ──────────────────────────────────────────────
        import pandas as pd
        st.dataframe(pd.DataFrame({
            "State":     _STATE_LABELS,
            "True V(s)": [f"{v:.4f}" for v in V_true],
            "MC est.":   [f"{v:.4f}" for v in V_final],
            "Error":     [f"{v-t:+.4f}" for v, t in zip(V_final, V_true)],
        }), use_container_width=True)
