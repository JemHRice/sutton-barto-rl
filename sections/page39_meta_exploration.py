import numpy as np
import plotly.graph_objects as go
import streamlit as st

# ── Gridworld ──────────────────────────────────────────────────────────────────

_ROWS, _COLS = 10, 10
_START = (9, 0)
_GOAL = (0, 9)
_ACTIONS = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # N, S, W, E


def _step(s: tuple, a: int) -> tuple:
    r = max(0, min(_ROWS - 1, s[0] + _ACTIONS[a][0]))
    c = max(0, min(_COLS - 1, s[1] + _ACTIONS[a][1]))
    ns = (r, c)
    if ns == _GOAL:
        return ns, 1.0, True
    return ns, -0.01, False


# ── Simulations ────────────────────────────────────────────────────────────────


@st.cache_data
def run_epsilon_greedy(
    n_episodes: int, alpha: float, gamma: float, epsilon: float, seed: int
) -> tuple:
    rng = np.random.default_rng(seed)
    Q = np.zeros((_ROWS, _COLS, 4))
    visits = np.zeros((_ROWS, _COLS), dtype=np.int32)
    steps_per_ep: list = []

    for _ in range(n_episodes):
        s = _START
        steps = 0
        for _ in range(500):
            r, c = s
            visits[r, c] += 1
            a = (
                int(rng.integers(4))
                if rng.random() < epsilon
                else int(np.argmax(Q[r, c]))
            )
            ns, reward, done = _step(s, a)
            nr, nc = ns
            Q[r, c, a] += alpha * (reward + gamma * np.max(Q[nr, nc]) - Q[r, c, a])
            s = ns
            steps += 1
            if done:
                break
        steps_per_ep.append(steps)

    return steps_per_ep, visits.tolist()


@st.cache_data
def run_count_based(
    n_episodes: int, alpha: float, gamma: float, beta: float, seed: int
) -> tuple:
    rng = np.random.default_rng(seed)
    Q = np.zeros((_ROWS, _COLS, 4))
    N = np.zeros((_ROWS, _COLS), dtype=np.int32)
    visits = np.zeros((_ROWS, _COLS), dtype=np.int32)
    steps_per_ep: list = []

    for _ in range(n_episodes):
        s = _START
        steps = 0
        for _ in range(500):
            r, c = s
            visits[r, c] += 1
            # Add intrinsic bonus to Q for action selection
            q_aug = Q[r, c].copy()
            for a in range(4):
                ns_r = max(0, min(_ROWS - 1, r + _ACTIONS[a][0]))
                ns_c = max(0, min(_COLS - 1, c + _ACTIONS[a][1]))
                q_aug[a] += beta / np.sqrt(N[ns_r, ns_c] + 1)
            a = int(np.argmax(q_aug))

            ns, extrinsic_r, done = _step(s, a)
            nr, nc = ns
            N[nr, nc] += 1
            intrinsic_r = beta / np.sqrt(N[nr, nc])
            total_r = extrinsic_r + intrinsic_r
            Q[r, c, a] += alpha * (total_r + gamma * np.max(Q[nr, nc]) - Q[r, c, a])
            s = ns
            steps += 1
            if done:
                break
        steps_per_ep.append(steps)

    return steps_per_ep, visits.tolist()


# ── Visualisations ─────────────────────────────────────────────────────────────


def _smooth(xs: list, w: int = 15) -> list:
    arr = np.array(xs, dtype=float)
    return np.convolve(arr, np.ones(w) / w, mode="valid").tolist()


def _steps_fig(eps_steps: list, cb_steps: list, w: int = 15) -> go.Figure:
    n = len(eps_steps)
    xs = list(range(w, n + 1))
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=xs,
            y=_smooth(eps_steps, w),
            mode="lines",
            name=f"ε-greedy (ε={0.1})",
            line=dict(color="#636EFA", width=2),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=xs,
            y=_smooth(cb_steps, w),
            mode="lines",
            name="Count-based bonus",
            line=dict(color="#7B2D8B", width=2),
        )
    )
    fig.update_layout(
        title=f"Steps per Episode — Rolling Mean (window {w})",
        xaxis_title="Episode",
        yaxis_title="Steps to goal (lower = better)",
        template="plotly_white",
        height=340,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


def _heatmap_fig(visits: list, title: str) -> go.Figure:
    z = np.log1p(np.array(visits, dtype=float)).tolist()
    fig = go.Figure(
        go.Heatmap(
            z=z,
            colorscale="Purples",
            showscale=False,
            xgap=1,
            ygap=1,
        )
    )
    fig.add_annotation(
        x=_START[1],
        y=_START[0],
        text="S",
        showarrow=False,
        font=dict(size=12, color="#008800", family="Arial Black"),
    )
    fig.add_annotation(
        x=_GOAL[1],
        y=_GOAL[0],
        text="G",
        showarrow=False,
        font=dict(size=12, color="#cc5500", family="Arial Black"),
    )
    fig.update_layout(
        title=title,
        xaxis=dict(visible=False, range=[-0.5, _COLS - 0.5]),
        yaxis=dict(visible=False, range=[_ROWS - 0.5, -0.5], scaleanchor="x"),
        template="plotly_white",
        height=320,
        margin=dict(l=10, r=10, t=40, b=10),
    )
    return fig


# ── Page ───────────────────────────────────────────────────────────────────────


def show():
    st.title("Meta-RL and Principled Exploration")
    st.markdown(
        '<div style="background:#7B2D8B;height:3px;'
        'border-radius:2px;margin-bottom:1rem;"></div>',
        unsafe_allow_html=True,
    )
    st.markdown("**Section 13 — Advanced Topics (Chapter 17)**")

    # ── Exploration concept ────────────────────────────────────────────────────
    st.header("Why Exploration Is Hard")
    st.markdown(r"""
ε-greedy and UCB work well in bandits because every arm is tried repeatedly and the visit
counts converge. In full RL with large or continuous state spaces, the same state may never
be visited twice — **counts become meaningless** without generalisation.

Two directions attack this:

1. **Principled exploration** — measure novelty and drive agents toward it as an explicit objective.
2. **Meta-RL** — learn an exploration *strategy* rather than hard-coding one, so the agent adapts its exploration to the structure of the current task.
""")

    st.header("Count-Based Exploration Bonus")
    st.markdown(r"""
The simplest principled exploration method adds an **intrinsic reward** that is large in
novel states and decays as a state is visited more often:

$$r_\text{total}(s, a) = r_\text{extrinsic}(s, a) + \frac{\beta}{\sqrt{N(s')}}$$

where $N(s')$ is the visit count of the resulting state $s'$ and $\beta > 0$ controls the
exploration strength.  States visited for the first time give bonus $\beta$; states visited
$k$ times give $\beta / \sqrt{k}$ — naturally decreasing incentive to revisit.

The agent remains greedy but its Q-values incorporate both task reward and novelty,
so it is *intrinsically motivated* to explore without a separate ε-greedy mechanism.
""")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Pseudo-count methods**")
        st.markdown(
            "When states are continuous, exact counts are always 1 or 0.  "
            "Pseudo-counts (Bellemare et al. 2016) use a density model $\\hat{\\rho}$ to "
            "estimate how familiar the agent is with a state, giving a *generalised count* "
            "$\\hat{N}(s) = \\hat{\\rho}(s) \\cdot n$ where $n$ is the total step count. "
            "This allows count-based bonuses in pixel-space Atari games."
        )
    with col2:
        st.markdown("**Intrinsic Curiosity Module (ICM)**")
        st.markdown(
            "ICM (Pathak et al. 2017) trains a forward dynamics model to predict "
            "the next state embedding from the current state and action.  "
            "The intrinsic reward is the model's **prediction error** — large for surprising "
            "transitions, small for familiar ones.  Unlike count-based methods, ICM "
            "generalises across visually similar states through the learned embedding."
        )

    with st.expander("Random Network Distillation (RND)"):
        st.markdown(r"""
RND (Burda et al. 2018) uses a fixed, randomly initialised target network $f$ and a
trained predictor network $\hat{f}$.  The intrinsic reward is:

$$r_i(s) = \|\hat{f}(s) - f(s)\|^2$$

The predictor reduces its error on frequently visited states but retains high error on
novel states.  RND is simple, scalable, and works directly from raw observations.
It achieved superhuman performance on *Montezuma's Revenge*, the canonical hard-exploration
Atari game that ε-greedy methods struggle to explore at all.
""")

    st.divider()

    # ── Meta-RL concept ────────────────────────────────────────────────────────
    st.header("Meta-RL — Learning to Learn")
    st.markdown(r"""
Meta-RL addresses a different question: rather than designing a fixed exploration strategy,
can an agent *learn* how to explore efficiently from experience across many tasks?

**RL² (Duan et al. 2016 / Wang et al. 2016)** is the canonical approach: the agent is a
recurrent network (LSTM) trained across a distribution of tasks.  The hidden state accumulates
experience within each episode, so the network's *behaviour* evolves over steps — it learns
to explore early and exploit later, without this being hardcoded.

**MAML (Model-Agnostic Meta-Learning, Finn et al. 2017)** optimises for parameters $\theta$
that can be adapted to a new task with a small number of gradient steps:

$$\theta^* = \arg\max_\theta\; \mathbb{E}_{\mathcal{T}}\!\left[J_{\mathcal{T}}(\theta - \alpha \nabla_\theta \mathcal{L}_{\mathcal{T}}(\theta))\right]$$

One gradient update from $\theta$ should perform well on the task.  Applied to RL, MAML
enables few-shot adaptation to new environments — the agent has learnt a prior over tasks
that makes individual tasks easy to fine-tune.
""")

    st.info(
        "**Key distinction**: principled exploration methods (count-based, ICM, RND) "
        "add a fixed intrinsic signal on top of any base algorithm.  Meta-RL "
        "learns the *entire exploration policy* from experience, allowing it to "
        "adapt to task structure that hand-crafted bonuses cannot anticipate."
    )

    with st.expander("Posterior Sampling / Thompson Sampling for RL"):
        st.markdown(r"""
**PSRL (Posterior Sampling for RL, Strens 2000 / Osband et al. 2013)** maintains a
Bayesian posterior over MDPs.  At the start of each episode, a single MDP is *sampled*
from the posterior and the agent acts optimally under that sample.

This is the RL analogue of Thompson Sampling for bandits and achieves the
$\tilde{O}(\sqrt{T})$ Bayesian regret bound — the *best possible* sample complexity for
exploration in a finite MDP.  Practical scalable versions use deep ensembles or
hypernetworks to approximate the posterior.
""")

    st.divider()

    # ── Simulation ─────────────────────────────────────────────────────────────
    st.header("Interactive Simulation — Exploration Comparison")
    st.markdown(
        "10×10 open gridworld: start **S** (bottom-left), goal **G** (top-right).  "
        "Compare ε-greedy Q-learning against count-based bonus Q-learning.  "
        "The visit heatmaps reveal how each method distributes its exploration."
    )

    col1, col2 = st.columns(2)
    with col1:
        n_episodes = st.slider("Episodes", 100, 500, 300, 50)
        alpha = st.slider("Learning rate α", 0.05, 0.5, 0.2, 0.05)
        gamma = st.slider("Discount γ", 0.90, 1.00, 0.99, 0.01)
    with col2:
        epsilon = st.slider("ε (ε-greedy agent)", 0.05, 0.40, 0.10, 0.05)
        beta = st.slider("β (count-based bonus strength)", 0.1, 3.0, 1.0, 0.1)
        seed = st.number_input("Random seed", value=42, step=1)

    if st.button("Run exploration comparison", type="primary"):
        with st.spinner("Running both agents…"):
            eps_steps, eps_visits = run_epsilon_greedy(
                n_episodes, alpha, float(gamma), epsilon, int(seed)
            )
            cb_steps, cb_visits = run_count_based(
                n_episodes, alpha, float(gamma), beta, int(seed)
            )

        st.plotly_chart(_steps_fig(eps_steps, cb_steps), use_container_width=True)

        col_a, col_b = st.columns(2)
        with col_a:
            st.plotly_chart(
                _heatmap_fig(eps_visits, f"ε-Greedy Visit Heatmap (ε={epsilon})"),
                use_container_width=True,
            )
        with col_b:
            st.plotly_chart(
                _heatmap_fig(cb_visits, f"Count-Based Visit Heatmap (β={beta})"),
                use_container_width=True,
            )
        st.caption("Colour intensity = log(visit count).  Darker = visited more often.")

        w = 15
        eps_end = float(np.mean(_smooth(eps_steps, w)[-20:]))
        cb_end = float(np.mean(_smooth(cb_steps, w)[-20:]))

        # First episode reaching the goal
        eps_first = next((i + 1 for i, s in enumerate(eps_steps) if s < 500), None)
        cb_first = next((i + 1 for i, s in enumerate(cb_steps) if s < 500), None)

        col_a, col_b = st.columns(2)
        with col_a:
            st.metric(
                "ε-Greedy — first goal episode",
                str(eps_first) if eps_first else "not found",
            )
            st.metric("ε-Greedy — final mean steps", f"{eps_end:.0f}")
        with col_b:
            st.metric(
                "Count-Based — first goal episode",
                str(cb_first) if cb_first else "not found",
            )
            st.metric("Count-Based — final mean steps", f"{cb_end:.0f}")

        st.header("Key Takeaways")
        st.success(
            "The visit heatmaps show the core difference: ε-greedy revisits familiar "
            "regions frequently (darker patches near the start), while the count-based "
            "agent's bonus decays in visited states, pushing it toward uncharted territory.  "
            "In sparse-reward problems, this directed exploration dramatically reduces "
            "the time to first find the goal."
        )
        st.info(
            f"**Tuning β**: a large β (>2) can cause the agent to over-explore and "
            "ignore the extrinsic goal signal.  A small β (≈0.5) provides gentle "
            "guidance.  In practice, β is annealed over training as the environment "
            "becomes familiar."
        )
