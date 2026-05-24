import numpy as np
import plotly.graph_objects as go
import streamlit as st

# ── Environment: Access-Control Queuing Task (S&B Example 10.2) ────────────────


class AccessControlQueue:
    """
    Access-control queuing task from Sutton & Barto Section 10.4.
    - k servers; each new customer has priority 1, 2, 4, or 8 (equal probability).
    - Agent accepts or rejects each customer:
      accept  → earn customer's priority, customer uses a server.
      reject  → earn 0.
    - Each busy server becomes free with probability p_free per step.
    - State: (n_free_servers, customer_priority).
    - Continuing task: no terminal state.
    """

    N_SERVERS = 10
    P_FREE = 0.06
    PRIORITIES = [1, 2, 4, 8]
    N_PRIORITIES = len(PRIORITIES)
    N_ACTIONS = 2  # 0=reject, 1=accept
    N_STATES = (N_SERVERS + 1) * N_PRIORITIES

    def __init__(self, seed: int | None = None):
        self.rng = np.random.default_rng(seed)
        self._n_free = self.N_SERVERS
        self._priority = int(self.rng.choice(self.PRIORITIES))

    def state_idx(self) -> int:
        p_idx = self.PRIORITIES.index(self._priority)
        return self._n_free * self.N_PRIORITIES + p_idx

    def step(self, action: int) -> tuple[int, float]:
        """Returns (next_state_idx, reward). Continuing: no done flag."""
        reward = 0.0
        if action == 1 and self._n_free > 0:
            reward = float(self._priority)
            self._n_free -= 1
        # Release busy servers
        if self._n_free < self.N_SERVERS:
            busy = self.N_SERVERS - self._n_free
            released = int(self.rng.binomial(busy, self.P_FREE))
            self._n_free = min(self.N_SERVERS, self._n_free + released)
        # Next customer
        self._priority = int(self.rng.choice(self.PRIORITIES))
        return self.state_idx(), reward


# ── Feature vectors ────────────────────────────────────────────────────────────


def _phi(state_idx: int) -> np.ndarray:
    """One-hot state encoding for the access-control task."""
    phi = np.zeros(AccessControlQueue.N_STATES)
    phi[state_idx] = 1.0
    return phi


# ── Training ───────────────────────────────────────────────────────────────────


@st.cache_data
def run_differential_sarsa(
    alpha: float,
    beta: float,
    epsilon: float,
    n_steps: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Differential semi-gradient SARSA for the access-control queuing task.
    Returns (avg_reward_trace, policy_grid [n_free × n_priorities], w).
    """
    env = AccessControlQueue(seed=seed)
    rng = np.random.default_rng(seed)
    n_s = AccessControlQueue.N_STATES
    n_a = AccessControlQueue.N_ACTIONS
    w = np.zeros((n_a, n_s))
    R_bar = 0.0  # running average reward estimate

    def q(s_idx: int, a: int) -> float:
        return float(w[a, s_idx])

    def eps_greedy(s_idx: int) -> int:
        if rng.random() < epsilon:
            return int(rng.integers(n_a))
        vals = np.array([q(s_idx, a) for a in range(n_a)])
        ties = np.flatnonzero(vals == vals.max())
        return int(rng.choice(ties))

    s = env.state_idx()
    a = eps_greedy(s)
    avg_trace = np.zeros(n_steps)

    for t in range(n_steps):
        s2, reward = env.step(a)
        a2 = eps_greedy(s2)
        delta = reward - R_bar + q(s2, a2) - q(s, a)
        R_bar += beta * delta
        w[a, s] += alpha * delta
        avg_trace[t] = R_bar
        s, a = s2, a2

    # Build greedy policy grid
    policy = np.zeros(
        (AccessControlQueue.N_SERVERS + 1, AccessControlQueue.N_PRIORITIES), dtype=int
    )
    for free in range(AccessControlQueue.N_SERVERS + 1):
        for p_idx, pri in enumerate(AccessControlQueue.PRIORITIES):
            s_idx = free * AccessControlQueue.N_PRIORITIES + p_idx
            vals = np.array([q(s_idx, a) for a in range(n_a)])
            policy[free, p_idx] = int(np.argmax(vals))

    return avg_trace, policy, w


@st.cache_data
def run_discounted_sarsa(
    alpha: float,
    gamma: float,
    epsilon: float,
    n_steps: int,
    seed: int,
    episode_len: int = 500,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Regular (discounted) semi-gradient SARSA on the same task, cut into
    artificial episodes of fixed length so it can be applied.
    Returns (cumulative_reward_trace, policy_grid).
    """
    env = AccessControlQueue(seed=seed)
    rng = np.random.default_rng(seed)
    n_s = AccessControlQueue.N_STATES
    n_a = AccessControlQueue.N_ACTIONS
    w = np.zeros((n_a, n_s))
    cum_r = 0.0
    cum_trace = np.zeros(n_steps)

    def q(s_idx: int, a: int) -> float:
        return float(w[a, s_idx])

    def eps_greedy(s_idx: int) -> int:
        if rng.random() < epsilon:
            return int(rng.integers(n_a))
        vals = np.array([q(s_idx, a) for a in range(n_a)])
        ties = np.flatnonzero(vals == vals.max())
        return int(rng.choice(ties))

    s = env.state_idx()
    a = eps_greedy(s)

    for t in range(n_steps):
        s2, reward = env.step(a)
        a2 = eps_greedy(s2)
        cum_r += reward
        delta = reward + gamma * q(s2, a2) - q(s, a)
        w[a, s] += alpha * delta
        cum_trace[t] = cum_r / (t + 1)
        s, a = s2, a2

    policy = np.zeros(
        (AccessControlQueue.N_SERVERS + 1, AccessControlQueue.N_PRIORITIES), dtype=int
    )
    for free in range(AccessControlQueue.N_SERVERS + 1):
        for p_idx, pri in enumerate(AccessControlQueue.PRIORITIES):
            s_idx = free * AccessControlQueue.N_PRIORITIES + p_idx
            vals = np.array([q(s_idx, a) for a in range(n_a)])
            policy[free, p_idx] = int(np.argmax(vals))

    return cum_trace, policy


# ── Figures ────────────────────────────────────────────────────────────────────


def _avg_reward_fig(
    trace_diff: np.ndarray, trace_disc: np.ndarray, smooth: int
) -> go.Figure:
    steps = list(range(1, len(trace_diff) + 1))
    fig = go.Figure()

    def add_smoothed(y, name, colour):
        fig.add_trace(
            go.Scatter(
                x=steps,
                y=y.tolist(),
                mode="lines",
                name=name,
                line=dict(color=colour, width=1),
                opacity=0.2,
                showlegend=False,
            )
        )
        if smooth > 1 and len(y) >= smooth:
            kernel = np.ones(smooth) / smooth
            sm = np.convolve(y, kernel, mode="valid")
            x_sm = list(range(smooth, len(y) + 1))
            fig.add_trace(
                go.Scatter(
                    x=x_sm,
                    y=sm.tolist(),
                    mode="lines",
                    name=name,
                    line=dict(color=colour, width=2),
                )
            )

    add_smoothed(trace_diff, "Differential SARSA (R̄)", "#636EFA")
    add_smoothed(trace_disc, "Discounted SARSA (avg cum. reward)", "#EF553B")

    fig.update_layout(
        title="Estimated Average Reward over Steps",
        xaxis_title="Step",
        yaxis_title="Reward estimate",
        template="plotly_white",
        height=350,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


def _policy_heatmap(policy: np.ndarray, title: str) -> go.Figure:
    fig = go.Figure(
        go.Heatmap(
            z=policy,
            x=["p=1", "p=2", "p=4", "p=8"],
            y=[str(f) for f in range(AccessControlQueue.N_SERVERS + 1)],
            colorscale=[[0, "#EF553B"], [1, "#636EFA"]],
            zmin=0,
            zmax=1,
            colorbar=dict(
                title="Action", tickvals=[0, 1], ticktext=["Reject", "Accept"]
            ),
        )
    )
    fig.update_layout(
        title=title,
        xaxis_title="Customer priority",
        yaxis_title="Free servers",
        template="plotly_white",
        height=380,
    )
    return fig


# ── Page ───────────────────────────────────────────────────────────────────────


def show():
    st.title("Average Reward vs Discounted Return")
    st.markdown(
        '<div style="background:#7B2D8B;height:3px;'
        'border-radius:2px;margin-bottom:1rem;"></div>',
        unsafe_allow_html=True,
    )
    st.markdown("**Section 8 — Function Approximation: Control (Chapter 10)**")

    # ── Concept ───────────────────────────────────────────────────────────────
    st.header("Continuing Tasks and Average Reward")
    st.markdown(r"""
All previous algorithms used **discounted return** $G_t = \sum_{k=0}^\infty \gamma^k R_{t+k+1}$.
Discounting works for episodic tasks (finite horizon) and can work for continuing tasks —
but it introduces a bias: distant rewards matter less, so the optimal policy can change
with $\gamma$.

For **continuing tasks** (no natural episode boundary), the more principled objective is
the **average reward**:

$$r(\pi) = \lim_{T\to\infty} \frac{1}{T}\sum_{t=0}^T R_{t+1}$$

We want to maximise the long-run reward per step.  The differential return replaces
absolute reward with *excess over average*:

$$\delta_t = R_{t+1} - \bar{R} + \hat{q}(S_{t+1}, A_{t+1}, \mathbf{w}) - \hat{q}(S_t, A_t, \mathbf{w})$$

where $\bar{R}$ is a running estimate of $r(\pi)$, updated with step size $\beta$:

$$\bar{R} \leftarrow \bar{R} + \beta \delta_t$$
""")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Discounted SARSA**")
        st.latex(
            r"\delta_t = R_{t+1} + \gamma\hat{q}(S_{t+1},A_{t+1}) - \hat{q}(S_t,A_t)"
        )
        st.caption(
            "Discounting with $\\gamma < 1$ implicitly assumes episodic structure.  "
            "On a continuing task, the discount horizon $1/(1-\\gamma)$ substitutes for "
            "episode length, but optimal policies can differ from the true average-reward optimum."
        )
    with col2:
        st.markdown("**Differential semi-gradient SARSA**")
        st.latex(
            r"\delta_t = R_{t+1} - \bar{R} + \hat{q}(S_{t+1},A_{t+1}) - \hat{q}(S_t,A_t)"
        )
        st.caption(
            "No discount — all future rewards count equally.  "
            "$\\bar{R}$ tracks the current policy's average reward and is subtracted to "
            "centre the TD error.  Converges to the true average-reward optimum for linear FAs."
        )

    st.info(
        "**When does it matter?**  For $\\gamma \\to 1$, discounted and average-reward objectives "
        "agree in the limit — but convergence is slower and weight magnitudes blow up.  "
        "For $\\gamma$ significantly less than 1, the agent becomes myopic and may miss "
        "high-value but delayed rewards.  Differential SARSA avoids this entirely."
    )

    with st.expander("The access-control queuing task (S&B Example 10.2)"):
        st.markdown(r"""
**Setup**: 10 servers; customers arrive one at a time with priority uniformly chosen
from $\{1, 2, 4, 8\}$.  The agent decides to **accept** (earn the priority and occupy
a server) or **reject** (earn 0).  Each busy server becomes free with probability 0.06
per step independently.

**Optimal policy**: accept high-priority customers more readily; reject low-priority
customers when few servers are free.  The exact threshold depends on the current number
of free servers.

This is a classic continuing task — there is no natural episode end — making it
ideal for demonstrating average-reward methods.
""")

    st.divider()

    # ── Simulation ─────────────────────────────────────────────────────────────
    st.header("Interactive Simulation")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Shared settings**")
        epsilon = st.slider("ε (exploration)", 0.0, 0.3, 0.1, 0.01)
        n_steps = st.slider(
            "Training steps", 10_000, 500_000, 100_000, 10_000, format="%d"
        )
        seed = st.number_input("Random seed", value=42, step=1)
        smooth = st.slider("Rolling mean window", 10, 5000, 500)
    with col2:
        st.markdown("**Differential SARSA**")
        alpha_d = st.slider("α (weight update)", 0.001, 0.5, 0.01, 0.001, format="%.3f")
        beta = st.slider(
            "β (avg reward update)", 0.001, 0.1, 0.01, 0.001, format="%.3f"
        )
        st.markdown("**Discounted SARSA**")
        alpha_g = st.slider("α (discounted)", 0.001, 0.5, 0.01, 0.001, format="%.3f")
        gamma = st.slider("γ (discount)", 0.9, 0.999, 0.99, 0.001, format="%.3f")

    if st.button("Run Comparison", type="primary"):
        with st.spinner("Running differential and discounted SARSA…"):
            avg_trace, policy_diff, _ = run_differential_sarsa(
                alpha_d, beta, epsilon, n_steps, int(seed)
            )
            cum_trace, policy_disc = run_discounted_sarsa(
                alpha_g, gamma, epsilon, n_steps, int(seed)
            )

        st.header("Results")
        st.plotly_chart(
            _avg_reward_fig(avg_trace, cum_trace, smooth), use_container_width=True
        )
        st.caption(
            "Blue: differential SARSA's running $\\bar{R}$ (average reward estimate per step).  "
            "Red: discounted SARSA's cumulative average reward per step.  "
            "Both should stabilise — differential converges to the true $r(\\pi^*)$."
        )

        col_a, col_b = st.columns(2)
        with col_a:
            st.metric("Differential SARSA — final R̄", f"{avg_trace[-1]:.4f}")
        with col_b:
            st.metric("Discounted SARSA — final avg reward", f"{cum_trace[-1]:.4f}")

        col_p1, col_p2 = st.columns(2)
        with col_p1:
            st.plotly_chart(
                _policy_heatmap(policy_diff, "Greedy Policy — Differential SARSA"),
                use_container_width=True,
            )
        with col_p2:
            st.plotly_chart(
                _policy_heatmap(policy_disc, "Greedy Policy — Discounted SARSA"),
                use_container_width=True,
            )

        st.header("Key Takeaways")
        st.success(
            "Both policies should show a **threshold structure**: accept all customers "
            "when many servers are free, become selective when servers are scarce.  "
            "Higher-priority customers should be accepted even with few free servers."
        )
        st.info(
            "**Compare the policies**: differential SARSA optimises average reward directly "
            "and typically rejects low-priority customers more aggressively when servers are "
            "scarce.  Discounted SARSA may be more conservative since $\\gamma < 1$ "
            "effectively limits the horizon."
        )

        with st.expander("Why do both policies look similar?"):
            st.markdown(r"""
For $\gamma$ close to 1, discounted and average-reward objectives are nearly equivalent —
the discount horizon $1/(1-\gamma)$ is large enough to capture most of the long-run structure.

The main practical difference emerges when:
- $\gamma$ is significantly less than 1 (short horizon → myopic policy)
- The task has very long optimal sequences
- You need to compare policies across tasks with different time scales

For the access-control task with 10 servers and $p\_free = 0.06$, the average busy time
per server is $1/0.06 \approx 17$ steps — within the discount horizon for $\gamma = 0.99$
($1/(1-0.99) = 100$ steps), so both methods find similar policies.
""")
