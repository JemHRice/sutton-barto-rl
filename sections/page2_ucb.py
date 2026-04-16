import numpy as np
import plotly.graph_objects as go
import streamlit as st


# ── Cached simulations ─────────────────────────────────────────────────────

@st.cache_data
def run_epsilon_greedy(
    n_arms: int, n_episodes: int, n_steps: int,
    epsilon: float, true_means: tuple,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(0)
    means = np.array(true_means)
    all_rewards = np.zeros((n_episodes, n_steps))
    all_counts = np.zeros((n_episodes, n_arms))

    for ep in range(n_episodes):
        Q = np.zeros(n_arms)
        N = np.zeros(n_arms)
        explore = rng.random(n_steps) < epsilon
        rand_a = rng.integers(0, n_arms, n_steps)
        for t in range(n_steps):
            action = int(rand_a[t]) if explore[t] else int(np.argmax(Q))
            reward = rng.normal(means[action], 1.0)
            N[action] += 1
            Q[action] += (reward - Q[action]) / N[action]
            all_rewards[ep, t] = reward
        all_counts[ep] = N

    return all_rewards.mean(axis=0), all_counts.mean(axis=0)


@st.cache_data
def run_ucb(
    n_arms: int, n_episodes: int, n_steps: int,
    c: float, true_means: tuple,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(1)
    means = np.array(true_means)
    all_rewards = np.zeros((n_episodes, n_steps))
    all_counts = np.zeros((n_episodes, n_arms))

    for ep in range(n_episodes):
        Q = np.zeros(n_arms)
        N = np.zeros(n_arms)
        for t in range(n_steps):
            if t < n_arms:
                action = t
            else:
                action = int(np.argmax(Q + c * np.sqrt(np.log(t + 1) / (N + 1e-9))))
            reward = rng.normal(means[action], 1.0)
            N[action] += 1
            Q[action] += (reward - Q[action]) / N[action]
            all_rewards[ep, t] = reward
        all_counts[ep] = N

    return all_rewards.mean(axis=0), all_counts.mean(axis=0)


# ── Page ───────────────────────────────────────────────────────────────────

def show():
    st.title("UCB — Upper Confidence Bound")
    st.markdown("**Section 1 — Bandit Algorithms**")

    # ── Concept explanation ────────────────────────────────────────────────
    st.header("The Problem with Random Exploration")
    st.markdown(
        """
ε-greedy works, but its exploration is wasteful. When it decides to explore, it picks a machine
completely at random — even one it's already tried 200 times and confirmed is terrible. It has
no memory of *which* machines it still knows little about.

**UCB** fixes that. The idea is simple: I should explore machines I'm uncertain about, not
machines I've already ruled out.

### Optimism in the Face of Uncertainty

When I haven't pulled a machine much, I genuinely don't know how good it is. UCB says:
*give every uncertain machine the benefit of the doubt — assume it could be the best.*
As I pull it more, my estimate gets sharper and the optimistic bonus shrinks. Eventually,
if it really is bad, I'll know that, and I'll stop pulling it.

The action I pick each step is the one with the highest **estimated value plus uncertainty bonus**:
"""
    )
    st.latex(
        r"A_t = \arg\max_a \left[ Q_t(a) + c \sqrt{\frac{\ln t}{N_t(a)}} \right]"
    )
    st.markdown(
        r"""
The left term $Q_t(a)$ is what I already think about arm $a$. The right term is the **bonus**:

- $N_t(a)$ is small (arm rarely pulled) → big bonus → I go explore it
- $N_t(a)$ is large (arm well understood) → tiny bonus → I rely on my estimate
- The $\ln t$ in the numerator grows slowly with time — just enough to ensure I revisit
  neglected arms occasionally, but not so fast that I waste pulls

### What Does **c** Control?

$c$ scales the bonus. A larger $c$ means I trust my estimates less and explore more aggressively.
A smaller $c$ means I lock in on promising arms faster. Unlike ε, there's no wasted exploration —
every "exploratory" pull goes to the arm I'm most uncertain about.
"""
    )

    col1, col2 = st.columns(2)
    with col1:
        st.success(
            "**UCB is deterministic.** Given the same history, it always picks the same arm. "
            "There's no randomness in the action selection — only in the reward outcomes."
        )
    with col2:
        st.info(
            "**UCB has provable guarantees.** Its regret (total missed reward vs always picking "
            "the best arm) grows only logarithmically with time — which is the best possible."
        )
    st.warning(
        "**Where UCB struggles:** If the reward distributions change over time, the count-based "
        "bonus $N_t(a)$ loses meaning — UCB was designed for stationary environments."
    )

    with st.expander("Deep Dive — Logarithmic Regret"):
        st.markdown(
            r"""
**Regret** is the total reward I missed by not always pulling the best arm:

$$L_T = \sum_{t=1}^{T} \left[ q^*(a^*) - q^*(A_t) \right]$$

UCB1 achieves $L_T = O(\ln T)$, which is **optimal** — no consistent algorithm can do better.
ε-greedy with a fixed ε suffers *linear* regret because it keeps pulling bad arms at rate ε forever.

The Lai & Robbins lower bound shows that for any algorithm that doesn't systematically waste pulls:

$$\liminf_{T\to\infty} \frac{L_T}{\ln T} \geq \sum_{a:\, \mu_a < \mu^*}
\frac{\mu^* - \mu_a}{\text{KL}(p_a \| p^*)}$$

UCB1 matches this up to a constant factor on the confidence width.
"""
        )

    st.divider()

    # ── Simulation ────────────────────────────────────────────────────────
    st.header("Interactive Simulation")
    st.markdown(
        "Run both algorithms on the same 10-armed testbed and compare directly. "
        "**Learning outcome:** UCB should achieve a higher reward floor and concentrate more "
        "pulls on the best arm — check the action frequency chart to see how differently each "
        "algorithm distributes its exploration."
    )

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        n_episodes = st.slider("Episodes", 50, 500, 200, 50,
                               help="Number of independent runs to average over. More episodes → smoother, more reliable curves.")
    with col2:
        n_steps = st.slider("Steps per episode", 100, 2000, 1000, 100,
                            help="Pulls per run. UCB needs enough steps to build up reliable counts before its bonus kicks in.")
    with col3:
        epsilon = st.slider("ε (ε-greedy)", 0.0, 1.0, 0.1, 0.01,
                            help="Exploration probability for the ε-greedy baseline. Controls how often it picks a random arm.")
    with col4:
        c = st.slider("c (UCB confidence)", 0.1, 5.0, 2.0, 0.1,
                      help="Scales the uncertainty bonus. Higher c → more exploration of uncertain arms; lower c → faster exploitation.")

    seed = st.number_input("Random seed", value=42, step=1)

    if st.button("Run Simulation", type="primary"):
        rng = np.random.default_rng(int(seed))
        true_means = tuple(rng.standard_normal(10).tolist())
        means_arr = np.array(true_means)

        eg_rewards, eg_counts = run_epsilon_greedy(10, n_episodes, n_steps, epsilon, true_means)
        ucb_rewards, ucb_counts = run_ucb(10, n_episodes, n_steps, c, true_means)

        st.header("Results")

        steps = list(range(1, n_steps + 1))
        fig_reward = go.Figure()
        fig_reward.add_trace(go.Scatter(
            x=steps, y=eg_rewards, mode="lines",
            name=f"ε-Greedy (ε={epsilon})", line=dict(color="#EF553B", width=2),
        ))
        fig_reward.add_trace(go.Scatter(
            x=steps, y=ucb_rewards, mode="lines",
            name=f"UCB (c={c})", line=dict(color="#636EFA", width=2),
        ))
        fig_reward.add_hline(
            y=float(means_arr.max()), line_dash="dash", line_color="gray",
            annotation_text="Best arm mean",
        )
        fig_reward.update_layout(
            title="Average Reward Over Time — ε-Greedy vs UCB",
            xaxis_title="Step", yaxis_title="Average Reward",
            template="plotly_white", height=420,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        st.plotly_chart(fig_reward, use_container_width=True)

        sorted_idx = np.argsort(means_arr)
        arm_labels = [f"Arm {i+1}" for i in range(10)]

        fig_counts = go.Figure()
        fig_counts.add_trace(go.Bar(
            x=[arm_labels[i] for i in sorted_idx], y=eg_counts[sorted_idx],
            name=f"ε-Greedy (ε={epsilon})", marker_color="#EF553B", opacity=0.75,
        ))
        fig_counts.add_trace(go.Bar(
            x=[arm_labels[i] for i in sorted_idx], y=ucb_counts[sorted_idx],
            name=f"UCB (c={c})", marker_color="#636EFA", opacity=0.75,
        ))
        fig_counts.update_layout(
            title="Action Selection Frequency — avg pulls per episode (sorted worst → best)",
            xaxis_title="Arm", yaxis_title="Avg pulls",
            barmode="group", template="plotly_white", height=400,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        st.plotly_chart(fig_counts, use_container_width=True)

        st.header("Key Takeaways")
        col_a, col_b, col_c = st.columns(3)
        with col_a:
            st.metric("ε-Greedy avg (last 100)", f"{eg_rewards[-100:].mean():.3f}")
        with col_b:
            st.metric("UCB avg (last 100)", f"{ucb_rewards[-100:].mean():.3f}")
        with col_c:
            delta = ucb_rewards[-100:].mean() - eg_rewards[-100:].mean()
            st.metric("UCB advantage", f"{delta:+.3f}")

        best_arm = int(np.argmax(means_arr))
        st.info(
            f"The best arm is **Arm {best_arm + 1}** (true mean = {means_arr[best_arm]:.3f}). "
            "UCB should concentrate more of its pulls on that arm compared to ε-greedy — "
            "check the action frequency chart above."
        )
