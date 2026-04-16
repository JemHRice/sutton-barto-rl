import math

import numpy as np
import plotly.graph_objects as go
import streamlit as st
import pandas as pd


# ── Cached simulations ─────────────────────────────────────────────────────

def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


@st.cache_data
def run_epsilon_greedy_bernoulli(
    n_arms: int, n_episodes: int, n_steps: int,
    epsilon: float, p_true: tuple,
) -> np.ndarray:
    rng = np.random.default_rng(0)
    p = np.array(p_true)
    all_rewards = np.zeros((n_episodes, n_steps))
    for ep in range(n_episodes):
        Q = np.zeros(n_arms)
        N = np.zeros(n_arms)
        explore = rng.random(n_steps) < epsilon
        rand_a = rng.integers(0, n_arms, n_steps)
        for t in range(n_steps):
            action = int(rand_a[t]) if explore[t] else int(np.argmax(Q))
            reward = float(rng.random() < p[action])
            N[action] += 1
            Q[action] += (reward - Q[action]) / N[action]
            all_rewards[ep, t] = reward
    return all_rewards.mean(axis=0)


@st.cache_data
def run_ucb_bernoulli(
    n_arms: int, n_episodes: int, n_steps: int,
    c: float, p_true: tuple,
) -> np.ndarray:
    rng = np.random.default_rng(1)
    p = np.array(p_true)
    all_rewards = np.zeros((n_episodes, n_steps))
    for ep in range(n_episodes):
        Q = np.zeros(n_arms)
        N = np.zeros(n_arms)
        for t in range(n_steps):
            if t < n_arms:
                action = t
            else:
                action = int(np.argmax(Q + c * np.sqrt(np.log(t + 1) / (N + 1e-9))))
            reward = float(rng.random() < p[action])
            N[action] += 1
            Q[action] += (reward - Q[action]) / N[action]
            all_rewards[ep, t] = reward
    return all_rewards.mean(axis=0)


@st.cache_data
def run_thompson_sampling(
    n_arms: int, n_episodes: int, n_steps: int,
    p_true: tuple,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Beta-Bernoulli Thompson Sampling.
    Returns (avg_rewards, final_alpha, final_beta) — final params from the last episode.
    """
    rng = np.random.default_rng(2)
    p = np.array(p_true)
    all_rewards = np.zeros((n_episodes, n_steps))
    final_alpha = np.ones(n_arms)
    final_beta  = np.ones(n_arms)

    for ep in range(n_episodes):
        alpha = np.ones(n_arms)
        beta  = np.ones(n_arms)
        for t in range(n_steps):
            action = int(np.argmax(rng.beta(alpha, beta)))
            reward = float(rng.random() < p[action])
            alpha[action] += reward
            beta[action]  += 1.0 - reward
            all_rewards[ep, t] = reward
        if ep == n_episodes - 1:
            final_alpha = alpha.copy()
            final_beta  = beta.copy()

    return all_rewards.mean(axis=0), final_alpha, final_beta


def _beta_pdf(alpha: float, beta: float, theta: np.ndarray) -> np.ndarray:
    """Beta PDF evaluated on theta ∈ (0,1), computed via log-gamma."""
    log_B = math.lgamma(alpha) + math.lgamma(beta) - math.lgamma(alpha + beta)
    log_pdf = (alpha - 1) * np.log(theta) + (beta - 1) * np.log(1 - theta) - log_B
    return np.nan_to_num(np.exp(log_pdf), nan=0.0, posinf=0.0)


# ── Page ───────────────────────────────────────────────────────────────────

def show():
    st.title("Thompson Sampling")
    st.markdown("**Section 1 — Bandit Algorithms**")

    # ── Concept explanation ────────────────────────────────────────────────
    st.header("A Different Kind of Uncertainty")
    st.markdown(
        """
Both ε-greedy and UCB keep a **point estimate** of each arm's value — a single number.
Thompson Sampling asks a different question: instead of *what's my best guess for this arm?*,
it asks *what do I actually believe about this arm?*

The answer isn't a number — it's a **probability distribution** over what the arm's true
success rate might be.

### Prior and Posterior — In Plain English

When I first sit down at the machines, I have no information. My belief about each arm is flat —
anything from 0% to 100% success rate seems equally possible. That's my **prior**.

After I pull an arm and see the result, I update. If I get a success, the distribution shifts
toward higher probabilities. If I fail, it shifts lower. That updated belief is my **posterior**.
The more pulls I have on an arm, the tighter and more confident the posterior becomes.

### The Decision Rule

At each step, I do this:

1. For every arm, I **sample one number** from its current posterior distribution.
2. I pull whichever arm gave the **highest sample**.

That's the whole algorithm. Arms with wide, uncertain posteriors will sometimes sample high and
win — getting explored. Arms with tight, well-understood posteriors consistently represent their
true quality — the best arm usually wins. Uncertainty and optimism emerge automatically from
the shape of the distribution.

### Why Beta Distributions?

For **Bernoulli rewards** (success or failure), the Beta distribution is the natural choice.
Starting with Beta(1, 1) — which is uniform, meaning I have no prior belief — the update
after seeing $s$ successes and $f$ failures is simply:
"""
    )
    st.latex(r"\text{Beta}(1 + s,\; 1 + f)")
    st.markdown(
        """
One line. No solver. The distribution updates itself as a closed-form calculation every pull.

**Note:** In this page, the Gaussian 10-armed testbed is converted to Bernoulli rewards via a
sigmoid mapping so all three algorithms compete on the same set of arms.
"""
    )

    col1, col2 = st.columns(2)
    with col1:
        st.success(
            "**Why this is Bayesian:** I maintain a full probability distribution over the unknown "
            "parameter — the Beta posterior — and update it every time I get new data. "
            "That's exactly what Bayesian inference does."
        )
    with col2:
        st.info(
            "**In practice:** Thompson Sampling often matches or beats UCB without any "
            "tuning. The uncertainty in the posterior naturally drives the right amount of "
            "exploration — it shrinks as I learn, so I don't keep exploring arms I've already understood."
        )

    with st.expander("Deep Dive — Conjugate Priors"):
        st.markdown(
            r"""
A prior $p(\theta)$ is called **conjugate** to a likelihood $p(x|\theta)$ when the posterior
has the same functional form as the prior — meaning the update is a simple parameter tweak,
not a full integration problem.

For Bernoulli observations and a Beta prior:

$$p(\theta | \text{data}) = \text{Beta}(\alpha + s,\; \beta + n - s)$$

where $s$ is successes and $n$ is total pulls. The posterior mean $\frac{\alpha}{\alpha+\beta}$
is our running estimate of the success probability.

As $\alpha + \beta \to \infty$ (many pulls), the Beta concentrates tightly around the true $p$,
and Thompson Sampling converges toward pure exploitation of the best arm.

Conjugate priors exist for many likelihood families: Beta-Binomial, Gamma-Poisson,
Normal-Normal. This is what makes Thompson Sampling tractable in practice.
"""
        )

    st.divider()

    # ── Simulation ────────────────────────────────────────────────────────
    st.header("Interactive Simulation — All 3 Algorithms")
    st.markdown(
        "Run all three exploration strategies on the same Bernoulli bandit. "
        "**Learning outcome:** Thompson Sampling typically converges fastest with no parameter "
        "tuning — its posterior uncertainty shrinks automatically as it gathers data. "
        "After the run, check the Beta posterior chart to see which arms the agent is still "
        "uncertain about, and which it has confidently identified."
    )

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        n_episodes = st.slider("Episodes", 50, 500, 200, 50,
                               help="Independent runs to average over. More episodes → smoother reward curves for all three algorithms.")
    with col2:
        n_steps = st.slider("Steps per episode", 100, 2000, 1000, 100,
                            help="Pulls per run. More steps gives each algorithm more time to identify and exploit the best arm.")
    with col3:
        epsilon = st.slider("ε (ε-greedy)", 0.0, 1.0, 0.1, 0.01,
                            help="Exploration probability for the ε-greedy algorithm. Thompson Sampling has no equivalent parameter — it self-regulates.")
    with col4:
        c = st.slider("c (UCB)", 0.1, 5.0, 2.0, 0.1,
                      help="UCB confidence scaling. Higher c → UCB explores more aggressively; Thompson Sampling doesn't need this parameter.")

    seed = st.number_input("Random seed", value=42, step=1)

    if st.button("Run All Three Algorithms", type="primary"):
        rng = np.random.default_rng(int(seed))
        true_means = rng.standard_normal(10)
        p_true = tuple(_sigmoid(true_means).tolist())
        p_arr = np.array(p_true)

        eg_rewards = run_epsilon_greedy_bernoulli(10, n_episodes, n_steps, epsilon, p_true)
        ucb_rewards = run_ucb_bernoulli(10, n_episodes, n_steps, c, p_true)
        ts_rewards, ts_alpha, ts_beta = run_thompson_sampling(10, n_episodes, n_steps, p_true)

        st.header("Results")

        steps = list(range(1, n_steps + 1))
        fig_cmp = go.Figure()
        fig_cmp.add_trace(go.Scatter(x=steps, y=eg_rewards, mode="lines",
            name=f"ε-Greedy (ε={epsilon})", line=dict(color="#EF553B", width=2)))
        fig_cmp.add_trace(go.Scatter(x=steps, y=ucb_rewards, mode="lines",
            name=f"UCB (c={c})", line=dict(color="#636EFA", width=2)))
        fig_cmp.add_trace(go.Scatter(x=steps, y=ts_rewards, mode="lines",
            name="Thompson Sampling", line=dict(color="#00CC96", width=2)))
        fig_cmp.add_hline(y=float(p_arr.max()), line_dash="dash", line_color="gray",
                          annotation_text="Best arm p")
        fig_cmp.update_layout(
            title="Average Reward Over Time — All 3 Algorithms (Bernoulli bandit)",
            xaxis_title="Step", yaxis_title="Average Reward",
            template="plotly_white", height=430,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        st.plotly_chart(fig_cmp, use_container_width=True)

        # Beta posteriors
        st.subheader("Beta Posteriors — Thompson Sampling After Final Episode")
        st.markdown(
            "Each curve is my current belief about that arm's true success probability. "
            "A narrow, tall peak means I'm confident. A wide, flat curve means I'm still uncertain."
        )

        theta_range = np.linspace(0.001, 0.999, 300)
        sorted_idx = np.argsort(p_arr)
        colors = [f"hsl({int(i * 240 / 9)}, 70%, 50%)" for i in range(10)]

        fig_beta = go.Figure()
        best_arm = int(np.argmax(p_arr))
        for rank, arm_idx in enumerate(sorted_idx):
            a, b = ts_alpha[arm_idx], ts_beta[arm_idx]
            pdf = _beta_pdf(a, b, theta_range)
            is_best = arm_idx == best_arm
            fig_beta.add_trace(go.Scatter(
                x=theta_range, y=pdf, mode="lines",
                name=f"Arm {arm_idx+1} (p*={p_arr[arm_idx]:.2f})",
                line=dict(color="gold" if is_best else colors[rank],
                          width=3 if is_best else 1.5),
                opacity=1.0 if is_best else 0.6,
            ))
        fig_beta.update_layout(
            title="Beta Posteriors for All Arms (gold = best arm)",
            xaxis_title="θ (success probability)", yaxis_title="Density",
            template="plotly_white", height=430,
            legend=dict(font=dict(size=10)),
        )
        st.plotly_chart(fig_beta, use_container_width=True)

        # Arm table
        st.subheader("Arm Parameters")
        st.dataframe(pd.DataFrame({
            "Arm":          [f"Arm {i+1}" for i in range(10)],
            "True p*":      [f"{p:.4f}" for p in p_arr],
            "Final α":      [f"{a:.1f}" for a in ts_alpha],
            "Final β":      [f"{b:.1f}" for b in ts_beta],
            "Est. mean":    [f"{a/(a+b):.4f}" for a, b in zip(ts_alpha, ts_beta)],
        }), use_container_width=True)

        # Comparison metrics
        st.header("Algorithm Comparison")
        s = slice(-100, None)
        col_a, col_b, col_c = st.columns(3)
        with col_a:
            st.metric("ε-Greedy avg (last 100)", f"{eg_rewards[s].mean():.4f}")
        with col_b:
            st.metric("UCB avg (last 100)", f"{ucb_rewards[s].mean():.4f}")
        with col_c:
            st.metric("Thompson Sampling avg (last 100)", f"{ts_rewards[s].mean():.4f}")

        # Summary
        st.divider()
        st.header("When to Use Each Algorithm")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.info(
                "**ε-Greedy**\n\n"
                "Simple and robust. Good as a baseline or when you want predictable, "
                "controllable exploration. Works in non-stationary settings with a decaying ε. "
                "Weakness: wastes exploration on arms already ruled out."
            )
        with col2:
            st.success(
                "**UCB**\n\n"
                "Deterministic and reproducible. Provably near-optimal regret in stationary "
                "environments. Directs every exploratory pull to the most uncertain arm. "
                "Weakness: the count-based bonus assumes rewards don't drift."
            )
        with col3:
            st.warning(
                "**Thompson Sampling**\n\n"
                "Often the best empirical performer with no parameter tuning. Handles delayed "
                "feedback and contextual bandits naturally. Weakness: requires a conjugate prior "
                "model; harder to extend beyond standard reward families."
            )
