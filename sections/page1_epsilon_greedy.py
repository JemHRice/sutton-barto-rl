import numpy as np
import plotly.graph_objects as go
import streamlit as st


# ── Cached simulation ──────────────────────────────────────────────────────

@st.cache_data
def run_epsilon_greedy(
    n_arms: int,
    n_episodes: int,
    n_steps: int,
    epsilon: float,
    true_means: tuple,          # tuple so st.cache_data can hash it
) -> tuple[np.ndarray, np.ndarray]:
    """
    Vectorised ε-greedy over n_episodes independent runs.
    Returns (avg_rewards [n_steps], final_Q [n_arms]).
    """
    rng = np.random.default_rng(0)
    means = np.array(true_means)
    all_rewards = np.zeros((n_episodes, n_steps))

    for ep in range(n_episodes):
        Q = np.zeros(n_arms)
        N = np.zeros(n_arms)
        explore = rng.random(n_steps) < epsilon
        random_actions = rng.integers(0, n_arms, n_steps)

        for t in range(n_steps):
            action = int(random_actions[t]) if explore[t] else int(np.argmax(Q))
            reward = rng.normal(means[action], 1.0)
            N[action] += 1
            Q[action] += (reward - Q[action]) / N[action]
            all_rewards[ep, t] = reward

    return all_rewards.mean(axis=0), Q


# ── Page ───────────────────────────────────────────────────────────────────

def show():
    st.title("ε-Greedy Bandit")
    st.markdown("**Section 1 — Bandit Algorithms**")

    # ── Concept explanation ────────────────────────────────────────────────
    st.header("The Problem")
    st.markdown(
        """
I have 10 slot machines in front of me. Each one pays out rewards drawn from some distribution
I don't know. My goal is to maximise my total reward over many pulls.

Here's the tension: the only way to find the best machine is to try them. But every pull I spend
*trying* is a pull I'm not spending on the machine I already think is best. Do I keep pulling the
one that's worked so far, or do I go looking for something better?

That's the **exploration-exploitation tradeoff** — and it's the central problem in bandit learning.

### How ε-Greedy Handles It

ε-greedy's solution is blunt but effective. Before every pull, I flip a biased coin:

- With probability **ε**, I ignore everything I know and pull a **random** machine.
- With probability **1 − ε**, I pull whichever machine has the **highest estimated value** so far.

That's it. The exploration is dumb — random — but having *any* exploration stops me from getting
permanently stuck on a mediocre machine I happened to try first.

After each pull, I update my estimate of that machine's value using a running average:
"""
    )
    st.latex(r"Q_{n+1}(a) = Q_n(a) + \frac{1}{N_n(a)} \bigl[ R_n - Q_n(a) \bigr]")
    st.markdown(
        r"""
Read that update as: *my new estimate = my old estimate, nudged toward the reward I just got.*
The step size $\frac{1}{N_n(a)}$ shrinks over time, so early observations matter less as I
accumulate more pulls on that arm.
"""
    )

    col1, col2 = st.columns(2)
    with col1:
        st.info(
            "**ε = 0 (pure greedy):** I never explore. I pull whatever looked best early on "
            "and never revisit the others — even if one of them is far better."
        )
    with col2:
        st.warning(
            "**ε = 1 (pure random):** I always explore. I'm collecting information I never "
            "use — reward stays near the average of all machines regardless of which is best."
        )
    st.success(
        "**The sweet spot** is a small ε (e.g. 0.1): I mostly exploit what I know, but "
        "keep poking at the others often enough to correct my mistakes."
    )

    with st.expander("Deep Dive — Where Does the Update Rule Come From?"):
        st.markdown(
            r"""
The update rule is just an incremental sample mean in disguise. Expanding the definition of a
running average over $n+1$ observations:

$$Q_{n+1} = \frac{1}{n+1}\sum_{i=1}^{n+1} R_i = Q_n + \frac{1}{n+1}(R_{n+1} - Q_n)$$

The term in brackets — $R_{n+1} - Q_n$ — is the **prediction error**: how surprised I was by
the reward. I take a step toward that surprise, with a step size that shrinks as I see more data.

In non-stationary problems (where the true means drift over time), replacing $\frac{1}{N}$ with
a fixed constant $\alpha$ keeps the estimate responsive to recent changes instead of averaging
over an ever-growing history. The general form:

$$\text{new estimate} \leftarrow \text{old estimate} + \alpha \bigl[\text{target} - \text{old estimate}\bigr]$$

reappears in almost every RL algorithm.
"""
        )

    st.divider()

    # ── Simulation ────────────────────────────────────────────────────────
    st.header("Interactive Simulation")
    st.markdown(
        "Adjust the parameters below and click **Run Simulation** to see how ε shapes the "
        "learning curve. **Learning outcome:** watch whether pure greedy (ε = 0) gets locked "
        "onto a suboptimal arm early — and how a small ε (try 0.1) trades a short initial dip "
        "for noticeably better long-run average reward."
    )

    col1, col2, col3 = st.columns(3)
    with col1:
        n_episodes = st.slider("Episodes", 50, 500, 200, 50,
                               help="Number of independent runs to average over. More episodes → smoother reward curve, less noise.")
    with col2:
        n_steps = st.slider("Steps per episode", 100, 2000, 1000, 100,
                            help="How many arm pulls per run. More steps gives the agent more time to learn and exploit.")
    with col3:
        epsilon = st.slider("ε (epsilon)", 0.0, 1.0, 0.1, 0.01,
                            help="Probability of picking a random arm each step. ε = 0 is pure greedy; ε = 1 is pure random.")

    seed = st.number_input("Random seed", value=42, step=1)

    if st.button("Run Simulation", type="primary"):
        rng = np.random.default_rng(int(seed))
        true_means = tuple(rng.standard_normal(10).tolist())

        avg_rewards, final_Q = run_epsilon_greedy(10, n_episodes, n_steps, epsilon, true_means)
        means_arr = np.array(true_means)

        # ── Results ───────────────────────────────────────────────────────
        st.header("Results")

        fig_reward = go.Figure()
        fig_reward.add_trace(go.Scatter(
            x=list(range(1, n_steps + 1)), y=avg_rewards,
            mode="lines", name=f"ε = {epsilon}",
            line=dict(color="#636EFA", width=2),
        ))
        fig_reward.add_hline(
            y=float(means_arr.max()), line_dash="dash", line_color="gray",
            annotation_text="Best arm mean",
        )
        fig_reward.update_layout(
            title="Average Reward Over Time",
            xaxis_title="Step", yaxis_title="Average Reward",
            template="plotly_white", height=400,
        )
        st.plotly_chart(fig_reward, use_container_width=True)

        sorted_idx = np.argsort(means_arr)
        arm_labels = [f"Arm {i+1}" for i in range(10)]

        fig_q = go.Figure()
        fig_q.add_trace(go.Bar(
            x=[arm_labels[i] for i in sorted_idx], y=means_arr[sorted_idx],
            name="True mean q*", marker_color="#EF553B", opacity=0.7,
        ))
        fig_q.add_trace(go.Bar(
            x=[arm_labels[i] for i in sorted_idx], y=final_Q[sorted_idx],
            name="Estimated Q (last episode)", marker_color="#636EFA", opacity=0.7,
        ))
        fig_q.update_layout(
            title="Estimated Q-Values vs True Means (sorted worst → best)",
            xaxis_title="Arm", yaxis_title="Value",
            barmode="group", template="plotly_white", height=400,
        )
        st.plotly_chart(fig_q, use_container_width=True)

        best_arm = int(np.argmax(means_arr))
        st.header("Key Takeaways")
        col_a, col_b, col_c = st.columns(3)
        with col_a:
            st.metric("Best arm", f"Arm {best_arm + 1}",
                      f"q* = {means_arr[best_arm]:.3f}")
        with col_b:
            st.metric("Final avg reward (last 100 steps)", f"{avg_rewards[-100:].mean():.3f}")
        with col_c:
            st.metric("ε used", f"{epsilon}")

        st.info(
            f"With ε = {epsilon}, roughly **{epsilon*100:.0f}%** of pulls are random exploration "
            f"and **{(1-epsilon)*100:.0f}%** exploit the current best estimate. "
            + ("Try ε = 0 to watch pure greedy get stuck early."
               if epsilon > 0
               else "Try ε = 0.1 to see exploration help the agent find better arms.")
        )
