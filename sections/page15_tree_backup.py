import numpy as np
import plotly.graph_objects as go
import streamlit as st

# ── Environment: 4×4 GridWorld (same as Page 14) ──────────────────────────

GRID_H, GRID_W = 4, 4
N_STATES = GRID_H * GRID_W
N_ACTIONS = 4
GOAL = N_STATES - 1
_DELTAS = [(-1, 0), (0, 1), (1, 0), (0, -1)]


def _step(state: int, action: int) -> tuple[int, float, bool]:
    if state == GOAL:
        return state, 0.0, True
    r, c = divmod(state, GRID_W)
    dr, dc = _DELTAS[action]
    nr, nc = r + dr, c + dc
    if not (0 <= nr < GRID_H and 0 <= nc < GRID_W):
        nr, nc = r, c
    next_state = nr * GRID_W + nc
    return next_state, -1.0, next_state == GOAL


def _epsilon_greedy(
    Q: np.ndarray, state: int, epsilon: float, rng: np.random.Generator
) -> int:
    return (
        int(rng.integers(0, N_ACTIONS))
        if rng.random() < epsilon
        else int(np.argmax(Q[state]))
    )


def _target_policy(Q: np.ndarray, state: int) -> int:
    """Greedy target policy."""
    return int(np.argmax(Q[state]))


# ── Off-policy n-step SARSA with importance sampling ──────────────────────


@st.cache_data
def run_offpolicy_nstep_sarsa(
    n: int,
    alpha: float,
    epsilon_behaviour: float,
    n_episodes: int,
) -> tuple[np.ndarray, list[float]]:
    """
    Off-policy n-step SARSA using importance sampling ratios.
    Behaviour policy: ε-greedy. Target policy: greedy.
    Returns (episode_returns, is_ratios_per_episode).
    """
    rng = np.random.default_rng(42)
    Q = np.zeros((N_STATES, N_ACTIONS))
    episode_returns = np.zeros(n_episodes)
    max_is_ratios = []

    for ep in range(n_episodes):
        state = 0
        action = _epsilon_greedy(Q, state, epsilon_behaviour, rng)
        states = [state]
        actions = [action]
        rewards = [0.0]
        T = float("inf")
        t = 0
        total_reward = 0.0
        ep_max_is = 0.0

        while True:
            if t < T:
                next_state, reward, done = _step(states[t], actions[t])
                rewards.append(reward)
                states.append(next_state)
                total_reward += reward
                if done:
                    T = t + 1
                else:
                    next_action = _epsilon_greedy(Q, next_state, epsilon_behaviour, rng)
                    actions.append(next_action)

            tau = t - n + 1
            if tau >= 0:
                # Importance sampling ratio: product of π(a|s)/b(a|s) for steps tau+1..min(tau+n,T-1)
                rho = 1.0
                for k in range(tau + 1, int(min(tau + n, T - 1)) + 1):
                    s_k, a_k = states[k], actions[k]
                    # target policy: greedy → prob 1 if best, 0 otherwise
                    target_prob = 1.0 if a_k == _target_policy(Q, s_k) else 0.0
                    # behaviour policy: ε-greedy
                    behaviour_prob = epsilon_behaviour / N_ACTIONS + (
                        (1 - epsilon_behaviour)
                        if a_k == int(np.argmax(Q[s_k]))
                        else 0.0
                    )
                    if behaviour_prob == 0:
                        rho = 0.0
                        break
                    rho *= target_prob / behaviour_prob

                ep_max_is = max(ep_max_is, rho)

                end = int(min(tau + n, T))
                G = sum(rewards[i] for i in range(tau + 1, end + 1))
                if tau + n < T:
                    G += Q[states[tau + n], actions[tau + n]]

                s_tau, a_tau = states[tau], actions[tau]
                Q[s_tau, a_tau] += alpha * rho * (G - Q[s_tau, a_tau])

            t += 1
            if tau == T - 1:
                break

        episode_returns[ep] = total_reward
        max_is_ratios.append(ep_max_is)

    return episode_returns, max_is_ratios


# ── n-step Tree Backup ─────────────────────────────────────────────────────


@st.cache_data
def run_tree_backup(
    n: int,
    alpha: float,
    epsilon: float,
    n_episodes: int,
) -> np.ndarray:
    """
    n-step Tree Backup: off-policy without importance sampling.
    Uses expected values over the target (greedy) policy at each step.
    """
    rng = np.random.default_rng(42)
    Q = np.zeros((N_STATES, N_ACTIONS))
    episode_returns = np.zeros(n_episodes)

    for ep in range(n_episodes):
        state = 0
        action = _epsilon_greedy(Q, state, epsilon, rng)
        states = [state]
        actions = [action]
        rewards = [0.0]
        T = float("inf")
        t = 0
        total_reward = 0.0

        while True:
            if t < T:
                next_state, reward, done = _step(states[t], actions[t])
                rewards.append(reward)
                states.append(next_state)
                total_reward += reward
                if done:
                    T = t + 1
                else:
                    next_action = _epsilon_greedy(Q, next_state, epsilon, rng)
                    actions.append(next_action)

            tau = t - n + 1
            if tau >= 0:
                if t + 1 >= T:
                    G = rewards[int(T)]
                else:
                    # Expected value over greedy policy at t+1
                    s_next = states[t + 1]
                    G = rewards[t + 1] + np.max(Q[s_next])

                # Back up through the tree
                for k in range(int(min(t, T - 1)), tau, -1):
                    s_k = states[k]
                    a_k = actions[k]
                    # Expected Q under greedy (target) policy
                    best_a = int(np.argmax(Q[s_k]))
                    expected_Q = Q[s_k, best_a]
                    # π(a_k|s_k) for greedy: 1 if best action, else 0
                    pi_a = 1.0 if a_k == best_a else 0.0
                    G = rewards[k] + pi_a * G + (expected_Q - pi_a * Q[s_k, a_k])

                s_tau, a_tau = states[tau], actions[tau]
                Q[s_tau, a_tau] += alpha * (G - Q[s_tau, a_tau])

            t += 1
            if tau == T - 1:
                break

        episode_returns[ep] = total_reward

    return episode_returns


# ── Page ───────────────────────────────────────────────────────────────────


def show():
    st.title("Tree Backup vs Importance Sampling")
    st.markdown("**Section 5 — n-step Bootstrapping · Chapter 7**")

    # ── Concept ───────────────────────────────────────────────────────────
    st.header("The Off-Policy Problem")
    st.markdown("""
Pages 13 and 14 used **on-policy** algorithms — the policy being learned is the same policy
used to collect data. Off-policy learning separates these two roles:

- **Behaviour policy** b(a|s): what the agent actually does (e.g. ε-greedy, exploratory)
- **Target policy** π(a|s): what we want to optimise (e.g. greedy, the final policy)

Off-policy learning is powerful — you can learn from data collected by a different agent,
a human demonstrator, or a historical dataset. But n-step off-policy updates need a
correction factor.
""")

    st.subheader("Method 1: Importance Sampling")
    st.latex(
        r"\rho_{t:t+n-1} = \prod_{k=t}^{\min(t+n-1,\,T-1)}"
        r"\frac{\pi(A_k \mid S_k)}{b(A_k \mid S_k)}"
    )
    st.markdown(
        "The importance sampling ratio ρ reweights the n-step return by how much more or less "
        "likely the target policy would have taken those same actions. The update then becomes:"
    )
    st.latex(
        r"Q(S_t, A_t) \leftarrow Q(S_t, A_t)"
        r"+ \alpha\,\rho_{t+1:t+n}\bigl[G_t^{(n)} - Q(S_t, A_t)\bigr]"
    )

    st.warning(
        "**The variance problem:** when the target and behaviour policies differ significantly, "
        "the IS ratio is a product of many terms — each potentially large. With n = 8 or 16, "
        "this product can explode or collapse to near zero, making learning unstable. "
        "This is the fundamental weakness of IS-based off-policy methods."
    )

    st.subheader("Method 2: n-step Tree Backup")
    st.markdown("""
Tree Backup avoids importance sampling entirely. Instead of following the behaviour policy's
actual actions, it **backs up expected values** over the target policy at every step except
the one actually taken.

At each intermediate step k, the backup branches over all actions the target policy might take,
weighted by their target policy probability — like a tree of possible continuations. Only the
actually-taken action continues the chain forward.
""")

    st.success(
        "**Why Tree Backup avoids IS:** because we never compare the behaviour and target policy "
        "probabilities directly. We just use the target policy's expected values as the "
        "bootstrapped estimate. No ratios → no variance explosion."
    )

    with st.expander("Deep Dive — Tree Backup Update in Detail"):
        st.markdown(r"""
For a greedy target policy (probability 1 on the best action, 0 on all others), the
tree backup return simplifies considerably. At each intermediate step $k$:

$$G_{k:k+n}^{\text{TB}} = R_{k+1} + \pi(A_{k+1}|S_{k+1}) \cdot G_{k+1:k+n}^{\text{TB}}
+ \sum_{a \neq A_{k+1}} \pi(a|S_{k+1}) Q(S_{k+1}, a)$$

With a greedy target policy, $\pi(A_{k+1}|S_{k+1})$ is 1 if $A_{k+1}$ is the greedy action,
0 otherwise. The sum over other actions uses the expected Q under the greedy policy. This
is the Expected SARSA update applied recursively — stable and IS-free.
""")

    st.divider()

    # ── Controls ──────────────────────────────────────────────────────────
    st.header("Interactive Comparison")

    col1, col2 = st.columns(2)
    with col1:
        n = st.slider("n (steps)", 1, 16, 4, 1)
        alpha = st.slider("α (step size)", 0.01, 0.5, 0.1, 0.01)
        n_episodes = st.slider("Episodes", 100, 2000, 500, 100)
    with col2:
        epsilon = st.slider(
            "ε (behaviour policy)",
            0.05,
            0.5,
            0.2,
            0.05,
            help="Exploration rate for the behaviour policy.",
        )
        smooth_window = st.slider("Smoothing window", 1, 50, 20)

    if st.button("Run Comparison", type="primary"):
        with st.spinner("Training both algorithms..."):
            is_returns, is_ratios = run_offpolicy_nstep_sarsa(
                n, alpha, epsilon, n_episodes
            )
            tb_returns = run_tree_backup(n, alpha, epsilon, n_episodes)

        st.header("Results")

        def _smooth(arr, w):
            if w <= 1:
                return arr
            return np.convolve(arr, np.ones(w) / w, mode="same")

        x = list(range(1, n_episodes + 1))

        # ── Learning curves ────────────────────────────────────────────
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=x,
                y=_smooth(is_returns, smooth_window),
                mode="lines",
                name="Off-policy IS",
                line=dict(color="#636EFA", width=2),
            )
        )
        fig.add_trace(
            go.Scatter(
                x=x,
                y=_smooth(tb_returns, smooth_window),
                mode="lines",
                name="Tree Backup",
                line=dict(color="#EF553B", width=2),
            )
        )
        fig.update_layout(
            title=f"Episode Returns (n={n}, smoothed over {smooth_window} episodes)",
            xaxis_title="Episode",
            yaxis_title="Total Return",
            template="plotly_white",
            height=380,
            legend=dict(
                orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1
            ),
        )
        st.plotly_chart(fig, use_container_width=True)

        # ── IS ratio variance ──────────────────────────────────────────
        fig2 = go.Figure()
        fig2.add_trace(
            go.Scatter(
                x=x,
                y=is_ratios,
                mode="lines",
                name="Max IS ratio per episode",
                line=dict(color="#AB63FA", width=1.5),
            )
        )
        fig2.add_hline(
            y=1.0,
            line_dash="dash",
            line_color="gray",
            annotation_text="ratio = 1 (no correction needed)",
        )
        fig2.update_layout(
            title="Importance Sampling Ratio (max per episode)",
            xaxis_title="Episode",
            yaxis_title="Max ρ",
            template="plotly_white",
            height=300,
        )
        st.plotly_chart(fig2, use_container_width=True)

        # ── Summary ────────────────────────────────────────────────────
        st.header("Key Takeaways")
        col_a, col_b, col_c = st.columns(3)
        with col_a:
            st.metric("IS final return (last 100)", f"{is_returns[-100:].mean():.1f}")
        with col_b:
            st.metric(
                "Tree Backup final return (last 100)", f"{tb_returns[-100:].mean():.1f}"
            )
        with col_c:
            st.metric(
                "Mean IS ratio",
                f"{np.mean(is_ratios):.2f}",
                help="Higher ratios → higher variance in IS updates",
            )

        st.info(
            f"With n = {n} and ε = {epsilon}, the IS ratio can become large, introducing "
            f"instability. Tree Backup sidesteps this entirely. Try increasing n to 8 or 16 "
            f"and a wide behaviour policy (high ε) to see the IS variance problem amplify."
        )
