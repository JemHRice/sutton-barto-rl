import numpy as np
import plotly.graph_objects as go
import streamlit as st


# ── Environment ────────────────────────────────────────────────────────────

def _true_values_random_walk(n_states: int = 19) -> np.ndarray:
    """True state values for the n-state random walk under uniform random policy."""
    return np.linspace(-1, 1, n_states + 2)[1:-1]


# ── Cached simulation ──────────────────────────────────────────────────────

@st.cache_data
def run_nstep_td(
    n: int,
    alpha: float,
    gamma: float,
    n_episodes: int,
    n_states: int = 19,
) -> tuple[np.ndarray, np.ndarray]:
    """
    n-step TD prediction on a random walk.
    Returns (rms_errors [n_episodes], final_V [n_states]).
    Terminal states are indices 0 (left) and n_states+1 (right).
    Reward: -1 on left terminal, +1 on right terminal, 0 otherwise.
    """
    rng = np.random.default_rng(42)
    true_V = _true_values_random_walk(n_states)
    V = np.zeros(n_states + 2)  # index 0 and n_states+1 are terminals
    rms_errors = np.zeros(n_episodes)

    for ep in range(n_episodes):
        state = n_states // 2 + 1  # start in the middle (1-indexed)
        states = [state]
        rewards = [0.0]
        T = float("inf")
        t = 0

        while True:
            if t < T:
                # random walk: move left or right
                move = rng.choice([-1, 1])
                next_state = state + move
                if next_state == 0:
                    reward = -1.0
                    T = t + 1
                elif next_state == n_states + 1:
                    reward = 1.0
                    T = t + 1
                else:
                    reward = 0.0
                states.append(next_state)
                rewards.append(reward)
                state = next_state

            tau = t - n + 1
            if tau >= 0:
                # Compute n-step return
                end = int(min(tau + n, T))
                G = sum(
                    gamma ** (i - tau - 1) * rewards[i]
                    for i in range(tau + 1, end + 1)
                )
                if tau + n < T:
                    G += gamma ** n * V[states[tau + n]]
                s_tau = states[tau]
                if s_tau not in (0, n_states + 1):
                    V[s_tau] += alpha * (G - V[s_tau])

            t += 1
            if tau == T - 1:
                break

        interior = V[1 : n_states + 1]
        rms_errors[ep] = float(np.sqrt(np.mean((interior - true_V) ** 2)))

    return rms_errors, V[1 : n_states + 1]


@st.cache_data
def run_all_n(
    n_values: tuple[int, ...],
    alpha: float,
    gamma: float,
    n_episodes: int,
) -> dict[int, np.ndarray]:
    return {n: run_nstep_td(n, alpha, gamma, n_episodes)[0] for n in n_values}


# ── Page ───────────────────────────────────────────────────────────────────

def show():
    st.title("n-step TD Prediction")
    st.markdown("**Section 5 — n-step Bootstrapping · Chapter 7**")

    # ── Concept ───────────────────────────────────────────────────────────
    st.header("The Idea")
    st.markdown(
        """
TD(0) updates using the reward from **one** step ahead, then bootstraps from the estimated
value of the next state. Monte Carlo waits for the **entire episode** to finish, using the
true return all the way to the end.

n-step TD sits between them. It looks **n steps** into the future, collects real rewards for
those n steps, then bootstraps from wherever it ends up. As you increase n:

- **n = 1** → TD(0): low variance, high bias (bootstraps immediately from rough estimates)
- **n = ∞** → Monte Carlo: zero bias, high variance (real returns, but noisy)
- **Small n (2–8)** → often the sweet spot: enough real rewards to reduce bias, not so many
  that variance explodes
"""
    )

    st.latex(
        r"G_t^{(n)} \;=\; R_{t+1} + \gamma R_{t+2} + \cdots + \gamma^{n-1} R_{t+n}"
        r"\;+\; \gamma^n V(S_{t+n})"
    )

    st.markdown("The update rule is then the same as TD(0), but using this n-step return:")
    st.latex(r"V(S_t) \;\leftarrow\; V(S_t) + \alpha\bigl[G_t^{(n)} - V(S_t)\bigr]")

    st.info(
        "**Why does intermediate n often learn fastest?** With small n you bootstrap from "
        "inaccurate early estimates (bias). With large n the return includes many random steps "
        "(variance). There is a bias-variance sweet spot — usually n = 2 to 8 — where learning "
        "curves drop fastest."
    )

    with st.expander("Deep Dive — The n-step Return in Detail"):
        st.markdown(
            r"""
For an episode that terminates at time $T$, the n-step return is:

$$G_t^{(n)} = \sum_{k=1}^{\min(n,T-t)} \gamma^{k-1} R_{t+k} \;+\; \gamma^n V(S_{t+n}) \cdot \mathbf{1}[t+n < T]$$

When $t + n \geq T$ (we reach the terminal state before looking ahead $n$ steps), there is
no bootstrapped term — the return is just the sum of discounted rewards to termination,
exactly like Monte Carlo for that suffix of the episode.

This means n-step TD unifies TD(0) and MC within a single framework. Setting $n = 1$ gives
TD(0); letting $n \to \infty$ recovers the full Monte Carlo return.
"""
        )

    st.divider()

    # ── Environment ───────────────────────────────────────────────────────
    st.header("Environment: 19-State Random Walk")
    st.markdown(
        """
The agent starts in the centre of a chain of **19 states**. At each step it moves left or right
with equal probability. The left terminal pays **−1**, the right terminal pays **+1**. All other
transitions give **0** reward.

The true state values under the random policy are known analytically — they form a straight line
from −1 to +1. We use root-mean-square error against these true values to measure learning
speed.
"""
    )

    st.divider()

    # ── Controls ──────────────────────────────────────────────────────────
    st.header("Interactive Simulation")
    st.markdown(
        "Pick which values of n to compare, set the learning rate and discount, then run. "
        "Watch how the RMS error curves differ across n values."
    )

    col1, col2, col3 = st.columns(3)
    with col1:
        n_episodes = st.slider("Episodes", 50, 500, 200, 50,
                               help="Training episodes per n value. More episodes → smoother curves.")
        alpha = st.slider("α (step size)", 0.01, 0.5, 0.1, 0.01,
                          help="Learning rate. Too high → unstable; too low → slow.")
    with col2:
        gamma = st.slider("γ (discount)", 0.9, 1.0, 1.0, 0.01,
                          help="Discount factor. For episodic random walk, γ=1 is standard.")
        n_values_options = [1, 2, 4, 8, 16, 32]
        selected_ns = st.multiselect(
            "n values to compare",
            n_values_options,
            default=[1, 2, 4, 8, 16],
            help="Select which n-step values to plot on the same chart.",
        )
    with col3:
        st.markdown("")
        st.markdown("")
        show_value_fn = st.checkbox("Show learned value functions", value=True)

    if not selected_ns:
        st.warning("Select at least one n value to run the simulation.")
        return

    if st.button("Run Simulation", type="primary"):
        with st.spinner("Running n-step TD for each selected n..."):
            results = run_all_n(tuple(sorted(selected_ns)), alpha, gamma, n_episodes)

        st.header("Results")

        # ── Learning curves ────────────────────────────────────────────
        COLOURS = ["#636EFA", "#EF553B", "#00CC96", "#AB63FA", "#FFA15A", "#19D3F3"]
        fig = go.Figure()
        for i, n in enumerate(sorted(selected_ns)):
            fig.add_trace(
                go.Scatter(
                    x=list(range(1, n_episodes + 1)),
                    y=results[n],
                    mode="lines",
                    name=f"n = {n}",
                    line=dict(color=COLOURS[i % len(COLOURS)], width=2),
                )
            )
        fig.update_layout(
            title="RMS Error vs Episodes for Different n",
            xaxis_title="Episode",
            yaxis_title="RMS Error",
            template="plotly_white",
            height=420,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        st.plotly_chart(fig, use_container_width=True)

        # ── Value function comparison ───────────────────────────────────
        if show_value_fn:
            true_V = _true_values_random_walk()
            fig2 = go.Figure()
            fig2.add_trace(
                go.Scatter(
                    x=list(range(1, 20)),
                    y=true_V,
                    mode="lines",
                    name="True values",
                    line=dict(color="black", dash="dash", width=2),
                )
            )
            for i, n in enumerate(sorted(selected_ns)):
                _, learned_V = run_nstep_td(n, alpha, gamma, n_episodes)
                fig2.add_trace(
                    go.Scatter(
                        x=list(range(1, 20)),
                        y=learned_V,
                        mode="lines+markers",
                        name=f"n = {n}",
                        line=dict(color=COLOURS[i % len(COLOURS)], width=1.5),
                        marker=dict(size=4),
                    )
                )
            fig2.update_layout(
                title=f"Learned Value Functions After {n_episodes} Episodes",
                xaxis_title="State",
                yaxis_title="Estimated Value",
                template="plotly_white",
                height=380,
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            )
            st.plotly_chart(fig2, use_container_width=True)

        # ── Takeaways ──────────────────────────────────────────────────
        st.header("Key Takeaways")
        best_n = min(results, key=lambda n: results[n][-1])
        cols = st.columns(len(selected_ns))
        for i, n in enumerate(sorted(selected_ns)):
            with cols[i]:
                st.metric(
                    f"n = {n}",
                    f"{results[n][-1]:.4f}",
                    label_visibility="visible",
                )

        st.success(
            f"With these settings, **n = {best_n}** achieved the lowest final RMS error. "
            "Try adjusting α — a different step size often changes which n wins."
        )
        st.info(
            "The curves often cross: small n learns faster early (low bias on short episodes) "
            "but may plateau higher. Large n is slower to start but can converge closer to the "
            "true values once estimates stabilise."
        )
