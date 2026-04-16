import numpy as np
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from utils.gridworld import (
    GRID_SIZE, TERMINAL_STATES,
    rc_to_state, ACTION_ARROWS,
    solve_value_iteration, solve_policy_iteration,
    policy_to_arrow_grid, make_value_heatmap,
)


def show():
    st.title("Value Iteration")
    st.markdown("**Section 2 — Dynamic Programming**")

    # ── Concept explanation ────────────────────────────────────────────────
    st.header("Skipping the Inner Loop")
    st.markdown(
        """
Policy iteration works, but it has a cost: before I can improve the policy, I have to evaluate
it *completely* — running sweeps until convergence. That inner evaluation loop can take many
sweeps, especially early on when the policy is far from optimal.

**Value iteration** makes a simple observation: I don't need to fully evaluate the policy before
improving it. I can collapse evaluation and improvement into a single update.

Instead of averaging over my policy, I just take the **maximum over all actions**. For each
state, I ask: *what's the best total return I could get from here if I acted optimally?*

That's the **Bellman optimality equation**:
"""
    )
    st.latex(
        r"V_{k+1}(s) = \max_a \left[ r(s,a) + \gamma V_k(s') \right]"
    )
    st.markdown(
        """
I apply this update to every state, over and over, until the values stop changing.
At convergence, I have $V^*$ — the optimal value function. I then extract the optimal
policy in a single greedy pass: for each state, pick whichever action achieved the max.

### How Does This Compare to Policy Iteration?

Think of policy iteration as two separate phases that alternate:

> **evaluate fully → improve once → evaluate fully → improve once → ...**

Value iteration fuses them into one:

> **one Bellman-max sweep → one Bellman-max sweep → ... → converge → extract π***

There's no inner loop at all. Each sweep does a little of both evaluation and improvement
simultaneously. The trade-off is that value iteration may need more total sweeps to converge —
but each sweep is cheap, and there's no overhead managing the inner evaluation loop.
"""
    )

    col1, col2 = st.columns(2)
    with col1:
        st.success(
            "**Advantage of value iteration:** No inner evaluation loop. Simpler to implement "
            "and easier to analyse. Forms the foundation for Q-learning and deep RL methods."
        )
    with col2:
        st.info(
            "**Advantage of policy iteration:** Fewer outer iterations (the policy converges "
            "fast). Each evaluation sweep can be parallelised. Often faster wall-clock time "
            "when states are many but policy iterations are few."
        )

    with st.expander("Deep Dive — Bellman Optimality Equations"):
        st.markdown(
            r"""
The full Bellman optimality equations for $V^*$ and $Q^*$:

$$V^*(s) = \max_a \mathbb{E}\bigl[R_{t+1} + \gamma V^*(S_{t+1}) \mid S_t=s, A_t=a\bigr]$$

$$Q^*(s,a) = \mathbb{E}\bigl[R_{t+1} + \gamma \max_{a'} Q^*(S_{t+1},a') \mid S_t=s, A_t=a\bigr]$$

The $Q^*$ equation is the foundation of **Q-learning**: if I don't know the model, I can
still approximate $Q^*$ by sampling transitions and bootstrapping off my current estimate
of $\max_{a'} Q(s', a')$. That's the jump from model-based DP to model-free RL.

Both equations involve a $\max$ operator, making them nonlinear. Despite this, the Bellman
optimality operator is still a contraction in the $\ell_\infty$ norm (for $\gamma < 1$),
guaranteeing a unique fixed point $V^*$.
"""
        )

    st.divider()

    # ── Simulation ────────────────────────────────────────────────────────
    st.header("Interactive Simulation — Value Iteration vs Policy Iteration")
    st.markdown(
        "Adjust θ and run both algorithms. Both use the same convergence criterion so the "
        "comparison is fair. **Learning outcome:** value iteration typically needs more total "
        "Bellman sweeps than policy iteration, but avoids the inner evaluation loop — check "
        "the sweep count bar chart and notice both algorithms converge to identical V* and π*."
    )

    theta = st.select_slider(
        "Convergence threshold θ",
        options=[1e-2, 1e-3, 1e-4, 1e-5, 1e-6],
        value=1e-4,
        format_func=lambda x: f"{x:.0e}",
        help="Stop iterating when the largest value change in a sweep falls below θ. Applied to both algorithms so the sweep count comparison is meaningful.",
    )

    if st.button("Run Both Algorithms", type="primary"):
        vi_V, vi_policy, vi_deltas = solve_value_iteration(float(theta))
        pi_history = solve_policy_iteration(theta=1e-6)
        pi_V = pi_history[-1]["V"]
        pi_policy = pi_history[-1]["policy"]

        # Count total eval sweeps across all PI iterations
        # (re-run to get sweep count — still cached so no extra work)
        pi_iters = len(pi_history)
        # Approximate PI sweep count: evaluate each policy until convergence.
        # We'll do a quick recount from scratch since solve_policy_iteration doesn't
        # surface per-iteration sweep counts in its return value.
        pi_total_sweeps = _count_pi_sweeps(float(theta))

        st.header("Results")

        # Side-by-side heatmaps
        col_vi, col_pi = st.columns(2)
        with col_vi:
            st.plotly_chart(
                make_value_heatmap(vi_V, "Value Iteration — V*(s)"),
                use_container_width=True,
            )
        with col_pi:
            st.plotly_chart(
                make_value_heatmap(pi_V, "Policy Iteration — V*(s)"),
                use_container_width=True,
            )

        max_diff = float(np.max(np.abs(vi_V - pi_V)))
        if max_diff < 1e-3:
            st.success(
                f"Both algorithms produced **identical value functions** "
                f"(max difference = {max_diff:.2e}). Same optimal V*, same optimal π*."
            )
        else:
            st.warning(f"Numerical difference between methods: {max_diff:.4f}")

        # Optimal policies side by side
        st.subheader("Optimal Policies")
        col_vip, col_pip = st.columns(2)
        with col_vip:
            st.markdown("**Value Iteration:**")
            st.code(_policy_str(vi_policy), language=None)
        with col_pip:
            st.markdown("**Policy Iteration:**")
            st.code(_policy_str(pi_policy), language=None)

        # Convergence comparison
        st.subheader("Convergence Speed")

        vi_sweeps = len(vi_deltas)
        fig_bar = go.Figure()
        fig_bar.add_trace(go.Bar(
            x=["Value Iteration", "Policy Iteration"],
            y=[vi_sweeps, pi_total_sweeps],
            marker_color=["#636EFA", "#EF553B"],
            text=[str(vi_sweeps), str(pi_total_sweeps)],
            textposition="auto",
        ))
        fig_bar.update_layout(
            title="Total Bellman Update Sweeps to Convergence",
            xaxis_title="Algorithm", yaxis_title="Sweeps",
            template="plotly_white", height=350,
        )
        st.plotly_chart(fig_bar, use_container_width=True)

        # VI delta plot
        fig_conv = go.Figure()
        fig_conv.add_trace(go.Scatter(
            x=list(range(1, vi_sweeps + 1)), y=vi_deltas,
            mode="lines+markers", name="VI max |ΔV|",
            line=dict(color="#636EFA", width=2), marker=dict(size=4),
        ))
        fig_conv.add_hline(y=float(theta), line_dash="dash", line_color="red",
                           annotation_text=f"θ = {theta:.0e}")
        fig_conv.update_layout(
            title="Value Iteration — Convergence per Sweep",
            xaxis_title="Sweep", yaxis_title="Max |ΔV|",
            yaxis_type="log", template="plotly_white", height=320,
        )
        st.plotly_chart(fig_conv, use_container_width=True)

        # Metrics
        st.header("Key Takeaways")
        col_a, col_b, col_c = st.columns(3)
        with col_a:
            st.metric("VI total sweeps", vi_sweeps)
        with col_b:
            st.metric("PI total eval sweeps", pi_total_sweeps)
        with col_c:
            st.metric("PI policy iterations", pi_iters)

        # Summary
        st.divider()
        st.header("Policy Iteration vs Value Iteration — When to Use Each")
        col1, col2 = st.columns(2)
        with col1:
            st.info(
                "**Policy Iteration — prefer when:**\n\n"
                "The policy converges in very few iterations (common even on large MDPs). "
                "Each evaluation pass can be parallelised across states. "
                "You need a valid policy at every intermediate step."
            )
        with col2:
            st.success(
                "**Value Iteration — prefer when:**\n\n"
                "A full evaluation to convergence is expensive. You want a simpler algorithm "
                "with no inner loop. You're building toward model-free methods — VI's structure "
                "maps directly onto Q-learning."
            )
        st.warning(
            "**The key insight:** Both converge to the same V* and the same π*. "
            "The choice is purely about computational efficiency for your specific problem."
        )


# ── Helpers ────────────────────────────────────────────────────────────────

@st.cache_data
def _count_pi_sweeps(theta: float) -> int:
    """Recount total inner-loop evaluation sweeps from policy iteration."""
    from utils.gridworld import (
        N_STATES, TERMINAL_STATES, GAMMA,
        next_state_reward, initial_policy, N_ACTIONS,
    )
    policy = initial_policy()
    total = 0
    for _ in range(100):
        V = np.zeros(N_STATES)
        while True:
            delta = 0.0
            for s in range(N_STATES):
                if s in TERMINAL_STATES:
                    continue
                ns, r = next_state_reward(s, policy[s])
                v_new = r + GAMMA * V[ns]
                delta = max(delta, abs(V[s] - v_new))
                V[s] = v_new
            total += 1
            if delta < theta:
                break
        new_policy = np.zeros(N_STATES, dtype=int)
        for s in range(N_STATES):
            if s in TERMINAL_STATES:
                continue
            q = [next_state_reward(s, a)[1] + GAMMA * V[next_state_reward(s, a)[0]]
                 for a in range(N_ACTIONS)]
            new_policy[s] = int(np.argmax(q))
        if np.array_equal(policy, new_policy):
            break
        policy = new_policy
    return total


def _policy_str(policy: np.ndarray) -> str:
    ag = policy_to_arrow_grid(policy)
    return "\n".join("  ".join(row) for row in ag)
