import numpy as np
import plotly.graph_objects as go
import streamlit as st

from utils.gridworld import (
    GRID_SIZE, TERMINAL_STATES,
    solve_random_policy, make_value_heatmap,
)


def show():
    st.title("Policy Evaluation")
    st.markdown("**Section 2 — Dynamic Programming**")

    # ── Concept explanation ────────────────────────────────────────────────
    st.header("A New Setting — I Know the Rules")
    st.markdown(
        """
In the bandit problems, I had to *discover* what the machines paid out through trial and error.
**Dynamic Programming** is different: I have a complete map of the environment. I know exactly
where every action takes me and what reward I'll get. The question is no longer *what happens
when I try this?* — I already know. The question is *what's the best plan?*

The environment here is a **4×4 GridWorld**:
- I start anywhere except the two terminal states (top-left and bottom-right corners).
- Every step costs **−1** reward, so I want to reach a terminal as quickly as possible.
- If I walk into a wall, I bounce back and still pay the −1 cost.
- I'm not trying to be optimal yet — first I just want to understand how good a particular
  policy is.

### What Is a Value Function?

The **value** of a state is the total reward I expect to collect from that state onward,
following my current policy forever. It's a single number that summarises how good or bad
it is to be in that state.

States close to a terminal are less bad — I'll reach the end quickly.
States far from both terminals are very negative — I'll take many steps.

### The Bellman Expectation Equation

Here's how I actually calculate the value of a state:

For each action my policy might take from this state, I look at where I'll end up
and add the reward to the (discounted) value of that next state — that gives me a single
number for that action. Then I take the weighted average of those numbers across all
actions according to my policy. The result is the value of the state.

That procedure, written as an equation:
"""
    )
    st.latex(
        r"V^\pi(s) = \sum_a \pi(a|s) \sum_{s',r} p(s', r \mid s, a)"
        r"\left[ r + \gamma V^\pi(s') \right]"
    )
    st.markdown(
        """
Because the right-hand side contains $V^\\pi$ itself, I can't solve this in one shot.
Instead, I start with all values at zero and apply this update repeatedly for every state —
each sweep uses the previous round's values to compute the next. Eventually the values
stop changing. That's **policy evaluation**: iterate the Bellman update until convergence.

### Why Does It Converge?

The Bellman update is a contraction: each sweep brings every value closer to the true $V^\\pi$.
For episodic tasks like this GridWorld — where every trajectory eventually reaches a terminal
— the convergence is guaranteed regardless of where I start.
"""
    )

    col1, col2 = st.columns(2)
    with col1:
        st.success(
            "**What I'm evaluating here:** the completely random policy — each of the four "
            "actions is chosen with equal probability 1/4. It's not smart, but its value "
            "function has a clean analytical solution I can verify against."
        )
    with col2:
        st.info(
            "**Convergence threshold θ:** I stop iterating when the largest change to any "
            "state's value in a full sweep falls below θ. Smaller θ → more sweeps → more "
            "accurate values."
        )

    with st.expander("Deep Dive — In-Place vs Two-Array Updates"):
        st.markdown(
            r"""
The textbook version of policy evaluation uses two arrays: a copy of the old values and the
new values being computed. This guarantees that each sweep uses only values from the
*previous* iteration.

In practice, **in-place** updates — overwriting $V$ as you go — converge faster using fewer
sweeps. The values are still correct at convergence; you just get there sooner because
later states in the sweep benefit from already-updated earlier states.

Both approaches converge to the same $V^\pi$. This implementation uses in-place updates.
"""
        )

    st.divider()

    # ── Simulation ────────────────────────────────────────────────────────
    st.header("Interactive Simulation")
    st.markdown(
        "Adjust θ and run policy evaluation on the random policy. "
        "**Learning outcome:** a smaller θ demands more sweeps but produces more accurate values — "
        "notice how the value function barely changes after the first few dozen sweeps even at "
        "very small θ. Most of the information is captured early."
    )

    theta = st.select_slider(
        "Convergence threshold θ",
        options=[1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6],
        value=1e-4,
        format_func=lambda x: f"{x:.0e}",
        help="Stop iterating when the largest value change in a full sweep falls below θ. Smaller θ → more sweeps → higher precision.",
    )

    if st.button("Run Policy Evaluation", type="primary"):
        V, deltas = solve_random_policy(float(theta))
        grid_V = V.reshape(GRID_SIZE, GRID_SIZE)

        st.header("Results")

        # Heatmap with terminal markers
        ann = []
        for r in range(GRID_SIZE):
            for c in range(GRID_SIZE):
                from utils.gridworld import rc_to_state
                s = rc_to_state(r, c)
                text = "T" if s in TERMINAL_STATES else f"{grid_V[r][c]:.2f}"
                ann.append(dict(
                    x=c, y=r, text=text, showarrow=False,
                    font=dict(size=16 if s not in TERMINAL_STATES else 20, color="black"),
                    xanchor="center", yanchor="middle",
                ))
        fig_heat = make_value_heatmap(V, "V(s) — Random Policy on 4×4 GridWorld", annotations=ann)
        # Add blue borders on terminal states
        for tr, tc in [(0, 0), (3, 3)]:
            fig_heat.add_shape(type="rect",
                x0=tc-0.5, x1=tc+0.5, y0=tr-0.5, y1=tr+0.5,
                line=dict(color="blue", width=3))
        st.plotly_chart(fig_heat, use_container_width=True)
        st.caption("Blue borders = terminal states (value = 0). Greener = closer to 0 = better.")

        # Convergence plot
        fig_conv = go.Figure()
        fig_conv.add_trace(go.Scatter(
            x=list(range(1, len(deltas) + 1)), y=deltas,
            mode="lines+markers", line=dict(color="#636EFA", width=2),
            marker=dict(size=4), name="Max |ΔV|",
        ))
        fig_conv.add_hline(y=float(theta), line_dash="dash", line_color="red",
                           annotation_text=f"θ = {theta:.0e}")
        fig_conv.update_layout(
            title="Convergence — Max Change per Sweep",
            xaxis_title="Sweep", yaxis_title="Max |ΔV|",
            yaxis_type="log", template="plotly_white", height=350,
        )
        st.plotly_chart(fig_conv, use_container_width=True)

        st.header("Key Takeaways")
        col_a, col_b, col_c = st.columns(3)
        with col_a:
            st.metric("Sweeps to converge", len(deltas))
        with col_b:
            st.metric("Worst state value", f"{V.min():.2f}")
        with col_c:
            worst = int(np.argmin(V))
            from utils.gridworld import state_to_rc
            wr, wc = state_to_rc(worst)
            st.metric("Worst state location", f"row {wr}, col {wc}")

        st.info(
            "Notice the symmetry: the GridWorld is symmetric about both diagonals, "
            "and the value function reflects that. States equidistant from both terminals "
            "have equal values. This matches the analytical solution in Sutton & Barto Example 4.1."
        )
        st.success(
            f"Converged in **{len(deltas)} sweeps** with θ = {theta:.0e}. "
            "Try a smaller θ to see how many more sweeps are needed for extra precision — "
            "but notice the values barely change after the first few dozen sweeps."
        )
