import numpy as np
import plotly.graph_objects as go
import streamlit as st

from utils.gridworld import (
    GRID_SIZE, N_STATES, TERMINAL_STATES,
    ACTION_ARROWS, rc_to_state,
    solve_policy_iteration, policy_to_arrow_grid, make_value_heatmap,
)


def _make_policy_heatmap(V: np.ndarray, policy: np.ndarray, iteration: int):
    """Heatmap with arrow + value annotation for each cell."""
    grid_V = V.reshape(GRID_SIZE, GRID_SIZE)
    arrow_grid = policy_to_arrow_grid(policy)
    annotations = []
    for r in range(GRID_SIZE):
        for c in range(GRID_SIZE):
            s = rc_to_state(r, c)
            arrow = arrow_grid[r][c]
            if s in TERMINAL_STATES:
                text = "<b>T</b>"
            else:
                text = f"<b>{arrow}</b><br><sub>{grid_V[r][c]:.1f}</sub>"
            annotations.append(dict(
                x=c, y=r, text=text, showarrow=False,
                font=dict(size=22 if s not in TERMINAL_STATES else 18, color="black"),
                xanchor="center", yanchor="middle",
            ))
    fig = make_value_heatmap(V, f"Iteration {iteration} — Policy (arrows) + Value Function",
                             annotations=annotations)
    return fig


def show():
    st.title("Policy Iteration")
    st.markdown("**Section 2 — Dynamic Programming**")

    # ── Concept explanation ────────────────────────────────────────────────
    st.header("From Evaluation to Improvement")
    st.markdown(
        """
Policy evaluation told me how good a given policy is. But what if the policy is bad?
**Policy iteration** uses that information to make it better — and keeps going until it can't.

The loop has two steps that alternate until nothing changes:

**Step 1 — Evaluate:** I take my current policy and run the Bellman expectation equation
to convergence. This gives me $V^\\pi$ — the value of every state under my current policy.

**Step 2 — Improve:** I go through every state and ask: *given what I now know about the
values of all states, is there a better action I could take here than my policy currently says?*
For each state, I look at all four possible actions, compute the immediate reward plus the
value of where I'd end up, and pick the best one:
"""
    )
    st.latex(
        r"\pi'(s) = \arg\max_a \left[ r(s,a) + \gamma V^\pi(s') \right]"
    )
    st.markdown(
        """
If the new policy $\\pi'$ is different from $\\pi$, I switch to it and repeat.
If nothing changed — every state already has the best possible action — the policy is **stable**,
and I'm done. That stable policy is the optimal policy $\\pi^*$.

### Why Is It Guaranteed to Improve?

The improvement step always picks the action that is at least as good as whatever the current
policy was doing. It can't make things worse. So each iteration produces a policy with
$V^{\\pi'} \\geq V^\\pi$ everywhere.

And since there are only finitely many deterministic policies, the process must terminate —
it can't keep improving forever without reaching optimality.
"""
    )

    col1, col2 = st.columns(2)
    with col1:
        st.info(
            "**Starting policy here:** DOWN in rows 0–2, RIGHT in row 3. "
            "This is chosen because it always reaches a terminal, which keeps the "
            "undiscounted evaluation well-defined."
        )
    with col2:
        st.success(
            "**What you'll see:** In just 3–4 iterations, the arrows reorganise themselves "
            "from a simple down-right pattern into the true optimal policy — the shortest "
            "path to the nearest terminal from every cell."
        )

    with st.expander("Deep Dive — Why Convergence Is Finite"):
        st.markdown(
            r"""
In a finite MDP with $|S|$ states and $|A|$ actions, there are $|A|^{|S|}$ deterministic policies.
Policy iteration produces a *strictly better* policy at each step (unless already optimal),
so it terminates in at most $|A|^{|S|}$ iterations.

For the 4×4 GridWorld: $4^{14} \approx 268$ million theoretical policies. In practice,
policy iteration converges in **3–4 iterations** — it's extremely efficient because each
evaluation step gives such rich information about how to improve.
"""
        )

    st.divider()

    # ── Simulation ────────────────────────────────────────────────────────
    st.header("Interactive Simulation")
    st.markdown(
        "Step through each iteration to watch the policy evolve from the initial "
        "seed toward the optimal solution."
    )

    if "pi_history" not in st.session_state:
        st.session_state.pi_history = None
        st.session_state.pi_step = 0

    col_run, _ = st.columns([2, 1])
    with col_run:
        if st.button("Run Policy Iteration", type="primary"):
            with st.spinner("Running…"):
                st.session_state.pi_history = solve_policy_iteration(theta=1e-6)
            st.session_state.pi_step = 0

    if st.session_state.pi_history is not None:
        history = st.session_state.pi_history
        n_iters = len(history)

        col_prev, col_next, col_final = st.columns(3)
        with col_prev:
            if st.button("← Previous") and st.session_state.pi_step > 0:
                st.session_state.pi_step -= 1
        with col_next:
            if st.button("Next →") and st.session_state.pi_step < n_iters - 1:
                st.session_state.pi_step += 1
        with col_final:
            if st.button("Jump to final"):
                st.session_state.pi_step = n_iters - 1

        step = st.session_state.pi_step
        snap = history[step]

        label = f"Iteration {step + 1} / {n_iters}"
        if snap["stable"]:
            label += " ✓ Optimal"
        st.progress((step + 1) / n_iters, text=label)

        st.plotly_chart(
            _make_policy_heatmap(snap["V"], snap["policy"], snap["iteration"]),
            use_container_width=True,
        )

        arrow_grid = policy_to_arrow_grid(snap["policy"])
        st.markdown("**Policy grid (arrows only):**")
        st.code("\n".join("  ".join(row) for row in arrow_grid), language=None)

        if snap["stable"]:
            st.success(
                f"Converged to the **optimal policy** in **{n_iters} iterations**. "
                "The improvement step found no state where switching actions would help — "
                "we've reached π*."
            )
        elif step + 1 < n_iters:
            n_changed = int(np.sum(snap["policy"] != history[step + 1]["policy"]))
            st.info(
                f"After this iteration's improvement step, **{n_changed} state(s)** will "
                "change their action. Press Next to see the updated policy."
            )

        st.header("Convergence Summary")
        col_a, col_b = st.columns(2)
        with col_a:
            st.metric("Policy iterations to converge", n_iters)
        with col_b:
            st.metric(
                "Optimal V at state (0,1)",
                f"{history[-1]['V'][1]:.2f}",
                help="Row 0, Col 1 — one step from the top-left terminal",
            )
