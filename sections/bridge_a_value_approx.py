import numpy as np
import plotly.graph_objects as go
import streamlit as st


# ── Page ───────────────────────────────────────────────────────────────────

def show():
    st.title("Value Function Approximation")
    st.markdown(
        '<div style="background:#7B2D8B;height:3px;border-radius:2px;margin-bottom:1rem;"></div>',
        unsafe_allow_html=True,
    )
    st.markdown("**Neural Networks Bridge — Transition to Deep RL**")

    st.info(
        "This is a conceptual bridge page — no simulations to run yet. It builds the "
        "intuition you need before the Deep RL sections. Read through it before continuing."
    )

    # ── Why tabular breaks down ────────────────────────────────────────────
    st.header("Why Tabular Methods Break Down")
    st.markdown(
        """
Every algorithm so far has stored a separate value estimate for every state (or state-action
pair) in a table. For small environments — 4×4 GridWorld, a 19-state random walk, a maze
with 70 cells — this is perfectly fine.

But now consider:

- **Backgammon:** roughly $10^{20}$ possible board positions
- **Go:** roughly $10^{170}$ positions
- **A robot arm** with continuous joint angles: *uncountably many* states
- **Atari games:** each frame is a 210×160 pixel image — the state space is astronomically large

A lookup table for any of these is impossible. We need a way to *generalise* — to estimate
values for states we have never explicitly visited, by recognising that similar states probably
have similar values.

This is the **curse of dimensionality**: as the number of state variables grows, the number
of possible states grows exponentially, and tabular methods become completely infeasible.
"""
    )

    st.warning(
        "**The fundamental shift:** instead of storing one value per state, we will learn "
        "a *function* parameterised by weights θ that maps any state to a value estimate. "
        "Generalisation across states comes for free — nearby states share weights."
    )

    st.divider()

    # ── What is function approximation ────────────────────────────────────
    st.header("Function Approximation")
    st.markdown(
        """
Instead of a table V[s], we approximate the value function as:
"""
    )
    st.latex(r"\hat{v}(s, \boldsymbol{\theta}) \approx v_\pi(s)")
    st.markdown(
        r"""
where $\boldsymbol{\theta}$ is a vector of learnable **parameters** (weights). The function
$\hat{v}$ can be anything differentiable:

- **Linear approximation:** $\hat{v}(s, \boldsymbol{\theta}) = \boldsymbol{\theta}^\top \mathbf{x}(s)$,
  where $\mathbf{x}(s)$ is a hand-crafted feature vector representing state $s$
- **Neural network:** $\hat{v}(s, \boldsymbol{\theta})$ is a deep network — the features
  are learned automatically from the raw state representation

The same applies to action values:
"""
    )
    st.latex(r"\hat{q}(s, a, \boldsymbol{\theta}) \approx q_\pi(s, a)")

    st.markdown(
        """
The critical difference from tabular methods: when we update $\boldsymbol{\theta}$ based on
one state, the change **propagates to all nearby states** through the shared weights. A robot
that learns the arm is in a good position at angle θ=45° will automatically have a slightly
updated estimate for θ=46° — because they share the same weight vector.
"""
    )

    with st.expander("Deep Dive — The Universal Approximation Theorem"):
        st.markdown(
            """
A neural network with at least one hidden layer of sufficient width can approximate *any*
continuous function on a compact domain to arbitrary precision (Cybenko, 1989; Hornik, 1991).

This means: in principle, a neural network value approximator is powerful enough to represent
any value function, no matter how complex the environment. The practical challenges are not
about expressiveness — they are about:

1. **Learning:** finding the right weights via gradient descent with limited data
2. **Stability:** function approximation + bootstrapping + off-policy learning can diverge
   (the *deadly triad* — covered in Section 9)
3. **Sample efficiency:** deep networks need many updates to converge

The theorem guarantees representation power exists; it says nothing about whether gradient
descent will find the right weights in reasonable time.
"""
        )

    st.divider()

    # ── MSVE objective ─────────────────────────────────────────────────────
    st.header("What Are We Optimising?")
    st.markdown(
        """
With tabular methods, the goal was simple: make V[s] equal to the true value v(s) for every s.
With function approximation, we cannot represent every state exactly — we have fewer parameters
than states. So we need to define what "as good as possible" means.

The standard objective is the **Mean Squared Value Error (MSVE)**:
"""
    )
    st.latex(
        r"\overline{VE}(\boldsymbol{\theta}) = "
        r"\sum_{s \in \mathcal{S}} \mu(s) \bigl[v_\pi(s) - \hat{v}(s, \boldsymbol{\theta})\bigr]^2"
    )
    st.markdown(
        r"""
where $\mu(s)$ is a **state distribution** — how often we care about getting state $s$ right.
Under the on-policy distribution (states the agent actually visits), $\mu(s)$ is the fraction
of time the agent spends in state $s$.

We minimise MSVE using **stochastic gradient descent**: after observing a target value for
state $s$, take a small step in the direction that reduces the squared error.
"""
    )
    st.latex(
        r"\boldsymbol{\theta}_{t+1} = \boldsymbol{\theta}_t + \alpha"
        r"\bigl[v_\pi(S_t) - \hat{v}(S_t, \boldsymbol{\theta}_t)\bigr]"
        r"\nabla_{\boldsymbol{\theta}} \hat{v}(S_t, \boldsymbol{\theta}_t)"
    )
    st.markdown(
        "The gradient $\\nabla_{\\boldsymbol{\\theta}} \\hat{v}$ tells us how to change each "
        "weight to increase the value estimate. We move in the *opposite* direction to decrease "
        "the error."
    )

    st.divider()

    # ── Interactive linear approximator ───────────────────────────────────
    st.header("Interactive: Linear Value Approximation")
    st.markdown(
        """
Below is a **1D value function problem**. The true values are shown in orange.
A linear approximator uses a small number of **basis functions** (Gaussian bumps) as features,
and learns weights for each one.

Drag the slider to see how approximator capacity (number of basis functions) affects
how closely it can match the true values.
"""
    )

    n_basis = st.slider(
        "Number of basis functions (approximator capacity)",
        2, 20, 5,
        help="More basis functions → richer representation → closer fit to true values.",
    )

    # Generate a wiggly true value function
    x = np.linspace(0, 1, 200)
    true_v = (
        0.5 * np.sin(2 * np.pi * x)
        + 0.3 * np.sin(6 * np.pi * x)
        + 0.2 * np.cos(4 * np.pi * x)
    )

    # Gaussian basis functions
    centers = np.linspace(0, 1, n_basis)
    sigma = 1.0 / n_basis
    Phi = np.exp(-0.5 * ((x[:, None] - centers[None, :]) / sigma) ** 2)

    # Least-squares fit (closed-form solution — conceptual demo, not gradient descent)
    theta, _, _, _ = np.linalg.lstsq(Phi, true_v, rcond=None)
    approx_v = Phi @ theta

    mse = float(np.mean((true_v - approx_v) ** 2))

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x, y=true_v,
        mode="lines", name="True value function v(s)",
        line=dict(color="#EF553B", width=2.5, dash="dash"),
    ))
    fig.add_trace(go.Scatter(
        x=x, y=approx_v,
        mode="lines", name=f"Linear approx. ({n_basis} basis fns)",
        line=dict(color="#636EFA", width=2),
    ))
    fig.update_layout(
        title=f"Linear Approximation  |  MSE = {mse:.4f}",
        xaxis_title="State",
        yaxis_title="Value",
        template="plotly_white",
        height=380,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    st.plotly_chart(fig, use_container_width=True)

    st.caption(
        "With very few basis functions the approximator is too rigid — high bias. "
        "With many basis functions it can represent the true function closely. "
        "In practice, we don't have the true values — we learn θ from sampled experience."
    )

    st.divider()

    # ── What's next ───────────────────────────────────────────────────────
    st.header("What's Coming Next")
    st.markdown(
        """
**Bridge Page B** implements the first actual learning algorithm with function approximation:
semi-gradient TD(0) with a linear approximator. You will see how the update rule above works
in practice on the Random Walk problem, using NumPy only — no neural networks yet.

After the Bridge, the Deep RL sections use **neural networks** as the function approximator,
bringing in PyTorch for the gradient computations. The theory is identical; only the
implementation changes.
"""
    )
    st.success(
        "**Key intuition to carry forward:** function approximation trades the ability to "
        "represent states exactly for the ability to *generalise* across states. This is "
        "what makes RL scalable to real-world problems."
    )
