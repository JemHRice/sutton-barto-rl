import numpy as np
import plotly.graph_objects as go
import streamlit as st

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim

    _TORCH_OK = True
except ImportError:
    _TORCH_OK = False

try:
    import gymnasium as gym

    _GYM_OK = True
except ImportError:
    _GYM_OK = False


def _policy_net(n_obs: int, n_actions: int, hidden: int) -> "nn.Sequential":
    return nn.Sequential(
        nn.Linear(n_obs, hidden),
        nn.Tanh(),
        nn.Linear(hidden, hidden),
        nn.Tanh(),
        nn.Linear(hidden, n_actions),
    )


# ── Training ───────────────────────────────────────────────────────────────────


@st.cache_data
def run_reinforce(
    hidden: int,
    alpha: float,
    gamma: float,
    n_episodes: int,
    seed: int,
) -> tuple[list, list]:
    """REINFORCE on CartPole-v1. Returns (episode_returns, smoothed)."""
    torch.manual_seed(seed)
    env = gym.make("CartPole-v1")
    n_obs = env.observation_space.shape[0]
    n_actions = env.action_space.n
    policy = _policy_net(n_obs, n_actions, hidden)
    opt = optim.Adam(policy.parameters(), lr=alpha)
    returns = []

    for ep in range(n_episodes):
        obs, _ = env.reset(seed=seed + ep)
        log_probs, rewards = [], []
        while True:
            probs = torch.softmax(policy(torch.FloatTensor(obs)), dim=-1)
            dist = torch.distributions.Categorical(probs)
            action = dist.sample()
            log_probs.append(dist.log_prob(action))
            obs, r, terminated, truncated, _ = env.step(action.item())
            rewards.append(float(r))
            if terminated or truncated:
                break

        G, Gs = 0.0, []
        for r in reversed(rewards):
            G = r + gamma * G
            Gs.insert(0, G)
        Gs_t = torch.FloatTensor(Gs)
        Gs_t = (Gs_t - Gs_t.mean()) / (Gs_t.std() + 1e-8)
        loss = -(torch.stack(log_probs) * Gs_t).sum()
        opt.zero_grad()
        loss.backward()
        opt.step()
        returns.append(float(sum(rewards)))

    env.close()
    w = max(1, min(20, n_episodes // 10))
    sm = np.convolve(returns, np.ones(w) / w, mode="valid").tolist()
    return returns, [None] * (w - 1) + sm


# ── Visualisation ──────────────────────────────────────────────────────────────


def _learning_curve(returns: list, smooth: list, title: str) -> go.Figure:
    eps = list(range(1, len(returns) + 1))
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=eps,
            y=returns,
            mode="lines",
            name="Return",
            line=dict(color="#636EFA", width=1),
            opacity=0.25,
        )
    )
    valid_x = [x for x, y in zip(eps, smooth) if y is not None]
    valid_y = [y for y in smooth if y is not None]
    if valid_y:
        fig.add_trace(
            go.Scatter(
                x=valid_x,
                y=valid_y,
                mode="lines",
                name="Rolling mean",
                line=dict(color="#EF553B", width=2),
            )
        )
    fig.add_hline(
        y=475,
        line_dash="dot",
        line_color="green",
        annotation_text="Solved (475)",
        annotation_position="bottom right",
    )
    fig.update_layout(
        title=title,
        xaxis_title="Episode",
        yaxis_title="Return",
        yaxis=dict(range=[0, 520]),
        template="plotly_white",
        height=360,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


# ── Page ───────────────────────────────────────────────────────────────────────


def show():
    st.title("REINFORCE — Monte Carlo Policy Gradient")
    st.markdown(
        '<div style="background:#7B2D8B;height:3px;'
        'border-radius:2px;margin-bottom:1rem;"></div>',
        unsafe_allow_html=True,
    )
    st.markdown("**Section 11 — Policy Gradients (Chapter 13)**")

    if not _TORCH_OK:
        st.error(
            "PyTorch is not installed.  Install it with `pip install torch` and restart the app."
        )
        return
    if not _GYM_OK:
        st.error(
            "Gymnasium is not installed.  Install it with `pip install gymnasium` and restart the app."
        )
        return

    # ── Concept ───────────────────────────────────────────────────────────────
    st.header("Direct Policy Optimisation")
    st.markdown(r"""
All previous chapters parameterised a **value function** and derived a policy from it.
Policy gradient methods take a different route: they directly parameterise the policy
$\pi(a \mid s, \boldsymbol{\theta})$ and optimise $\boldsymbol{\theta}$ by gradient ascent
on the expected return:

$$J(\boldsymbol{\theta}) = \mathbb{E}_\pi\!\left[\sum_{t=0}^T \gamma^t R_{t+1}\right]$$

The **Policy Gradient Theorem** (Sutton et al., 2000) gives the gradient without a model:

$$\nabla J(\boldsymbol{\theta}) \propto \sum_s \mu(s) \sum_a q_\pi(s,a)\,\nabla\pi(a \mid s, \boldsymbol{\theta})$$

Multiplying and dividing by $\pi(a \mid s)$ rewrites this as an expectation:

$$\nabla J(\boldsymbol{\theta}) = \mathbb{E}_\pi\!\left[G_t\,\nabla\ln\pi(A_t \mid S_t, \boldsymbol{\theta})\right]$$

**REINFORCE** estimates this expectation from a single complete episode:

$$\boldsymbol{\theta} \leftarrow \boldsymbol{\theta} + \alpha\,\gamma^t G_t\,\nabla\ln\pi(A_t \mid S_t, \boldsymbol{\theta})$$
""")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Log-derivative trick**")
        st.latex(
            r"\nabla \ln \pi(a|s,\boldsymbol{\theta}) = \frac{\nabla \pi(a|s,\boldsymbol{\theta})}{\pi(a|s,\boldsymbol{\theta})}"
        )
        st.caption(
            "Converts a gradient-of-expectation into an expectation-of-gradient. "
            "The score function $\\nabla \\ln \\pi$ is computed via back-propagation "
            "— no environment model is needed."
        )
    with col2:
        st.markdown("**Intuition**")
        st.markdown(
            "Actions that led to above-average returns have their log-probability increased; "
            "actions that led to below-average returns have it decreased. "
            "The magnitude of each update scales with $|G_t|$."
        )

    st.info(
        "**REINFORCE is unbiased but high-variance.** The Monte Carlo return $G_t$ is an "
        "exact sample of $q_\\pi(S_t, A_t)$, so the gradient estimate has no bias — but "
        "$G_t$ fluctuates enormously between episodes.  Page 32 shows how a baseline "
        "dramatically reduces this variance without introducing any bias."
    )

    with st.expander("Policy Gradient Theorem — proof sketch"):
        st.markdown(r"""
For episodic tasks, $J(\boldsymbol{\theta}) = v_{\pi_\theta}(s_0)$.
Differentiating the Bellman equation for $v_\pi$ with respect to $\boldsymbol{\theta}$,
unrolling the resulting recursion in $\nabla q_\pi$, and collecting the geometric series gives:

$$\nabla J(\boldsymbol{\theta}) \propto \sum_s \mu(s) \sum_a q_\pi(s,a)\,\nabla\pi(a \mid s, \boldsymbol{\theta})$$

where $\mu(s)$ is the on-policy state-visitation distribution.
Multiplying and dividing by $\pi(a|s)$:

$$= \mathbb{E}_\pi\!\left[q_\pi(S_t, A_t)\,\nabla\ln\pi(A_t \mid S_t, \boldsymbol{\theta})\right]$$

Replacing $q_\pi(S_t, A_t)$ with its unbiased Monte Carlo sample $G_t$ gives REINFORCE.
The key insight: the gradient is now an expectation that can be sampled from episodes
— no transition model $p(s' \mid s, a)$ required.
""")

    with st.expander("Return normalisation"):
        st.markdown(r"""
In practice, REINFORCE is stabilised by normalising the returns before each update:

$$\tilde{G}_t = \frac{G_t - \bar{G}}{\sigma_G + \varepsilon}$$

Subtracting the episode mean $\bar{G}$ is equivalent to using a constant within-episode
baseline — it does not change the expected gradient direction but keeps the effective
step size $\alpha \tilde{G}_t$ bounded regardless of episode length.
Without normalisation, long episodes produce very large gradient magnitudes and short
episodes produce very small ones, causing instability.
""")

    st.divider()

    # ── Simulation ─────────────────────────────────────────────────────────────
    st.header("Interactive Simulation — CartPole-v1")
    st.markdown(
        "Train a two-layer MLP policy on CartPole-v1.  "
        "The environment gives +1 reward per step; the maximum return is 500.  "
        "The agent must learn to balance the pole by pushing the cart left or right."
    )

    col1, col2 = st.columns(2)
    with col1:
        hidden = st.slider("Hidden units per layer", 16, 256, 64, 16)
        alpha = st.slider("α (learning rate)", 1e-4, 5e-3, 1e-3, 1e-4, format="%.4f")
        gamma = st.slider("γ (discount)", 0.90, 1.00, 0.99, 0.01)
    with col2:
        n_episodes = st.slider("Episodes", 100, 1000, 500, 50)
        seed = st.number_input("Random seed", value=42, step=1)

    if st.button("Train REINFORCE", type="primary"):
        with st.spinner("Running REINFORCE on CartPole-v1…"):
            returns, smooth = run_reinforce(
                hidden, alpha, float(gamma), n_episodes, int(seed)
            )

        st.header("Results")
        st.plotly_chart(
            _learning_curve(returns, smooth, "REINFORCE — Episode Return"),
            use_container_width=True,
        )

        early = max(1, n_episodes // 5)
        col_a, col_b, col_c = st.columns(3)
        with col_a:
            st.metric("Mean return (first 20%)", f"{np.mean(returns[:early]):.1f}")
        with col_b:
            st.metric("Mean return (last 20%)", f"{np.mean(returns[-early:]):.1f}")
        with col_c:
            solved = sum(1 for r in returns[-early:] if r >= 475)
            st.metric("Solved episodes (≥475, last 20%)", f"{solved}/{early}")

        st.header("Key Takeaways")
        final_mean = float(np.mean(returns[-early:]))
        if final_mean >= 400:
            st.success(
                f"Policy has converged ({final_mean:.0f} mean return in last 20%).  "
                "The agent has discovered the balancing strategy — notice the high variance "
                "even in the later learning curve: REINFORCE is inherently noisy due to its "
                "Monte Carlo nature."
            )
        else:
            st.warning(
                f"Still learning ({final_mean:.0f} mean return in last 20%).  "
                "REINFORCE often needs many episodes to stabilise.  "
                "Try α=0.001, 500 episodes with seed 42 for consistent results."
            )
        st.info(
            "**Compare with Page 32**: the same task trained with REINFORCE+Baseline typically "
            "converges faster and with noticeably lower variance.  The baseline reduces the "
            "variance of each gradient estimate without changing its expected direction."
        )
