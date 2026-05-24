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


def _value_net(n_obs: int, hidden: int) -> "nn.Sequential":
    return nn.Sequential(
        nn.Linear(n_obs, hidden),
        nn.Tanh(),
        nn.Linear(hidden, hidden),
        nn.Tanh(),
        nn.Linear(hidden, 1),
    )


# ── Training ───────────────────────────────────────────────────────────────────


@st.cache_data
def run_plain_reinforce(
    hidden: int,
    alpha: float,
    gamma: float,
    n_episodes: int,
    seed: int,
) -> tuple[list, list]:
    """REINFORCE without baseline. Returns (returns, smoothed)."""
    torch.manual_seed(seed)
    env = gym.make("CartPole-v1")
    n_obs = env.observation_space.shape[0]
    n_act = env.action_space.n
    policy = _policy_net(n_obs, n_act, hidden)
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


@st.cache_data
def run_reinforce_with_baseline(
    hidden: int,
    alpha_pol: float,
    alpha_val: float,
    gamma: float,
    n_episodes: int,
    seed: int,
) -> tuple[list, list]:
    """REINFORCE with learned value-function baseline. Returns (returns, smoothed)."""
    torch.manual_seed(seed)
    env = gym.make("CartPole-v1")
    n_obs = env.observation_space.shape[0]
    n_act = env.action_space.n
    policy = _policy_net(n_obs, n_act, hidden)
    value = _value_net(n_obs, hidden)
    opt_p = optim.Adam(policy.parameters(), lr=alpha_pol)
    opt_v = optim.Adam(value.parameters(), lr=alpha_val)
    returns = []

    for ep in range(n_episodes):
        obs, _ = env.reset(seed=seed + ep)
        log_probs, rewards, states = [], [], []
        while True:
            obs_t = torch.FloatTensor(obs)
            states.append(obs_t)
            probs = torch.softmax(policy(obs_t), dim=-1)
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

        states_t = torch.stack(states)
        values = value(states_t).squeeze()
        advantages = Gs_t - values.detach()
        adv_norm = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        loss_p = -(torch.stack(log_probs) * adv_norm).sum()
        loss_v = nn.functional.mse_loss(values, Gs_t)

        opt_p.zero_grad()
        loss_p.backward()
        opt_p.step()
        opt_v.zero_grad()
        loss_v.backward()
        opt_v.step()
        returns.append(float(sum(rewards)))

    env.close()
    w = max(1, min(20, n_episodes // 10))
    sm = np.convolve(returns, np.ones(w) / w, mode="valid").tolist()
    return returns, [None] * (w - 1) + sm


# ── Visualisations ─────────────────────────────────────────────────────────────


def _comparison_fig(
    r_plain: list,
    sm_plain: list,
    r_base: list,
    sm_base: list,
) -> go.Figure:
    eps_p = list(range(1, len(r_plain) + 1))
    eps_b = list(range(1, len(r_base) + 1))
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=eps_p,
            y=r_plain,
            mode="lines",
            name="REINFORCE (raw)",
            line=dict(color="#636EFA", width=1),
            opacity=0.20,
            showlegend=True,
        )
    )
    vx = [x for x, y in zip(eps_p, sm_plain) if y is not None]
    vy = [y for y in sm_plain if y is not None]
    if vy:
        fig.add_trace(
            go.Scatter(
                x=vx,
                y=vy,
                mode="lines",
                name="REINFORCE (mean)",
                line=dict(color="#636EFA", width=2),
            )
        )

    fig.add_trace(
        go.Scatter(
            x=eps_b,
            y=r_base,
            mode="lines",
            name="With baseline (raw)",
            line=dict(color="#EF553B", width=1),
            opacity=0.20,
            showlegend=True,
        )
    )
    vx2 = [x for x, y in zip(eps_b, sm_base) if y is not None]
    vy2 = [y for y in sm_base if y is not None]
    if vy2:
        fig.add_trace(
            go.Scatter(
                x=vx2,
                y=vy2,
                mode="lines",
                name="With baseline (mean)",
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
        title="REINFORCE vs REINFORCE+Baseline — Episode Return",
        xaxis_title="Episode",
        yaxis_title="Return",
        yaxis=dict(range=[0, 520]),
        template="plotly_white",
        height=380,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


def _variance_fig(r_plain: list, r_base: list, window: int) -> go.Figure:
    def rolling_std(data, w):
        result = []
        for i in range(len(data)):
            chunk = data[max(0, i - w + 1) : i + 1]
            result.append(float(np.std(chunk)) if len(chunk) > 1 else 0.0)
        return result

    eps = list(range(1, len(r_plain) + 1))
    std_p = rolling_std(r_plain, window)
    std_b = rolling_std(r_base, window)
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=eps,
            y=std_p,
            mode="lines",
            name="REINFORCE",
            line=dict(color="#636EFA", width=2),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=eps,
            y=std_b,
            mode="lines",
            name="With baseline",
            line=dict(color="#EF553B", width=2),
        )
    )
    fig.update_layout(
        title=f"Rolling Return Std-Dev (window = {window})",
        xaxis_title="Episode",
        yaxis_title="Std Dev",
        template="plotly_white",
        height=320,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


# ── Page ───────────────────────────────────────────────────────────────────────


def show():
    st.title("REINFORCE with Baseline")
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
    st.header("Reducing Variance with a Baseline")
    st.markdown(r"""
The REINFORCE gradient estimate has high variance because the return $G_t$ fluctuates
widely between episodes.  We can subtract any **baseline** $b(S_t)$ from the return
without changing the expected gradient:

$$\nabla J(\boldsymbol{\theta}) = \mathbb{E}_\pi\!\left[(G_t - b(S_t))\,\nabla\ln\pi(A_t \mid S_t, \boldsymbol{\theta})\right]$$

**Why is the baseline unbiased?** Because $\mathbb{E}_\pi[\nabla\ln\pi(A_t|S_t)] = 0$
for any baseline $b$ that does not depend on the action:

$$\sum_a \pi(a|s)\,\nabla\ln\pi(a|s) = \nabla\sum_a\pi(a|s) = \nabla 1 = 0$$

The best baseline minimises the variance of the update.  Using the learned state-value
function $\hat{v}(S_t, \mathbf{w})$ is a natural choice — the difference
$G_t - \hat{v}(S_t, \mathbf{w})$ is an estimate of the **advantage**:
how much better action $A_t$ was than average in state $S_t$.

$$\boldsymbol{\theta} \leftarrow \boldsymbol{\theta} + \alpha_\theta\,\delta_t\,\nabla\ln\pi(A_t \mid S_t, \boldsymbol{\theta})$$
$$\mathbf{w} \leftarrow \mathbf{w} + \alpha_w\,\nabla_\mathbf{w}\frac{1}{2}(\delta_t)^2 \quad\text{where } \delta_t = G_t - \hat{v}(S_t, \mathbf{w})$$
""")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Without baseline**")
        st.latex(r"\delta_t = G_t")
        st.caption(
            "Returns that are above zero increase log-probability; below zero decrease it.  "
            "But even a bad action might get reinforced if the whole episode was profitable."
        )
    with col2:
        st.markdown("**With value baseline**")
        st.latex(r"\delta_t = G_t - \hat{v}(S_t, \mathbf{w})")
        st.caption(
            "Only actions that were *better than expected* for that state get reinforced.  "
            "The baseline shifts the signal to be centred around zero, reducing variance."
        )

    st.info(
        "**The value network is trained simultaneously** using the MC return $G_t$ as a "
        "supervised regression target.  Early in training the baseline is noisy, so the "
        "variance reduction is modest — it improves as the value function becomes accurate."
    )

    with st.expander("Optimal baseline — why the value function?"):
        st.markdown(r"""
The variance of the REINFORCE update for a single step $t$ is proportional to
$\text{Var}[(G_t - b)\,\nabla\ln\pi]$.  The exact optimal baseline is:

$$b^*(s) = \frac{\mathbb{E}_\pi\!\left[G_t^2\,\|\nabla\ln\pi\|^2 \mid S_t = s\right]}
               {\mathbb{E}_\pi\!\left[\|\nabla\ln\pi\|^2 \mid S_t = s\right]}$$

This is a weighted average of $G_t^2$ — hard to compute exactly.  The state-value
function $v_\pi(s) = \mathbb{E}[G_t \mid S_t = s]$ is a simpler but very effective
approximation and is what is used in practice.
""")

    st.divider()

    # ── Simulation ─────────────────────────────────────────────────────────────
    st.header("Interactive Comparison — CartPole-v1")
    st.markdown(
        "Run both REINFORCE and REINFORCE+Baseline with identical seeds.  "
        "Both use the same two-layer MLP policy architecture.  "
        "Watch the variance difference in the learning curves."
    )

    col1, col2 = st.columns(2)
    with col1:
        hidden = st.slider("Hidden units per layer", 16, 256, 64, 16)
        alpha_pol = st.slider("α policy", 1e-4, 5e-3, 1e-3, 1e-4, format="%.4f")
        alpha_val = st.slider(
            "α value (baseline)", 1e-4, 5e-3, 3e-3, 1e-4, format="%.4f"
        )
    with col2:
        gamma = st.slider("γ (discount)", 0.90, 1.00, 0.99, 0.01)
        n_episodes = st.slider("Episodes", 100, 1000, 500, 50)
        seed = st.number_input("Random seed", value=42, step=1)
        window = st.slider("Variance window", 5, 50, 20)

    if st.button("Compare Methods", type="primary"):
        with st.spinner("Training both methods…"):
            r_plain, sm_plain = run_plain_reinforce(
                hidden, alpha_pol, float(gamma), n_episodes, int(seed)
            )
            r_base, sm_base = run_reinforce_with_baseline(
                hidden, alpha_pol, alpha_val, float(gamma), n_episodes, int(seed)
            )

        st.header("Results")
        tab1, tab2 = st.tabs(["Learning Curves", "Return Variance"])
        with tab1:
            st.plotly_chart(
                _comparison_fig(r_plain, sm_plain, r_base, sm_base),
                use_container_width=True,
            )
        with tab2:
            st.plotly_chart(
                _variance_fig(r_plain, r_base, window),
                use_container_width=True,
            )

        early = max(1, n_episodes // 5)
        col_a, col_b = st.columns(2)
        with col_a:
            st.metric(
                "REINFORCE — mean return (last 20%)", f"{np.mean(r_plain[-early:]):.1f}"
            )
        with col_b:
            st.metric(
                "With baseline — mean return (last 20%)",
                f"{np.mean(r_base[-early:]):.1f}",
            )

        st.header("Key Takeaways")
        var_plain = float(np.std(r_plain[-early:]))
        var_base = float(np.std(r_base[-early:]))
        if var_base < var_plain:
            reduction = 100 * (var_plain - var_base) / (var_plain + 1e-8)
            st.success(
                f"The baseline reduced return variance by **{reduction:.0f}%** in the last 20% "
                f"(std: {var_plain:.1f} → {var_base:.1f}).  "
                "More consistent training translates directly to faster convergence."
            )
        else:
            st.warning(
                "With these settings the baseline did not reduce variance — the value network "
                "may still be too noisy.  Try more episodes or a larger α value."
            )
        st.info(
            "**Next step**: Page 33 (Actor-Critic) replaces the Monte Carlo return $G_t$ with "
            "a one-step TD bootstrap $R + \\gamma\\hat{v}(S')$, making updates fully online "
            "— no need to wait for episode end.  This trades some variance for bias."
        )
