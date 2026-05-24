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


def _actor_net(n_obs: int, n_actions: int, hidden: int) -> "nn.Sequential":
    return nn.Sequential(
        nn.Linear(n_obs, hidden),
        nn.Tanh(),
        nn.Linear(hidden, hidden),
        nn.Tanh(),
        nn.Linear(hidden, n_actions),
    )


def _critic_net(n_obs: int, hidden: int) -> "nn.Sequential":
    return nn.Sequential(
        nn.Linear(n_obs, hidden),
        nn.Tanh(),
        nn.Linear(hidden, hidden),
        nn.Tanh(),
        nn.Linear(hidden, 1),
    )


# ── Training ───────────────────────────────────────────────────────────────────


@st.cache_data
def run_actor_critic(
    hidden: int,
    alpha_actor: float,
    alpha_critic: float,
    gamma: float,
    n_episodes: int,
    seed: int,
) -> tuple[list, list, list]:
    """
    One-step Actor-Critic on CartPole-v1.
    Returns (episode_returns, smoothed, td_errors_mean_per_ep).
    """
    torch.manual_seed(seed)
    env = gym.make("CartPole-v1")
    n_obs = env.observation_space.shape[0]
    n_act = env.action_space.n
    actor = _actor_net(n_obs, n_act, hidden)
    critic = _critic_net(n_obs, hidden)
    opt_a = optim.Adam(actor.parameters(), lr=alpha_actor)
    opt_c = optim.Adam(critic.parameters(), lr=alpha_critic)

    returns, td_errors = [], []

    for ep in range(n_episodes):
        obs, _ = env.reset(seed=seed + ep)
        done = False
        I = 1.0  # γ^t discount accumulator
        ep_ret = 0.0
        ep_td = []

        while not done:
            obs_t = torch.FloatTensor(obs)
            v_s = critic(obs_t).squeeze()

            probs = torch.softmax(actor(obs_t), dim=-1)
            dist = torch.distributions.Categorical(probs)
            action = dist.sample()
            log_p = dist.log_prob(action)

            obs2, r, terminated, truncated, _ = env.step(action.item())
            done = terminated or truncated
            ep_ret += float(r)

            obs2_t = torch.FloatTensor(obs2)
            with torch.no_grad():
                v_s2 = critic(obs2_t).squeeze() if not done else torch.tensor(0.0)
            delta = float(r) + gamma * v_s2.item() - v_s.item()
            ep_td.append(abs(delta))

            # Critic update (MSE on TD target)
            td_target = float(r) + gamma * v_s2.item()
            critic_loss = (torch.tensor(td_target) - v_s) ** 2
            opt_c.zero_grad()
            critic_loss.backward()
            opt_c.step()

            # Actor update (policy gradient with TD error as advantage)
            actor_loss = -I * delta * log_p
            opt_a.zero_grad()
            actor_loss.backward()
            opt_a.step()

            I *= gamma
            obs = obs2

        returns.append(ep_ret)
        td_errors.append(float(np.mean(ep_td)) if ep_td else 0.0)

    env.close()
    w = max(1, min(20, n_episodes // 10))
    sm = np.convolve(returns, np.ones(w) / w, mode="valid").tolist()
    return returns, [None] * (w - 1) + sm, td_errors


# ── Visualisations ─────────────────────────────────────────────────────────────


def _returns_fig(returns: list, smooth: list) -> go.Figure:
    eps = list(range(1, len(returns) + 1))
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=eps,
            y=returns,
            mode="lines",
            name="Return",
            line=dict(color="#00CC96", width=1),
            opacity=0.25,
        )
    )
    vx = [x for x, y in zip(eps, smooth) if y is not None]
    vy = [y for y in smooth if y is not None]
    if vy:
        fig.add_trace(
            go.Scatter(
                x=vx,
                y=vy,
                mode="lines",
                name="Rolling mean",
                line=dict(color="#19D3F3", width=2),
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
        title="Actor-Critic — Episode Return",
        xaxis_title="Episode",
        yaxis_title="Return",
        yaxis=dict(range=[0, 520]),
        template="plotly_white",
        height=350,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


def _td_error_fig(td_errors: list, window: int) -> go.Figure:
    eps = list(range(1, len(td_errors) + 1))
    kernel = np.ones(window) / window
    sm = np.convolve(td_errors, kernel, mode="valid").tolist()
    sm_x = list(range(window, len(td_errors) + 1))
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=eps,
            y=td_errors,
            mode="lines",
            name="Mean |δ|",
            line=dict(color="#FFA15A", width=1),
            opacity=0.30,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=sm_x,
            y=sm,
            mode="lines",
            name=f"Rolling mean ({window})",
            line=dict(color="#EF553B", width=2),
        )
    )
    fig.update_layout(
        title="Mean Absolute TD Error per Episode",
        xaxis_title="Episode",
        yaxis_title="|δ| mean",
        template="plotly_white",
        height=300,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


# ── Page ───────────────────────────────────────────────────────────────────────


def show():
    st.title("One-step Actor-Critic")
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
    st.header("Bootstrapping the Policy Gradient")
    st.markdown(r"""
REINFORCE with baseline uses the full Monte Carlo return $G_t$ to estimate the advantage.
**Actor-Critic** replaces $G_t$ with a one-step TD bootstrap, making updates **fully online**
— no need to wait until episode end:

$$\delta_t = R_{t+1} + \gamma\,\hat{v}(S_{t+1}, \mathbf{w}) - \hat{v}(S_t, \mathbf{w})$$

The **critic** $\hat{v}(s, \mathbf{w})$ is updated by minimising the squared TD error:

$$\mathbf{w} \leftarrow \mathbf{w} + \alpha_w\,\delta_t\,\nabla\hat{v}(S_t, \mathbf{w})$$

The **actor** $\pi(a|s, \boldsymbol{\theta})$ is updated using $\delta_t$ as the advantage:

$$\boldsymbol{\theta} \leftarrow \boldsymbol{\theta} + \alpha_\theta\,I\,\delta_t\,\nabla\ln\pi(A_t \mid S_t, \boldsymbol{\theta})$$

where $I = \gamma^t$ is a discount factor that downweights later steps (per S&B Alg. 13.5).
""")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Bias–variance tradeoff**")
        st.markdown(
            "Replacing $G_t$ (low bias, high variance) with a one-step TD target "
            "(low variance, some bias) speeds training at the cost of a small approximation error.  "
            "The TD error $\\delta_t$ is a biased but consistent estimate of the advantage."
        )
    with col2:
        st.markdown("**Two separate networks**")
        st.markdown(
            "The actor and critic are updated independently with different learning rates.  "
            "The critic loss is a simple MSE on the TD target.  "
            "The actor loss is the negative log-probability scaled by the detached TD error."
        )

    st.info(
        "**Actor-Critic is the foundation of modern deep RL.**  A2C (Advantage Actor-Critic) "
        "collects mini-batches in parallel; PPO (Page 34) adds a clipped surrogate to "
        "prevent destructively large updates.  Both are descendants of this one-step algorithm."
    )

    with st.expander("Why detach δ before the actor update?"):
        st.markdown(r"""
The TD error $\delta_t = r + \gamma V(s') - V(s)$ depends on both $V(s)$ and $V(s')$.
If we kept the gradient flowing through $\delta_t$, the actor update would also
differentiate through the critic — mixing two separate objectives.

By calling `.detach()` on $\delta_t$ before the actor loss, we treat it as a scalar
weighting constant.  The critic is updated separately to reduce $\delta_t$ toward zero.
This clean separation is standard practice and avoids unstable cross-network gradients.
""")

    with st.expander("The I = γ^t discount factor"):
        st.markdown(r"""
The $I = \gamma^t$ term appears in the formal derivation of the policy gradient theorem
for episodic tasks.  It down-weights updates for actions taken later in the episode,
reflecting that they contributed less to the total discounted return.

In practice many implementations omit $I$ (setting it to 1 throughout), which corresponds
to an **undiscounted average-reward** objective on each step.  Omitting $I$ often works
better empirically — the correct discount can shrink late-episode gradients so much that
the policy stops learning from long episodes.  Both variants appear in the literature.
""")

    st.divider()

    # ── Simulation ─────────────────────────────────────────────────────────────
    st.header("Interactive Simulation — CartPole-v1")
    st.markdown(
        "One-step actor-critic with separate actor and critic networks.  "
        "The actor is updated at every step — the episode return is only used for evaluation."
    )

    col1, col2 = st.columns(2)
    with col1:
        hidden = st.slider("Hidden units per layer", 16, 256, 64, 16)
        alpha_actor = st.slider("α actor", 1e-4, 5e-3, 5e-4, 1e-4, format="%.4f")
        alpha_critic = st.slider("α critic", 1e-4, 1e-2, 1e-3, 1e-4, format="%.4f")
    with col2:
        gamma = st.slider("γ (discount)", 0.90, 1.00, 0.99, 0.01)
        n_episodes = st.slider("Episodes", 100, 1000, 500, 50)
        seed = st.number_input("Random seed", value=42, step=1)
        window = st.slider("Rolling mean window", 5, 50, 20)

    if st.button("Train Actor-Critic", type="primary"):
        with st.spinner("Running one-step Actor-Critic on CartPole-v1…"):
            returns, smooth, td_errors = run_actor_critic(
                hidden, alpha_actor, alpha_critic, float(gamma), n_episodes, int(seed)
            )

        st.header("Results")
        tab1, tab2 = st.tabs(["Episode Return", "TD Error"])
        with tab1:
            st.plotly_chart(_returns_fig(returns, smooth), use_container_width=True)
        with tab2:
            st.plotly_chart(_td_error_fig(td_errors, window), use_container_width=True)
            st.caption(
                "Falling TD error indicates the critic is getting more accurate — "
                "the advantage estimates used by the actor become more informative over time."
            )

        early = max(1, n_episodes // 5)
        col_a, col_b, col_c = st.columns(3)
        with col_a:
            st.metric("Mean return (first 20%)", f"{np.mean(returns[:early]):.1f}")
        with col_b:
            st.metric("Mean return (last 20%)", f"{np.mean(returns[-early:]):.1f}")
        with col_c:
            st.metric("Final mean |δ|", f"{np.mean(td_errors[-early:]):.3f}")

        st.header("Key Takeaways")
        final_mean = float(np.mean(returns[-early:]))
        if final_mean >= 400:
            st.success(
                f"Actor-Critic converged ({final_mean:.0f} mean return).  "
                "One-step bootstrapping typically produces a smoother learning curve than "
                "pure Monte Carlo REINFORCE by reducing gradient variance."
            )
        else:
            st.warning(
                f"Still training ({final_mean:.0f} mean return).  "
                "One-step Actor-Critic is sensitive to the ratio α_actor / α_critic.  "
                "Try α_critic 2–5× larger than α_actor."
            )
        st.info(
            "**Bias vs variance**: compared with REINFORCE (page 31), Actor-Critic typically "
            "converges faster but to a slightly worse policy — the bootstrapped TD estimate "
            "introduces bias.  Page 34 (PPO) addresses the stability of larger updates."
        )
