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


def _shared_net(n_obs: int, n_actions: int, hidden: int):
    """Shared trunk + separate actor/critic heads."""

    class ActorCritic(nn.Module):
        def __init__(self):
            super().__init__()
            self.trunk = nn.Sequential(
                nn.Linear(n_obs, hidden),
                nn.Tanh(),
                nn.Linear(hidden, hidden),
                nn.Tanh(),
            )
            self.actor = nn.Linear(hidden, n_actions)
            self.critic = nn.Linear(hidden, 1)

        def forward(self, x):
            h = self.trunk(x)
            return self.actor(h), self.critic(h).squeeze(-1)

        def get_log_prob_and_value(self, obs, action):
            logits, value = self(obs)
            dist = torch.distributions.Categorical(logits=logits)
            log_p = dist.log_prob(action)
            entropy = dist.entropy()
            return log_p, value, entropy

    return ActorCritic()


# ── Training ───────────────────────────────────────────────────────────────────


@st.cache_data
def run_ppo(
    hidden: int,
    lr: float,
    gamma: float,
    lam: float,
    clip_eps: float,
    n_epochs: int,
    batch_steps: int,
    n_updates: int,
    entropy_coef: float,
    seed: int,
) -> tuple[list, list]:
    """
    Simplified PPO on CartPole-v1.
    Collects batch_steps transitions, updates for n_epochs with clipped surrogate.
    Returns (episode_returns, smoothed).
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    env = gym.make("CartPole-v1")
    net = _shared_net(env.observation_space.shape[0], env.action_space.n, hidden)
    opt = optim.Adam(net.parameters(), lr=lr)

    all_returns: list[float] = []
    ep_reward = 0.0
    obs, _ = env.reset(seed=seed)

    for _ in range(n_updates):
        # ── Collect rollout ───────────────────────────────────────────────────
        obs_buf, act_buf, rew_buf, done_buf, logp_buf, val_buf = [], [], [], [], [], []

        for _ in range(batch_steps):
            obs_t = torch.FloatTensor(obs)
            with torch.no_grad():
                logits, value = net(obs_t)
                dist = torch.distributions.Categorical(logits=logits)
                action = dist.sample()
                log_p = dist.log_prob(action)

            obs_buf.append(obs_t)
            act_buf.append(action)
            logp_buf.append(log_p)
            val_buf.append(value)

            obs, r, terminated, truncated, _ = env.step(action.item())
            done = terminated or truncated
            rew_buf.append(float(r))
            done_buf.append(float(done))
            ep_reward += float(r)

            if done:
                all_returns.append(ep_reward)
                ep_reward = 0.0
                obs, _ = env.reset()

        # ── Compute GAE advantages ────────────────────────────────────────────
        with torch.no_grad():
            last_val = net(torch.FloatTensor(obs))[1].item()

        advantages = np.zeros(batch_steps, dtype=np.float32)
        returns_buf = np.zeros(batch_steps, dtype=np.float32)
        gae = 0.0
        vals_np = np.array([v.item() for v in val_buf] + [last_val])
        for t in reversed(range(batch_steps)):
            next_non_term = 1.0 - done_buf[t]
            delta = rew_buf[t] + gamma * vals_np[t + 1] * next_non_term - vals_np[t]
            gae = delta + gamma * lam * next_non_term * gae
            advantages[t] = gae
            returns_buf[t] = gae + vals_np[t]

        obs_t = torch.stack(obs_buf)
        act_t = torch.stack(act_buf)
        old_logp = torch.stack(logp_buf).detach()
        adv_t = torch.FloatTensor(advantages)
        ret_t = torch.FloatTensor(returns_buf)
        adv_t = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)

        # ── PPO update epochs ────────────────────────────────────────────────
        for _ in range(n_epochs):
            log_p, value, entropy = net.get_log_prob_and_value(obs_t, act_t)
            ratio = torch.exp(log_p - old_logp)
            surr1 = ratio * adv_t
            surr2 = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * adv_t
            actor_loss = -torch.min(surr1, surr2).mean()
            critic_loss = nn.functional.mse_loss(value, ret_t)
            loss = actor_loss + 0.5 * critic_loss - entropy_coef * entropy.mean()
            opt.zero_grad()
            loss.backward()
            opt.step()

    env.close()
    if not all_returns:
        all_returns = [0.0]
    w = max(1, min(20, len(all_returns) // 10))
    sm = np.convolve(all_returns, np.ones(w) / w, mode="valid").tolist()
    return all_returns, [None] * (w - 1) + sm


# ── Visualisations ─────────────────────────────────────────────────────────────


def _clipping_fig(clip_eps: float) -> go.Figure:
    r = np.linspace(0.0, 2.5, 400)
    fig = go.Figure()
    colours = {"A > 0 (good action)": "#00CC96", "A < 0 (bad action)": "#EF553B"}

    for label, A in [("A > 0 (good action)", 1.0), ("A < 0 (bad action)", -1.0)]:
        surr_unclipped = r * A
        surr_clipped = np.clip(r, 1 - clip_eps, 1 + clip_eps) * A
        ppo_obj = np.minimum(surr_unclipped, surr_clipped)
        col = colours[label]
        fig.add_trace(
            go.Scatter(
                x=r.tolist(),
                y=surr_unclipped.tolist(),
                mode="lines",
                name=f"Unclipped ({label})",
                line=dict(color=col, width=1, dash="dot"),
                opacity=0.5,
            )
        )
        fig.add_trace(
            go.Scatter(
                x=r.tolist(),
                y=ppo_obj.tolist(),
                mode="lines",
                name=f"PPO objective ({label})",
                line=dict(color=col, width=2),
            )
        )

    fig.add_vline(
        x=1 - clip_eps,
        line_dash="dash",
        line_color="grey",
        annotation_text=f"1−ε={1-clip_eps:.2f}",
        annotation_position="top left",
    )
    fig.add_vline(
        x=1 + clip_eps,
        line_dash="dash",
        line_color="grey",
        annotation_text=f"1+ε={1+clip_eps:.2f}",
        annotation_position="top right",
    )
    fig.add_vline(x=1.0, line_dash="solid", line_color="black", line_width=1)
    fig.update_layout(
        title="PPO Clipped Surrogate Objective  (normalised |A| = 1)",
        xaxis_title="Probability ratio  r = π_new / π_old",
        yaxis_title="Objective value",
        template="plotly_white",
        height=420,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


def _returns_fig(returns: list, smooth: list) -> go.Figure:
    eps = list(range(1, len(returns) + 1))
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=eps,
            y=returns,
            mode="lines",
            name="Return",
            line=dict(color="#AB63FA", width=1),
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
                line=dict(color="#7B2D8B", width=2),
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
        title="PPO — Episode Return",
        xaxis_title="Episode",
        yaxis_title="Return",
        yaxis=dict(range=[0, 520]),
        template="plotly_white",
        height=350,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


# ── Page ───────────────────────────────────────────────────────────────────────


def show():
    st.title("PPO — Proximal Policy Optimisation")
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
    st.header("Stable Policy Updates via Clipping")
    st.markdown(r"""
Vanilla policy gradient methods suffer from a destructive update problem: a single large
gradient step can collapse the policy to a degenerate distribution, and the next update
may be based on data collected under that degenerate policy.

**TRPO** (Trust Region Policy Optimisation, Schulman et al. 2015) constrains updates to a
trust region defined by a KL divergence limit, requiring a second-order optimisation step.

**PPO** (Proximal Policy Optimisation, Schulman et al. 2017) achieves similar stability
using a simple first-order trick: the **clipped surrogate objective**.

Define the probability ratio $r_t(\theta) = \pi_\theta(A_t|S_t) / \pi_{\theta_\text{old}}(A_t|S_t)$.
The PPO objective is:

$$L^\text{CLIP}(\theta) = \mathbb{E}_t\!\left[\min\!\left(r_t(\theta)\,A_t,\;
\text{clip}(r_t(\theta),\,1-\varepsilon,\,1+\varepsilon)\,A_t\right)\right]$$

The $\min$ pessimistically clips the objective: large policy changes are penalised by
refusing to take credit for them.
""")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**When the advantage is positive** ($A_t > 0$)")
        st.markdown(
            "The action was good.  We want to increase $r_t$ (make the action more likely), "
            "but the objective is flat above $1+\\varepsilon$ — "
            "we get no gradient for pushing the ratio further."
        )
    with col2:
        st.markdown("**When the advantage is negative** ($A_t < 0$)")
        st.markdown(
            "The action was bad.  We want to decrease $r_t$, "
            "but the objective is flat below $1-\\varepsilon$ — "
            "the gradient signal cuts off once the action is sufficiently suppressed."
        )

    st.info(
        "**PPO is the most widely used policy gradient algorithm in practice.** "
        "It combines the stability of TRPO with the simplicity of vanilla policy gradient.  "
        "ChatGPT and most modern RLHF systems use PPO or close variants."
    )

    with st.expander("GAE — Generalised Advantage Estimation"):
        st.markdown(r"""
PPO uses **GAE** (Schulman et al. 2015b) to estimate the advantage $A_t$:

$$\hat{A}_t = \sum_{l=0}^{\infty} (\gamma\lambda)^l\,\delta_{t+l}
\quad\text{where}\quad
\delta_t = R_{t+1} + \gamma V(S_{t+1}) - V(S_t)$$

GAE is a weighted average of $n$-step TD errors — analogous to TD(λ) but for the
advantage.  The parameter $\lambda$ controls the bias–variance tradeoff:
- $\lambda = 0$: pure one-step TD advantage (low variance, higher bias)
- $\lambda = 1$: Monte Carlo advantage (high variance, lower bias)
Typical values: $\lambda = 0.95$, $\gamma = 0.99$.
""")

    with st.expander("TRPO vs PPO"):
        st.markdown(r"""
**TRPO** optimises the surrogate objective subject to a hard constraint:

$$\max_\theta\, L^\text{surr}(\theta) \quad\text{s.t.}\quad
\mathbb{E}_t\!\left[D_\text{KL}(\pi_{\theta_\text{old}}(\cdot|S_t) \| \pi_\theta(\cdot|S_t))\right] \leq \delta$$

This requires computing the Fisher information matrix (or an approximation via conjugate
gradient), making it expensive.

**PPO** achieves similar empirical performance with a first-order update and simple
gradient descent, making it ~10× faster to implement and run.

In practice, PPO often matches or outperforms TRPO on continuous control benchmarks
(MuJoCo) while being far simpler to tune and debug.
""")

    st.divider()

    # ── Clipping visualisation ─────────────────────────────────────────────────
    st.header("Clipping Function Explorer")
    st.markdown(
        "Adjust ε to see how the clipped objective shapes the gradient signal for good "
        "and bad actions.  The flat regions are where gradient flow is cut off."
    )
    clip_eps_viz = st.slider("ε (clip range)", 0.05, 0.50, 0.20, 0.05)
    st.plotly_chart(_clipping_fig(clip_eps_viz), use_container_width=True)

    st.divider()

    # ── Simulation ─────────────────────────────────────────────────────────────
    st.header("Interactive Simulation — CartPole-v1")
    st.markdown(
        "PPO with a shared actor-critic network and GAE advantages.  "
        "The agent collects a batch of transitions, then updates for multiple epochs on that batch."
    )

    col1, col2 = st.columns(2)
    with col1:
        hidden = st.slider("Hidden units per layer", 16, 256, 64, 16)
        lr = st.slider("Learning rate", 1e-4, 5e-3, 3e-4, 1e-4, format="%.4f")
        clip_eps = st.slider("ε (clip range)", 0.05, 0.40, 0.20, 0.05)
        entropy_coef = st.slider(
            "Entropy coefficient", 0.0, 0.05, 0.01, 0.005, format="%.3f"
        )
    with col2:
        gamma = st.slider("γ (discount)", 0.90, 1.00, 0.99, 0.01)
        lam = st.slider("λ (GAE)", 0.80, 1.00, 0.95, 0.01)
        n_epochs = st.slider("Epochs per update", 1, 10, 4)
        batch_steps = st.select_slider(
            "Steps per batch", [64, 128, 256, 512], value=256
        )
        n_updates = st.slider("Number of updates", 10, 200, 80, 10)
        seed = st.number_input("Random seed", value=42, step=1)

    if st.button("Train PPO", type="primary"):
        with st.spinner("Running PPO on CartPole-v1…"):
            returns, smooth = run_ppo(
                hidden,
                lr,
                float(gamma),
                float(lam),
                clip_eps,
                n_epochs,
                batch_steps,
                n_updates,
                entropy_coef,
                int(seed),
            )

        st.header("Results")
        st.plotly_chart(_returns_fig(returns, smooth), use_container_width=True)
        st.caption(f"Total episodes completed: {len(returns)}")

        if len(returns) >= 10:
            early = max(1, len(returns) // 5)
            col_a, col_b = st.columns(2)
            with col_a:
                st.metric("Mean return (first 20%)", f"{np.mean(returns[:early]):.1f}")
            with col_b:
                st.metric("Mean return (last 20%)", f"{np.mean(returns[-early:]):.1f}")

            final_mean = float(np.mean(returns[-early:]))
            st.header("Key Takeaways")
            if final_mean >= 400:
                st.success(
                    f"PPO converged ({final_mean:.0f} mean return).  "
                    "Notice the much smoother learning curve compared with REINFORCE — "
                    "the clipped objective prevents catastrophic policy collapses."
                )
            else:
                st.warning(
                    f"Still training ({final_mean:.0f} mean return).  "
                    "Try more updates, a larger batch, or reduce ε slightly."
                )
        st.info(
            "**Clipping in action**: increase ε toward 0.4 and watch for instability "
            "as large policy changes are allowed.  Decrease it toward 0.05 and training "
            "slows as the trust region becomes very tight.  ε ≈ 0.1–0.2 is typical."
        )
