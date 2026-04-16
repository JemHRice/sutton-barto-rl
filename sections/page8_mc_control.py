import numpy as np
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from utils.blackjack import BlackjackEnv


# ── Cached training ────────────────────────────────────────────────────────

@st.cache_data
def train_mc_control(
    n_episodes: int,
    epsilon: float,
    gamma: float,
    seed: int,
) -> dict:
    """
    On-policy first-visit MC control with ε-greedy exploration.

    Q-table axes: [player_sum−12, dealer_card−1, usable_ace, action]
      player_sum : 12–21  → index 0–9
      dealer_card: 1–10   → index 0–9
      usable_ace : False/True → 0/1
      action     : stand/hit  → 0/1

    Returns Q, N, and the reward for every episode.
    """
    rng = np.random.default_rng(seed)
    env = BlackjackEnv(natural=True)
    env.seed(seed)

    Q = np.zeros((10, 10, 2, 2))    # (player, dealer, ace, action)
    N = np.zeros((10, 10, 2, 2), dtype=np.int32)
    rewards = np.zeros(n_episodes)

    for ep in range(n_episodes):
        state = env.reset()
        episode = []     # list of (state_idx, action, reward)
        done = False

        while not done:
            p, d, ua = state
            if not (12 <= p <= 21):
                # outside tracked state space — force stand
                state, reward, done = env.step(0)
                episode.append(((0, 0, 0), 0, reward))
                continue

            idx = (p - 12, d - 1, int(ua))

            # ε-greedy action selection
            if rng.random() < epsilon:
                action = int(rng.integers(0, 2))
            else:
                action = int(np.argmax(Q[idx]))

            state, reward, done = env.step(action)
            episode.append((idx, action, reward))

        # ── First-visit MC return update ───────────────────────────────
        G           = 0.0
        first_visit = {}     # (idx, action) → return at first occurrence

        for t in range(len(episode) - 1, -1, -1):
            idx, action, r = episode[t]
            G = r + gamma * G
            first_visit[(idx, action)] = G    # backward pass → keeps earliest

        for (idx, action), G_sa in first_visit.items():
            pi, di, uai = idx
            N[pi, di, uai, action] += 1
            Q[pi, di, uai, action] += (G_sa - Q[pi, di, uai, action]) / N[pi, di, uai, action]

        rewards[ep] = episode[-1][2] if episode else 0.0

    return {"Q": Q, "N": N, "rewards": rewards}


# ── Plotting helpers ───────────────────────────────────────────────────────

_DEALER_LABELS = ["A", "2", "3", "4", "5", "6", "7", "8", "9", "10"]
_PLAYER_LABELS = [str(s) for s in range(12, 22)]


def _q_heatmap(Q: np.ndarray, usable_ace: int, action: int, title: str):
    """
    Return a Plotly heatmap of Q[player, dealer, usable_ace, action].
    Rows = player_sum (12→21), columns = dealer_card (A→10).
    """
    z = Q[:, :, usable_ace, action]          # shape (10, 10)
    fig = go.Figure(data=go.Heatmap(
        z=z,
        x=_DEALER_LABELS,
        y=_PLAYER_LABELS,
        colorscale="RdYlGn",
        colorbar=dict(title="Q"),
        text=[[f"{z[r][c]:.2f}" for c in range(10)] for r in range(10)],
        texttemplate="%{text}",
        textfont=dict(size=9),
    ))
    fig.update_layout(
        title=title,
        xaxis_title="Dealer showing",
        yaxis_title="Player sum",
        height=420,
        template="plotly_white",
    )
    return fig


# ── Page ───────────────────────────────────────────────────────────────────

def show():
    st.title("Monte Carlo Control")
    st.markdown("**Section 3 — Monte Carlo Methods**")

    # ── Concept ───────────────────────────────────────────────────────────
    st.header("Finding the Optimal Policy Without a Model")
    st.markdown(
        """
MC prediction told me how good a fixed policy is. Now I want to find the *best* policy —
without a model.

Here's the problem: in DP, when I wanted to improve a policy, I did a greedy update.
For each state I asked: "which action leads to the best next state?"
But to answer that, I needed to know *where each action takes me* — the transition function.
I don't have that here.

The fix is to shift from estimating **V(s)** (value of a state) to estimating **Q(s, a)**
(value of taking action $a$ from state $s$). If I know Q, I can pick the best action
without knowing where it leads:
"""
    )
    st.latex(r"\pi(s) = \arg\max_a\; Q(s, a)")
    st.markdown(
        """
The Q-value for a (state, action) pair is updated the same way as before — I track the
return I collected from each (state, action) visit and average over episodes:
"""
    )
    st.latex(
        r"Q(s, a) \;\leftarrow\; Q(s, a) + \frac{1}{N(s,a)}\bigl[G_t - Q(s,a)\bigr]"
    )
    st.markdown(
        """
The full MC control loop is:

1. **Generate an episode** using my current ε-greedy policy
2. **Work backwards** through the episode computing returns
3. **Update Q(s, a)** at every first visit using the incremental average
4. **Improve the policy** greedily: π(s) = argmax Q(s, a)
5. Repeat

### Why ε-Greedy? — Exploration Is Non-Negotiable

If I always act greedily, I'll never visit (state, action) pairs that my current policy avoids.
Those Q-values will stay at their initial estimates and I'll never know if they were good.
ε-greedy ensures I keep visiting every (state, action) pair — the estimates for all actions
will eventually converge.

This is **on-policy** control: I'm using the same policy I'm learning about to generate experience.
"""
    )

    col1, col2 = st.columns(2)
    with col1:
        st.info(
            "**On-policy:** I generate data using the policy I'm trying to improve — the same ε-greedy "
            "policy. I learn about the ε-greedy policy, not the fully greedy one."
        )
    with col2:
        st.warning(
            "**Off-policy** (preview): I could instead generate data with a separate *behaviour* policy "
            "(e.g., random) and learn about the *target* greedy policy. This uses importance sampling "
            "and is more complex but more flexible."
        )

    with st.expander("Deep Dive — Why Q and Not V?"):
        st.markdown(
            r"""
With a model, the greedy improvement step from $V$ works:

$$\pi'(s) = \arg\max_a \sum_{s',r} p(s',r|s,a)\bigl[r + \gamma V(s')\bigr]$$

Without a model, I don't know $p(s', r \mid s, a)$. I can't compute that sum.

With $Q$, the improvement step becomes trivial:

$$\pi'(s) = \arg\max_a\; Q(s, a)$$

No model needed. $Q(s, a)$ already encodes the expected return after taking action $a$,
including all the downstream consequences. That's why Q-values — not V-values — are the
workhorse of model-free RL.
"""
        )

    st.divider()

    # ── Environment intro ─────────────────────────────────────────────────
    st.header("The Environment — Blackjack")
    st.markdown(
        """
**State:** (player sum, dealer's face-up card, whether I have a usable ace)

A **usable ace** is one I can count as 11 without busting. If I later draw a card
that would bust me, the ace flips to 1 automatically.

**Actions:** Stand (0) or Hit (1).

**Reward:** +1 for a win, 0 for a draw, −1 for a loss. A natural blackjack
(two-card 21 when the dealer doesn't also have it) pays +1.5.

**Rule:** The dealer always hits until their sum reaches 17 or higher.

I start with two cards; if my sum is below 12, I auto-hit until it reaches 12
(no meaningful decision below 12 — I can't bust on the next card).
"""
    )

    st.divider()

    # ── Simulation ────────────────────────────────────────────────────────
    st.header("Interactive Simulation")
    st.markdown(
        "Adjust the training parameters and run MC control on Blackjack. "
        "**Learning outcome:** at 100k episodes the Q-value heatmaps are noisy; at 500k they "
        "start to resemble the known optimal strategy — stand on high sums against any dealer "
        "card, and hit more aggressively when you have a usable ace."
    )

    col1, col2, col3 = st.columns(3)
    with col1:
        n_episodes = st.slider("Number of episodes", 10_000, 500_000, 100_000, 10_000,
                               help="Hands of Blackjack to play. More episodes → rarer (state, action) pairs get visited and Q-values converge.")
    with col2:
        epsilon = st.slider("ε (exploration)", 0.01, 0.5, 0.1, 0.01,
                            help="Probability of taking a random action. Ensures every (state, action) pair gets visited — without it, unvisited pairs never get Q-value estimates.")
    with col3:
        gamma = st.slider("γ (discount)", 0.5, 1.0, 1.0, 0.05,
                          help="Discount factor. γ = 1 is standard for Blackjack — episodes always end, so no need to discount future rewards.")

    seed = st.number_input("Random seed", value=42, step=1)

    if st.button("Train MC Control", type="primary"):
        with st.spinner(f"Running {n_episodes:,} episodes…"):
            result = train_mc_control(n_episodes, epsilon, gamma, int(seed))

        Q       = result["Q"]
        rewards = result["rewards"]

        st.header("Results")

        # ── Q-value heatmaps — no usable ace ──────────────────────────────
        st.subheader("Q-Values — No Usable Ace")
        col_l, col_r = st.columns(2)
        with col_l:
            st.plotly_chart(
                _q_heatmap(Q, usable_ace=0, action=0, title="Q(s, Stand) — No Usable Ace"),
                use_container_width=True,
            )
        with col_r:
            st.plotly_chart(
                _q_heatmap(Q, usable_ace=0, action=1, title="Q(s, Hit) — No Usable Ace"),
                use_container_width=True,
            )

        st.subheader("Q-Values — Usable Ace")
        col_l, col_r = st.columns(2)
        with col_l:
            st.plotly_chart(
                _q_heatmap(Q, usable_ace=1, action=0, title="Q(s, Stand) — Usable Ace"),
                use_container_width=True,
            )
        with col_r:
            st.plotly_chart(
                _q_heatmap(Q, usable_ace=1, action=1, title="Q(s, Hit) — Usable Ace"),
                use_container_width=True,
            )

        # ── Win rate curve ─────────────────────────────────────────────────
        st.subheader("Win Rate Over Training")
        window = max(1000, n_episodes // 100)
        wins = (rewards > 0).astype(float)
        rolling_win = np.convolve(wins, np.ones(window) / window, mode="valid")

        fig_wr = go.Figure()
        fig_wr.add_trace(go.Scatter(
            x=list(range(window, n_episodes + 1)),
            y=rolling_win,
            mode="lines",
            line=dict(color="#00CC96", width=2),
            name=f"Win rate (window={window:,})",
        ))
        fig_wr.update_layout(
            title=f"Rolling Win Rate (window = {window:,} episodes)",
            xaxis_title="Episode",
            yaxis_title="Win rate",
            template="plotly_white",
            height=360,
            yaxis=dict(range=[0, 1]),
        )
        st.plotly_chart(fig_wr, use_container_width=True)

        # ── Metrics ────────────────────────────────────────────────────────
        st.header("Key Takeaways")
        last_10pct = slice(int(n_episodes * 0.9), None)
        win_rate  = float((rewards[last_10pct] > 0).mean())
        loss_rate = float((rewards[last_10pct] < 0).mean())
        draw_rate = float((rewards[last_10pct] == 0).mean())

        col_a, col_b, col_c = st.columns(3)
        with col_a:
            st.metric("Win rate (last 10%)", f"{win_rate:.1%}")
        with col_b:
            st.metric("Loss rate (last 10%)", f"{loss_rate:.1%}")
        with col_c:
            st.metric("Draw rate (last 10%)", f"{draw_rate:.1%}")

        st.info(
            "The Q-value heatmaps show what the agent learned about the value of each "
            "action from every (player sum × dealer card) situation. "
            "Compare Stand vs Hit — the agent should learn to stand on high sums "
            "and hit aggressively when showing a usable ace."
        )
        st.success(
            f"After {n_episodes:,} episodes, the win rate is stabilising. "
            "Head to **Page 9** to run 500k+ episodes and extract the full optimal policy."
        )
