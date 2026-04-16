import numpy as np
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from utils.blackjack import (
    BlackjackEnv, apply_card, simulate_dealer,
    CARD_POOL, CARD_LABEL,
)


# ── Cached training (extended) ─────────────────────────────────────────────

@st.cache_data
def train_extended(n_episodes: int, seed: int) -> dict:
    """
    On-policy first-visit MC control — fixed ε=0.05, γ=1.0 for convergence.
    Returns Q, win/draw/loss per episode, and rolling win rate.
    """
    rng = np.random.default_rng(seed)
    env = BlackjackEnv(natural=True)
    env.seed(seed)

    epsilon = 0.05
    gamma   = 1.0

    Q = np.zeros((10, 10, 2, 2))
    N = np.zeros((10, 10, 2, 2), dtype=np.int32)

    outcomes = np.zeros(n_episodes)   # +1 win, 0 draw, -1 loss

    for ep in range(n_episodes):
        state = env.reset()
        episode = []
        done = False

        while not done:
            p, d, ua = state
            if not (12 <= p <= 21):
                state, reward, done = env.step(0)
                episode.append(((0, 0, 0), 0, reward))
                continue

            idx    = (p - 12, d - 1, int(ua))
            action = (int(rng.integers(0, 2))
                      if rng.random() < epsilon
                      else int(np.argmax(Q[idx])))

            state, reward, done = env.step(action)
            episode.append((idx, action, reward))

        G = 0.0
        first_visit: dict = {}
        for t in range(len(episode) - 1, -1, -1):
            idx, action, r = episode[t]
            G = r + gamma * G
            first_visit[(idx, action)] = G

        for (idx, action), G_sa in first_visit.items():
            pi, di, uai = idx
            N[pi, di, uai, action] += 1
            Q[pi, di, uai, action] += (
                (G_sa - Q[pi, di, uai, action]) / N[pi, di, uai, action]
            )

        final_r = episode[-1][2] if episode else 0.0
        outcomes[ep] = 1.0 if final_r > 0 else (-1.0 if final_r < 0 else 0.0)

    return {"Q": Q, "outcomes": outcomes}


# ── Plotting helpers ───────────────────────────────────────────────────────

_DEALER_LABELS = ["A", "2", "3", "4", "5", "6", "7", "8", "9", "10"]
_PLAYER_LABELS = [str(s) for s in range(12, 22)]


def _policy_heatmap(Q: np.ndarray, usable_ace: int, title: str):
    """
    Colour each cell green (stand) or red (hit) based on argmax Q.
    Text shows the action name.
    """
    policy = np.argmax(Q[:, :, usable_ace, :], axis=-1)   # (10,10): 0=stand,1=hit
    z      = policy.astype(float)

    text = [["Stand" if policy[r][c] == 0 else "Hit"
             for c in range(10)] for r in range(10)]

    fig = go.Figure(data=go.Heatmap(
        z=z,
        x=_DEALER_LABELS,
        y=_PLAYER_LABELS,
        colorscale=[[0, "#2ecc71"], [1, "#e74c3c"]],  # green=stand, red=hit
        showscale=False,
        zmin=0, zmax=1,
        text=text,
        texttemplate="%{text}",
        textfont=dict(size=11, color="white"),
    ))
    fig.update_layout(
        title=title,
        xaxis_title="Dealer showing",
        yaxis_title="Player sum",
        height=400,
        template="plotly_white",
    )
    return fig


# ── Page ───────────────────────────────────────────────────────────────────

def show():
    st.title("Solving Blackjack")
    st.markdown("**Section 3 — Monte Carlo Methods**")

    # ── Concept ───────────────────────────────────────────────────────────
    st.header("What Does 'Solved' Look Like?")
    st.markdown(
        """
On page 8, I ran 100k episodes and the Q-values started taking shape.
But Blackjack has a large state space — 10 × 10 × 2 = 200 states, each with 2 actions —
and many (state, action) pairs are visited rarely. The estimates are still noisy.

With enough episodes — typically 500k or more — the MC estimates converge and the learned
policy stabilises into something recognisable: it closely approximates the **basic strategy**
that professional gamblers use.

The basic strategy tells you exactly when to hit and when to stand, based purely on your
sum and the dealer's upcard. It's been proven optimal for the standard Blackjack rules.
The question is: can a learning agent discover it from scratch, with no prior knowledge,
just by playing many hands?
"""
    )

    col1, col2 = st.columns(2)
    with col1:
        st.info(
            "**More episodes → better policy.** Each episode adds one data point per "
            "(state, action) pair visited. Rare state-action pairs — like player sum 20 "
            "with a usable ace — take longer to get good estimates."
        )
    with col2:
        st.warning(
            "**High variance is MC's weakness.** Each return is a single noisy sample of "
            "the true Q-value. With ε-greedy exploration (ε=0.05 here), some episodes "
            "will take suboptimal actions that add noise to the estimates."
        )

    with st.expander("Deep Dive — Bias, Variance, and Why More Episodes Help"):
        st.markdown(
            r"""
The MC estimate of $Q(s, a)$ is an **unbiased** estimator of the true $Q^\pi(s,a)$ under the
behaviour policy — each sample return $G_t$ is an unbiased sample. But it has **high variance**:
different episodes visiting the same (s,a) return very different $G_t$ values depending on
subsequent random actions and dealer behaviour.

Variance reduces as $1/N(s,a)$ — the reciprocal of visit count. So:
- After 10 visits: high variance, estimate is noisy
- After 1,000 visits: much tighter estimate
- After 100,000 visits: nearly converged

This is why 500k+ episodes matter. The rare states (high sums, usable ace) simply need many
more total episodes before their specific (state, action) pairs are visited enough times.
"""
        )

    st.divider()

    # ── Training ──────────────────────────────────────────────────────────
    st.header("Training")

    st.markdown(
        "Choose how many hands to train on. **Learning outcome:** at 100k episodes the policy "
        "is still noisy in rare states; at 500k–1M it converges toward the known basic strategy "
        "— compare the heatmaps to what a professional Blackjack chart recommends."
    )

    col1, col2 = st.columns(2)
    with col1:
        n_episodes = st.select_slider(
            "Number of episodes",
            options=[100_000, 250_000, 500_000, 750_000, 1_000_000],
            value=500_000,
            format_func=lambda x: f"{x:,}",
            help="More episodes → rarer (state, action) pairs like 'player 20 with usable ace' get visited enough to converge.",
        )
    with col2:
        seed = st.number_input("Random seed", value=42, step=1)

    st.caption("Fixed: ε = 0.05, γ = 1.0 — standard settings for Blackjack convergence.")

    if st.button("Train Agent", type="primary"):
        with st.spinner(f"Running {n_episodes:,} episodes… (cached after first run)"):
            result = train_extended(n_episodes, int(seed))

        Q        = result["Q"]
        outcomes = result["outcomes"]

        st.session_state["bj_Q"]        = Q
        st.session_state["bj_outcomes"] = outcomes
        st.session_state["bj_n_ep"]     = n_episodes

    if "bj_Q" not in st.session_state:
        st.info("Run the training above to see results.")
        return

    Q        = st.session_state["bj_Q"]
    outcomes = st.session_state["bj_outcomes"]
    n_ep     = st.session_state["bj_n_ep"]

    # ── Results ───────────────────────────────────────────────────────────
    st.header("Results")

    # Policy heatmaps
    st.subheader("Learned Policy")
    st.markdown(
        "Green = **Stand**, Red = **Hit**. "
        "Each cell shows what the agent does at that (player sum, dealer card) combination."
    )
    col_l, col_r = st.columns(2)
    with col_l:
        st.plotly_chart(
            _policy_heatmap(Q, usable_ace=0, title="No Usable Ace"),
            use_container_width=True,
        )
    with col_r:
        st.plotly_chart(
            _policy_heatmap(Q, usable_ace=1, title="Usable Ace"),
            use_container_width=True,
        )

    st.info(
        "**What to look for:** Without a usable ace, the agent should stand on 17–21 "
        "against almost any dealer card, and stand on 12–16 when the dealer shows a weak "
        "card (2–6). With a usable ace, it should be more aggressive — hitting on soft hands "
        "up to soft 18 is often correct."
    )

    # Win rate curve
    st.subheader("Win Rate Over Training")
    window = max(5_000, n_ep // 100)
    wins   = (outcomes > 0).astype(float)
    rolling = np.convolve(wins, np.ones(window) / window, mode="valid")

    fig_wr = go.Figure()
    fig_wr.add_trace(go.Scatter(
        x=list(range(window, n_ep + 1)),
        y=rolling,
        mode="lines",
        line=dict(color="#00CC96", width=1.5),
        name=f"Win rate (window={window:,})",
    ))
    fig_wr.add_hline(
        y=float(wins.mean()), line_dash="dash", line_color="gray",
        annotation_text=f"Overall avg: {wins.mean():.1%}",
    )
    fig_wr.update_layout(
        title=f"Rolling Win Rate (window = {window:,} episodes)",
        xaxis_title="Episode",
        yaxis_title="Win rate",
        template="plotly_white",
        height=360,
        yaxis=dict(range=[0.25, 0.65]),
    )
    st.plotly_chart(fig_wr, use_container_width=True)

    # Outcome summary
    last_10 = slice(int(n_ep * 0.9), None)
    win_r  = float((outcomes[last_10] > 0).mean())
    loss_r = float((outcomes[last_10] < 0).mean())
    draw_r = float((outcomes[last_10] == 0).mean())

    col_a, col_b, col_c = st.columns(3)
    with col_a:
        st.metric("Win rate (last 10%)", f"{win_r:.1%}")
    with col_b:
        st.metric("Loss rate (last 10%)", f"{loss_r:.1%}")
    with col_c:
        st.metric("Draw rate (last 10%)", f"{draw_r:.1%}")

    st.success(
        "A win rate around 42–44% is expected — Blackjack has a slight house edge even "
        "with optimal play (~0.5% depending on rules). The agent converging to ~42% "
        "win rate suggests it has found a near-optimal strategy."
    )

    # ── Play vs Agent ──────────────────────────────────────────────────────
    st.divider()
    st.header("Play vs Agent — Interactive Demo")
    st.markdown(
        """
Set your starting hand and watch the trained agent play it out, with the reasoning
behind each decision shown step by step.
"""
    )

    col1, col2, col3 = st.columns(3)
    with col1:
        player_sum = st.selectbox(
            "Your starting sum",
            list(range(12, 22)),
            index=4,      # default = 16
            help="12–21. We'll build a compatible hand automatically.",
        )
    with col2:
        dealer_label = st.selectbox(
            "Dealer's upcard",
            ["A", "2", "3", "4", "5", "6", "7", "8", "9", "10"],
            index=5,      # default = 6
        )
    with col3:
        usable_ace = st.checkbox(
            "Starting hand has a usable ace",
            value=False,
            help="Usable ace = ace counted as 11 without busting",
        )

    dealer_card = 1 if dealer_label == "A" else int(dealer_label)

    if st.button("▶ Play the Hand", type="primary"):
        _play_demo(Q, player_sum, dealer_card, usable_ace)

    # ── Section summary ────────────────────────────────────────────────────
    st.divider()
    st.header("When to Use Monte Carlo")
    col1, col2 = st.columns(2)
    with col1:
        st.success(
            "**MC is the right tool when:**\n\n"
            "- The task is **episodic** — episodes end, and I can compute returns\n"
            "- I have **no model** of the environment and can only learn from experience\n"
            "- The environment is **complex enough** that DP is intractable (too many states)\n"
            "- I want a **simple, unbiased** estimator — no bootstrapping, no model error"
        )
    with col2:
        st.warning(
            "**MC struggles when:**\n\n"
            "- Tasks are **continuing** (no natural episode end) — returns are infinite\n"
            "- Episodes are **very long** — returns are delayed and variance is high\n"
            "- I need to learn **online** mid-episode rather than waiting for the end\n"
            "- The state space is large and many states are **rarely visited**\n\n"
            "**Temporal Difference (TD) methods** fix most of these — they bootstrap "
            "mid-episode and don't need complete returns."
        )


# ── Demo helper ────────────────────────────────────────────────────────────

def _play_demo(Q: np.ndarray, start_sum: int, dealer_card: int, usable_ace: bool):
    """Step through a hand using the trained agent, showing Q-value reasoning."""
    rng = np.random.default_rng()   # fresh RNG each demo run

    p_sum = start_sum
    ua    = usable_ace
    steps = []
    done  = False

    while not done:
        if not (12 <= p_sum <= 21):
            break

        idx    = (p_sum - 12, dealer_card - 1, int(ua))
        q_vals = Q[idx]                        # [Q(stand), Q(hit)]
        action = int(np.argmax(q_vals))        # greedy (no ε in demo)

        step_info = {
            "p_sum":     p_sum,
            "ua":        ua,
            "action":    action,
            "q_stand":   float(q_vals[0]),
            "q_hit":     float(q_vals[1]),
            "card_drawn": None,
        }

        if action == 1:   # hit
            rank        = int(rng.choice(CARD_POOL))
            card_val    = min(rank, 10)
            card_name   = CARD_LABEL[card_val]
            new_sum, new_ua = apply_card(p_sum, ua, rank)
            step_info["card_drawn"] = card_name
            step_info["new_sum"]    = new_sum
            steps.append(step_info)
            if new_sum > 21:
                done = True
                steps[-1]["result"] = "bust"
            else:
                p_sum, ua = new_sum, new_ua
        else:             # stand
            steps.append(step_info)
            done = True

    # Determine outcome
    if steps and steps[-1].get("result") == "bust":
        outcome = "bust"
        d_total = None
    else:
        # Simulate dealer
        d_total = simulate_dealer(dealer_card, rng)
        if d_total > 21 or p_sum > d_total:
            outcome = "win"
        elif p_sum == d_total:
            outcome = "draw"
        else:
            outcome = "loss"

    # ── Display ────────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown(
        f"**Starting hand:** Sum = **{start_sum}**"
        + (" (usable ace)" if usable_ace else "")
        + f" | Dealer shows **{CARD_LABEL.get(dealer_card, str(dealer_card))}**"
    )

    for i, step in enumerate(steps):
        action_name = "Hit" if step["action"] == 1 else "Stand"
        q_winner    = "Q(hit)" if step["action"] == 1 else "Q(stand)"
        q_loser     = "Q(stand)" if step["action"] == 1 else "Q(hit)"
        q_win_val   = step["q_hit"] if step["action"] == 1 else step["q_stand"]
        q_lose_val  = step["q_stand"] if step["action"] == 1 else step["q_hit"]

        ace_str = " (usable ace)" if step["ua"] else ""
        st.markdown(f"**Step {i+1}:** Player sum = **{step['p_sum']}**{ace_str}")

        reasoning = (
            f"Agent **{action_name}s** — "
            f"{q_winner} = {q_win_val:.3f} > {q_loser} = {q_lose_val:.3f}"
        )

        if step["action"] == 1 and step.get("card_drawn"):
            if step.get("result") == "bust":
                st.error(
                    f"{reasoning}\n\n"
                    f"Drew **{step['card_drawn']}** → new sum = **{step['new_sum']}** — **Bust!**"
                )
            else:
                st.info(
                    f"{reasoning}\n\n"
                    f"Drew **{step['card_drawn']}** → new sum = **{step['new_sum']}**"
                    + (" (usable ace)" if step.get("new_ua") else "")
                )
        else:
            st.info(reasoning)

    # Final outcome
    st.markdown(f"**Dealer:** final total = **{d_total if d_total else '—'}**")
    if outcome == "win":
        st.success(f"Result: **Win** — player {p_sum} > dealer {d_total}")
    elif outcome == "draw":
        st.warning(f"Result: **Draw** — player {p_sum} = dealer {d_total}")
    elif outcome == "loss":
        st.error(f"Result: **Loss** — player {p_sum} < dealer {d_total}")
    else:
        st.error("Result: **Bust** — player went over 21")
