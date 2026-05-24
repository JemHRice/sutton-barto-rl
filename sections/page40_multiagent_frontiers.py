import numpy as np
import plotly.graph_objects as go
import streamlit as st

# ── Game theory constants ──────────────────────────────────────────────────────

_R, _P, _T, _S = 3, 1, 5, 0  # Reward, Punishment, Temptation, Sucker
_STRATEGIES = ["Always Defect", "Always Cooperate", "Tit-for-Tat", "Random (50%)"]
_COLOURS = ["#EF553B", "#00CC96", "#7B2D8B", "#FFA15A"]


def _move(strategy: int, opp_last: int, rng) -> int:
    """Return 1 (cooperate) or 0 (defect)."""
    if strategy == 0:
        return 0
    if strategy == 1:
        return 1
    if strategy == 2:
        return opp_last
    return int(rng.random() < 0.5)


def _match_payoff(si: int, sj: int, n_rounds: int, rng) -> float:
    """Average payoff per round for strategy si against strategy sj."""
    prev_i = prev_j = 1
    total = 0
    for _ in range(n_rounds):
        mi = _move(si, prev_j, rng)
        mj = _move(sj, prev_i, rng)
        if mi == 1 and mj == 1:
            total += _R
        elif mi == 0 and mj == 0:
            total += _P
        elif mi == 0 and mj == 1:
            total += _T
        else:
            total += _S
        prev_i, prev_j = mi, mj
    return total / n_rounds


def _compute_payoff_matrix(n_rounds: int = 20, n_trials: int = 100) -> np.ndarray:
    rng = np.random.default_rng(0)
    n = len(_STRATEGIES)
    M = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            M[i, j] = np.mean(
                [_match_payoff(i, j, n_rounds, rng) for _ in range(n_trials)]
            )
    return M


_PAYOFF_MATRIX = _compute_payoff_matrix()


# ── Simulation ─────────────────────────────────────────────────────────────────


@st.cache_data
def run_replicator(
    n_generations: int,
    mutation_rate: float,
    seed: int,
    initial_ad: float,
    initial_ac: float,
    initial_tft: float,
    initial_rand: float,
) -> list:
    """Replicator dynamics with mutation. Returns list of freq arrays (n_gen+1, 4)."""
    rng = np.random.default_rng(seed)
    freqs = np.array([initial_ad, initial_ac, initial_tft, initial_rand], dtype=float)
    freqs = np.clip(freqs, 1e-6, None)
    freqs /= freqs.sum()
    history = [freqs.copy().tolist()]

    for _ in range(n_generations):
        fitness = _PAYOFF_MATRIX @ freqs
        mean_fitness = float(freqs @ fitness)
        if mean_fitness > 1e-9:
            new_freqs = freqs * fitness / mean_fitness
        else:
            new_freqs = freqs.copy()
        new_freqs = (1 - mutation_rate) * new_freqs + mutation_rate / len(_STRATEGIES)
        new_freqs = np.clip(new_freqs, 0, None)
        freqs = new_freqs / new_freqs.sum()
        history.append(freqs.copy().tolist())

    return history


# ── Visualisations ─────────────────────────────────────────────────────────────


def _payoff_heatmap() -> go.Figure:
    M = _PAYOFF_MATRIX
    labels = ["Always<br>Defect", "Always<br>Coop.", "Tit-for-<br>Tat", "Random"]
    fig = go.Figure(
        go.Heatmap(
            z=M.tolist(),
            x=labels,
            y=labels,
            colorscale="RdYlGn",
            zmin=0,
            zmax=5,
            text=[[f"{v:.2f}" for v in row] for row in M],
            texttemplate="%{text}",
            showscale=True,
        )
    )
    fig.update_layout(
        title="Average Payoff per Round (row = my strategy, col = opponent)",
        xaxis_title="Opponent strategy",
        yaxis_title="My strategy",
        template="plotly_white",
        height=360,
        margin=dict(l=10, r=10, t=50, b=10),
    )
    return fig


def _evolution_fig(history: list) -> go.Figure:
    arr = np.array(history)
    gens = list(range(len(arr)))
    fig = go.Figure()
    for i, (name, color) in enumerate(zip(_STRATEGIES, _COLOURS)):
        fig.add_trace(
            go.Scatter(
                x=gens,
                y=arr[:, i].tolist(),
                mode="lines",
                name=name,
                line=dict(color=color, width=2),
                stackgroup="one",
            )
        )
    fig.update_layout(
        title="Strategy Frequency over Generations (replicator dynamics)",
        xaxis_title="Generation",
        yaxis_title="Population frequency",
        yaxis=dict(range=[0, 1]),
        template="plotly_white",
        height=380,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


# ── Page ───────────────────────────────────────────────────────────────────────


def show():
    st.title("Multi-Agent RL and Frontiers")
    st.markdown(
        '<div style="background:#7B2D8B;height:3px;'
        'border-radius:2px;margin-bottom:1rem;"></div>',
        unsafe_allow_html=True,
    )
    st.markdown("**Section 13 — Advanced Topics (Chapter 17)**")

    # ── Concept ───────────────────────────────────────────────────────────────
    st.header("Beyond Single-Agent RL")
    st.markdown(r"""
Single-agent RL assumes a stationary environment: the dynamics and reward function do not
change over time.  When multiple learning agents share an environment, this assumption fails —
each agent's policy change alters the effective dynamics experienced by all others.

Multi-agent RL (MARL) studies how agents can learn in the presence of other learners,
and encompasses three broad settings:
""")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("**Cooperative**")
        st.markdown(
            "Agents share a common reward and must coordinate to maximise it together.  "
            "Examples: multi-robot warehouse logistics, sensor network coordination.  "
            "Challenge: credit assignment — which agent caused the shared reward?"
        )
    with col2:
        st.markdown("**Competitive**")
        st.markdown(
            "Agents have opposing objectives (zero-sum or adversarial).  "
            "Examples: Go (AlphaGo), poker, adversarial robustness.  "
            "Challenge: arms race dynamics — agents co-evolve against each other."
        )
    with col3:
        st.markdown("**Mixed (general-sum)**")
        st.markdown(
            "Agents have partially overlapping interests.  "
            "Examples: traffic routing, negotiation, social dilemmas.  "
            "Challenge: cooperation may not be individually rational."
        )

    st.header("Game Theory Foundations")
    st.markdown(r"""
In a **normal-form game** each agent $i$ chooses a strategy from its action set and
receives a payoff that depends on *all* agents' actions simultaneously.

A **Nash equilibrium** (NE) is a joint strategy profile $(\pi_1^*, \pi_2^*, \ldots)$ such
that no agent can unilaterally improve its payoff by deviating:

$$J_i(\pi_i^*, \pi_{-i}^*) \geq J_i(\pi_i, \pi_{-i}^*) \quad \forall i,\, \forall \pi_i$$

Every finite game has at least one NE (possibly in mixed strategies — Nash 1950).  In RL,
self-play methods like those used in AlphaGo Zero aim to converge to an NE by letting
agents train against themselves.
""")

    with st.expander("Independent Q-Learning (IQL)"):
        st.markdown(r"""
The simplest multi-agent extension: each agent runs its own Q-learning algorithm,
treating other agents as part of the (non-stationary) environment.

**Problem**: the environment is no longer stationary from any single agent's perspective —
other agents' policies change, violating the Markov assumption Q-learning relies on.
Convergence guarantees break down.

In practice, IQL often works reasonably well when agents reach approximate equilibria,
but can cycle, diverge, or converge to suboptimal equilibria.  It scales easily to many
agents and requires no explicit communication between learners.
""")

    with st.expander("Centralised Training, Decentralised Execution (CTDE)"):
        st.markdown(r"""
CTDE resolves the non-stationarity problem during training: a centralised critic receives
the joint observation $(o_1, o_2, \ldots)$ and joint actions $(a_1, a_2, \ldots)$, giving
each agent's policy an accurate Q-estimate.  At deployment, each agent acts using only its
own local observation $o_i$.

Key CTDE algorithms:

- **QMIX** (Rashid et al. 2018): a monotonic mixing network combines individual Q-values
  into a joint Q-value, enabling efficient credit assignment.
- **MADDPG** (Lowe et al. 2017): each actor is conditioned on local observations; each
  critic sees the joint state and all agents' actions during training.
- **MAPPO**: applies PPO in the CTDE framework with a shared centralised value function.
""")

    st.divider()

    # ── PD section ────────────────────────────────────────────────────────────
    st.header("The Prisoner's Dilemma — A Social Dilemma")
    st.markdown(r"""
Two players simultaneously choose to **cooperate (C)** or **defect (D)**.
Payoffs follow the ordering $T > R > P > S$ with $2R > T + S$:

| | Opponent: C | Opponent: D |
|---|---|---|
| **My action: C** | R, R (mutual benefit) | S, T (exploited) |
| **My action: D** | T, S (exploit) | P, P (mutual loss) |

With $T=5, R=3, P=1, S=0$: defecting is always individually rational (it dominates C for any
fixed opponent action), yet mutual defection $(P=1, P=1)$ is worse for both than mutual
cooperation $(R=3, R=3)$.  This is the **social dilemma**: individual rationality leads to
collective irrationality.

In the **iterated** version (many rounds), cooperation can emerge: **Tit-for-Tat** (TFT)
cooperates initially and then copies the opponent's last move.  It is never exploited for long,
and rewards cooperation, making it a successful strategy in evolutionary tournaments.
""")

    st.plotly_chart(_payoff_heatmap(), use_container_width=True)
    st.caption(
        "Each cell shows the average payoff per round when the row strategy faces the column "
        "strategy over 20 rounds.  Tit-for-Tat earns R=3 against cooperators and itself, "
        "and limits losses against defectors by quickly retaliating."
    )

    st.divider()

    # ── Simulation ─────────────────────────────────────────────────────────────
    st.header("Evolutionary Dynamics — Replicator Equation")
    st.markdown(
        "Strategies compete in a population: each generation, strategies with above-average "
        "fitness grow in proportion, those with below-average fitness shrink.  "
        "Mutation prevents any strategy from dying out entirely.  "
        "Adjust the initial population mix and watch which strategies dominate."
    )

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Initial population fractions**")
        initial_ad = st.slider("Always Defect", 0.0, 1.0, 0.60, 0.05)
        initial_ac = st.slider("Always Cooperate", 0.0, 1.0, 0.10, 0.05)
        initial_tft = st.slider("Tit-for-Tat", 0.0, 1.0, 0.20, 0.05)
        initial_rand = st.slider("Random (50%)", 0.0, 1.0, 0.10, 0.05)
        total = initial_ad + initial_ac + initial_tft + initial_rand
        if abs(total - 1.0) > 0.05:
            st.warning(
                f"Fractions sum to {total:.2f} — they will be normalised automatically."
            )
    with col2:
        n_generations = st.slider("Generations", 50, 500, 200, 50)
        mutation_rate = st.slider(
            "Mutation rate", 0.0, 0.10, 0.01, 0.005, format="%.3f"
        )
        seed = st.number_input("Random seed", value=0, step=1)

    if st.button("Run evolutionary simulation", type="primary"):
        history = run_replicator(
            n_generations,
            mutation_rate,
            int(seed),
            initial_ad,
            initial_ac,
            initial_tft,
            initial_rand,
        )

        st.plotly_chart(_evolution_fig(history), use_container_width=True)

        final = np.array(history[-1])
        winner_idx = int(np.argmax(final))
        winner_name = _STRATEGIES[winner_idx]
        winner_freq = float(final[winner_idx])

        st.header("Key Takeaways")
        if winner_idx == 2 and winner_freq > 0.4:
            st.success(
                f"**Tit-for-Tat** dominated ({winner_freq:.0%} of population).  "
                "TFT resists exploitation because it immediately retaliates against defection "
                "and rewards cooperation — the evolutionary stable outcome when TFT starts "
                "with sufficient frequency to form cooperative clusters."
            )
        elif winner_idx == 0 and winner_freq > 0.4:
            st.warning(
                f"**Always Defect** dominated ({winner_freq:.0%}).  "
                "When defectors are too numerous, TFT clusters cannot form — cooperators are "
                "exploited before they can establish mutual-cooperation pairs.  "
                "Try increasing TFT's initial fraction above 20% to flip the outcome."
            )
        else:
            st.info(
                f"**{winner_name}** is most prevalent ({winner_freq:.0%}).  "
                "With mutation keeping all strategies alive, the dynamics depend critically "
                "on initial frequencies and the mutation rate."
            )
        st.info(
            "**RL connection**: self-play RL (AlphaGo, OpenAI Five, AlphaStar) is an "
            "evolutionary process — the agent trains against versions of itself, creating "
            "competitive pressure that drives improvement.  The policies that survive are "
            "approximate Nash equilibria of the game."
        )

    st.divider()

    # ── Frontiers ──────────────────────────────────────────────────────────────
    st.header("Open Frontiers in Reinforcement Learning")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Offline RL**")
        st.markdown(
            "Learning from a fixed logged dataset without any environment interaction.  "
            "Critical for safety-sensitive domains (healthcare, robotics) where online "
            "exploration is too risky or expensive.  Key challenge: *distributional shift* — "
            "the learned policy may visit states absent from the dataset, producing "
            "unreliable Q-estimates.  Conservative Q-Learning (CQL) and Decision Transformer "
            "are leading approaches."
        )
        st.markdown("**Reward Specification**")
        st.markdown(
            "Designing reward functions that capture human intent without unintended "
            "loopholes is extremely difficult.  Reward hacking — the agent finds unexpected "
            "ways to achieve high reward that violate the designer's intent — is a major "
            "failure mode.  RLHF (Reinforcement Learning from Human Feedback) learns a "
            "reward model from human preference comparisons rather than a hand-coded signal."
        )
    with col2:
        st.markdown("**Model-Based RL**")
        st.markdown(
            "Learning an explicit environment model and using it for planning or "
            "data augmentation.  Sample-efficient but sensitive to model errors — "
            "Dyna-Q showed the idea at tabular scale; modern methods (Dreamer, MBPO) "
            "use neural world models.  The key challenge is compounding model error "
            "over long rollouts."
        )
        st.markdown("**Real-World Deployment**")
        st.markdown(
            "Bridging the sim-to-real gap (policies trained in simulation fail in the real "
            "world due to unmodelled physics and sensor noise), safe exploration, partial "
            "observability, non-stationary environments, and multi-task generalisation "
            "are active areas of research.  Large pre-trained models (RT-2, VPT) trained "
            "on diverse internet-scale data are beginning to address the generalisation gap."
        )

    st.success(
        "**From bandits to frontiers**: every method in this app — ε-greedy, policy gradients, "
        "eligibility traces, function approximation — appears as a component in modern systems.  "
        "ChatGPT uses RLHF with PPO; AlphaGo uses MCTS with value networks and self-play; "
        "robotics controllers use model-based RL with sim-to-real transfer.  "
        "The core ideas from Sutton & Barto remain the foundation."
    )
