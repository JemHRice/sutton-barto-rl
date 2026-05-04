import numpy as np
import plotly.graph_objects as go
import streamlit as st


# ── Environment ────────────────────────────────────────────────────────────────

class WindyGridWorldEnv:
    """
    Windy GridWorld (Sutton & Barto Example 6.5).
    7 rows x 10 columns. Wind pushes agent upward (decreases row index).
    Wind strengths per column: [0,0,0,1,1,1,2,2,1,0].
    Reward -1 per step. Episode ends on reaching goal (3, 7).
    """
    ROWS     = 7
    COLS     = 10
    WIND     = np.array([0, 0, 0, 1, 1, 1, 2, 2, 1, 0])
    START_RC = (3, 0)
    GOAL_RC  = (3, 7)

    # 4-action deltas: up, down, right, left
    _D4 = np.array([[-1, 0], [1, 0], [0, 1], [0, -1]])
    # 8-action deltas: N, NE, E, SE, S, SW, W, NW
    _D8 = np.array([[-1, 0], [-1, 1], [0, 1], [1, 1],
                    [1, 0],  [1, -1], [0, -1], [-1, -1]])

    def __init__(self, stochastic_wind: bool = False,
                 kings_moves: bool = False, seed: int | None = None):
        self.stochastic = stochastic_wind
        self.n_actions  = 8 if kings_moves else 4
        self._deltas    = self._D8 if kings_moves else self._D4
        self.n_states   = self.ROWS * self.COLS
        self.rng        = np.random.default_rng(seed)
        self._state     = self._rc(self.START_RC[0], self.START_RC[1])

    def _rc(self, r: int, c: int) -> int:
        return r * self.COLS + c

    def _s_to_rc(self, s: int) -> tuple[int, int]:
        return s // self.COLS, s % self.COLS

    def reset(self) -> int:
        self._state = self._rc(*self.START_RC)
        return self._state

    def step(self, action: int) -> tuple[int, float, bool]:
        r, c   = self._s_to_rc(self._state)
        dr, dc = self._deltas[action]
        nr     = r + dr
        nc     = c + dc
        # apply wind on departure column
        w = int(self.WIND[c])
        if self.stochastic and w > 0:
            w = max(0, w + int(self.rng.integers(-1, 2)))
        nr = int(np.clip(nr - w, 0, self.ROWS - 1))
        nc = int(np.clip(nc,     0, self.COLS - 1))
        self._state = self._rc(nr, nc)
        done        = (nr, nc) == self.GOAL_RC
        return self._state, -1.0, done

    def state_to_rc(self, s: int) -> tuple[int, int]:
        return self._s_to_rc(s)


# ── Training ───────────────────────────────────────────────────────────────────

_MAX_STEPS = 5000


def _eps_greedy(Q: np.ndarray, s: int, eps: float,
                n_actions: int, rng: np.random.Generator) -> int:
    if rng.random() < eps:
        return int(rng.integers(n_actions))
    return int(np.argmax(Q[s]))


@st.cache_data
def run_windy_comparison(
    stochastic_wind: bool, kings_moves: bool,
    epsilon: float, alpha: float, gamma: float,
    n_episodes: int, seed: int,
) -> dict:
    n_actions = 8 if kings_moves else 4

    def _sarsa(env_seed):
        env   = WindyGridWorldEnv(stochastic_wind, kings_moves, seed=env_seed)
        rng   = np.random.default_rng(env_seed)
        Q     = np.zeros((env.n_states, n_actions))
        steps = []
        for _ in range(n_episodes):
            s    = env.reset()
            a    = _eps_greedy(Q, s, epsilon, n_actions, rng)
            t    = 0
            done = False
            while not done and t < _MAX_STEPS:
                ns, r, done = env.step(a)
                na          = _eps_greedy(Q, ns, epsilon, n_actions, rng)
                Q[s, a]    += alpha * (r + gamma * Q[ns, na] - Q[s, a])
                s, a        = ns, na
                t          += 1
            steps.append(t)
        return Q, steps

    def _qlearning(env_seed):
        env   = WindyGridWorldEnv(stochastic_wind, kings_moves, seed=env_seed)
        rng   = np.random.default_rng(env_seed)
        Q     = np.zeros((env.n_states, n_actions))
        steps = []
        for _ in range(n_episodes):
            s    = env.reset()
            t    = 0
            done = False
            while not done and t < _MAX_STEPS:
                a       = _eps_greedy(Q, s, epsilon, n_actions, rng)
                ns, r, done = env.step(a)
                Q[s, a] += alpha * (r + gamma * np.max(Q[ns]) - Q[s, a])
                s        = ns
                t       += 1
            steps.append(t)
        return Q, steps

    Q_s, sarsa_lengths = _sarsa(seed)
    Q_q, ql_lengths    = _qlearning(seed + 1)

    # Extract greedy paths with fixed seed for reproducibility
    def _greedy_path(Q, stoch, kings):
        env  = WindyGridWorldEnv(stoch, kings, seed=99)
        s    = env.reset()
        path = [env.state_to_rc(s)]
        for _ in range(800):
            a       = int(np.argmax(Q[s]))
            ns, _, done = env.step(a)
            s       = ns
            path.append(env.state_to_rc(s))
            if done:
                break
        return path

    sarsa_path = _greedy_path(Q_s, stochastic_wind, kings_moves)
    ql_path    = _greedy_path(Q_q, stochastic_wind, kings_moves)

    return {
        "sarsa_lengths": sarsa_lengths,
        "ql_lengths":    ql_lengths,
        "sarsa_path":    sarsa_path,
        "ql_path":       ql_path,
        "wind":          WindyGridWorldEnv.WIND.tolist(),
    }


# ── Visualisations ─────────────────────────────────────────────────────────────

def _make_grid_fig(sarsa_path: list, ql_path: list, wind: list) -> go.Figure:
    ROWS, COLS = WindyGridWorldEnv.ROWS, WindyGridWorldEnv.COLS
    GR, GC     = WindyGridWorldEnv.GOAL_RC

    z = np.zeros((ROWS, COLS))
    z[WindyGridWorldEnv.START_RC] = 1
    z[GR, GC]                     = 2

    fig = go.Figure()
    fig.add_trace(go.Heatmap(
        z=z,
        colorscale=[[0, "#F2F3F4"], [0.5, "#AED6F1"], [1.0, "#A9DFBF"]],
        showscale=False, zmin=0, zmax=2,
        xgap=2, ygap=2,
    ))

    if sarsa_path:
        rs = [p[0] for p in sarsa_path]
        cs = [p[1] for p in sarsa_path]
        fig.add_trace(go.Scatter(
            x=cs, y=rs, mode="lines+markers", name="SARSA",
            line=dict(color="#636EFA", width=2.5),
            marker=dict(size=7, symbol="circle"),
        ))

    if ql_path:
        rq = [p[0] for p in ql_path]
        cq = [p[1] for p in ql_path]
        fig.add_trace(go.Scatter(
            x=cq, y=rq, mode="lines+markers", name="Q-Learning",
            line=dict(color="#EF553B", width=2.5),
            marker=dict(size=7, symbol="diamond"),
        ))

    # Labels: S and G
    fig.add_annotation(x=0,  y=3, text="<b>S</b>", showarrow=False,
                       font=dict(size=15, color="#1A5276"))
    fig.add_annotation(x=GC, y=GR, text="<b>G</b>", showarrow=False,
                       font=dict(size=15, color="#1E8449"))

    # Wind indicators below grid
    fig.add_annotation(x=-0.75, y=7.4, text="<b>Wind</b>", showarrow=False,
                       font=dict(size=11, color="#555"))
    for c, w in enumerate(wind):
        label = f"(↑{w})" if w > 0 else "(0)"
        fig.add_annotation(x=c, y=7.4, text=label, showarrow=False,
                           font=dict(size=13, color="#AAAAAA"))

    fig.update_layout(
        title="Windy GridWorld — Learned Paths (greedy policy)",
        xaxis=dict(tickvals=list(range(COLS)), title="Column",
                   showgrid=False, zeroline=False),
        yaxis=dict(autorange="reversed", tickvals=list(range(ROWS)),
                   title="Row", range=[-0.5, 8.2], showgrid=False, zeroline=False),
        height=500, template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


def _make_steps_fig(sarsa: list, ql: list) -> go.Figure:
    n      = len(sarsa)
    eps    = list(range(1, n + 1))
    window = max(10, n // 20)

    def smooth(data):
        arr = np.array(data, dtype=float)
        return np.convolve(arr, np.ones(window) / window, mode="valid").tolist()

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=eps, y=sarsa, mode="lines", name="SARSA (raw)",
                             line=dict(color="#636EFA", width=1), opacity=0.25))
    fig.add_trace(go.Scatter(x=eps, y=ql,    mode="lines", name="Q-Learning (raw)",
                             line=dict(color="#EF553B", width=1), opacity=0.25))

    offset = window - 1
    if n >= window:
        x_s = eps[offset:]
        fig.add_trace(go.Scatter(x=x_s, y=smooth(sarsa), mode="lines",
                                 name=f"SARSA ({window}-ep avg)",
                                 line=dict(color="#636EFA", width=2.5)))
        fig.add_trace(go.Scatter(x=x_s, y=smooth(ql), mode="lines",
                                 name=f"Q-Learning ({window}-ep avg)",
                                 line=dict(color="#EF553B", width=2.5)))

    fig.update_layout(
        title="Steps per Episode over Training",
        xaxis_title="Episode", yaxis_title="Steps to Goal",
        template="plotly_white", height=380,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


# ── Page ───────────────────────────────────────────────────────────────────────

def show():
    st.title("Windy GridWorld")
    st.markdown("**Section 4 — Temporal Difference Learning**")

    # ── Concept ───────────────────────────────────────────────────────────────
    st.header("How Wind Modifies the Transition Function")
    st.markdown(
        r"""
The Windy GridWorld (Sutton & Barto Example 6.5) is a 7×10 grid where columns 4–8 have an
upward current — a fixed wind strength that shifts the agent toward row 0 **after every step**,
regardless of the chosen action.

This means the transition $p(s' \mid s, a)$ is no longer "move one cell in direction $a$".
The actual next state combines the intended move with the wind:
"""
    )
    st.latex(
        r"s' = \mathrm{clip}\!\Bigl(\mathrm{move}(s,a) - \mathbf{w}_{c(s)},\;"
        r"[0, R-1]\Bigr)"
    )
    st.markdown(
        r"""
where $\mathbf{w}_{c(s)}$ is the wind strength of the **current column** and $R = 7$ is
the number of rows. Because the wind depends only on the current column (not history),
the environment remains Markov and TD methods apply directly. The SARSA Bellman update:
"""
    )
    st.latex(
        r"Q(s,a) \;\leftarrow\; Q(s,a) + \alpha\!\left[R_{t+1}"
        r"+ \gamma\, Q(S_{t+1}, A_{t+1}) - Q(s,a)\right]"
    )
    st.markdown(
        r"""
$A_{t+1}$ is chosen **on-policy** (ε-greedy from the current $Q$), making this SARSA.
Q-Learning instead bootstraps from the greedy action:
$R_{t+1} + \gamma \max_{a'} Q(S_{t+1}, a')$, decoupling the behaviour and target policies.

TD methods learn from **experience**: when the wind pushes the agent to an unexpected cell,
the resulting TD error propagates backward and adjusts Q-values. With enough exploration
the agent discovers paths that **account for the wind** — navigating at angles to compensate
for the upward push in the high-wind columns.
"""
    )

    col1, col2 = st.columns(2)
    with col1:
        st.info(
            "**King's Moves** add 4 diagonal actions (NE, SE, SW, NW) to the standard 4. "
            "In windy columns this is very powerful: the agent can move diagonally downwind, "
            "compensating for the upward gusts and reaching the goal in significantly fewer steps."
        )
    with col2:
        st.success(
            "**Why TD works despite stochastic wind:** each update uses the *actual* next state "
            "the agent landed in — no transition model is needed. The Q-values converge to the "
            "expected value under the wind distribution as long as all state-action pairs are "
            "visited sufficiently often."
        )

    with st.expander("Deep Dive — Stochastic Wind and Convergence"):
        st.markdown(
            r"""
With stochastic wind, the same $(s, a)$ pair can lead to different next states across episodes.
The Q-update becomes a noisy sample of the true expected Bellman target.

Under stochastic approximation conditions (all pairs visited ∞ often, $\sum \alpha_t = \infty$,
$\sum \alpha_t^2 < \infty$), both SARSA and Q-Learning converge. With a **fixed** $\alpha$
the values don't converge exactly — they track a neighbourhood of the true values, with
residual oscillations proportional to $\alpha \sigma^2_{\text{wind}}$. Smaller $\alpha$
reduces tracking noise but slows early learning.

SARSA vs Q-Learning under stochastic wind: SARSA accounts for the exploration ε in its
target (it uses the *actual* next action, which is sometimes random), making it slightly more
conservative but stable. Q-Learning targets the greedy action regardless of ε, so it learns
the optimal policy faster but can be noisier during learning.
"""
        )

    st.divider()

    # ── Simulation ─────────────────────────────────────────────────────────────
    st.header("Interactive Simulation")
    st.markdown(
        "Train SARSA and Q-Learning simultaneously on the Windy GridWorld. "
        "After training, the **greedy paths** for both algorithms are overlaid on the grid. "
        "Lower final episode length = faster navigation."
    )

    col1, col2 = st.columns(2)
    with col1:
        wind_type = st.radio(
            "Wind type",
            ["Standard (deterministic)", "Stochastic (±1 random perturbation)"],
            help="Stochastic: each step randomly adds -1, 0, or +1 to the column's base wind strength.",
        )
        kings = st.checkbox("King's Moves (8 directional actions)", value=False)
    with col2:
        seed = st.number_input("Random seed", value=42, step=1)

    col3, col4, col5, col6 = st.columns(4)
    with col3:
        eps = st.slider("ε (exploration)", 0.01, 0.5, 0.1, 0.01)
    with col4:
        alpha = st.slider("α (learning rate)", 0.01, 1.0, 0.5, 0.01)
    with col5:
        gamma = st.slider("γ (discount)", 0.5, 1.0, 1.0, 0.05)
    with col6:
        n_episodes = st.slider("Episodes", 100, 3000, 500, 100)

    stoch = wind_type.startswith("Stochastic")

    if st.button("Run Simulation", type="primary"):
        with st.spinner("Training SARSA and Q-Learning…"):
            result = run_windy_comparison(
                stoch, kings, eps, alpha, gamma, n_episodes, int(seed)
            )

        st.header("Results")

        # Grid with paths
        fig_grid = _make_grid_fig(result["sarsa_path"], result["ql_path"], result["wind"])
        st.plotly_chart(fig_grid, use_container_width=True)

        # Steps per episode
        fig_steps = _make_steps_fig(result["sarsa_lengths"], result["ql_lengths"])
        st.plotly_chart(fig_steps, use_container_width=True)

        # Metrics
        col_a, col_b, col_c = st.columns(3)
        with col_a:
            avg_s = np.mean(result["sarsa_lengths"][-max(1, n_episodes // 10):])
            st.metric("SARSA — final avg steps", f"{avg_s:.0f}")
        with col_b:
            avg_q = np.mean(result["ql_lengths"][-max(1, n_episodes // 10):])
            st.metric("Q-Learning — final avg steps", f"{avg_q:.0f}")
        with col_c:
            sp = len(result["sarsa_path"]) - 1
            qp = len(result["ql_path"]) - 1
            st.metric("Greedy path length (SARSA / Q-L)", f"{sp} / {qp}")

        st.header("Takeaways")
        st.markdown(
            """
Both algorithms learn to navigate the windy columns, but they reflect different trade-offs:

- **SARSA** (on-policy) adapts its Q-values to include the ε-noise in its own behaviour.
  During training with ε > 0 it tends to take a slightly safer path away from the wind zone.
- **Q-Learning** (off-policy) always bootstraps from the greedy maximum, so it more directly
  pursues the theoretically shortest path — but during training it may accumulate more negative
  reward from exploratory steps in the high-wind region.
"""
        )
        if kings:
            st.info(
                "With King's Moves you should see a noticeably shorter greedy path. "
                "Diagonal actions let the agent cut through the windy columns at an angle, "
                "compensating for the upward drift and shaving off several steps."
            )
        if stoch:
            st.warning(
                "With stochastic wind the greedy path shown is one sample — it may differ "
                "slightly between runs. The Q-values encode the *expected* best policy over "
                "the wind distribution, so the greedy policy is optimal on average."
            )
