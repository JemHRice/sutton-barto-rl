import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st


# ── Environment ────────────────────────────────────────────────────────────────

class CliffWalkingEnv:
    """
    Cliff Walking (Sutton & Barto Example 6.6).
    4 rows x 12 columns. Start: (3,0). Goal: (3,11).
    Cliff: row 3, cols 1–10 → reward -100, reset to start (episode continues).
    All other steps: reward -1. Episode terminates on reaching goal.
    Actions: 0=up, 1=down, 2=right, 3=left.
    """
    ROWS       = 4
    COLS       = 12
    N_STATES   = 48
    N_ACTIONS  = 4
    START      = 3 * 12 + 0   # state 36
    GOAL       = 3 * 12 + 11  # state 47
    _CLIFF_ROW = 3
    _CLIFF_COLS = set(range(1, 11))
    _DELTAS    = [(-1, 0), (1, 0), (0, 1), (0, -1)]  # up, down, right, left
    ACTION_SYMBOLS = ["↑", "↓", "→", "←"]

    @classmethod
    def _rc(cls, r: int, c: int) -> int:
        return r * cls.COLS + c

    @classmethod
    def _s_to_rc(cls, s: int) -> tuple[int, int]:
        return s // cls.COLS, s % cls.COLS

    @classmethod
    def reset(cls) -> int:
        return cls.START

    @classmethod
    def step(cls, state: int, action: int) -> tuple[int, float, bool]:
        r, c   = cls._s_to_rc(state)
        dr, dc = cls._DELTAS[action]
        nr     = int(np.clip(r + dr, 0, cls.ROWS - 1))
        nc     = int(np.clip(c + dc, 0, cls.COLS - 1))
        ns     = cls._rc(nr, nc)

        if nr == cls._CLIFF_ROW and nc in cls._CLIFF_COLS:
            return cls.START, -100.0, False   # cliff — reset, episode continues
        if ns == cls.GOAL:
            return ns, -1.0, True
        return ns, -1.0, False


# ── Training ───────────────────────────────────────────────────────────────────

_SNAP_INTERVAL = 50


def _eps_greedy(Q: np.ndarray, s: int, eps: float, rng: np.random.Generator) -> int:
    if rng.random() < eps:
        return int(rng.integers(CliffWalkingEnv.N_ACTIONS))
    return int(np.argmax(Q[s]))


def _expected_value(Q: np.ndarray, s: int, eps: float) -> float:
    n      = CliffWalkingEnv.N_ACTIONS
    best_a = int(np.argmax(Q[s]))
    probs  = np.full(n, eps / n)
    probs[best_a] += 1.0 - eps
    return float(np.dot(probs, Q[s]))


@st.cache_data
def run_all_td_control(
    epsilon: float, alpha: float, gamma: float,
    n_episodes: int, seed: int,
) -> dict:
    rng_s = np.random.default_rng(seed)
    rng_q = np.random.default_rng(seed + 1)
    rng_e = np.random.default_rng(seed + 2)

    env = CliffWalkingEnv
    NA  = env.N_ACTIONS
    NS  = env.N_STATES

    Q_sarsa    = np.zeros((NS, NA))
    Q_ql       = np.zeros((NS, NA))
    Q_esarsa   = np.zeros((NS, NA))

    rewards_sarsa  = []
    rewards_ql     = []
    rewards_esarsa = []

    snap_episodes   = []
    snaps_sarsa:  list[np.ndarray] = []
    snaps_ql:     list[np.ndarray] = []
    snaps_esarsa: list[np.ndarray] = []

    for ep in range(1, n_episodes + 1):
        # ── SARSA ─────────────────────────────────────────────────────────────
        s    = env.reset()
        a    = _eps_greedy(Q_sarsa, s, epsilon, rng_s)
        tot  = 0.0
        done = False
        while not done:
            ns, r, done = env.step(s, a)
            na           = _eps_greedy(Q_sarsa, ns, epsilon, rng_s)
            Q_sarsa[s, a] += alpha * (r + gamma * Q_sarsa[ns, na] - Q_sarsa[s, a])
            s, a   = ns, na
            tot   += r
        rewards_sarsa.append(tot)

        # ── Q-Learning ────────────────────────────────────────────────────────
        s    = env.reset()
        tot  = 0.0
        done = False
        while not done:
            a           = _eps_greedy(Q_ql, s, epsilon, rng_q)
            ns, r, done = env.step(s, a)
            Q_ql[s, a] += alpha * (r + gamma * np.max(Q_ql[ns]) - Q_ql[s, a])
            s    = ns
            tot += r
        rewards_ql.append(tot)

        # ── Expected SARSA ────────────────────────────────────────────────────
        s    = env.reset()
        tot  = 0.0
        done = False
        while not done:
            a           = _eps_greedy(Q_esarsa, s, epsilon, rng_e)
            ns, r, done = env.step(s, a)
            ev          = _expected_value(Q_esarsa, ns, epsilon)
            Q_esarsa[s, a] += alpha * (r + gamma * ev - Q_esarsa[s, a])
            s    = ns
            tot += r
        rewards_esarsa.append(tot)

        # ── Snapshots every _SNAP_INTERVAL episodes ───────────────────────────
        if ep % _SNAP_INTERVAL == 0 or ep == 1:
            snap_episodes.append(ep)
            snaps_sarsa.append(Q_sarsa.copy())
            snaps_ql.append(Q_ql.copy())
            snaps_esarsa.append(Q_esarsa.copy())

    def greedy_path(Q):
        s    = env.reset()
        path = [env._s_to_rc(s)]
        seen = set()
        for _ in range(500):
            if s in seen:
                break
            seen.add(s)
            a       = int(np.argmax(Q[s]))
            ns, _, done = env.step(s, a)
            s       = ns
            path.append(env._s_to_rc(s))
            if done:
                break
        return path

    return {
        "rewards_sarsa":   rewards_sarsa,
        "rewards_ql":      rewards_ql,
        "rewards_esarsa":  rewards_esarsa,
        "Q_sarsa_final":   Q_sarsa,
        "Q_ql_final":      Q_ql,
        "Q_esarsa_final":  Q_esarsa,
        "path_sarsa":      greedy_path(Q_sarsa),
        "path_ql":         greedy_path(Q_ql),
        "path_esarsa":     greedy_path(Q_esarsa),
        "snap_episodes":   snap_episodes,
        "snaps_sarsa":     snaps_sarsa,
        "snaps_ql":        snaps_ql,
        "snaps_esarsa":    snaps_esarsa,
    }


# ── Visualisations ─────────────────────────────────────────────────────────────

_ALGO_COLORS = {
    "SARSA":          "#636EFA",
    "Q-Learning":     "#EF553B",
    "Expected SARSA": "#00CC96",
}

_CLIFF_STATES = {CliffWalkingEnv._rc(3, c) for c in range(1, 11)}


def _make_cliff_grid(paths: dict[str, list]) -> go.Figure:
    env  = CliffWalkingEnv
    ROWS, COLS = env.ROWS, env.COLS

    z = np.zeros((ROWS, COLS))
    z[3, 0]  = 1   # start
    z[3, 11] = 2   # goal
    for c in range(1, 11):
        z[3, c] = 3   # cliff

    fig = go.Figure()
    fig.add_trace(go.Heatmap(
        z=z,
        colorscale=[
            [0,    "#F2F3F4"],
            [0.33, "#AED6F1"],
            [0.66, "#A9DFBF"],
            [1.0,  "#E74C3C"],
        ],
        showscale=False, zmin=0, zmax=3,
        xgap=2, ygap=2,
    ))

    # Cell labels
    fig.add_annotation(x=0,  y=3, text="<b>S</b>", showarrow=False,
                       font=dict(size=14, color="#1A5276"))
    fig.add_annotation(x=11, y=3, text="<b>G</b>", showarrow=False,
                       font=dict(size=14, color="#1E8449"))
    for c in range(1, 11):
        fig.add_annotation(x=c, y=3, text="☠", showarrow=False,
                           font=dict(size=13, color="white"))

    for algo, path in paths.items():
        if not path:
            continue
        color = _ALGO_COLORS[algo]
        rs = [p[0] for p in path]
        cs = [p[1] for p in path]
        fig.add_trace(go.Scatter(
            x=cs, y=rs, mode="lines+markers", name=algo,
            line=dict(color=color, width=2.5),
            marker=dict(size=7),
        ))

    fig.update_layout(
        title="Cliff Walking — Greedy Paths",
        xaxis=dict(tickvals=list(range(COLS)), title="Column",
                   showgrid=False, zeroline=False),
        yaxis=dict(autorange="reversed", tickvals=list(range(ROWS)),
                   title="Row", showgrid=False, zeroline=False),
        height=320, template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


def _make_learning_curves(
    rewards: dict[str, list], algos: list[str], window: int = 50
) -> go.Figure:
    fig = go.Figure()
    for algo in algos:
        data  = rewards[algo]
        n     = len(data)
        eps   = list(range(1, n + 1))
        color = _ALGO_COLORS[algo]
        w     = min(window, max(1, n // 10))

        # Raw (faint)
        fig.add_trace(go.Scatter(
            x=eps, y=data, mode="lines", name=f"{algo} (raw)",
            line=dict(color=color, width=1), opacity=0.2,
            showlegend=False,
        ))

        # Rolling average
        if n >= w:
            arr    = np.array(data, dtype=float)
            smooth = np.convolve(arr, np.ones(w) / w, mode="valid").tolist()
            fig.add_trace(go.Scatter(
                x=list(range(w, n + 1)), y=smooth, mode="lines", name=algo,
                line=dict(color=color, width=2.5),
            ))

    fig.update_layout(
        title=f"Learning Curves — Episode Reward ({window}-ep rolling avg)",
        xaxis_title="Episode", yaxis_title="Total Reward",
        template="plotly_white", height=380,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


def _make_qvalue_heatmap(Q: np.ndarray, title: str) -> go.Figure:
    env  = CliffWalkingEnv
    ROWS, COLS = env.ROWS, env.COLS

    # Best Q-value per state
    best_q = np.max(Q, axis=1).reshape(ROWS, COLS)
    best_a = np.argmax(Q, axis=1).reshape(ROWS, COLS)

    fig = go.Figure()
    fig.add_trace(go.Heatmap(
        z=best_q,
        colorscale="RdYlGn",
        colorbar=dict(title="Max Q"),
        xgap=1, ygap=1,
    ))

    # Greedy policy arrows
    symbols = env.ACTION_SYMBOLS
    for r in range(ROWS):
        for c in range(COLS):
            s = env._rc(r, c)
            if s in _CLIFF_STATES:
                text = "☠"
            elif s == env.GOAL:
                text = "G"
            else:
                text = symbols[best_a[r, c]]
            fig.add_annotation(
                x=c, y=r, text=text, showarrow=False,
                font=dict(size=14, color="black"),
            )

    fig.update_layout(
        title=title,
        xaxis=dict(tickvals=list(range(COLS)), showgrid=False, zeroline=False),
        yaxis=dict(autorange="reversed", tickvals=list(range(ROWS)),
                   showgrid=False, zeroline=False),
        height=280, template="plotly_white",
    )
    return fig


def _make_qbar(Q: np.ndarray, state: int, algo: str) -> go.Figure:
    env    = CliffWalkingEnv
    labels = ["Up (↑)", "Down (↓)", "Right (→)", "Left (←)"]
    values = Q[state].tolist()
    best   = int(np.argmax(values))
    colors = [
        _ALGO_COLORS[algo] if i == best else "#C0C0C0"
        for i in range(env.N_ACTIONS)
    ]
    r, c = env._s_to_rc(state)
    fig = go.Figure(go.Bar(x=labels, y=values, marker_color=colors))
    fig.update_layout(
        title=f"Q-values at state ({r},{c}) — {algo}",
        xaxis_title="Action", yaxis_title="Q(s, a)",
        template="plotly_white", height=300,
    )
    return fig


# ── Page ───────────────────────────────────────────────────────────────────────

def show():
    st.title("Cliff Walking: SARSA vs Q-Learning vs Expected SARSA")
    st.markdown("**Section 4 — Temporal Difference Learning**")

    # ── Concept ───────────────────────────────────────────────────────────────
    st.header("Three TD Control Algorithms, One Environment")
    st.markdown(
        r"""
Cliff Walking (Sutton & Barto Example 6.6) is a 4×12 grid with a cliff running along the
bottom row. Falling off the cliff gives a −100 penalty and resets the agent to the start;
the episode continues. Reward is −1 per step. This design creates a tension between two
strategies: hug the cliff-edge (fewer steps, risky) or take the safe path one row above
(more steps, never falls).

All three algorithms use **TD control** — they maintain Q-value estimates and improve the
policy greedily. The key difference is *which target* they use in the update.
"""
    )

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("**SARSA (on-policy)**")
        st.latex(
            r"Q(s,a) \leftarrow Q(s,a) + \alpha\!\left[r"
            r"+ \gamma\, Q(s', a') - Q(s,a)\right]"
        )
        st.markdown(r"$a'$ is chosen ε-greedy from $Q(s', \cdot)$")
    with col2:
        st.markdown("**Q-Learning (off-policy)**")
        st.latex(
            r"Q(s,a) \leftarrow Q(s,a) + \alpha\!\left[r"
            r"+ \gamma \max_{a'} Q(s', a') - Q(s,a)\right]"
        )
        st.markdown(r"Target always uses the greedy action regardless of ε")
    with col3:
        st.markdown("**Expected SARSA**")
        st.latex(
            r"Q(s,a) \leftarrow Q(s,a) + \alpha\!\left[r"
            r"+ \gamma \sum_{a'}\pi(a'|s')\,Q(s',a') - Q(s,a)\right]"
        )
        st.markdown(r"Target is the expectation over the ε-greedy policy $\pi$")

    st.markdown(
        r"""
### On-Policy vs Off-Policy — Why the Path Differs

**SARSA** is on-policy: it evaluates the policy it is *actually following*, including the
ε-random steps. Near the cliff, a random step can be fatal (−100). SARSA learns to be
cautious — it "knows" about its own exploration noise and steers away from cliff edges.
It finds the **safe path** (row 2), which is safer under the ε-greedy behaviour policy.

**Q-Learning** is off-policy: it learns the *optimal greedy policy* regardless of what it
actually does during exploration. It discovers the **optimal path** right along the cliff-edge
(row 3), which is only 12 steps. But during training with ε > 0, it keeps falling off the
cliff, accumulating large negative rewards — hence the noisier learning curve.

**Expected SARSA** takes the average over all actions weighted by the ε-greedy probabilities.
This eliminates the variance introduced by the random action selection in SARSA's target,
making it more stable. It finds a path between SARSA's safe route and Q-Learning's risky one.
"""
    )

    st.info(
        "**Why Expected SARSA reduces variance vs SARSA:** SARSA's target $r + \\gamma Q(s', a')$ "
        "is random because $a'$ is sampled. Expected SARSA replaces that random sample with the "
        "full expectation $\\sum_{a'} \\pi(a'|s') Q(s', a')$, computed analytically. "
        "Same mean, zero additional variance from $a'$ — always a strict improvement over SARSA."
    )

    with st.expander("Deep Dive — When to Use Each Algorithm"):
        import pandas as pd
        table = pd.DataFrame({
            "Algorithm":     ["SARSA", "Q-Learning", "Expected SARSA"],
            "On/Off Policy": ["On-policy", "Off-policy", "On-policy"],
            "Update Target": [
                "r + γ Q(s′, a′)  where a′ ~ ε-greedy",
                "r + γ max Q(s′, ·)",
                "r + γ Σ π(a′|s′) Q(s′, a′)",
            ],
            "Stability":     ["Good", "Good (can be noisy near cliffs)", "Best"],
            "Best Used When": [
                "Agent interacts with real environment; safe exploration matters",
                "Offline / batch learning; optimal policy is the goal",
                "Variance reduction is important; computational cost of expectation is acceptable",
            ],
        })
        st.dataframe(table, use_container_width=True)

    st.success(
        "**When to use each TD control method:** "
        "Use SARSA when safe on-policy learning matters (e.g. real robots). "
        "Use Q-Learning when you need the optimal greedy policy and can tolerate training noise. "
        "Use Expected SARSA as a drop-in replacement for SARSA whenever you want lower variance — "
        "it is strictly better in theory and usually in practice."
    )

    st.divider()

    # ── Simulation controls ────────────────────────────────────────────────────
    st.header("Interactive Simulation")
    st.markdown("Shared hyperparameters for all three algorithms.")

    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        eps = st.slider("ε (exploration)", 0.01, 0.5, 0.1, 0.01)
    with col2:
        alpha = st.slider("α (learning rate)", 0.01, 1.0, 0.5, 0.01)
    with col3:
        gamma = st.slider("γ (discount)", 0.5, 1.0, 1.0, 0.05)
    with col4:
        n_eps = st.slider("Episodes", 100, 2000, 500, 100)
    with col5:
        seed = st.number_input("Random seed", value=42, step=1)

    if st.button("Train All Algorithms", type="primary"):
        with st.spinner("Training SARSA, Q-Learning, and Expected SARSA…"):
            result = run_all_td_control(eps, alpha, gamma, n_eps, int(seed))
        st.session_state["cliff_result"] = result
        st.session_state["cliff_eps"]    = eps

    if "cliff_result" not in st.session_state:
        return

    result = st.session_state["cliff_result"]
    st.header("Results")

    # ── Learning curves ────────────────────────────────────────────────────────
    all_algos = ["SARSA", "Q-Learning", "Expected SARSA"]
    selected  = st.multiselect(
        "Show on learning curves", all_algos, default=all_algos,
        help="Toggle which algorithms appear on the episode reward chart.",
    )
    if selected:
        reward_map = {
            "SARSA":          result["rewards_sarsa"],
            "Q-Learning":     result["rewards_ql"],
            "Expected SARSA": result["rewards_esarsa"],
        }
        fig_lc = _make_learning_curves(reward_map, selected)
        st.plotly_chart(fig_lc, use_container_width=True)

    st.divider()

    # ── Greedy paths on grid ───────────────────────────────────────────────────
    st.header("Greedy Paths (after training)")
    paths = {
        "SARSA":          result["path_sarsa"],
        "Q-Learning":     result["path_ql"],
        "Expected SARSA": result["path_esarsa"],
    }
    fig_grid = _make_cliff_grid(paths)
    st.plotly_chart(fig_grid, use_container_width=True)

    st.divider()

    # ── Policy & Q-value explorer ──────────────────────────────────────────────
    st.header("Policy & Q-Value Explorer")
    st.markdown(
        "Use the episode slider to watch Q-values evolve through training. "
        "Snapshots are stored every 50 episodes."
    )

    snap_eps = result["snap_episodes"]
    sel_ep   = st.select_slider(
        "Training snapshot (episode)", options=snap_eps, value=snap_eps[-1]
    )
    snap_idx = snap_eps.index(sel_ep)

    snap_map = {
        "SARSA":          result["snaps_sarsa"][snap_idx],
        "Q-Learning":     result["snaps_ql"][snap_idx],
        "Expected SARSA": result["snaps_esarsa"][snap_idx],
    }

    tab_s, tab_q, tab_e = st.tabs(["SARSA", "Q-Learning", "Expected SARSA"])

    for tab, algo in zip([tab_s, tab_q, tab_e], all_algos):
        with tab:
            Q_snap = snap_map[algo]
            fig_hm = _make_qvalue_heatmap(Q_snap, f"{algo} — Q-value heatmap (ep {sel_ep})")
            st.plotly_chart(fig_hm, use_container_width=True)

            st.markdown("**State Q-Value Inspector** — select a cell to see all action values.")
            c1, c2 = st.columns(2)
            with c1:
                row_sel = st.number_input(
                    "Row (0 = top)", 0, CliffWalkingEnv.ROWS - 1, 3,
                    key=f"row_{algo}",
                )
            with c2:
                col_sel = st.number_input(
                    "Column (0 = left)", 0, CliffWalkingEnv.COLS - 1, 0,
                    key=f"col_{algo}",
                )
            state_idx = CliffWalkingEnv._rc(int(row_sel), int(col_sel))
            fig_bar   = _make_qbar(Q_snap, state_idx, algo)
            st.plotly_chart(fig_bar, use_container_width=True)

    st.divider()

    # ── Summary metrics ────────────────────────────────────────────────────────
    st.header("Takeaways")
    tail = max(1, n_eps // 10)
    col_a, col_b, col_c = st.columns(3)
    with col_a:
        st.metric("SARSA — final avg reward",
                  f"{np.mean(result['rewards_sarsa'][-tail:]):.1f}")
    with col_b:
        st.metric("Q-Learning — final avg reward",
                  f"{np.mean(result['rewards_ql'][-tail:]):.1f}")
    with col_c:
        st.metric("Expected SARSA — final avg reward",
                  f"{np.mean(result['rewards_esarsa'][-tail:]):.1f}")

    path_lens = {
        "SARSA":          len(result["path_sarsa"]) - 1,
        "Q-Learning":     len(result["path_ql"]) - 1,
        "Expected SARSA": len(result["path_esarsa"]) - 1,
    }
    st.dataframe(pd.DataFrame({
        "Algorithm":            list(path_lens.keys()),
        "Greedy path length":   list(path_lens.values()),
        "On/Off Policy":        ["On", "Off", "On"],
        "Final avg reward":     [
            f"{np.mean(result['rewards_sarsa'][-tail:]):.1f}",
            f"{np.mean(result['rewards_ql'][-tail:]):.1f}",
            f"{np.mean(result['rewards_esarsa'][-tail:]):.1f}",
        ],
    }), use_container_width=True)

    st.info(
        "**SARSA vs Q-Learning on the grid:** you should see SARSA's greedy path run through "
        "row 2 (safe route) while Q-Learning hugs row 3 (cliff-edge optimal route). "
        "If both take the safe path, try lowering ε — the difference is most visible around ε = 0.1."
    )
