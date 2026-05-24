import numpy as np
import plotly.graph_objects as go
import streamlit as st
from collections import deque

# ── Four-Rooms environment ─────────────────────────────────────────────────────

_ROWS, _COLS = 13, 13
_ACTIONS = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # N, S, W, E


def _build_walls() -> frozenset:
    w: set = set()
    for r in range(_ROWS):
        w.add((r, 0))
        w.add((r, _COLS - 1))
    for c in range(_COLS):
        w.add((0, c))
        w.add((_ROWS - 1, c))
    for c in range(1, _COLS - 1):
        if c not in (2, 9):
            w.add((6, c))
    for r in range(1, _ROWS - 1):
        if r not in (2, 9):
            w.add((r, 6))
    return frozenset(w)


_WALLS = _build_walls()
_DOORWAYS = [(6, 2), (6, 9), (2, 6), (9, 6)]
_START = (10, 2)
_GOAL = (2, 10)
_VALID = sorted(
    [(r, c) for r in range(_ROWS) for c in range(_COLS) if (r, c) not in _WALLS]
)
_STATE_IDX = {s: i for i, s in enumerate(_VALID)}
_N_STATES = len(_VALID)


def _step(s: tuple, a: int) -> tuple:
    dr, dc = _ACTIONS[a]
    ns = (s[0] + dr, s[1] + dc)
    if ns in _WALLS:
        return s, -0.001, False
    if ns == _GOAL:
        return ns, 1.0, True
    return ns, -0.001, False


def _bfs_policy(target: tuple) -> dict:
    dist: dict = {target: 0}
    q = deque([target])
    while q:
        s = q.popleft()
        for dr, dc in _ACTIONS:
            ns = (s[0] + dr, s[1] + dc)
            if ns not in _WALLS and ns not in dist:
                dist[ns] = dist[s] + 1
                q.append(ns)
    pol: dict = {}
    for s in _VALID:
        best_a, best_d = None, float("inf")
        for i, (dr, dc) in enumerate(_ACTIONS):
            ns = (s[0] + dr, s[1] + dc)
            d = dist.get(ns, float("inf"))
            if d < best_d:
                best_d = d
                best_a = i
        if best_a is not None:
            pol[s] = best_a
    return pol


_OPTION_POLICIES = [_bfs_policy(d) for d in _DOORWAYS]


def _execute_option(s: tuple, opt: int, max_steps: int = 80) -> tuple:
    target = _DOORWAYS[opt]
    if s == target:
        return s, 0.0, 1, False
    pol = _OPTION_POLICIES[opt]
    total_r, steps, done = 0.0, 0, False
    for _ in range(max_steps):
        if s not in pol:
            break
        a = pol[s]
        s, r, done = _step(s, a)
        total_r += r
        steps += 1
        if done or s == target:
            break
    return s, total_r, max(steps, 1), done


# ── Simulations ────────────────────────────────────────────────────────────────


@st.cache_data
def run_flat_q(
    n_episodes: int, alpha: float, gamma: float, epsilon: float, seed: int
) -> list:
    rng = np.random.default_rng(seed)
    Q = np.zeros((_N_STATES, 4))
    out = []
    for _ in range(n_episodes):
        s = _START
        steps = 0
        for _ in range(600):
            si = _STATE_IDX[s]
            a = (
                int(rng.integers(4))
                if rng.random() < epsilon
                else int(np.argmax(Q[si]))
            )
            ns, r, done = _step(s, a)
            nsi = _STATE_IDX[ns]
            Q[si, a] += alpha * (r + gamma * np.max(Q[nsi]) - Q[si, a])
            s = ns
            steps += 1
            if done:
                break
        out.append(steps)
    return out


@st.cache_data
def run_options_q(
    n_episodes: int, alpha: float, gamma: float, epsilon: float, seed: int
) -> list:
    rng = np.random.default_rng(seed)
    N_ACT = 4 + len(_DOORWAYS)
    Q = np.zeros((_N_STATES, N_ACT))
    out = []
    for _ in range(n_episodes):
        s = _START
        steps = 0
        done = False
        while steps < 600 and not done:
            si = _STATE_IDX[s]
            a = (
                int(rng.integers(N_ACT))
                if rng.random() < epsilon
                else int(np.argmax(Q[si]))
            )
            if a < 4:
                ns, r, done = _step(s, a)
                nsi = _STATE_IDX[ns]
                Q[si, a] += alpha * (r + gamma * np.max(Q[nsi]) - Q[si, a])
                s = ns
                steps += 1
            else:
                ns, total_r, k, done = _execute_option(s, a - 4)
                nsi = _STATE_IDX[ns]
                eff_gamma = gamma**k
                Q[si, a] += alpha * (total_r + eff_gamma * np.max(Q[nsi]) - Q[si, a])
                s = ns
                steps += k
        out.append(steps)
    return out


# ── Visualisations ─────────────────────────────────────────────────────────────


def _grid_fig() -> go.Figure:
    z = [[0.0] * _COLS for _ in range(_ROWS)]
    for r, c in _WALLS:
        z[r][c] = 1.0
    for r, c in _DOORWAYS:
        z[r][c] = 0.4

    fig = go.Figure(
        go.Heatmap(
            z=z,
            colorscale=[[0.0, "#F5F5F5"], [0.4, "#76B7B2"], [1.0, "#3A3A3A"]],
            showscale=False,
            xgap=1,
            ygap=1,
        )
    )
    for r, c in _DOORWAYS:
        fig.add_annotation(
            x=c, y=r, text="D", showarrow=False, font=dict(size=9, color="#003380")
        )
    fig.add_annotation(
        x=_START[1],
        y=_START[0],
        text="S",
        showarrow=False,
        font=dict(size=13, color="#1a6600", family="Arial Black"),
    )
    fig.add_annotation(
        x=_GOAL[1],
        y=_GOAL[0],
        text="G",
        showarrow=False,
        font=dict(size=13, color="#cc5500", family="Arial Black"),
    )
    # Room labels
    for label, r, c in [
        ("Room 1", 9, 3),
        ("Room 2", 9, 9),
        ("Room 3", 3, 3),
        ("Room 4", 3, 9),
    ]:
        fig.add_annotation(
            x=c, y=r, text=label, showarrow=False, font=dict(size=9, color="#777")
        )
    fig.update_layout(
        title="Four-Rooms Environment (S = start, G = goal, D = doorway)",
        xaxis=dict(visible=False, range=[-0.5, _COLS - 0.5]),
        yaxis=dict(visible=False, range=[_ROWS - 0.5, -0.5], scaleanchor="x"),
        template="plotly_white",
        height=380,
        margin=dict(l=10, r=10, t=40, b=10),
    )
    return fig


def _smooth(xs: list, w: int = 20) -> list:
    arr = np.array(xs, dtype=float)
    return np.convolve(arr, np.ones(w) / w, mode="valid").tolist()


def _convergence_fig(flat: list, opts: list, w: int = 20) -> go.Figure:
    eps = list(range(w, len(flat) + 1))
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=eps,
            y=_smooth(flat, w),
            mode="lines",
            name="Flat Q-learning (4 actions)",
            line=dict(color="#636EFA", width=2),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=eps,
            y=_smooth(opts, w),
            mode="lines",
            name="Q-learning + Options (8 actions)",
            line=dict(color="#7B2D8B", width=2),
        )
    )
    fig.update_layout(
        title=f"Steps per Episode — Rolling Mean (window {w})",
        xaxis_title="Episode",
        yaxis_title="Steps to goal",
        template="plotly_white",
        height=360,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


# ── Page ───────────────────────────────────────────────────────────────────────


def show():
    st.title("Hierarchical RL — Options Framework")
    st.markdown(
        '<div style="background:#7B2D8B;height:3px;'
        'border-radius:2px;margin-bottom:1rem;"></div>',
        unsafe_allow_html=True,
    )
    st.markdown("**Section 13 — Advanced Topics (Chapter 17)**")

    # ── Concept ───────────────────────────────────────────────────────────────
    st.header("Flat Policies and Temporal Scale")
    st.markdown(r"""
Standard RL agents choose one primitive action per time step. For tasks requiring many
coordinated sub-behaviours — reaching a door, navigating between rooms, assembling a
sequence of motor primitives — the credit assignment problem becomes severe: a reward
hundreds of steps away must propagate back through hundreds of one-step updates.

**Hierarchical RL** introduces *temporally extended actions* that run for multiple steps,
allowing the high-level agent to reason at a coarser time scale and dramatically shortening
the effective credit-assignment distance.
""")

    st.header("The Options Framework")
    st.markdown(r"""
An **option** $o = \langle \mathcal{I}_o,\, \pi_o,\, \beta_o \rangle$ has three components:

| Component | Symbol | Role |
|---|---|---|
| Initiation set | $\mathcal{I}_o \subseteq \mathcal{S}$ | States from which the option may be invoked |
| Intra-option policy | $\pi_o(a \mid s)$ | Primitive-action policy executed while option runs |
| Termination condition | $\beta_o(s) \in [0,1]$ | Probability of stopping the option in state $s$ |

Once invoked, the option's internal policy $\pi_o$ drives the agent until $\beta_o$ fires,
producing a trajectory of variable length. This creates a **semi-Markov decision process
(SMDP)**: transitions have variable duration rather than a fixed one step.
""")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Temporal abstraction**")
        st.markdown(
            "Options compress multi-step behaviours into single decisions, reducing the "
            "number of high-level choices between reward and the state where it was earned. "
            "This is the key advantage for tasks with sparse, delayed rewards."
        )
    with col2:
        st.markdown("**Transfer and reuse**")
        st.markdown(
            "A 'navigate to doorway' option learned once transfers to any task sharing "
            "the same layout, even when the goal changes. Humans routinely reuse motor "
            "primitives (grip, push, reach) across completely different objectives."
        )

    st.info(
        "**Intra-option Q-learning** (Sutton, Precup & Singh 1999) propagates reward "
        "information backwards *during* option execution — not just at termination — "
        "so credit assignment is still efficient even within a long option."
    )

    with st.expander("SMDP Q-learning update"):
        st.markdown(r"""
When option $o$ is invoked from state $s$, executes for $k$ steps, and terminates in $s'$
with cumulative discounted reward $R = \sum_{i=0}^{k-1} \gamma^i r_{t+i}$:

$$Q(s, o) \;\leftarrow\; Q(s, o) + \alpha \!\left[R + \gamma^k \max_{o'} Q(s', o') - Q(s, o)\right]$$

The $\gamma^k$ factor discounts the future value appropriately for the option's duration.
When $k = 1$ (primitive action), this reduces exactly to standard one-step Q-learning.
""")

    with st.expander("Learned vs hand-crafted options"):
        st.markdown(r"""
**Hand-crafted options** (used in this demo) are designed by a human specifying useful
sub-goals — e.g., reaching each doorway in the four-rooms problem. They work immediately
but require domain knowledge.

**Learned options** discover sub-goals automatically by identifying *bottleneck states* —
states visited disproportionately often on near-optimal paths.  Key methods include:

- **Eigenoptions** (Machado et al. 2017): options derived from the eigenvectors of the
  environment's graph Laplacian, capturing the principal directions of reachability.
- **Option-Critic** (Bacon et al. 2017): end-to-end gradient optimisation that jointly
  learns the option policies, termination functions, and the high-level policy.
- **DADS** (Sharma et al. 2020): unsupervised skill discovery using mutual information
  between skills and the transitions they produce.
""")

    st.divider()
    st.plotly_chart(_grid_fig(), use_container_width=True)
    st.caption(
        "To reach **G** from **S** the agent must cross two rooms and pass through two "
        "doorways (**D**).  Flat Q-learning must propagate reward across every individual "
        "grid cell step; options can jump directly to each doorway."
    )

    st.divider()

    # ── Simulation ─────────────────────────────────────────────────────────────
    st.header("Interactive Simulation — Four-Rooms")
    st.markdown(
        "Compare flat Q-learning (4 primitive actions: N/S/E/W) against Q-learning "
        "augmented with 4 doorway-navigation options (8 total actions).  "
        "The option subpolicies are shortest-path BFS policies; the high-level agent "
        "must still learn *when* to invoke each option."
    )

    col1, col2 = st.columns(2)
    with col1:
        n_episodes = st.slider("Episodes", 200, 1000, 600, 100)
        alpha = st.slider("Learning rate α", 0.05, 0.5, 0.2, 0.05)
    with col2:
        gamma = st.slider("Discount γ", 0.90, 1.00, 0.99, 0.01)
        epsilon = st.slider("Exploration ε", 0.05, 0.40, 0.15, 0.05)
        seed = st.number_input("Random seed", value=42, step=1)

    if st.button("Run comparison", type="primary"):
        with st.spinner("Training both agents…"):
            flat = run_flat_q(n_episodes, alpha, float(gamma), epsilon, int(seed))
            opts = run_options_q(n_episodes, alpha, float(gamma), epsilon, int(seed))

        st.plotly_chart(_convergence_fig(flat, opts), use_container_width=True)

        w = 20
        flat_end = float(np.mean(_smooth(flat, w)[-30:]))
        opts_end = float(np.mean(_smooth(opts, w)[-30:]))
        speedup = flat_end / opts_end if opts_end > 0 else 1.0

        col_a, col_b, col_c = st.columns(3)
        with col_a:
            st.metric("Flat Q — final steps (mean)", f"{flat_end:.0f}")
        with col_b:
            st.metric("Options Q — final steps (mean)", f"{opts_end:.0f}")
        with col_c:
            st.metric("Steps reduction", f"{speedup:.1f}×")

        st.header("Key Takeaways")
        if speedup >= 1.3:
            st.success(
                f"Options reduced the final mean steps by {speedup:.1f}×.  "
                "With options, the high-level agent chains three or four option decisions "
                "(jump to doorway, jump to next doorway, reach goal) instead of navigating "
                "every individual grid cell — credit assignment is far shorter."
            )
        else:
            st.info(
                "The gap is modest with these settings.  Try more episodes (≥600), "
                "higher ε (0.2+), or lower α (0.1) to see the advantage widen.  "
                "Options help most when the reward is distant and the sub-goals are reusable."
            )
        st.info(
            "**Why options converge faster**: flat Q-learning must back-propagate the "
            "reward signal across every primitive step separating goal from start — often "
            "100+ steps.  Options compress this to 3–4 high-level decisions, making each "
            "episode far more informative about which sub-goal sequence leads to the reward."
        )
