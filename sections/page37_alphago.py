import numpy as np
import plotly.graph_objects as go
import streamlit as st
from collections import defaultdict
import math

# ── MCTS on a toy tree ────────────────────────────────────────────────────────


class _Node:
    """One node in the MCTS tree."""

    __slots__ = ("N", "W", "Q", "P", "children", "terminal", "value")

    def __init__(self, prior: float = 0.0):
        self.N = 0  # visit count
        self.W = 0.0  # total value
        self.Q = 0.0  # mean value
        self.P = prior  # prior probability from policy network
        self.children: dict[int, "_Node"] = {}
        self.terminal = False
        self.value = 0.0  # terminal value if terminal


def _build_toy_tree(depth: int, branching: int, rng: np.random.Generator) -> _Node:
    """
    Build a random toy game tree.
    Leaf values are random in [-1, 1].  Prior probabilities are uniform + noise.
    """

    def _build(d):
        node = _Node()
        if d == 0:
            node.terminal = True
            node.value = float(rng.uniform(-1, 1))
            return node
        for a in range(branching):
            child = _build(d - 1)
            child.P = float(rng.dirichlet(np.ones(branching))[a])
            node.children[a] = child
        return node

    return _build(depth)


def _ucb_score(parent: _Node, child: _Node, c_puct: float) -> float:
    """PUCT formula used in AlphaGo Zero."""
    u = c_puct * child.P * math.sqrt(parent.N) / (1 + child.N)
    return child.Q + u


def _mcts_simulate(root: _Node, c_puct: float) -> float:
    """One MCTS simulation: selection → evaluation → backup."""
    path: list[_Node] = [root]
    node = root

    # Selection: traverse down choosing max UCB
    while node.children and not node.terminal:
        best_a, best_child = max(
            node.children.items(),
            key=lambda kv: _ucb_score(node, kv[1], c_puct),
        )
        node = best_child
        path.append(node)

    # Evaluation
    if node.terminal:
        value = node.value
    else:
        # Leaf not yet expanded — use random rollout as value estimate
        value = float(np.random.uniform(-1, 1))

    # Backup
    for n in path:
        n.N += 1
        n.W += value
        n.Q = n.W / n.N
    return value


@st.cache_data
def run_mcts(
    depth: int,
    branching: int,
    n_simulations: int,
    c_puct: float,
    seed: int,
) -> tuple[list, list, list, list]:
    """
    Run MCTS on a toy tree.
    Returns (sim_indices, root_Q_over_time, best_action_visits, action_visit_counts).
    """
    rng = np.random.default_rng(seed)
    np.random.seed(seed)
    root = _build_toy_tree(depth, branching, rng)

    root_Q_over_time: list[float] = []
    best_action_hist: list[int] = []

    for i in range(n_simulations):
        _mcts_simulate(root, c_puct)
        root_Q_over_time.append(root.Q)
        if root.children:
            best_a = max(root.children, key=lambda a: root.children[a].N)
            best_action_hist.append(best_a)
        else:
            best_action_hist.append(-1)

    # Final action visit counts
    visit_counts = {a: root.children[a].N for a in root.children}
    return (
        list(range(1, n_simulations + 1)),
        root_Q_over_time,
        best_action_hist,
        [
            {"action": a, "visits": v, "Q": round(root.children[a].Q, 3)}
            for a, v in sorted(visit_counts.items(), key=lambda kv: -kv[1])
        ],
    )


# ── Visualisations ─────────────────────────────────────────────────────────────


def _q_convergence_fig(sim_idxs: list, q_vals: list, branching: int) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=sim_idxs,
            y=q_vals,
            mode="lines",
            name="Root Q value",
            line=dict(color="#7B2D8B", width=2),
        )
    )
    fig.add_hline(y=0, line_dash="dot", line_color="grey")
    fig.update_layout(
        title="Root Q-value Estimate over MCTS Simulations",
        xaxis_title="Simulations",
        yaxis_title="Q(root)",
        yaxis=dict(range=[-1.1, 1.1]),
        template="plotly_white",
        height=320,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


def _visit_bar_fig(visit_data: list, branching: int) -> go.Figure:
    actions = [f"Action {d['action']}" for d in visit_data]
    visits = [d["visits"] for d in visit_data]
    q_vals = [d["Q"] for d in visit_data]
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=actions,
            y=visits,
            name="Visit count",
            marker_color="#AB63FA",
            text=[f"Q={q:.2f}" for q in q_vals],
            textposition="outside",
        )
    )
    fig.update_layout(
        title="Root Action Visit Counts",
        xaxis_title="Action",
        yaxis_title="Visits",
        template="plotly_white",
        height=320,
    )
    return fig


def _architecture_fig() -> go.Figure:
    """Simplified AlphaGo architecture as an annotated Sankey-style diagram."""
    fig = go.Figure()
    boxes = [
        (0.05, 0.50, "Board State<br><i>19×19 position</i>", "#636EFA"),
        (0.28, 0.75, "Supervised<br>Policy Net<br><i>SL-policy π_σ</i>", "#00CC96"),
        (0.28, 0.35, "Rollout<br>Policy Net<br><i>fast π_π</i>", "#FFA15A"),
        (0.55, 0.75, "RL Policy<br>Network<br><i>π_ρ</i>", "#AB63FA"),
        (0.55, 0.35, "Value<br>Network<br><i>V_θ(s)</i>", "#EF553B"),
        (0.80, 0.55, "MCTS<br>Action<br>Selection", "#7B2D8B"),
    ]
    for x, y, label, col in boxes:
        fig.add_shape(
            type="rect",
            x0=x - 0.08,
            y0=y - 0.12,
            x1=x + 0.08,
            y1=y + 0.12,
            fillcolor=col,
            opacity=0.25,
            line=dict(color=col, width=2),
        )
        fig.add_annotation(
            x=x, y=y, text=label, showarrow=False, font=dict(size=11), align="center"
        )

    arrows = [
        (0.13, 0.50, 0.20, 0.75),
        (0.13, 0.50, 0.20, 0.35),
        (0.36, 0.75, 0.47, 0.75),
        (0.36, 0.75, 0.47, 0.35),
        (0.63, 0.75, 0.72, 0.60),
        (0.63, 0.35, 0.72, 0.50),
    ]
    for x0, y0, x1, y1 in arrows:
        fig.add_annotation(
            x=x1,
            y=y1,
            ax=x0,
            ay=y0,
            xref="paper",
            yref="paper",
            axref="paper",
            ayref="paper",
            arrowhead=2,
            arrowwidth=2,
            arrowcolor="#444",
            showarrow=True,
            text="",
        )

    fig.update_layout(
        title="AlphaGo Architecture Overview",
        xaxis=dict(visible=False, range=[0, 1]),
        yaxis=dict(visible=False, range=[0, 1]),
        template="plotly_white",
        height=380,
        margin=dict(l=10, r=10, t=50, b=10),
    )
    return fig


# ── Page ───────────────────────────────────────────────────────────────────────


def show():
    st.title("AlphaGo — Combining Search with Learned Knowledge")
    st.markdown(
        '<div style="background:#7B2D8B;height:3px;'
        'border-radius:2px;margin-bottom:1rem;"></div>',
        unsafe_allow_html=True,
    )
    st.markdown("**Section 12 — Applications (Chapter 16)**")

    # ── Concept ───────────────────────────────────────────────────────────────
    st.header("Why Go Was Considered Unsolvable")
    st.markdown(r"""
Go has a branching factor of ~250 (vs ~35 for Chess) and game trees of depth 150+.
Pure alpha-beta search, which defeated Kasparov in Chess, would need to search
$250^{150} \approx 10^{359}$ nodes — completely infeasible.

**AlphaGo** (Silver et al., 2016) solved this by combining four components:

1. **Supervised learning policy network** $\pi_\sigma$ — trained on 30 million human expert moves to predict the next move.
2. **Reinforcement learning policy network** $\pi_\rho$ — fine-tuned by self-play to win games, not just predict expert moves.
3. **Value network** $V_\theta(s)$ — trained to predict who wins from a given position, replacing expensive rollouts.
4. **Monte Carlo Tree Search (MCTS)** — guided by $\pi_\rho$ for move selection and $V_\theta$ for position evaluation.

The key insight: **use neural networks to reduce both the breadth and the depth of search**.
""")

    st.plotly_chart(_architecture_fig(), use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Reducing breadth** (policy network)")
        st.markdown(
            "Instead of considering all 250 legal moves, MCTS samples only the "
            "most promising ones according to $\\pi_\\rho$.  "
            "This focuses computation on moves that a strong player would consider."
        )
    with col2:
        st.markdown("**Reducing depth** (value network)")
        st.markdown(
            "Instead of rolling out to the game end, MCTS evaluates leaf nodes "
            "using $V_\\theta(s)$ — a trained estimate of win probability.  "
            "This replaces thousands of random moves with one forward pass."
        )

    st.info(
        "**AlphaGo Zero** (2017) removed the supervised learning phase entirely — "
        "it learned solely through self-play with MCTS, starting from random play.  "
        "It surpassed AlphaGo in 3 days and is considered the cleaner, more elegant system."
    )

    with st.expander("MCTS — Selection, Expansion, Evaluation, Backup"):
        st.markdown(r"""
Monte Carlo Tree Search builds a partial game tree through repeated simulations:

1. **Selection**: from the root, traverse the tree by choosing actions that maximise the
   **PUCT** (Predictor + UCT) score:
   $$a^* = \arg\max_a \left[Q(s,a) + c_\text{puct}\,P(s,a)\,\frac{\sqrt{N(s)}}{1+N(s,a)}\right]$$
   $Q(s,a)$ is the average value seen from $(s,a)$; $P(s,a)$ is the policy prior;
   $N(s,a)$ is the visit count.  The second term encourages exploration of
   low-visit / high-prior moves.

2. **Expansion**: when we reach a node not yet in the tree, add it and query
   the policy network for priors $P(s, \cdot)$.

3. **Evaluation**: estimate the leaf value using the value network $V_\theta(s)$,
   mixed with a fast random rollout result:
   $v = (1-\lambda)\,V_\theta(s) + \lambda\,z_\text{rollout}$.

4. **Backup**: propagate the value back up the path, updating $N$ and $Q$ at each node.

After $n$ simulations, the MCTS policy is $\pi_\text{MCTS}(a|s) \propto N(s,a)^{1/\tau}$.
""")

    with st.expander("AlphaGo Zero — simplification without supervised learning"):
        st.markdown(r"""
AlphaGo Zero replaced the four separate networks with a **single residual network**
with two heads:
- Policy head: outputs move probabilities.
- Value head: outputs win probability.

Training loop (entirely self-play):
1. Play games using MCTS + current network to generate training data.
2. The MCTS action distribution $\pi_\text{MCTS}$ is the policy target.
3. The game outcome $z \in \{-1, +1\}$ is the value target.
4. Train the network to minimise $(z - v)^2 - \pi_\text{MCTS}^\top \ln p$.

After 4.9 million self-play games, AlphaGo Zero surpassed all prior versions
and all human players.  The entire training took 40 days on 64 TPUs.
""")

    st.divider()

    # ── MCTS demo ──────────────────────────────────────────────────────────────
    st.header("Interactive MCTS Demo")
    st.markdown(
        "Explore how MCTS concentrates simulations on the best actions over time.  "
        "A toy random game tree is used — the value network and policy are simulated "
        "by uniform priors + random leaf values."
    )

    col1, col2 = st.columns(2)
    with col1:
        depth = st.slider("Tree depth", 2, 5, 3)
        branching = st.slider("Branching factor (actions)", 2, 6, 4)
        n_simulations = st.slider("Number of simulations", 10, 500, 100, 10)
    with col2:
        c_puct = st.slider("c_puct (exploration constant)", 0.5, 5.0, 1.5, 0.5)
        seed = st.number_input("Random seed", value=42, step=1)

    st.caption(
        f"Total nodes in tree: {branching}^0 + ... + {branching}^{depth} = "
        f"{sum(branching**d for d in range(depth + 1)):,}  "
        f"(MCTS explores a tiny fraction)"
    )

    if st.button("Run MCTS", type="primary"):
        with st.spinner("Running MCTS on toy game tree…"):
            sim_idxs, q_vals, _, visit_data = run_mcts(
                depth, branching, n_simulations, c_puct, int(seed)
            )

        st.header("Results")
        tab1, tab2 = st.tabs(["Q-value Convergence", "Action Visit Distribution"])
        with tab1:
            st.plotly_chart(
                _q_convergence_fig(sim_idxs, q_vals, branching),
                use_container_width=True,
            )
            st.caption(
                "The root Q-value estimate converges as more simulations are run.  "
                "High c_puct = more exploration; low c_puct = more exploitation of "
                "initially promising actions."
            )
        with tab2:
            st.plotly_chart(
                _visit_bar_fig(visit_data, branching),
                use_container_width=True,
            )
            st.caption(
                "MCTS concentrates visits on the most promising action (highest Q).  "
                "The visit count distribution is the policy AlphaGo would use at this node."
            )

        best = visit_data[0] if visit_data else {}
        st.header("Key Takeaways")
        st.success(
            f"After {n_simulations} simulations, MCTS selected Action {best.get('action', '?')} "
            f"({best.get('visits', 0)} visits, Q = {best.get('Q', 0):.3f}).  "
            "The visit distribution is sharply peaked — MCTS found the best action "
            f"while visiting only {n_simulations} of "
            f"{sum(branching**d for d in range(depth + 1)):,} total tree nodes."
        )
        st.info(
            "**In AlphaGo**, c_puct ≈ 5 during self-play (more exploration) and ≈ 1 at "
            "test time (more exploitation).  1600 simulations per move were used.  "
            "Each simulation takes ~2ms on a GPU — the whole move selection runs in 3 seconds."
        )
