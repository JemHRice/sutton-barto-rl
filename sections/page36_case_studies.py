import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

# ── Data ───────────────────────────────────────────────────────────────────────

SYSTEMS = [
    {
        "name": "TD-Gammon",
        "year": 1992,
        "domain": "Backgammon",
        "org": "IBM",
        "algo": "TD(λ) + Neural Net",
        "key": "First neural-net game AI to reach expert level via self-play",
        "elo_vs_best": 100,  # approximate ELO above random; rough relative scale
        "params": 8_000,
        "compute": 1,
        "colour": "#636EFA",
    },
    {
        "name": "Deep Blue",
        "year": 1997,
        "domain": "Chess",
        "org": "IBM",
        "algo": "Alpha-Beta Search (not RL)",
        "key": "Defeated Kasparov — engineering rather than learning",
        "elo_vs_best": 2850,
        "params": 0,  # handcrafted evaluation function
        "compute": 1_000,
        "colour": "#EF553B",
    },
    {
        "name": "Atari DQN",
        "year": 2015,
        "domain": "49 Atari games",
        "org": "DeepMind",
        "algo": "DQN (TD + replay + target net)",
        "key": "Human-level control from raw pixels on diverse tasks",
        "elo_vs_best": None,
        "params": 1_800_000,
        "compute": 50_000,
        "colour": "#00CC96",
    },
    {
        "name": "AlphaGo",
        "year": 2016,
        "domain": "Go",
        "org": "DeepMind",
        "algo": "Policy gradient + Value net + MCTS",
        "key": "Defeated Lee Sedol — decades ahead of predictions",
        "elo_vs_best": 3500,
        "params": 12_000_000,
        "compute": 50_000_000,
        "colour": "#AB63FA",
    },
    {
        "name": "AlphaZero",
        "year": 2017,
        "domain": "Chess / Go / Shogi",
        "org": "DeepMind",
        "algo": "MCTS + policy/value net + self-play",
        "key": "Mastered three games from scratch, no human data",
        "elo_vs_best": 3700,
        "params": 20_000_000,
        "compute": 5_000_000_000,
        "colour": "#FFA15A",
    },
    {
        "name": "OpenAI Five",
        "year": 2019,
        "domain": "Dota 2",
        "org": "OpenAI",
        "algo": "PPO + LSTM + self-play",
        "key": "Defeated world champions in a real-time team game",
        "elo_vs_best": None,
        "params": 158_000_000,
        "compute": 800_000_000_000,
        "colour": "#19D3F3",
    },
    {
        "name": "AlphaStar",
        "year": 2019,
        "domain": "StarCraft II",
        "org": "DeepMind",
        "algo": "Policy gradient + league-based self-play",
        "key": "Grandmaster level in a game with huge action and state spaces",
        "elo_vs_best": None,
        "params": 200_000_000,
        "compute": 1_000_000_000_000,
        "colour": "#FF6692",
    },
    {
        "name": "ChatGPT (RLHF)",
        "year": 2022,
        "domain": "Language",
        "org": "OpenAI",
        "algo": "PPO + reward model from human feedback",
        "key": "RL fine-tuning of LLMs — largest deployment of RL to date",
        "elo_vs_best": None,
        "params": 175_000_000_000,
        "compute": None,
        "colour": "#B6E880",
    },
]


# ── Visualisations ─────────────────────────────────────────────────────────────


def _timeline_fig() -> go.Figure:
    domains = sorted({s["domain"] for s in SYSTEMS})
    domain_y = {d: i for i, d in enumerate(domains)}

    fig = go.Figure()
    for sys in SYSTEMS:
        y = domain_y[sys["domain"]]
        fig.add_trace(
            go.Scatter(
                x=[sys["year"]],
                y=[y],
                mode="markers+text",
                name=sys["name"],
                text=[sys["name"]],
                textposition="top center",
                marker=dict(
                    size=14, color=sys["colour"], line=dict(width=1, color="white")
                ),
                hovertemplate=(
                    f"<b>{sys['name']}</b> ({sys['year']})<br>"
                    f"Domain: {sys['domain']}<br>"
                    f"Algorithm: {sys['algo']}<br>"
                    f"{sys['key']}<extra></extra>"
                ),
            )
        )

    fig.update_layout(
        title="Major RL Milestones",
        xaxis=dict(title="Year", range=[1990, 2025], dtick=2),
        yaxis=dict(
            tickvals=list(domain_y.values()),
            ticktext=list(domain_y.keys()),
            title="",
        ),
        showlegend=False,
        template="plotly_white",
        height=450,
    )
    return fig


def _params_fig() -> go.Figure:
    names = [s["name"] for s in SYSTEMS if s["params"] > 0]
    params = [s["params"] for s in SYSTEMS if s["params"] > 0]
    years = [s["year"] for s in SYSTEMS if s["params"] > 0]
    cols = [s["colour"] for s in SYSTEMS if s["params"] > 0]
    fig = go.Figure(
        go.Bar(
            x=names,
            y=params,
            marker_color=cols,
            text=[f"{p:,}" for p in params],
            textposition="outside",
        )
    )
    fig.update_layout(
        title="Number of Parameters",
        xaxis_title="System",
        yaxis=dict(title="Parameters (log scale)", type="log"),
        template="plotly_white",
        height=380,
    )
    return fig


def _algo_comparison_fig(selected: list[str]) -> go.Figure:
    sel = [s for s in SYSTEMS if s["name"] in selected]
    if not sel:
        return go.Figure()

    categories = [
        "Self-play",
        "Neural Net",
        "Search / Lookahead",
        "Multi-agent",
        "Continuous Action",
    ]
    scores = {
        "TD-Gammon": [5, 3, 1, 0, 0],
        "Deep Blue": [0, 0, 5, 0, 0],
        "Atari DQN": [0, 5, 1, 0, 0],
        "AlphaGo": [5, 5, 5, 0, 0],
        "AlphaZero": [5, 5, 5, 0, 0],
        "OpenAI Five": [5, 5, 1, 5, 0],
        "AlphaStar": [5, 5, 2, 5, 3],
        "ChatGPT (RLHF)": [0, 5, 0, 0, 2],
    }
    fig = go.Figure()
    for sys in sel:
        sc = scores.get(sys["name"], [0] * 5)
        fig.add_trace(
            go.Scatterpolar(
                r=sc + [sc[0]],
                theta=categories + [categories[0]],
                fill="toself",
                name=sys["name"],
                line=dict(color=sys["colour"]),
                opacity=0.6,
            )
        )
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 5])),
        showlegend=True,
        title="Algorithmic Comparison (qualitative)",
        template="plotly_white",
        height=420,
    )
    return fig


# ── Page ───────────────────────────────────────────────────────────────────────


def show():
    st.title("Real-World RL — Case Studies")
    st.markdown(
        '<div style="background:#7B2D8B;height:3px;'
        'border-radius:2px;margin-bottom:1rem;"></div>',
        unsafe_allow_html=True,
    )
    st.markdown("**Section 12 — Applications (Chapter 16)**")

    # ── Concept ───────────────────────────────────────────────────────────────
    st.header("From Theory to Practice")
    st.markdown("""
Reinforcement learning has moved from academic benchmarks to real-world systems that
affect millions of people.  The progression from TD-Gammon (1992) to ChatGPT's RLHF
training (2022) traces thirty years of algorithmic advances and scaling:

- **Function approximation** allowed RL to handle continuous, high-dimensional state spaces.
- **Experience replay and target networks** (DQN) stabilised training with neural networks.
- **Self-play** provides unlimited on-policy data without human demonstrations.
- **Monte Carlo Tree Search** combines planning with learned priors for combinatorial domains.
- **Policy gradients and PPO** scaled to real-time multi-agent games.
- **RLHF** brought RL to language modelling, aligning large models to human preferences.
""")

    st.info(
        "All the algorithms in this app — from TD(0) to Actor-Critic to PPO — appear in "
        "some form in at least one of the systems below.  The S&B textbook covers "
        "the theory that makes each one work."
    )

    st.divider()

    # ── Timeline ──────────────────────────────────────────────────────────────
    st.header("Milestones Timeline")
    st.plotly_chart(_timeline_fig(), use_container_width=True)
    st.caption("Hover over a point to see the algorithm and key achievement.")

    st.divider()

    # ── System cards ──────────────────────────────────────────────────────────
    st.header("System Profiles")
    for sys in SYSTEMS:
        with st.expander(f"**{sys['name']}** — {sys['year']} · {sys['org']}"):
            col1, col2 = st.columns([2, 1])
            with col1:
                st.markdown(f"**Domain:** {sys['domain']}")
                st.markdown(f"**Algorithm:** {sys['algo']}")
                st.markdown(f"**Key achievement:** {sys['key']}")
            with col2:
                if sys["params"] and sys["params"] > 0:
                    st.metric("Parameters", f"{sys['params']:,}")
                else:
                    st.metric("Parameters", "Handcrafted")

    st.divider()

    # ── Parameters chart ──────────────────────────────────────────────────────
    st.header("Scale Over Time")
    st.plotly_chart(_params_fig(), use_container_width=True)
    st.caption(
        "Parameter counts have grown by ~7 orders of magnitude from TD-Gammon (8k) "
        "to ChatGPT (175B).  Compute has grown even faster."
    )

    st.divider()

    # ── Algorithmic comparison ────────────────────────────────────────────────
    st.header("Algorithmic Comparison")
    st.markdown("Select systems to compare on key algorithmic dimensions.")
    all_names = [s["name"] for s in SYSTEMS]
    selected = st.multiselect(
        "Systems to compare",
        all_names,
        default=["TD-Gammon", "Atari DQN", "AlphaGo", "ChatGPT (RLHF)"],
    )
    if selected:
        st.plotly_chart(_algo_comparison_fig(selected), use_container_width=True)
        st.caption(
            "Scores are qualitative (0–5).  Self-play = relies on self-generated data; "
            "Search = explicit tree search at inference; Multi-agent = trained in multi-agent setting."
        )

    st.divider()

    # ── Takeaways ──────────────────────────────────────────────────────────────
    st.header("Key Takeaways")
    st.success(
        "Every system in this timeline applies the core ideas from S&B: "
        "value functions, policy gradients, bootstrapping, and the exploration–exploitation "
        "balance.  What changed over 30 years is scale, stability techniques, and "
        "the ability to represent complex functions with deep neural networks."
    )
    st.info(
        "**What comes next**: Page 37 dives into AlphaGo's architecture — the most "
        "sophisticated combination of RL ideas on this list — and shows how MCTS "
        "and learned policy/value networks work together."
    )
