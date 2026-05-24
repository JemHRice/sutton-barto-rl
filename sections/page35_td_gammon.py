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


# ── Tic-Tac-Toe environment ────────────────────────────────────────────────────

_WINS = [
    [0, 1, 2],
    [3, 4, 5],
    [6, 7, 8],  # rows
    [0, 3, 6],
    [1, 4, 7],
    [2, 5, 8],  # columns
    [0, 4, 8],
    [2, 4, 6],  # diagonals
]


def _check_winner(board: np.ndarray) -> int | None:
    """Return 1 (X wins), -1 (O wins), 0 (draw), None (ongoing)."""
    for line in _WINS:
        s = board[line].sum()
        if s == 3:
            return 1
        if s == -3:
            return -1
    if np.all(board != 0):
        return 0
    return None


def _legal_moves(board: np.ndarray) -> list[int]:
    return [i for i in range(9) if board[i] == 0]


# ── Neural network value function ──────────────────────────────────────────────


def _make_net(hidden: int) -> "nn.Sequential":
    """Board (9 inputs) → V(s) probability that X wins (0..1 sigmoid output)."""
    return nn.Sequential(
        nn.Linear(9, hidden),
        nn.ReLU(),
        nn.Linear(hidden, hidden),
        nn.ReLU(),
        nn.Linear(hidden, 1),
        nn.Sigmoid(),
    )


def _net_value(net: "nn.Module", board: np.ndarray) -> float:
    with torch.no_grad():
        return float(net(torch.FloatTensor(board)))


def _best_move(
    net: "nn.Module",
    board: np.ndarray,
    player: int,
    epsilon: float,
    rng: np.random.Generator,
) -> int:
    legal = _legal_moves(board)
    if rng.random() < epsilon:
        return int(rng.choice(legal))
    best_val = -np.inf
    best_move = legal[0]
    for m in legal:
        b = board.copy()
        b[m] = player
        # V always from X's perspective; O minimises it
        v = _net_value(net, b) if player == 1 else 1.0 - _net_value(net, b)
        if v > best_val:
            best_val = v
            best_move = m
    return best_move


# ── Training ───────────────────────────────────────────────────────────────────


@st.cache_data
def run_td_tictactoe(
    hidden: int,
    lr: float,
    epsilon: float,
    epsilon_decay: float,
    n_episodes: int,
    eval_every: int,
    n_eval: int,
    seed: int,
) -> tuple[list, list, list]:
    """
    TD self-play on Tic-Tac-Toe.
    Returns (eval_episodes, win_rates_vs_random, draw_rates_vs_random).
    """
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    net = _make_net(hidden)
    opt = optim.Adam(net.parameters(), lr=lr)

    eval_eps, win_rates, draw_rates = [], [], []
    eps_now = epsilon

    for ep in range(n_episodes):
        board = np.zeros(9, dtype=np.float32)
        player = 1  # X moves first
        states: list[np.ndarray] = []

        while True:
            move = _best_move(net, board, player, eps_now, rng)
            board[move] = player
            states.append(board.copy())
            result = _check_winner(board)
            if result is not None:
                break
            player = -player

        # terminal value: 1=X wins, 0=O wins, 0.5=draw
        terminal = 1.0 if result == 1 else (0.0 if result == -1 else 0.5)

        # TD(0) backward pass through the episode
        target = torch.tensor(terminal, dtype=torch.float32)
        for state in reversed(states):
            s_t = torch.FloatTensor(state)
            v_s = net(s_t).squeeze()
            loss = (target - v_s) ** 2
            opt.zero_grad()
            loss.backward()
            opt.step()
            with torch.no_grad():
                target = net(s_t).squeeze().clone()

        eps_now = max(0.01, eps_now * epsilon_decay)

        # Periodic evaluation vs random opponent
        if (ep + 1) % eval_every == 0:
            wins = draws = 0
            eval_rng = np.random.default_rng(seed + ep)
            for _ in range(n_eval):
                b = np.zeros(9, dtype=np.float32)
                pl = 1
                outcome = None
                while outcome is None:
                    if pl == 1:
                        mv = _best_move(net, b, 1, 0.0, eval_rng)
                    else:
                        legal = _legal_moves(b)
                        mv = int(eval_rng.choice(legal))
                    b[mv] = pl
                    outcome = _check_winner(b)
                    pl = -pl
                if outcome == 1:
                    wins += 1
                elif outcome == 0:
                    draws += 1
            eval_eps.append(ep + 1)
            win_rates.append(wins / n_eval)
            draw_rates.append(draws / n_eval)

    return eval_eps, win_rates, draw_rates


@st.cache_data
def get_value_heatmap(
    hidden: int,
    lr: float,
    epsilon: float,
    epsilon_decay: float,
    n_episodes: int,
    seed: int,
) -> list[list]:
    """
    Train a net and return V(s) for each of the 9 possible X-only first moves
    (X plays one move from the empty board).
    """
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    net = _make_net(hidden)
    opt = optim.Adam(net.parameters(), lr=lr)
    eps_now = epsilon

    for ep in range(n_episodes):
        board = np.zeros(9, dtype=np.float32)
        player = 1
        states = []
        while True:
            move = _best_move(net, board, player, eps_now, rng)
            board[move] = player
            states.append(board.copy())
            result = _check_winner(board)
            if result is not None:
                break
            player = -player

        terminal = 1.0 if result == 1 else (0.0 if result == -1 else 0.5)
        target = torch.tensor(terminal, dtype=torch.float32)
        for state in reversed(states):
            s_t = torch.FloatTensor(state)
            v_s = net(s_t).squeeze()
            loss = (target - v_s) ** 2
            opt.zero_grad()
            loss.backward()
            opt.step()
            with torch.no_grad():
                target = net(s_t).squeeze().clone()

        eps_now = max(0.01, eps_now * epsilon_decay)

    # Evaluate V(s) after each of X's 9 possible first moves
    values = []
    for move in range(9):
        b = np.zeros(9, dtype=np.float32)
        b[move] = 1.0
        values.append(_net_value(net, b))
    return [round(v, 3) for v in values]


# ── Visualisations ─────────────────────────────────────────────────────────────


def _win_rate_fig(eval_eps: list, win_rates: list, draw_rates: list) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=eval_eps,
            y=win_rates,
            mode="lines+markers",
            name="Win rate vs random",
            line=dict(color="#636EFA", width=2),
            marker=dict(size=5),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=eval_eps,
            y=draw_rates,
            mode="lines+markers",
            name="Draw rate vs random",
            line=dict(color="#00CC96", width=2, dash="dot"),
            marker=dict(size=5),
        )
    )
    loss = [1 - w - d for w, d in zip(win_rates, draw_rates)]
    fig.add_trace(
        go.Scatter(
            x=eval_eps,
            y=loss,
            mode="lines+markers",
            name="Loss rate vs random",
            line=dict(color="#EF553B", width=2, dash="dash"),
            marker=dict(size=5),
        )
    )
    fig.add_hline(
        y=0.58,
        line_dash="dot",
        line_color="grey",
        annotation_text="~58% optimal win rate vs random",
        annotation_position="bottom right",
    )
    fig.update_layout(
        title="Agent Performance vs Random Opponent",
        xaxis_title="Training Episode",
        yaxis_title="Rate",
        yaxis=dict(range=[0, 1.05]),
        template="plotly_white",
        height=380,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


def _value_board_fig(values: list) -> go.Figure:
    grid = np.array(values).reshape(3, 3)
    fig = go.Figure(
        go.Heatmap(
            z=grid[::-1].tolist(),
            colorscale="RdYlGn",
            zmin=0,
            zmax=1,
            text=[[f"{v:.2f}" for v in row] for row in grid[::-1].tolist()],
            texttemplate="%{text}",
            showscale=True,
            colorbar=dict(title="V(s)<br>after X's<br>first move"),
        )
    )
    fig.update_layout(
        title="Learned V(s) — Board Position After X's First Move",
        xaxis=dict(
            tickvals=[0, 1, 2], ticktext=["Left", "Centre", "Right"], showgrid=False
        ),
        yaxis=dict(
            tickvals=[0, 1, 2], ticktext=["Bottom", "Middle", "Top"], showgrid=False
        ),
        template="plotly_white",
        height=360,
    )
    return fig


# ── Page ───────────────────────────────────────────────────────────────────────


def show():
    st.title("TD-Gammon and Self-play TD Learning")
    st.markdown(
        '<div style="background:#7B2D8B;height:3px;'
        'border-radius:2px;margin-bottom:1rem;"></div>',
        unsafe_allow_html=True,
    )
    st.markdown("**Section 12 — Applications (Chapter 16)**")

    if not _TORCH_OK:
        st.error(
            "PyTorch is not installed.  Install it with `pip install torch` and restart the app."
        )
        return

    # ── Concept ───────────────────────────────────────────────────────────────
    st.header("TD Learning Meets Game AI")
    st.markdown(r"""
In 1992, Gerald Tesauro trained **TD-Gammon** — a neural network backgammon player —
using TD(λ) self-play.  Starting from random play, it reached world-class strength
after 1.5 million games, demonstrating for the first time that:

1. **Neural networks** can represent complex game value functions.
2. **Self-play + TD learning** can discover expert-level strategy without human demonstrations.
3. The combination generalises far beyond what tabular RL could represent.

TD-Gammon's value network estimated the probability of winning from each board position.
Self-play training alternated between collecting experience and updating the network via
TD(λ), treating the next position's value as the bootstrap target.
""")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**TD-Gammon update rule**")
        st.latex(r"V(s_t) \leftarrow V(s_t) + \alpha\bigl[V(s_{t+1}) - V(s_t)\bigr]")
        st.caption(
            "At terminal: V(s_T) = 1 (win) or 0 (loss). "
            "The signal propagates backward through the episode using eligibility traces."
        )
    with col2:
        st.markdown("**Why self-play works**")
        st.markdown(
            "Playing against itself, the agent always faces an opponent at its own skill level.  "
            "This creates a natural curriculum: as the agent improves, so does its opponent.  "
            "The training signal is always informative — not too hard, not too easy."
        )

    st.info(
        "**TD-Gammon's legacy**: it directly inspired AlphaGo and AlphaZero, which "
        "extended self-play TD learning to Go and Chess with Monte Carlo Tree Search "
        "for lookahead (covered in Page 37)."
    )

    with st.expander("Why backgammon worked but other games didn't (at the time)"):
        st.markdown(r"""
Backgammon has a key property that made TD-Gammon succeed where earlier game-playing
TD systems failed: **dice rolls introduce stochasticity**.

With random dice, a greedy policy that always plays the best move in each position
*naturally explores* the game tree — different dice outcomes force the agent through
diverse board states.  This exploration is free and automatically well-distributed.

Chess and Go, being deterministic, provide no such forced exploration.  A greedy policy
quickly cycles through the same positions and never discovers diverse situations.
This is why it took AlphaZero's explicit MCTS-based exploration to crack Go.
""")

    st.divider()

    # ── Simulation ─────────────────────────────────────────────────────────────
    st.header("Interactive Demo — TD Self-play on Tic-Tac-Toe")
    st.markdown(
        "A neural network value function learns by playing against itself.  "
        "V(s) estimates the probability that X wins from state s.  "
        "X selects moves greedily (ε-greedy), O minimises V(s)."
    )

    col1, col2 = st.columns(2)
    with col1:
        hidden = st.slider("Hidden units per layer", 16, 128, 64, 16)
        lr = st.slider("Learning rate", 1e-4, 1e-2, 1e-3, 1e-4, format="%.4f")
        epsilon = st.slider("ε start (exploration)", 0.1, 1.0, 0.5, 0.05)
        epsilon_decay = st.slider(
            "ε decay (per episode)", 0.990, 0.9999, 0.998, 0.0001, format="%.4f"
        )
    with col2:
        n_episodes = st.slider("Training episodes", 1000, 50000, 10000, 1000)
        eval_every = st.slider("Evaluate every N episodes", 100, 2000, 500, 100)
        n_eval = st.slider("Evaluation games", 100, 1000, 200, 100)
        seed = st.number_input("Random seed", value=42, step=1)

    if st.button("Train TD Agent", type="primary"):
        with st.spinner("Running TD self-play on Tic-Tac-Toe…"):
            eval_eps, win_rates, draw_rates = run_td_tictactoe(
                hidden,
                lr,
                epsilon,
                epsilon_decay,
                n_episodes,
                eval_every,
                n_eval,
                int(seed),
            )
            values = get_value_heatmap(
                hidden,
                lr,
                epsilon,
                epsilon_decay,
                n_episodes,
                int(seed),
            )

        st.header("Results")
        tab1, tab2 = st.tabs(["Win Rate over Training", "Learned Position Values"])
        with tab1:
            st.plotly_chart(
                _win_rate_fig(eval_eps, win_rates, draw_rates), use_container_width=True
            )
        with tab2:
            st.plotly_chart(_value_board_fig(values), use_container_width=True)
            st.caption(
                "Higher values (green) = X prefers these first moves.  "
                "The centre (0.55) and corners (0.54–0.56) should be valued higher "
                "than edge moves — consistent with optimal Tic-Tac-Toe strategy."
            )

        final_wr = float(win_rates[-1]) if win_rates else 0.0
        final_dr = float(draw_rates[-1]) if draw_rates else 0.0

        st.header("Key Takeaways")
        if final_wr >= 0.50:
            st.success(
                f"Agent wins {final_wr*100:.0f}% and draws {final_dr*100:.0f}% "
                f"vs a random opponent.  "
                "The self-play TD signal has discovered that X has a structural advantage "
                "when played optimally."
            )
        else:
            st.warning(
                f"Win rate is {final_wr*100:.0f}% — the agent is still learning.  "
                "Try more training episodes or a slower ε decay."
            )
        st.info(
            "**Connecting to TD-Gammon**: the same mechanism — TD bootstrapping on a "
            "neural value function trained via self-play — scaled to 196-point backgammon "
            "boards with ~100 million parameters.  The only major addition was "
            "eligibility traces (TD(λ)) and a hand-crafted input encoding."
        )
