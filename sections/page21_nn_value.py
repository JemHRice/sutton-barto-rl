import numpy as np
import streamlit as st
import plotly.graph_objects as go

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim

    _TORCH_OK = True
except ImportError:
    _TORCH_OK = False

from utils.random_walk_1000 import RandomWalk1000, compute_true_values
from utils.feature_bases import BASIS_REGISTRY, make_feature_matrix

# ── Helpers ────────────────────────────────────────────────────────────────────


@st.cache_data
def _true_values() -> np.ndarray:
    return compute_true_values()


# ── MLP definition (inside function so it can run without torch at import time)


def _build_mlp(hidden: int, n_hidden: int):
    """Return a simple MLP: 1 → hidden×n_hidden → 1."""
    layers: list[nn.Module] = []
    in_dim = 1
    for _ in range(n_hidden):
        layers += [nn.Linear(in_dim, hidden), nn.ReLU()]
        in_dim = hidden
    layers.append(nn.Linear(in_dim, 1))
    return nn.Sequential(*layers)


# ── Training routines ──────────────────────────────────────────────────────────


@st.cache_data
def run_nn_gradient_mc(
    hidden: int,
    n_hidden: int,
    alpha: float,
    n_episodes: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Gradient MC with an MLP value function approximator.
    Returns (rms_errors [n_episodes], V_learned [1000]).
    """
    torch.manual_seed(seed)
    net = _build_mlp(hidden, n_hidden)
    opt = optim.Adam(net.parameters(), lr=alpha)
    env = RandomWalk1000(seed=seed)
    V_true = _true_values()
    rms = np.zeros(n_episodes)
    states_t = torch.linspace(1 / 1000, 1.0, 1000).unsqueeze(1)

    for ep in range(n_episodes):
        traj: list[tuple[int, float]] = []
        s = env.reset()
        while True:
            ns, r, done = env.step()
            traj.append((s, r))
            s = ns
            if done:
                break

        G = 0.0
        visited: set[int] = set()
        loss_sum = torch.tensor(0.0)
        count = 0
        for t in range(len(traj) - 1, -1, -1):
            sv, rv = traj[t]
            G = rv + G
            if sv not in visited:
                visited.add(sv)
                x = torch.tensor([[sv / RandomWalk1000.N_STATES]], dtype=torch.float32)
                v_s = net(x).squeeze()
                loss_sum = loss_sum + (G - v_s) ** 2
                count += 1

        if count > 0:
            opt.zero_grad()
            (loss_sum / count).backward()
            opt.step()

        with torch.no_grad():
            v_hat = net(states_t).squeeze().numpy()
        rms[ep] = float(np.sqrt(np.mean((v_hat - V_true) ** 2)))

    with torch.no_grad():
        V_hat = net(states_t).squeeze().numpy()
    return rms, V_hat


@st.cache_data
def run_linear_mc(
    basis_name: str,
    basis_param: int,
    alpha: float,
    n_episodes: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Gradient MC with a linear approximator for comparison."""
    V_true = _true_values()
    info = BASIS_REGISTRY[basis_name]
    basis_fn = info["fn"]
    n_feat = info["n_feat"](basis_param)
    w = np.zeros(n_feat)
    env = RandomWalk1000(seed=seed)
    rms = np.zeros(n_episodes)

    for ep in range(n_episodes):
        traj: list[tuple[int, float]] = []
        s = env.reset()
        while True:
            ns, r, done = env.step()
            traj.append((s, r))
            s = ns
            if done:
                break

        G = 0.0
        visited: set[int] = set()
        for t in range(len(traj) - 1, -1, -1):
            sv, rv = traj[t]
            G = rv + G
            if sv not in visited:
                visited.add(sv)
                phi = basis_fn(sv, RandomWalk1000.N_STATES, basis_param)
                w += alpha * (G - float(w @ phi)) * phi

        V_hat = np.array(
            [
                float(w @ basis_fn(s + 1, RandomWalk1000.N_STATES, basis_param))
                for s in range(RandomWalk1000.N_STATES)
            ]
        )
        rms[ep] = float(np.sqrt(np.mean((V_hat - V_true) ** 2)))

    V_hat = np.array(
        [
            float(w @ basis_fn(s + 1, RandomWalk1000.N_STATES, basis_param))
            for s in range(RandomWalk1000.N_STATES)
        ]
    )
    return rms, V_hat


# ── Figures ────────────────────────────────────────────────────────────────────


def _rms_fig(rms_nn: np.ndarray, rms_lin: np.ndarray, lin_label: str) -> go.Figure:
    eps = list(range(1, len(rms_nn) + 1))
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=eps,
            y=rms_nn,
            mode="lines",
            name="MLP (neural net)",
            line=dict(color="#AB63FA", width=2),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=eps,
            y=rms_lin,
            mode="lines",
            name=f"Linear ({lin_label})",
            line=dict(color="#636EFA", width=2),
        )
    )
    fig.update_layout(
        title="RMS Error vs Episodes",
        xaxis_title="Episodes",
        yaxis_title="RMS Error",
        template="plotly_white",
        height=380,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


def _value_fig(
    V_true: np.ndarray, V_nn: np.ndarray, V_lin: np.ndarray, lin_label: str
) -> go.Figure:
    states = list(range(1, RandomWalk1000.N_STATES + 1))
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=states,
            y=V_true,
            mode="lines",
            name="True V(s)",
            line=dict(color="black", width=2, dash="dot"),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=states,
            y=V_nn,
            mode="lines",
            name="MLP",
            line=dict(color="#AB63FA", width=2),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=states,
            y=V_lin,
            mode="lines",
            name=f"Linear ({lin_label})",
            line=dict(color="#636EFA", width=2),
        )
    )
    fig.update_layout(
        title="Learned Value Functions vs True",
        xaxis_title="State",
        yaxis_title="V(s)",
        template="plotly_white",
        height=380,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


# ── Page ───────────────────────────────────────────────────────────────────────


def show():
    st.title("Neural Network Value Approximation")
    st.markdown(
        '<div style="background:#7B2D8B;height:3px;'
        'border-radius:2px;margin-bottom:1rem;"></div>',
        unsafe_allow_html=True,
    )
    st.markdown("**Section 7 — Function Approximation: Prediction (Chapter 9)**")

    if not _TORCH_OK:
        st.error(
            "PyTorch is not installed.  Install it with `pip install torch` and restart the app."
        )
        return

    # ── Concept ───────────────────────────────────────────────────────────────
    st.header("Going Non-Linear")
    st.markdown(r"""
Linear approximators are powerful and theoretically tractable, but they are limited to
functions in the span of the chosen basis.  A **neural network** lifts this restriction:
with enough hidden units, a two-layer MLP is a universal function approximator.

The algorithm is still **gradient Monte Carlo**:

$$\mathbf{w} \leftarrow \mathbf{w} - \alpha \nabla_\mathbf{w} \frac{1}{2}\bigl[G_t - \hat{v}(S_t, \mathbf{w})\bigr]^2$$

but $\hat{v}(s, \mathbf{w})$ is now the output of the network, and $\mathbf{w}$ refers to
all network parameters.  Back-propagation computes the gradient automatically.

The input is simply $x = s / n_\text{states} \in (0, 1]$ — a single normalised scalar.
""")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Linear approximator**")
        st.latex(r"\hat{v}(s, \mathbf{w}) = \mathbf{w}^\top \boldsymbol{\phi}(s)")
        st.caption(
            "Gradient is $\\boldsymbol{\\phi}(s)$ — closed form.  "
            "Converges to the MSVE minimum for the chosen basis.  "
            "Theoretically well-understood."
        )
    with col2:
        st.markdown("**Neural network approximator**")
        st.latex(r"\hat{v}(s, \mathbf{w}) = f_\mathbf{w}(s)")
        st.caption(
            "Gradient computed by back-propagation.  "
            "Can represent a richer class of functions — but no convergence guarantee for "
            "semi-gradient methods (though gradient MC still has one in theory)."
        )

    st.info(
        "**Why semi-gradient TD + neural nets is tricky.**  Semi-gradient TD(0) with a "
        "linear approximator converges to the TD fixed point — but with a non-linear "
        "approximator like an MLP, the update is no longer a contraction and may diverge "
        "(the *deadly triad*).  Deep Q-Networks (DQN) add experience replay and a "
        "target network to stabilise training — covered in Chapter 16."
    )

    with st.expander(
        "The deadly triad: function approximation + bootstrapping + off-policy"
    ):
        st.markdown(r"""
Convergence of TD methods breaks down when **all three** of these are present:

1. **Function approximation** — the value function is parameterised (e.g. a neural net)
2. **Bootstrapping** — targets depend on current estimates (TD, not MC)
3. **Off-policy training** — the update distribution differs from the behaviour policy

With any two of the three, stable algorithms exist.  Adding the third can cause
divergence.  Gradient MC avoids this because it does not bootstrap (condition 2 is absent),
making it safe to combine with neural nets and off-policy data.
""")

    st.divider()

    # ── Simulation ─────────────────────────────────────────────────────────────
    st.header("Interactive Simulation")
    st.markdown(
        "Train an MLP alongside a linear approximator using gradient MC on the 1000-state "
        "random walk.  Compare convergence speed and final approximation quality."
    )

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**MLP settings**")
        hidden = st.slider("Hidden units per layer", 8, 256, 64, 8)
        n_hidden = st.slider("Hidden layers", 1, 4, 2)
        alpha_nn = st.slider("α — MLP (Adam)", 1e-4, 1e-2, 1e-3, 1e-4, format="%.4f")
        n_params = hidden * (1 + hidden * (n_hidden - 1) + 1) + hidden * n_hidden + 1
        st.caption(f"Approx. network parameters: **~{n_params}**")
    with col2:
        st.markdown("**Linear baseline settings**")
        basis_name = st.selectbox("Feature basis", list(BASIS_REGISTRY.keys()), index=2)
        info = BASIS_REGISTRY[basis_name]
        plabel, pdef, pmin, pmax, pstep = info["param"]
        basis_param = st.slider(plabel, pmin, pmax, pdef, pstep)
        alpha_lin = st.slider("α — Linear", 0.0001, 0.05, 0.002, 0.0001, format="%.4f")
        n_feat = info["n_feat"](basis_param)
        st.caption(f"Linear feature dimension: **{n_feat}**")

    n_episodes = st.slider("Episodes", 100, 2000, 500, 100)
    seed = st.number_input("Random seed", value=42, step=1)

    if st.button("Run Comparison", type="primary"):
        with st.spinner("Training MLP and linear approximator…"):
            rms_nn, V_nn = run_nn_gradient_mc(
                hidden, n_hidden, alpha_nn, n_episodes, int(seed)
            )
            rms_lin, V_lin = run_linear_mc(
                basis_name, basis_param, alpha_lin, n_episodes, int(seed)
            )
            V_true = _true_values()

        st.header("Results")
        st.plotly_chart(_rms_fig(rms_nn, rms_lin, basis_name), use_container_width=True)
        st.plotly_chart(
            _value_fig(V_true, V_nn, V_lin, basis_name), use_container_width=True
        )

        col_a, col_b = st.columns(2)
        with col_a:
            st.metric("MLP — final RMS", f"{rms_nn[-1]:.4f}")
        with col_b:
            st.metric(f"Linear ({basis_name}) — final RMS", f"{rms_lin[-1]:.4f}")

        st.header("Key Takeaways")
        winner = "MLP" if rms_nn[-1] < rms_lin[-1] else f"Linear ({basis_name})"
        st.success(
            f"**{winner}** achieved lower final RMS with these settings. "
            "On this nearly-linear task the MLP needs many more gradient steps to match a "
            "well-chosen linear basis — but the MLP has no built-in knowledge of the function's "
            "shape and must discover it from data."
        )
        st.info(
            "Try a **Fourier Cosine order 2** linear baseline — it often achieves near-zero "
            "RMS because the true value function is almost linear, matching what the Fourier "
            "basis naturally represents.  An MLP with a small α and enough episodes can "
            "eventually match it, but usually needs more data."
        )
