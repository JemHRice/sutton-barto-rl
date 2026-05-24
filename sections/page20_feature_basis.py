import numpy as np
import plotly.graph_objects as go
import streamlit as st

from utils.random_walk_1000 import RandomWalk1000, compute_true_values
from utils.feature_bases import BASIS_REGISTRY, make_feature_matrix


@st.cache_data
def _true_values() -> np.ndarray:
    return compute_true_values()


@st.cache_data
def _feature_matrix(basis_name: str, basis_param: int) -> np.ndarray:
    info = BASIS_REGISTRY[basis_name]
    return make_feature_matrix(info["fn"], RandomWalk1000.N_STATES, basis_param)


# ── Figures ────────────────────────────────────────────────────────────────────


def _basis_heatmap(Phi: np.ndarray, basis_name: str, basis_param: int) -> go.Figure:
    fig = go.Figure(
        go.Heatmap(
            z=Phi,
            colorscale="RdBu",
            zmid=0,
            colorbar=dict(title="φ value"),
        )
    )
    fig.update_layout(
        title=f"Feature Matrix Φ — {basis_name} (param={basis_param})",
        xaxis_title="Feature index",
        yaxis_title="State",
        template="plotly_white",
        height=350,
    )
    return fig


def _individual_features_fig(Phi: np.ndarray, n_show: int) -> go.Figure:
    n_feat = Phi.shape[1]
    n_show = min(n_show, n_feat)
    states = list(range(1, RandomWalk1000.N_STATES + 1))
    colours = [
        "#636EFA",
        "#EF553B",
        "#00CC96",
        "#AB63FA",
        "#FFA15A",
        "#19D3F3",
        "#FF6692",
        "#B6E880",
        "#FF97FF",
        "#FECB52",
    ]
    fig = go.Figure()
    for i in range(n_show):
        fig.add_trace(
            go.Scatter(
                x=states,
                y=Phi[:, i],
                mode="lines",
                name=f"φ_{i}",
                line=dict(color=colours[i % len(colours)], width=1.5),
            )
        )
    fig.update_layout(
        title=f"First {n_show} Basis Functions",
        xaxis_title="State",
        yaxis_title="φ(s)",
        template="plotly_white",
        height=350,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


def _approximation_fig(
    V_true: np.ndarray, Phi: np.ndarray, basis_name: str
) -> go.Figure:
    w = np.linalg.lstsq(Phi, V_true, rcond=None)[0]
    V_hat = Phi @ w
    rms = float(np.sqrt(np.mean((V_hat - V_true) ** 2)))
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
            y=V_hat,
            mode="lines",
            name=f"Best-fit ({basis_name})",
            line=dict(color="#636EFA", width=2),
        )
    )
    fig.update_layout(
        title=f"Best Linear Approximation — RMS = {rms:.4f}",
        xaxis_title="State",
        yaxis_title="V(s)",
        template="plotly_white",
        height=350,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


@st.cache_data
def _compare_all_bases(
    params: tuple[tuple[str, int], ...],
) -> dict[str, tuple[float, int]]:
    """Returns {basis_name: (rms, n_features)} for each basis."""
    V_true = _true_values()
    results: dict[str, tuple[float, int]] = {}
    params_dict = dict(params)
    for name, info in BASIS_REGISTRY.items():
        p = params_dict[name]
        Phi = make_feature_matrix(info["fn"], RandomWalk1000.N_STATES, p)
        w = np.linalg.lstsq(Phi, V_true, rcond=None)[0]
        rms = float(np.sqrt(np.mean((Phi @ w - V_true) ** 2)))
        results[name] = (rms, info["n_feat"](p))
    return results


# ── Page ───────────────────────────────────────────────────────────────────────


def show():
    st.title("Feature Basis Explorer")
    st.markdown(
        '<div style="background:#7B2D8B;height:3px;'
        'border-radius:2px;margin-bottom:1rem;"></div>',
        unsafe_allow_html=True,
    )
    st.markdown("**Section 7 — Function Approximation: Prediction (Chapter 9)**")

    st.header("What Is a Feature Basis?")
    st.markdown(r"""
A **feature vector** $\boldsymbol{\phi}(s) \in \mathbb{R}^d$ encodes state $s$ so that a linear
weight vector $\mathbf{w}$ can approximate the value function:

$$\hat{v}(s, \mathbf{w}) = \mathbf{w}^\top \boldsymbol{\phi}(s)$$

The choice of basis determines what the approximator can represent.  A basis with $d$ features
can only represent functions in the span of those $d$ vectors — if the true $v_\pi$ lies
outside that span, some **approximation error** is irreducible regardless of how long you train.
""")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**State Aggregation**")
        st.latex(r"\phi_i(s) = \mathbf{1}[s \in \text{group } i]")
        st.caption(
            "Partitions states into disjoint groups.  Fast and interpretable, but produces "
            "a staircase approximation — cannot represent smooth gradients within a group."
        )
        st.markdown("**Polynomial**")
        st.latex(r"\phi_i(s) = x^i, \quad x = s / n_\text{states}")
        st.caption(
            "Degree-$d$ polynomial: $d+1$ features.  Can fit smooth curves; high-degree "
            "polynomials can oscillate near boundaries (Runge's phenomenon)."
        )
    with col2:
        st.markdown("**Fourier Cosine**")
        st.latex(r"\phi_i(s) = \cos(i \pi x), \quad x = s / n_\text{states}")
        st.caption(
            "Order-$k$ Fourier basis: $k+1$ orthogonal features.  Excellent at capturing "
            "smooth, gradual structure.  Often outperforms polynomials of the same dimension."
        )
        st.markdown("**Radial Basis Functions (RBF)**")
        st.latex(r"\phi_i(s) = \exp\!\left(-\frac{(x - c_i)^2}{2\sigma^2}\right)")
        st.caption(
            "Gaussian bumps at evenly-spaced centres.  Local support means each weight "
            "affects a neighbourhood rather than the whole state space."
        )

    st.info(
        "**Best-fit RMS** on this page uses **least-squares projection** — the optimal "
        "linear weights for the given basis — not gradient descent.  This reveals the "
        "*representational capacity* of each basis: the error floor that any gradient "
        "algorithm approaches in the infinite-data limit."
    )

    st.divider()

    # ── Interactive explorer ───────────────────────────────────────────────────
    st.header("Interactive Explorer")

    col1, col2 = st.columns(2)
    with col1:
        basis_name = st.selectbox("Feature basis", list(BASIS_REGISTRY.keys()), index=2)
        info = BASIS_REGISTRY[basis_name]
        plabel, pdef, pmin, pmax, pstep = info["param"]
        basis_param = st.slider(plabel, pmin, pmax, pdef, pstep)
    with col2:
        n_feat = info["n_feat"](basis_param)
        n_show = st.slider(
            "Individual features to plot", 1, min(10, n_feat), min(6, n_feat)
        )
        st.caption(f"Feature vector dimension: **{n_feat}**")

    Phi = _feature_matrix(basis_name, basis_param)
    V_true = _true_values()

    tab1, tab2, tab3 = st.tabs(
        ["Feature Matrix", "Individual Basis Functions", "Best-fit Approximation"]
    )

    with tab1:
        st.plotly_chart(
            _basis_heatmap(Phi, basis_name, basis_param), use_container_width=True
        )
        st.caption(
            "Each row is a state (1–1000); each column is one feature.  "
            "State aggregation shows hard block boundaries; Fourier shows smooth oscillations."
        )

    with tab2:
        st.plotly_chart(_individual_features_fig(Phi, n_show), use_container_width=True)
        st.caption(
            "Each line is one basis function $\\phi_i(s)$ plotted over all 1000 states.  "
            "Together these vectors span the representable function space."
        )

    with tab3:
        st.plotly_chart(
            _approximation_fig(V_true, Phi, basis_name), use_container_width=True
        )
        st.caption(
            "Least-squares best fit to the true value function.  "
            "Increasing the parameter (order / groups / degree) generally reduces RMS "
            "until the basis is expressive enough to capture $v_\\pi$."
        )

    st.divider()

    # ── Cross-basis comparison ─────────────────────────────────────────────────
    st.header("Cross-Basis Comparison")
    st.markdown(
        "Set a parameter for each basis, then compare their best-fit RMS on the same true "
        "value function at a glance."
    )

    cols = st.columns(len(BASIS_REGISTRY))
    cmp_params: dict[str, int] = {}
    for col, (name, binfo) in zip(cols, BASIS_REGISTRY.items()):
        with col:
            lbl, dfl, bmin, bmax, bstep = binfo["param"]
            cmp_params[name] = st.slider(
                name, bmin, bmax, dfl, bstep, key=f"cmp_{name}"
            )

    if st.button("Compare Bases", type="primary"):
        param_tuple = tuple(cmp_params.items())
        rms_map = _compare_all_bases(param_tuple)  # type: ignore[arg-type]

        names = list(rms_map.keys())
        rms_vals = [rms_map[n][0] for n in names]
        n_feats = [rms_map[n][1] for n in names]
        labels = [f"{n}<br>({f} feat)" for n, f in zip(names, n_feats)]

        fig = go.Figure(
            go.Bar(
                x=labels,
                y=rms_vals,
                marker_color=["#636EFA", "#EF553B", "#00CC96", "#AB63FA"],
                text=[f"{v:.4f}" for v in rms_vals],
                textposition="outside",
            )
        )
        fig.update_layout(
            title="Best-fit RMS by Basis (lower is better)",
            xaxis_title="Basis",
            yaxis_title="RMS Error",
            template="plotly_white",
            height=380,
            yaxis=dict(range=[0, max(rms_vals) * 1.3]),
        )
        st.plotly_chart(fig, use_container_width=True)

        best = min(rms_map, key=lambda k: rms_map[k][0])
        st.success(
            f"**{best}** achieved the lowest best-fit RMS "
            f"({rms_map[best][0]:.4f}) with {rms_map[best][1]} features."
        )

    with st.expander("Why does Fourier often win?"):
        st.markdown(r"""
The true value function for the 1000-state random walk is nearly linear:
$v^*(s) \approx (2s - 1001)/1001$.  A Fourier cosine basis captures the dominant
low-frequency component directly: $\phi_1(s) = \cos(\pi x)$ approximates a linear
ramp, and $\phi_0(s) = 1$ provides the constant offset.  Order 2 typically
achieves near-zero RMS.

Polynomials can also fit this well (degree 1 suffices for a linear $v^*$), but
face numerical conditioning problems at higher degrees.  State aggregation is
inherently stepwise and needs many groups to approach a smooth fit.  RBFs depend
on $\sigma$ — with the default $\sigma=0.1$ and evenly-spaced centres, enough RBFs
can fit the function but require more features than Fourier for the same accuracy.
""")
