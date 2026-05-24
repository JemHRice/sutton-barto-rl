"""
RL Foundations: Interactive Sutton & Barto companion
Two-mode app: Tabular RL (Ch. 1-8) and Deep RL (Ch. 9-13)
"""

import importlib
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

import streamlit as st

st.set_page_config(
    page_title="RL Foundations",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .sidebar-section {
        font-size: 0.72rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: #aaa;
        margin-top: 1rem;
        margin-bottom: 0.1rem;
        padding-left: 0.1rem;
    }
    .block-container { padding-top: 1.5rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Page registry ──────────────────────────────────────────────────────────

TABULAR_SECTIONS = [
    {
        "label": "Section 1 · Bandit Algorithms",
        "key": "tab_s1",
        "pages": [
            ("1 — ε-Greedy Bandit", "sections.page1_epsilon_greedy"),
            ("2 — UCB", "sections.page2_ucb"),
            ("3 — Thompson Sampling", "sections.page3_thompson"),
        ],
    },
    {
        "label": "Section 2 · Dynamic Programming",
        "key": "tab_s2",
        "pages": [
            ("4 — Policy Evaluation", "sections.page4_policy_eval"),
            ("5 — Policy Iteration", "sections.page5_policy_iteration"),
            ("6 — Value Iteration", "sections.page6_value_iteration"),
        ],
    },
    {
        "label": "Section 3 · Monte Carlo Methods",
        "key": "tab_s3",
        "pages": [
            ("7 — MC Prediction", "sections.page7_mc_prediction"),
            ("8 — MC Control", "sections.page8_mc_control"),
            ("9 — Solving Blackjack", "sections.page9_blackjack"),
        ],
    },
    {
        "label": "Section 4 · Temporal Difference",
        "key": "tab_s4",
        "pages": [
            ("10 — TD(0) Prediction", "sections.page10_td0_prediction"),
            ("11 — Windy GridWorld", "sections.page11_windy_gridworld"),
            ("12 — Cliff Walking", "sections.page12_cliff_walking"),
        ],
    },
    {
        "label": "Section 5 · n-step Bootstrapping",
        "key": "tab_s5",
        "pages": [
            ("13 — n-step TD Prediction", "sections.page13_nstep_td"),
            ("14 — n-step SARSA", "sections.page14_nstep_sarsa"),
            ("15 — Tree Backup vs IS", "sections.page15_tree_backup"),
        ],
    },
    {
        "label": "Section 6 · Planning and Learning",
        "key": "tab_s6",
        "pages": [
            ("16 — Dyna-Q", "sections.page16_dynaq"),
            ("17 — Dyna-Q+ Changing Maze", "sections.page17_dynaq_plus"),
            ("18 — Prioritised Sweeping", "sections.page18_prioritized"),
        ],
    },
]

TABULAR_SUMMARY = ("📋 Summary — Key Concepts", "sections.page_summary")

DEEP_SECTIONS = [
    {
        "label": "Bridge · Neural Networks",
        "key": "deep_bridge",
        "pages": [
            ("Bridge A — Value Function Approx.", "sections.bridge_a_value_approx"),
            ("Bridge B — Semi-Gradient Methods", "sections.bridge_b_semi_gradient"),
        ],
    },
    {
        "label": "Section 7 · FA: Prediction",
        "key": "deep_s7",
        "pages": [
            ("19 — Gradient MC vs Semi-grad TD", "sections.page19_gradient_mc_td"),
            ("20 — Feature Basis Explorer", "sections.page20_feature_basis"),
            ("21 — Neural Net Value Approx.", "sections.page21_nn_value"),
        ],
    },
    {
        "label": "Section 8 · FA: Control",
        "key": "deep_s8",
        "pages": [
            ("22 — Semi-gradient SARSA", "sections.page22_semigradient_sarsa"),
            ("23 — Mountain Car Solver", "sections.page23_mountain_car"),
            ("24 — Average vs Discounted Reward", "sections.page24_average_reward"),
        ],
    },
    {
        "label": "Section 9 · Off-Policy with FA",
        "key": "deep_s9",
        "pages": [
            (
                "25 — IS with Function Approximation",
                "sections.page25_importance_sampling",
            ),
            ("26 — Deadly Triad Demo", "sections.page26_deadly_triad"),
            ("27 — Gradient TD Methods", "sections.page27_gradient_td"),
        ],
    },
    {
        "label": "Section 10 · Eligibility Traces",
        "key": "deep_s10",
        "pages": [
            ("28 — λ-return and TD(λ)", "sections.page28_td_lambda"),
            ("29 — SARSA(λ)", "sections.page29_sarsa_lambda"),
            ("30 — Unifying n-step and Traces", "sections.page30_unifying"),
        ],
    },
    {
        "label": "Section 11 · Policy Gradients",
        "key": "deep_s11",
        "pages": [
            ("31 — REINFORCE", "sections.page31_reinforce"),
            ("32 — REINFORCE with Baseline", "sections.page32_reinforce_baseline"),
            ("33 — Actor-Critic", "sections.page33_actor_critic"),
            ("34 — PPO/TRPO Conceptual", "sections.page34_ppo"),
        ],
    },
    {
        "label": "Section 12 · Applications",
        "key": "deep_s12",
        "pages": [
            ("35 — TD-Gammon & Game Playing", "sections.page35_td_gammon"),
            ("36 — Real-World RL Case Studies", "sections.page36_case_studies"),
            ("37 — AlphaGo Architecture", "sections.page37_alphago"),
        ],
    },
    {
        "label": "Section 13 · Advanced Topics",
        "key": "deep_s13",
        "pages": [
            ("38 — Hierarchical RL & Options", "sections.page38_hierarchical_rl"),
            ("39 — Meta-RL & Exploration", "sections.page39_meta_exploration"),
            ("40 — Multi-Agent & Frontiers", "sections.page40_multiagent_frontiers"),
        ],
    },
]

# ── Flat page lookup ───────────────────────────────────────────────────────

ALL_PAGES: dict[str, str] = {}
for _sec in TABULAR_SECTIONS:
    for _name, _mod in _sec["pages"]:
        ALL_PAGES[_name] = _mod
ALL_PAGES[TABULAR_SUMMARY[0]] = TABULAR_SUMMARY[1]
for _sec in DEEP_SECTIONS:
    for _name, _mod in _sec["pages"]:
        ALL_PAGES[_name] = _mod

FIRST_TABULAR = TABULAR_SECTIONS[0]["pages"][0][0]
FIRST_DEEP = DEEP_SECTIONS[0]["pages"][0][0]

_TABULAR_PAGE_NAMES = {p[0] for s in TABULAR_SECTIONS for p in s["pages"]} | {
    TABULAR_SUMMARY[0]
}
_DEEP_PAGE_NAMES = {p[0] for s in DEEP_SECTIONS for p in s["pages"]}

# ── Session state init ─────────────────────────────────────────────────────

if "mode" not in st.session_state:
    st.session_state.mode = "home"
if "selected_page" not in st.session_state:
    st.session_state.selected_page = FIRST_TABULAR

for _sec in TABULAR_SECTIONS:
    if f"r_{_sec['key']}" not in st.session_state:
        st.session_state[f"r_{_sec['key']}"] = None
if "r_tab_sum" not in st.session_state:
    st.session_state.r_tab_sum = None
for _sec in DEEP_SECTIONS:
    if f"r_{_sec['key']}" not in st.session_state:
        st.session_state[f"r_{_sec['key']}"] = None


def _sync_radio() -> None:
    """Keep radio button state consistent with selected_page."""
    sp = st.session_state.selected_page
    for sec in TABULAR_SECTIONS:
        rk = f"r_{sec['key']}"
        names = [p[0] for p in sec["pages"]]
        st.session_state[rk] = sp if sp in names else None
    st.session_state.r_tab_sum = (
        TABULAR_SUMMARY[0] if sp == TABULAR_SUMMARY[0] else None
    )
    for sec in DEEP_SECTIONS:
        rk = f"r_{sec['key']}"
        names = [p[0] for p in sec["pages"]]
        st.session_state[rk] = sp if sp in names else None


_sync_radio()

# ── Callbacks ──────────────────────────────────────────────────────────────


def _pick(key: str) -> None:
    val = st.session_state.get(key)
    if val is not None:
        st.session_state.selected_page = val


# ── Sidebar ────────────────────────────────────────────────────────────────


def _section_has_page(sec: dict) -> bool:
    return st.session_state.selected_page in [p[0] for p in sec["pages"]]


with st.sidebar:
    st.title("RL Foundations")

    if st.button("⌂ Home", use_container_width=True):
        st.session_state.mode = "home"
        st.rerun()

    st.divider()

    if st.session_state.mode == "tabular":
        st.caption("📊 Tabular RL · Chapters 1–8")
        st.markdown("")

        for sec in TABULAR_SECTIONS:
            rk = f"r_{sec['key']}"
            page_names = [p[0] for p in sec["pages"]]
            with st.expander(sec["label"], expanded=_section_has_page(sec)):
                st.radio(
                    sec["label"],
                    page_names,
                    label_visibility="collapsed",
                    key=rk,
                    on_change=_pick,
                    args=(rk,),
                )

        st.divider()
        st.radio(
            "summary",
            [TABULAR_SUMMARY[0]],
            label_visibility="collapsed",
            key="r_tab_sum",
            on_change=_pick,
            args=("r_tab_sum",),
        )

    elif st.session_state.mode == "deep":
        st.markdown(
            '<div style="background:#7B2D8B;height:3px;'
            'border-radius:2px;margin-bottom:0.75rem;"></div>',
            unsafe_allow_html=True,
        )
        st.caption("🧠 Deep RL · Chapters 9–13")
        st.markdown("")

        for sec in DEEP_SECTIONS:
            rk = f"r_{sec['key']}"
            page_names = [p[0] for p in sec["pages"]]
            with st.expander(sec["label"], expanded=_section_has_page(sec)):
                st.radio(
                    sec["label"],
                    page_names,
                    label_visibility="collapsed",
                    key=rk,
                    on_change=_pick,
                    args=(rk,),
                )

    st.divider()
    st.markdown(
        "<small>📘 Based on <em>Reinforcement Learning: An Introduction</em>"
        " — Sutton &amp; Barto (2nd ed.)<br><br>"
        "Tabular methods: NumPy only.<br>"
        "Deep RL: PyTorch + Gymnasium.</small>",
        unsafe_allow_html=True,
    )

# ── Landing page ───────────────────────────────────────────────────────────


def show_landing() -> None:
    st.title("RL Foundations")
    st.markdown(
        "#### An interactive companion to *Reinforcement Learning: An Introduction*"
        " — Sutton & Barto (2nd ed.)"
    )
    st.divider()

    col1, col2 = st.columns(2, gap="large")

    with col1:
        st.subheader("📊 Tabular RL")
        st.caption("Chapters 1–8 · Pure NumPy · No GPU required")
        st.markdown("""
- Multi-Armed Bandits
- Dynamic Programming
- Monte Carlo Methods
- Temporal Difference Learning
- n-step Bootstrapping
- Planning & Learning (Dyna-Q)
            """)
        if st.button(
            "Enter Tabular RL →",
            type="primary",
            use_container_width=True,
            key="btn_tabular",
        ):
            st.session_state.mode = "tabular"
            if st.session_state.selected_page not in _TABULAR_PAGE_NAMES:
                st.session_state.selected_page = FIRST_TABULAR
            st.rerun()

    with col2:
        st.subheader("🧠 Deep RL")
        st.caption("Chapters 9–13 · Requires PyTorch + Gymnasium")
        st.markdown("""
- Neural Networks Bridge *(available now)*
- Function Approximation: Prediction *(available now)*
- Function Approximation: Control *(available now)*
- Off-Policy Methods with FA *(available now)*
- Eligibility Traces *(available now)*
- Policy Gradients *(available now)*
- Applications & Case Studies *(available now)*
- Advanced Topics *(coming soon)*
            """)
        if st.button(
            "Enter Deep RL →",
            use_container_width=True,
            key="btn_deep",
        ):
            st.session_state.mode = "deep"
            if st.session_state.selected_page not in _DEEP_PAGE_NAMES:
                st.session_state.selected_page = FIRST_DEEP
            st.rerun()

    st.divider()
    st.caption(
        "Start with Tabular RL if you are new to reinforcement learning. "
        "Deep RL sections build directly on the tabular foundations. "
        "Install optional dependencies with: `pip install torch gymnasium`"
    )


# ── Coming soon ────────────────────────────────────────────────────────────


def show_coming_soon() -> None:
    st.title(st.session_state.selected_page)
    st.info(
        "This section is under construction and will be added in a future update. "
        "Use the sidebar to navigate to an available page, or return Home to choose a mode."
    )


# ── Route ──────────────────────────────────────────────────────────────────

if st.session_state.mode == "home":
    show_landing()
else:
    _module = ALL_PAGES.get(st.session_state.selected_page, "coming_soon")
    if _module == "coming_soon":
        show_coming_soon()
    else:
        try:
            importlib.import_module(_module).show()
        except ModuleNotFoundError:
            show_coming_soon()
