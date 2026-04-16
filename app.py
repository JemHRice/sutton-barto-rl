"""
RL Foundations: Bandits, Dynamic Programming & Monte Carlo Methods
A beginner-friendly Streamlit educational app.
"""

import importlib
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

import streamlit as st

st.set_page_config(
    page_title="RL Foundations",
    page_icon="🎰",
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

PAGES = {
    "1 — ε-Greedy Bandit": "sections.page1_epsilon_greedy",
    "2 — UCB": "sections.page2_ucb",
    "3 — Thompson Sampling": "sections.page3_thompson",
    "4 — Policy Evaluation": "sections.page4_policy_eval",
    "5 — Policy Iteration": "sections.page5_policy_iteration",
    "6 — Value Iteration": "sections.page6_value_iteration",
    "7 — MC Prediction": "sections.page7_mc_prediction",
    "8 — MC Control": "sections.page8_mc_control",
    "9 — Solving Blackjack": "sections.page9_blackjack",
    "📋 Summary — Key Concepts": "sections.page_summary",
}

PAGE_KEYS = list(PAGES.keys())

SEC1   = PAGE_KEYS[0:3]
SEC2   = PAGE_KEYS[3:6]
SEC3   = PAGE_KEYS[6:9]
SECSUM = [PAGE_KEYS[9]]

# ── Session state ──────────────────────────────────────────────────────────

if "selected_page" not in st.session_state:
    st.session_state.selected_page = SEC1[0]
    st.session_state.r1   = SEC1[0]
    st.session_state.r2   = None
    st.session_state.r3   = None
    st.session_state.rsum = None


def _pick(key: str, others: list):
    """On-change callback: update selected_page and clear the other radios."""
    val = st.session_state[key]
    if val is not None:
        st.session_state.selected_page = val
        for k in others:
            st.session_state[k] = None


# ── Sidebar ────────────────────────────────────────────────────────────────

def _section_label(text: str):
    st.sidebar.markdown(
        f'<div class="sidebar-section">{text}</div>', unsafe_allow_html=True
    )


with st.sidebar:
    st.title("RL Foundations")
    st.caption("Bandits · DP · Monte Carlo")
    st.divider()

    _section_label("Section 1 · Bandit Algorithms")
    st.radio("s1", SEC1, label_visibility="collapsed", key="r1",
             on_change=_pick, args=("r1", ["r2", "r3", "rsum"]))

    _section_label("Section 2 · Dynamic Programming")
    st.radio("s2", SEC2, label_visibility="collapsed", key="r2",
             on_change=_pick, args=("r2", ["r1", "r3", "rsum"]))

    _section_label("Section 3 · Monte Carlo Methods")
    st.radio("s3", SEC3, label_visibility="collapsed", key="r3",
             on_change=_pick, args=("r3", ["r1", "r2", "rsum"]))

    st.divider()
    st.radio("s4", SECSUM, label_visibility="collapsed", key="rsum",
             on_change=_pick, args=("rsum", ["r1", "r2", "r3"]))

    st.divider()
    st.markdown(
        "<small>📘 Based on <em>Reinforcement Learning: An Introduction</em>"
        " — Sutton &amp; Barto (2nd ed.)<br><br>"
        "All algorithms implemented from scratch with NumPy.</small>",
        unsafe_allow_html=True,
    )

# ── Route ──────────────────────────────────────────────────────────────────

importlib.import_module(PAGES[st.session_state.selected_page]).show()
