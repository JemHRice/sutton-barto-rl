# RL Foundations: An Interactive Sutton & Barto Companion

## [🚀 Try the Live App](https://sutton-barto-rl-eajnsbsvvdoygeyktohrju.streamlit.app/)

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://sutton-barto-rl-eajnsbsvvdoygeyktohrju.streamlit.app/)
![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![NumPy](https://img.shields.io/badge/tabular%20methods-NumPy%20only-orange)
![License](https://img.shields.io/badge/license-MIT-green)

An interactive, beginner-friendly Streamlit app for learning reinforcement learning from the ground up — built alongside a study of Sutton & Barto's *Reinforcement Learning: An Introduction* (2nd ed.).

The app has two modes: **Tabular RL** (Chapters 1–8, pure NumPy, no GPU) and **Deep RL** (Chapters 9–13, PyTorch + Gymnasium). Every algorithm is implemented from scratch. Every concept has a hands-on simulation.

---

## Who Is This For?

Anyone learning RL who knows Python and basic probability. No prior RL knowledge required — each section builds directly on what came before. Each page follows a consistent structure:

> **Concept explanation → Interactive controls → Run simulation → Visualisation → Key takeaways**

---

## Installation

```bash
git clone https://github.com/JemHRice/sutton-barto-rl.git
cd sutton-barto-rl

python -m venv .venv

# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate

pip install -r requirements.txt

# Optional: for Deep RL sections (Chapters 9–13)
pip install torch gymnasium

streamlit run app.py
```

The app opens at `http://localhost:8501`.

---

## What's Inside

### Tabular RL Mode — Chapters 1–8

#### Section 1 — Bandit Algorithms (Chapter 2)

| Page | Algorithm | Key Idea |
|------|-----------|----------|
| 1 | ε-Greedy | Explore randomly with probability ε; exploit best known arm otherwise |
| 2 | UCB | Explore arms that are uncertain, not just randomly |
| 3 | Thompson Sampling | Maintain a full Bayesian posterior; sample to decide |

#### Section 2 — Dynamic Programming (Chapters 3–4)

| Page | Topic | Key Idea |
|------|-------|----------|
| 4 | Policy Evaluation | Compute how good a fixed policy is via iterative Bellman updates |
| 5 | Policy Iteration | Alternate evaluate/improve until optimal; step through interactively |
| 6 | Value Iteration | Combine evaluation and improvement into a single sweep |

#### Section 3 — Monte Carlo Methods (Chapter 5)

| Page | Topic | Key Idea |
|------|-------|----------|
| 7 | MC Prediction | Average observed returns to estimate value; no model needed |
| 8 | MC Control | Learn optimal Q-function with ε-greedy on-policy MC |
| 9 | Solving Blackjack | 500k+ episodes to converge; play interactively against the trained agent |

#### Section 4 — Temporal Difference Learning (Chapter 6)

| Page | Topic | Key Idea |
|------|-------|----------|
| 10 | TD(0) Prediction | Bootstrap after every step; compare with MC on Random Walk |
| 11 | Windy GridWorld | SARSA vs Q-Learning with wind; toggle stochastic wind and King's moves |
| 12 | Cliff Walking | SARSA finds the safe path; Q-Learning finds the risky optimal one |

#### Section 5 — n-step Bootstrapping (Chapter 7)

| Page | Topic | Key Idea |
|------|-------|----------|
| 13 | n-step TD Prediction | Slider from n=1 (TD) to n=∞ (MC); bias-variance sweet spot on Random Walk |
| 14 | n-step SARSA | n-step control on GridWorld; learned policy heatmap |
| 15 | Tree Backup vs IS | Off-policy without IS variance explosion |

#### Section 6 — Planning and Learning (Chapter 8)

| Page | Topic | Key Idea |
|------|-------|----------|
| 16 | Dyna-Q | Learn a model, plan with it; n planning steps = n free simulated updates |
| 17 | Dyna-Q+ (Changing Maze) | Exploration bonus adapts to blocking/shortcut maze changes |
| 18 | Prioritised Sweeping | Priority queue targets highest-error states first |

---

### Neural Networks Bridge — Transition Section

| Page | Topic | Key Idea |
|------|-------|----------|
| Bridge A | Value Function Approximation | Why tabular breaks down; MSVE objective; interactive linear approximator |
| Bridge B | Semi-Gradient Methods | Gradient MC vs semi-gradient TD(0) with linear features on Random Walk |

---

### Deep RL Mode — Chapters 9–13

> Deep RL sections require `pip install torch gymnasium`

#### Section 7 — Function Approximation: Prediction (Chapter 9) — *coming soon*

| Page | Topic |
|------|-------|
| 19 | Gradient MC vs Semi-gradient TD with feature basis selector |
| 20 | Feature Basis Explorer: tile coding, RBF, Fourier basis |
| 21 | Neural Network Value Approximator with PyTorch |

#### Section 8 — Function Approximation: Control (Chapter 10) — *coming soon*

| Page | Topic |
|------|-------|
| 22 | Semi-gradient SARSA on Mountain Car |
| 23 | Mountain Car Solver: cost-to-go surface plot |
| 24 | Average Reward vs Discounted formulations |

#### Section 9 — Off-Policy with Approximation (Chapter 11) — *coming soon*

| Page | Topic |
|------|-------|
| 25 | Importance Sampling with Function Approximation |
| 26 | Deadly Triad Demonstration |
| 27 | Gradient TD Methods (TDC, GTD2) |

#### Section 10 — Eligibility Traces (Chapter 12) — *coming soon*

| Page | Topic |
|------|-------|
| 28 | λ-return and TD(λ) |
| 29 | SARSA(λ) on Cliff Walking |
| 30 | Unifying n-step and Traces |

#### Section 11 — Policy Gradient Methods (Chapter 13) — *coming soon*

| Page | Topic |
|------|-------|
| 31 | REINFORCE on CartPole |
| 32 | REINFORCE with Baseline |
| 33 | Actor-Critic |
| 34 | PPO/TRPO Conceptual |

#### Section 12 — Applications & Case Studies (Chapter 16) — *coming soon*

| Page | Topic |
|------|-------|
| 35 | TD-Gammon & Game Playing |
| 36 | Real-World RL Case Studies |
| 37 | AlphaGo Architecture |

#### Section 13 — Advanced Topics & Frontiers (Chapter 17) — *coming soon*

| Page | Topic |
|------|-------|
| 38 | Hierarchical RL & Options |
| 39 | Meta-RL & Exploration Frontiers |
| 40 | Multi-Agent & Future Directions |

---

## Dependencies

| Package | Purpose |
|---------|---------|
| `streamlit` | App framework |
| `numpy` | All tabular algorithm implementations |
| `plotly` | Interactive charts |
| `pandas` | Table display |
| `torch` | Deep RL sections (optional) |
| `gymnasium` | Continuous environments — Mountain Car, CartPole (optional) |

---

## Architecture

```
app.py                        # Landing page, two-mode routing, collapsible sidebar
requirements.txt              # NumPy-only; PyTorch and Gymnasium are optional
sections/
  page1_epsilon_greedy.py     # One file per page — tabular sections 1–4
  ...
  page13_nstep_td.py          # Section 5 — n-step Bootstrapping
  page14_nstep_sarsa.py
  page15_tree_backup.py
  page16_dynaq.py             # Section 6 — Planning and Learning
  page17_dynaq_plus.py
  page18_prioritized.py
  bridge_a_value_approx.py    # Neural Networks Bridge
  bridge_b_semi_gradient.py
  page_summary.py             # Accumulated concept notes
utils/
  gridworld.py                # Shared GridWorld + cached DP solvers
  blackjack.py                # Blackjack environment
  random_walk.py              # Random Walk environment
```

**Caching pattern:** every expensive simulation function is decorated with `@st.cache_data`
at module level and called only inside `st.button` blocks — nothing runs at import or page load.

**PyTorch guard pattern** (Deep RL pages):
```python
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
```

---

## Reference

Sutton, R. S., & Barto, A. G. (2018). *Reinforcement Learning: An Introduction* (2nd ed.). MIT Press. [Available free online](http://incompleteideas.net/book/the-book-2nd.html).
