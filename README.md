# RL Foundations: Bandits, Dynamic Programming & Monte Carlo

An interactive, beginner-friendly Streamlit app for learning the core ideas in reinforcement learning — from multi-armed bandits through dynamic programming to Monte Carlo methods — through live simulations and plain-English explanations.

Built on Sutton & Barto's *Reinforcement Learning: An Introduction* (2nd ed.). All algorithms are implemented from scratch using NumPy only — no RL libraries.

---

## Who Is This For?

Anyone learning RL who knows Python and basic probability but has never implemented RL algorithms before. Each page follows a consistent structure:

> **Concept explanation → Interactive simulation → Results visualisation → Key takeaways**

---

## Installation & Running

```bash
# 1. Clone or download this repo
git clone <your-repo-url>
cd sutton-barto-rl

# 2. Create a virtual environment (recommended)
python -m venv .venv

# Windows (PowerShell)
.venv\Scripts\activate

# Mac / Linux
source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Launch the app
python -m streamlit run app.py
```

The app will open at `http://localhost:8501`.

---

## What's Inside

### Section 1 — Bandit Algorithms

Bandits are the simplest RL setting: one state, many actions, unknown reward distributions. They isolate the **exploration-exploitation tradeoff** in its purest form.

| Page | Algorithm | Key Idea |
|------|-----------|----------|
| 1 | **ε-Greedy** | Explore randomly with probability ε; exploit best known arm otherwise |
| 2 | **UCB (Upper Confidence Bound)** | Explore arms that are uncertain, not just randomly |
| 3 | **Thompson Sampling** | Maintain a full Bayesian posterior over each arm's reward probability |

All three run on the same 10-armed testbed — pages 2 and 3 compare algorithms head-to-head.

### Section 2 — Dynamic Programming

DP methods solve RL problems exactly when you have a **complete model** of the environment (transition probabilities and rewards). All pages use a 4×4 GridWorld (Sutton & Barto Example 4.1).

| Page | Topic | Key Idea |
|------|-------|----------|
| 4 | **Policy Evaluation** | Compute how good a fixed policy is using iterative Bellman updates |
| 5 | **Policy Iteration** | Alternate between evaluating and improving the policy until optimal; step through each iteration interactively |
| 6 | **Value Iteration** | Combine evaluation and improvement into a single Bellman-max sweep; compare convergence speed against Policy Iteration |

### Section 3 — Monte Carlo Methods

MC methods learn from experience — no model required. They run complete episodes and use the actual returns to estimate values and improve policies.

| Page | Topic | Key Idea |
|------|-------|----------|
| 7 | **MC Prediction** | Estimate a value function by averaging observed returns; demonstrated on a 5-state Random Walk with true-value comparison |
| 8 | **MC Control** | Learn the optimal Q-function with ε-greedy on-policy MC; environment is Blackjack, implemented from scratch |
| 9 | **Solving Blackjack** | Run 500k–1M episodes to converge on the optimal Blackjack policy; visualise learned strategy heatmaps and play interactively against the trained agent |

### Summary — Key Concepts

A growing reference page of plain-English summaries for every algorithm and concept covered in the app. Updated as new sections are added.

---

## Dependencies

| Package | Purpose |
|---------|---------|
| `streamlit` | App framework and UI components |
| `numpy` | All algorithm implementations |
| `plotly` | Interactive charts |
| `pandas` | Table display |

No RL libraries (gym, stable-baselines, etc.) are used.

---

## Project Structure

```
sutton-barto-rl/
├── app.py                          # Main entry point, sidebar navigation
├── requirements.txt
├── README.md
├── bugs.md                         # Known bugs and feature requests
├── utils/
│   ├── gridworld.py                # Shared 4×4 GridWorld + cached DP solvers
│   └── blackjack.py                # Blackjack environment (reset/step API)
└── sections/
    ├── page1_epsilon_greedy.py     # ε-Greedy bandit
    ├── page2_ucb.py                # UCB1 bandit
    ├── page3_thompson.py           # Thompson Sampling (Beta-Bernoulli)
    ├── page4_policy_eval.py        # Iterative policy evaluation
    ├── page5_policy_iteration.py   # Policy iteration with step-through UI
    ├── page6_value_iteration.py    # Value iteration vs policy iteration
    ├── page7_mc_prediction.py      # First-visit MC prediction (Random Walk)
    ├── page8_mc_control.py         # On-policy MC control (Blackjack)
    ├── page9_blackjack.py          # Solving Blackjack + interactive demo
    └── page_summary.py             # Summary — Key Concepts (accumulates over time)
```
