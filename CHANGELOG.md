# Changelog

## [Unreleased]

### Fixed
- **Page 10 TD(0) Prediction** — MC comparison was unfair: MC used sample averaging while TD(0) used constant-α, making MC always appear to converge faster. Both methods now use constant-α updates, matching S&B Figure 6.2.
- **Page 11 Windy GridWorld** — Greedy path visualisation used stochastic wind during rollout, causing SARSA to be blown off-course and never reach the goal. Path display now always uses deterministic wind.
- **Page 11 Windy GridWorld** — Steps-per-episode chart was unreadable due to noisy raw traces and early 800-step spikes distorting the y-axis. Raw traces reduced to 8% opacity and excluded from legend; y-axis capped at 95th percentile.
- **Page 11 Windy GridWorld** — Unconverged greedy paths cycled for all 800 steps, rendering as a dense blob. Added cycle detection and a convergence warning when the goal is not reached.
- **Pages 13–15 (n-step methods)** — OverflowError crash: `int(T)` was called before `min()` could limit the value, so `float('inf')` hit `int()` when the episode had not yet terminated. Fixed by moving `int()` outside `min()` in all three files.
- **Page 16 Dyna-Q** — Agent never moved through the maze: `state = next_state` was missing from the training loop, so every step re-acted from the same state. Fixed and default steps increased to 3 000.
- **Pages 16 and 18** — `np.argmax` tie-breaking on all-zero Q-values always picked action 0 (up), driving the agent into a corner at episode start. Replaced with random tie-breaking among equally-valued actions. Default steps on Page 18 also increased to 3 000.
- **Page 18 Priority Heatmap** — Colorscale auto-ranged into negatives when priorities were near zero. Fixed by pinning `zmin=0`.

### Added
- Two-mode app architecture: Tabular RL (Chapters 1–8) and Deep RL (Chapters 9–13)
- Landing page routing users to their chosen mode
- Collapsible section groups in the sidebar for both modes
- Purple colour accent for Deep RL mode
- Section 5 — n-step Bootstrapping (Chapter 7)
  - Page 13: n-step TD Prediction on 19-state Random Walk
  - Page 14: n-step SARSA Control on GridWorld
  - Page 15: Tree Backup vs Importance Sampling
- Section 6 — Planning and Learning (Chapter 8)
  - Page 16: Dyna-Q Maze Solver
  - Page 17: Dyna-Q vs Dyna-Q+ on Changing Mazes
  - Page 18: Prioritised Sweeping
- Neural Networks Bridge (transition section)
  - Bridge A: Value Function Approximation (conceptual + interactive linear approximator)
  - Bridge B: Semi-Gradient Methods (gradient MC vs semi-gradient TD on Random Walk)
- "Coming soon" placeholder pages for Sections 7–13 (Chapters 9–13)
- Summary page updated with Sections 5 and 6 concept notes
- CHANGELOG.md and CONTRIBUTING.md

## [0.1.0] — Initial Release

### Added
- Section 1 — Bandit Algorithms (Chapter 2)
  - Page 1: ε-Greedy Bandit
  - Page 2: UCB Action Selection
  - Page 3: Thompson Sampling
- Section 2 — Dynamic Programming (Chapters 3–4)
  - Page 4: Policy Evaluation
  - Page 5: Policy Iteration
  - Page 6: Value Iteration
- Section 3 — Monte Carlo Methods (Chapter 5)
  - Page 7: MC Prediction
  - Page 8: MC Control
  - Page 9: Solving Blackjack
- Section 4 — Temporal Difference Learning (Chapter 6)
  - Page 10: TD(0) Prediction
  - Page 11: Windy GridWorld
  - Page 12: Cliff Walking
- Summary page with accumulated concept notes
