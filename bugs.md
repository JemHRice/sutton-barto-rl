# Bug Reports

> All 16 bugs SQUISHED as of 2026-05-21. Bug 17 logged and fixed same session.

---

## Bug N — Page X: Short title

**Trigger:** What you clicked / selected / set before the bug appeared.  
**Expected:** What should have happened.  
**Actual:** What actually happened (error message if there is one).

---

## 🐛 SQUISHED — Bug 17 — Filesystem: Two conflicting Page 21 files

**Trigger:** Inspecting the `sections/` directory.  
**Expected:** One canonical file for Page 21 (`page21_nn_value.py`) wired into `app.py`.  
**Actual:** A second file `page21_neural_value.py` existed alongside it — a stale draft that imports `utils.feature_bases` rather than PyTorch, and is not referenced by `app.py`.  
**Fix:** Deleted `sections/page21_neural_value.py`.

---

## 🐛 SQUISHED — Bug 1 — Sidebar: Blank page entries for page1 through page9

**Trigger:** App loads normally.  
**Expected:** Only the custom sidebar navigation (Section 1/2/3 radio buttons) is visible.  
**Actual:** Streamlit auto-discovers the `pages/` directory and adds blank entries for all 9 page files in the sidebar above the custom navigation.

---

## 🐛 SQUISHED — Bug 2 — Sidebar: Three separate radio groups instead of one

**Trigger:** App loads normally.  
**Expected:** A single radio group covering all 9 pages across all 3 sections — only one page selected at a time across the whole app.  
**Actual:** Three independent radio buttons (one per section), each with its own selection tick. Multiple sections show a selected page simultaneously.

---

## 🐛 SQUISHED — Bug 3 — All Pages: Page numbering in subtitle

**Trigger:** Navigate to any page.  
**Expected:** Subtitle shows only the section name and page title, e.g. "Section 1 — Bandit Algorithms".  
**Actual:** Subtitle reads "Section X — Title · Page Y of Z" — the page count indicator is unnecessary and should be removed.

---

## 🐛 SQUISHED — Bug 4 — All Pages: Simulation sliders have no context

**Trigger:** Navigate to any page with interactive sliders before running a simulation.  
**Expected:** Each slider (and the simulation block overall) has a brief description explaining what parameter it controls and a learning outcome statement telling the user what to look for in the results.  
**Actual:** Sliders are labelled but have no surrounding explanation of what varying them demonstrates or what the user should learn from the simulation.

---

## 🐛 SQUISHED — Bug 5 — Sidebar: Section labels not interspersed with their pages

**Trigger:** App loads normally.  
**Expected:** The 3 section labels (Section 1 · Bandit Algorithms, etc.) each appear directly above their 3 relevant page options, so the sidebar reads as grouped: label → 3 pages → label → 3 pages → label → 3 pages. Only one page across all 9 can be selected at a time.  
**Actual:** All 3 section labels are stacked at the top of the sidebar, followed by all 9 page options as a flat ungrouped list below them.

---

## 🐛 SQUISHED — Bug 6 — README: Out of date

**Trigger:** Open README.md.  
**Expected:** README reflects the current project structure, including the `sections/` directory (not `pages/`), the Summary page, and the venv setup instructions for Windows.  
**Actual:** README references a `pages/` directory that doesn't exist, omits the Summary page, and gives Unix-only activation instructions.

---

## 🐛 SQUISHED — Bug 7 — App: Page 10 and Page 11 need to swap positions

**Trigger:** Navigate to Section 4 in the sidebar.
**Expected:** TD(0) Prediction appears before Windy GridWorld.
**Actual:** Windy GridWorld is page 10 and TD(0) Prediction is page 11 — they should be swapped.

---

## 🐛 SQUISHED — Bug 8 — Summary Page: Needs updating for new pages (10, 11, 12)

**Trigger:** Navigate to the Summary page.
**Expected:** Summary reflects all current pages including the TD Learning section (pages 10–12).
**Actual:** Summary section is out of date and does not include content for the new pages.

---

## 🐛 SQUISHED — Bug 9 — Page 11 Windy GridWorld: Wind factor numbers hard to read

**Trigger:** View the Windy GridWorld grid with wind strength numbers above columns.
**Expected:** Wind factor numbers are clearly visible, in a lighter grey colour, and wrapped in brackets e.g. (↑2).
**Actual:** Numbers are too hard to see — need lighter grey colour and brackets added.

---

## 🐛 SQUISHED — Bug 10 — Page 10 TD(0) Prediction: MC converges faster than TD(0)

**Trigger:** Run TD(0) vs MC simulation with any settings.  
**Expected:** TD(0) converges faster than MC, matching Sutton & Barto Figure 6.2.  
**Actual:** MC consistently showed lower RMS error and converged faster than TD(0).  
**Root cause:** MC used sample averaging (`returns_sum / returns_count`) while TD(0) used constant-α. Sample-averaging MC is an unbiased estimator whose variance goes to zero, so it always beats constant-α TD eventually — but that's not a fair comparison. S&B Figure 6.2 uses constant-α for both methods.  
**Fix:** Changed MC update to `V(S) += α * (G - V(S))` so both methods use the same update rule, matching the book's comparison.

---

## 🐛 SQUISHED — Bug 11 — Page 11 Windy GridWorld: SARSA greedy path broken with stochastic wind

**Trigger:** Select Stochastic wind, run simulation, view the learned paths grid.  
**Expected:** Both SARSA and Q-Learning greedy paths navigate cleanly from start to goal.  
**Actual:** SARSA path was pushed to row 0 by random wind during the greedy rollout and never reached the goal (showing 800-step path that went off the top of the grid).  
**Root cause:** `_greedy_path` passed `stochastic_wind=True` into the rollout environment, so the visualisation was one random stochastic sample rather than a clean display of the learned policy.  
**Fix:** Changed `_greedy_path` to always use `stochastic_wind=False` for the visualisation. Training still runs with the selected wind type; the path display shows the policy under deterministic (average-case) conditions.

---

## 🐛 SQUISHED — Bug 12 — Page 11 Windy GridWorld: Steps-per-episode chart unreadable

**Trigger:** Run any simulation and view the Steps per Episode chart.  
**Expected:** Convergence trend is clearly visible for both algorithms.  
**Actual:** Raw episode noise dominated the chart and the y-axis was distorted by early 800-step spikes, compressing the interesting part of the curve.  
**Fix:** Reduced raw trace opacity to 8% and removed raw traces from the legend (smoothed averages only in legend); capped y-axis at the 95th percentile of all episode lengths.

---

## 🐛 SQUISHED — Bug 13 — Page 11 Windy GridWorld: Unconverged greedy path renders as dense blob

**Trigger:** Run with Stochastic wind + King's Moves and low episode count (e.g. 500), view the learned paths grid.  
**Expected:** Either a clean path to the goal, or a clear message that the policy hasn't converged.  
**Actual:** The greedy path cycled between a small set of states for all 800 steps, rendering as a dense cluster of overlapping dots that looked like a short dead-end path.  
**Root cause:** No cycle detection in `_greedy_path` — the agent looped indefinitely on states with near-zero Q-values (argmax of all-zeros always picks action 0 = North).  
**Fix:** Added visited-state tracking to `_greedy_path`; breaks on first revisited state. Added a visible warning when either algorithm's path doesn't reach the goal, prompting the user to increase episodes or α.

---

## 🐛 SQUISHED — Bug 14 — Pages 13–15 (n-step methods): OverflowError on first run

**Trigger:** Click Run Simulation on any of Pages 13, 14, or 15 with any settings.  
**Expected:** Simulation runs and results are displayed.  
**Actual:** `OverflowError: cannot convert float infinity to integer` crash on every run.  
**Root cause:** `T` (episode termination time) is initialised to `float('inf')` in the n-step TD algorithm. `int(T)` was called before `min()` could limit the value, so infinity hit `int()` when the episode had not yet terminated. Affected `min(tau + n, int(T))` in all three files, plus two additional occurrences in Page 15's tree-backup loop.  
**Fix:** Moved `int()` outside `min()` in all affected lines — e.g. `int(min(tau + n, T))`. Safe because `min(x, float('inf'))` always returns `x` (a plain int), so `int()` never sees infinity.

---

## 🐛 SQUISHED — Bug 15 — Page 16 Dyna-Q: Agent never moves through the maze

**Trigger:** Click Run Simulation on Page 16 with any settings.  
**Expected:** Agents with more planning steps reach the goal more often, showing the Dyna-Q planning advantage.  
**Actual:** All n values reported 0 goals reached regardless of settings.  
**Root cause:** `state = next_state` was missing from the `run_dynaq` training loop. The agent re-acted from the same state every step and never advanced through the maze.  
**Fix:** Added `else: state = next_state` after the episode-end check. Default steps also increased from 2 000 to 3 000 to match S&B Figure 8.2.

---

## 🐛 SQUISHED — Bug 16 — Pages 16 and 18: Agents stuck in corner due to argmax tie-breaking

**Trigger:** Click Run Simulation on Page 16 or Page 18 with default settings.  
**Expected:** Agents explore the maze and find the goal within the step budget.  
**Actual:** 0 goals reached; agents visually stuck near the start.  
**Root cause:** `np.argmax` on an all-zero Q-table (before any reward is seen) always returns index 0, which maps to action 0 = "up". From the start cell (bottom-left), repeatedly going up drives the agent into the top-left corner where it stays for the entire episode budget.  
**Fix:** Replaced `int(np.argmax(Q[state]))` with random tie-breaking among equally-valued actions: `int(rng.choice(np.flatnonzero(q_s == q_s.max())))`. Applied to all three training functions across Pages 16 and 18.

---
