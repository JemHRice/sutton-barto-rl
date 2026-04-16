# Bug Reports

> All 6 bugs resolved. Bugs 1 & 2 were already fixed (pages moved to `sections/`, unified radio). Bugs 3 & 4 fixed 2026-04-16. Bugs 5 & 6 fixed 2026-04-16.



Use the format below. One section per bug. Don't worry about root cause or fixes — just describe what you saw.

---

## Bug N — Page X: Short title

**Trigger:** What you clicked / selected / set before the bug appeared.  
**Expected:** What should have happened.  
**Actual:** What actually happened (error message if there is one).

---

## Bug 1 — Sidebar: Blank page entries for page1 through page9

**Trigger:** App loads normally.  
**Expected:** Only the custom sidebar navigation (Section 1/2/3 radio buttons) is visible.  
**Actual:** Streamlit auto-discovers the `pages/` directory and adds blank entries for all 9 page files in the sidebar above the custom navigation.

---

## Bug 2 — Sidebar: Three separate radio groups instead of one

**Trigger:** App loads normally.  
**Expected:** A single radio group covering all 9 pages across all 3 sections — only one page selected at a time across the whole app.  
**Actual:** Three independent radio buttons (one per section), each with its own selection tick. Multiple sections show a selected page simultaneously.

---

## Bug 3 — All Pages: Page numbering in subtitle

**Trigger:** Navigate to any page.  
**Expected:** Subtitle shows only the section name and page title, e.g. "Section 1 — Bandit Algorithms".  
**Actual:** Subtitle reads "Section X — Title · Page Y of Z" — the page count indicator is unnecessary and should be removed.

---

## Bug 4 — All Pages: Simulation sliders have no context

**Trigger:** Navigate to any page with interactive sliders before running a simulation.  
**Expected:** Each slider (and the simulation block overall) has a brief description explaining what parameter it controls and a learning outcome statement telling the user what to look for in the results.  
**Actual:** Sliders are labelled but have no surrounding explanation of what varying them demonstrates or what the user should learn from the simulation.

---

## Bug 6 — README: Out of date

**Trigger:** Open README.md.  
**Expected:** README reflects the current project structure, including the `sections/` directory (not `pages/`), the Summary page, and the venv setup instructions for Windows.  
**Actual:** README references a `pages/` directory that doesn't exist, omits the Summary page, and gives Unix-only activation instructions.

---

## Bug 5 — Sidebar: Section labels not interspersed with their pages

**Trigger:** App loads normally.  
**Expected:** The 3 section labels (Section 1 · Bandit Algorithms, etc.) each appear directly above their 3 relevant page options, so the sidebar reads as grouped: label → 3 pages → label → 3 pages → label → 3 pages. Only one page across all 9 can be selected at a time.  
**Actual:** All 3 section labels are stacked at the top of the sidebar, followed by all 9 page options as a flat ungrouped list below them.
