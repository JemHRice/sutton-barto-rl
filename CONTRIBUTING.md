# Contributing

This is a personal learning project and evolving alongside a study of Sutton & Barto.
Contributions are welcome — particularly corrections, improved explanations, or new
interactive visualisations for existing pages.

## Getting Started

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

## Project Structure

```
app.py                  # routing, sidebar, landing page
sections/
  page1_*.py            # one file per page — tabular sections
  page13_*.py           # Chapter 7 (n-step)
  page16_*.py           # Chapter 8 (planning)
  bridge_a_*.py         # Neural Networks Bridge
  bridge_b_*.py
  page_summary.py       # accumulated concept notes
utils/
  blackjack.py
  gridworld.py
  random_walk.py
requirements.txt        # NumPy, Streamlit, Plotly, Pandas (no PyTorch)
```

## Page Conventions

Every page file must follow this pattern:

1. **Module-level cached functions** decorated with `@st.cache_data` for all expensive computations
2. **A single `show()` function** — called by `app.py` when the page is selected
3. **No computation at import time** — everything runs inside `show()` or inside a cached function called from `show()`
4. **All simulations gated behind `st.button`** — nothing runs on page load
5. **All charts use Plotly** — no Matplotlib
6. **Equations use `st.latex`**
7. **Callouts**: `st.info` for insights, `st.success` for positive outcomes, `st.warning` for caveats
8. **Deep dives** in `st.expander`
9. **Page structure**: Concept explanation → Interactive controls → Run simulation → Visualisation → Key takeaways

## Deep RL Pages (Chapters 9–13)

Pages using PyTorch must include a guarded import at the top:

```python
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
```

And inside `show()`, return early if PyTorch is missing:

```python
def show():
    if not TORCH_AVAILABLE:
        st.error(
            "PyTorch is required for this page. "
            "Install it with: `pip install torch`"
        )
        return
    # rest of page
```

Pages using Gymnasium follow the same pattern with `gymnasium` in place of `torch`.

## Adding a New Page

1. Create `sections/pageN_description.py` following the conventions above
2. Add the page to the appropriate section in `app.py` under `TABULAR_SECTIONS` or `DEEP_SECTIONS`
3. Add concept notes to the relevant section in `sections/page_summary.py`
4. Update `CHANGELOG.md`

## Style

- Beginner-friendly tone — assume Python and basic probability, no prior RL knowledge beyond
  what earlier sections have covered
- Algorithms from scratch with NumPy (tabular sections) — no RL libraries
- No comments explaining what the code does — only comments explaining non-obvious why
- No docstrings beyond a one-line description on cached functions if needed
