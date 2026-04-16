import streamlit as st


# ── Content ────────────────────────────────────────────────────────────────
#
# Each concept is a dict with keys:
#   title    : str   — expander label
#   body     : str   — markdown content
#
# Add new entries here as you work through the book.
# Sections automatically render in order.

SECTIONS = [
    {
        "label": "Section 1 — Bandit Algorithms",
        "concepts": [
            {
                "title": "ε-Greedy",
                "body": """
For each possible action, I keep track of the expected reward based on what has happened before,
and most of the time I choose the action that currently has the highest expected reward. A small
percentage of the time, controlled by ε, I deliberately choose a random action instead of the
best one so that I continue gathering information about actions I have not tried enough. Over
many steps, this allows the expected rewards of each action to become more accurate, while still
ensuring that all actions are explored enough to discover which one is actually best.
""",
            },
            {
                "title": "UCB (Upper Confidence Bound)",
                "body": """
For each possible action, I calculate not only the expected reward based on past outcomes, but
also add a bonus that depends on how few times that action has been tried so far. Actions that
have been tried many times get a small bonus, while actions that have been tried rarely get a
larger bonus, reflecting greater uncertainty about how good they really are. I then choose the
action with the highest combined value of expected reward plus uncertainty bonus, which causes
the system to naturally explore uncertain actions while still favouring actions that appear to
perform well.
""",
            },
            {
                "title": "UCB vs ε-Greedy",
                "body": """
With ε-greedy, I explore by randomly choosing actions some percentage of the time, regardless
of whether those actions are uncertain or already well understood. With UCB, I explore by
deliberately choosing actions that have been tried less often, meaning the exploration is
directed toward actions where my estimates are still uncertain rather than purely random. Over
time, this usually leads to faster learning because effort is focused on resolving uncertainty
rather than randomly testing actions that are already well known.
""",
            },
            {
                "title": "Thompson Sampling",
                "body": """
For each possible action, I maintain a belief about how good that action might be, represented
as a range of possible values rather than a single fixed estimate. At each step, I randomly
sample one possible value from each action's belief range, and then choose the action whose
sampled value is highest at that moment. Over time, actions that appear more promising are
sampled more often, but uncertain actions are still occasionally chosen because their belief
ranges allow for the possibility that they might be better than expected.
""",
            },
            {
                "title": "Thompson Sampling vs ε-Greedy and UCB",
                "body": """
With ε-greedy, I occasionally explore by choosing actions completely at random, regardless of
what I currently believe about their quality. With UCB, I explore by adding an uncertainty
bonus to actions that have been tried less frequently, ensuring that poorly tested actions are
revisited until their values are better understood. With Thompson Sampling, I explore by
randomly sampling from my beliefs about each action's quality, meaning actions that might
reasonably be good still get chosen even if they are not currently the most certain option.
""",
            },
        ],
    },
    {
        "label": "Section 2 — Dynamic Programming",
        "concepts": [
            {
                "title": "Policy Evaluation",
                "body": """
Given a fixed policy that tells me how likely I am to take each action in every state, I
repeatedly calculate the expected reward plus discounted value of the next state for all
possible next states and rewards that could result from following that policy. I then update the
value of the current state to be the weighted average of those outcomes, where the weights
depend on both the transition probabilities and the policy's action choices. By repeating this
process over and over, the state values gradually stabilise, giving me an accurate measure of
how good the policy actually is without changing the policy itself.
""",
            },
            {
                "title": "Policy Iteration",
                "body": """
Starting with some initial policy, I first evaluate how good that policy is by repeatedly
updating state values until they reflect the expected rewards that occur when following that
policy. Once the values are known, I improve the policy by selecting, in each state, the action
that produces the highest expected reward plus discounted next-state value. I then repeat this
cycle of evaluating the policy and improving it, continuing until the policy stops changing,
which means that the optimal policy has been found.
""",
            },
            {
                "title": "Value Iteration",
                "body": """
Instead of fully evaluating a policy before improving it, I directly update each state by
calculating the expected reward plus discounted value of the next state for all possible actions
and immediately selecting the action that produces the highest result. This means that each
update step both evaluates possible outcomes and pushes the values toward the best possible
decisions at the same time. By repeatedly updating state values in this way, the values move
directly toward the optimal values without needing separate policy evaluation steps.
""",
            },
            {
                "title": "Why Value Iteration Is Better",
                "body": """
Policy iteration requires fully evaluating a policy before making improvements, which can take
many repeated updates before the values become accurate enough to guide changes. Value iteration
skips this long evaluation phase by always choosing the best possible action during each value
update, meaning policy improvement happens automatically as the values are updated. Because it
combines evaluation and improvement into a single repeated step, value iteration usually reaches
the optimal values and policy faster than policy iteration.
""",
            },
        ],
    },
    {
        "label": "Section 3 — Monte Carlo Methods",
        "concepts": [
            {
                "title": "Monte Carlo Prediction (and how it fundamentally differs from DP)",
                "body": """
Given a fixed policy, I run full episodes using that policy and record what actually happens
from each state until the episode ends, collecting the total reward that occurs after each state
appears. For each state, I calculate the return that followed that state in the episode, and
over many episodes I take the average of those observed returns to estimate how good that state
really is under the policy. Unlike dynamic programming, I do not calculate expected rewards
using known transition probabilities or bootstrapped next-state values, and instead rely
entirely on rewards that actually occurred during real sampled episodes, which allows learning
even when the model of the environment is unknown.
""",
            },
            {
                "title": "Monte Carlo Control",
                "body": """
Starting with an initial policy, I run full episodes using that policy while occasionally
trying different actions so that all possible actions get tested enough times. For each
state-action pair that appears in an episode, I calculate the return that followed that action
and update the expected value of that action by averaging the returns observed across many
episodes. After updating these action values, I adjust the policy to favour the actions with
the highest expected returns, and by repeating this cycle of running episodes, updating values,
and improving the policy, the policy gradually shifts toward the actions that produce the best
long-term rewards.
""",
            },
            {
                "title": "Solving Blackjack with Monte Carlo",
                "body": """
For each round of blackjack, I treat the entire hand as one episode and follow the current
policy to decide whether to hit or stick at each state, continuing until the hand ends with a
win, loss, or draw. Whenever a specific state-action pair appears, such as choosing to hit on
a particular player total against a dealer card, I record the total reward that eventually
occurs at the end of the hand and associate that return with the decision that was made
earlier. Over many episodes, I average the returns observed after each state-action pair and
update the policy to favour actions that historically produced better outcomes, and by
repeatedly playing hands, updating expected values, and improving decisions, the strategy
gradually converges toward an optimal way of playing blackjack without ever needing to know
the exact probabilities of the game.
""",
            },
        ],
    },
]


# ── Page ───────────────────────────────────────────────────────────────────

def show():
    st.title("Summary — Key Concepts")
    st.markdown("**Accumulated notes from working through Sutton & Barto**")
    st.markdown(
        "Each entry is a plain-English summary of one concept or comparison. "
        "Expand any card to read it. This page grows as new sections are added to the app."
    )

    for section in SECTIONS:
        st.divider()
        st.subheader(section["label"])

        if not section["concepts"]:
            st.caption("No entries yet — check back after completing this section.")
            continue

        for concept in section["concepts"]:
            with st.expander(concept["title"]):
                st.markdown(concept["body"].strip())
