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
    {
        "label": "Section 4 — Temporal Difference Learning",
        "concepts": [
            {
                "title": "TD(0) Prediction",
                "body": """
Instead of waiting until the end of an episode to calculate the actual return, I update my
estimate of how good each state is after every single step, using the reward I just received
plus my current estimate of how good the next state is. This means I am updating one estimate
using another estimate, which introduces a small amount of bias but dramatically reduces the
amount of noise in each update compared to waiting for the full episode return. Because updates
happen immediately at each step rather than at episode end, this approach also works in
continuing tasks that never terminate.
""",
            },
            {
                "title": "TD(0) vs Monte Carlo",
                "body": """
Monte Carlo methods wait until the episode ends and then use the actual observed return to
update state values, which means each update is an unbiased sample of the true value but is
also noisy because a single episode is just one random outcome. TD(0) updates state values
immediately after each step using the current reward plus the estimated value of the next
state, which introduces some bias from the bootstrapped estimate but produces much lower
variance because each update depends on only one step rather than an entire sequence of
random events. In practice, the lower variance of TD often allows it to converge faster than
Monte Carlo, especially in problems with long episodes.
""",
            },
            {
                "title": "Windy GridWorld — SARSA vs Q-Learning",
                "body": """
In the Windy GridWorld, some columns push the agent upward regardless of the action chosen,
which means the agent must learn to account for this wind when planning a path to the goal.
SARSA updates its action values using the action that was actually taken in the next state,
which means it learns the value of its own behaviour including any random exploratory actions,
making it more cautious in risky parts of the grid during training. Q-Learning always updates
using the best possible action in the next state regardless of what was actually taken, so it
learns the value of the optimal policy even while exploring, which typically finds a more
direct path but may take more risk during training when exploration sends it into the wind.
""",
            },
            {
                "title": "Cliff Walking — Why SARSA and Q-Learning Take Different Paths",
                "body": """
In the cliff walking problem, there is a short path along the edge of a cliff that leads
quickly to the goal but risks a large negative reward if an exploratory action steps off the
edge, and a longer but safer path that stays away from the cliff. Q-Learning learns the
value of always taking the optimal action, so it converges to the short path along the cliff
edge because in theory that is the best route. SARSA learns the value of its actual behaviour
including exploration, so it recognises that occasionally taking a random action near the
cliff is dangerous and prefers the longer safe path instead. This difference between learning
the optimal policy versus the current behaviour policy means SARSA and Q-Learning can
converge to genuinely different solutions in the same environment.
""",
            },
        ],
    },
    {
        "label": "Section 5 — n-step Bootstrapping",
        "concepts": [
            {
                "title": "n-step TD Prediction",
                "body": """
Instead of waiting for the full episode return like Monte Carlo or bootstrapping from just one
step ahead like TD(0), n-step TD collects real rewards for n steps into the future and then
bootstraps from the estimated value of the state it ends up in. This creates a spectrum between
TD(0) and Monte Carlo controlled by the single parameter n. Small n gives low variance but high
bias because early value estimates are rough; large n gives low bias but high variance because
more random steps are included in the return. The best n depends on the task, but intermediate
values of roughly two to eight often learn fastest by balancing the two error sources.
""",
            },
            {
                "title": "n-step SARSA",
                "body": """
n-step SARSA extends the n-step bootstrapping idea from prediction to control by replacing
state values with action values and applying ε-greedy action selection. After taking n real
steps the algorithm computes the n-step return using the real rewards collected and the current
Q value of the state-action pair it landed on, then updates the Q value of the original
state-action pair from which the n steps began. This allows reward information to propagate
n steps backward in a single update, which is particularly helpful in tasks with delayed
rewards where pure one-step methods require many episodes to propagate credit back to the
decisions that actually caused a good outcome.
""",
            },
            {
                "title": "Tree Backup vs Importance Sampling for Off-Policy n-step",
                "body": """
When the policy being evaluated differs from the policy used to collect data, n-step updates
need a correction. Importance sampling multiplies the return by the ratio of target policy
probability to behaviour policy probability for each step in the n-step window. When the two
policies differ substantially, this ratio is a product of many terms that can become very large
or very small, causing high variance that destabilises learning especially for large n.
Tree backup avoids importance sampling entirely by replacing the continuation of the trajectory
with the expected value under the target policy at each intermediate step, effectively backing
up a weighted tree of possible continuations rather than a single observed path. This produces
an off-policy update without any ratio products, at the cost of slightly more computation per
update.
""",
            },
        ],
    },
    {
        "label": "Section 6 — Planning and Learning",
        "concepts": [
            {
                "title": "Dyna-Q and Model-Based RL",
                "body": """
Dyna-Q combines direct model-free reinforcement learning with planning using a learned model
of the environment. After each real interaction the agent updates Q values directly from the
observed transition, updates its internal model to record that the observed state and action
led to the observed reward and next state, then runs n additional planning steps by sampling
previously observed state-action pairs from the model and performing Q-learning updates on
those simulated transitions. Because simulated experience is cheap to generate once the model
is built, each real environment step effectively produces n plus one Q-learning updates.
With enough planning steps an agent can learn a good policy after many fewer real interactions
than a purely model-free algorithm, which matters greatly when real experience is expensive.
""",
            },
            {
                "title": "Dyna-Q+ and Adapting to Change",
                "body": """
Plain Dyna-Q assumes the environment is stationary and its model remains accurate over time.
When the environment changes, the stored model becomes incorrect and Dyna-Q may continue
planning with stale transitions, failing to discover new paths or adapting slowly to blocked
paths. Dyna-Q+ addresses this by adding an exploration bonus to the reward of any state-action
pair that has not been visited recently, scaled by the square root of the time since it was
last tried. This bonus encourages the agent to revisit parts of the environment it has not
checked in a while, allowing it to detect environmental changes much faster than random
exploration would manage on its own.
""",
            },
            {
                "title": "Prioritised Sweeping",
                "body": """
Dyna-Q selects which state-action pairs to update during planning uniformly at random, which
wastes computation on pairs whose Q values are already accurate. Prioritised sweeping instead
maintains a priority queue ordered by the magnitude of the TD error for each state-action pair.
After each real or simulated update, any state-action pairs whose predecessors just had their
Q values changed are recomputed and added to the queue if their TD errors exceed a threshold.
The pair with the highest priority is always updated next, so computation focuses on the parts
of the state space where value estimates are most wrong. When reward is first discovered,
prioritized sweeping immediately propagates this information backward through all predecessor
states in order of importance, reaching the goal state in far fewer real environment steps
than random Dyna-Q planning.
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
