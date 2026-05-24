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
    {
        "label": "Section 7 — Function Approximation: Prediction",
        "concepts": [
            {
                "title": "Why We Need Function Approximation",
                "body": """
Tabular methods store one value per state, which works when the state space is small but becomes
impossible when states are continuous or enormous. Function approximation replaces the table with a
parameterised function that generalises across states, so updating the value of one state
automatically improves estimates of nearby states. The tradeoff is that the function can only
represent values in the span of its chosen basis, so some approximation error is irreducible
regardless of how long training continues.
""",
            },
            {
                "title": "Gradient Monte Carlo",
                "body": """
Gradient Monte Carlo runs full episodes to collect unbiased return estimates and then uses
gradient descent to update the function approximator's weights toward each observed return.
Because the target is the actual return and not a bootstrapped estimate, the gradient is exact
and the method converges to the true minimum of the mean squared value error for linear
approximators. The cost is that the agent must wait until episode end before any update can
happen, which makes learning slower per unit of real time and impossible for continuing tasks.
""",
            },
            {
                "title": "Semi-gradient TD(0)",
                "body": """
Semi-gradient TD(0) updates the approximator's weights after every step using a bootstrapped
target of the current reward plus the discounted estimated value of the next state. Since this
target itself depends on the current weights, taking the full gradient of the loss would require
differentiating through the target — instead, the target is treated as a constant and only the
gradient of the predicted value is used, which is why the method is called semi-gradient. For
linear approximators this still converges, but to a TD fixed point rather than the true MSVE
minimum, meaning there is some residual bias that cannot be eliminated even with infinite data.
""",
            },
            {
                "title": "Choosing a Feature Basis",
                "body": """
The feature basis determines the class of functions that can be represented at all. State
aggregation partitions states into groups and produces a staircase approximation that cannot
capture smooth gradients within a group. Polynomial features can fit smooth curves but can
oscillate badly at high degree. Fourier cosine features are orthogonal, smooth, and excel at
representing functions with gradual or periodic structure, often outperforming polynomials at
equal feature count. Radial basis functions place Gaussian bumps at fixed centres and provide
local generalisation, so updating one region does not change distant regions. The best basis
depends on the known structure of the value function being approximated.
""",
            },
            {
                "title": "Neural Networks as Non-linear Approximators",
                "body": """
A multi-layer perceptron with nonlinear activations can represent a much richer class of
functions than any fixed linear basis, and back-propagation automatically computes gradients
through any depth. With gradient Monte Carlo the convergence guarantee that holds for linear
approximators extends in principle to neural networks because the target is unbiased, though
in practice training is more sensitive to learning rate and architecture. Combining neural
networks with bootstrapped targets such as TD(0) breaks the convergence guarantee for
prediction and creates the risk of divergence known as the deadly triad, which motivates
the stabilisation techniques used in deep RL such as experience replay and target networks.
""",
            },
        ],
    },
    {
        "label": "Section 8 — Function Approximation: Control",
        "concepts": [
            {
                "title": "Semi-gradient SARSA for Control",
                "body": """
Extending function approximation from prediction to control replaces the state-value function
with an action-value function that takes both the state and the chosen action as input. Semi-
gradient SARSA applies the on-policy TD update to these approximate action values, using the
next chosen action to form the bootstrapped target. For linear approximators combined with on-
policy data, this remains stable because the update distribution matches the behaviour, and the
method converges to the action-value TD fixed point rather than the true optimum, but the
resulting greedy policy is often close to optimal in practice.
""",
            },
            {
                "title": "Tile Coding for Continuous States",
                "body": """
Tile coding is a hand-crafted feature basis for continuous state spaces that uses multiple
overlapping grids called tilings, each covering the full space with a rectangular grid of tiles.
For any given state exactly one tile per tiling is active, giving a sparse binary feature vector
of length equal to the total number of tiles across all tilings. Because the features are binary,
computing the approximate value is a fast sum of the active weights rather than a dot product.
Overlapping tilings with asymmetric offsets ensure that nearby states share some tiles but not
all, providing smooth generalisation without sharp discontinuities at tile boundaries.
""",
            },
            {
                "title": "Mountain Car and Sparse Rewards",
                "body": """
Mountain Car is the canonical benchmark for continuous-state control with sparse rewards because
the car receives only minus one per step and the optimal solution requires the counter-intuitive
strategy of first reversing to build kinetic energy before accelerating toward the goal. With
zero Q-values at initialisation and no positive reinforcement until the goal is actually reached,
early episodes rely entirely on random exploration to stumble on the goal state and propagate
that reward signal backward. Once even a small portion of the Q-values near the goal become
negative rather than zero, the agent can begin following a consistent policy toward them and
learning accelerates substantially.
""",
            },
            {
                "title": "Average Reward for Continuing Tasks",
                "body": """
Discounted return with gamma less than one is designed for episodic tasks and when applied to
continuing tasks it implicitly limits the agent's planning horizon to roughly one over one minus
gamma steps, which can cause suboptimal behaviour if that horizon is shorter than important
delayed consequences. The average reward formulation instead optimises the long-run reward per
step with no discounting, and the differential TD error subtracts the running estimate of average
reward from each reward signal before bootstrapping. This centres the updates around zero on
average, ensuring stability, and converges to the true average-reward optimum for linear
approximators on ergodic continuing tasks where every state is eventually visited under any
reasonable policy.
""",
            },
        ],
    },
    {
        "label": "Section 9 — Off-Policy Methods with Function Approximation",
        "concepts": [
            {
                "title": "Importance Sampling with Function Approximation",
                "body": """
Off-policy learning uses experience generated by a behaviour policy b to evaluate or improve a
different target policy pi. With tabular methods, importance sampling weights each return by the
product of per-step ratios rho equal to pi of the action divided by b of the action. With function
approximation, the per-step ratio is folded directly into the semi-gradient update: the weight
increment is multiplied by rho before being applied. This re-weights the update distribution from
the behaviour distribution to the target distribution in expectation, giving an unbiased estimate
of the gradient of the value function under pi. The cost is higher variance, since large rho values
amplify individual updates and can cause erratic behaviour when pi and b differ substantially.
""",
            },
            {
                "title": "The Deadly Triad",
                "body": """
Three conditions individually permit convergent reinforcement learning algorithms, but combining
all three simultaneously can cause divergence. Function approximation means the value function
is represented by a parameterised family such as a neural network or linear combination of
features and cannot represent all value functions exactly. Bootstrapping means the update target
depends on the current estimate of the value function rather than on observed returns, as in TD
methods. Off-policy training means the distribution of states used in updates is generated by a
behaviour policy different from the one being evaluated. When all three are present, the update
operator is no longer a contraction and weights can grow without bound. Baird's counterexample
is the minimal known MDP that triggers this divergence with linear function approximation.
""",
            },
            {
                "title": "Gradient TD Methods (TDC and GTD2)",
                "body": """
Semi-gradient TD is not a true gradient method because it ignores the gradient of the bootstrap
target. Gradient TD methods instead minimise the Mean Squared Projected Bellman Error, which is a
well-defined scalar objective whose gradient is computable. TDC adds a correction term to the
semi-gradient update that involves a secondary weight vector h converging to the product of the
inverse feature covariance matrix and the expected TD error vector. At each step the primary
weights w are updated with a corrected gradient and the secondary weights h are updated with a
simple residual gradient step using a smaller step size beta. The result is an algorithm that
converges to the MSPBE minimum for linear function approximation even when all three deadly triad
conditions are active, at the cost of tracking an additional d-dimensional vector and tuning a
second learning rate.
""",
            },
        ],
    },
    {
        "label": "Section 10 — Eligibility Traces",
        "concepts": [
            {
                "title": "Eligibility Traces and the λ-Return",
                "body": """
An eligibility trace is a short-term memory vector z of the same dimension as the weight vector
that accumulates the recent history of feature activations. At each step the trace decays by
gamma times lambda and the current feature vector is added to it. When a TD error is computed,
the weight update is proportional to delta times z rather than delta times the current feature
vector alone. This means that all recently active features receive a share of the credit for the
current error, with the share decaying geometrically the further back in time the feature was
active. The effective depth of credit assignment is approximately one over one minus gamma lambda
steps, so lambda equal to zero gives pure one-step TD and lambda equal to one gives Monte Carlo.
The lambda-return is the forward-view equivalent: a geometric mixture of all n-step returns
weighted by lambda to the power of n minus one. Eligibility traces compute this mixture online
in constant memory rather than storing the full trajectory.
""",
            },
            {
                "title": "SARSA(λ) — Traces for Control",
                "body": """
SARSA(λ) extends TD(λ) to action-value function approximation by maintaining one eligibility
trace vector per action. At each step, the trace for the selected action is updated with the
current feature vector while all action traces decay by gamma lambda. The semi-gradient update
applies the TD error multiplied by the trace to the weights of every action. Replacing traces
are preferred for tile-coded features: instead of accumulating the feature vector into the
trace, active components are clamped to one. This prevents the trace from growing large when
the same tile is activated repeatedly within a single episode, which is common in tasks like
Mountain Car where the agent may oscillate before finding the goal. Replacing traces typically
converge faster and more stably than accumulating traces in control tasks.
""",
            },
            {
                "title": "Unifying n-step Returns and Eligibility Traces",
                "body": """
n-step TD and TD(λ) both interpolate between one-step TD and Monte Carlo returns and achieve
similar performance when their effective horizons are matched through the equivalence n
approximately equal to one over one minus lambda. The key practical advantage of eligibility
traces is memory and computational efficiency: n-step methods must store the last n transitions
and can only update weights n steps after an experience is observed, while traces require only
one extra d-dimensional vector and update weights at every step. For online learning and
streaming data, eligibility traces are therefore the preferred implementation. The
forward-view and backward-view are mathematically equivalent for on-policy linear TD but
diverge for off-policy settings, where corrections such as Retrace(λ) are needed to maintain
convergence while controlling the variance of importance sampling ratios.
""",
            },
        ],
    },
    {
        "label": "Section 11 — Policy Gradients",
        "concepts": [
            {
                "title": "REINFORCE — Monte Carlo Policy Gradient",
                "body": """
Rather than learning a value function and deriving a policy from it, REINFORCE directly
parameterises the policy and optimises it by gradient ascent on expected return. The Policy
Gradient Theorem gives the gradient as an expectation of the log probability of the chosen
action multiplied by the total return from that step onward. REINFORCE estimates this
expectation from a single complete episode, making it a Monte Carlo method. Because the
return is an unbiased sample of the true action value, the gradient estimate has no bias,
but it has very high variance because a single episode can have wildly different outcomes
depending on random transitions and the stochasticity of the policy itself.
""",
            },
            {
                "title": "REINFORCE with Baseline",
                "body": """
Adding a baseline to the REINFORCE update subtracts a state-dependent value from the
return before multiplying by the log probability gradient. Because the expected value of
the log probability over all actions is zero, the baseline does not change the expected
gradient direction, but it can dramatically reduce its variance. Using a learned state
value function as the baseline means the policy update is weighted by how much better the
chosen action turned out to be compared to what was expected on average from that state.
Actions that exceeded expectations get reinforced and actions that fell short get
suppressed, even if the overall episode was profitable, which produces a cleaner credit
assignment signal than the raw return.
""",
            },
            {
                "title": "Actor-Critic",
                "body": """
Actor-Critic replaces the Monte Carlo return used in REINFORCE with a one-step TD
bootstrap: the immediate reward plus the discounted estimated value of the next state
minus the estimated value of the current state. This TD error serves as a biased but
lower-variance estimate of the advantage, allowing the policy to be updated after every
single step rather than at episode end. The critic network is trained to minimise the TD
error and thereby provide better advantage estimates for the actor. Compared with
REINFORCE, Actor-Critic typically learns faster because each transition produces an
immediate update, but the bootstrapped advantage is biased since the value estimates are
initially inaccurate, so the policy can be misled early in training.
""",
            },
            {
                "title": "PPO — Proximal Policy Optimisation",
                "body": """
Vanilla policy gradient methods can take a destructively large update step if the gradient
happens to point strongly in one direction for a given batch of data, collapsing the
policy to a degenerate distribution from which recovery is slow. PPO prevents this by
clipping the probability ratio between the new and old policy before multiplying by the
advantage, so the objective becomes flat once the ratio moves outside a narrow interval
around one. This means that even if the gradient points strongly toward a large update,
the objective saturates and the gradient signal cuts off, keeping the policy within a
trust region. In practice PPO uses multiple passes of gradient descent on each batch of
collected transitions, the generalised advantage estimator for lower-variance advantage
estimates, and a shared actor-critic architecture, making it the dominant policy gradient
algorithm for complex environments including game playing and language model fine-tuning.
""",
            },
        ],
    },
    {
        "label": "Section 12 — Applications and Case Studies",
        "concepts": [
            {
                "title": "TD-Gammon and Self-Play",
                "body": """
TD-Gammon demonstrated in 1992 that a neural network trained by TD self-play could reach
world-class performance in backgammon without any human-designed evaluation function. The
system used TD lambda with a small multilayer perceptron whose output was the estimated
probability of winning from a given board position. Self-play naturally provides an
infinite supply of on-policy training data and an automatically calibrated opponent,
since the agent always faces an opponent at its own current skill level. Backgammon was
particularly well-suited because the dice rolls provide stochastic forced exploration,
ensuring that training data covers diverse board states. The success of TD-Gammon directly
inspired AlphaGo and AlphaZero, which extended the same self-play paradigm to Go using
Monte Carlo Tree Search for lookahead.
""",
            },
            {
                "title": "AlphaGo — Combining Search with Learned Knowledge",
                "body": """
Go has a branching factor of around 250 and game lengths of over 150 moves, making
exhaustive tree search completely infeasible. AlphaGo solved this by using neural networks
to reduce both the breadth and depth of Monte Carlo Tree Search. A policy network trained
first by supervised learning on expert moves and then fine-tuned by policy gradient
self-play provides move priors that concentrate search on promising actions, reducing
effective branching. A value network trained on self-play outcomes evaluates board positions
without simulating to game end, reducing effective depth. MCTS combines these using the
PUCT formula to balance exploitation of high-value actions with exploration of
high-prior low-visit actions. AlphaGo Zero later removed the supervised learning phase
entirely, training a single combined policy-value network from scratch through self-play
and surpassing the original AlphaGo in three days.
""",
            },
            {
                "title": "Scaling RL — From Games to Language",
                "body": """
The progression from TD-Gammon to modern systems traces a thirty-year arc of algorithmic
advances and scaling. Deep Q-Networks extended tabular TD to raw pixel inputs using
experience replay and target networks to stabilise training against the deadly triad.
Policy gradient methods scaled further by operating on continuous action spaces and
directly optimising parameterised policies. PPO became the dominant algorithm for
real-time multi-agent games like Dota 2 and StarCraft II, where the action space is too
large for value-based methods. Reinforcement learning from human feedback applied PPO to
language model fine-tuning, using a reward model trained on human preference comparisons
as the reward signal. At each stage the core algorithmic ideas remained recognisable
descendants of what is covered in Sutton and Barto, with engineering for stability and
scale as the primary differentiator.
""",
            },
        ],
    },
    {
        "label": "Section 13 — Advanced Topics",
        "concepts": [
            {
                "title": "Hierarchical RL and the Options Framework",
                "body": """
Standard RL agents choose one primitive action per step, which means that rewards hundreds
of steps away must propagate through hundreds of individual updates before the early actions
that caused them receive any useful credit signal. Hierarchical RL breaks this by introducing
temporally extended actions called options, each consisting of an initiation set specifying
where the option can start, an intra-option policy that selects primitive actions for as long
as the option runs, and a termination condition that decides when the option ends. When a
high-level agent invokes an option it receives a single lumped reward and transition after
potentially many primitive steps, so the credit assignment distance shrinks dramatically.
The canonical four-rooms problem illustrates this: without options the agent must bridge the
full chain of primitive steps between rooms, but with doorway-navigation options it only
needs to learn a three-decision chain regardless of how many steps each option takes.
Intra-option Q-learning extends credit further by updating option values during execution
rather than only at termination, and learned option methods such as Option-Critic discover
useful sub-goals automatically from self-play without domain knowledge.
""",
            },
            {
                "title": "Count-Based Exploration and Intrinsic Motivation",
                "body": """
The exploration-exploitation tradeoff becomes much harder in large state spaces because
epsilon-greedy exploration is undirected and will repeatedly revisit familiar states without
any memory of what has already been seen. Count-based exploration adds an intrinsic reward
equal to beta divided by the square root of the visit count for the reached state, which
gives a large bonus for genuinely novel states and a decaying bonus as states are revisited.
This automatically directs the agent toward unexplored regions without requiring a separate
exploration schedule. When states are continuous and exact counts are always one, pseudo-count
methods use a density model to estimate how familiar the agent is with a region of state
space and compute a generalised count from that density. The Intrinsic Curiosity Module takes
a different approach by using the prediction error of a learned forward dynamics model as the
novelty signal, which generalises across visually similar states through the shared embedding.
Random Network Distillation achieves similar results more simply by measuring how well a
trained predictor network matches a fixed random target network, with high prediction error
indicating states that the predictor has not been trained on enough, which is precisely the
states that have been visited least.
""",
            },
            {
                "title": "Meta-RL — Learning to Explore",
                "body": """
Rather than designing a fixed exploration strategy such as epsilon-greedy or a specific
intrinsic bonus, meta-RL aims to learn an exploration policy from experience across many
related tasks so that the agent adapts its exploration to the structure of new tasks it
encounters. RL-squared trains a recurrent policy across a distribution of tasks and finds
that the hidden state naturally develops an adaptive exploration strategy where the agent
explores broadly early in an episode and exploits more aggressively as it accumulates
evidence about the current task. MAML optimises network parameters so that a single gradient
step adapts them to a new task, effectively learning a prior over tasks that makes individual
tasks easy to fine-tune. The key distinction from intrinsic motivation is that meta-RL learns
what to explore given task structure, rather than simply rewarding novelty regardless of
whether that novelty is relevant to the current objective.
""",
            },
            {
                "title": "Multi-Agent RL and Game Theory",
                "body": """
When multiple learning agents share an environment, each agent's changing policy makes the
environment non-stationary from every other agent's perspective, breaking the convergence
guarantees that single-agent Q-learning relies on. Game theory provides the solution concept:
a Nash equilibrium is a joint policy where no agent can unilaterally improve its payoff by
deviating, and self-play methods like those used in AlphaGo Zero converge to approximate
Nash equilibria by training each agent against copies of itself. Independent Q-learning, where
each agent runs its own Q-learning while ignoring the others, is the simplest approach and
works in practice despite lacking theoretical guarantees. Centralised training with
decentralised execution resolves the non-stationarity problem by giving each agent a critic
that sees the joint observation during training but acting on local observations at deployment,
enabling efficient credit assignment without requiring communication at test time. The
iterated prisoner's dilemma illustrates the social dilemma structure that pervades multi-agent
problems: individually rational behaviour leads to collectively worse outcomes, but strategies
like Tit-for-Tat that reward cooperation and retaliate against defection can sustain
cooperation when they form a sufficient fraction of the population.
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
