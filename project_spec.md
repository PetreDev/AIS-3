This file contains the full project specification. Cursor should follow this document exactly when generating the solution.

YOU ARE TO IMPLEMENT THE FOLLOWING FULL HOMEWORK PROJECT.
READ EVERYTHING CAREFULLY BEFORE GENERATING CODE.

I need you to create a complete solution (Python project or single notebook) implementing:

1. DATA PREPARATION
   Dataset

Use the CSIC 2010 HTTP dataset(its in the folder dataset and it is called csic_database.csv).

Each sample contains:

URL string

Label: 0 = benign, 1 = attack

Feature Extraction

Extract numerical features from each URL, including:

URL length

Number of special characters (?, &, =, %, ', ", <, >, ;, @, {}, [])

Number of digits

Number of parameters

Presence of suspicious keywords (binary feature): ["union", "select", "drop", "script", "alert", "insert"]

Dataset Split

Prepare:

training set

validation set

test set

Minimum: 10,000 total events.

2. STATE DISCRETIZATION (VERY IMPORTANT)

Implement both:

A) Binning (fixed intervals)

Example:

URL length: 0–100 = bin 0, 100–200 = bin 1, >200 = bin 2

Special chars: 0–5 = bin 0, 6–10 = bin 1, >10 = bin 2

B) Quantization (quantiles)

Example:

4 quantiles: 0–25%, 25–50%, 50–75%, 75–100%

The discretized features must be combined to form a discrete state index.

Provide a function:

state = encode_state(features)
that maps features → discrete state index.

3. RL ENVIRONMENT (MDP)

Implement a custom environment class with:

State (sᵗ)

Discrete integer representing:

url_length_bin

specchar_bin

keyword_presence
(and possibly more bins)

Actions

Two simple actions:

0 = ALLOW

1 = BLOCK

Reward Function
TP (attack & BLOCK): +2.0
TN (benign & ALLOW): +0.5
FP (benign & BLOCK): -5.0
FN (attack & ALLOW): -8.0

Episode Structure

Each episode = 1000 consecutive requests.

Total training = ≥ 10 episodes (≥ 10,000 events).

The environment must expose:

reset()

step(action)

get_state()

is_done()

4. Q-LEARNING ALGORITHM

Implement Tabular Q-learning with the Bellman update:

Q(s*t, a_t) ← Q(s_t, a_t) + α [ r_t + γ max_a Q(s*{t+1}, a) − Q(s_t, a_t) ]

Hyperparameters (use and justify inside comments)
α = 0.1
γ = 0.90
ϵ_start = 1.0
ϵ_min = 0.05
ϵ_decay = 0.98 per episode

Exploration Strategy

Use ϵ-greedy.

Produce:

Q-table as a 2D numpy array [num_states × 2]

Save final Q-table to file.

5. BASELINE APPROACH

Implement a simple rule-based baseline:

If number_of_suspicious_features > K:
BLOCK
else:
ALLOW

Use K = 2 (or tune automatically).

6. EVALUATION

Evaluate both RL agent and baseline on the test set using:

Metrics:

Precision

Recall

F1-score

Accuracy

Average reward per episode

Visualizations:

Learning curve (avg reward per episode)

Confusion matrix for RL agent

Confusion matrix for baseline

Optional: heatmap of Q-table

Save all graphs as PNG.

7. REPORT (GENERATE THIS TOO)

Auto-generate a clear report (Markdown or TXT) including:

A) Description of the RL environment

state

actions

rewards

B) Explanation of Binning & Quantization

Use the examples:

Binning:

special chars 0–5 → bin 1

6–10 → bin 2

10 → bin 3

Quantization:

automatically compute quantile boundaries

example boundaries: 2,7,15

C) Q-table Explanation

rows = states

columns = actions ALLOW/BLOCK

Q(s,a) meaning

Bellman update equation

Example calculation (use the one I provided):

update of Q(s₀, BLOCK) with reward -5 → becomes -0.5

D) Comparison of RL vs Baseline

Include:

tables

confusion matrices

learning curves

E) How Reward Shaping Affects FP/FN

Explain:

high FN penalty pushes recall

high FP penalty pushes precision

8. DELIVERABLES

Produce all of this inside Cursor:

✔ Complete working Python code
✔ A reproducible training pipeline
✔ Saved plots (PNG)
✔ Saved Q-table
✔ evaluation files (JSON or TXT)
✔ Final report (Markdown/TXT)

IMPORTANT REQUIREMENTS

Code must run end-to-end with no missing variables.

Use numpy + pandas + matplotlib only (no Gym needed).

If CSIC 2010 CSV is missing, generate a synthetic dataset automatically.

Add detailed comments for every major step for clarity.

Now generate the full solution as described above.
