# Complete Codebase Explanation

This document provides a comprehensive explanation of every file, module, class, and function in the URL Security Classification using Q-Learning project. This is designed to help you understand and explain the entire codebase.

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Project Structure](#project-structure)
3. [Detailed File-by-File Explanation](#detailed-file-by-file-explanation)
   - [data_preparation.py](#data_preparationpy)
   - [baseline.py](#baselinepy)
   - [environment.py](#environmentpy)
   - [q_learning.py](#q_learningpy)
   - [training.py](#trainingpy)
   - [evaluation.py](#evaluationpy)
   - [main.py](#mainpy)
4. [Data Flow and Pipeline](#data-flow-and-pipeline)
5. [Key Algorithms and Concepts](#key-algorithms-and-concepts)

---

## Project Overview

This project implements a **Reinforcement Learning (RL) solution for URL-based attack detection** using the Q-learning algorithm. The system learns to classify HTTP requests as either **benign (ALLOW)** or **malicious (BLOCK)** by training on the CSIC 2010 HTTP dataset.

### Core Concept

The problem is framed as a **Markov Decision Process (MDP)**:
- **States**: Discretized representations of URL features (length, special characters, parameters, etc.)
- **Actions**: ALLOW (0) or BLOCK (1) the request
- **Rewards**: Based on classification correctness with emphasis on security (higher penalty for missing attacks)
- **Goal**: Learn the optimal policy that maximizes cumulative reward

---

## Project Structure

```
.
├── data_preparation.py    # Handles data loading, feature extraction, state discretization
├── baseline.py            # Rule-based baseline classifier for comparison
├── environment.py         # RL environment (MDP) implementation
├── q_learning.py          # Q-learning algorithm implementation
├── training.py            # Training pipeline orchestration
├── evaluation.py          # Evaluation metrics, visualizations, and comparison
├── main.py                # Main script that orchestrates the entire pipeline
├── requirements.txt       # Python dependencies
├── dataset/
│   └── csic_database.csv  # CSIC 2010 HTTP dataset
├── plots/                 # Generated visualizations
└── CODEBASE_EXPLANATION.md # This file
```

---

## Detailed File-by-File Explanation

### data_preparation.py

**Purpose**: Handles all data-related operations including loading, cleaning, feature extraction, normalization, discretization, and state encoding.

#### Class: `DataPreparator`

The main class that encapsulates all data preprocessing functionality.

##### Key Attributes:

- **`dataset_path`**: Path to the CSIC 2010 dataset CSV file
- **`suspicious_keywords`**: List of suspicious keywords (e.g., "union", "select", "script", "alert") used to detect attacks
- **`special_chars`**: List of special characters that may indicate attacks
- **`sql_patterns`**: Regular expressions for detecting SQL injection patterns
- **`xss_patterns`**: Regular expressions for detecting XSS (Cross-Site Scripting) patterns
- **`binning_bounds`** & **`quantile_bounds`**: Parameters for discretization methods
- **`normalize_features`**: Flag to enable/disable feature normalization
- **`scaler`**: Scaler object (RobustScaler or MinMaxScaler) for normalization
- **`handle_imbalance`**: Flag to enable/disable class imbalance handling

##### Key Methods:

1. **`load_data()`**
   - Loads the CSIC 2010 dataset from CSV
   - Performs data cleaning (removes duplicates, handles missing values)
   - Validates data quality and class distribution
   - Handles class imbalance if detected (upsampling or downsampling)
   - Falls back to synthetic data generation if dataset not found
   - Returns: DataFrame with 'URL' and 'label' columns

2. **`_clean_data(data)`**
   - Removes HTTP/1.1 suffixes
   - Strips whitespace
   - Normalizes URL encoding
   - Removes empty URLs

3. **`_validate_data(data)`**
   - Checks for missing values
   - Analyzes URL length distribution
   - Detects class imbalance (warns if ratio > 30%)
   - Returns: Boolean indicating if imbalance detected

4. **`_balance_classes(data)`**
   - Handles class imbalance using upsampling or downsampling
   - Uses sklearn's `resample` function
   - Targets a 50/50 class ratio by default
   - Shuffles the balanced dataset

5. **`_generate_synthetic_data(n_samples=15000)`**
   - Generates synthetic URL data for testing when real dataset unavailable
   - Creates benign URLs (normal patterns) and attack URLs (SQL injection, XSS patterns)
   - Returns: DataFrame with synthetic URLs and labels

6. **`extract_features(data)`**
   - **Main feature extraction function** - extracts 20+ features from URLs
   - **Basic Features**:
     - `url_length`: Total character count
     - `num_special_chars`: Count of special characters
     - `num_digits`: Count of digits
     - `num_parameters`: Number of query parameters
   - **Advanced Features**:
     - `url_entropy`: Shannon entropy (measure of randomness/complexity)
     - `path_depth`: Number of path segments
     - `has_file_extension`: Binary feature
     - `special_char_ratio`: Ratio of special characters
     - `digit_ratio`: Ratio of digits
   - **Security-Specific Features**:
     - `has_suspicious_keywords`: Binary (1 if keywords found)
     - `suspicious_keyword_count`: Count of keywords
     - `has_sql_pattern`: Binary (1 if SQL injection pattern detected)
     - `has_xss_pattern`: Binary (1 if XSS pattern detected)
     - `has_encoding_patterns`: Binary (double encoding, etc.)
     - `has_suspicious_combos`: Binary (path traversal, etc.)
   - **URL Structure Features**:
     - `has_query`: Binary (has query string)
     - `has_fragment`: Binary (has fragment)
     - `avg_param_length`: Average length of parameter values
   - Handles NaN and infinite values
   - Clips outliers (beyond 3 standard deviations)
   - Returns: DataFrame with features and label column

7. **Helper Methods for Feature Extraction**:
   - **`_count_parameters(url)`**: Counts query parameters using URL parsing
   - **`_calculate_entropy(url)`**: Calculates Shannon entropy (information content measure)
   - **`_get_path_depth(url)`**: Counts path segments
   - **`_has_file_extension(url)`**: Checks for file extension
   - **`_detect_sql_patterns(url)`**: Uses regex to detect SQL injection attempts
   - **`_detect_xss_patterns(url)`**: Uses regex to detect XSS attempts
   - **`_detect_encoding_patterns(url)`**: Detects URL encoding anomalies
   - **`_detect_suspicious_combinations(url)`**: Detects suspicious character combinations (e.g., "..", "//")
   - **`_get_avg_param_length(url)`**: Calculates average parameter value length

8. **`fit_normalization(features)`**
   - Fits a scaler (RobustScaler or MinMaxScaler) on training data
   - Only normalizes core features: url_length, num_special_chars, num_digits, num_parameters
   - RobustScaler is preferred as it handles outliers better

9. **`normalize_features_data(features)`**
   - Applies fitted normalization to features
   - Returns normalized DataFrame

10. **`fit_discretization(features, method='binning')`**
    - Fits discretization parameters based on training data
    - **Two methods**:
      - **Binning**: Fixed intervals (e.g., URL length: 0-100, 100-200, >200)
      - **Quantization**: Quantile-based (4 quantiles: 0-25%, 25-50%, 50-75%, 75-100%)
    - Optionally normalizes features before discretization for better bin boundaries
    - Stores bin boundaries and number of bins per feature

11. **`discretize_feature(value, feature_name)`**
    - Discretizes a single feature value into a bin index
    - Uses either binning or quantization based on fitted method
    - Returns: Bin index (integer)

12. **`encode_state(features_row)`**
    - **Critical function**: Converts feature vector to discrete state index
    - **State encoding combines**:
      - `url_length_bin` (0-2 or 0-3)
      - `specchar_bin` (0-2 or 0-3)
      - `digits_bin` (0-2 or 0-3)
      - `params_bin` (0-2 or 0-3)
      - `keyword_presence` (0 or 1)
      - `sql_pattern` (0 or 1)
      - `xss_pattern` (0 or 1)
    - Uses **base conversion** to combine all features into a single integer state
    - Formula: `state = url_length_bin * (specchar_max * digits_max * params_max * 2 * 2 * 2) + specchar_bin * (digits_max * params_max * 2 * 2 * 2) + ...`
    - Returns: Discrete state index (integer)

13. **`get_num_states()`**
    - Calculates total number of possible discrete states
    - Formula: `url_length_bins × specchar_bins × digits_bins × params_bins × 2 × 2 × 2`
    - Returns: Total number of states

14. **`split_data(data, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15)`**
    - Splits data into training, validation, and test sets
    - Uses stratified sampling to maintain class distribution
    - Ensures minimum 10,000 samples required
    - Returns: Tuple of (train_data, val_data, test_data)

---

### baseline.py

**Purpose**: Implements a simple rule-based baseline classifier for comparison with the RL agent.

#### Class: `RuleBasedBaseline`

A simple threshold-based classifier that counts suspicious features.

##### Key Attributes:

- **`K`**: Threshold for number of suspicious features (default: 2)
- **`special_chars_threshold`**: 5 (if more → suspicious)
- **`parameters_threshold`**: 3 (if more → suspicious)
- **`url_length_threshold`**: 150 (if longer → suspicious)
- **`suspicious_keywords`**: List of suspicious keywords

##### Key Methods:

1. **`count_suspicious_features(features_row)`**
   - Counts how many suspicious features are present:
     - Special chars > threshold? → +1
     - Parameters > threshold? → +1
     - Has suspicious keywords? → +1
     - URL length > threshold? → +1
   - Returns: Count (0-4)

2. **`predict(features_row)`**
   - Main prediction function
   - Rule: If `count_suspicious_features > K`, then **BLOCK (1)**, else **ALLOW (0)**
   - Returns: Action (0 = ALLOW, 1 = BLOCK)

3. **`predict_batch(features_df)`**
   - Applies prediction to a batch of URLs
   - Returns: Array of predictions

4. **`tune_threshold(features_df, labels, K_values=None)`**
   - Automatically tunes the threshold K to maximize accuracy
   - Tests K values [0, 1, 2, 3, 4] by default
   - Returns: Best K value and corresponding accuracy

**Why it's useful**: Provides a simple baseline to compare against the RL agent's performance. Demonstrates that RL can learn more sophisticated patterns than simple rule-based approaches.

---

### environment.py

**Purpose**: Implements the RL environment as a Markov Decision Process (MDP) for URL security classification.

#### Class: `URLSecurityEnvironment`

Implements the OpenAI Gym-like interface for RL training.

##### Key Attributes:

- **`data`**: DataFrame with features and labels
- **`data_preparator`**: DataPreparator instance for state encoding
- **`episode_length`**: Number of steps per episode (default: 1000)
- **Reward structure**:
  - `reward_tp = 2.5`: True Positive (attack & BLOCK)
  - `reward_tn = 0.5`: True Negative (benign & ALLOW)
  - `reward_fp = -5.0`: False Positive (benign & BLOCK)
  - `reward_fn = -8.0`: False Negative (attack & ALLOW) - **Highest penalty**

##### Key Methods:

1. **`reset()`**
   - Resets environment to start a new episode
   - Samples a random starting point in the dataset
   - Encodes initial state using `data_preparator.encode_state()`
   - Resets step counter and reward tracker
   - Returns: Initial state (discrete integer)

2. **`step(action)`**
   - **Core environment step function**
   - Takes action (0 = ALLOW, 1 = BLOCK)
   - Compares action with true label to calculate reward:
     - Action 1 (BLOCK) + Label 1 (Attack) → TP → +2.5
     - Action 1 (BLOCK) + Label 0 (Benign) → FP → -5.0
     - Action 0 (ALLOW) + Label 0 (Benign) → TN → +0.5
     - Action 0 (ALLOW) + Label 1 (Attack) → FN → -8.0
   - Moves to next sample (wraps around if needed)
   - Encodes next state
   - Returns: Tuple of `(next_state, reward, done, info)`
     - `next_state`: Next discrete state (None if done)
     - `reward`: Calculated reward
     - `done`: Boolean indicating if episode finished
     - `info`: Dictionary with true_label, prediction, correctness, step number

3. **`get_state()`**: Returns current state
4. **`is_done()`**: Checks if episode is finished
5. **`get_episode_reward()`**: Returns total reward for current episode
6. **`get_episode_average_reward()`**: Returns average reward per step

**Reward Design Philosophy**:
- **FN penalty (-8.0) is highest**: Missing an attack is the worst outcome (security-critical)
- **FP penalty (-5.0) is moderate**: Blocking benign requests is bad but not as critical
- **TP reward (+2.5)**: Rewards correct attack detection
- **TN reward (+0.5)**: Small reward for allowing legitimate traffic

This design encourages the agent to prioritize security (high recall) while balancing precision.

---

### q_learning.py

**Purpose**: Implements the Q-learning algorithm with tabular Q-table.

#### Class: `QLearningAgent`

Implements tabular Q-learning with ε-greedy exploration.

##### Key Attributes:

- **`num_states`**: Total number of discrete states
- **`num_actions`**: Number of actions (2: ALLOW=0, BLOCK=1)
- **Hyperparameters**:
  - **`alpha` (α)**: Learning rate (default: 0.1) - controls how quickly Q-values update
  - **`gamma` (γ)**: Discount factor (default: 0.90) - values future rewards
  - **`epsilon` (ε)**: Current exploration rate (starts at 1.0, decays to 0.01)
  - **`epsilon_start`**: 1.0 (100% exploration initially)
  - **`epsilon_min`**: 0.01 (1% exploration minimum)
  - **`epsilon_decay`**: 0.95 (exponential decay per episode)
  - **`alpha_decay`**: 0.9995 (learning rate decay for stability)
- **`q_table`**: NumPy array of shape `[num_states × num_actions]`
  - Initialized optimistically with 0.1 (encourages exploration)
  - Each cell `Q[s, a]` represents expected cumulative reward for action `a` in state `s`

##### Key Methods:

1. **`select_action(state, training=True)`**
   - **ε-greedy policy**: Balances exploration vs exploitation
   - **During training**: 
     - With probability ε: **Explore** (random action)
     - With probability (1-ε): **Exploit** (greedy action: `argmax(Q[state])`)
   - **During evaluation** (`training=False`): Always greedy (no exploration)
   - Returns: Selected action (0 or 1)

2. **`update(state, action, reward, next_state, done)`**
   - **Bellman update equation**:
     ```
     Q(s_t, a_t) ← Q(s_t, a_t) + α [ r_t + γ max_a Q(s_{t+1}, a) − Q(s_t, a_t) ]
     ```
   - **Steps**:
     1. Get current Q-value: `Q[state, action]`
     2. If done (terminal): `target_q = reward`
     3. If not done: `target_q = reward + γ * max(Q[next_state])`
     4. Update: `Q[state, action] = current_q + α * (target_q - current_q)`
   - This update moves Q-values toward the optimal action-value function

3. **`decay_epsilon()`**
   - Decays exploration rate after each episode
   - Formula: `ε = max(ε_min, ε * ε_decay)`
   - Gradually shifts from exploration to exploitation

4. **`decay_alpha()`**
   - Decays learning rate for stable convergence
   - Formula: `α = max(0.01, α * α_decay)`

5. **`get_q_value(state, action)`**: Returns Q-value for state-action pair
6. **`get_policy(state)`**: Returns greedy action (best action according to Q-table)

7. **`save_q_table(filepath)`**: Saves Q-table as NumPy file
8. **`load_q_table(filepath)`**: Loads Q-table from NumPy file
9. **`save_agent(filepath)`**: Saves entire agent (Q-table + hyperparameters) as pickle
10. **`load_agent(filepath)`**: Loads entire agent from pickle

**Algorithm Explanation**:

Q-learning is a **model-free, off-policy** RL algorithm:
- **Model-free**: Doesn't need a model of the environment (learns from experience)
- **Off-policy**: Can learn optimal policy while following exploratory policy (ε-greedy)
- **Tabular**: Uses a table to store Q-values (works for discrete state spaces)

The algorithm converges to optimal Q-values Q*(s,a) through repeated Bellman updates, allowing the agent to learn the best action in each state.

---

### training.py

**Purpose**: Orchestrates the training of the Q-learning agent.

#### Function: `train_q_learning_agent(env, agent, num_episodes=10, verbose=True)`

Main training loop that runs episodes of agent-environment interaction.

##### Process:

1. **For each episode**:
   - Reset environment (get initial state)
   - Initialize episode variables (total_reward, steps)
   
2. **For each step in episode** (until done):
   - Agent selects action using ε-greedy policy: `action = agent.select_action(state, training=True)`
   - Environment executes action: `next_state, reward, done, info = env.step(action)`
   - Agent updates Q-table: `agent.update(state, action, reward, next_state, done)`
   - Move to next state: `state = next_state`
   - Accumulate reward and step count
   
3. **After episode**:
   - Store episode statistics (total reward, length)
   - Decay exploration rate: `agent.decay_epsilon()`
   - Decay learning rate: `agent.decay_alpha()`
   - Print progress (if verbose)

4. **After all episodes**:
   - Store training statistics in agent
   - Return dictionary with training metrics

##### Returns:

Dictionary with:
- `episode_rewards`: List of total rewards per episode
- `episode_lengths`: List of episode lengths
- `average_reward`: Average reward across all episodes
- `final_epsilon`: Final exploration rate

**Training Statistics**: Tracked for visualization (learning curve) and analysis.

---

### evaluation.py

**Purpose**: Handles evaluation metrics, visualizations, and comparison between RL agent and baseline.

#### Key Functions:

1. **`evaluate_agent(agent, env, data_preparator, test_data, verbose=True)`**
   - Evaluates Q-learning agent on test data
   - **Process**:
     - For each test sample:
       - Encode state using `data_preparator.encode_state()`
       - Get action from agent using greedy policy (no exploration)
       - Calculate reward based on action vs true label
       - Store prediction and true label
   - **Calculates metrics**:
     - Accuracy, Precision, Recall, F1-Score
     - Average reward
     - Confusion matrix (TP, TN, FP, FN)
   - Returns: Dictionary with all metrics

2. **`evaluate_baseline(baseline, test_data, verbose=True)`**
   - Evaluates baseline classifier on test data
   - Gets batch predictions using `baseline.predict_batch()`
   - Calculates same metrics as RL agent
   - Returns: Dictionary with metrics

3. **`plot_learning_curve(agent, save_path)`**
   - Plots training progress over episodes
   - Shows:
     - Raw episode rewards (light blue, semi-transparent)
     - Moving average (red line, smoothed trend)
   - Helps visualize learning progress and convergence

4. **`plot_confusion_matrix(cm, title, save_path)`**
   - Creates visual confusion matrix
   - Shows TP, TN, FP, FN with color intensity
   - Labels: ALLOW vs BLOCK (rows and columns)

5. **`plot_q_table_heatmap(agent, save_path, max_states=100)`**
   - Visualizes Q-table values as heatmap
   - Shows Q-values for each state-action pair
   - Color scale: Red (low Q-values) to Green (high Q-values)
   - Limited to first 100 states for readability

6. **`save_evaluation_results(rl_results, baseline_results, save_path)`**
   - Saves evaluation metrics to JSON file
   - Excludes predictions/true_labels (too large)
   - Useful for comparison and reporting

7. **`generate_all_visualizations(agent, rl_results, baseline_results, output_dir)`**
   - Generates all required visualizations:
     - Learning curve
     - RL agent confusion matrix
     - Baseline confusion matrix
     - Q-table heatmap
   - Saves all plots to specified directory

**Metrics Explained**:
- **Accuracy**: (TP + TN) / (TP + TN + FP + FN) - Overall correctness
- **Precision**: TP / (TP + FP) - Of all blocked, how many were attacks?
- **Recall**: TP / (TP + FN) - Of all attacks, how many were detected?
- **F1-Score**: 2 × (Precision × Recall) / (Precision + Recall) - Harmonic mean

---

### main.py

**Purpose**: Main orchestration script that runs the entire pipeline from start to finish.

#### Function: `main()`

Executes the complete pipeline in 7 steps:

1. **Data Preparation**:
   - Creates `DataPreparator` instance
   - Loads data from CSV
   - Extracts features from URLs
   - Fits discretization (quantization method)
   - Calculates number of states
   - Splits data into train/val/test sets

2. **Baseline Approach**:
   - Creates `RuleBasedBaseline` instance
   - Tunes threshold K on validation set
   - Sets up baseline for comparison

3. **RL Environment Setup**:
   - Creates `URLSecurityEnvironment` with training data
   - Configures episode length (1000 steps)
   - Prints reward structure

4. **Q-Learning Agent Setup**:
   - Creates `QLearningAgent` with hyperparameters:
     - Learning rate: 0.2
     - Discount factor: 0.95
     - Exploration: ε starts at 1.0, decays to 0.01
   - Initializes Q-table

5. **Training**:
   - Trains agent for 15 episodes (≥ 10,000 events requirement)
   - Saves Q-table and agent to files
   - Stores training statistics

6. **Evaluation**:
   - Evaluates RL agent on test set
   - Evaluates baseline on test set
   - Generates all visualizations (learning curve, confusion matrices, Q-table heatmap)
   - Saves evaluation results to JSON

7. **Report Generation**:
   - Calls `generate_report()` to create comprehensive markdown report
   - Includes all sections required for documentation

#### Function: `generate_report(agent, preparator, rl_results, baseline_results, training_stats, env)`

Generates a comprehensive markdown report (`report.md`) with:

- **Section A**: RL Environment Description (states, actions, rewards)
- **Section B**: State Discretization (binning & quantization explanation)
- **Section C**: Q-Learning Algorithm Implementation and Configuration
- **Section D**: Comparison: RL Agent vs Baseline (metrics, confusion matrices)
- **Section E**: How Reward Shaping Affects FP/FN
- **Section F**: Conclusion and Summary

**Output Files Created**:
- `q_table.npy`: Saved Q-table
- `q_learning_agent.pkl`: Saved trained agent
- `evaluation_results.json`: Evaluation metrics
- `plots/learning_curve.png`: Training progress
- `plots/confusion_matrix_rl.png`: RL agent performance
- `plots/confusion_matrix_baseline.png`: Baseline performance
- `plots/q_table_heatmap.png`: Q-table visualization
- `report.md`: Comprehensive report

---

## Data Flow and Pipeline

### Complete Pipeline Flow

```
1. DATA LOADING (data_preparation.py)
   ↓
   [CSIC 2010 Dataset] → Clean → Validate → Balance → Split
   ↓
   [Train/Val/Test Sets]

2. FEATURE EXTRACTION (data_preparation.py)
   ↓
   [URLs] → Extract Features → Normalize → Discretize
   ↓
   [Feature Vectors] → Encode State
   ↓
   [Discrete States]

3. ENVIRONMENT SETUP (environment.py)
   ↓
   [Training Data] → Create Environment
   ↓
   [URLSecurityEnvironment with MDP]

4. AGENT SETUP (q_learning.py)
   ↓
   [Number of States] → Initialize Q-table
   ↓
   [QLearningAgent]

5. TRAINING LOOP (training.py)
   ↓
   For each episode:
     Reset Environment → [Initial State]
     ↓
     For each step:
       Agent selects action (ε-greedy)
       ↓
       Environment returns (next_state, reward, done)
       ↓
       Agent updates Q-table (Bellman update)
       ↓
       Move to next state
     ↓
     Decay epsilon & alpha
   ↓
   [Trained Agent with Q-table]

6. EVALUATION (evaluation.py)
   ↓
   [Test Data] → Evaluate Agent → Metrics
   [Test Data] → Evaluate Baseline → Metrics
   ↓
   Generate Visualizations
   ↓
   [Plots + JSON Results]

7. REPORT GENERATION (main.py)
   ↓
   [All Results] → Generate Report
   ↓
   [report.md]
```

### State Encoding Flow

```
URL → Extract Features → Normalize → Discretize → Encode State
"http://example.com/page?id=1' OR '1'='1"
    ↓
[url_length=45, num_special_chars=8, num_digits=1, 
 num_parameters=1, has_suspicious_keywords=1, 
 has_sql_pattern=1, has_xss_pattern=0]
    ↓
[url_length_bin=0, specchar_bin=1, digits_bin=0, 
 params_bin=0, keyword=1, sql=1, xss=0]
    ↓
State Index = (0 × 3 × 3 × 3 × 2 × 2 × 2) + 
              (1 × 3 × 3 × 2 × 2 × 2) + 
              (0 × 3 × 2 × 2 × 2) + 
              (0 × 2 × 2 × 2) + 
              (1 × 2 × 2) + 
              (1 × 2) + 
              0
    ↓
State = 1234 (example)
```

---

## Key Algorithms and Concepts

### 1. Q-Learning Algorithm

**Type**: Model-free, off-policy, value-based RL

**Key Equation (Bellman Update)**:
```
Q(s_t, a_t) ← Q(s_t, a_t) + α [ r_t + γ max_a Q(s_{t+1}, a) − Q(s_t, a_t) ]
```

**Steps**:
1. Initialize Q-table with optimistic values (0.1)
2. For each episode:
   - Start from initial state
   - For each step:
     - Select action using ε-greedy policy
     - Execute action, observe reward and next state
     - Update Q-value using Bellman equation
     - Move to next state
   - Decay exploration rate

**Convergence**: Q-values converge to optimal Q*(s,a) under conditions:
- All state-action pairs visited infinitely often
- Learning rate satisfies: Σα_t = ∞ and Σα_t² < ∞
- Exploration schedule ensures sufficient exploration

### 2. State Discretization

**Problem**: Continuous features need to be converted to discrete states for tabular Q-learning.

**Methods**:

1. **Binning (Fixed Intervals)**:
   - URL length: [0-100, 100-200, >200]
   - Special chars: [0-5, 6-10, >10]
   - Simple and interpretable

2. **Quantization (Quantiles)**:
   - Uses data distribution (4 quantiles: 0-25%, 25-50%, 50-75%, 75-100%)
   - Adapts to data distribution
   - Better for skewed distributions

**State Encoding**: Combines multiple discrete features into single integer using base conversion.

### 3. Reward Shaping

**Design Principle**: Prioritize security (high recall) while maintaining reasonable precision.

**Reward Structure**:
- **FN = -8.0** (highest penalty): Missing attacks is worst
- **FP = -5.0** (moderate penalty): False alarms are bad but acceptable
- **TP = +2.5**: Reward correct detection
- **TN = +0.5**: Small reward for allowing legitimate traffic

**Effect**: Agent learns to be conservative (block suspicious requests) while not being too aggressive.

### 4. ε-Greedy Exploration

**Strategy**: Balance exploration vs exploitation

- **Exploration** (with probability ε): Try random actions to discover better policies
- **Exploitation** (with probability 1-ε): Use best known action (greedy)

**Decay Schedule**: ε starts at 1.0 (full exploration), decays to 0.01 (mostly exploitation)

**Why it works**: Early exploration helps discover good policies; later exploitation refines them.

### 5. Feature Engineering

**Categories**:

1. **Basic Features**: Simple counts (length, special chars, digits, parameters)
2. **Advanced Features**: Derived metrics (entropy, ratios, depth)
3. **Security Features**: Attack pattern detection (SQL, XSS, keywords, encoding)

**Why important**: Good features enable the agent to learn meaningful patterns. Bad features lead to poor performance.

---

## Key Design Decisions

1. **Tabular Q-Learning**: Chosen for interpretability and sufficient for discrete state space. Would need function approximation (DQN) for larger/continuous states.

2. **Discrete States**: Necessary for tabular methods. Trade-off between granularity (more states = better representation) and computational cost.

3. **Reward Structure**: Security-focused design prioritizes recall over precision. Adjustable for different security requirements.

4. **Feature Selection**: 20+ features capture URL characteristics. Balance between information content and computational efficiency.

5. **Normalization**: Helps discretization work better by ensuring features are on similar scales.

6. **Class Balancing**: Important for fair training when dataset is imbalanced.

---

## How to Use This Codebase

### For Understanding:
1. Start with `main.py` to see the overall flow
2. Read `data_preparation.py` to understand data processing
3. Read `environment.py` to understand the MDP formulation
4. Read `q_learning.py` to understand the learning algorithm
5. Read `evaluation.py` to understand metrics and comparison

### For Modification:
- **Change reward structure**: Modify `environment.py` reward values
- **Add features**: Extend `extract_features()` in `data_preparation.py`
- **Change discretization**: Modify `fit_discretization()` method
- **Adjust hyperparameters**: Modify agent initialization in `main.py`
- **Change baseline**: Modify `baseline.py` rules

### For Extension:
- **Deep Q-Network (DQN)**: Replace tabular Q-table with neural network
- **Different RL algorithms**: Implement SARSA, Actor-Critic, etc.
- **More features**: Add domain-specific features
- **Different datasets**: Adapt `load_data()` for other datasets

---

## Summary

This codebase implements a complete RL solution for URL security classification:

1. **Data Preprocessing**: Professional pipeline with feature extraction, normalization, discretization
2. **RL Environment**: Well-defined MDP with security-focused rewards
3. **Q-Learning**: Tabular implementation with ε-greedy exploration
4. **Training**: Episodic training with decay schedules
5. **Evaluation**: Comprehensive metrics and visualizations
6. **Comparison**: Baseline for performance comparison
7. **Reporting**: Automated report generation

The system demonstrates that RL can learn effective security policies that balance attack detection with false positive rates, outperforming simple rule-based approaches.

---

**End of Codebase Explanation**

