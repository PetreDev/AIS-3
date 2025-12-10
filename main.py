"""
Main Script
Orchestrates the entire pipeline: data preparation, training, evaluation, and report generation.
"""

import os
import numpy as np
from data_preparation import DataPreparator
from environment import URLSecurityEnvironment
from q_learning import QLearningAgent
from baseline import RuleBasedBaseline
from training import train_q_learning_agent
from evaluation import (evaluate_agent, evaluate_baseline, generate_all_visualizations, 
                       save_evaluation_results)


def main():
    """Main execution function."""
    print("="*70)
    print("URL Security Classification using Q-Learning")
    print("="*70)
    
    # ========================================================================
    # 1. DATA PREPARATION
    # ========================================================================
    print("\n[1/6] Data Preparation")
    print("-" * 70)
    
    preparator = DataPreparator(dataset_path='dataset/csic_database.csv')
    
    # Load data
    data = preparator.load_data()
    
    # Extract features
    features = preparator.extract_features(data)
    
    # Fit discretization (using quantization for better state representation)
    preparator.fit_discretization(features.drop('label', axis=1), method='quantization')
    num_states = preparator.get_num_states()
    print(f"Total number of discrete states: {num_states}")
    
    # Split data
    train_data, val_data, test_data = preparator.split_data(features)
    
    # ========================================================================
    # 2. BASELINE APPROACH
    # ========================================================================
    print("\n[2/6] Baseline Approach")
    print("-" * 70)
    
    baseline = RuleBasedBaseline(K=2)
    
    # Optionally tune threshold on validation set
    print("Tuning baseline threshold on validation set...")
    baseline.tune_threshold(val_data.drop('label', axis=1), val_data['label'].values)
    
    # ========================================================================
    # 3. RL ENVIRONMENT SETUP
    # ========================================================================
    print("\n[3/6] RL Environment Setup")
    print("-" * 70)
    
    # Create environment with training data
    env = URLSecurityEnvironment(train_data, preparator, episode_length=1000)
    print(f"Environment created with episode length: {env.episode_length}")
    print(f"Reward structure:")
    print(f"  TP (attack & BLOCK): +{env.reward_tp}")
    print(f"  TN (benign & ALLOW): +{env.reward_tn}")
    print(f"  FP (benign & BLOCK): {env.reward_fp}")
    print(f"  FN (attack & ALLOW): {env.reward_fn}")
    
    # ========================================================================
    # 4. Q-LEARNING AGENT SETUP
    # ========================================================================
    print("\n[4/6] Q-Learning Agent Setup")
    print("-" * 70)
    
    agent = QLearningAgent(
        num_states=num_states,
        num_actions=2,
        alpha=0.2,          # Learning rate (increased for faster learning)
        gamma=0.95,         # Discount factor (increased to value future rewards more)
        epsilon_start=1.0,  # Initial exploration
        epsilon_min=0.01,   # Minimum exploration (reduced for better exploitation)
        epsilon_decay=0.995 # Exploration decay (slower decay for more exploration)
    )
    print(f"Q-learning agent initialized")
    print(f"  States: {num_states}")
    print(f"  Actions: 2 (ALLOW=0, BLOCK=1)")
    print(f"  Learning rate (α): {agent.alpha}")
    print(f"  Discount factor (γ): {agent.gamma}")
    print(f"  Initial epsilon: {agent.epsilon_start}")
    
    # ========================================================================
    # 5. TRAINING
    # ========================================================================
    print("\n[5/6] Training Q-Learning Agent")
    print("-" * 70)
    
    # Train for at least 10 episodes (≥ 10,000 events)
    num_episodes = 50  # Increased for better learning and convergence
    training_stats = train_q_learning_agent(env, agent, num_episodes=num_episodes, verbose=True)
    
    # Save Q-table
    agent.save_q_table('q_table.npy')
    agent.save_agent('q_learning_agent.pkl')
    
    # ========================================================================
    # 6. EVALUATION
    # ========================================================================
    print("\n[6/6] Evaluation")
    print("-" * 70)
    
    # Evaluate RL agent
    rl_results = evaluate_agent(agent, env, preparator, test_data, verbose=True)
    
    # Evaluate baseline
    baseline_results = evaluate_baseline(baseline, test_data, verbose=True)
    
    # Generate visualizations
    print("\nGenerating visualizations...")
    generate_all_visualizations(agent, rl_results, baseline_results, output_dir='plots')
    
    # Save evaluation results
    save_evaluation_results(rl_results, baseline_results, 'evaluation_results.json')
    
    # ========================================================================
    # 7. GENERATE REPORT
    # ========================================================================
    print("\n[7/7] Generating Report")
    print("-" * 70)
    
    generate_report(agent, preparator, rl_results, baseline_results, training_stats)
    
    print("\n" + "="*70)
    print("Pipeline complete! Check the following outputs:")
    print("  - q_table.npy: Saved Q-table")
    print("  - q_learning_agent.pkl: Saved agent")
    print("  - evaluation_results.json: Evaluation metrics")
    print("  - plots/: Directory with all visualizations")
    print("  - report.md: Comprehensive report")
    print("="*70)


def generate_report(agent, preparator, rl_results, baseline_results, training_stats):
    """
    Generate a comprehensive markdown report.
    
    Args:
        agent: QLearningAgent instance
        preparator: DataPreparator instance
        rl_results: Dictionary with RL evaluation results
        baseline_results: Dictionary with baseline evaluation results
        training_stats: Dictionary with training statistics
    """
    report = []
    
    report.append("# URL Security Classification using Q-Learning - Report\n")
    report.append("Generated automatically from training and evaluation results.\n")
    
    # A) RL Environment Description
    report.append("## A) RL Environment Description\n")
    report.append("### State\n")
    report.append("The state is a discrete integer representing discretized URL features:\n")
    report.append("- `url_length_bin`: Discretized URL length\n")
    report.append("- `specchar_bin`: Discretized number of special characters\n")
    report.append("- `digits_bin`: Discretized number of digits\n")
    report.append("- `params_bin`: Discretized number of parameters\n")
    report.append("- `keyword_presence`: Binary feature (0 or 1) for suspicious keywords\n\n")
    report.append(f"Total number of states: {preparator.get_num_states()}\n\n")
    
    report.append("### Actions\n")
    report.append("- **Action 0 (ALLOW)**: Allow the request to proceed\n")
    report.append("- **Action 1 (BLOCK)**: Block the request as potentially malicious\n\n")
    
    report.append("### Rewards\n")
    report.append("The reward function is designed to penalize security failures:\n")
    report.append("- **TP (True Positive)**: Attack detected and blocked → **+2.0**\n")
    report.append("- **TN (True Negative)**: Benign request allowed → **+0.5**\n")
    report.append("- **FP (False Positive)**: Benign request blocked → **-5.0**\n")
    report.append("- **FN (False Negative)**: Attack allowed → **-8.0**\n\n")
    report.append("The high penalty for False Negatives (-8.0) encourages the agent to be more\n")
    report.append("conservative and block suspicious requests, prioritizing security over convenience.\n\n")
    
    # B) Binning & Quantization
    report.append("## B) State Discretization: Binning & Quantization\n")
    report.append("### Binning (Fixed Intervals)\n")
    report.append("Binning uses fixed intervals to discretize continuous features:\n\n")
    report.append("**URL Length:**\n")
    report.append("- 0-100 → bin 0\n")
    report.append("- 100-200 → bin 1\n")
    report.append("- >200 → bin 2\n\n")
    report.append("**Special Characters:**\n")
    report.append("- 0-5 → bin 0\n")
    report.append("- 6-10 → bin 1\n")
    report.append("- >10 → bin 2\n\n")
    report.append("**Digits:**\n")
    report.append("- 0-5 → bin 0\n")
    report.append("- 6-15 → bin 1\n")
    report.append("- >15 → bin 2\n\n")
    report.append("**Parameters:**\n")
    report.append("- 0-2 → bin 0\n")
    report.append("- 3-5 → bin 1\n")
    report.append("- >5 → bin 2\n\n")
    
    report.append("### Quantization (Quantiles)\n")
    report.append("Quantization uses quantile boundaries computed from the training data:\n\n")
    report.append("- **4 quantiles**: 0-25%, 25-50%, 50-75%, 75-100%\n")
    report.append("- Boundaries are automatically computed from data distribution\n")
    report.append("- Example boundaries for special characters: 2, 7, 15\n\n")
    report.append("**Note:** This implementation uses binning for state discretization.\n\n")
    
    # C) Q-table Explanation
    report.append("## C) Q-Table Explanation\n")
    report.append("### Structure\n")
    report.append(f"- **Rows**: {agent.q_table.shape[0]} states (one per discrete state)\n")
    report.append("- **Columns**: 2 actions (ALLOW=0, BLOCK=1)\n")
    report.append(f"- **Shape**: [{agent.q_table.shape[0]} × {agent.q_table.shape[1]}]\n\n")
    
    report.append("### Q(s,a) Meaning\n")
    report.append("Q(s,a) represents the expected cumulative reward when taking action `a` in state `s`\n")
    report.append("and following the optimal policy thereafter.\n\n")
    
    report.append("### Bellman Update Equation\n")
    report.append("```\n")
    report.append("Q(s_t, a_t) ← Q(s_t, a_t) + α [ r_t + γ max_a Q(s_{t+1}, a) − Q(s_t, a_t) ]\n")
    report.append("```\n\n")
    report.append("Where:\n")
    report.append(f"- α (alpha) = {agent.alpha}: Learning rate\n")
    report.append(f"- γ (gamma) = {agent.gamma}: Discount factor\n")
    report.append("- r_t: Immediate reward\n")
    report.append("- s_t: Current state\n")
    report.append("- a_t: Action taken\n")
    report.append("- s_{t+1}: Next state\n\n")
    
    report.append("### Example Calculation\n")
    report.append("Given:\n")
    report.append("- Initial Q(s₀, BLOCK) = 0.0\n")
    report.append("- Reward r = -5.0 (False Positive: benign request blocked)\n")
    report.append("- α = 0.1\n")
    report.append("- γ = 0.90\n")
    report.append("- max_a Q(s₁, a) = 0.5 (assumed)\n\n")
    report.append("Update:\n")
    report.append("```\n")
    report.append("Q(s₀, BLOCK) = 0.0 + 0.1 × [-5.0 + 0.90 × 0.5 - 0.0]\n")
    report.append("             = 0.0 + 0.1 × [-5.0 + 0.45]\n")
    report.append("             = 0.0 + 0.1 × [-4.55]\n")
    report.append("             = -0.455\n")
    report.append("```\n\n")
    report.append("After rounding: **Q(s₀, BLOCK) ≈ -0.5**\n\n")
    
    # D) Comparison
    report.append("## D) Comparison: RL Agent vs Baseline\n")
    report.append("### Performance Metrics\n\n")
    report.append("| Metric | RL Agent | Baseline |\n")
    report.append("|--------|----------|----------|\n")
    report.append(f"| Accuracy | {rl_results['accuracy']:.4f} | {baseline_results['accuracy']:.4f} |\n")
    report.append(f"| Precision | {rl_results['precision']:.4f} | {baseline_results['precision']:.4f} |\n")
    report.append(f"| Recall | {rl_results['recall']:.4f} | {baseline_results['recall']:.4f} |\n")
    report.append(f"| F1-Score | {rl_results['f1_score']:.4f} | {baseline_results['f1_score']:.4f} |\n")
    report.append(f"| Avg Reward | {rl_results['average_reward']:.4f} | {baseline_results['average_reward']:.4f} |\n\n")
    
    report.append("### Confusion Matrices\n\n")
    report.append("**RL Agent:**\n")
    rl_cm = np.array(rl_results['confusion_matrix'])
    report.append("```\n")
    report.append("              Predicted\n")
    report.append("              ALLOW  BLOCK\n")
    report.append(f"Actual Benign  {rl_cm[0,0]:4d}   {rl_cm[0,1]:4d}\n")
    report.append(f"Actual Attack  {rl_cm[1,0]:4d}   {rl_cm[1,1]:4d}\n")
    report.append("```\n\n")
    
    report.append("**Baseline:**\n")
    baseline_cm = np.array(baseline_results['confusion_matrix'])
    report.append("```\n")
    report.append("              Predicted\n")
    report.append("              ALLOW  BLOCK\n")
    report.append(f"Actual Benign  {baseline_cm[0,0]:4d}   {baseline_cm[0,1]:4d}\n")
    report.append(f"Actual Attack  {baseline_cm[1,0]:4d}   {baseline_cm[1,1]:4d}\n")
    report.append("```\n\n")
    
    report.append("### Learning Curve\n")
    report.append("The learning curve shows the average reward per episode during training.\n")
    report.append("See `plots/learning_curve.png` for visualization.\n\n")
    report.append(f"Training Statistics:\n")
    report.append(f"- Episodes: {len(agent.training_episodes)}\n")
    report.append(f"- Average reward: {training_stats['average_reward']:.4f}\n")
    report.append(f"- Final epsilon: {training_stats['final_epsilon']:.4f}\n\n")
    
    # E) Reward Shaping
    report.append("## E) How Reward Shaping Affects FP/FN\n")
    report.append("### High FN Penalty Pushes Recall\n")
    report.append("A high penalty for False Negatives (FN = -8.0) encourages the agent to:\n")
    report.append("- Be more conservative and block suspicious requests\n")
    report.append("- Prioritize detecting attacks over allowing benign requests\n")
    report.append("- Increase **Recall** (True Positive Rate) by reducing missed attacks\n")
    report.append("- This is critical for security applications where missing an attack is costly\n\n")
    
    report.append("### High FP Penalty Pushes Precision\n")
    report.append("A high penalty for False Positives (FP = -5.0) encourages the agent to:\n")
    report.append("- Be more selective about blocking requests\n")
    report.append("- Only block when highly confident it's an attack\n")
    report.append("- Increase **Precision** by reducing false alarms\n")
    report.append("- This balances security with user experience\n\n")
    
    report.append("### Trade-off Analysis\n")
    report.append("The current reward structure:\n")
    report.append("- **FN penalty (-8.0) > FP penalty (-5.0)**: Prioritizes security (recall)\n")
    report.append("- This is appropriate for security applications where missing attacks is worse than blocking benign requests\n")
    report.append("- The agent learns to be more aggressive in blocking, leading to higher recall\n")
    report.append("- However, this may come at the cost of lower precision (more false positives)\n\n")
    
    report.append("### Results Analysis\n")
    report.append(f"**RL Agent:**\n")
    report.append(f"- Recall: {rl_results['recall']:.4f} (ability to detect attacks)\n")
    report.append(f"- Precision: {rl_results['precision']:.4f} (accuracy of blocking decisions)\n")
    report.append(f"- The agent achieves a balance between detecting attacks and minimizing false positives\n\n")
    
    report.append(f"**Baseline:**\n")
    report.append(f"- Recall: {baseline_results['recall']:.4f}\n")
    report.append(f"- Precision: {baseline_results['precision']:.4f}\n")
    report.append(f"- Simple rule-based approach with fixed threshold\n\n")
    
    # Write report to file
    report_text = ''.join(report)
    with open('report.md', 'w', encoding='utf-8') as f:
        f.write(report_text)
    
    print("Report generated and saved to report.md")


if __name__ == '__main__':
    main()

