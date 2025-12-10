"""
Evaluation Module
Handles evaluation metrics, visualizations, and comparison of RL agent vs baseline.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, confusion_matrix
import json
import os


def evaluate_agent(agent, env, data_preparator, test_data, verbose=True):
    """
    Evaluate the Q-learning agent on test data.
    
    Args:
        agent: QLearningAgent instance
        data_preparator: DataPreparator instance
        test_data: DataFrame with test features and labels
        verbose: Whether to print results
        
    Returns:
        Dictionary with evaluation metrics
    """
    predictions = []
    true_labels = []
    rewards = []
    
    # Reward structure (same as environment - updated for better balance)
    reward_tp = 2.5
    reward_tn = 0.5
    reward_fp = -5.0
    reward_fn = -8.0
    
    for idx, row in test_data.iterrows():
        # Get state
        state = data_preparator.encode_state(row)
        true_label = row['label']
        
        # Get action from agent (greedy policy, no exploration)
        action = agent.select_action(state, training=False)
        
        # Calculate reward
        if action == 1:  # BLOCK
            if true_label == 1:  # Attack
                reward = reward_tp
            else:  # Benign
                reward = reward_fp
        else:  # ALLOW
            if true_label == 0:  # Benign
                reward = reward_tn
            else:  # Attack
                reward = reward_fn
        
        predictions.append(action)
        true_labels.append(true_label)
        rewards.append(reward)
    
    predictions = np.array(predictions)
    true_labels = np.array(true_labels)
    rewards = np.array(rewards)
    
    # Calculate metrics
    accuracy = accuracy_score(true_labels, predictions)
    precision = precision_score(true_labels, predictions, zero_division=0)
    recall = recall_score(true_labels, predictions, zero_division=0)
    f1 = f1_score(true_labels, predictions, zero_division=0)
    avg_reward = np.mean(rewards)
    
    # Confusion matrix
    cm = confusion_matrix(true_labels, predictions)
    
    results = {
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1_score': float(f1),
        'average_reward': float(avg_reward),
        'confusion_matrix': cm.tolist(),
        'predictions': predictions.tolist(),
        'true_labels': true_labels.tolist()
    }
    
    if verbose:
        print("\n" + "="*50)
        print("RL Agent Evaluation Results")
        print("="*50)
        print(f"Accuracy:  {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall:    {recall:.4f}")
        print(f"F1-Score:  {f1:.4f}")
        print(f"Avg Reward: {avg_reward:.4f}")
        print("\nConfusion Matrix:")
        print(f"              Predicted")
        print(f"              ALLOW  BLOCK")
        print(f"Actual Benign  {cm[0,0]:4d}   {cm[0,1]:4d}")
        print(f"Actual Attack  {cm[1,0]:4d}   {cm[1,1]:4d}")
        print("="*50)
    
    return results


def evaluate_baseline(baseline, test_data, verbose=True):
    """
    Evaluate the baseline classifier on test data.
    
    Args:
        baseline: RuleBasedBaseline instance
        test_data: DataFrame with test features and labels
        verbose: Whether to print results
        
    Returns:
        Dictionary with evaluation metrics
    """
    # Get predictions
    predictions = baseline.predict_batch(test_data)
    true_labels = test_data['label'].values
    
    # Reward structure (same as environment - updated for better balance)
    reward_tp = 2.5
    reward_tn = 0.5
    reward_fp = -5.0
    reward_fn = -8.0
    
    # Calculate rewards
    rewards = []
    for pred, true_label in zip(predictions, true_labels):
        if pred == 1:  # BLOCK
            if true_label == 1:  # Attack
                rewards.append(reward_tp)
            else:  # Benign
                rewards.append(reward_fp)
        else:  # ALLOW
            if true_label == 0:  # Benign
                rewards.append(reward_tn)
            else:  # Attack
                rewards.append(reward_fn)
    
    rewards = np.array(rewards)
    
    # Calculate metrics
    accuracy = accuracy_score(true_labels, predictions)
    precision = precision_score(true_labels, predictions, zero_division=0)
    recall = recall_score(true_labels, predictions, zero_division=0)
    f1 = f1_score(true_labels, predictions, zero_division=0)
    avg_reward = np.mean(rewards)
    
    # Confusion matrix
    cm = confusion_matrix(true_labels, predictions)
    
    results = {
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1_score': float(f1),
        'average_reward': float(avg_reward),
        'confusion_matrix': cm.tolist(),
        'predictions': predictions.tolist(),
        'true_labels': true_labels.tolist()
    }
    
    if verbose:
        print("\n" + "="*50)
        print("Baseline Evaluation Results")
        print("="*50)
        print(f"Accuracy:  {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall:    {recall:.4f}")
        print(f"F1-Score:  {f1:.4f}")
        print(f"Avg Reward: {avg_reward:.4f}")
        print("\nConfusion Matrix:")
        print(f"              Predicted")
        print(f"              ALLOW  BLOCK")
        print(f"Actual Benign  {cm[0,0]:4d}   {cm[0,1]:4d}")
        print(f"Actual Attack  {cm[1,0]:4d}   {cm[1,1]:4d}")
        print("="*50)
    
    return results


def plot_learning_curve(agent, save_path='learning_curve.png'):
    """
    Plot the learning curve (average reward per episode).
    
    Args:
        agent: QLearningAgent instance with training_rewards
        save_path: Path to save the plot
    """
    if not agent.training_rewards:
        print("No training rewards available for plotting")
        return
    
    plt.figure(figsize=(10, 6))
    episodes = agent.training_episodes
    rewards = agent.training_rewards
    
    # Plot raw rewards
    plt.plot(episodes, rewards, alpha=0.3, color='blue', label='Episode Reward')
    
    # Plot moving average
    if len(rewards) > 1:
        window = min(10, len(rewards) // 2)
        moving_avg = np.convolve(rewards, np.ones(window)/window, mode='valid')
        moving_avg_episodes = episodes[window-1:]
        plt.plot(moving_avg_episodes, moving_avg, color='red', linewidth=2, 
                label=f'Moving Average (window={window})')
    
    plt.xlabel('Episode')
    plt.ylabel('Total Reward per Episode')
    plt.title('Q-Learning Training: Learning Curve')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Learning curve saved to {save_path}")
    plt.close()


def plot_confusion_matrix(cm, title, save_path):
    """
    Plot a confusion matrix.
    
    Args:
        cm: Confusion matrix (2x2 numpy array)
        title: Title for the plot
        save_path: Path to save the plot
    """
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(title)
    plt.colorbar()
    
    classes = ['ALLOW', 'BLOCK']
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes)
    plt.yticks(tick_marks, classes)
    
    # Add text annotations
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                    horizontalalignment="center",
                    color="white" if cm[i, j] > thresh else "black")
    
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Confusion matrix saved to {save_path}")
    plt.close()


def plot_q_table_heatmap(agent, save_path='q_table_heatmap.png', max_states=100):
    """
    Plot a heatmap of the Q-table (optional visualization).
    
    Args:
        agent: QLearningAgent instance
        save_path: Path to save the plot
        max_states: Maximum number of states to display (for readability)
    """
    q_table = agent.q_table
    
    # Limit number of states for visualization
    if q_table.shape[0] > max_states:
        # Sample states or show first max_states
        q_table_vis = q_table[:max_states, :]
        print(f"Q-table has {q_table.shape[0]} states, showing first {max_states} for visualization")
    else:
        q_table_vis = q_table
    
    plt.figure(figsize=(10, max(6, q_table_vis.shape[0] // 10)))
    plt.imshow(q_table_vis, aspect='auto', cmap='RdYlGn', interpolation='nearest')
    plt.colorbar(label='Q-value')
    plt.xlabel('Action (0=ALLOW, 1=BLOCK)')
    plt.ylabel('State')
    plt.title('Q-Table Heatmap')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Q-table heatmap saved to {save_path}")
    plt.close()


def save_evaluation_results(rl_results, baseline_results, save_path='evaluation_results.json'):
    """
    Save evaluation results to JSON file.
    
    Args:
        rl_results: Dictionary with RL agent evaluation results
        baseline_results: Dictionary with baseline evaluation results
        save_path: Path to save the results
    """
    # Remove predictions and true_labels for JSON (too large)
    rl_results_save = {k: v for k, v in rl_results.items() 
                       if k not in ['predictions', 'true_labels']}
    baseline_results_save = {k: v for k, v in baseline_results.items() 
                            if k not in ['predictions', 'true_labels']}
    
    results = {
        'rl_agent': rl_results_save,
        'baseline': baseline_results_save
    }
    
    with open(save_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Evaluation results saved to {save_path}")


def generate_all_visualizations(agent, rl_results, baseline_results, output_dir='plots'):
    """
    Generate all required visualizations.
    
    Args:
        agent: QLearningAgent instance
        rl_results: Dictionary with RL agent evaluation results
        baseline_results: Dictionary with baseline evaluation results
        output_dir: Directory to save plots
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Learning curve
    plot_learning_curve(agent, os.path.join(output_dir, 'learning_curve.png'))
    
    # Confusion matrices
    rl_cm = np.array(rl_results['confusion_matrix'])
    baseline_cm = np.array(baseline_results['confusion_matrix'])
    
    plot_confusion_matrix(rl_cm, 'RL Agent Confusion Matrix', 
                         os.path.join(output_dir, 'confusion_matrix_rl.png'))
    plot_confusion_matrix(baseline_cm, 'Baseline Confusion Matrix', 
                         os.path.join(output_dir, 'confusion_matrix_baseline.png'))
    
    # Q-table heatmap (optional)
    plot_q_table_heatmap(agent, os.path.join(output_dir, 'q_table_heatmap.png'))

