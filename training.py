"""
Training Module
Orchestrates the training of the Q-learning agent.
"""

import numpy as np
from environment import URLSecurityEnvironment
from q_learning import QLearningAgent


def train_q_learning_agent(env, agent, num_episodes=10, verbose=True):
    """
    Train the Q-learning agent in the environment.
    
    Args:
        env: URLSecurityEnvironment instance
        agent: QLearningAgent instance
        num_episodes: Number of episodes to train (default: 10, minimum required)
        verbose: Whether to print training progress
        
    Returns:
        Dictionary with training statistics
    """
    episode_rewards = []
    episode_lengths = []
    
    if verbose:
        print(f"Starting training for {num_episodes} episodes...")
        print(f"Initial epsilon: {agent.epsilon:.4f}")
    
    for episode in range(num_episodes):
        # Reset environment
        state = env.reset()
        total_reward = 0.0
        steps = 0
        
        # Run episode
        done = False
        while not done:
            # Select action using ε-greedy policy
            action = agent.select_action(state, training=True)
            
            # Take step in environment
            next_state, reward, done, info = env.step(action)
            
            # Update Q-table
            agent.update(state, action, reward, next_state, done)
            
            # Move to next state
            state = next_state
            total_reward += reward
            steps += 1
        
        # Store episode statistics
        episode_rewards.append(total_reward)
        episode_lengths.append(steps)
        
        # Decay exploration rate and learning rate
        agent.decay_epsilon()
        agent.decay_alpha()
        
        # Print progress
        if verbose and (episode + 1) % max(1, num_episodes // 10) == 0:
            avg_reward = np.mean(episode_rewards[-10:]) if len(episode_rewards) >= 10 else total_reward
            print(f"Episode {episode + 1}/{num_episodes} - "
                  f"Reward: {total_reward:.2f}, "
                  f"Avg Reward (last 10): {avg_reward:.2f}, "
                  f"Epsilon: {agent.epsilon:.4f}")
    
    if verbose:
        print(f"\nTraining complete!")
        print(f"Average reward per episode: {np.mean(episode_rewards):.2f}")
        print(f"Final epsilon: {agent.epsilon:.4f}")
    
    # Store training statistics in agent
    agent.training_rewards = episode_rewards
    agent.training_episodes = list(range(1, num_episodes + 1))
    
    return {
        'episode_rewards': episode_rewards,
        'episode_lengths': episode_lengths,
        'average_reward': np.mean(episode_rewards),
        'final_epsilon': agent.epsilon
    }

