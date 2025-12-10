"""
RL Environment Module
Implements a custom MDP environment for URL-based attack detection.
"""

import numpy as np


class URLSecurityEnvironment:
    """
    Custom MDP environment for URL security classification.
    
    State: Discrete integer representing discretized URL features
    Actions: 0 = ALLOW, 1 = BLOCK
    Rewards:
        TP (attack & BLOCK): +2.0
        TN (benign & ALLOW): +0.5
        FP (benign & BLOCK): -5.0
        FN (attack & ALLOW): -8.0
    """
    
    def __init__(self, data, data_preparator, episode_length=1000):
        """
        Initialize the environment.
        
        Args:
            data: DataFrame with features and labels
            data_preparator: DataPreparator instance with fitted discretization
            episode_length: Number of steps per episode (default: 1000)
        """
        self.data = data.reset_index(drop=True)
        self.data_preparator = data_preparator
        self.episode_length = episode_length
        
        # Reward structure (adjusted for better precision/recall balance)
        self.reward_tp = 2.5   # True Positive: attack & BLOCK (increased)
        self.reward_tn = 0.5   # True Negative: benign & ALLOW
        self.reward_fp = -5.0  # False Positive: benign & BLOCK (less penalty to reduce over-blocking)
        self.reward_fn = -8.0  # False Negative: attack & ALLOW (keep high penalty)
        
        # Environment state
        self.current_step = 0
        self.current_index = 0
        self.current_state = None
        self.current_label = None
        self.episode_rewards = []
        
    def reset(self):
        """
        Reset the environment to start a new episode.
        
        Returns:
            Initial state (discrete integer)
        """
        # Reset episode counters
        self.current_step = 0
        self.episode_rewards = []
        
        # Sample a random starting point in the dataset
        self.current_index = np.random.randint(0, len(self.data))
        
        # Get initial state
        features_row = self.data.iloc[self.current_index]
        self.current_state = self.data_preparator.encode_state(features_row)
        self.current_label = features_row['label']
        
        return self.current_state
    
    def step(self, action):
        """
        Execute one step in the environment.
        
        Args:
            action: 0 = ALLOW, 1 = BLOCK
            
        Returns:
            Tuple of (next_state, reward, done, info)
            - next_state: Next discrete state
            - reward: Reward for this step
            - done: Whether episode is finished
            - info: Dictionary with additional information
        """
        # Get true label (0 = benign, 1 = attack)
        true_label = self.current_label
        
        # Calculate reward based on action and true label
        if action == 1:  # BLOCK
            if true_label == 1:  # Attack
                reward = self.reward_tp  # True Positive
                prediction = 1
            else:  # Benign
                reward = self.reward_fp  # False Positive
                prediction = 1
        else:  # ALLOW (action == 0)
            if true_label == 0:  # Benign
                reward = self.reward_tn  # True Negative
                prediction = 0
            else:  # Attack
                reward = self.reward_fn  # False Negative
                prediction = 0
        
        # Store reward for episode statistics
        self.episode_rewards.append(reward)
        
        # Move to next step
        self.current_step += 1
        
        # Check if episode is done
        done = (self.current_step >= self.episode_length)
        
        # Get next state if not done
        if not done:
            # Move to next sample (wrap around if needed)
            self.current_index = (self.current_index + 1) % len(self.data)
            features_row = self.data.iloc[self.current_index]
            self.current_state = self.data_preparator.encode_state(features_row)
            self.current_label = features_row['label']
        else:
            # Episode finished, state doesn't matter
            self.current_state = None
        
        # Prepare info dictionary
        info = {
            'true_label': true_label,
            'prediction': prediction,
            'correct': (true_label == prediction),
            'step': self.current_step
        }
        
        return self.current_state, reward, done, info
    
    def get_state(self):
        """
        Get the current state.
        
        Returns:
            Current discrete state (integer)
        """
        return self.current_state
    
    def is_done(self):
        """
        Check if the episode is done.
        
        Returns:
            True if episode is finished, False otherwise
        """
        return self.current_step >= self.episode_length
    
    def get_episode_reward(self):
        """
        Get the total reward for the current episode.
        
        Returns:
            Sum of all rewards in the current episode
        """
        return sum(self.episode_rewards) if self.episode_rewards else 0.0
    
    def get_episode_average_reward(self):
        """
        Get the average reward per step for the current episode.
        
        Returns:
            Average reward per step
        """
        if not self.episode_rewards:
            return 0.0
        return sum(self.episode_rewards) / len(self.episode_rewards)

