"""
Q-Learning Algorithm Module
Implements tabular Q-learning for URL security classification.
"""

import numpy as np
import pickle


class QLearningAgent:
    """
    Tabular Q-learning agent for URL security classification.
    
    Implements the Bellman update:
    Q(s_t, a_t) ← Q(s_t, a_t) + α [ r_t + γ max_a Q(s_{t+1}, a) − Q(s_t, a_t) ]
    
    Hyperparameters:
    - α (alpha): Learning rate = 0.1
      Justification: Small learning rate ensures stable convergence without overshooting
    - γ (gamma): Discount factor = 0.90
      Justification: High discount factor values future rewards, important for sequential decisions
    - ε_start: Initial exploration rate = 1.0
      Justification: Start with full exploration to learn the state space
    - ε_min: Minimum exploration rate = 0.05
      Justification: Maintain some exploration even after convergence
    - ε_decay: Exploration decay per episode = 0.98
      Justification: Gradually shift from exploration to exploitation
    """
    
    def __init__(self, num_states, num_actions=2, alpha=0.1, gamma=0.90, 
                 epsilon_start=1.0, epsilon_min=0.05, epsilon_decay=0.98):
        """
        Initialize the Q-learning agent.
        
        Args:
            num_states: Total number of discrete states
            num_actions: Number of actions (default: 2 for ALLOW/BLOCK)
            alpha: Learning rate (default: 0.1)
            gamma: Discount factor (default: 0.90)
            epsilon_start: Initial exploration rate (default: 1.0)
            epsilon_min: Minimum exploration rate (default: 0.05)
            epsilon_decay: Exploration decay per episode (default: 0.98)
        """
        self.num_states = num_states
        self.num_actions = num_actions
        
        # Hyperparameters
        self.alpha = alpha  # Learning rate
        self.alpha_start = alpha  # Initial learning rate (for decay)
        self.gamma = gamma  # Discount factor
        self.epsilon = epsilon_start  # Current exploration rate
        self.epsilon_start = epsilon_start
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.alpha_decay = 0.9995  # Learning rate decay factor per episode
        
        # Q-table: [num_states × num_actions]
        # Optimistic initialization: small positive values encourage exploration
        # This helps the agent try all actions before converging
        self.q_table = np.ones((num_states, num_actions)) * 0.1
        
        # Training statistics
        self.training_rewards = []
        self.training_episodes = []
        
    def select_action(self, state, training=True):
        """
        Select an action using ε-greedy policy.
        
        Args:
            state: Current discrete state
            training: If True, use ε-greedy; if False, use greedy (exploitation only)
            
        Returns:
            Selected action (0 = ALLOW, 1 = BLOCK)
        """
        if training and np.random.random() < self.epsilon:
            # Exploration: random action
            return np.random.randint(0, self.num_actions)
        else:
            # Exploitation: greedy action (action with highest Q-value)
            return np.argmax(self.q_table[state])
    
    def update(self, state, action, reward, next_state, done):
        """
        Update Q-table using Bellman equation.
        
        Q(s_t, a_t) ← Q(s_t, a_t) + α [ r_t + γ max_a Q(s_{t+1}, a) − Q(s_t, a_t) ]
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state (None if done)
            done: Whether episode is finished
        """
        # Current Q-value
        current_q = self.q_table[state, action]
        
        if done:
            # Terminal state: no future rewards
            target_q = reward
        else:
            # Non-terminal: include discounted future reward
            max_next_q = np.max(self.q_table[next_state])
            target_q = reward + self.gamma * max_next_q
        
        # Bellman update
        self.q_table[state, action] = current_q + self.alpha * (target_q - current_q)
    
    def decay_epsilon(self):
        """
        Decay exploration rate after each episode.
        """
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
    
    def decay_alpha(self):
        """
        Decay learning rate after each episode for more stable convergence.
        """
        self.alpha = max(0.01, self.alpha * self.alpha_decay)
    
    def get_q_value(self, state, action):
        """
        Get Q-value for a state-action pair.
        
        Args:
            state: Discrete state
            action: Action
            
        Returns:
            Q-value
        """
        return self.q_table[state, action]
    
    def get_policy(self, state):
        """
        Get the greedy policy (best action) for a state.
        
        Args:
            state: Discrete state
            
        Returns:
            Best action according to current Q-table
        """
        return np.argmax(self.q_table[state])
    
    def save_q_table(self, filepath='q_table.npy'):
        """
        Save Q-table to file.
        
        Args:
            filepath: Path to save the Q-table
        """
        np.save(filepath, self.q_table)
        print(f"Q-table saved to {filepath}")
    
    def load_q_table(self, filepath='q_table.npy'):
        """
        Load Q-table from file.
        
        Args:
            filepath: Path to load the Q-table from
        """
        self.q_table = np.load(filepath)
        print(f"Q-table loaded from {filepath}")
    
    def save_agent(self, filepath='q_learning_agent.pkl'):
        """
        Save entire agent (Q-table and hyperparameters) to file.
        
        Args:
            filepath: Path to save the agent
        """
        agent_data = {
            'q_table': self.q_table,
            'num_states': self.num_states,
            'num_actions': self.num_actions,
            'alpha': self.alpha,
            'gamma': self.gamma,
            'epsilon': self.epsilon,
            'epsilon_start': self.epsilon_start,
            'epsilon_min': self.epsilon_min,
            'epsilon_decay': self.epsilon_decay
        }
        with open(filepath, 'wb') as f:
            pickle.dump(agent_data, f)
        print(f"Agent saved to {filepath}")
    
    def load_agent(self, filepath='q_learning_agent.pkl'):
        """
        Load entire agent from file.
        
        Args:
            filepath: Path to load the agent from
        """
        with open(filepath, 'rb') as f:
            agent_data = pickle.load(f)
        
        self.q_table = agent_data['q_table']
        self.num_states = agent_data['num_states']
        self.num_actions = agent_data['num_actions']
        self.alpha = agent_data['alpha']
        self.gamma = agent_data['gamma']
        self.epsilon = agent_data['epsilon']
        self.epsilon_start = agent_data['epsilon_start']
        self.epsilon_min = agent_data['epsilon_min']
        self.epsilon_decay = agent_data['epsilon_decay']
        
        print(f"Agent loaded from {filepath}")

