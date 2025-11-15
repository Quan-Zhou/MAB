import numpy as np
from collections import defaultdict

class Discretizer:
    def __init__(self, bins):
        """
        bins = [b_x, b_xdot, b_theta, b_thetadot]
        """
        self.bins = np.array(bins)
        self.low = np.array([-4.8, -3.0, -0.418, -3.5])
        self.high = np.array([4.8, 3.0, 0.418, 3.5])
        self.width = (self.high - self.low) / self.bins

    def discretize(self, state):
        ratios = (state - self.low) / self.width
        indices = np.floor(ratios).astype(int)
        indices = np.clip(indices, 0, self.bins - 1)
        return tuple(indices)
    
    def get_all_discrete_states(self):
        """Generator that yields all possible discrete states"""
        ranges = [range(bin_size) for bin_size in self.bins]
        from itertools import product
        for state_tuple in product(*ranges):
            yield state_tuple

class TabularQLearningAgent:
    def __init__(self, bins, num_actions=10, lr=0.1, gamma=0.99, epsilon=0.1, force_mag=10.0):
        self.disc = Discretizer(bins)
        self.num_actions = num_actions
        self.discrete_actions = np.linspace(-force_mag, force_mag, self.num_actions)
        
        print(f"Continuous action space discretized into {self.num_actions} actions:")
        print(f"Discrete actions: {self.discrete_actions}")

        self.Q = np.zeros((*bins, num_actions))
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon

    def choose_action(self, state):
        s = self.disc.discretize(state)
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.num_actions)
        return np.argmax(self.Q[s])

    def update(self, state, action, reward, next_state, done):
        s = self.disc.discretize(state)
        ns = self.disc.discretize(next_state)

        current_q = self.Q[s][action]
        if done:
            target = reward
        else:
            target = reward + self.gamma * np.max(self.Q[ns])
        
        self.Q[s][action] += self.lr * (target - current_q)

    def decrease_epsilon(self, factor=0.99, min_epsilon=0.01):
        """Gradually decrease exploration rate"""
        self.epsilon = max(min_epsilon, self.epsilon * factor)

    def get_policy_dict(self):
        """
        Returns the policy as a dictionary mapping discrete states to actions
        
        Returns:
            dict: {state_tuple: action_index} where action_index corresponds 
                  to self.discrete_actions[action_index]
        """
        policy = {}
        
        # Iterate through all possible discrete states
        for state_tuple in self.disc.get_all_discrete_states():
            # Get the best action for this state
            best_action = np.argmax(self.Q[state_tuple])
            policy[state_tuple] = best_action
        
        return policy

    def get_policy_with_continuous_actions(self):
        """
        Returns the policy as a dictionary mapping discrete states to continuous actions
        
        Returns:
            dict: {state_tuple: continuous_action_value}
        """
        policy = {}
        
        for state_tuple in self.disc.get_all_discrete_states():
            best_action_idx = np.argmax(self.Q[state_tuple])
            continuous_action = self.discrete_actions[best_action_idx]
            policy[state_tuple] = continuous_action
        
        return policy

    def get_detailed_policy(self):
        """
        Returns a detailed policy with Q-values and action information
        
        Returns:
            dict: {state_tuple: {'action_index': int, 
                                'continuous_action': float,
                                'q_value': float,
                                'all_q_values': list}}
        """
        detailed_policy = {}
        
        for state_tuple in self.disc.get_all_discrete_states():
            best_action_idx = np.argmax(self.Q[state_tuple])
            best_q_value = self.Q[state_tuple][best_action_idx]
            continuous_action = self.discrete_actions[best_action_idx]
            all_q_values = self.Q[state_tuple].tolist()
            
            detailed_policy[state_tuple] = {
                'action_index': best_action_idx,
                'continuous_action': continuous_action,
                'q_value': best_q_value,
                'all_q_values': all_q_values
            }
        
        return detailed_policy

    def print_policy_sample(self, num_samples=10):
        """Print a sample of the policy for inspection"""
        policy = self.get_policy_with_continuous_actions()
        states = list(policy.keys())[:num_samples]
        
        print(f"\nPolicy Sample (first {num_samples} states):")
        print("=" * 60)
        for i, state in enumerate(states):
            action = policy[state]
            print(f"State {state} -> Action: {action:.3f}")
        
        print(f"\nTotal policy size: {len(policy)} states")
        
        # Also show action distribution
        all_actions = list(policy.values())
        unique_actions = set(all_actions)
        print(f"Unique continuous actions used: {len(unique_actions)}")
        
        # Show most common actions
        from collections import Counter
        action_counts = Counter(all_actions)
        print("\nMost common actions:")
        for action, count in action_counts.most_common(5):
            print(f"  Action {action:.3f}: {count} states")
