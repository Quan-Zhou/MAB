import numpy as np

class Discretizer:
    def __init__(self, bins):
        """
        bins = [b_x, b_xdot, b_theta, b_thetadot]
        """
        self.bins = np.array(bins)

        # Approx CartPole observation bounds
        self.low = np.array([-4.8, -3.0, -0.418, -3.5])
        self.high = np.array([4.8, 3.0, 0.418, 3.5])

        self.width = (self.high - self.low) / self.bins

    def discretize(self, state):
        ratios = (state - self.low) / self.width
        indices = np.floor(ratios).astype(int)
        indices = np.clip(indices, 0, self.bins - 1)
        return tuple(indices)

class TabularQLearningAgent:
    def __init__(self, bins, action_dim=2, lr=0.1, gamma=0.99, epsilon=0.1):
        self.disc = Discretizer(bins)
        self.action_dim = action_dim

        # Create Q-table with shape: (b1, b2, b3, b4, action_dim)
        self.Q = np.zeros((*bins, action_dim))

        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon

    def choose_action(self, state):
        s = self.disc.discretize(state)
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.action_dim)
        return np.argmax(self.Q[s])

    def update(self, state, action, reward, next_state, done):
        s = self.disc.discretize(state)
        ns = self.disc.discretize(next_state)

        target = reward + self.gamma * np.max(self.Q[ns]) * (1 - done)
        self.Q[s][action] += self.lr * (target - self.Q[s][action])
