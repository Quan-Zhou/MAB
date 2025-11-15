import numpy as np
from gymnasium.envs.classic_control.cartpole import CartPoleEnv
from gymnasium import spaces


class ContinuousCartPoleEnv(CartPoleEnv):
    """CartPole with continuous action space."""

    def __init__(self, **custom_params):
        super().__init__()  # cannot accept custom args

        # Overwrite internal parameters after init
        for k, v in custom_params.items():
            setattr(self, k, v)

        # Continuous action space
        self.action_space = spaces.Box(
            low=np.array([-self.force_mag], dtype=np.float32),
            high=np.array([self.force_mag], dtype=np.float32),
            dtype=np.float32,
        )

        # Recompute dependent quantities
        self.total_mass = self.masscart + self.masspole
        self.polemass_length = self.masspole * self.length

    def step(self, action):
        # Allow both discrete (0/1) and continuous actions
        # if np.isscalar(action):  
        #     # Discrete action → convert to continuous force
        #     force = self.force_mag if action == 1 else -self.force_mag
        # else:
        # Continuous action
        force = float(np.clip(action, -self.force_mag, self.force_mag))

        self.last_u = force

        # Same as original CartPole step except using continuous force
        x, x_dot, theta, theta_dot = self.state
        costheta = np.cos(theta)
        sintheta = np.sin(theta)

        temp = (force + self.polemass_length * theta_dot**2 * sintheta) / self.total_mass
        thetaacc = (self.gravity * sintheta - costheta * temp) / (
            self.length * (4.0/3.0 - self.masspole * costheta**2 / self.total_mass)
        )
        xacc = temp - self.polemass_length * thetaacc * costheta / self.total_mass

        x = x + self.tau * x_dot
        x_dot = x_dot + self.tau * xacc
        theta = theta + self.tau * theta_dot
        theta_dot = theta_dot + self.tau * thetaacc

        # self.state = (x, x_dot, theta, theta_dot)
        self.state = np.array([x, x_dot, theta, theta_dot], dtype=np.float32)

        terminated = (
            x < -self.x_threshold
            or x > self.x_threshold
            or theta < -self.theta_threshold_radians
            or theta > self.theta_threshold_radians
        )
        reward = 1.0
        return np.array(self.state), reward, terminated, False, {}
    
    def reset(self):
        """
        Reset the environment and return a proper 4-element state array
        """
        result = super().reset()  # This returns (state, info) tuple
        
        # Extract just the state array from the tuple
        if isinstance(result, tuple):
            state = result[0]  # Get the first element (the state array)
        else:
            state = result
        
        # Ensure it's a 4-element numpy array
        if not isinstance(state, np.ndarray) or len(state) != 4:
            state = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        
        self.state = state
        return state

class CartPoleCategoryGenerator:
    def __init__(self):
        self.categories = {
            'easy':     {'gravity': 8.0,  'masspole': 0.08, 'length': 0.4, 'variation': 0.1},
            'medium':   {'gravity': 9.8,  'masspole': 0.1,  'length': 0.5, 'variation': 0.15},
            'hard':     {'gravity': 11.0, 'masspole': 0.15, 'length': 0.6, 'variation': 0.1},
            'very_hard':{'gravity': 12.0, 'masspole': 0.2,  'length': 0.7, 'variation': 0.08},
            'unstable': {'gravity': 10.5, 'masspole': 0.18, 'length': 0.65,'variation': 0.2},
        }

    def generate_env(self, category):
        base = self.categories[category]
        variation = base['variation']

        # Randomize each physical parameter
        params = {}
        for k, v in base.items():
            if k != "variation":
                params[k] = v * np.random.uniform(1 - variation, 1 + variation)

        class RandomContinuousCartPole(ContinuousCartPoleEnv):
            def __init__(self):
                super().__init__(**params)

        return RandomContinuousCartPole()

    def get_available_categories(self):
        return list(self.categories.keys())

    def get_category_info(self, category):
        return self.categories[category]

