import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
import pandas as pd

class GaussianCTS:
    def __init__(self, n_arms, budget, prior_var=1.0, noise_var=1.0):
        """
        n_arms: number of base arms
        budget: number of arms to select per round (top-k)
        prior_var: prior variance of each arm
        noise_var: observation noise variance
        """
        self.n_arms = n_arms
        self.budget = budget
        self.prior_var = prior_var
        self.noise_var = noise_var

        # Posterior parameters
        self.mu = np.zeros(n_arms)
        self.var = np.ones(n_arms) * prior_var

    def step(self, get_rewards):
        """
        Perform one round of Gaussian CTS (top-k oracle)

        get_rewards: function(chosen) -> rewards for chosen arms
        """
        # 1. Sample theta from posterior
        theta = np.random.normal(self.mu, np.sqrt(self.var))

        # 2. Top-k oracle: choose arms with largest sampled values
        chosen = np.argsort(theta)[-self.budget:]

        # 3. Observe rewards from the environment
        rewards = get_rewards[chosen]

        # 4. Update posterior for chosen arms
        for i, r in zip(chosen, rewards):
            prior_prec = 1.0 / self.var[i]
            like_prec = 1.0 / self.noise_var
            post_prec = prior_prec + like_prec
            self.mu[i] = (self.mu[i]*prior_prec + r*like_prec) / post_prec
            self.var[i] = 1.0 / post_prec

        return chosen

class CombinatorialUCB:
    def __init__(self, n_arms, budget, noise_var=1.0):
        """
        n_arms: number of base arms
        budget: number of arms to select per round (top-k)
        noise_var: known variance of rewards
        """
        self.n_arms = n_arms
        self.budget = budget
        self.noise_var = noise_var

        self.means = np.zeros(n_arms)    # empirical means
        self.counts = np.zeros(n_arms)   # number of pulls
        self.t = 1                       # time step

    def step(self, get_rewards):
        """
        Perform one round of combinatorial UCB

        get_rewards: function(chosen) -> rewards for chosen arms
        """
        # 1. Compute UCB for each arm
        ucb = np.zeros(self.n_arms)
        for i in range(self.n_arms):
            if self.counts[i] == 0:
                ucb[i] = float('inf')  # ensure each arm is pulled at least once
            else:
                ucb[i] = self.means[i] + np.sqrt(2 * self.noise_var * np.log(self.t) / self.counts[i])

        # 2. Top-k selection
        chosen = np.argsort(ucb)[-self.budget:]

        # 3. Observe rewards
        rewards = get_rewards[chosen]

        # 4. Update empirical means and counts
        for i, r in zip(chosen, rewards):
            self.counts[i] += 1
            self.means[i] += (r - self.means[i]) / self.counts[i]

        self.t += 1
        return chosen

class GPfunctions:
    def __init__(self, K, length_scale=None, IfStationary=True):
        self.K=K
        self.num_points= 500 #number of actions
        actionspace =  np.linspace(-5,5,self.num_points) #.reshape(-1, 1) # grid points
        self.actionspace = np.sort(actionspace,axis=0)
        self.length_scale=length_scale
        # Compute covariance matrix
        if IfStationary == True:
            self.kernel = self.rbf_kernel()
        else:
            self.kernel = self.gibbs_kernel()
   
        self.subset = self.algorithm()

    # Stationary Gaussian Kernel
    def rbf_kernel(self):
        """Computes the RBF kernel matrix."""
        actionset=self.actionspace.reshape((-1,1))
        sq_dist = cdist(actionset,actionset, 'sqeuclidean')
        return np.exp(-sq_dist / (2 * self.length_scale ** 2))
    
    # Non-stationary Gibbs Kernel
    def gibbs_kernel(self):
        """Computes the Gibbs kernel matrix."""
        K = np.zeros((self.num_points,self.num_points))
        # Compute the kernel matrix
        for i in range(self.num_points):
            for j in range(self.num_points):
                K[i,j] = self.gibbs_kernel_fun(self.actionspace[i],self.actionspace[j])
        return K

    # Define an input-dependent length scale function l(x)
    def length_scale_fun(self, x):
        return 0.5 + 0.5* np.exp(-(x/self.length_scale)**2)  # Short length scale near 0, longer away

    # Define the 1D Gibbs kernel function
    def gibbs_kernel_fun(self, x, x_prime):
        l_x = self.length_scale_fun(x)
        l_xp = self.length_scale_fun(x_prime)
        numerator = 2 * l_x * l_xp
        denominator = l_x**2 + l_xp**2
        prefactor = np.sqrt(numerator / denominator)
        exponent = - (x - x_prime)**2 / denominator
        return prefactor * np.exp(exponent)

    def samples(self,size):
        # Sample multiple functions from the GP
        return np.random.multivariate_normal(mean=np.zeros(self.num_points), cov=self.kernel,size=size)
    
    def algorithm(self):
        # Sample multiple functions from the GP
        f_samples = self.samples(size=self.K)
        # Find the max index for each batch
        max_indices = np.argmax(f_samples, axis=1)  # Shape: (num_batches,)
        # Get unique max indices
        subset = np.unique(max_indices)

        while len(subset) < self.K: # add more items until K distinct actions are found
            f_samples = self.samples(size=self.K-len(subset))
            max_indices = np.argmax(f_samples, axis=1)
            subset = np.unique(np.append(subset,max_indices))
  
        return subset
    
    def test(self,subset):
        num_batches = 10**5  # Number of function samples for testing
        # Sample multiple functions from the GP
        f_samples = self.samples(size=num_batches) # np.random.multivariate_normal(mean=np.zeros(self.num_points), cov=self.kernel, size=num_batches)
        return np.average(np.max(f_samples, axis=1)-np.max(f_samples[:,subset], axis=1))

    def ucb_action_selection(self, N):
        """
        Selects a subset of K actions using Upper Confidence Bound (UCB).
        Returns: List of selected action indices.
        """
        # Sample multiple functions from the GP
        f_samples = self.samples(size=N) 
        ucb = CombinatorialUCB(self.num_points, self.K, noise_var=1.0)

        for t in range(N):
            selected_actions = ucb.step(f_samples[t,:])
        return selected_actions
    
    def ts_action_selection(self, N):
        """
        Selects a subset of K actions using Thompson Sampling.
        Returns: List of selected action indices.
        """
        # Sample multiple functions from the GP
        f_samples = self.samples(size=N) 
        
        cts = GaussianCTS(self.num_points, self.K, prior_var=1.0, noise_var=1.0)

        for t in range(N):
            selected_actions = cts.step(f_samples[t,:])
        return selected_actions  