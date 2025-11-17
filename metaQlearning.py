import time
import random
from typing import Dict, List, Tuple, Any
import numpy as np

class QLearningExperimentRunner:
    def __init__(self, gen, low: float, high: float, num_actions: int, 
                 actionset_dict: Dict, lr: float = 0.1, gamma: float = 0.99, 
                 epsilon: float = 1.0, force_mag: float = 10.0,
                 min_td_error: float = 0.001, consecutive_small_errors: int = 10):
        """
        Initialize the Q-learning experiment runner.
        
        Args:
            gen: Environment generator
            low: Lower bound for state space discretization
            high: Upper bound for state space discretization  
            num_actions: Number of discrete actions
            actionset_dict: Dictionary to store learned actions
            lr: Learning rate
            gamma: Discount factor
            epsilon: Exploration rate
            force_mag: Force magnitude for agent
            min_td_error: Minimum TD error threshold for early stopping
            consecutive_small_errors: Consecutive small errors needed for early stop
        """
        self.gen = gen
        self.low = low
        self.high = high
        self.num_actions = num_actions
        self.actionset_dict = actionset_dict
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.force_mag = force_mag
        self.min_td_error = min_td_error
        self.consecutive_small_errors = consecutive_small_errors
        
        # Store run results
        self.run_results = []
    
    def run_single_experiment(self, episodes: int, category: str = None, 
                            verbose: bool = False) -> Dict[str, Any]:
        """
        Run a single Q-learning experiment.
        
        Args:
            episodes: Number of episodes to run
            category: Specific category to use, if None chooses randomly from first 3
            verbose: Whether to print progress information
            
        Returns:
            Dictionary containing run results and metrics
        """
        start_time = time.time()
        
        # Choose category if not specified
        if category is None:
            category = random.choice(list(self.gen.categories.keys())[:3])
        
        if verbose:
            print(f"Starting experiment with category: {category}")
        
        # Generate environment and agent
        env = self.gen.generate_env(category)
        agent = TabularQLearningAgent(
            statespace=[self.low, self.high],
            num_actions=self.num_actions,
            actionspace=self.actionset_dict, 
            lr=self.lr,
            gamma=self.gamma,
            epsilon=self.epsilon,
            force_mag=self.force_mag
        )
        
        # Track episode rewards and steps
        episode_rewards = []
        episode_steps = []
        early_stops = 0
        
        # Training loop
        for episode in range(episodes):
            state = env.reset()
            total_reward = 0
            done = False
            steps = 0
            small_error_count = 0
            
            while not done:
                # Choose action (returns index 0-9)
                a = agent.choose_action(state)
                action = agent.discrete_actions[a]
                
                # Take action in environment
                result = env.step(action)
                if len(result) == 4:
                    next_state, reward, done, info = result
                else:
                    next_state, reward, done, truncated, info = result
                    done = done or truncated
                
                # Update Q-table and get TD error
                td_error = agent.update(state, a, reward, next_state, done)
                
                state = next_state
                total_reward += reward
                steps += 1
                
                # Check if TD error is small enough to stop episode
                if abs(td_error) < self.min_td_error:
                    small_error_count += 1
                else:
                    small_error_count = 0
                    
                # Stop episode if TD error has been small for consecutive steps
                if small_error_count >= self.consecutive_small_errors:
                    done = True
                    early_stops += 1
                    if verbose and episode % 1000 == 0:
                        print(f"Episode {episode} stopped early due to small TD error")
            
            episode_rewards.append(total_reward)
            episode_steps.append(steps)
            
            # Decrease exploration over time
            if episode % 100 == 0:
                agent.decrease_epsilon()
        
        env.close()
        
        # Update actionset_dict with learned policies
        initial_action_count = len(self.actionset_dict)
        for state_tuple in agent.disc.get_all_discrete_states():
            self.actionset_dict[state_tuple].append(np.argmax(agent.Q[state_tuple]))
        
        end_time = time.time()
        runtime = end_time - start_time
        
        # Compile results
        result = {
            'category': category,
            'runtime_seconds': runtime,
            'total_episodes': episodes,
            'early_stops': early_stops,
            'mean_reward': np.mean(episode_rewards),
            'std_reward': np.std(episode_rewards),
            'mean_steps': np.mean(episode_steps),
            'final_epsilon': agent.epsilon,
            'actions_added': len(self.actionset_dict) - initial_action_count,
            'agent': agent
        }
        
        if verbose:
            print(f"Experiment completed in {runtime:.2f} seconds")
            print(f"Mean reward: {result['mean_reward']:.2f} ± {result['std_reward']:.2f}")
            print(f"Early stops: {early_stops}/{episodes}")
        
        return result
    
    def run_multiple_experiments(self, num_runs: int, episodes_per_run: int, 
                               categories: List[str] = None, verbose: bool = True) -> List[Dict]:
        """
        Run multiple Q-learning experiments.
        
        Args:
            num_runs: Number of experiments to run
            episodes_per_run: Number of episodes per experiment
            categories: List of categories to use (cycles through if provided)
            verbose: Whether to print progress information
            
        Returns:
            List of results from each run
        """
        self.run_results = []
        
        for run_idx in range(num_runs):
            if verbose:
                print(f"\n--- Starting Run {run_idx + 1}/{num_runs} ---")
            
            # Select category for this run
            if categories is not None:
                category = categories[run_idx % len(categories)]
            else:
                category = None
            
            result = self.run_single_experiment(
                episodes=episodes_per_run,
                category=category,
                verbose=verbose
            )
            
            result['run_id'] = run_idx
            self.run_results.append(result)
            
            if verbose:
                print(f"Run {run_idx + 1} completed in {result['runtime_seconds']:.2f}s")
        
        return self.run_results
    
    def print_summary(self):
        """Print a summary of all runs."""
        if not self.run_results:
            print("No results to summarize. Run experiments first.")
            return
        
        print("\n" + "="*50)
        print("EXPERIMENT SUMMARY")
        print("="*50)
        
        runtimes = [r['runtime_seconds'] for r in self.run_results]
        mean_rewards = [r['mean_reward'] for r in self.run_results]
        
        print(f"Total runs: {len(self.run_results)}")
        print(f"Average runtime: {np.mean(runtimes):.2f} ± {np.std(runtimes):.2f} seconds")
        print(f"Average mean reward: {np.mean(mean_rewards):.2f} ± {np.std(mean_rewards):.2f}")
        print(f"Total actions in actionset_dict: {len(self.actionset_dict)}")
        
        print("\nDetailed results:")
        for result in self.run_results:
            print(f"Run {result['run_id']+1}: {result['category']} - "
                  f"{result['runtime_seconds']:.2f}s, "
                  f"reward: {result['mean_reward']:.2f} ± {result['std_reward']:.2f}")

# Usage example:
if __name__ == "__main__":
    # Initialize with your parameters
    runner = QLearningExperimentRunner(
        gen=gen,  # your environment generator
        low=low,
        high=high, 
        num_actions=num_actions,
        actionset_dict=actionset_dict,
        lr=lr,
        gamma=gamma,
        epsilon=epsilon,
        force_mag=force_mag
    )
    
    # Run multiple experiments
    results = runner.run_multiple_experiments(
        num_runs=5,
        episodes_per_run=1000,
        verbose=True
    )
    
    # Print summary
    runner.print_summary()
    
    # Access individual results
    for i, result in enumerate(results):
        print(f"\nRun {i+1}:")
        print(f"  Category: {result['category']}")
        print(f"  Runtime: {result['runtime_seconds']:.2f}s")
        print(f"  Mean reward: {result['mean_reward']:.2f}")