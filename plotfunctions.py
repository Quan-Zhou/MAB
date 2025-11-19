import matplotlib.pyplot as plt
import numpy as np
import time

def plot_training_comparison(comparison_results):
    """Plot training runtime and evaluation rewards"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Training runtime
    ax1.boxplot([comparison_results['actionspace_training_runtimes'], 
                 comparison_results['actionset_training_runtimes']],
                labels=['ActionSpace', 'ActionSet'])
    ax1.set_ylabel('Training Time (seconds)')
    ax1.set_title('Training Runtime')
    
    # Evaluation rewards
    ax2.boxplot([comparison_results['actionspace_eval_rewards'], 
                 comparison_results['actionset_eval_rewards']],
                labels=['ActionSpace', 'ActionSet'])
    ax2.set_ylabel('Average Reward')
    ax2.set_title('Policy Performance')
    
    plt.tight_layout()
    plt.show()

def plot_runtime_vs_reward(comparison_results):
    """Trade-off between training time and performance"""
    plt.figure(figsize=(10, 6))
    
    plt.scatter(comparison_results['actionspace_training_runtimes'], 
                comparison_results['actionspace_eval_rewards'],
                alpha=0.7, label='ActionSpace', s=80)
    plt.scatter(comparison_results['actionset_training_runtimes'], 
                comparison_results['actionset_eval_rewards'],
                alpha=0.7, label='ActionSet', s=80)
    
    plt.xlabel('Training Time (seconds)')
    plt.ylabel('Average Reward')
    plt.title('Training Time vs Performance')
    plt.legend()
    plt.show()

def plot_statistical_summary(comparison_results):
    """Bar plot with error bars for key metrics"""
    metrics = ['Training Time', 'Average Reward']
    actionspace_means = [
        np.mean(comparison_results['actionspace_training_runtimes']),
        np.mean(comparison_results['actionspace_eval_rewards'])
    ]
    actionset_means = [
        np.mean(comparison_results['actionset_training_runtimes']),
        np.mean(comparison_results['actionset_eval_rewards'])
    ]
    
    actionspace_stds = [
        np.std(comparison_results['actionspace_training_runtimes']),
        np.std(comparison_results['actionspace_eval_rewards'])
    ]
    actionset_stds = [
        np.std(comparison_results['actionset_training_runtimes']),
        np.std(comparison_results['actionset_eval_rewards'])
    ]
    
    x = np.arange(len(metrics))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x - width/2, actionspace_means, width, yerr=actionspace_stds, 
           label='ActionSpace', capsize=5, alpha=0.8)
    ax.bar(x + width/2, actionset_means, width, yerr=actionset_stds, 
           label='ActionSet', capsize=5, alpha=0.8)
    
    ax.set_ylabel('Performance')
    ax.set_title('ActionSpace vs ActionSet Performance')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.legend()
    
    plt.show()

def collect_runs_tradeoff_data(runner, max_runs=10, episodes_per_run=1000):
    """Collect data on how performance changes with number of training runs"""
    tradeoff_data = {
        'num_runs': [],
        'final_sizes': [],
        'total_runtimes': [],
        'average_rewards': [],
        'runtime_stds': [],
        'reward_stds': []
    }
    
    # Store initial state to reset between tests
    initial_actionset = runner.actionset_dict.copy()
    
    for num_runs in range(1, max_runs + 1):
        print(f"Testing with {num_runs} training runs...")
        
        # Reset actionset_dict for each test
        runner.actionset_dict = initial_actionset.copy()
        
        # Run the specified number of training runs
        start_time = time.time()
        results = runner.run_multiple_experiments(
            num_runs=num_runs,
            episodes_per_run=episodes_per_run,
            use_actionset_as_actionspace=False,
            update_actionset_dict=True,
            verbose=False
        )
        total_runtime = time.time() - start_time
        
        # Evaluate the final policy using the accumulated action set
        eval_result = runner.evaluate_policy(
            episodes=500,
            use_actionset_as_actionspace=True,  # Use the built action set
            max_steps_per_episode=1000,
            verbose=False
        )
        
        # Store data
        tradeoff_data['num_runs'].append(num_runs)
        tradeoff_data['final_sizes'].append(len(runner.actionset_dict))
        tradeoff_data['total_runtimes'].append(total_runtime)
        tradeoff_data['average_rewards'].append(eval_result['mean_reward'])
        tradeoff_data['reward_stds'].append(eval_result['std_reward'])
        
        # Calculate runtime std from individual runs
        run_runtimes = [result['runtime_seconds'] for result in results]
        tradeoff_data['runtime_stds'].append(np.std(run_runtimes))
        
        print(f"  Runs: {num_runs}, Size: {len(runner.actionset_dict)}, "
              f"Reward: {eval_result['mean_reward']:.2f}, Runtime: {total_runtime:.2f}s")
    
    return tradeoff_data

def plot_runs_vs_runtime(tradeoff_data):
    """Plot number of training runs vs total runtime"""
    plt.figure(figsize=(10, 6))
    
    num_runs = tradeoff_data['num_runs']
    total_runtimes = tradeoff_data['total_runtimes']
    runtime_stds = tradeoff_data['runtime_stds']
    
    plt.errorbar(num_runs, total_runtimes, yerr=runtime_stds, 
                 marker='o', capsize=5, linewidth=2, markersize=8)
    
    plt.xlabel('Number of Training Runs')
    plt.ylabel('Total Training Runtime (seconds)')
    plt.title('Training Cost vs Number of Runs')
    plt.grid(True, alpha=0.3)
    
    # Add trend line
    z = np.polyfit(num_runs, total_runtimes, 1)
    p = np.poly1d(z)
    plt.plot(num_runs, p(num_runs), 'r--', alpha=0.8, label=f'Linear trend')
    plt.legend()
    
    plt.show()

def plot_runs_vs_reward(tradeoff_data):
    """Plot number of training runs vs evaluation reward"""
    plt.figure(figsize=(10, 6))
    
    num_runs = tradeoff_data['num_runs']
    average_rewards = tradeoff_data['average_rewards']
    reward_stds = tradeoff_data['reward_stds']
    
    plt.errorbar(num_runs, average_rewards, yerr=reward_stds,
                 marker='s', capsize=5, linewidth=2, markersize=8, color='orange')
    
    plt.xlabel('Number of Training Runs')
    plt.ylabel('Average Evaluation Reward')
    plt.title('Performance vs Number of Training Runs')
    plt.grid(True, alpha=0.3)
    
    plt.show()

def plot_size_vs_performance(tradeoff_data):
    """Plot final action set size vs performance metrics"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    sizes = tradeoff_data['final_sizes']
    rewards = tradeoff_data['average_rewards']
    runtimes = tradeoff_data['total_runtimes']
    
    # Size vs Reward
    ax1.errorbar(sizes, rewards, yerr=tradeoff_data['reward_stds'],
                 marker='o', capsize=5, linewidth=2, markersize=6)
    ax1.set_xlabel('Final Action Set Size')
    ax1.set_ylabel('Average Reward')
    ax1.set_title('Action Set Size vs Performance')
    ax1.grid(True, alpha=0.3)
    
    # Size vs Runtime
    ax2.errorbar(sizes, runtimes, yerr=tradeoff_data['runtime_stds'],
                 marker='s', capsize=5, linewidth=2, markersize=6, color='red')
    ax2.set_xlabel('Final Action Set Size')
    ax2.set_ylabel('Total Runtime (seconds)')
    ax2.set_title('Action Set Size vs Training Cost')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def plot_comprehensive_tradeoff(tradeoff_data):
    """Comprehensive plot showing all trade-offs"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    num_runs = tradeoff_data['num_runs']
    sizes = tradeoff_data['final_sizes']
    rewards = tradeoff_data['average_rewards']
    runtimes = tradeoff_data['total_runtimes']
    
    # Plot 1: Runs vs Reward
    ax1.errorbar(num_runs, rewards, yerr=tradeoff_data['reward_stds'],
                 marker='o', capsize=5, linewidth=2, markersize=6)
    ax1.set_xlabel('Number of Training Runs')
    ax1.set_ylabel('Average Reward')
    ax1.set_title('Performance vs Training Runs')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Runs vs Runtime
    ax2.errorbar(num_runs, runtimes, yerr=tradeoff_data['runtime_stds'],
                 marker='s', capsize=5, linewidth=2, markersize=6, color='red')
    ax2.set_xlabel('Number of Training Runs')
    ax2.set_ylabel('Total Runtime (seconds)')
    ax2.set_title('Training Cost vs Training Runs')
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Size vs Reward
    ax3.scatter(sizes, rewards, s=100, alpha=0.7)
    ax3.set_xlabel('Action Set Size')
    ax3.set_ylabel('Average Reward')
    ax3.set_title('Action Set Size vs Performance')
    ax3.grid(True, alpha=0.3)
    
    # Add size labels
    for i, (size, reward) in enumerate(zip(sizes, rewards)):
        ax3.annotate(f'{num_runs[i]} runs', (size, reward), 
                    xytext=(5, 5), textcoords='offset points', fontsize=8)
    
    # Plot 4: Size vs Runtime
    ax4.scatter(sizes, runtimes, s=100, alpha=0.7, color='red')
    ax4.set_xlabel('Action Set Size')
    ax4.set_ylabel('Total Runtime (seconds)')
    ax4.set_title('Action Set Size vs Training Cost')
    ax4.grid(True, alpha=0.3)
    
    # Add size labels
    for i, (size, runtime) in enumerate(zip(sizes, runtimes)):
        ax4.annotate(f'{num_runs[i]} runs', (size, runtime), 
                    xytext=(5, 5), textcoords='offset points', fontsize=8)
    
    plt.tight_layout()
    plt.show()

def analyze_runs_tradeoff(runner, max_runs=8, episodes_per_run=1000):
    """Complete analysis of trade-offs between number of runs and performance"""
    print("="*60)
    print("ANALYZING TRADE-OFF: Number of Runs vs Performance")
    print("="*60)
    
    # Collect data
    tradeoff_data = collect_runs_tradeoff_data(
        runner, 
        max_runs=max_runs, 
        episodes_per_run=episodes_per_run
    )
    
    # Generate all plots
    plot_runs_vs_runtime(tradeoff_data)
    plot_runs_vs_reward(tradeoff_data)
    plot_size_vs_performance(tradeoff_data)
    plot_comprehensive_tradeoff(tradeoff_data)
    
    # Print summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    for i, num_runs in enumerate(tradeoff_data['num_runs']):
        print(f"{num_runs} runs: Size={tradeoff_data['final_sizes'][i]}, "
              f"Reward={tradeoff_data['average_rewards'][i]:.2f} ± {tradeoff_data['reward_stds'][i]:.2f}, "
              f"Runtime={tradeoff_data['total_runtimes'][i]:.2f}s")
    
    # Find optimal point (highest reward per runtime)
    efficiencies = [reward / runtime for reward, runtime in 
                   zip(tradeoff_data['average_rewards'], tradeoff_data['total_runtimes'])]
    optimal_idx = np.argmax(efficiencies)
    
    print(f"\n🎯 Most efficient: {tradeoff_data['num_runs'][optimal_idx]} runs "
          f"(Reward/Runtime: {efficiencies[optimal_idx]:.4f})")
    
    return tradeoff_data
