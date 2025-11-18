    def run_single_experiment(self, episodes: int, category: str = None, 
                            use_actionset_as_actionspace: bool = False,
                            update_actionset_dict: bool = True,
                            env=None,  # Accept existing environment
                            verbose: bool = False) -> Dict[str, Any]:
        """
        Run a single Q-learning experiment.
        
        Args:
            episodes: Number of episodes to run
            category: Specific category to use, if None chooses randomly from first 3
            use_actionset_as_actionspace: If True, use actionset_dict as action space instead of actionspace_dict
            update_actionset_dict: Whether to update actionset_dict with learned policy
            env: Existing environment to use (if None, creates new one)
            verbose: Whether to print progress information
        """
        start_time = time.time()
        
        print('env',env)

        # Choose category if not specified and no env provided
        if category is None and env is None:
            category = random.choice(list(self.gen.categories.keys())[:3])
        
        # Generate environment if not provided
        if env is None:
            env = self.gen.generate_env(category)
            close_env = True
        else:
            close_env = False
            category = "provided_env"  # Placeholder since we don't know the category
        
        if verbose:
            print(f"Starting experiment with {'provided environment' if env else f'category: {category}'}")
            if use_actionset_as_actionspace:
                print("Using actionset_dict as action space")
            else:
                mode = "training" + (" (updating actionset_dict)" if update_actionset_dict else " (not updating actionset_dict)")
                print(f"Using actionspace_dict as action space ({mode})")
        
        # Choose which action space to use
        if use_actionset_as_actionspace:
            action_space = self.actionset_dict
        else:
            action_space = self.actionspace_dict
        
        # Create agent
        agent = TabularQLearningAgent(
            statespace=[self.low, self.high],
            num_actions=self.num_actions,
            actionspace=action_space, 
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

        # Close environment only if we created it
        if close_env:
            env.close()

        end_time = time.time()
        runtime = end_time - start_time
        
        # Update actionset_dict only if requested AND we're not using actionset_dict as action space
        if update_actionset_dict and not use_actionset_as_actionspace:
            for state_tuple in agent.disc.get_all_discrete_states():
                self.actionset_dict[state_tuple].append(np.argmax(agent.Q[state_tuple]))
        
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
            'mode': 'evaluation' if use_actionset_as_actionspace else 'training',
            'actionset_updated': update_actionset_dict and not use_actionset_as_actionspace
        }
        
        if verbose:
            print(f"Experiment completed in {runtime:.2f} seconds")
            print(f"Mean reward: {result['mean_reward']:.2f} ± {result['std_reward']:.2f}")
            print(f"Early stops: {early_stops}/{episodes}")
            print(f"Mode: {result['mode']}")
            if result['actionset_updated']:
                print("Actionset dictionary was updated")
        
        return result

 