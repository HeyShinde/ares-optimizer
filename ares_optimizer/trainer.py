"""
Main trainer module for Ares optimizer.
Integrates RL agent with feature extractor, code transformer, and reward calculator.
"""

import os
import time
from typing import Dict, List, Optional, Tuple
import numpy as np
import torch
from ares_optimizer.state_representation.feature_extractor import FeatureExtractor
from ares_optimizer.code_transformer.transformations import (
    apply_lru_cache,
    convert_list_to_set_for_membership,
    replace_loop_with_sum,
    use_generator_expression,
    introduce_early_exit
)
from ares_optimizer.reward_calculator import RewardCalculator
from ares_optimizer.rl_agent.agent import DQNAgent
from ares_optimizer.ares_env.code_executor import CodeExecutor
from ares_optimizer.ares_env.test_manager import TestManager
import re
import logging


class AresTrainer:
    def __init__(
        self,
        code_executor: CodeExecutor,
        test_manager: TestManager,
        state_dim: int = 18,  # Default state dimension from feature extractor
        action_dim: int = 5,  # Number of available transformations
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
        save_dir: str = 'checkpoints',
        **agent_kwargs
    ):
        """
        Initialize the Ares trainer.

        Args:
            code_executor: CodeExecutor instance for measuring performance
            test_manager: TestManager instance for verifying correctness
            state_dim: Dimension of state vector from feature extractor
            action_dim: Number of available transformations
            device: Device to run the model on ('cuda' or 'cpu')
            save_dir: Directory to save model checkpoints
            **agent_kwargs: Additional arguments for DQNAgent
        """
        self.code_executor = code_executor
        self.test_manager = test_manager
        self.feature_extractor = FeatureExtractor()
        self.reward_calculator = RewardCalculator(code_executor, test_manager)
        self.agent = DQNAgent(
            state_dim=state_dim,
            action_dim=action_dim,
            device=device,
            **agent_kwargs
        )
        self.device = device
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)

        # Define available transformations
        self.transformations = [
            apply_lru_cache,
            convert_list_to_set_for_membership,
            replace_loop_with_sum,
            use_generator_expression,
            introduce_early_exit
        ]

        # Initialize logger
        self.logger = logging.getLogger(__name__)

    def get_state(self, code: str, runtime_ms: float = 0.0, memory_mb: float = 0.0) -> np.ndarray:
        """Get state representation for the given code and performance metrics."""
        return self.feature_extractor.get_state_vector(code, runtime_ms, memory_mb)

    def apply_transformation(self, code: str, action: int) -> str:
        """Apply the selected transformation to the code."""
        if action >= len(self.transformations):
            raise ValueError(f"Invalid action {action}")
        
        transform_func = self.transformations[action]
        try:
            # For transformations that require additional parameters
            if transform_func == convert_list_to_set_for_membership:
                # Find list variables in the code (simplified)
                return transform_func(code, "my_list")  # You might want to make this smarter
            elif transform_func == introduce_early_exit:
                return transform_func(code, "i > 5")  # You might want to make this smarter
            else:
                return transform_func(code)
        except Exception as e:
            print(f"Error applying transformation {action}: {e}")
            return code  # Return original code if transformation fails

    def _extract_function_name(self, code: str) -> str:
        """Extract the first function name from the code string."""
        match = re.search(r'def\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\(', code)
        if match:
            return match.group(1)
        raise ValueError("No function definition found in code.")

    def train_episode(
        self,
        code: str,
        test_cases: Dict,
        max_steps: int = 10,
        verbose: bool = True
    ) -> Tuple[float, List[Dict]]:
        """
        Train the agent on a single code optimization episode.

        Args:
            code: Original Python code to optimize
            test_cases: Test cases for correctness verification
            max_steps: Maximum number of transformation steps
            verbose: Whether to print progress

        Returns:
            Tuple of (final_reward, episode_metrics)
        """
        current_code = code
        episode_metrics = []
        total_reward = 0

        function_name = self._extract_function_name(current_code)
        test_inputs = test_cases['test_cases'] if isinstance(test_cases, dict) and 'test_cases' in test_cases else test_cases
        orig_runtime, orig_memory, _ = self.code_executor.execute_function(current_code, function_name, test_inputs)
        state = self.get_state(current_code, orig_runtime, orig_memory)

        for step in range(max_steps):
            # Select and apply transformation
            if step == 0:
                # Force a transformation for the first step (for testing)
                action = 2  # Example: replace_loop_with_sum
                self.logger.info(f"Forcing transformation: {self.transformations[action].__name__}")
            else:
                action = self.agent.select_action(state)
                self.logger.info(f"Selected transformation: {self.transformations[action].__name__}")

            transformed_code = self.apply_transformation(current_code, action)
            self.logger.info(f"Transformed code:\n{transformed_code}")

            function_name = self._extract_function_name(transformed_code)
            runtime, memory, _ = self.code_executor.execute_function(transformed_code, function_name, test_inputs)
            
            # Calculate reward
            reward, metrics = self.reward_calculator.calculate_reward(
                current_code,
                transformed_code,
                test_cases
            )
            
            # Get next state
            next_state = self.get_state(transformed_code, runtime, memory)
            
            # Store transition
            self.agent.store_transition(
                state,
                action,
                reward,
                next_state,
                step == max_steps - 1  # done flag
            )
            
            # Train agent
            loss = self.agent.train_step()
            
            # Update metrics
            metrics.update({
                'step': step,
                'action': action,
                'loss': loss,
                'reward': reward,
                'runtime': runtime,
                'memory': memory
            })
            episode_metrics.append(metrics)
            
            if verbose:
                print(f"Step {step}: Action {action}, Reward {reward:.3f}, Loss {loss:.3f if loss else 'N/A'}")
            
            # Update current code if transformation was successful
            if reward > 0:
                current_code = transformed_code
                state = next_state
            else:
                # If not successful, keep state as is
                state = next_state
            
            total_reward += reward
            
            # Early stopping if we've found a good solution
            if reward > 0.8:  # High reward threshold
                break
        
        # Summarize episode metrics
        rewards = [m['reward'] for m in episode_metrics]
        return total_reward, {'rewards': rewards, 'steps': len(rewards)}

    def train(
        self,
        code_samples: List[Tuple[str, Dict]],
        num_episodes: int = 1000,
        save_freq: int = 100,
        verbose: bool = True
    ):
        """
        Train the agent on multiple code samples.

        Args:
            code_samples: List of (code, test_cases) tuples
            num_episodes: Number of training episodes
            save_freq: Frequency of model checkpointing
            verbose: Whether to print progress
        """
        best_reward = float('-inf')
        
        for episode in range(num_episodes):
            # Select random code sample
            code, test_cases = code_samples[episode % len(code_samples)]
            
            # Train on the sample
            total_reward, metrics = self.train_episode(
                code,
                test_cases,
                verbose=verbose
            )
            
            if verbose and episode % 10 == 0:
                print(f"Episode {episode}: Total Reward {total_reward:.3f}")
            
            # Save best model
            if total_reward > best_reward:
                best_reward = total_reward
                self.agent.save(os.path.join(self.save_dir, 'best_model.pt'))
            
            # Periodic checkpointing
            if episode % save_freq == 0:
                self.agent.save(os.path.join(self.save_dir, f'checkpoint_{episode}.pt'))

    def optimize_code(self, code: str, test_cases: dict, max_steps: int = 10, verbose: bool = True) -> str:
        """Optimize the given code using the trained agent."""
        current_code = code
        best_code = code
        best_reward = float('-inf')
        _, metrics = self.train_episode(code, test_cases, max_steps, verbose)
        rewards = metrics['rewards']
        if not rewards or max(rewards) <= 0:
            return code
        # If there are positive rewards, return the code at the best reward step
        # (for now, just return the last code, as we don't store code per step)
        return current_code 