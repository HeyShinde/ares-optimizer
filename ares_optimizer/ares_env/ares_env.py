"""RL environment module for Ares Optimizer.

This module implements the reinforcement learning environment for code optimization,
providing the interface between the RL agent and the code transformation system.
"""

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from typing import Any, Dict, List, Optional, Tuple, Union

from ..code_transformer.transformations import Transformations
from ..state_representation.state_representation import StateRepresentation, OptimizationState
from .code_executor import CodeExecutor
from .test_manager import TestManager


class AresEnv(gym.Env):
    """Reinforcement learning environment for code optimization."""

    def __init__(
        self,
        initial_code: str,
        function_name: str,
        test_cases: List[Dict[str, Any]],
        max_steps: int = 50,
        reward_weights: Optional[Dict[str, float]] = None,
    ):
        """Initialize the Ares environment.

        Args:
            initial_code: The initial Python code to optimize.
            function_name: Name of the function to optimize.
            test_cases: List of test cases for validation.
            max_steps: Maximum number of optimization steps.
            reward_weights: Weights for different reward components.
        """
        super().__init__()

        # Initialize components
        self.transformations = Transformations()
        self.state_representation = StateRepresentation()
        self.code_executor = CodeExecutor()
        self.test_manager = TestManager()

        # Store initial state
        self.initial_code = initial_code
        self.function_name = function_name
        self.test_cases = test_cases
        self.max_steps = max_steps
        self.current_step = 0

        # Set up reward weights
        self.reward_weights = reward_weights or {
            "runtime": 0.4,
            "memory": 0.3,
            "correctness": 0.3,
        }

        # Initialize action and observation spaces
        self._setup_spaces()

        # Load test cases
        self.test_manager.load_test_cases(function_name, test_cases)

    def _setup_spaces(self):
        """Set up the action and observation spaces."""
        # Action space: discrete space for transformation selection
        num_transformations = len(self.transformations.get_available_transformations())
        self.action_space = spaces.Discrete(num_transformations)

        # Observation space: continuous space for state features
        state_size = self.state_representation.get_state_size()
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(state_size,), dtype=np.float32
        )

    def reset(self) -> np.ndarray:
        """Reset the environment to initial state.

        Returns:
            Initial state observation.
        """
        self.current_step = 0

        # Get initial performance metrics
        initial_metrics = self._evaluate_code(self.initial_code)

        # Create initial state
        self.state = self.state_representation.create_initial_state(
            self.initial_code, self.function_name, initial_metrics
        )

        return self.state_representation.get_state_vector(self.state)

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """Take a step in the environment.

        Args:
            action: Index of the transformation to apply.

        Returns:
            Tuple of (observation, reward, done, info).
        """
        self.current_step += 1

        # Get available transformations
        available_transformations = self.transformations.get_available_transformations()
        if action >= len(available_transformations):
            return self._get_observation(), -1.0, True, {"error": "Invalid action"}

        # Apply transformation
        transformation = available_transformations[action]
        new_code, success = getattr(self.transformations, transformation["name"])(
            self.state.code, self.function_name
        )

        if not success:
            return self._get_observation(), -0.5, False, {"error": "Transformation failed"}

        # Evaluate new code
        new_metrics = self._evaluate_code(new_code)

        # Update state
        self.state = self.state_representation.update_state(
            self.state,
            new_code,
            new_metrics,
            {
                "name": transformation["name"],
                "type": transformation["type"],
                "description": transformation["description"],
            },
        )

        # Calculate reward
        reward = self._calculate_reward(new_metrics)

        # Check if done
        done = self.current_step >= self.max_steps

        return self._get_observation(), reward, done, {
            "metrics": new_metrics,
            "transformation": transformation,
        }

    def _evaluate_code(self, code: str) -> Dict[str, float]:
        """Evaluate code performance and correctness.

        Args:
            code: The code to evaluate.

        Returns:
            Dictionary of performance metrics.
        """
        # Execute code and get performance metrics
        result, metrics = self.code_executor.execute_function(code, self.function_name)
        if result is None:
            return {
                "runtime": float("inf"),
                "memory_usage": float("inf"),
                "correctness": 0.0,
            }

        # Validate against test cases
        is_correct, test_results = self.test_manager.validate_function(
            code, self.function_name
        )

        # Combine metrics
        return {
            "runtime": metrics["runtime"],
            "memory_usage": metrics["memory_usage"],
            "correctness": 1.0 if is_correct else 0.0,
        }

    def _calculate_reward(self, metrics: Dict[str, float]) -> float:
        """Calculate reward based on performance metrics.

        Args:
            metrics: Dictionary of performance metrics.

        Returns:
            Reward value.
        """
        # Normalize metrics
        normalized_metrics = {
            "runtime": self._normalize_metric(metrics["runtime"], "runtime"),
            "memory": self._normalize_metric(metrics["memory_usage"], "memory"),
            "correctness": metrics["correctness"],
        }

        # Calculate weighted reward
        reward = sum(
            normalized_metrics[metric] * weight
            for metric, weight in self.reward_weights.items()
        )

        return reward

    def _normalize_metric(self, value: float, metric_type: str) -> float:
        """Normalize a metric value to [0, 1] range.

        Args:
            value: The metric value to normalize.
            metric_type: Type of the metric.

        Returns:
            Normalized value.
        """
        if value == float("inf"):
            return 0.0

        # Get baseline from initial state
        baseline = self.state.performance_metrics.get(
            "runtime" if metric_type == "runtime" else "memory_usage", value
        )

        if baseline == 0:
            return 1.0 if value == 0 else 0.0

        # Calculate improvement ratio
        improvement = (baseline - value) / baseline
        return max(0.0, min(1.0, improvement))

    def _get_observation(self) -> np.ndarray:
        """Get the current state observation.

        Returns:
            State observation as numpy array.
        """
        return self.state_representation.get_state_vector(self.state)

    def render(self, mode: str = "human"):
        """Render the current state of the environment.

        Args:
            mode: Rendering mode.
        """
        if mode == "human":
            print(f"\nStep: {self.current_step}")
            print(f"Function: {self.function_name}")
            print("\nCurrent Code:")
            print(self.state.code)
            print("\nPerformance Metrics:")
            for metric, value in self.state.performance_metrics.items():
                print(f"{metric}: {value}")
            print("\nOptimization History:")
            for entry in self.state.optimization_history:
                print(f"- {entry['transformation']['description']}")
                print(f"  Improvement: {entry['improvement']}") 