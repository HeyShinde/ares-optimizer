"""
Reward calculation module for Ares optimizer.
Calculates rewards based on performance improvements and correctness of transformed code.
"""

import numpy as np
from typing import Dict, Tuple, Optional
from ares_optimizer.ares_env.code_executor import CodeExecutor
from ares_optimizer.ares_env.test_manager import TestManager
import re


class RewardCalculator:
    def __init__(
        self,
        code_executor: CodeExecutor,
        test_manager: TestManager,
        performance_weight: float = 0.7,
        correctness_weight: float = 0.3,
        min_performance_improvement: float = 0.01,  # 1% minimum improvement
        max_performance_improvement: float = 0.5,   # 50% maximum improvement
    ):
        """
        Initialize the reward calculator.

        Args:
            code_executor: CodeExecutor instance for measuring performance
            test_manager: TestManager instance for verifying correctness
            performance_weight: Weight for performance improvement in reward calculation
            correctness_weight: Weight for correctness in reward calculation
            min_performance_improvement: Minimum improvement threshold for positive reward
            max_performance_improvement: Maximum improvement threshold for reward scaling
        """
        self.code_executor = code_executor
        self.test_manager = test_manager
        self.performance_weight = performance_weight
        self.correctness_weight = correctness_weight
        self.min_performance_improvement = min_performance_improvement
        self.max_performance_improvement = max_performance_improvement

    def _extract_function_name(self, code: str) -> str:
        match = re.search(r'def\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\(', code)
        if match:
            return match.group(1)
        raise ValueError("No function definition found in code.")

    def calculate_reward(
        self,
        original_code: str,
        transformed_code: str,
        test_cases: Optional[Dict] = None
    ) -> Tuple[float, Dict]:
        """
        Calculate the reward for a code transformation.

        Args:
            original_code: Original Python code
            transformed_code: Transformed Python code
            test_cases: Optional test cases to use for correctness verification

        Returns:
            Tuple of (reward, metrics) where metrics contains:
            - performance_improvement: Relative improvement in runtime
            - memory_improvement: Relative improvement in memory usage
            - is_correct: Whether the transformed code passes all tests
            - performance_reward: Reward component from performance
            - correctness_reward: Reward component from correctness
        """
        # Extract function name and test inputs
        function_name = self._extract_function_name(original_code)
        test_inputs = test_cases['test_cases']
        # Measure performance of original and transformed code
        original_metrics = self.code_executor.execute_function(original_code, function_name, test_inputs)
        function_name_trans = self._extract_function_name(transformed_code)
        transformed_metrics = self.code_executor.execute_function(transformed_code, function_name_trans, test_inputs)
        orig_runtime, orig_memory, orig_error = original_metrics
        trans_runtime, trans_memory, trans_error = transformed_metrics

        # Calculate performance improvements
        runtime_improvement = self._calculate_improvement(
            orig_runtime,
            trans_runtime
        )
        memory_improvement = self._calculate_improvement(
            orig_memory,
            trans_memory
        )

        # Verify correctness
        test_cases_list = test_cases['test_cases'] if isinstance(test_cases, dict) and 'test_cases' in test_cases else test_cases
        is_correct = self.test_manager.validate_function_output(
            transformed_code,
            function_name_trans,
            test_cases_list
        )

        # Calculate reward components
        performance_reward = self._calculate_performance_reward(
            runtime_improvement,
            memory_improvement
        )
        correctness_reward = 1.0 if is_correct else -1.0

        # Calculate final reward
        # If the transformation is incorrect, the reward should be negative
        # regardless of performance improvement
        if not is_correct:
            reward = -1.0
        else:
            reward = (
                self.performance_weight * performance_reward +
                self.correctness_weight * correctness_reward
            )

        metrics = {
            "runtime_improvement": runtime_improvement,
            "memory_improvement": memory_improvement,
            "is_correct": is_correct,
            "performance_reward": performance_reward,
            "correctness_reward": correctness_reward,
            "original_metrics": original_metrics,
            "transformed_metrics": transformed_metrics
        }

        return reward, metrics

    def _calculate_improvement(self, original: float, transformed: float) -> float:
        """
        Calculate relative improvement between original and transformed values.
        Positive values indicate improvement (transformed is better).
        """
        if original == 0:
            return 0.0
        return (original - transformed) / original

    def _calculate_performance_reward(
        self,
        runtime_improvement: float,
        memory_improvement: float
    ) -> float:
        """
        Calculate performance reward based on runtime and memory improvements.
        """
        # Combine runtime and memory improvements (weighted average)
        combined_improvement = 0.7 * runtime_improvement + 0.3 * memory_improvement

        # Apply thresholds and scaling
        if combined_improvement < self.min_performance_improvement:
            return -0.5  # Small penalty for no significant improvement
        elif combined_improvement > self.max_performance_improvement:
            return 1.0  # Maximum reward for large improvements
        else:
            # Scale reward between -0.5 and 1.0 based on improvement
            return -0.5 + 1.5 * (
                (combined_improvement - self.min_performance_improvement) /
                (self.max_performance_improvement - self.min_performance_improvement)
            ) 