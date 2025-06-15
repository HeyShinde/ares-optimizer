"""State representation module for Ares Optimizer.

This module handles the representation of the optimization state for the RL agent,
including code state, performance metrics, and optimization history.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .feature_extractor import FeatureExtractor


@dataclass
class OptimizationState:
    """Represents the current state of the optimization process."""

    code: str
    function_name: str
    performance_metrics: Dict[str, float]
    optimization_history: List[Dict[str, Any]]
    feature_vector: Optional[np.ndarray] = None

    def __post_init__(self):
        """Initialize the feature vector after object creation."""
        if self.feature_vector is None:
            self.feature_vector = self._compute_feature_vector()

    def _compute_feature_vector(self) -> np.ndarray:
        """Compute the feature vector for the current state.

        Returns:
            Feature vector as numpy array.
        """
        extractor = FeatureExtractor()
        code_features = extractor.extract_code_features(self.code, self.function_name)
        perf_features = extractor.extract_performance_features(self.performance_metrics)
        return extractor.combine_features(code_features, perf_features)


class StateRepresentation:
    """Manages the state representation for the RL environment."""

    def __init__(self, max_history_length: int = 10):
        """Initialize the state representation manager.

        Args:
            max_history_length: Maximum number of optimization steps to keep in history.
        """
        self.max_history_length = max_history_length
        self.feature_extractor = FeatureExtractor()

    def create_initial_state(
        self, code: str, function_name: str, initial_metrics: Dict[str, float]
    ) -> OptimizationState:
        """Create the initial state for optimization.

        Args:
            code: The initial Python code.
            function_name: Name of the function to optimize.
            initial_metrics: Initial performance metrics.

        Returns:
            Initial optimization state.
        """
        return OptimizationState(
            code=code,
            function_name=function_name,
            performance_metrics=initial_metrics,
            optimization_history=[],
        )

    def update_state(
        self,
        current_state: OptimizationState,
        new_code: str,
        new_metrics: Dict[str, float],
        transformation_applied: Dict[str, Any],
    ) -> OptimizationState:
        """Update the state with new optimization results.

        Args:
            current_state: Current optimization state.
            new_code: New code after transformation.
            new_metrics: New performance metrics.
            transformation_applied: Details of the applied transformation.

        Returns:
            Updated optimization state.
        """
        # Create new history entry
        history_entry = {
            "transformation": transformation_applied,
            "old_metrics": current_state.performance_metrics,
            "new_metrics": new_metrics,
            "improvement": self._calculate_improvement(
                current_state.performance_metrics, new_metrics
            ),
        }

        # Update history
        new_history = current_state.optimization_history + [history_entry]
        if len(new_history) > self.max_history_length:
            new_history = new_history[-self.max_history_length :]

        # Create new state
        return OptimizationState(
            code=new_code,
            function_name=current_state.function_name,
            performance_metrics=new_metrics,
            optimization_history=new_history,
        )

    def get_state_vector(self, state: OptimizationState) -> np.ndarray:
        """Get the complete state vector for the RL agent.

        Args:
            state: Current optimization state.

        Returns:
            Complete state vector as numpy array.
        """
        # Get current state features
        current_features = state.feature_vector

        # Get history features
        history_features = self._get_history_features(state.optimization_history)

        # Combine features
        return np.concatenate([current_features, history_features])

    def _calculate_improvement(
        self, old_metrics: Dict[str, float], new_metrics: Dict[str, float]
    ) -> Dict[str, float]:
        """Calculate improvement in performance metrics.

        Args:
            old_metrics: Previous performance metrics.
            new_metrics: New performance metrics.

        Returns:
            Dictionary of improvement values.
        """
        improvement = {}
        for metric in old_metrics:
            if metric in new_metrics:
                old_val = old_metrics[metric]
                new_val = new_metrics[metric]
                if old_val != 0:
                    improvement[metric] = (old_val - new_val) / old_val
                else:
                    improvement[metric] = 0.0
        return improvement

    def _get_history_features(self, history: List[Dict[str, Any]]) -> np.ndarray:
        """Extract features from optimization history.

        Args:
            history: List of optimization history entries.

        Returns:
            History features as numpy array.
        """
        if not history:
            return np.zeros(5)  # Default features when no history

        # Calculate aggregate statistics
        improvements = [entry["improvement"] for entry in history]
        avg_improvement = np.mean([imp.get("runtime", 0.0) for imp in improvements])
        max_improvement = np.max([imp.get("runtime", 0.0) for imp in improvements])
        success_rate = np.mean([imp.get("runtime", 0.0) > 0 for imp in improvements])
        transformation_types = [entry["transformation"]["type"] for entry in history]
        unique_transformations = len(set(transformation_types))

        return np.array(
            [
                avg_improvement,
                max_improvement,
                success_rate,
                unique_transformations,
                len(history) / self.max_history_length,
            ],
            dtype=np.float32,
        )

    def get_state_size(self) -> int:
        """Get the size of the state vector.

        Returns:
            Size of the state vector.
        """
        # Create a dummy state to get the feature vector size
        dummy_state = OptimizationState(
            code="def dummy(): pass",
            function_name="dummy",
            performance_metrics={"runtime": 0.0, "memory_usage": 0.0},
            optimization_history=[],
        )
        return len(self.get_state_vector(dummy_state)) 