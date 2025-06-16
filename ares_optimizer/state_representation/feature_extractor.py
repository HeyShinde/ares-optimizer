"""
Feature extraction module for Ares - converts code and performance metrics into numerical state vectors.
"""

import ast
from typing import Dict, List, Tuple, Union

import numpy as np


class FeatureExtractor:
    """
    Extracts numerical features from code and performance metrics for RL state representation.
    """

    def __init__(self):
        """Initialize the feature extractor."""
        # Define feature names for better tracking
        self.ast_features = [
            "num_lines",
            "num_chars",
            "num_functions",
            "num_loops",
            "num_conditionals",
            "num_assignments",
            "num_returns",
            "max_depth",
            "has_yield",
            "has_lambda",
            "has_list_comp",
            "has_dict_comp",
            "has_set_comp",
            "has_generator",
        ]

    def extract_ast_features(self, code_string: str) -> Dict[str, float]:
        """
        Extract features from the AST of the code.

        Args:
            code_string: The Python code to analyze.

        Returns:
            Dictionary of feature names and their values.
        """
        try:
            tree = ast.parse(code_string)
        except SyntaxError:
            return {feature: 0.0 for feature in self.ast_features}

        features = {
            "num_lines": len(code_string.splitlines()),
            "num_chars": len(code_string),
            "num_functions": len([node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]),
            "num_loops": len([node for node in ast.walk(tree) if isinstance(node, (ast.For, ast.While))]),
            "num_conditionals": len([node for node in ast.walk(tree) if isinstance(node, ast.If)]),
            "num_assignments": len([node for node in ast.walk(tree) if isinstance(node, ast.Assign)]),
            "num_returns": len([node for node in ast.walk(tree) if isinstance(node, ast.Return)]),
            "max_depth": self._calculate_max_depth(tree),
            "has_yield": float(any(isinstance(node, ast.Yield) for node in ast.walk(tree))),
            "has_lambda": float(any(isinstance(node, ast.Lambda) for node in ast.walk(tree))),
            "has_list_comp": float(any(isinstance(node, ast.ListComp) for node in ast.walk(tree))),
            "has_dict_comp": float(any(isinstance(node, ast.DictComp) for node in ast.walk(tree))),
            "has_set_comp": float(any(isinstance(node, ast.SetComp) for node in ast.walk(tree))),
            "has_generator": float(any(isinstance(node, ast.GeneratorExp) for node in ast.walk(tree))),
        }

        return features

    def extract_performance_features(
        self, runtime_ms: float, memory_mb: float
    ) -> Dict[str, float]:
        """
        Extract features from performance metrics.

        Args:
            runtime_ms: Runtime in milliseconds
            memory_mb: Memory usage in megabytes

        Returns:
            Dictionary of performance features
        """
        # Normalize performance metrics (using log scale for better distribution)
        features = {
            "log_runtime": np.log1p(runtime_ms),
            "log_memory": np.log1p(memory_mb),
            "runtime": runtime_ms,
            "memory": memory_mb,
        }
        return features

    def get_state_vector(
        self, code_string: str, runtime_ms: float, memory_mb: float
    ) -> np.ndarray:
        """
        Combine AST and performance features into a single state vector.

        Args:
            code_string: The Python code to analyze
            runtime_ms: Runtime in milliseconds
            memory_mb: Memory usage in megabytes

        Returns:
            NumPy array containing all features
        """
        ast_features = self.extract_ast_features(code_string)
        perf_features = self.extract_performance_features(runtime_ms, memory_mb)

        # Combine features in a consistent order
        feature_vector = []
        for feature in self.ast_features:
            feature_vector.append(ast_features[feature])
        for feature in ["log_runtime", "log_memory", "runtime", "memory"]:
            feature_vector.append(perf_features[feature])

        return np.array(feature_vector, dtype=np.float32)

    def _calculate_max_depth(self, tree: ast.AST) -> float:
        """
        Calculate the maximum depth of the AST.

        Args:
            tree: The AST to analyze

        Returns:
            Maximum depth of the AST
        """
        def _get_depth(node: ast.AST, current_depth: int = 0) -> int:
            if not isinstance(node, ast.AST):
                return current_depth
            max_depth = current_depth
            for child in ast.iter_child_nodes(node):
                max_depth = max(max_depth, _get_depth(child, current_depth + 1))
            return max_depth

        return float(_get_depth(tree)) 