"""Feature extraction module for Ares Optimizer.

This module handles the extraction of features from code and performance metrics
to represent the state of the optimization process.
"""

import ast
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from ..code_transformer.ast_manipulator import ASTManipulator


class FeatureExtractor:
    """Extracts features from code and performance metrics for state representation."""

    def __init__(self):
        """Initialize the feature extractor."""
        self.ast_manipulator = ASTManipulator()

    def extract_code_features(self, code: str, function_name: str) -> Dict[str, float]:
        """Extract features from the code structure.

        Args:
            code: The Python code to analyze.
            function_name: Name of the function to analyze.

        Returns:
            Dictionary of code features.
        """
        tree = self.ast_manipulator.parse_code(code)
        if not tree:
            return {}

        func_def = self.ast_manipulator.get_function_by_name(tree, function_name)
        if not func_def:
            return {}

        features = {
            "cyclomatic_complexity": self._calculate_cyclomatic_complexity(func_def),
            "num_loops": self._count_loops(func_def),
            "num_conditionals": self._count_conditionals(func_def),
            "num_function_calls": self._count_function_calls(func_def),
            "num_variables": len(self._get_variables(func_def)),
            "num_parameters": len(func_def.args.args),
            "has_recursion": self._has_recursion(func_def),
            "has_list_operations": self._has_list_operations(func_def),
            "has_dict_operations": self._has_dict_operations(func_def),
            "has_set_operations": self._has_set_operations(func_def),
            "has_numpy_operations": self._has_numpy_operations(func_def),
            "has_generator_expression": self._has_generator_expression(func_def),
            "has_list_comprehension": self._has_list_comprehension(func_def),
            "has_decorators": len(func_def.decorator_list) > 0,
            "has_type_hints": self._has_type_hints(func_def),
            "code_length": len(code.splitlines()),
        }

        return features

    def extract_performance_features(self, metrics: Dict[str, float]) -> Dict[str, float]:
        """Extract features from performance metrics.

        Args:
            metrics: Dictionary of performance metrics.

        Returns:
            Dictionary of performance features.
        """
        features = {
            "runtime": metrics.get("runtime", 0.0),
            "memory_usage": metrics.get("memory_usage", 0.0),
            "cpu_usage": metrics.get("cpu_usage", 0.0),
            "io_operations": metrics.get("io_operations", 0.0),
            "cache_hits": metrics.get("cache_hits", 0.0),
            "cache_misses": metrics.get("cache_misses", 0.0),
        }

        # Calculate derived features
        if features["cache_hits"] + features["cache_misses"] > 0:
            features["cache_hit_ratio"] = (
                features["cache_hits"] / (features["cache_hits"] + features["cache_misses"])
            )
        else:
            features["cache_hit_ratio"] = 0.0

        return features

    def combine_features(
        self, code_features: Dict[str, float], perf_features: Dict[str, float]
    ) -> np.ndarray:
        """Combine code and performance features into a single feature vector.

        Args:
            code_features: Dictionary of code features.
            perf_features: Dictionary of performance features.

        Returns:
            Combined feature vector as numpy array.
        """
        # Combine all features
        all_features = {**code_features, **perf_features}

        # Convert to numpy array
        feature_vector = np.array(list(all_features.values()), dtype=np.float32)

        # Normalize features
        feature_vector = self._normalize_features(feature_vector)

        return feature_vector

    def _calculate_cyclomatic_complexity(self, node: ast.AST) -> int:
        """Calculate cyclomatic complexity of a function.

        Args:
            node: The AST node to analyze.

        Returns:
            Cyclomatic complexity score.
        """
        complexity = 1  # Base complexity

        for child in ast.walk(node):
            if isinstance(child, (ast.If, ast.While, ast.For, ast.FunctionDef)):
                complexity += 1
            elif isinstance(child, ast.BoolOp):
                complexity += len(child.values) - 1
            elif isinstance(child, ast.Try):
                complexity += len(child.handlers)

        return complexity

    def _count_loops(self, node: ast.AST) -> int:
        """Count the number of loops in the code.

        Args:
            node: The AST node to analyze.

        Returns:
            Number of loops.
        """
        return sum(1 for child in ast.walk(node) if isinstance(child, (ast.For, ast.While)))

    def _count_conditionals(self, node: ast.AST) -> int:
        """Count the number of conditional statements.

        Args:
            node: The AST node to analyze.

        Returns:
            Number of conditionals.
        """
        return sum(1 for child in ast.walk(node) if isinstance(child, ast.If))

    def _count_function_calls(self, node: ast.AST) -> int:
        """Count the number of function calls.

        Args:
            node: The AST node to analyze.

        Returns:
            Number of function calls.
        """
        return sum(1 for child in ast.walk(node) if isinstance(child, ast.Call))

    def _get_variables(self, node: ast.AST) -> set:
        """Get all variables used in the code.

        Args:
            node: The AST node to analyze.

        Returns:
            Set of variable names.
        """
        variables = set()
        for child in ast.walk(node):
            if isinstance(child, ast.Name):
                variables.add(child.id)
        return variables

    def _has_recursion(self, node: ast.AST) -> bool:
        """Check if the function contains recursion.

        Args:
            node: The AST node to analyze.

        Returns:
            True if recursion is present, False otherwise.
        """
        if not isinstance(node, ast.FunctionDef):
            return False

        function_name = node.name
        for child in ast.walk(node):
            if isinstance(child, ast.Call):
                if isinstance(child.func, ast.Name) and child.func.id == function_name:
                    return True
        return False

    def _has_list_operations(self, node: ast.AST) -> bool:
        """Check if the code contains list operations.

        Args:
            node: The AST node to analyze.

        Returns:
            True if list operations are present, False otherwise.
        """
        for child in ast.walk(node):
            if isinstance(child, ast.List):
                return True
            if isinstance(child, ast.Call):
                if isinstance(child.func, ast.Name) and child.func.id in ("list", "append", "extend"):
                    return True
        return False

    def _has_dict_operations(self, node: ast.AST) -> bool:
        """Check if the code contains dictionary operations.

        Args:
            node: The AST node to analyze.

        Returns:
            True if dictionary operations are present, False otherwise.
        """
        for child in ast.walk(node):
            if isinstance(child, ast.Dict):
                return True
            if isinstance(child, ast.Call):
                if isinstance(child.func, ast.Name) and child.func.id in ("dict", "get", "update"):
                    return True
        return False

    def _has_set_operations(self, node: ast.AST) -> bool:
        """Check if the code contains set operations.

        Args:
            node: The AST node to analyze.

        Returns:
            True if set operations are present, False otherwise.
        """
        for child in ast.walk(node):
            if isinstance(child, ast.Set):
                return True
            if isinstance(child, ast.Call):
                if isinstance(child.func, ast.Name) and child.func.id in ("set", "add", "remove"):
                    return True
        return False

    def _has_numpy_operations(self, node: ast.AST) -> bool:
        """Check if the code contains NumPy operations.

        Args:
            node: The AST node to analyze.

        Returns:
            True if NumPy operations are present, False otherwise.
        """
        for child in ast.walk(node):
            if isinstance(child, ast.Call):
                if isinstance(child.func, ast.Attribute):
                    if isinstance(child.func.value, ast.Name) and child.func.value.id == "np":
                        return True
        return False

    def _has_generator_expression(self, node: ast.AST) -> bool:
        """Check if the code contains generator expressions.

        Args:
            node: The AST node to analyze.

        Returns:
            True if generator expressions are present, False otherwise.
        """
        return any(isinstance(child, ast.GeneratorExp) for child in ast.walk(node))

    def _has_list_comprehension(self, node: ast.AST) -> bool:
        """Check if the code contains list comprehensions.

        Args:
            node: The AST node to analyze.

        Returns:
            True if list comprehensions are present, False otherwise.
        """
        return any(isinstance(child, ast.ListComp) for child in ast.walk(node))

    def _has_type_hints(self, node: ast.AST) -> bool:
        """Check if the code contains type hints.

        Args:
            node: The AST node to analyze.

        Returns:
            True if type hints are present, False otherwise.
        """
        if not isinstance(node, ast.FunctionDef):
            return False

        # Check return type annotation
        if node.returns is not None:
            return True

        # Check argument type annotations
        for arg in node.args.args:
            if arg.annotation is not None:
                return True

        return False

    def _normalize_features(self, features: np.ndarray) -> np.ndarray:
        """Normalize feature values to [0, 1] range.

        Args:
            features: Feature vector to normalize.

        Returns:
            Normalized feature vector.
        """
        # Handle zero division
        max_val = np.max(features)
        if max_val == 0:
            return features

        return features / max_val 