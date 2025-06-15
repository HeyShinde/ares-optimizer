"""Test management module for Ares Optimizer.

This module handles the management and validation of test cases for code optimization,
ensuring that optimizations maintain code correctness.
"""

import ast
import importlib.util
import os
from typing import Any, Dict, List, Optional, Tuple, Union


class TestManager:
    """Manages test cases and validates function correctness."""

    def __init__(self, test_dir: str = "test_cases"):
        """Initialize the test manager.

        Args:
            test_dir: Directory containing test case files.
        """
        self.test_dir = test_dir
        self.test_cases = {}

    def load_test_cases(self, function_name: str, test_cases: List[Dict[str, Any]]):
        """Load test cases for a function.

        Args:
            function_name: Name of the function to test.
            test_cases: List of test case dictionaries.
        """
        self.test_cases[function_name] = []
        for test_case in test_cases:
            if not all(key in test_case for key in ["test_input", "expected_output"]):
                continue

            self.test_cases[function_name].append({
                "name": test_case.get("name", f"test_{len(self.test_cases[function_name])}"),
                "test_input": test_case["test_input"],
                "expected_output": test_case["expected_output"],
                "description": test_case.get("description", ""),
            })

    def validate_function(
        self, code: str, function_name: str
    ) -> Tuple[bool, List[Dict[str, Any]]]:
        """Validate a function against its test cases.

        Args:
            code: The code to validate.
            function_name: Name of the function to validate.

        Returns:
            Tuple of (all_tests_passed, test_results).
        """
        if function_name not in self.test_cases:
            return False, [{"error": "No test cases found"}]

        # Create a temporary module to execute the code
        spec = importlib.util.spec_from_loader(
            "temp_module",
            loader=importlib.machinery.SourceFileLoader("temp_module", "<string>"),
        )
        module = importlib.util.module_from_spec(spec)
        exec(code, module.__dict__)

        # Get the function to test
        function = getattr(module, function_name, None)
        if function is None:
            return False, [{"error": f"Function {function_name} not found"}]

        # Run test cases
        test_results = []
        all_passed = True

        for test_case in self.test_cases[function_name]:
            try:
                # Execute the function with test input
                actual_output = function(*test_case["test_input"])

                # Compare outputs
                is_correct = self._compare_outputs(
                    actual_output, test_case["expected_output"]
                )

                test_results.append({
                    "name": test_case["name"],
                    "passed": is_correct,
                    "expected": test_case["expected_output"],
                    "actual": actual_output,
                    "description": test_case["description"],
                })

                if not is_correct:
                    all_passed = False

            except Exception as e:
                test_results.append({
                    "name": test_case["name"],
                    "passed": False,
                    "error": str(e),
                    "description": test_case["description"],
                })
                all_passed = False

        return all_passed, test_results

    def _compare_outputs(self, actual: Any, expected: Any) -> bool:
        """Compare actual and expected outputs.

        Args:
            actual: Actual output from the function.
            expected: Expected output from the test case.

        Returns:
            True if outputs match, False otherwise.
        """
        # Handle None values
        if actual is None and expected is None:
            return True
        if actual is None or expected is None:
            return False

        # Handle numeric types with tolerance
        if isinstance(actual, (int, float)) and isinstance(expected, (int, float)):
            return abs(actual - expected) < 1e-10

        # Handle sequences
        if isinstance(actual, (list, tuple)) and isinstance(expected, (list, tuple)):
            if len(actual) != len(expected):
                return False
            return all(self._compare_outputs(a, e) for a, e in zip(actual, expected))

        # Handle dictionaries
        if isinstance(actual, dict) and isinstance(expected, dict):
            if actual.keys() != expected.keys():
                return False
            return all(
                self._compare_outputs(actual[key], expected[key])
                for key in actual.keys()
            )

        # Handle sets
        if isinstance(actual, set) and isinstance(expected, set):
            return actual == expected

        # Default comparison
        return actual == expected

    def get_test_coverage(self, function_name: str) -> float:
        """Calculate test coverage for a function.

        Args:
            function_name: Name of the function to analyze.

        Returns:
            Test coverage ratio (0.0 to 1.0).
        """
        if function_name not in self.test_cases:
            return 0.0

        # TODO: Implement actual test coverage analysis
        # This would require analyzing the function's code and test cases
        # to determine which lines/branches are covered by tests
        return 1.0

    def add_test_case(
        self, function_name: str, test_input: Any, expected_output: Any, **kwargs
    ):
        """Add a new test case.

        Args:
            function_name: Name of the function to test.
            test_input: Input arguments for the function.
            expected_output: Expected output from the function.
            **kwargs: Additional test case attributes.
        """
        if function_name not in self.test_cases:
            self.test_cases[function_name] = []

        test_case = {
            "name": kwargs.get("name", f"test_{len(self.test_cases[function_name])}"),
            "test_input": test_input,
            "expected_output": expected_output,
            "description": kwargs.get("description", ""),
        }

        self.test_cases[function_name].append(test_case)

    def remove_test_case(self, function_name: str, test_name: str) -> bool:
        """Remove a test case.

        Args:
            function_name: Name of the function.
            test_name: Name of the test case to remove.

        Returns:
            True if test case was removed, False otherwise.
        """
        if function_name not in self.test_cases:
            return False

        initial_length = len(self.test_cases[function_name])
        self.test_cases[function_name] = [
            test for test in self.test_cases[function_name]
            if test["name"] != test_name
        ]

        return len(self.test_cases[function_name]) < initial_length

    def get_test_cases(self, function_name: str) -> List[Dict[str, Any]]:
        """Get all test cases for a function.

        Args:
            function_name: Name of the function.

        Returns:
            List of test cases.
        """
        return self.test_cases.get(function_name, []) 