"""Test case management and correctness validation for Ares Optimizer."""

import importlib.util
import os
import sys
import tempfile
from typing import Any, List, Tuple, Optional

class TestManager:
    """Manages test cases and validates function correctness."""

    def __init__(self, test_cases_dir: str = "test_cases"):
        self.test_cases_dir = test_cases_dir

    def load_test_cases(self, function_name: str) -> List[Tuple[Any, Any]]:
        """Load test cases for a given function from the test_cases directory.

        Args:
            function_name: Name of the function to load test cases for.
        Returns:
            List of (input, expected_output) tuples.
        """
        # Convention: test case files are named <function_name>_tests.py
        test_file = os.path.join(self.test_cases_dir, f"{function_name}_tests.py")
        if not os.path.exists(test_file):
            raise FileNotFoundError(f"Test case file not found: {test_file}")

        # Dynamically import the test case file
        spec = importlib.util.spec_from_file_location(f"{function_name}_tests", test_file)
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)

        # Expect a variable named TEST_CASES = [(input, expected_output), ...]
        if not hasattr(module, "TEST_CASES"):
            raise AttributeError(f"No TEST_CASES variable found in {test_file}")
        return getattr(module, "TEST_CASES")

    def validate_function_output(
        self,
        code_string: str,
        function_name: str,
        test_cases: List[Tuple[Any, Any]],
    ) -> Tuple[bool, Optional[str]]:
        """Validate that the function produces correct output for all test cases.

        Args:
            code_string: The Python code containing the function to test.
            function_name: Name of the function to test.
            test_cases: List of (input, expected_output) tuples.
        Returns:
            (is_correct, failed_test_details)
        """
        # Write the code to a temporary file and import it as a module
        with tempfile.TemporaryDirectory() as tmpdir:
            code_file = os.path.join(tmpdir, "user_func.py")
            with open(code_file, "w") as f:
                f.write(code_string)
            spec = importlib.util.spec_from_file_location("user_func", code_file)
            module = importlib.util.module_from_spec(spec)
            sys.modules[spec.name] = module
            try:
                spec.loader.exec_module(module)
            except Exception as e:
                return False, f"Code import error: {e}"

            if not hasattr(module, function_name):
                return False, f"Function '{function_name}' not found in code."
            func = getattr(module, function_name)

            for idx, (test_input, expected_output) in enumerate(test_cases):
                try:
                    # Support both single and tuple input
                    if isinstance(test_input, tuple):
                        result = func(*test_input)
                    else:
                        result = func(test_input)
                except Exception as e:
                    return False, f"Test case {idx+1} raised exception: {e} (input: {test_input})"
                if result != expected_output:
                    return (
                        False,
                        f"Test case {idx+1} failed: input={test_input}, expected={expected_output}, got={result}",
                    )
            return True, None 