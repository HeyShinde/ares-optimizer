"""
Integration tests for CodeExecutor and TestManager.
"""

import pytest
from ares_optimizer.ares_env.code_executor import CodeExecutor
from ares_optimizer.ares_env.test_manager import TestManager

# Test function and its test cases
TEST_FUNCTION = """
def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)
"""

TEST_CASES = [
    (0, 0),
    (1, 1),
    (2, 1),
    (3, 2),
    (4, 3),
    (5, 5),
    (6, 8),
]

def test_code_executor():
    """Test that CodeExecutor correctly measures performance."""
    executor = CodeExecutor(timeout_seconds=5)
    runtime, memory, error = executor.execute_function(
        TEST_FUNCTION,
        "fibonacci",
        test_inputs=(5,),
        num_runs=3,
    )
    assert error is None, f"Execution failed with error: {error}"
    assert runtime > 0, "Runtime should be positive"
    assert memory >= 0, "Memory usage should be non-negative"

def test_test_manager():
    """Test that TestManager correctly validates function output."""
    manager = TestManager(test_cases_dir="ares_optimizer/test_cases")
    is_correct, error = manager.validate_function_output(
        TEST_FUNCTION,
        "fibonacci",
        TEST_CASES,
    )
    assert is_correct, f"Correct implementation failed: {error}"
    # Test with incorrect implementation
    incorrect_function = """
def fibonacci(n):
    return n  # Incorrect implementation
"""
    is_correct, error = manager.validate_function_output(
        incorrect_function,
        "fibonacci",
        TEST_CASES,
    )
    assert not is_correct, "Incorrect implementation should fail validation"
    assert error is not None, "Should provide error details for incorrect implementation"

def test_integration():
    """Test that CodeExecutor and TestManager work together."""
    executor = CodeExecutor(timeout_seconds=5)
    manager = TestManager(test_cases_dir="ares_optimizer/test_cases")
    is_correct, error = manager.validate_function_output(
        TEST_FUNCTION,
        "fibonacci",
        TEST_CASES,
    )
    assert is_correct, f"Function validation failed: {error}"
    runtime, memory, error = executor.execute_function(
        TEST_FUNCTION,
        "fibonacci",
        test_inputs=(5,),
        num_runs=3,
    )
    assert error is None, f"Performance measurement failed: {error}"
    assert runtime > 0, "Runtime should be positive"
    assert memory >= 0, "Memory usage should be non-negative" 