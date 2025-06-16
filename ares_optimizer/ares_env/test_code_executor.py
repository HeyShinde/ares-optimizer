"""Tests for the CodeExecutor module."""

import pytest

from ares_optimizer.ares_env.code_executor import CodeExecutor


def test_successful_execution():
    """Test successful execution of a simple function."""
    code = """
def add(a, b):
    return a + b
"""
    executor = CodeExecutor()
    runtime, memory, error = executor.execute_function(code, "add", (1, 2))
    
    assert error is None
    assert runtime > 0
    assert memory >= 0


def test_syntax_error():
    """Test handling of syntax errors."""
    code = """
def add(a, b)
    return a + b
"""
    executor = CodeExecutor()
    runtime, memory, error = executor.execute_function(code, "add", (1, 2))
    
    assert error is not None
    assert "Syntax error" in error
    assert runtime == 0
    assert memory == 0


def test_runtime_error():
    """Test handling of runtime errors."""
    code = """
def divide(a, b):
    return a / b
"""
    executor = CodeExecutor()
    runtime, memory, error = executor.execute_function(code, "divide", (1, 0))
    
    assert error is not None
    assert "division by zero" in error
    assert runtime == 0
    assert memory == 0


def test_timeout():
    """Test handling of timeout."""
    code = """
def infinite_loop():
    while True:
        pass
"""
    executor = CodeExecutor(timeout_seconds=1)
    runtime, memory, error = executor.execute_function(code, "infinite_loop", ())
    
    assert error == "Execution timed out"
    assert runtime == 0
    assert memory == 0


def test_memory_measurement():
    """Test memory usage measurement."""
    code = """
def create_large_list():
    return [0] * 1000000
"""
    executor = CodeExecutor()
    runtime, memory, error = executor.execute_function(code, "create_large_list", ())
    
    assert error is None
    assert runtime > 0
    assert memory > 0  # Should use some memory for the large list 