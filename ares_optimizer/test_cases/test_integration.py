"""
Integration tests for the full optimization pipeline.
"""

import pytest
import numpy as np
import torch
from ares_optimizer.trainer import AresTrainer
from ares_optimizer.ares_env.code_executor import CodeExecutor
from ares_optimizer.ares_env.test_manager import TestManager
from ares_optimizer.rl_agent.agent import DQNAgent
import logging
import os
import tempfile
import shutil

@pytest.fixture
def temp_dir():
    """Create a temporary directory for test artifacts."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)

@pytest.fixture
def sample_code():
    """Sample code for testing."""
    return """
def sum_list(numbers):
    total = 0
    for num in numbers:
        total += num
    return total
    """

@pytest.fixture
def sample_test_cases():
    """Sample test cases for the code."""
    return {
        "test_cases": [
            ([1, 2, 3, 4, 5], 15),
            ([10, 20, 30], 60),
            ([0, 0, 0], 0),
            ([-1, -2, -3], -6)
        ]
    }

@pytest.fixture
def trainer(temp_dir):
    """Create a trainer instance for testing."""
    code_executor = CodeExecutor()
    test_manager = TestManager()
    return AresTrainer(
        code_executor=code_executor,
        test_manager=test_manager,
        device='cpu',
        save_dir=temp_dir
    )

def test_full_optimization_pipeline(trainer, sample_code, sample_test_cases):
    """Test the complete optimization pipeline."""
    # Train the agent
    trainer.train(
        code_samples=[(sample_code, sample_test_cases)],
        num_episodes=10,
        save_freq=5,
        verbose=False
    )

    # Optimize the code
    optimized_code = trainer.optimize_code(
        sample_code,
        sample_test_cases,
        max_steps=5,
        verbose=False
    )

    # Verify the optimized code
    assert optimized_code != sample_code, "Code should be transformed"
    assert "sum" in optimized_code, "Should use built-in sum function"

    # Verify correctness
    function_name = "sum_list"
    test_inputs = [tc[0] for tc in sample_test_cases["test_cases"]]
    test_outputs = [tc[1] for tc in sample_test_cases["test_cases"]]
    
    for inputs, expected in zip(test_inputs, test_outputs):
        result = trainer.code_executor.execute_function(
            optimized_code,
            function_name,
            [inputs]
        )
        assert result[2] is None, "No errors should occur"
        assert abs(result[0] - expected) < 1e-6, "Results should match"

def test_optimization_with_multiple_transformations(trainer, sample_code, sample_test_cases):
    """Test optimization with multiple transformations."""
    # Create a more complex code sample
    complex_code = """
def process_list(numbers):
    result = []
    for num in numbers:
        if num > 0:
            result.append(num * 2)
    return sum(result)
    """
    
    complex_test_cases = {
        "test_cases": [
            ([1, 2, 3, 4, 5], 30),
            ([-1, 2, -3, 4], 12),
            ([0, 0, 0], 0)
        ]
    }

    # Train and optimize
    trainer.train(
        code_samples=[(complex_code, complex_test_cases)],
        num_episodes=10,
        save_freq=5,
        verbose=False
    )

    optimized_code = trainer.optimize_code(
        complex_code,
        complex_test_cases,
        max_steps=5,
        verbose=False
    )

    # Verify transformations
    assert optimized_code != complex_code
    assert "sum" in optimized_code
    assert "if" not in optimized_code or "for" not in optimized_code

def test_optimization_with_error_handling(trainer):
    """Test optimization with error handling."""
    # Code with potential errors
    error_code = """
def divide_list(numbers, divisor):
    result = []
    for num in numbers:
        result.append(num / divisor)
    return result
    """
    
    error_test_cases = {
        "test_cases": [
            ([1, 2, 3], 2),
            ([10, 20, 30], 5),
            ([1, 2, 3], 0)  # Division by zero
        ]
    }

    # Train and optimize
    trainer.train(
        code_samples=[(error_code, error_test_cases)],
        num_episodes=10,
        save_freq=5,
        verbose=False
    )

    optimized_code = trainer.optimize_code(
        error_code,
        error_test_cases,
        max_steps=5,
        verbose=False
    )

    # Verify error handling
    assert "try" in optimized_code or "if" in optimized_code
    assert "ZeroDivisionError" in optimized_code

def test_optimization_with_performance_improvement(trainer):
    """Test optimization with performance improvement."""
    # Code with performance issues
    slow_code = """
def find_duplicates(numbers):
    result = []
    for i in range(len(numbers)):
        for j in range(i + 1, len(numbers)):
            if numbers[i] == numbers[j]:
                result.append(numbers[i])
    return result
    """
    
    perf_test_cases = {
        "test_cases": [
            ([1, 2, 3, 4, 5], []),
            ([1, 2, 2, 3, 3], [2, 3]),
            ([1, 1, 1, 1], [1, 1, 1])
        ]
    }

    # Train and optimize
    trainer.train(
        code_samples=[(slow_code, perf_test_cases)],
        num_episodes=10,
        save_freq=5,
        verbose=False
    )

    optimized_code = trainer.optimize_code(
        slow_code,
        perf_test_cases,
        max_steps=5,
        verbose=False
    )

    # Verify performance improvement
    assert "set" in optimized_code or "Counter" in optimized_code
    assert "for" not in optimized_code or "range" not in optimized_code

def test_optimization_with_memory_improvement(trainer):
    """Test optimization with memory improvement."""
    # Code with memory issues
    memory_code = """
def process_large_list(numbers):
    result = []
    for num in numbers:
        if num > 0:
            result.append(num * 2)
    return result
    """
    
    memory_test_cases = {
        "test_cases": [
            ([1, 2, 3, 4, 5], [2, 4, 6, 8, 10]),
            ([10, 20, 30], [20, 40, 60]),
            ([0, 0, 0], [])
        ]
    }

    # Train and optimize
    trainer.train(
        code_samples=[(memory_code, memory_test_cases)],
        num_episodes=10,
        save_freq=5,
        verbose=False
    )

    optimized_code = trainer.optimize_code(
        memory_code,
        memory_test_cases,
        max_steps=5,
        verbose=False
    )

    # Verify memory improvement
    assert "yield" in optimized_code or "generator" in optimized_code
    assert "append" not in optimized_code 