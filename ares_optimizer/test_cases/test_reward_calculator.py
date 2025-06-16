"""
Tests for the reward calculator module.
"""

import pytest
from unittest.mock import Mock, patch
from ares_optimizer.reward_calculator import RewardCalculator
from ares_optimizer.ares_env.code_executor import CodeExecutor
from ares_optimizer.ares_env.test_manager import TestManager


@pytest.fixture
def mock_code_executor():
    """Create a mock CodeExecutor."""
    executor = Mock(spec=CodeExecutor)
    return executor


@pytest.fixture
def mock_test_manager():
    """Create a mock TestManager."""
    manager = Mock(spec=TestManager)
    return manager


@pytest.fixture
def reward_calculator(mock_code_executor, mock_test_manager):
    """Create a RewardCalculator instance with mock dependencies."""
    return RewardCalculator(
        code_executor=mock_code_executor,
        test_manager=mock_test_manager,
        performance_weight=0.7,
        correctness_weight=0.3
    )


def test_calculate_reward_improvement(reward_calculator, mock_code_executor, mock_test_manager):
    """Test reward calculation with performance improvement."""
    # Mock performance metrics
    mock_code_executor.execute_function.side_effect = [
        {"runtime": 1.0, "memory": 100.0},  # Original code
        {"runtime": 0.5, "memory": 80.0}    # Transformed code (50% faster, 20% less memory)
    ]
    mock_test_manager.validate_function_output.return_value = True

    original_code = "def add(a, b): return a + b"
    transformed_code = "def add(a, b): return a + b  # Optimized version"
    test_cases = {"test1": (1, 2, 3)}

    reward, metrics = reward_calculator.calculate_reward(
        original_code,
        transformed_code,
        test_cases
    )

    # Verify reward calculation
    assert reward > 0  # Should be positive due to improvement
    assert metrics["runtime_improvement"] == 0.5  # 50% faster
    assert metrics["memory_improvement"] == 0.2   # 20% less memory
    assert metrics["is_correct"] is True
    assert metrics["performance_reward"] > 0
    assert metrics["correctness_reward"] == 1.0


def test_calculate_reward_degradation(reward_calculator, mock_code_executor, mock_test_manager):
    """Test reward calculation with performance degradation."""
    # Mock performance metrics
    mock_code_executor.execute_function.side_effect = [
        {"runtime": 1.0, "memory": 100.0},  # Original code
        {"runtime": 1.5, "memory": 120.0}   # Transformed code (50% slower, 20% more memory)
    ]
    mock_test_manager.validate_function_output.return_value = True

    original_code = "def add(a, b): return a + b"
    transformed_code = "def add(a, b): return a + b  # Degraded version"
    test_cases = {"test1": (1, 2, 3)}

    reward, metrics = reward_calculator.calculate_reward(
        original_code,
        transformed_code,
        test_cases
    )

    # Verify reward calculation
    assert reward < 0  # Should be negative due to degradation
    assert metrics["runtime_improvement"] == -0.5  # 50% slower
    assert metrics["memory_improvement"] == -0.2   # 20% more memory
    assert metrics["is_correct"] is True
    assert metrics["performance_reward"] < 0
    assert metrics["correctness_reward"] == 1.0


def test_calculate_reward_incorrect(reward_calculator, mock_code_executor, mock_test_manager):
    """Test reward calculation with incorrect transformation."""
    # Mock performance metrics
    mock_code_executor.execute_function.side_effect = [
        {"runtime": 1.0, "memory": 100.0},  # Original code
        {"runtime": 0.5, "memory": 80.0}    # Transformed code (faster but incorrect)
    ]
    mock_test_manager.validate_function_output.return_value = False

    original_code = "def add(a, b): return a + b"
    transformed_code = "def add(a, b): return a - b  # Incorrect version"
    test_cases = {"test1": (1, 2, 3)}

    reward, metrics = reward_calculator.calculate_reward(
        original_code,
        transformed_code,
        test_cases
    )

    # Verify reward calculation
    assert reward < 0  # Should be negative due to incorrectness
    assert metrics["runtime_improvement"] == 0.5  # 50% faster
    assert metrics["memory_improvement"] == 0.2   # 20% less memory
    assert metrics["is_correct"] is False
    assert metrics["performance_reward"] > 0
    assert metrics["correctness_reward"] == -1.0


def test_calculate_reward_execution_error(reward_calculator, mock_code_executor, mock_test_manager):
    """Test reward calculation when code execution fails."""
    # Mock execution failure
    mock_code_executor.execute_function.return_value = None

    original_code = "def add(a, b): return a + b"
    transformed_code = "def add(a, b): return a + b  # Invalid syntax"
    test_cases = {"test1": (1, 2, 3)}

    reward, metrics = reward_calculator.calculate_reward(
        original_code,
        transformed_code,
        test_cases
    )

    # Verify reward calculation
    assert reward == -1.0  # Should be maximum penalty
    assert "error" in metrics


def test_calculate_reward_zero_original(reward_calculator, mock_code_executor, mock_test_manager):
    """Test reward calculation when original runtime is zero."""
    # Mock performance metrics with zero original runtime
    mock_code_executor.execute_function.side_effect = [
        {"runtime": 0.0, "memory": 100.0},  # Original code
        {"runtime": 0.0, "memory": 80.0}    # Transformed code
    ]
    mock_test_manager.validate_function_output.return_value = True

    original_code = "def add(a, b): return a + b"
    transformed_code = "def add(a, b): return a + b  # Same performance"
    test_cases = {"test1": (1, 2, 3)}

    reward, metrics = reward_calculator.calculate_reward(
        original_code,
        transformed_code,
        test_cases
    )

    # Verify reward calculation
    assert metrics["runtime_improvement"] == 0.0  # No improvement possible
    assert metrics["memory_improvement"] == 0.2   # 20% less memory
    assert metrics["is_correct"] is True
 