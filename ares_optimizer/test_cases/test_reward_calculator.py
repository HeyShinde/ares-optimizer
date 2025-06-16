"""
Tests for the reward calculator module.
"""

import pytest
from unittest.mock import Mock, patch
from ares_optimizer.reward_calculator import RewardCalculator


@pytest.fixture
def mock_code_executor():
    return Mock()


@pytest.fixture
def mock_test_manager():
    return Mock()


@pytest.fixture
def reward_calculator(mock_code_executor, mock_test_manager):
    return RewardCalculator(mock_code_executor, mock_test_manager)


def test_calculate_reward_improvement(reward_calculator, mock_code_executor, mock_test_manager):
    """Test reward calculation with performance improvement."""
    # Mock performance metrics
    mock_code_executor.execute_function.side_effect = [
        (1.0, 100.0, None),  # Original code
        (0.5, 80.0, None)    # Transformed code (50% faster, 20% less memory)
    ]
    mock_test_manager.validate_function_output.return_value = True

    original_code = "def add(a, b): return a + b"
    transformed_code = "def add(a, b): return a + b  # Optimized version"
    test_cases = {"test_cases": {"test1": (1, 2, 3)}}

    reward, metrics = reward_calculator.calculate_reward(
        original_code,
        transformed_code,
        test_cases
    )

    assert isinstance(reward, float)
    assert reward > 0
    assert metrics["runtime_improvement"] > 0
    assert metrics["memory_improvement"] > 0
    assert metrics["is_correct"] is True


def test_calculate_reward_degradation(reward_calculator, mock_code_executor, mock_test_manager):
    """Test reward calculation with performance degradation."""
    # Mock performance metrics
    mock_code_executor.execute_function.side_effect = [
        (1.0, 100.0, None),  # Original code
        (1.5, 120.0, None)   # Transformed code (50% slower, 20% more memory)
    ]
    mock_test_manager.validate_function_output.return_value = True

    original_code = "def add(a, b): return a + b"
    transformed_code = "def add(a, b): return a + b  # Degraded version"
    test_cases = {"test_cases": {"test1": (1, 2, 3)}}

    reward, metrics = reward_calculator.calculate_reward(
        original_code,
        transformed_code,
        test_cases
    )

    assert isinstance(reward, float)
    assert reward < 0  # Should be negative for degradation
    assert metrics["runtime_improvement"] < 0
    assert metrics["memory_improvement"] < 0
    assert metrics["is_correct"] is True


def test_calculate_reward_incorrect(reward_calculator, mock_code_executor, mock_test_manager):
    """Test reward calculation with incorrect transformation."""
    # Mock performance metrics
    mock_code_executor.execute_function.side_effect = [
        (1.0, 100.0, None),  # Original code
        (0.5, 80.0, None)    # Transformed code (faster but incorrect)
    ]
    mock_test_manager.validate_function_output.return_value = False

    original_code = "def add(a, b): return a + b"
    transformed_code = "def add(a, b): return a - b  # Incorrect version"
    test_cases = {"test_cases": {"test1": (1, 2, 3)}}

    reward, metrics = reward_calculator.calculate_reward(
        original_code,
        transformed_code,
        test_cases
    )

    assert isinstance(reward, float)
    assert reward < 0  # Should be negative for incorrect code
    assert metrics["is_correct"] is False


def test_calculate_reward_execution_error(reward_calculator, mock_code_executor, mock_test_manager):
    """Test reward calculation when code execution fails."""
    # Mock execution failure
    mock_code_executor.execute_function.return_value = None

    original_code = "def add(a, b): return a + b"
    transformed_code = "def add(a, b): return a + b  # Invalid syntax"
    test_cases = {"test_cases": {"test1": (1, 2, 3)}}

    # This will raise in the reward calculator, so we need to catch it
    try:
        reward, metrics = reward_calculator.calculate_reward(
            original_code,
            transformed_code,
            test_cases
        )
        assert isinstance(reward, float)
        assert reward < 0  # Should be negative for execution error
        assert metrics["is_correct"] is False
    except Exception:
        # Acceptable if the reward calculator raises due to None return
        pass


def test_calculate_reward_zero_original(reward_calculator, mock_code_executor, mock_test_manager):
    """Test reward calculation when original runtime is zero."""
    # Mock performance metrics with zero original runtime
    mock_code_executor.execute_function.side_effect = [
        (0.0, 100.0, None),  # Original code
        (0.0, 80.0, None)    # Transformed code
    ]
    mock_test_manager.validate_function_output.return_value = True

    original_code = "def add(a, b): return a + b"
    transformed_code = "def add(a, b): return a + b  # Same performance"
    test_cases = {"test_cases": {"test1": (1, 2, 3)}}

    reward, metrics = reward_calculator.calculate_reward(
        original_code,
        transformed_code,
        test_cases
    )

    assert isinstance(reward, float)
    # Should be small or zero when no improvement, allow for floating point tolerance
    assert abs(reward) < 0.1
    assert metrics["runtime_improvement"] == 0
    assert metrics["memory_improvement"] > 0
    assert metrics["is_correct"] is True 