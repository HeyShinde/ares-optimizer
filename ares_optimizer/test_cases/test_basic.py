"""
Basic tests for Ares optimizer setup.
"""

import numpy as np
import pytest
from gymnasium import spaces

from ares_optimizer.ares_env.ares_env import AresEnv
from ares_optimizer.functions_to_optimize.test_functions import (
    get_test_function_spaces,
    sphere,
)
from ares_optimizer.optimizer import AresOptimizer


def test_environment_initialization():
    """Test that the environment initializes correctly."""
    input_space, output_space = get_test_function_spaces("sphere", dim=2)
    env = AresEnv(
        target_function=sphere,
        input_space=input_space,
        output_space=output_space,
        max_steps=100,
    )

    # Test reset
    obs, info = env.reset()
    assert isinstance(obs, dict)
    assert "current_input" in obs
    assert "current_output" in obs
    assert "step" in obs
    assert obs["step"] == 0

    # Test step
    action = np.array([1.0, 1.0], dtype=np.float32)
    obs, reward, terminated, truncated, info = env.step(action)
    assert isinstance(obs, dict)
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert isinstance(info, dict)


def test_optimizer_initialization():
    """Test that the optimizer initializes correctly."""
    input_space, output_space = get_test_function_spaces("sphere", dim=2)
    optimizer = AresOptimizer(
        target_function=sphere,
        input_space=input_space,
        output_space=output_space,
        max_steps=100,
    )

    assert optimizer.target_function == sphere
    assert optimizer.max_steps == 100
    assert optimizer.current_step == 0
    assert optimizer.best_reward == float("-inf")
    assert optimizer.best_params is None
    assert isinstance(optimizer.optimization_history, list)


def test_optimizer_not_implemented():
    """Test that optimize() raises NotImplementedError."""
    input_space, output_space = get_test_function_spaces("sphere", dim=2)
    optimizer = AresOptimizer(
        target_function=sphere,
        input_space=input_space,
        output_space=output_space,
    )

    with pytest.raises(NotImplementedError):
        optimizer.optimize() 