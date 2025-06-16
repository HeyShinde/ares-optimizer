"""
Test functions for optimization.
"""

import numpy as np
from gymnasium import spaces


def sphere(x: np.ndarray) -> float:
    """
    Sphere function: f(x) = sum(x^2)
    Minimum at x = [0, 0, ..., 0]
    """
    return float(np.sum(x**2))


def rosenbrock(x: np.ndarray) -> float:
    """
    Rosenbrock function: f(x) = sum(100*(x[i+1] - x[i]^2)^2 + (1-x[i])^2)
    Minimum at x = [1, 1, ..., 1]
    """
    return float(
        np.sum(100.0 * (x[1:] - x[:-1] ** 2.0) ** 2.0 + (1 - x[:-1]) ** 2.0)
    )


def get_test_function_spaces(
    function_name: str, dim: int = 2
) -> tuple[spaces.Space, spaces.Space]:
    """
    Get input and output spaces for test functions.

    Args:
        function_name: Name of the test function ('sphere' or 'rosenbrock')
        dim: Dimension of the input space

    Returns:
        Tuple of (input_space, output_space)
    """
    if function_name == "sphere":
        input_space = spaces.Box(
            low=-5.12, high=5.12, shape=(dim,), dtype=np.float32
        )
        output_space = spaces.Box(
            low=0.0, high=float("inf"), shape=(), dtype=np.float32
        )
    elif function_name == "rosenbrock":
        input_space = spaces.Box(
            low=-2.048, high=2.048, shape=(dim,), dtype=np.float32
        )
        output_space = spaces.Box(
            low=0.0, high=float("inf"), shape=(), dtype=np.float32
        )
    else:
        raise ValueError(f"Unknown test function: {function_name}")

    return input_space, output_space 