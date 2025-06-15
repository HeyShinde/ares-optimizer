"""Example script demonstrating Ares Optimizer with a Fibonacci function.

This script shows how to use Ares Optimizer to automatically optimize a recursive
Fibonacci function using reinforcement learning.
"""

import os
import sys
from typing import List, Dict, Any

# Add the parent directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ares_optimizer.ares_env.ares_env import AresEnv


def fibonacci(n: int) -> int:
    """Calculate the nth Fibonacci number recursively.

    Args:
        n: The position in the Fibonacci sequence.

    Returns:
        The nth Fibonacci number.
    """
    if n <= 1:
        return n
    return fibonacci(n - 1) + fibonacci(n - 2)


def main():
    """Run the optimization example."""
    # Initial code
    initial_code = """
def fibonacci(n: int) -> int:
    if n <= 1:
        return n
    return fibonacci(n - 1) + fibonacci(n - 2)
"""

    # Test cases
    test_cases = [
        {
            "name": "test_fibonacci_0",
            "test_input": (0,),
            "expected_output": 0,
            "description": "Test fibonacci(0)",
        },
        {
            "name": "test_fibonacci_1",
            "test_input": (1,),
            "expected_output": 1,
            "description": "Test fibonacci(1)",
        },
        {
            "name": "test_fibonacci_5",
            "test_input": (5,),
            "expected_output": 5,
            "description": "Test fibonacci(5)",
        },
        {
            "name": "test_fibonacci_10",
            "test_input": (10,),
            "expected_output": 55,
            "description": "Test fibonacci(10)",
        },
    ]

    # Create the environment
    env = AresEnv(
        initial_code=initial_code,
        function_name="fibonacci",
        test_cases=test_cases,
        max_steps=50,
    )

    # Run optimization
    state = env.reset()
    done = False
    total_reward = 0

    print("Starting optimization...")
    print("\nInitial code:")
    print(initial_code)

    while not done:
        # Choose a random action (in a real scenario, this would be from an RL agent)
        action = env.action_space.sample()
        
        # Take a step
        state, reward, done, info = env.step(action)
        total_reward += reward

        # Print progress
        if info.get("error"):
            print(f"\nStep failed: {info['error']}")
        else:
            print(f"\nStep {env.current_step}:")
            print(f"Applied transformation: {info['transformation']['description']}")
            print(f"Reward: {reward:.4f}")
            print("\nCurrent code:")
            print(env.state.code)
            print("\nPerformance metrics:")
            for metric, value in info["metrics"].items():
                print(f"{metric}: {value}")

    print("\nOptimization complete!")
    print(f"Total reward: {total_reward:.4f}")
    print("\nFinal code:")
    print(env.state.code)
    print("\nOptimization history:")
    for entry in env.state.optimization_history:
        print(f"- {entry['transformation']['description']}")
        print(f"  Improvement: {entry['improvement']}")


if __name__ == "__main__":
    main() 