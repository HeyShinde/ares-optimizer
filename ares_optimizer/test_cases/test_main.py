"""
Tests for the main Ares Optimizer orchestrator.
"""

import unittest
import tempfile
import shutil
import os
from pathlib import Path
import json
import torch

from ares_optimizer.main import AresOptimizer

class TestAresOptimizer(unittest.TestCase):
    def setUp(self):
        """Set up test environment."""
        # Create temporary directories
        self.test_dir = tempfile.mkdtemp()
        self.functions_dir = Path(self.test_dir) / "functions_to_optimize"
        self.checkpoint_dir = Path(self.test_dir) / "checkpoints"
        self.log_dir = Path(self.test_dir) / "logs"
        
        # Create directories
        self.functions_dir.mkdir()
        self.checkpoint_dir.mkdir()
        self.log_dir.mkdir()
        
        # Create test functions
        self._create_test_functions()
        
        # Initialize optimizer
        self.optimizer = AresOptimizer(
            functions_dir=str(self.functions_dir),
            checkpoint_dir=str(self.checkpoint_dir),
            log_dir=str(self.log_dir),
            num_episodes=10,  # Small number for testing
            max_steps_per_episode=3,
            save_freq=5,
            device="cpu"  # Use CPU for testing
        )

    def tearDown(self):
        """Clean up test environment."""
        shutil.rmtree(self.test_dir)

    def _create_test_functions(self):
        """Create test functions and their test cases."""
        # Test function 1: Simple list sum
        func1_code = """
def sum_list(numbers):
    total = 0
    for num in numbers:
        total += num
    return total
"""
        func1_tests = {
            "test_cases": [
                {"input": [1, 2, 3, 4, 5], "expected": 15},
                {"input": [0, 0, 0], "expected": 0},
                {"input": [-1, 1], "expected": 0}
            ]
        }
        
        # Test function 2: List comprehension
        func2_code = """
def square_numbers(numbers):
    result = []
    for num in numbers:
        result.append(num * num)
    return result
"""
        func2_tests = {
            "test_cases": [
                {"input": [1, 2, 3], "expected": [1, 4, 9]},
                {"input": [0], "expected": [0]},
                {"input": [-1, -2], "expected": [1, 4]}
            ]
        }
        
        # Save functions and test cases
        with open(self.functions_dir / "sum_list.py", "w") as f:
            f.write(func1_code)
        with open(self.functions_dir / "sum_list.json", "w") as f:
            json.dump(func1_tests, f)
            
        with open(self.functions_dir / "square_numbers.py", "w") as f:
            f.write(func2_code)
        with open(self.functions_dir / "square_numbers.json", "w") as f:
            json.dump(func2_tests, f)

    def test_initialization(self):
        """Test optimizer initialization."""
        self.assertTrue(self.functions_dir.exists())
        self.assertTrue(self.checkpoint_dir.exists())
        self.assertTrue(self.log_dir.exists())
        self.assertIsNotNone(self.optimizer)

    def test_load_functions(self):
        """Test function loading."""
        functions = self.optimizer.load_functions()
        self.assertEqual(len(functions), 2)
        
        # Check if both functions are loaded
        function_names = [self.optimizer._extract_function_name(code) 
                         for code, _ in functions]
        self.assertIn("sum_list", function_names)
        self.assertIn("square_numbers", function_names)

    def test_optimize_function(self):
        """Test function optimization."""
        # Load a test function
        functions = self.optimizer.load_functions()
        code, test_cases = functions[0]
        
        # Optimize the function
        optimized_code, report = self.optimizer.optimize_function(
            code,
            test_cases,
            verbose=False
        )
        
        # Check report structure
        self.assertIn("original", report)
        self.assertIn("optimized", report)
        self.assertIn("improvements", report)
        
        # Check if optimization preserved functionality
        self.assertNotEqual(code, optimized_code)  # Code should be modified
        self.assertIn("def sum_list", optimized_code)  # Function name preserved

    def test_training(self):
        """Test training process."""
        functions = self.optimizer.load_functions()
        
        # Train the optimizer
        self.optimizer.train(functions, verbose=False)
        
        # Check if checkpoints were created
        checkpoint_files = list(self.checkpoint_dir.glob("*.pt"))
        self.assertTrue(len(checkpoint_files) > 0)

if __name__ == "__main__":
    unittest.main() 