"""Test runner for verifying function implementations."""

import importlib
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple

# Add parent directory to path to import functions_to_optimize
sys.path.append(str(Path(__file__).parent.parent))

def run_tests(module_name: str, test_cases: Dict[str, List[Tuple]]) -> Dict[str, List[Tuple[bool, str]]]:
    """Run tests for a specific module.
    
    Args:
        module_name: Name of the module containing the functions to test
        test_cases: Dictionary mapping function names to their test cases
        
    Returns:
        Dictionary mapping function names to lists of (success, message) tuples
    """
    results = {}
    module = importlib.import_module(f"functions_to_optimize.{module_name}")
    
    for func_name, cases in test_cases.items():
        func = getattr(module, func_name)
        func_results = []
        
        for case in cases:
            try:
                # Always unpack all but the last element as arguments
                args = case[:-1]
                expected = case[-1]
                result = func(*args)
                success = result == expected
                message = f"Expected {expected}, got {result}"
                func_results.append((success, message))
            except Exception as e:
                func_results.append((False, f"Error: {str(e)}"))
        
        results[func_name] = func_results
    
    return results

def print_results(results: Dict[str, List[Tuple[bool, str]]]) -> None:
    """Print test results in a readable format."""
    for func_name, func_results in results.items():
        print(f"\nTesting {func_name}:")
        for i, (success, message) in enumerate(func_results, 1):
            status = "✓" if success else "✗"
            print(f"  Test {i}: {status} - {message}")

def main():
    """Run all test cases."""
    # Import all test case modules
    from numerical_ops_tests import TEST_CASES as numerical_tests
    from list_manipulation_tests import TEST_CASES as list_tests
    from string_processing_tests import TEST_CASES as string_tests
    from data_structures_usage_tests import TEST_CASES as data_structure_tests
    
    # Run tests for each module
    print("Running numerical operations tests...")
    numerical_results = run_tests("numerical_ops", numerical_tests)
    print_results(numerical_results)
    
    print("\nRunning list manipulation tests...")
    list_results = run_tests("list_manipulation", list_tests)
    print_results(list_results)
    
    print("\nRunning string processing tests...")
    string_results = run_tests("string_processing", string_tests)
    print_results(string_results)
    
    print("\nRunning data structure usage tests...")
    data_structure_results = run_tests("data_structures_usage", data_structure_tests)
    print_results(data_structure_results)

if __name__ == "__main__":
    main() 