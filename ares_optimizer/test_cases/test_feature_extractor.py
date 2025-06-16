"""
Tests for the feature extractor module.
"""

import pytest
import numpy as np
from ares_optimizer.state_representation.feature_extractor import FeatureExtractor


def test_extract_ast_features():
    """Test AST feature extraction."""
    extractor = FeatureExtractor()
    
    # Test simple function
    code = """
def add(a, b):
    return a + b
"""
    features = extractor.extract_ast_features(code)
    assert features["num_functions"] == 1
    assert features["num_returns"] == 1
    assert features["num_loops"] == 0
    assert features["num_conditionals"] == 0
    assert features["has_yield"] == 0.0
    assert features["has_lambda"] == 0.0

    # Test function with loops and conditionals
    code = """
def process_list(lst):
    result = []
    for item in lst:
        if item > 0:
            result.append(item)
    return result
"""
    features = extractor.extract_ast_features(code)
    assert features["num_functions"] == 1
    assert features["num_loops"] == 1
    assert features["num_conditionals"] == 1
    assert features["num_assignments"] == 1

    # Test function with comprehensions
    code = """
def square_list(lst):
    return [x * x for x in lst]
"""
    features = extractor.extract_ast_features(code)
    assert features["has_list_comp"] == 1.0

    # Test invalid code
    features = extractor.extract_ast_features("def invalid code")
    assert all(features[feature] == 0.0 for feature in extractor.ast_features)


def test_extract_performance_features():
    """Test performance feature extraction."""
    extractor = FeatureExtractor()
    
    # Test with zero values
    features = extractor.extract_performance_features(0.0, 0.0)
    assert features["runtime"] == 0.0
    assert features["memory"] == 0.0
    assert features["log_runtime"] == 0.0
    assert features["log_memory"] == 0.0

    # Test with positive values
    features = extractor.extract_performance_features(100.0, 50.0)
    assert features["runtime"] == 100.0
    assert features["memory"] == 50.0
    assert features["log_runtime"] == np.log1p(100.0)
    assert features["log_memory"] == np.log1p(50.0)


def test_get_state_vector():
    """Test state vector generation."""
    extractor = FeatureExtractor()
    
    code = """
def example():
    return 42
"""
    state_vector = extractor.get_state_vector(code, 10.0, 5.0)
    
    # Check vector shape (14 AST features + 4 performance features)
    assert state_vector.shape == (18,)
    assert state_vector.dtype == np.float32
    
    # Check that all values are finite
    assert np.all(np.isfinite(state_vector))
    
    # Check that AST features are present
    assert state_vector[0] > 0  # num_lines
    assert state_vector[1] > 0  # num_chars
    assert state_vector[2] == 1.0  # num_functions
    
    # Check that performance features are present (with float32 precision)
    np.testing.assert_array_almost_equal(
        state_vector[-4:],
        np.array([
            np.log1p(10.0),
            np.log1p(5.0),
            10.0,
            5.0
        ], dtype=np.float32),
        decimal=6
    ) 