"""Tests for the TestManager module."""

from ares_optimizer.ares_env.test_manager import TestManager


def test_load_test_cases():
    tm = TestManager(test_cases_dir="test_cases")
    test_cases = tm.load_test_cases("add")
    assert isinstance(test_cases, list)
    assert ((1, 2), 3) in test_cases


def test_validate_function_output_success():
    tm = TestManager(test_cases_dir="test_cases")
    code = """
def add(a, b):
    return a + b
"""
    test_cases = tm.load_test_cases("add")
    is_correct, details = tm.validate_function_output(code, "add", test_cases)
    assert is_correct
    assert details is None


def test_validate_function_output_failure():
    tm = TestManager(test_cases_dir="test_cases")
    code = """
def add(a, b):
    return a - b  # Intentional bug
"""
    test_cases = tm.load_test_cases("add")
    is_correct, details = tm.validate_function_output(code, "add", test_cases)
    assert not is_correct
    assert details is not None
    assert "failed" in details or "got" in details 