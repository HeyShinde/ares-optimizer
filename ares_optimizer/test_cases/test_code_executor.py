"""Tests for the CodeExecutor module."""

import pytest
import logging
from ares_optimizer.ares_env.code_executor import CodeExecutor

# Configure logging for tests
logging.basicConfig(level=logging.DEBUG)

def test_successful_execution():
    """Test successful execution of a simple function."""
    code = """
def add(a, b):
    return a + b
"""
    executor = CodeExecutor()
    runtime, memory, error = executor.execute_function(code, "add", (1, 2))
    
    assert error is None
    assert runtime > 0
    assert memory >= 0

def test_syntax_error(executor):
    """Test handling of syntax errors."""
    code = """
def add(a, b)
    return a + b
"""
    runtime, memory, error = executor.execute_function(code, "add", (1, 2))
    
    assert error is not None
    assert "Syntax error" in error
    assert runtime == 0
    assert memory == 0

def test_runtime_error():
    """Test handling of runtime errors."""
    code = """
def divide(a, b):
    return a / b
"""
    executor = CodeExecutor()
    runtime, memory, error = executor.execute_function(code, "divide", (1, 0))
    
    assert error is not None
    assert "division by zero" in error
    assert runtime == 0
    assert memory == 0

def test_timeout():
    """Test handling of timeout."""
    code = """
def infinite_loop():
    while True:
        pass
"""
    executor = CodeExecutor(timeout_seconds=1)
    runtime, memory, error = executor.execute_function(code, "infinite_loop", ())
    
    assert error == "Execution timed out"
    assert runtime == 0
    assert memory == 0

def test_memory_measurement():
    """Test memory usage measurement."""
    code = """
def create_large_list():
    return [0] * 1000000
"""
    executor = CodeExecutor()
    runtime, memory, error = executor.execute_function(code, "create_large_list", ())
    
    assert error is None
    assert runtime > 0
    assert memory > 0  # Should use some memory for the large list

def test_memory_measurement_small():
    """Test memory measurement for a small operation."""
    code = """
def small_operation():
    return 1 + 1
"""
    executor = CodeExecutor()
    runtime, memory, error = executor.execute_function(code, "small_operation", ())
    
    assert error is None
    assert runtime > 0
    assert memory >= 0  # Should use minimal memory

def test_memory_measurement_nested():
    """Test memory measurement for nested data structures."""
    code = """
def create_nested_structure():
    return [[i for i in range(1000)] for _ in range(100)]
"""
    executor = CodeExecutor()
    runtime, memory, error = executor.execute_function(code, "create_nested_structure", ())
    
    assert error is None
    assert runtime > 0
    assert memory > 0  # Should use significant memory for nested lists

def test_memory_measurement_generator():
    """Test memory measurement for generator expressions."""
    code = """
def use_generator():
    return sum(i for i in range(1000000))
"""
    executor = CodeExecutor()
    runtime, memory, error = executor.execute_function(code, "use_generator", ())
    
    assert error is None
    assert runtime > 0
    assert memory >= 0  # Should use minimal memory due to generator

def test_memory_measurement_recursive():
    """Test memory measurement for recursive function."""
    code = """
def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)
"""
    executor = CodeExecutor()
    runtime, memory, error = executor.execute_function(code, "fibonacci", (10,))
    
    assert error is None
    assert runtime > 0
    assert memory > 0  # Should use some memory for recursion stack 

"""
Edge case tests for the code executor.
"""

import pytest
import time
import numpy as np
from ares_optimizer.ares_env.code_executor import CodeExecutor
import logging
import os
import tempfile
import shutil

@pytest.fixture
def executor():
    """Create a code executor for testing."""
    return CodeExecutor()

def test_infinite_loop(executor):
    """Test handling of infinite loops."""
    code = """
def infinite_loop():
    while True:
        pass
    """
    runtime, memory, error = executor.execute_function(code, "infinite_loop", [])
    assert error is not None, "Should detect infinite loop"
    assert "timeout" in str(error).lower(), "Should timeout"

def test_memory_overflow(executor):
    """Test handling of memory overflow."""
    code = """
def memory_overflow():
    result = []
    while True:
        result.append([0] * 1000000)
    """
    runtime, memory, error = executor.execute_function(code, "memory_overflow", [])
    assert error is not None, "Should detect memory overflow"
    assert "memory" in str(error).lower(), "Should be memory related error"

def test_syntax_error(executor):
    """Test handling of syntax errors."""
    code = """
def syntax_error():
    if True
        return 1
    """
    runtime, memory, error = executor.execute_function(code, "syntax_error", [])
    assert error is not None, "Should detect syntax error"
    assert "syntax" in str(error).lower(), "Should be syntax error"

def test_import_error(executor):
    """Test handling of import errors."""
    code = """
def import_error():
    import nonexistent_module
    return 1
    """
    runtime, memory, error = executor.execute_function(code, "import_error", [])
    assert error is not None, "Should detect import error"
    assert "import" in str(error).lower(), "Should be import error"

def test_division_by_zero(executor):
    """Test handling of division by zero."""
    code = """
def divide_by_zero():
    return 1 / 0
    """
    runtime, memory, error = executor.execute_function(code, "divide_by_zero", [])
    assert error is not None, "Should detect division by zero"
    assert "zero" in str(error).lower(), "Should be division by zero error"

def test_recursion_depth(executor):
    """Test handling of recursion depth limit."""
    code = """
def recursive_function(n):
    if n <= 0:
        return 1
    return recursive_function(n - 1) + 1
    """
    runtime, memory, error = executor.execute_function(code, "recursive_function", [1000])
    assert error is not None, "Should detect recursion depth limit"
    assert "recursion" in str(error).lower(), "Should be recursion error"

def test_large_output(executor):
    """Test handling of large output."""
    code = """
def large_output():
    return [0] * 1000000
    """
    runtime, memory, error = executor.execute_function(code, "large_output", [])
    assert error is None, "Should handle large output"
    assert memory > 0, "Should measure memory usage"

def test_multiple_exceptions(executor):
    """Test handling of multiple exceptions."""
    code = """
def multiple_exceptions():
    try:
        return 1 / 0
    except ZeroDivisionError:
        return None
    finally:
        return "finally"
    """
    runtime, memory, error = executor.execute_function(code, "multiple_exceptions", [])
    assert error is None, "Should handle multiple exceptions"
    assert runtime > 0, "Should measure runtime"

def test_global_variables(executor):
    """Test handling of global variables."""
    code = """
global_var = 0
def modify_global():
    global global_var
    global_var += 1
    return global_var
    """
    runtime, memory, error = executor.execute_function(code, "modify_global", [])
    assert error is None, "Should handle global variables"
    assert runtime > 0, "Should measure runtime"

def test_file_operations(executor):
    """Test handling of file operations."""
    with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
        f.write("test content")
        temp_file = f.name

    code = f"""
def file_operations():
    with open("{temp_file}", 'r') as f:
        return f.read()
    """
    runtime, memory, error = executor.execute_function(code, "file_operations", [])
    assert error is None, "Should handle file operations"
    assert runtime > 0, "Should measure runtime"

    os.unlink(temp_file)

def test_network_operations(executor):
    """Test handling of network operations."""
    code = """
def network_operations():
    import socket
    s = socket.socket()
    s.connect(('localhost', 12345))
    return s.recv(1024)
    """
    runtime, memory, error = executor.execute_function(code, "network_operations", [])
    assert error is not None, "Should detect network error"
    assert "connection" in str(error).lower(), "Should be connection error"

def test_system_operations(executor):
    """Test handling of system operations."""
    code = """
def system_operations():
    import os
    return os.system('rm -rf /')
    """
    runtime, memory, error = executor.execute_function(code, "system_operations", [])
    assert error is not None, "Should detect system operation error"
    assert "permission" in str(error).lower(), "Should be permission error"

def test_concurrent_operations(executor):
    """Test handling of concurrent operations."""
    code = """
def concurrent_operations():
    import threading
    def worker():
        while True:
            pass
    threads = [threading.Thread(target=worker) for _ in range(10)]
    for t in threads:
        t.start()
    return len(threads)
    """
    runtime, memory, error = executor.execute_function(code, "concurrent_operations", [])
    assert error is not None, "Should detect concurrent operation error"
    assert "timeout" in str(error).lower(), "Should timeout"

def test_resource_cleanup(executor):
    """Test proper resource cleanup."""
    code = """
def resource_cleanup():
    import tempfile
    f = tempfile.NamedTemporaryFile(delete=False)
    f.write(b'test')
    f.close()
    return f.name
    """
    runtime, memory, error = executor.execute_function(code, "resource_cleanup", [])
    assert error is None, "Should handle resource cleanup"
    assert runtime > 0, "Should measure runtime"

def test_unicode_handling(executor):
    """Test handling of unicode characters."""
    code = """
def unicode_handling():
    return "你好，世界"
    """
    runtime, memory, error = executor.execute_function(code, "unicode_handling", [])
    assert error is None, "Should handle unicode"
    assert runtime > 0, "Should measure runtime"

def test_large_input(executor):
    """Test handling of large input."""
    code = """
def large_input(x):
    return len(x)
    """
    large_input = [0] * 1000000
    runtime, memory, error = executor.execute_function(code, "large_input", [large_input])
    assert error is None, "Should handle large input"
    assert memory > 0, "Should measure memory usage" 