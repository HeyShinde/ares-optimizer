"""Code execution module for Ares Optimizer.

This module handles the safe execution of Python code and measurement of performance
metrics such as runtime and memory usage.
"""

import ast
import os
import psutil
import subprocess
import tempfile
import time
import sys
from typing import Any, Dict, List, Optional, Tuple, Union


class CodeExecutor:
    """Executes Python code and measures performance metrics."""

    def __init__(self, timeout: int = 30):
        """Initialize the code executor.

        Args:
            timeout: Maximum execution time in seconds.
        """
        self.timeout = timeout
        self.process = psutil.Process()

    def execute_function(
        self, code: str, function_name: str, *args, **kwargs
    ) -> Tuple[Optional[Any], Dict[str, float]]:
        """Execute a function and measure its performance.

        Args:
            code: The Python code containing the function.
            function_name: Name of the function to execute.
            *args: Positional arguments for the function.
            **kwargs: Keyword arguments for the function.

        Returns:
            Tuple of (result, metrics).
        """
        # Create a temporary file for the code
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            # Add necessary imports
            f.write("import time\n")
            f.write("import psutil\n")
            f.write("import sys\n\n")
            
            # Write the function code
            f.write(code)
            f.write("\n\n")
            
            # Add execution code
            f.write(f"""
if __name__ == "__main__":
    process = psutil.Process()
    start_time = time.time()
    start_memory = process.memory_info().rss
    
    try:
        result = {function_name}(*{args}, **{kwargs})
        end_time = time.time()
        end_memory = process.memory_info().rss
        
        print("RESULT:", result)
        print("RUNTIME:", end_time - start_time)
        print("MEMORY:", end_memory - start_memory)
        print("CPU:", process.cpu_percent())
        print("IO:", process.io_counters().read_bytes + process.io_counters().write_bytes)
    except Exception as e:
        print("ERROR:", str(e))
        sys.exit(1)
""")
            temp_file = f.name

        try:
            # Execute the code
            start_time = time.time()
            process = subprocess.Popen(
                [sys.executable, temp_file],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            try:
                stdout, stderr = process.communicate(timeout=self.timeout)
            except subprocess.TimeoutExpired:
                process.kill()
                stdout, stderr = process.communicate()
                return None, {
                    "runtime": float("inf"),
                    "memory_usage": float("inf"),
                    "cpu_usage": 0.0,
                    "io_operations": 0.0,
                }

            # Parse output
            if process.returncode != 0:
                return None, {
                    "runtime": float("inf"),
                    "memory_usage": float("inf"),
                    "cpu_usage": 0.0,
                    "io_operations": 0.0,
                }

            # Extract results
            result = None
            metrics = {
                "runtime": 0.0,
                "memory_usage": 0.0,
                "cpu_usage": 0.0,
                "io_operations": 0.0,
            }

            for line in stdout.splitlines():
                if line.startswith("RESULT:"):
                    result = eval(line[7:].strip())
                elif line.startswith("RUNTIME:"):
                    metrics["runtime"] = float(line[8:].strip())
                elif line.startswith("MEMORY:"):
                    metrics["memory_usage"] = float(line[7:].strip())
                elif line.startswith("CPU:"):
                    metrics["cpu_usage"] = float(line[4:].strip())
                elif line.startswith("IO:"):
                    metrics["io_operations"] = float(line[3:].strip())

            return result, metrics

        finally:
            # Clean up
            try:
                os.unlink(temp_file)
            except:
                pass

    def benchmark_function(
        self, code: str, function_name: str, num_runs: int = 5, *args, **kwargs
    ) -> Dict[str, float]:
        """Benchmark a function over multiple runs.

        Args:
            code: The Python code containing the function.
            function_name: Name of the function to benchmark.
            num_runs: Number of benchmark runs.
            *args: Positional arguments for the function.
            **kwargs: Keyword arguments for the function.

        Returns:
            Dictionary of average performance metrics.
        """
        metrics_list = []
        for _ in range(num_runs):
            _, metrics = self.execute_function(code, function_name, *args, **kwargs)
            if metrics["runtime"] != float("inf"):
                metrics_list.append(metrics)

        if not metrics_list:
            return {
                "runtime": float("inf"),
                "memory_usage": float("inf"),
                "cpu_usage": 0.0,
                "io_operations": 0.0,
            }

        # Calculate averages
        return {
            "runtime": sum(m["runtime"] for m in metrics_list) / len(metrics_list),
            "memory_usage": sum(m["memory_usage"] for m in metrics_list) / len(metrics_list),
            "cpu_usage": sum(m["cpu_usage"] for m in metrics_list) / len(metrics_list),
            "io_operations": sum(m["io_operations"] for m in metrics_list) / len(metrics_list),
        }

    def validate_syntax(self, code: str) -> Tuple[bool, Optional[str]]:
        """Validate Python code syntax.

        Args:
            code: The Python code to validate.

        Returns:
            Tuple of (is_valid, error_message).
        """
        try:
            ast.parse(code)
            return True, None
        except SyntaxError as e:
            return False, str(e)

    def get_function_signature(self, code: str, function_name: str) -> Optional[Dict[str, Any]]:
        """Get the signature of a function.

        Args:
            code: The Python code containing the function.
            function_name: Name of the function to analyze.

        Returns:
            Dictionary containing function signature information.
        """
        try:
            tree = ast.parse(code)
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef) and node.name == function_name:
                    return {
                        "name": node.name,
                        "args": [arg.arg for arg in node.args.args],
                        "defaults": [ast.unparse(d) for d in node.args.defaults],
                        "returns": ast.unparse(node.returns) if node.returns else None,
                        "decorators": [ast.unparse(d) for d in node.decorator_list],
                    }
        except:
            pass
        return None 