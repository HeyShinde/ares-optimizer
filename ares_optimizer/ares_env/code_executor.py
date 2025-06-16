"""Code execution and performance measurement module for Ares Optimizer."""

import ast
import multiprocessing
import os
import time
import timeit
from typing import Any, Dict, Optional, Tuple, Union

import psutil


class CodeExecutor:
    """Safely executes Python code snippets and measures their performance."""

    def __init__(self, timeout_seconds: int = 30):
        """Initialize the CodeExecutor.

        Args:
            timeout_seconds: Maximum time (in seconds) to allow for code execution.
        """
        self.timeout_seconds = timeout_seconds

    def execute_function(
        self,
        code_string: str,
        function_name: str,
        test_inputs: Union[Any, Tuple[Any, ...]],
        num_runs: int = 5,
    ) -> Tuple[float, float, Optional[str]]:
        """Execute a function and measure its performance.

        Args:
            code_string: The Python code containing the function to execute.
            function_name: Name of the function to execute.
            test_inputs: Input arguments for the function. Can be a single value or a tuple of values.
            num_runs: Number of times to run the function for performance measurement.

        Returns:
            Tuple containing:
            - Average runtime in milliseconds
            - Peak memory usage in megabytes
            - Exception info if any error occurred, None otherwise
        """
        # Validate the code string
        try:
            ast.parse(code_string)
        except SyntaxError as e:
            return 0.0, 0.0, f"Syntax error: {str(e)}"

        # Create a temporary module to execute the code
        module_code = f"""
{code_string}

def _run_test():
    return {function_name}(*{test_inputs if isinstance(test_inputs, tuple) else (test_inputs,)})
"""
        # Create a process to run the code
        result_queue = multiprocessing.Queue()
        process = multiprocessing.Process(
            target=self._run_in_process,
            args=(module_code, result_queue),
        )

        try:
            process.start()
            process.join(timeout=self.timeout_seconds)

            if process.is_alive():
                process.terminate()
                process.join()
                return 0.0, 0.0, "Execution timed out"

            if not result_queue.empty():
                result = result_queue.get()
                if isinstance(result, Exception):
                    return 0.0, 0.0, str(result)
                return result

            return 0.0, 0.0, "Unknown error occurred"

        except Exception as e:
            return 0.0, 0.0, str(e)

    def _run_in_process(self, code: str, result_queue: multiprocessing.Queue) -> None:
        """Run the code in a separate process and measure its performance.

        Args:
            code: The Python code to execute.
            result_queue: Queue to store the results.
        """
        try:
            # Create a new process for memory measurement
            process = psutil.Process(os.getpid())
            initial_memory = process.memory_info().rss / 1024 / 1024  # Convert to MB

            # Create a namespace for the code
            namespace = {}
            exec(code, namespace)

            # Measure runtime
            timer = timeit.Timer("_run_test()", globals=namespace)
            runtime_ms = timer.timeit(number=1) * 1000  # Convert to milliseconds

            # Measure peak memory
            peak_memory = process.memory_info().rss / 1024 / 1024  # Convert to MB
            memory_used = peak_memory - initial_memory

            result_queue.put((runtime_ms, memory_used, None))

        except Exception as e:
            result_queue.put(e) 