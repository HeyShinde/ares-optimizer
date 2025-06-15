"""Code transformation module for Ares Optimizer.

This module defines specific code transformations that can be applied to optimize
Python functions for better performance.
"""

import ast
from typing import Any, Dict, List, Optional, Tuple, Union

from .ast_manipulator import ASTManipulator


class Transformations:
    """Defines and implements specific code optimization transformations."""

    def __init__(self):
        """Initialize the transformations manager."""
        self.ast_manipulator = ASTManipulator()

    def apply_lru_cache(self, code: str, function_name: str) -> Tuple[str, bool]:
        """Add LRU cache decorator to a function.

        Args:
            code: The Python code containing the function.
            function_name: Name of the function to optimize.

        Returns:
            Tuple of (transformed_code, success).
        """
        tree = self.ast_manipulator.parse_code(code)
        if not tree:
            return code, False

        func_def = self.ast_manipulator.get_function_by_name(tree, function_name)
        if not func_def:
            return code, False

        # Check if function is already cached
        for decorator in func_def.decorator_list:
            if isinstance(decorator, ast.Call):
                if isinstance(decorator.func, ast.Attribute):
                    if decorator.func.attr == "lru_cache":
                        return code, False

        # Add the decorator
        lru_cache = ast.Call(
            func=ast.Attribute(
                value=ast.Name(id="functools", ctx=ast.Load()),
                attr="lru_cache",
                ctx=ast.Load()
            ),
            args=[],
            keywords=[]
        )
        func_def.decorator_list.append(lru_cache)

        # Add import if needed
        if not any(isinstance(node, ast.Import) and any(alias.name == "functools" for alias in node.names)
                  for node in ast.walk(tree)):
            import_stmt = ast.Import(names=[ast.alias(name="functools", asname=None)])
            tree.body.insert(0, import_stmt)

        return self.ast_manipulator.unparse_code(tree), True

    def convert_list_to_set(self, code: str, function_name: str) -> Tuple[str, bool]:
        """Convert list to set for membership testing.

        Args:
            code: The Python code containing the function.
            function_name: Name of the function to optimize.

        Returns:
            Tuple of (transformed_code, success).
        """
        tree = self.ast_manipulator.parse_code(code)
        if not tree:
            return code, False

        func_def = self.ast_manipulator.get_function_by_name(tree, function_name)
        if not func_def:
            return code, False

        # Find list variables used in membership tests
        list_vars = set()
        for node in ast.walk(func_def):
            if isinstance(node, ast.Compare):
                if isinstance(node.left, ast.Name):
                    for op, comparator in zip(node.ops, node.comparators):
                        if isinstance(op, ast.In) and isinstance(comparator, ast.Name):
                            list_vars.add(comparator.id)

        if not list_vars:
            return code, False

        # Transform list initializations to sets
        for node in ast.walk(func_def):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id in list_vars:
                        if isinstance(node.value, ast.List):
                            node.value = ast.Call(
                                func=ast.Name(id="set", ctx=ast.Load()),
                                args=[node.value],
                                keywords=[]
                            )

        return self.ast_manipulator.unparse_code(tree), True

    def replace_loop_with_sum(self, code: str, function_name: str) -> Tuple[str, bool]:
        """Replace loop-based summation with built-in sum().

        Args:
            code: The Python code containing the function.
            function_name: Name of the function to optimize.

        Returns:
            Tuple of (transformed_code, success).
        """
        tree = self.ast_manipulator.parse_code(code)
        if not tree:
            return code, False

        func_def = self.ast_manipulator.get_function_by_name(tree, function_name)
        if not func_def:
            return code, False

        # Find loops that perform summation
        for node in ast.walk(func_def):
            if isinstance(node, ast.For):
                # Check if loop body contains a single assignment that adds to a variable
                if len(node.body) == 1 and isinstance(node.body[0], ast.AugAssign):
                    aug_assign = node.body[0]
                    if isinstance(aug_assign.op, ast.Add):
                        # Replace with sum()
                        new_node = ast.Assign(
                            targets=[aug_assign.target],
                            value=ast.Call(
                                func=ast.Name(id="sum", ctx=ast.Load()),
                                args=[node.iter],
                                keywords=[]
                            )
                        )
                        # Find the parent node and replace the loop
                        for parent in ast.walk(func_def):
                            if isinstance(parent, ast.For) and parent == node:
                                parent = new_node
                                return self.ast_manipulator.unparse_code(tree), True

        return code, False

    def use_generator_expression(self, code: str, function_name: str) -> Tuple[str, bool]:
        """Replace list comprehension with generator expression when appropriate.

        Args:
            code: The Python code containing the function.
            function_name: Name of the function to optimize.

        Returns:
            Tuple of (transformed_code, success).
        """
        tree = self.ast_manipulator.parse_code(code)
        if not tree:
            return code, False

        func_def = self.ast_manipulator.get_function_by_name(tree, function_name)
        if not func_def:
            return code, False

        # Find list comprehensions that are immediately consumed
        for node in ast.walk(func_def):
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name) and node.func.id in ("sum", "any", "all", "min", "max"):
                    if len(node.args) == 1 and isinstance(node.args[0], ast.ListComp):
                        # Replace ListComp with GeneratorExp
                        list_comp = node.args[0]
                        gen_exp = ast.GeneratorExp(
                            elt=list_comp.elt,
                            generators=list_comp.generators
                        )
                        node.args[0] = gen_exp
                        return self.ast_manipulator.unparse_code(tree), True

        return code, False

    def introduce_early_exit(self, code: str, function_name: str) -> Tuple[str, bool]:
        """Add early exit conditions to loops when possible.

        Args:
            code: The Python code containing the function.
            function_name: Name of the function to optimize.

        Returns:
            Tuple of (transformed_code, success).
        """
        tree = self.ast_manipulator.parse_code(code)
        if not tree:
            return code, False

        func_def = self.ast_manipulator.get_function_by_name(tree, function_name)
        if not func_def:
            return code, False

        # Find loops that could benefit from early exit
        for node in ast.walk(func_def):
            if isinstance(node, ast.For):
                # Check if loop body contains a condition that could trigger early exit
                for stmt in node.body:
                    if isinstance(stmt, ast.If):
                        # Add break statement to the if block
                        stmt.body.append(ast.Break())
                        return self.ast_manipulator.unparse_code(tree), True

        return code, False

    def apply_numpy_vectorization(self, code: str, function_name: str) -> Tuple[str, bool]:
        """Replace explicit loops with NumPy vectorized operations.

        Args:
            code: The Python code containing the function.
            function_name: Name of the function to optimize.

        Returns:
            Tuple of (transformed_code, success).
        """
        tree = self.ast_manipulator.parse_code(code)
        if not tree:
            return code, False

        func_def = self.ast_manipulator.get_function_by_name(tree, function_name)
        if not func_def:
            return code, False

        # Add numpy import if needed
        if not any(isinstance(node, ast.Import) and any(alias.name == "numpy" for alias in node.names)
                  for node in ast.walk(tree)):
            import_stmt = ast.Import(names=[ast.alias(name="numpy", asname="np")])
            tree.body.insert(0, import_stmt)

        # Find loops that perform numerical operations
        for node in ast.walk(func_def):
            if isinstance(node, ast.For):
                # Check if loop body contains numerical operations
                if any(isinstance(stmt, ast.AugAssign) and isinstance(stmt.op, (ast.Add, ast.Sub, ast.Mult, ast.Div))
                      for stmt in node.body):
                    # TODO: Implement NumPy vectorization transformation
                    # This is a complex transformation that requires careful analysis
                    # of the loop body and array operations
                    pass

        return code, False

    def get_available_transformations(self) -> List[Dict[str, Any]]:
        """Get a list of available transformations with their descriptions.

        Returns:
            List of dictionaries containing transformation information.
        """
        return [
            {
                "name": "apply_lru_cache",
                "description": "Add LRU cache decorator for memoization",
                "type": "decorator"
            },
            {
                "name": "convert_list_to_set",
                "description": "Convert list to set for membership testing",
                "type": "data_structure"
            },
            {
                "name": "replace_loop_with_sum",
                "description": "Replace loop-based summation with built-in sum()",
                "type": "algorithm"
            },
            {
                "name": "use_generator_expression",
                "description": "Replace list comprehension with generator expression",
                "type": "memory"
            },
            {
                "name": "introduce_early_exit",
                "description": "Add early exit conditions to loops",
                "type": "algorithm"
            },
            {
                "name": "apply_numpy_vectorization",
                "description": "Replace explicit loops with NumPy vectorized operations",
                "type": "performance"
            }
        ] 