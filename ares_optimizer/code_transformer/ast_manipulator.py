"""AST manipulation module for Ares Optimizer.

This module provides functionality to parse, analyze, and transform Python code
using the Abstract Syntax Tree (AST).
"""

import ast
from typing import Any, Dict, List, Optional, Set, Tuple, Union, Callable
from dataclasses import dataclass

@dataclass
class Transformation:
    """Represents a code transformation with its description and confidence."""
    description: str
    confidence: float  # 0.0 to 1.0
    transformed_code: str

class ASTManipulator:
    """Handles parsing and manipulation of Python code using AST."""

    def __init__(self):
        """Initialize the AST manipulator."""
        self._cache: Dict[str, ast.AST] = {}
        self.transformations: List[Transformation] = []

    def parse_code(self, code: str) -> Optional[ast.AST]:
        """Parse Python code into an AST.

        Args:
            code: The Python code to parse.

        Returns:
            The parsed AST, or None if parsing failed.
        """
        try:
            return ast.parse(code)
        except SyntaxError:
            return None

    def unparse_code(self, tree: ast.AST) -> str:
        """Convert an AST back to Python code.

        Args:
            tree: The AST to convert.

        Returns:
            The generated Python code.
        """
        # Note: ast.unparse is available in Python 3.9+
        # For earlier versions, we would need to use astor or similar
        return ast.unparse(tree)

    def get_function_definitions(self, tree: ast.AST) -> List[ast.FunctionDef]:
        """Get all function definitions from an AST.

        Args:
            tree: The AST to analyze.

        Returns:
            List of function definition nodes.
        """
        functions = []
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                functions.append(node)
        return functions

    def get_function_by_name(self, tree: ast.AST, name: str) -> Optional[ast.FunctionDef]:
        """Get a specific function definition by name.

        Args:
            tree: The AST to analyze.
            name: Name of the function to find.

        Returns:
            The function definition node, or None if not found.
        """
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == name:
                return node
        return None

    def analyze_function(self, func_def: ast.FunctionDef) -> Dict[str, Any]:
        """Analyze a function definition for various metrics.

        Args:
            func_def: The function definition node to analyze.

        Returns:
            Dictionary containing various metrics about the function.
        """
        metrics = {
            "name": func_def.name,
            "args": len(func_def.args.args),
            "returns": isinstance(func_def.returns, ast.Name),
            "docstring": ast.get_docstring(func_def),
            "complexity": self._calculate_complexity(func_def),
            "loops": self._count_loops(func_def),
            "conditionals": self._count_conditionals(func_def),
            "calls": self._count_function_calls(func_def),
            "variables": self._get_variable_usage(func_def)
        }
        return metrics

    def _calculate_complexity(self, node: ast.AST) -> int:
        """Calculate cyclomatic complexity of an AST node.

        Args:
            node: The AST node to analyze.

        Returns:
            The cyclomatic complexity score.
        """
        complexity = 1  # Base complexity

        for child in ast.walk(node):
            if isinstance(child, (ast.If, ast.While, ast.For, ast.Try,
                                ast.ExceptHandler, ast.With, ast.Assert)):
                complexity += 1
            elif isinstance(child, ast.BoolOp):
                complexity += len(child.values) - 1

        return complexity

    def _count_loops(self, node: ast.AST) -> Dict[str, int]:
        """Count different types of loops in an AST node.

        Args:
            node: The AST node to analyze.

        Returns:
            Dictionary with counts of different loop types.
        """
        counts = {"for": 0, "while": 0}
        for child in ast.walk(node):
            if isinstance(child, ast.For):
                counts["for"] += 1
            elif isinstance(child, ast.While):
                counts["while"] += 1
        return counts

    def _count_conditionals(self, node: ast.AST) -> Dict[str, int]:
        """Count different types of conditionals in an AST node.

        Args:
            node: The AST node to analyze.

        Returns:
            Dictionary with counts of different conditional types.
        """
        counts = {"if": 0, "elif": 0, "else": 0}
        for child in ast.walk(node):
            if isinstance(child, ast.If):
                counts["if"] += 1
                if child.orelse:
                    if isinstance(child.orelse[0], ast.If):
                        counts["elif"] += 1
                    else:
                        counts["else"] += 1
        return counts

    def _count_function_calls(self, node: ast.AST) -> Dict[str, int]:
        """Count function calls in an AST node.

        Args:
            node: The AST node to analyze.

        Returns:
            Dictionary mapping function names to their call counts.
        """
        calls = {}
        for child in ast.walk(node):
            if isinstance(child, ast.Call):
                if isinstance(child.func, ast.Name):
                    name = child.func.id
                    calls[name] = calls.get(name, 0) + 1
                elif isinstance(child.func, ast.Attribute):
                    name = child.func.attr
                    calls[name] = calls.get(name, 0) + 1
        return calls

    def _get_variable_usage(self, node: ast.AST) -> Dict[str, Set[str]]:
        """Analyze variable usage in an AST node.

        Args:
            node: The AST node to analyze.

        Returns:
            Dictionary mapping variable names to their usage types (read/write).
        """
        usage = {}
        
        for child in ast.walk(node):
            # Handle assignments
            if isinstance(child, ast.Assign):
                for target in child.targets:
                    if isinstance(target, ast.Name):
                        name = target.id
                        if name not in usage:
                            usage[name] = set()
                        usage[name].add("write")
            
            # Handle variable reads
            elif isinstance(child, ast.Name) and isinstance(child.ctx, ast.Load):
                name = child.id
                if name not in usage:
                    usage[name] = set()
                usage[name].add("read")
        
        return usage

    def transform_function(self, func_def: ast.FunctionDef,
                         transformations: List[Tuple[str, Any]]) -> ast.FunctionDef:
        """Apply a series of transformations to a function definition.

        Args:
            func_def: The function definition to transform.
            transformations: List of (transformation_type, parameters) tuples.

        Returns:
            The transformed function definition.
        """
        # Create a deep copy of the function definition
        new_func = ast.FunctionDef(
            name=func_def.name,
            args=func_def.args,
            body=func_def.body.copy(),
            decorator_list=func_def.decorator_list.copy(),
            returns=func_def.returns
        )

        # Apply each transformation
        for transform_type, params in transformations:
            if transform_type == "add_decorator":
                new_func.decorator_list.append(params)
            elif transform_type == "modify_body":
                new_func.body = params
            elif transform_type == "modify_args":
                new_func.args = params
            elif transform_type == "modify_returns":
                new_func.returns = params

        return new_func

    def validate_transformation(self, original: ast.AST,
                              transformed: ast.AST) -> bool:
        """Validate that a transformation maintains code correctness.

        Args:
            original: The original AST.
            transformed: The transformed AST.

        Returns:
            Boolean indicating if the transformation is valid.
        """
        # Basic validation: check if the transformed code is syntactically valid
        try:
            ast.unparse(transformed)
        except Exception:
            return False

        # TODO: Add more sophisticated validation rules
        # - Check for variable scope changes
        # - Verify function signatures remain compatible
        # - Ensure no critical functionality is removed
        # - Validate that the transformation doesn't introduce infinite loops

        return True

    def find_nodes(self, tree: ast.AST, node_type: type) -> List[ast.AST]:
        """
        Find all nodes of a specific type in the AST.
        
        Args:
            tree: The AST to search
            node_type: The type of nodes to find
            
        Returns:
            List of matching nodes
        """
        nodes = []
        for node in ast.walk(tree):
            if isinstance(node, node_type):
                nodes.append(node)
        return nodes
    
    def apply_transformation(self, tree: ast.AST, 
                           transformer: Callable[[ast.AST], Optional[ast.AST]]) -> Optional[ast.AST]:
        """
        Apply a transformation to the AST.
        
        Args:
            tree: The AST to transform
            transformer: Function that takes a node and returns a transformed node
            
        Returns:
            The transformed AST if successful, None otherwise
        """
        try:
            return transformer(tree)
        except Exception:
            return None
    
    def add_transformation(self, description: str, confidence: float, transformed_code: str):
        """
        Add a transformation to the list of applied transformations.
        
        Args:
            description: Description of what was changed
            confidence: Confidence in the transformation (0.0 to 1.0)
            transformed_code: The transformed code
        """
        self.transformations.append(Transformation(
            description=description,
            confidence=confidence,
            transformed_code=transformed_code
        ))
    
    def get_transformations(self) -> List[Transformation]:
        """
        Get the list of applied transformations.
        
        Returns:
            List of Transformation objects
        """
        return self.transformations
    
    def clear_transformations(self):
        """Clear the list of applied transformations."""
        self.transformations.clear()
    
    def analyze_complexity(self, tree: ast.AST) -> dict:
        """
        Analyze the complexity of the code.
        
        Args:
            tree: The AST to analyze
            
        Returns:
            Dictionary containing complexity metrics
        """
        metrics = {
            'function_count': 0,
            'loop_count': 0,
            'if_count': 0,
            'max_depth': 0,
            'variable_count': 0,
            'import_count': 0
        }
        
        def update_metrics(node: ast.AST, depth: int = 0):
            metrics['max_depth'] = max(metrics['max_depth'], depth)
            
            if isinstance(node, ast.FunctionDef):
                metrics['function_count'] += 1
            elif isinstance(node, (ast.For, ast.While)):
                metrics['loop_count'] += 1
            elif isinstance(node, ast.If):
                metrics['if_count'] += 1
            elif isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
                metrics['variable_count'] += 1
            elif isinstance(node, ast.Import) or isinstance(node, ast.ImportFrom):
                metrics['import_count'] += 1
            
            for child in ast.iter_child_nodes(node):
                update_metrics(child, depth + 1)
        
        update_metrics(tree)
        return metrics 