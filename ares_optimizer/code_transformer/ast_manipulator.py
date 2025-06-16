"""
AST manipulation utilities for Ares code transformer.
"""

import ast
import astunparse  # You may need to add this to requirements.txt if not present
from typing import Any


def parse_code_to_ast(code_string: str) -> ast.AST:
    """
    Parse Python code string into an AST.
    """
    return ast.parse(code_string)


def ast_to_code(ast_tree: ast.AST) -> str:
    """
    Convert an AST back to Python code string.
    """
    return astunparse.unparse(ast_tree)


def find_nodes_by_type(ast_tree: ast.AST, node_type: Any) -> list:
    """
    Find all nodes of a given type in the AST.
    """
    return [node for node in ast.walk(ast_tree) if isinstance(node, node_type)]


def replace_node(parent: ast.AST, old_node: ast.AST, new_node: ast.AST) -> None:
    """
    Replace old_node with new_node in the parent's body.
    """
    for field, value in ast.iter_fields(parent):
        if isinstance(value, list):
            for i, item in enumerate(value):
                if item is old_node:
                    value[i] = new_node
        elif value is old_node:
            setattr(parent, field, new_node) 