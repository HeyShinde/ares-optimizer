"""
Defines and implements specific code transformation actions.
"""

import ast
import astunparse
import functools
from typing import Optional
import logging

logger = logging.getLogger(__name__)

# --- Transformation 1: Apply @functools.lru_cache ---
def apply_lru_cache(code_string: str, function_name: str = None) -> str:
    """
    Adds @functools.lru_cache to the specified function (or the first function if not specified).
    """
    logger.info(f"Applying lru_cache to function: {function_name}")
    tree = ast.parse(code_string)
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            if function_name is None or node.name == function_name:
                # Create the decorator node directly
                decorator = ast.Call(
                    func=ast.Attribute(
                        value=ast.Name(id='functools', ctx=ast.Load()),
                        attr='lru_cache',
                        ctx=ast.Load()
                    ),
                    args=[],
                    keywords=[ast.keyword(arg='maxsize', value=ast.Constant(value=None))]
                )
                node.decorator_list.insert(0, decorator)
                break
    result = astunparse.unparse(tree)
    logger.info(f"Transformed code: {result}")
    return result

# --- Transformation 2: Convert list to set for membership checks ---
def convert_list_to_set_for_membership(code_string: str, list_var: str) -> str:
    """
    Converts a list variable to a set if used for membership checks (item in list_var).
    """
    logger.info(f"Converting list to set for membership checks: {list_var}")
    tree = ast.parse(code_string)
    class ListToSetTransformer(ast.NodeTransformer):
        def visit_Assign(self, node):
            # Replace list assignment with set assignment
            if (
                isinstance(node.value, ast.List)
                and any(
                    isinstance(target, ast.Name) and target.id == list_var
                    for target in node.targets
                )
            ):
                node.value = ast.Call(func=ast.Name(id="set", ctx=ast.Load()), args=[node.value], keywords=[])
            return node
        def visit_Compare(self, node):
            # Replace 'in list_var' with 'in set_var'
            if (
                isinstance(node.ops[0], ast.In)
                and isinstance(node.comparators[0], ast.Name)
                and node.comparators[0].id == list_var
            ):
                node.comparators[0].id = list_var
            return node
    tree = ListToSetTransformer().visit(tree)
    ast.fix_missing_locations(tree)
    result = astunparse.unparse(tree).strip()
    logger.info(f"Transformed code: {result}")
    return result

# --- Transformation 3: Replace loop with sum ---
def replace_loop_with_sum(code_string: str, total_var: str = "total", list_var: str = "my_list") -> str:
    """
    Replaces a for loop summing a list with total = sum(list_var).
    """
    logger.info(f"Replacing loop with sum for total: {total_var}, list: {list_var}")
    tree = ast.parse(code_string)
    class LoopToSumTransformer(ast.NodeTransformer):
        def visit_For(self, node):
            # Look for: for x in my_list: total += x
            if (
                isinstance(node.target, ast.Name)
                and isinstance(node.iter, ast.Name)
                and node.iter.id == list_var
                and len(node.body) == 1
                and isinstance(node.body[0], ast.AugAssign)
                and isinstance(node.body[0].target, ast.Name)
                and node.body[0].target.id == total_var
                and isinstance(node.body[0].op, ast.Add)
                and isinstance(node.body[0].value, ast.Name)
                and node.body[0].value.id == node.target.id
            ):
                # Replace with: total = sum(my_list)
                return ast.Assign(
                    targets=[ast.Name(id=total_var, ctx=ast.Store())],
                    value=ast.Call(
                        func=ast.Name(id="sum", ctx=ast.Load()),
                        args=[ast.Name(id=list_var, ctx=ast.Load())],
                        keywords=[],
                    ),
                )
            return node
    tree = LoopToSumTransformer().visit(tree)
    ast.fix_missing_locations(tree)
    result = astunparse.unparse(tree).strip()
    logger.info(f"Transformed code: {result}")
    return result

# --- Transformation 4: Use generator expression ---
def use_generator_expression(code_string: str) -> str:
    """
    Replaces list comprehensions inside sum() with generator expressions.
    """
    logger.info("Using generator expression")
    tree = ast.parse(code_string)
    class ListCompToGenExp(ast.NodeTransformer):
        def visit_Call(self, node):
            self.generic_visit(node)
            if (
                isinstance(node.func, ast.Name)
                and node.func.id == "sum"
                and len(node.args) == 1
                and isinstance(node.args[0], ast.ListComp)
            ):
                node.args[0] = ast.GeneratorExp(
                    elt=node.args[0].elt,
                    generators=node.args[0].generators,
                )
            return node
    tree = ListCompToGenExp().visit(tree)
    ast.fix_missing_locations(tree)
    result = astunparse.unparse(tree).strip()
    logger.info(f"Transformed code: {result}")
    return result

# --- Transformation 5: Introduce early exit in loops ---
def introduce_early_exit(code_string: str, break_condition: str) -> str:
    """
    Adds a break statement for early exit in a loop if the break_condition is met.
    break_condition should be a valid Python expression as a string.
    """
    logger.info(f"Introducing early exit with condition: {break_condition}")
    tree = ast.parse(code_string)
    class EarlyExitTransformer(ast.NodeTransformer):
        def visit_For(self, node):
            # Insert 'if <break_condition>: break' at the start of the loop body
            break_if = ast.If(
                test=ast.parse(break_condition, mode="eval").body,
                body=[ast.Break()],
                orelse=[],
            )
            node.body.insert(0, break_if)
            return node
    tree = EarlyExitTransformer().visit(tree)
    ast.fix_missing_locations(tree)
    result = astunparse.unparse(tree).strip()
    logger.info(f"Transformed code: {result}")
    return result

# --- Transformation 6: Replace Manual List Append with List Comprehension ---
def replace_manual_list_append_with_list_comp(code_string: str) -> str:
    """
    Replaces manual list append loops with list comprehensions.
    Example:
        result = []
        for x in xs:
            result.append(f(x))
        # becomes
        result = [f(x) for x in xs]
    """
    logger.info("Replacing manual list append with list comprehension")
    tree = ast.parse(code_string)
    class ManualListAppendTransformer(ast.NodeTransformer):
        def __init__(self):
            super().__init__()
            self.to_remove = set()
        def visit_Module(self, node):
            new_body = []
            i = 0
            while i < len(node.body):
                stmt = node.body[i]
                # Look for result = [] followed by for loop
                if (
                    isinstance(stmt, ast.Assign)
                    and len(stmt.targets) == 1
                    and isinstance(stmt.targets[0], ast.Name)
                    and isinstance(stmt.value, ast.List)
                    and i + 1 < len(node.body)
                    and isinstance(node.body[i + 1], ast.For)
                ):
                    for_node = node.body[i + 1]
                    # Check if for loop matches append pattern
                    if (
                        len(for_node.body) == 1
                        and isinstance(for_node.body[0], ast.Expr)
                        and isinstance(for_node.body[0].value, ast.Call)
                        and isinstance(for_node.body[0].value.func, ast.Attribute)
                        and for_node.body[0].value.func.attr == "append"
                        and isinstance(for_node.body[0].value.func.value, ast.Name)
                        and for_node.body[0].value.func.value.id == stmt.targets[0].id
                    ):
                        # Replace both with a single list comp assignment
                        append_target = stmt.targets[0].id
                        append_arg = for_node.body[0].value.args[0]
                        new_body.append(
                            ast.Assign(
                                targets=[ast.Name(id=append_target, ctx=ast.Store())],
                                value=ast.ListComp(
                                    elt=append_arg,
                                    generators=[ast.comprehension(
                                        target=for_node.target,
                                        iter=for_node.iter,
                                        ifs=[],
                                        is_async=0
                                    )]
                                )
                            )
                        )
                        i += 2
                        continue
                new_body.append(stmt)
                i += 1
            node.body = new_body
            return node
    tree = ManualListAppendTransformer().visit(tree)
    ast.fix_missing_locations(tree)
    result = astunparse.unparse(tree).strip()
    logger.info(f"Transformed code: {result}")
    return result

# --- Transformation 7: Replace Manual List Construction with map() ---
def replace_manual_list_with_map(code_string: str) -> str:
    """
    Replaces manual list construction loops with map().
    Example:
        result = []
        for x in xs:
            result.append(f(x))
        # becomes
        result = list(map(f, xs))
    """
    logger.info("Replacing manual list construction with map")
    logger.info(f"Original code: {code_string}")
    tree = ast.parse(code_string)
    class ManualListMapTransformer(ast.NodeTransformer):
        def visit_Module(self, node):
            new_body = []
            i = 0
            while i < len(node.body):
                stmt = node.body[i]
                # Look for result = [] followed by for loop
                if (
                    isinstance(stmt, ast.Assign)
                    and len(stmt.targets) == 1
                    and isinstance(stmt.targets[0], ast.Name)
                    and isinstance(stmt.value, ast.List)
                    and i + 1 < len(node.body)
                    and isinstance(node.body[i + 1], ast.For)
                ):
                    for_node = node.body[i + 1]
                    # Check if for loop matches append pattern
                    if (
                        len(for_node.body) == 1
                        and isinstance(for_node.body[0], ast.Expr)
                        and isinstance(for_node.body[0].value, ast.Call)
                        and isinstance(for_node.body[0].value.func, ast.Attribute)
                        and for_node.body[0].value.func.attr == "append"
                        and isinstance(for_node.body[0].value.func.value, ast.Name)
                        and for_node.body[0].value.func.value.id == stmt.targets[0].id
                    ):
                        append_target = stmt.targets[0].id
                        append_arg = for_node.body[0].value.args[0]
                        lambda_func = ast.Lambda(
                            args=ast.arguments(
                                posonlyargs=[],
                                args=[for_node.target],
                                vararg=None,
                                kwonlyargs=[],
                                kw_defaults=[],
                                kwarg=None,
                                defaults=[]
                            ),
                            body=append_arg
                        )
                        new_body.append(
                            ast.Assign(
                                targets=[ast.Name(id=append_target, ctx=ast.Store())],
                                value=ast.Call(
                                    func=ast.Name(id="list", ctx=ast.Load()),
                                    args=[ast.Call(
                                        func=ast.Name(id="map", ctx=ast.Load()),
                                        args=[lambda_func, for_node.iter],
                                        keywords=[],
                                    )],
                                    keywords=[],
                                )
                            )
                        )
                        i += 2
                        continue
                new_body.append(stmt)
                i += 1
            node.body = new_body
            return node
    tree = ManualListMapTransformer().visit(tree)
    ast.fix_missing_locations(tree)
    result = astunparse.unparse(tree).strip()
    logger.info(f"Transformed code: {result}")
    return result

# --- Transformation 8: Replace range(len(xs)) with enumerate(xs) ---
def replace_range_len_with_enumerate(code_string: str) -> str:
    """
    Replaces loops using range(len(xs)) with enumerate(xs).
    Example:
        for i in range(len(xs)):
            x = xs[i]
        # becomes
        for i, x in enumerate(xs):
    """
    tree = ast.parse(code_string)
    class RangeLenToEnumerateTransformer(ast.NodeTransformer):
        def visit_For(self, node):
            if (
                isinstance(node.iter, ast.Call)
                and isinstance(node.iter.func, ast.Name)
                and node.iter.func.id == "range"
                and len(node.iter.args) == 1
                and isinstance(node.iter.args[0], ast.Call)
                and isinstance(node.iter.args[0].func, ast.Name)
                and node.iter.args[0].func.id == "len"
                and isinstance(node.iter.args[0].args[0], ast.Name)
            ):
                xs_var = node.iter.args[0].args[0].id
                return ast.For(
                    target=ast.Tuple(
                        elts=[
                            node.target,
                            ast.Name(id=xs_var, ctx=ast.Store())
                        ],
                        ctx=ast.Store()
                    ),
                    iter=ast.Call(
                        func=ast.Name(id="enumerate", ctx=ast.Load()),
                        args=[ast.Name(id=xs_var, ctx=ast.Load())],
                        keywords=[]
                    ),
                    body=node.body,
                    orelse=node.orelse
                )
            return node
    tree = RangeLenToEnumerateTransformer().visit(tree)
    ast.fix_missing_locations(tree)
    return astunparse.unparse(tree).strip()

# --- Transformation 9: Use Built-in Functions (max, min, any, all) ---
def use_builtin_functions(code_string: str) -> str:
    """
    Replaces manual implementations of max, min, any, all with built-in functions.
    Example:
        max_val = xs[0]
        for x in xs[1:]:
            if x > max_val:
                max_val = x
        # becomes
        max_val = max(xs)
    """
    tree = ast.parse(code_string)
    class BuiltinFunctionTransformer(ast.NodeTransformer):
        def visit_For(self, node):
            if (
                len(node.body) == 1
                and isinstance(node.body[0], ast.If)
                and isinstance(node.body[0].test, ast.Compare)
                and isinstance(node.body[0].test.ops[0], ast.Gt)
                and isinstance(node.body[0].test.left, ast.Name)
                and isinstance(node.body[0].test.comparators[0], ast.Name)
            ):
                max_var = node.body[0].test.left.id
                return ast.Assign(
                    targets=[ast.Name(id=max_var, ctx=ast.Store())],
                    value=ast.Call(
                        func=ast.Name(id="max", ctx=ast.Load()),
                        args=[node.iter],
                        keywords=[]
                    )
                )
            return node
    tree = BuiltinFunctionTransformer().visit(tree)
    ast.fix_missing_locations(tree)
    return astunparse.unparse(tree).strip()

# --- Transformation 10: Loop Unrolling ---
def unroll_small_loop(code_string: str, unroll_factor: int = 2) -> str:
    """
    Unrolls small loops by duplicating the loop body unroll_factor times.
    Example (unroll_factor=2):
        for i in range(4):
            do_something(i)
        # becomes
        do_something(0)
        do_something(1)
        do_something(2)
        do_something(3)
    """
    tree = ast.parse(code_string)
    class LoopUnrollTransformer(ast.NodeTransformer):
        def visit_For(self, node):
            if (
                isinstance(node.iter, ast.Call)
                and isinstance(node.iter.func, ast.Name)
                and node.iter.func.id == "range"
                and len(node.iter.args) == 1
                and isinstance(node.iter.args[0], ast.Constant)
                and node.iter.args[0].value <= unroll_factor
            ):
                unrolled_body = []
                for i in range(node.iter.args[0].value):
                    for stmt in node.body:
                        unrolled_body.append(ast.copy_location(stmt, node))
                return unrolled_body
            return node
    tree = LoopUnrollTransformer().visit(tree)
    ast.fix_missing_locations(tree)
    return astunparse.unparse(tree).strip()

# --- Transformation 11: Apply NumPy Vectorization ---
def apply_numpy_vectorization(code_string: str) -> str:
    """
    Replaces loops with NumPy vectorized operations.
    Example:
        result = []
        for x in xs:
            result.append(x * 2)
        # becomes
        import numpy as np
        result = np.array(xs) * 2
    """
    tree = ast.parse(code_string)
    class NumpyVectorizationTransformer(ast.NodeTransformer):
        def visit_For(self, node):
            if (
                len(node.body) == 1
                and isinstance(node.body[0], ast.Expr)
                and isinstance(node.body[0].value, ast.Call)
                and isinstance(node.body[0].value.func, ast.Attribute)
                and node.body[0].value.func.attr == "append"
                and isinstance(node.body[0].value.func.value, ast.Name)
            ):
                append_target = node.body[0].value.func.value.id
                append_arg = node.body[0].value.args[0]
                return ast.Assign(
                    targets=[ast.Name(id=append_target, ctx=ast.Store())],
                    value=ast.Call(
                        func=ast.Attribute(
                            value=ast.Call(
                                func=ast.Name(id="np.array", ctx=ast.Load()),
                                args=[node.iter],
                                keywords=[]
                            ),
                            attr="__mul__",
                            ctx=ast.Load()
                        ),
                        args=[append_arg],
                        keywords=[]
                    )
                )
            return node
    tree = NumpyVectorizationTransformer().visit(tree)
    ast.fix_missing_locations(tree)
    return astunparse.unparse(tree).strip() 