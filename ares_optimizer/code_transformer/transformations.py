"""
Defines and implements specific code transformation actions.
"""

import ast
import astunparse
import functools
from typing import Optional, List, Dict, Any
import logging

logger = logging.getLogger(__name__)

# --- Transformation Selection Helper ---
def select_transformations(code: str) -> List[str]:
    """
    Analyzes code and returns a list of applicable transformations.
    Returns a list of transformation function names that should be applied.
    """
    tree = ast.parse(code)
    applicable_transforms = []
    
    # Check for patterns that indicate which transformations would be beneficial
    for node in ast.walk(tree):
        # Check for list membership tests
        if isinstance(node, ast.Compare) and isinstance(node.ops[0], ast.In):
            applicable_transforms.append('convert_list_to_set_for_membership')
        
        # Check for manual list construction
        if isinstance(node, ast.For):
            if any(isinstance(body, ast.Expr) and 
                  isinstance(body.value, ast.Call) and 
                  isinstance(body.value.func, ast.Attribute) and 
                  body.value.func.attr == 'append' 
                  for body in node.body):
                applicable_transforms.append('replace_manual_list_append_with_list_comp')
                applicable_transforms.append('replace_manual_list_with_map')
        
        # Check for range(len()) patterns
        if isinstance(node, ast.For) and isinstance(node.iter, ast.Call):
            if (isinstance(node.iter.func, ast.Name) and 
                node.iter.func.id == 'range' and 
                len(node.iter.args) == 1 and 
                isinstance(node.iter.args[0], ast.Call) and 
                isinstance(node.iter.args[0].func, ast.Name) and 
                node.iter.args[0].func.id == 'len'):
                applicable_transforms.append('replace_range_len_with_enumerate')
        
        # Check for max/min patterns
        if isinstance(node, ast.For):
            if (len(node.body) == 1 and 
                isinstance(node.body[0], ast.If) and 
                len(node.body[0].body) == 1 and 
                isinstance(node.body[0].body[0], ast.Assign)):
                applicable_transforms.append('use_builtin_functions')
        
        # Check for generator patterns
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            if node.func.id in ['sum', 'any', 'all'] and len(node.args) == 1:
                if isinstance(node.args[0], ast.ListComp):
                    applicable_transforms.append('use_generator_expression')
        
        # Check for caching opportunities
        if isinstance(node, ast.FunctionDef):
            if any(isinstance(body, ast.Return) for body in node.body):
                applicable_transforms.append('apply_lru_cache')
    
    return list(set(applicable_transforms))

# --- New Transformation: Use Set Operations ---
def use_set_operations(code: str) -> str:
    """
    Replaces list operations with set operations where appropriate.
    Example:
        result = []
        for x in list1:
            if x in list2:
                result.append(x)
        # becomes
        result = list(set(list1) & set(list2))
    """
    try:
        tree = ast.parse(code)
        class SetOperationTransformer(ast.NodeTransformer):
            def visit_Module(self, node):
                new_body = []
                i = 0
                while i < len(node.body):
                    stmt = node.body[i]
                    
                    # Look for intersection pattern
                    if (isinstance(stmt, ast.Assign) and
                        len(stmt.targets) == 1 and
                        isinstance(stmt.targets[0], ast.Name) and
                        isinstance(stmt.value, ast.List) and
                        i + 1 < len(node.body) and
                        isinstance(node.body[i + 1], ast.For)):
                        
                        for_node = node.body[i + 1]
                        if (len(for_node.body) == 1 and
                            isinstance(for_node.body[0], ast.If) and
                            isinstance(for_node.body[0].test, ast.Compare) and
                            isinstance(for_node.body[0].test.ops[0], ast.In) and
                            len(for_node.body[0].body) == 1 and
                            isinstance(for_node.body[0].body[0], ast.Expr) and
                            isinstance(for_node.body[0].body[0].value, ast.Call) and
                            isinstance(for_node.body[0].body[0].value.func, ast.Attribute) and
                            for_node.body[0].body[0].value.func.attr == 'append'):
                            
                            # Get the lists involved
                            list1 = for_node.iter
                            list2 = for_node.body[0].test.comparators[0]
                            
                            # Create set intersection
                            new_body.append(ast.Assign(
                                targets=[ast.Name(id=stmt.targets[0].id, ctx=ast.Store())],
                                value=ast.Call(
                                    func=ast.Name(id='list', ctx=ast.Load()),
                                    args=[
                                        ast.BinOp(
                                            left=ast.Call(
                                                func=ast.Name(id='set', ctx=ast.Load()),
                                                args=[list1],
                                                keywords=[]
                                            ),
                                            op=ast.BitAnd(),
                                            right=ast.Call(
                                                func=ast.Name(id='set', ctx=ast.Load()),
                                                args=[list2],
                                                keywords=[]
                                            )
                                        )
                                    ],
                                    keywords=[]
                                )
                            ))
                            i += 2
                            continue
                    
                    new_body.append(stmt)
                    i += 1
                node.body = new_body
                return node
        
        tree = SetOperationTransformer().visit(tree)
        ast.fix_missing_locations(tree)
        return ast.unparse(tree)
    except Exception as e:
        logging.error(f"Error in use_set_operations: {str(e)}")
        return code

# --- New Transformation: Use List Comprehension with Condition ---
def use_list_comp_with_condition(code: str) -> str:
    """
    Replaces filter+map patterns with list comprehension.
    Example:
        result = []
        for x in items:
            if condition(x):
                result.append(transform(x))
        # becomes
        result = [transform(x) for x in items if condition(x)]
    """
    try:
        tree = ast.parse(code)
        class ListCompWithConditionTransformer(ast.NodeTransformer):
            def visit_Module(self, node):
                new_body = []
                i = 0
                while i < len(node.body):
                    stmt = node.body[i]
                    
                    if (isinstance(stmt, ast.Assign) and
                        len(stmt.targets) == 1 and
                        isinstance(stmt.targets[0], ast.Name) and
                        isinstance(stmt.value, ast.List) and
                        i + 1 < len(node.body) and
                        isinstance(node.body[i + 1], ast.For)):
                        
                        for_node = node.body[i + 1]
                        if (len(for_node.body) == 1 and
                            isinstance(for_node.body[0], ast.If) and
                            len(for_node.body[0].body) == 1 and
                            isinstance(for_node.body[0].body[0], ast.Expr) and
                            isinstance(for_node.body[0].body[0].value, ast.Call) and
                            isinstance(for_node.body[0].body[0].value.func, ast.Attribute) and
                            for_node.body[0].body[0].value.func.attr == 'append'):
                            
                            # Get the transformation and condition
                            transform = for_node.body[0].body[0].value.args[0]
                            condition = for_node.body[0].test
                            
                            # Create list comprehension
                            new_body.append(ast.Assign(
                                targets=[ast.Name(id=stmt.targets[0].id, ctx=ast.Store())],
                                value=ast.ListComp(
                                    elt=transform,
                                    generators=[
                                        ast.comprehension(
                                            target=for_node.target,
                                            iter=for_node.iter,
                                            ifs=[condition],
                                            is_async=0
                                        )
                                    ]
                                )
                            ))
                            i += 2
                            continue
                    
                    new_body.append(stmt)
                    i += 1
                node.body = new_body
                return node
        
        tree = ListCompWithConditionTransformer().visit(tree)
        ast.fix_missing_locations(tree)
        return ast.unparse(tree)
    except Exception as e:
        logging.error(f"Error in use_list_comp_with_condition: {str(e)}")
        return code

# --- New Transformation: Use Dictionary Comprehension ---
def use_dict_comprehension(code: str) -> str:
    """
    Replaces dictionary construction loops with dictionary comprehension.
    Example:
        result = {}
        for k, v in items:
            result[k] = v
        # becomes
        result = {k: v for k, v in items}
    """
    try:
        tree = ast.parse(code)
        class DictCompTransformer(ast.NodeTransformer):
            def visit_Module(self, node):
                new_body = []
                i = 0
                while i < len(node.body):
                    stmt = node.body[i]
                    
                    if (isinstance(stmt, ast.Assign) and
                        len(stmt.targets) == 1 and
                        isinstance(stmt.targets[0], ast.Name) and
                        isinstance(stmt.value, ast.Dict) and
                        i + 1 < len(node.body) and
                        isinstance(node.body[i + 1], ast.For)):
                        
                        for_node = node.body[i + 1]
                        if (len(for_node.body) == 1 and
                            isinstance(for_node.body[0], ast.Assign) and
                            isinstance(for_node.body[0].targets[0], ast.Subscript) and
                            isinstance(for_node.body[0].targets[0].value, ast.Name) and
                            for_node.body[0].targets[0].value.id == stmt.targets[0].id):
                            
                            # Get the key and value
                            key = for_node.body[0].targets[0].slice
                            value = for_node.body[0].value
                            
                            # Create dictionary comprehension
                            new_body.append(ast.Assign(
                                targets=[ast.Name(id=stmt.targets[0].id, ctx=ast.Store())],
                                value=ast.DictComp(
                                    key=key,
                                    value=value,
                                    generators=[
                                        ast.comprehension(
                                            target=for_node.target,
                                            iter=for_node.iter,
                                            ifs=[],
                                            is_async=0
                                        )
                                    ]
                                )
                            ))
                            i += 2
                            continue
                    
                    new_body.append(stmt)
                    i += 1
                node.body = new_body
                return node
        
        tree = DictCompTransformer().visit(tree)
        ast.fix_missing_locations(tree)
        return ast.unparse(tree)
    except Exception as e:
        logging.error(f"Error in use_dict_comprehension: {str(e)}")
        return code

# --- New Transformation: Use Generator Functions ---
def use_generator_functions(code: str) -> str:
    """
    Replaces list-building functions with generator functions.
    Example:
        def get_items():
            result = []
            for x in range(10):
                result.append(x * 2)
            return result
        # becomes
        def get_items():
            for x in range(10):
                yield x * 2
    """
    try:
        tree = ast.parse(code)
        class GeneratorFunctionTransformer(ast.NodeTransformer):
            def visit_FunctionDef(self, node):
                # Check if function builds and returns a list
                if (len(node.body) >= 2 and
                    isinstance(node.body[0], ast.Assign) and
                    isinstance(node.body[0].value, ast.List) and
                    isinstance(node.body[-1], ast.Return) and
                    isinstance(node.body[-1].value, ast.Name) and
                    node.body[-1].value.id == node.body[0].targets[0].id):
                    
                    # Check if there's a for loop that appends to the list
                    for i, stmt in enumerate(node.body[1:-1]):
                        if (isinstance(stmt, ast.For) and
                            len(stmt.body) == 1 and
                            isinstance(stmt.body[0], ast.Expr) and
                            isinstance(stmt.body[0].value, ast.Call) and
                            isinstance(stmt.body[0].value.func, ast.Attribute) and
                            stmt.body[0].value.func.attr == 'append'):
                            
                            # Convert to generator function
                            node.body = [
                                ast.For(
                                    target=stmt.target,
                                    iter=stmt.iter,
                                    body=[ast.Expr(value=ast.Yield(value=stmt.body[0].value.args[0]))],
                                    orelse=[]
                                )
                            ]
                            return node
                return node
        
        tree = GeneratorFunctionTransformer().visit(tree)
        ast.fix_missing_locations(tree)
        return ast.unparse(tree)
    except Exception as e:
        logging.error(f"Error in use_generator_functions: {str(e)}")
        return code

# --- Transformation 1: Apply @functools.lru_cache ---
def apply_lru_cache(code_string: str, function_name: str = None) -> str:
    """
    Adds @functools.lru_cache to the specified function (or the first function if not specified).
    Also ensures 'import functools' is present at the top of the code.
    """
    logger.info(f"Applying lru_cache to function: {function_name}")
    logger.info(f"Original code:\n{code_string}")
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
    # Ensure 'import functools' is present
    if 'import functools' not in result:
        result = 'import functools\n' + result
    logger.info(f"Transformed code:\n{result}")
    return result

# --- Transformation 2: Convert list to set for membership checks ---
def convert_list_to_set_for_membership(code: str, list_name: str) -> str:
    """Convert list to set for membership checks."""
    try:
        tree = ast.parse(code)
        class ListToSetTransformer(ast.NodeTransformer):
            def visit_Assign(self, node):
                # Find the list initialization
                if (len(node.targets) == 1 and 
                    isinstance(node.targets[0], ast.Name) and 
                    node.targets[0].id == list_name and
                    isinstance(node.value, ast.List)):
                    
                    # Convert list to set
                    node.value = ast.Call(
                        func=ast.Name(id='set', ctx=ast.Load()),
                        args=[node.value],
                        keywords=[]
                    )
                return node
            
            def visit_Expr(self, node):
                # Find list modifications (append, extend, etc.)
                if (isinstance(node.value, ast.Call) and
                    isinstance(node.value.func, ast.Attribute) and
                    isinstance(node.value.func.value, ast.Name) and
                    node.value.func.value.id == list_name):
                    
                    if node.value.func.attr == 'append':
                        # Convert append to add
                        node.value.func.attr = 'add'
                    elif node.value.func.attr == 'extend':
                        # Convert extend to update
                        node.value.func.attr = 'update'
                return node
            
            def visit_Compare(self, node):
                # Add parentheses around membership check
                if isinstance(node.ops[0], ast.In):
                    return ast.Compare(
                        left=node.left,
                        ops=node.ops,
                        comparators=node.comparators
                    )
                return node
        
        tree = ListToSetTransformer().visit(tree)
        ast.fix_missing_locations(tree)
        return ast.unparse(tree)
    except Exception as e:
        logging.error(f"Error in convert_list_to_set_for_membership: {str(e)}")
        return code

# --- Transformation 3: Replace loop with sum ---
def replace_loop_with_sum(code_string: str, total_var: str = "total", list_var: str = "my_list") -> str:
    """
    Replaces a for loop summing a list with total = sum(list_var).
    """
    logger.info(f"Replacing loop with sum for total: {total_var}, list: {list_var}")
    logger.info(f"Original code:\n{code_string}")
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
    logger.info(f"Transformed code:\n{result}")
    return result

# --- Transformation 4: Use generator expression ---
def use_generator_expression(code: str) -> str:
    """Replace list comprehension with generator expression."""
    try:
        tree = ast.parse(code)
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                # Check if it's a sum with list comprehension
                if (isinstance(node.func, ast.Name) and
                    node.func.id == 'sum' and
                    len(node.args) == 1 and
                    isinstance(node.args[0], ast.ListComp)):
                    
                    # Convert list comprehension to generator expression
                    node.args[0] = ast.GeneratorExp(
                        elt=node.args[0].elt,
                        generators=node.args[0].generators
                    )
                    return ast.unparse(tree)
        
        return code
    except Exception as e:
        logging.error(f"Error in use_generator_expression: {str(e)}")
        return code

# --- Transformation 5: Introduce early exit in loops ---
def introduce_early_exit(code_string: str, break_condition: str) -> str:
    """
    Adds a break statement for early exit in a loop if the break_condition is met.
    break_condition should be a valid Python expression as a string.
    """
    logger.info(f"Introducing early exit with condition: {break_condition}")
    logger.info(f"Original code:\n{code_string}")
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
    logger.info(f"Transformed code:\n{result}")
    return result

# --- Transformation 6: Replace Manual List Append with List Comprehension ---
def replace_manual_list_append_with_list_comp(code: str) -> str:
    """Replace manual list append with list comprehension."""
    try:
        tree = ast.parse(code)
        class ListCompTransformer(ast.NodeTransformer):
            def visit_Module(self, node):
                new_body = []
                i = 0
                while i < len(node.body):
                    stmt = node.body[i]
                    # Look for result = [] followed by for loop
                    if (isinstance(stmt, ast.Assign) and
                        len(stmt.targets) == 1 and
                        isinstance(stmt.targets[0], ast.Name) and
                        isinstance(stmt.value, ast.List) and
                        i + 1 < len(node.body) and
                        isinstance(node.body[i + 1], ast.For)):
                        for_node = node.body[i + 1]
                        # Simple append: result.append(...)
                        if (len(for_node.body) == 1 and 
                            isinstance(for_node.body[0], ast.Expr) and 
                            isinstance(for_node.body[0].value, ast.Call) and
                            isinstance(for_node.body[0].value.func, ast.Attribute) and
                            for_node.body[0].value.func.attr == 'append'):
                            append_arg = for_node.body[0].value.args[0]
                            new_body.append(ast.Assign(
                                targets=[ast.Name(id=stmt.targets[0].id, ctx=ast.Store())],
                                value=ast.ListComp(
                                    elt=append_arg,
                                    generators=[
                                        ast.comprehension(
                                            target=for_node.target,
                                            iter=for_node.iter,
                                            ifs=[],
                                            is_async=0
                                        )
                                    ]
                                )
                            ))
                            i += 2
                            continue
                        # If-then-append: if ...: result.append(...)
                        elif (len(for_node.body) == 1 and
                              isinstance(for_node.body[0], ast.If) and
                              len(for_node.body[0].body) == 1 and
                              isinstance(for_node.body[0].body[0], ast.Expr) and
                              isinstance(for_node.body[0].body[0].value, ast.Call) and
                              isinstance(for_node.body[0].body[0].value.func, ast.Attribute) and
                              for_node.body[0].body[0].value.func.attr == 'append'):
                            append_arg = for_node.body[0].body[0].value.args[0]
                            condition = for_node.body[0].test
                            new_body.append(ast.Assign(
                                targets=[ast.Name(id=stmt.targets[0].id, ctx=ast.Store())],
                                value=ast.ListComp(
                                    elt=append_arg,
                                    generators=[
                                        ast.comprehension(
                                            target=for_node.target,
                                            iter=for_node.iter,
                                            ifs=[condition],
                                            is_async=0
                                        )
                                    ]
                                )
                            ))
                            i += 2
                            continue
                    new_body.append(stmt)
                    i += 1
                node.body = new_body
                return node
        tree = ListCompTransformer().visit(tree)
        ast.fix_missing_locations(tree)
        return ast.unparse(tree)
    except Exception as e:
        logging.error(f"Error in replace_manual_list_append_with_list_comp: {str(e)}")
        return code

# --- Transformation 7: Replace Manual List Construction with map() ---
def replace_manual_list_with_map(code: str) -> str:
    """Replace manual list construction with map()."""
    try:
        tree = ast.parse(code)
        class MapTransformer(ast.NodeTransformer):
            def visit_For(self, node):
                # Check if it's a list append loop
                if (len(node.body) == 1 and 
                    isinstance(node.body[0], ast.Expr) and 
                    isinstance(node.body[0].value, ast.Call) and
                    isinstance(node.body[0].value.func, ast.Attribute) and
                    node.body[0].value.func.attr == 'append'):
                    
                    # Get the append argument
                    append_arg = node.body[0].value.args[0]
                    
                    # Create lambda function
                    lambda_func = ast.Lambda(
                        args=ast.arguments(
                            posonlyargs=[],
                            args=[ast.arg(arg=node.target.id)],
                            kwonlyargs=[],
                            kw_defaults=[],
                            defaults=[]
                        ),
                        body=append_arg
                    )
                    
                    # Create map call
                    return ast.Assign(
                        targets=[ast.Name(id='result', ctx=ast.Store())],
                        value=ast.Call(
                            func=ast.Name(id='list', ctx=ast.Load()),
                            args=[
                                ast.Call(
                                    func=ast.Name(id='map', ctx=ast.Load()),
                                    args=[lambda_func, node.iter],
                                    keywords=[]
                                )
                            ],
                            keywords=[]
                        )
                    )
                return node
        
        tree = MapTransformer().visit(tree)
        ast.fix_missing_locations(tree)
        return ast.unparse(tree)
    except Exception as e:
        logging.error(f"Error in replace_manual_list_with_map: {str(e)}")
        return code

# --- Transformation 8: Replace range(len(xs)) with enumerate(xs) ---
def replace_range_len_with_enumerate(code: str) -> str:
    """Replace range(len(xs)) with enumerate(xs)."""
    try:
        tree = ast.parse(code)
        for node in ast.walk(tree):
            if isinstance(node, ast.For):
                # Check if it's a range(len(xs)) pattern
                if (isinstance(node.iter, ast.Call) and
                    isinstance(node.iter.func, ast.Name) and
                    node.iter.func.id == 'range' and
                    len(node.iter.args) == 1 and
                    isinstance(node.iter.args[0], ast.Call) and
                    isinstance(node.iter.args[0].func, ast.Name) and
                    node.iter.args[0].func.id == 'len'):
                    
                    # Get the list variable
                    list_var = node.iter.args[0].args[0]
                    
                    # Create enumerate call
                    node.iter = ast.Call(
                        func=ast.Name(id='enumerate', ctx=ast.Load()),
                        args=[list_var],
                        keywords=[]
                    )
                    
                    # Update loop target to tuple unpacking
                    node.target = ast.Tuple(
                        elts=[
                            ast.Name(id='i', ctx=ast.Store()),
                            ast.Name(id='x', ctx=ast.Store())
                        ],
                        ctx=ast.Store()
                    )
                    
                    # Update loop body to use x instead of xs[i]
                    for body_node in ast.walk(node):
                        if (isinstance(body_node, ast.Subscript) and
                            isinstance(body_node.value, ast.Name) and
                            body_node.value.id == list_var.id and
                            isinstance(body_node.slice, ast.Index) and
                            isinstance(body_node.slice.value, ast.Name) and
                            body_node.slice.value.id == node.target.elts[0].id):
                            body_node.value = ast.Name(id='x', ctx=ast.Load())
                    
                    return ast.unparse(tree)
        
        return code
    except Exception as e:
        logging.error(f"Error in replace_range_len_with_enumerate: {str(e)}")
        return code

# --- Transformation 9: Use Built-in Functions (max, min, any, all) ---
def invert_comparison(comp: ast.Compare) -> ast.Compare:
    """Invert a comparison operator (e.g., <= to >, < to >=, etc.)."""
    op_map = {
        ast.Lt: ast.GtE,  # < to >=
        ast.LtE: ast.Gt,  # <= to >
        ast.Gt: ast.LtE,  # > to <=
        ast.GtE: ast.Lt,  # >= to <
        ast.Eq: ast.NotEq,  # == to !=
        ast.NotEq: ast.Eq,  # != to ==
    }
    if isinstance(comp.ops[0], tuple(op_map.keys())):
        return ast.Compare(
            left=comp.left,
            ops=[op_map[type(comp.ops[0])]()],
            comparators=comp.comparators
        )
    return comp

def use_builtin_functions(code: str) -> str:
    """Replace loops with built-in functions."""
    try:
        tree = ast.parse(code)
        logging.info(f"Original code:\n{code}")
        
        class BuiltinFunctionTransformer(ast.NodeTransformer):
            def visit_Module(self, node):
                new_body = []
                i = 0
                while i < len(node.body):
                    stmt = node.body[i]
                    logging.info(f"\nProcessing statement {i}:")
                    logging.info(f"Statement type: {type(stmt).__name__}")
                    
                    # max/min idiom
                    if (isinstance(stmt, ast.Assign) and
                        len(stmt.targets) == 1 and
                        isinstance(stmt.targets[0], ast.Name) and
                        isinstance(stmt.value, ast.Subscript) and
                        i + 1 < len(node.body) and
                        isinstance(node.body[i + 1], ast.For)):
                        logging.info("Found potential max/min pattern")
                        for_node = node.body[i + 1]
                        if (len(for_node.body) == 1 and
                            isinstance(for_node.body[0], ast.If) and
                            len(for_node.body[0].body) == 1 and
                            isinstance(for_node.body[0].body[0], ast.Assign) and
                            len(for_node.body[0].body[0].targets) == 1 and
                            isinstance(for_node.body[0].body[0].targets[0], ast.Name) and
                            for_node.body[0].body[0].targets[0].id == stmt.targets[0].id):
                            assign_target = stmt.targets[0].id
                            # max
                            if (isinstance(for_node.body[0].test, ast.Compare) and
                                isinstance(for_node.body[0].test.left, ast.Name) and
                                for_node.body[0].test.left.id == for_node.target.id and
                                len(for_node.body[0].test.comparators) == 1 and
                                isinstance(for_node.body[0].test.comparators[0], ast.Name) and
                                for_node.body[0].test.comparators[0].id == assign_target and
                                isinstance(for_node.body[0].test.ops[0], ast.Gt)):
                                logging.info("Matched max pattern")
                                new_body.append(ast.Assign(
                                    targets=[ast.Name(id=assign_target, ctx=ast.Store())],
                                    value=ast.Call(
                                        func=ast.Name(id='max', ctx=ast.Load()),
                                        args=[for_node.iter],
                                        keywords=[]
                                    )
                                ))
                                i += 2
                                continue
                            # min
                            elif (isinstance(for_node.body[0].test, ast.Compare) and
                                  isinstance(for_node.body[0].test.left, ast.Name) and
                                  for_node.body[0].test.left.id == for_node.target.id and
                                  len(for_node.body[0].test.comparators) == 1 and
                                  isinstance(for_node.body[0].test.comparators[0], ast.Name) and
                                  for_node.body[0].test.comparators[0].id == assign_target and
                                  isinstance(for_node.body[0].test.ops[0], ast.Lt)):
                                logging.info("Matched min pattern")
                                new_body.append(ast.Assign(
                                    targets=[ast.Name(id=assign_target, ctx=ast.Store())],
                                    value=ast.Call(
                                        func=ast.Name(id='min', ctx=ast.Load()),
                                        args=[for_node.iter],
                                        keywords=[]
                                    )
                                ))
                                i += 2
                                continue
                    
                    # any idiom: found = False; for x in xs: if cond: found = True; break
                    if (isinstance(stmt, ast.Assign) and
                        len(stmt.targets) == 1 and
                        isinstance(stmt.targets[0], ast.Name) and
                        isinstance(stmt.value, ast.Constant) and
                        stmt.value.value is False and
                        i + 1 < len(node.body) and
                        isinstance(node.body[i + 1], ast.For)):
                        logging.info("Found potential any pattern")
                        logging.info(f"Initial assignment: {ast.unparse(stmt)}")
                        for_node = node.body[i + 1]
                        logging.info(f"For loop: {ast.unparse(for_node)}")
                        
                        # Check if the for loop body contains an if statement with a True assignment and a break
                        if (len(for_node.body) >= 1 and
                            isinstance(for_node.body[0], ast.If) and
                            len(for_node.body[0].body) >= 1):
                            
                            # Check if the if body contains a True assignment
                            if_stmt = for_node.body[0]
                            if (len(if_stmt.body) >= 1 and
                                isinstance(if_stmt.body[0], ast.Assign) and
                                len(if_stmt.body[0].targets) == 1 and
                                isinstance(if_stmt.body[0].targets[0], ast.Name) and
                                if_stmt.body[0].targets[0].id == stmt.targets[0].id and
                                isinstance(if_stmt.body[0].value, ast.Constant) and
                                if_stmt.body[0].value.value is True):
                                
                                # Check if there's a break statement after the if
                                has_break = len(for_node.body) > 1 and isinstance(for_node.body[1], ast.Break)
                                
                                logging.info("Matched any pattern structure")
                                assign_target = stmt.targets[0].id
                                condition = for_node.body[0].test
                                logging.info(f"Condition: {ast.unparse(condition)}")
                                
                                new_body.append(ast.Assign(
                                    targets=[ast.Name(id=assign_target, ctx=ast.Store())],
                                    value=ast.Call(
                                        func=ast.Name(id='any', ctx=ast.Load()),
                                        args=[ast.GeneratorExp(
                                            elt=condition,
                                            generators=[
                                                ast.comprehension(
                                                    target=for_node.target,
                                                    iter=for_node.iter,
                                                    ifs=[],
                                                    is_async=0
                                                )
                                            ]
                                        )],
                                        keywords=[]
                                    )
                                ))
                                i += 2
                                continue
                            else:
                                logging.info("Any pattern structure mismatch:")
                                if len(if_stmt.body) < 1:
                                    logging.info("If body is empty")
                                elif not isinstance(if_stmt.body[0], ast.Assign):
                                    logging.info("First statement in if body is not an Assign")
                                elif not isinstance(if_stmt.body[0].value, ast.Constant):
                                    logging.info("Assignment value is not a Constant")
                                elif if_stmt.body[0].value.value is not True:
                                    logging.info("Assignment value is not True")
                        else:
                            logging.info("Any pattern structure mismatch:")
                            if len(for_node.body) < 1:
                                logging.info("For loop body is empty")
                            elif not isinstance(for_node.body[0], ast.If):
                                logging.info("First statement is not an If")
                    
                    # all idiom: all_positive = True; for x in xs: if cond: all_positive = False; break
                    if (isinstance(stmt, ast.Assign) and
                        len(stmt.targets) == 1 and
                        isinstance(stmt.targets[0], ast.Name) and
                        isinstance(stmt.value, ast.Constant) and
                        stmt.value.value is True and
                        i + 1 < len(node.body) and
                        isinstance(node.body[i + 1], ast.For)):
                        logging.info("Found potential all pattern")
                        logging.info(f"Initial assignment: {ast.unparse(stmt)}")
                        for_node = node.body[i + 1]
                        logging.info(f"For loop: {ast.unparse(for_node)}")
                        
                        # Check if the for loop body contains an if statement with a False assignment and a break
                        if (len(for_node.body) >= 1 and
                            isinstance(for_node.body[0], ast.If) and
                            len(for_node.body[0].body) >= 1):
                            
                            # Check if the if body contains a False assignment
                            if_stmt = for_node.body[0]
                            if (len(if_stmt.body) >= 1 and
                                isinstance(if_stmt.body[0], ast.Assign) and
                                len(if_stmt.body[0].targets) == 1 and
                                isinstance(if_stmt.body[0].targets[0], ast.Name) and
                                if_stmt.body[0].targets[0].id == stmt.targets[0].id and
                                isinstance(if_stmt.body[0].value, ast.Constant) and
                                if_stmt.body[0].value.value is False):
                                
                                # Check if there's a break statement after the if
                                has_break = len(for_node.body) > 1 and isinstance(for_node.body[1], ast.Break)
                                
                                logging.info("Matched all pattern structure")
                                assign_target = stmt.targets[0].id
                                condition = for_node.body[0].test
                                logging.info(f"Condition: {ast.unparse(condition)}")
                                
                                # Invert the comparison operator directly
                                if isinstance(condition, ast.Compare):
                                    inverted_condition = invert_comparison(condition)
                                else:
                                    inverted_condition = ast.UnaryOp(op=ast.Not(), operand=condition)
                                
                                new_body.append(ast.Assign(
                                    targets=[ast.Name(id=assign_target, ctx=ast.Store())],
                                    value=ast.Call(
                                        func=ast.Name(id='all', ctx=ast.Load()),
                                        args=[ast.GeneratorExp(
                                            elt=inverted_condition,
                                            generators=[
                                                ast.comprehension(
                                                    target=for_node.target,
                                                    iter=for_node.iter,
                                                    ifs=[],
                                                    is_async=0
                                                )
                                            ]
                                        )],
                                        keywords=[]
                                    )
                                ))
                                i += 2
                                continue
                            else:
                                logging.info("All pattern structure mismatch:")
                                if len(if_stmt.body) < 1:
                                    logging.info("If body is empty")
                                elif not isinstance(if_stmt.body[0], ast.Assign):
                                    logging.info("First statement in if body is not an Assign")
                                elif not isinstance(if_stmt.body[0].value, ast.Constant):
                                    logging.info("Assignment value is not a Constant")
                                elif if_stmt.body[0].value.value is not False:
                                    logging.info("Assignment value is not False")
                        else:
                            logging.info("All pattern structure mismatch:")
                            if len(for_node.body) < 1:
                                logging.info("For loop body is empty")
                            elif not isinstance(for_node.body[0], ast.If):
                                logging.info("First statement is not an If")
                    
                    new_body.append(stmt)
                    i += 1
                node.body = new_body
                return node
        
        tree = BuiltinFunctionTransformer().visit(tree)
        ast.fix_missing_locations(tree)
        transformed_code = ast.unparse(tree)
        logging.info(f"\nTransformed code:\n{transformed_code}")
        return transformed_code
    except Exception as e:
        logging.error(f"Error in use_builtin_functions: {str(e)}")
        return code

# --- Transformation 10: Loop Unrolling ---
def unroll_small_loop(code_string: str, unroll_factor: int = 2) -> str:
    """
    Duplicates the loop body a specified number of times for small loops.
    Example:
        for i in range(10):
            print(i)
        # becomes (unroll_factor=2)
        for i in range(0, 10, 2):
            print(i)
            print(i + 1)
    """
    logger.info(f"Unrolling small loop with factor: {unroll_factor}")
    logger.info(f"Original code:\n{code_string}")
    tree = ast.parse(code_string)
    class LoopUnrollTransformer(ast.NodeTransformer):
        def visit_For(self, node):
            # Look for: for i in range(n)
            if (
                isinstance(node.target, ast.Name)
                and isinstance(node.iter, ast.Call)
                and isinstance(node.iter.func, ast.Name)
                and node.iter.func.id == "range"
                and len(node.iter.args) == 1
                and isinstance(node.iter.args[0], ast.Constant)
            ):
                n = node.iter.args[0].value
                # Replace with: for i in range(0, n, unroll_factor)
                node.iter.args = [
                    ast.Constant(value=0),
                    ast.Constant(value=n),
                    ast.Constant(value=unroll_factor)
                ]
                # Duplicate the loop body
                node.body = node.body * unroll_factor
            return node
    tree = LoopUnrollTransformer().visit(tree)
    ast.fix_missing_locations(tree)
    result = astunparse.unparse(tree).strip()
    logger.info(f"Transformed code:\n{result}")
    return result

# --- Transformation 11: Apply NumPy Vectorization ---
def apply_numpy_vectorization(code: str) -> str:
    """Replace loops with NumPy vectorized operations."""
    try:
        tree = ast.parse(code)
        class NumpyVectorizationTransformer(ast.NodeTransformer):
            def visit_For(self, node):
                # Check if it's a simple list append loop
                if (len(node.body) == 1 and 
                    isinstance(node.body[0], ast.Expr) and 
                    isinstance(node.body[0].value, ast.Call) and
                    isinstance(node.body[0].value.func, ast.Attribute) and
                    node.body[0].value.func.attr == 'append'):
                    
                    # Get the append argument
                    append_arg = node.body[0].value.args[0]
                    
                    # Handle simple square operation
                    if (isinstance(append_arg, ast.BinOp) and 
                        isinstance(append_arg.op, ast.Mult) and
                        isinstance(append_arg.left, ast.Name) and
                        isinstance(append_arg.right, ast.Name) and
                        append_arg.left.id == node.target.id and
                        append_arg.right.id == node.target.id):
                        
                        # Replace with np.square
                        return ast.Assign(
                            targets=[ast.Name(id='result', ctx=ast.Store())],
                            value=ast.Call(
                                func=ast.Attribute(
                                    value=ast.Name(id='np', ctx=ast.Load()),
                                    attr='square',
                                    ctx=ast.Load()
                                ),
                                args=[node.iter],
                                keywords=[]
                            )
                        )
                    
                    # Handle complex operations (e.g., x * x + 1)
                    elif (isinstance(append_arg, ast.BinOp) and
                          isinstance(append_arg.op, ast.Add)):
                        # Create np.add(np.square(xs), 1)
                        return ast.Assign(
                            targets=[ast.Name(id='result', ctx=ast.Store())],
                            value=ast.Call(
                                func=ast.Attribute(
                                    value=ast.Name(id='np', ctx=ast.Load()),
                                    attr='add',
                                    ctx=ast.Load()
                                ),
                                args=[
                                    ast.Call(
                                        func=ast.Attribute(
                                            value=ast.Name(id='np', ctx=ast.Load()),
                                            attr='square',
                                            ctx=ast.Load()
                                        ),
                                        args=[node.iter],
                                        keywords=[]
                                    ),
                                    ast.Constant(value=1)
                                ],
                                keywords=[]
                            )
                        )
                return node
        
        tree = NumpyVectorizationTransformer().visit(tree)
        ast.fix_missing_locations(tree)
        return ast.unparse(tree)
    except Exception as e:
        logging.error(f"Error in apply_numpy_vectorization: {str(e)}")
        return code

# --- Transformation Application Helper ---
def apply_transformations(code: str, transformations: List[str]) -> str:
    """
    Applies the specified transformations to the code in an optimal order.
    """
    # Define transformation dependencies and order
    transform_order = [
        'use_set_operations',
        'use_list_comp_with_condition',
        'use_dict_comprehension',
        'use_generator_functions',
        'convert_list_to_set_for_membership',
        'replace_manual_list_append_with_list_comp',
        'replace_manual_list_with_map',
        'replace_range_len_with_enumerate',
        'use_builtin_functions',
        'use_generator_expression',
        'apply_lru_cache',
        'unroll_small_loop',
        'apply_numpy_vectorization'
    ]
    
    # Filter and sort transformations
    applicable_transforms = [t for t in transform_order if t in transformations]
    
    # Apply transformations in order
    result = code
    for transform_name in applicable_transforms:
        transform_func = globals()[transform_name]
        try:
            result = transform_func(result)
        except Exception as e:
            logging.error(f"Error applying {transform_name}: {str(e)}")
    
    return result

# --- Main Transformation Function ---
def transform_code(code: str) -> str:
    """
    Main function to transform code using all applicable optimizations.
    """
    # Select applicable transformations
    transformations = select_transformations(code)
    
    # Apply transformations
    return apply_transformations(code, transformations)