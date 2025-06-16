"""
Tests for code transformation functions.
"""

import unittest
from ares_optimizer.code_transformer.transformations import (
    apply_lru_cache,
    convert_list_to_set_for_membership,
    replace_loop_with_sum,
    use_generator_expression,
    introduce_early_exit,
    replace_manual_list_append_with_list_comp,
    replace_manual_list_with_map,
    replace_range_len_with_enumerate,
    use_builtin_functions,
    unroll_small_loop,
    apply_numpy_vectorization
)


class TestTransformations(unittest.TestCase):
    def test_apply_lru_cache(self):
        """Test applying @functools.lru_cache decorator."""
        # Test with default function (first function)
        code = """
def add(a, b):
    return a + b

def subtract(a, b):
    return a - b
"""
        transformed_code = apply_lru_cache(code)
        self.assertIn("@functools.lru_cache", transformed_code)
        self.assertIn("def add", transformed_code)
        self.assertIn("def subtract", transformed_code)

        # Test with specific function
        transformed_code = apply_lru_cache(code, "subtract")
        self.assertNotIn("@functools.lru_cache", transformed_code.split("def add")[0])
        self.assertIn("@functools.lru_cache", transformed_code.split("def subtract")[0])

        # Test with nested function
        code = """
def outer():
    def inner():
        return 42
    return inner()
"""
        transformed_code = apply_lru_cache(code, "inner")
        self.assertIn("@functools.lru_cache", transformed_code)

    def test_convert_list_to_set_for_membership(self):
        """Test converting list to set for membership checks."""
        code = """
my_list = [1, 2, 3, 4, 5]
if 3 in my_list:
    print("Found!")
"""
        transformed_code = convert_list_to_set_for_membership(code, "my_list")
        self.assertIn("my_list = set([1, 2, 3, 4, 5])", transformed_code)
        self.assertIn("if 3 in my_list:", transformed_code)

        # Test with non-existent list variable
        code = """
other_list = [1, 2, 3]
if 3 in my_list:
    print("Found!")
"""
        transformed_code = convert_list_to_set_for_membership(code, "my_list")
        self.assertIn("other_list = [1, 2, 3]", transformed_code)
        self.assertIn("if 3 in my_list:", transformed_code)

        # Test with modified list
        code = """
my_list = [1, 2, 3]
my_list.append(4)
if 3 in my_list:
    print("Found!")
"""
        transformed_code = convert_list_to_set_for_membership(code, "my_list")
        self.assertIn("my_list = set([1, 2, 3])", transformed_code)
        self.assertIn("my_list.add(4)", transformed_code)

    def test_replace_loop_with_sum(self):
        """Test replacing loop with sum function."""
        # Test basic summation loop
        code = """
total = 0
for x in my_list:
    total += x
"""
        transformed_code = replace_loop_with_sum(code)
        self.assertIn("total = sum(my_list)", transformed_code)

        # Test with custom variable names
        code = """
result = 0
for num in numbers:
    result += num
"""
        transformed_code = replace_loop_with_sum(code, "result", "numbers")
        self.assertIn("result = sum(numbers)", transformed_code)

        # Test with non-matching loop
        code = """
total = 0
for x in my_list:
    total += x * 2
"""
        transformed_code = replace_loop_with_sum(code)
        self.assertNotIn("total = sum(my_list)", transformed_code)

        # Test with loop variable used elsewhere
        code = """
total = 0
for x in my_list:
    total += x
    print(x)
"""
        transformed_code = replace_loop_with_sum(code)
        self.assertNotIn("total = sum(my_list)", transformed_code)

    def test_use_generator_expression(self):
        """Test replacing list comprehension with generator expression."""
        # Test basic sum with list comprehension
        code = """
result = sum([x * x for x in range(10)])
"""
        transformed_code = use_generator_expression(code)
        self.assertTrue(
            "sum(((x * x) for x in range(10)))" in transformed_code or
            "sum((x * x for x in range(10)))" in transformed_code
        )

        # Test nested list comprehension
        code = """
result = sum([x * y for x in range(5) for y in range(5)])
"""
        transformed_code = use_generator_expression(code)
        self.assertTrue(
            "sum(((x * y) for x in range(5) for y in range(5)))" in transformed_code or
            "sum((x * y for x in range(5) for y in range(5)))" in transformed_code
        )

        # Test with non-sum function
        code = """
result = max([x * x for x in range(10)])
"""
        transformed_code = use_generator_expression(code)
        self.assertTrue("max([(" in transformed_code or "max([x * x" in transformed_code)

        # Test with complex comprehension
        code = """
result = sum([x * x for x in range(10) if x % 2 == 0])
"""
        transformed_code = use_generator_expression(code)
        self.assertTrue(
            "sum(((x * x) for x in range(10) if x % 2 == 0))" in transformed_code or
            "sum((x * x for x in range(10) if x % 2 == 0))" in transformed_code
        )

    def test_introduce_early_exit(self):
        """Test introducing early exit in loops."""
        # Test basic early exit
        code = """
for i in range(10):
    print(i)
"""
        transformed_code = introduce_early_exit(code, "i > 5")
        self.assertIn("if (i > 5):", transformed_code)
        self.assertIn("break", transformed_code)

        # Test with nested loops
        code = """
for i in range(5):
    for j in range(5):
        print(i, j)
"""
        transformed_code = introduce_early_exit(code, "i + j > 5")
        # Accept any if statement with i + j > 5 and break
        self.assertIn("i+j", transformed_code.replace(" ", ""))
        self.assertIn("break", transformed_code)

        # Test with complex condition
        code = """
for item in items:
    if item > 0:
        print(item)
"""
        transformed_code = introduce_early_exit(code, "item < 0")
        self.assertIn("if (item < 0):", transformed_code)
        self.assertIn("break", transformed_code)

        # Test with loop variable used after break
        code = """
for i in range(10):
    print(i)
    if i > 5:
        break
print(i)
"""
        transformed_code = introduce_early_exit(code, "i > 5")
        self.assertIn("if (i > 5):", transformed_code)
        self.assertIn("break", transformed_code)

    def test_replace_manual_list_append_with_list_comp(self):
        code = "result = []\nfor x in my_list:\n    result.append(x * x)"
        transformed_code = replace_manual_list_append_with_list_comp(code)
        self.assertTrue(
            "result = [x * x for x in my_list]" in transformed_code or
            "result = [(x * x) for x in my_list]" in transformed_code
        )

        # Test with complex append logic
        code = "result = []\nfor x in my_list:\n    if x > 0:\n        result.append(x * x)"
        transformed_code = replace_manual_list_append_with_list_comp(code)
        self.assertTrue(
            "result = [x * x for x in my_list if x > 0]" in transformed_code or
            "result = [(x * x) for x in my_list if x > 0]" in transformed_code
        )

        # Test with list modified elsewhere
        code = "result = []\nfor x in my_list:\n    result.append(x * x)\nresult.append(42)"
        transformed_code = replace_manual_list_append_with_list_comp(code)
        self.assertTrue(
            "result = [x * x for x in my_list]" in transformed_code or
            "result = [(x * x) for x in my_list]" in transformed_code
        )

    def test_replace_manual_list_with_map(self):
        code = "result = []\nfor x in my_list:\n    result.append(x * x)"
        transformed_code = replace_manual_list_with_map(code)
        self.assertTrue(
            "result = list(map(lambda x: x * x, my_list))" in transformed_code or
            "result = list(map((x * x), my_list))" in transformed_code or
            "result = list(map((lambda x: (x * x)), my_list))" in transformed_code
        )

        # Test with complex lambda function
        code = "result = []\nfor x in my_list:\n    result.append(x * x + 1)"
        transformed_code = replace_manual_list_with_map(code)
        self.assertTrue(
            "result = list(map(lambda x: x * x + 1, my_list))" in transformed_code or
            "result = list(map((x * x + 1), my_list))" in transformed_code or
            "result = list(map((lambda x: (x * x + 1)), my_list))" in transformed_code
        )

        # Test with list modified elsewhere
        code = "result = []\nfor x in my_list:\n    result.append(x * x)\nresult.append(42)"
        transformed_code = replace_manual_list_with_map(code)
        self.assertTrue(
            "result = list(map(lambda x: x * x, my_list))" in transformed_code or
            "result = list(map((x * x), my_list))" in transformed_code or
            "result = list(map((lambda x: (x * x)), my_list))" in transformed_code
        )

    def test_replace_range_len_with_enumerate(self):
        code = "for i in range(len(xs)):\n    print(i, xs[i])"
        transformed_code = replace_range_len_with_enumerate(code)
        self.assertTrue(
            "for i, x in enumerate(xs):" in transformed_code or
            "for (i, x) in enumerate(xs):" in transformed_code
        )

        # Test with index used elsewhere
        code = "for i in range(len(xs)):\n    print(i, xs[i])\n    if i > 5:\n        break"
        transformed_code = replace_range_len_with_enumerate(code)
        self.assertTrue(
            "for i, x in enumerate(xs):" in transformed_code or
            "for (i, x) in enumerate(xs):" in transformed_code
        )

        # Test with complex loop body
        code = "for i in range(len(xs)):\n    if xs[i] > 0:\n        print(i, xs[i])"
        transformed_code = replace_range_len_with_enumerate(code)
        self.assertTrue(
            "for i, x in enumerate(xs):" in transformed_code or
            "for (i, x) in enumerate(xs):" in transformed_code
        )

    def test_use_builtin_functions(self):
        code = "max_val = xs[0]\nfor x in xs:\n    if x > max_val:\n        max_val = x"
        transformed_code = use_builtin_functions(code)
        self.assertIn("max_val = max(xs)", transformed_code)

        # Test with min function
        code = "min_val = xs[0]\nfor x in xs:\n    if x < min_val:\n        min_val = x"
        transformed_code = use_builtin_functions(code)
        self.assertIn("min_val = min(xs)", transformed_code)

        # Test with any function
        code = "found = False\nfor x in xs:\n    if x > 0:\n        found = True\n        break"
        transformed_code = use_builtin_functions(code)
        self.assertTrue(
            "found = any(x > 0 for x in xs)" in transformed_code or
            "found = any((x > 0 for x in xs))" in transformed_code
        )

        # Test with all function
        code = "all_positive = True\nfor x in xs:\n    if x <= 0:\n        all_positive = False\n        break"
        transformed_code = use_builtin_functions(code)
        self.assertTrue(
            "all_positive = all(x > 0 for x in xs)" in transformed_code or
            "all_positive = all((x > 0 for x in xs))" in transformed_code
        )

    def test_unroll_small_loop(self):
        code = "for i in range(10):\n    print(i)"
        transformed_code = unroll_small_loop(code, 2)
        self.assertTrue(
            "for i in range(0, 10, 2):" in transformed_code or
            "for i in range(0, 10, 2):" in transformed_code
        )

        # Test with loop variable used elsewhere
        code = "for i in range(10):\n    print(i)\n    if i > 5:\n        break"
        transformed_code = unroll_small_loop(code, 2)
        self.assertTrue(
            "for i in range(0, 10, 2):" in transformed_code or
            "for i in range(0, 10, 2):" in transformed_code
        )

        # Test with complex loop body
        code = "for i in range(10):\n    if i % 2 == 0:\n        print(i)"
        transformed_code = unroll_small_loop(code, 2)
        self.assertTrue(
            "for i in range(0, 10, 2):" in transformed_code or
            "for i in range(0, 10, 2):" in transformed_code
        )

    def test_apply_numpy_vectorization(self):
        code = "result = []\nfor x in xs:\n    result.append(x * x)"
        transformed_code = apply_numpy_vectorization(code)
        self.assertIn("result = np.square(xs)", transformed_code)

        # Test with complex operation
        code = "result = []\nfor x in xs:\n    result.append(x * x + 1)"
        transformed_code = apply_numpy_vectorization(code)
        self.assertIn("result = np.add(np.square(xs), 1)", transformed_code)

        # Test with list modified elsewhere
        code = "result = []\nfor x in xs:\n    result.append(x * x)\nresult.append(42)"
        transformed_code = apply_numpy_vectorization(code)
        self.assertIn("result = np.square(xs)", transformed_code)


if __name__ == "__main__":
    unittest.main() 