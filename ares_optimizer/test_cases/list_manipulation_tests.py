"""Test cases for list manipulation functions."""

# Test cases for find_duplicates
DUPLICATES_TESTS = [
    ([1, 2, 3, 4, 5], []),  # No duplicates
    ([1, 2, 2, 3, 4, 4], [2, 4]),  # Multiple duplicates
    ([1, 1, 1], [1]),  # All same elements
    ([], []),  # Empty list
    ([1, 2, 3, 1, 2, 3], [1, 2, 3]),  # Multiple duplicates
]

# Test cases for reverse_in_place
REVERSE_TESTS = [
    ([1, 2, 3, 4, 5], [5, 4, 3, 2, 1]),  # Basic case
    ([1, 2, 3], [3, 2, 1]),  # Odd length
    ([1], [1]),  # Single element
    ([], []),  # Empty list
    ([1, 2, 3, 4], [4, 3, 2, 1]),  # Even length
]

# Test cases for remove_duplicates
REMOVE_DUPLICATES_TESTS = [
    ([1, 2, 3, 4, 5], [1, 2, 3, 4, 5]),  # No duplicates
    ([1, 2, 2, 3, 4, 4], [1, 2, 3, 4]),  # Multiple duplicates
    ([1, 1, 1], [1]),  # All same elements
    ([], []),  # Empty list
    ([1, 2, 3, 1, 2, 3], [1, 2, 3]),  # Multiple duplicates
]

# Test cases for find_common_elements
COMMON_ELEMENTS_TESTS = [
    ([1, 2, 3], [4, 5, 6], []),  # No common elements
    ([1, 2, 3], [3, 4, 5], [3]),  # One common element
    ([1, 2, 3], [1, 2, 3], [1, 2, 3]),  # All elements common
    ([], [1, 2, 3], []),  # Empty list
    ([1, 2, 2, 3], [2, 3, 3, 4], [2, 3]),  # Duplicates in both lists
]

# Test cases for rotate_list
ROTATE_TESTS = [
    ([1, 2, 3, 4, 5], 2, [4, 5, 1, 2, 3]),  # Rotate by 2
    ([1, 2, 3], 0, [1, 2, 3]),  # No rotation
    ([1, 2, 3], 3, [1, 2, 3]),  # Full rotation
    ([1, 2, 3], 5, [2, 3, 1]),  # Rotation > length
    ([], 1, []),  # Empty list
]

# Combine all test cases
TEST_CASES = {
    "find_duplicates": DUPLICATES_TESTS,
    "reverse_in_place": REVERSE_TESTS,
    "remove_duplicates": REMOVE_DUPLICATES_TESTS,
    "find_common_elements": COMMON_ELEMENTS_TESTS,
    "rotate_list": ROTATE_TESTS,
} 