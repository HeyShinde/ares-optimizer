"""Test cases for data structure usage functions."""

# Test cases for check_membership_large_list
MEMBERSHIP_TESTS = [
    ([1, 2, 3, 4, 5], 3, True),  # Item exists
    ([1, 2, 3, 4, 5], 6, False),  # Item doesn't exist
    ([], 1, False),  # Empty list
    ([1, 1, 1], 1, True),  # Multiple occurrences
    (["a", "b", "c"], "b", True),  # String items
]

# Test cases for count_frequency_dict
FREQUENCY_TESTS = [
    ([1, 2, 2, 3, 3, 3], {1: 1, 2: 2, 3: 3}),  # Numbers with different frequencies
    ([], {}),  # Empty list
    (["a", "a", "b"], {"a": 2, "b": 1}),  # String items
    ([1, 1, 1, 1], {1: 4}),  # All same items
    ([1, 2, 3], {1: 1, 2: 1, 3: 1}),  # All unique items
]

# Test cases for find_intersection_sets
INTERSECTION_TESTS = [
    ({1, 2, 3}, {2, 3, 4}, {2, 3}),  # Partial intersection
    ({1, 2, 3}, {4, 5, 6}, set()),  # No intersection
    ({1, 2, 3}, {1, 2, 3}, {1, 2, 3}),  # Complete intersection
    (set(), {1, 2, 3}, set()),  # Empty set
    ({"a", "b"}, {"b", "c"}, {"b"}),  # String items
]

# Test cases for stack_operations
STACK_TESTS = [
    ([("push", 1), ("push", 2), ("pop", None)], [1]),  # Basic operations
    ([("push", 1), ("pop", None), ("pop", None)], []),  # Pop empty stack
    ([], []),  # No operations
    ([("push", 1), ("push", 2), ("push", 3)], [1, 2, 3]),  # Only pushes
    ([("pop", None), ("push", 1)], [1]),  # Pop before push
]

# Test cases for queue_operations
QUEUE_TESTS = [
    ([("enqueue", 1), ("enqueue", 2), ("dequeue", None)], [2]),  # Basic operations
    ([("enqueue", 1), ("dequeue", None), ("dequeue", None)], []),  # Dequeue empty queue
    ([], []),  # No operations
    ([("enqueue", 1), ("enqueue", 2), ("enqueue", 3)], [1, 2, 3]),  # Only enqueues
    ([("dequeue", None), ("enqueue", 1)], [1]),  # Dequeue before enqueue
]

# Combine all test cases
TEST_CASES = {
    "check_membership_large_list": MEMBERSHIP_TESTS,
    "count_frequency_dict": FREQUENCY_TESTS,
    "find_intersection_sets": INTERSECTION_TESTS,
    "stack_operations": STACK_TESTS,
    "queue_operations": QUEUE_TESTS,
} 