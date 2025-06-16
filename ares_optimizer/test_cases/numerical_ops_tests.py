"""Test cases for numerical operations."""

# Test cases for fibonacci_recursive
FIBONACCI_TESTS = [
    (0, 0),  # Base case 1
    (1, 1),  # Base case 2
    (2, 1),  # First recursive case
    (3, 2),  # Second recursive case
    (4, 3),  # Third recursive case
    (5, 5),  # Fourth recursive case
    (6, 8),  # Fifth recursive case
    (7, 13),  # Sixth recursive case
    (8, 21),  # Seventh recursive case
    (9, 34),  # Eighth recursive case
    (10, 55),  # Ninth recursive case
]

# Test cases for sum_list_conditional
SUM_LIST_TESTS = [
    ([1, 2, 3, 4, 5], 3, 9),  # Sum numbers > 3
    ([1, 2, 3, 4, 5], 0, 15),  # Sum all numbers
    ([1, 2, 3, 4, 5], 5, 0),  # No numbers > 5
    ([], 1, 0),  # Empty list
    ([-1, -2, -3], -2, -1),  # Negative numbers
]

# Test cases for factorial_recursive
FACTORIAL_TESTS = [
    (0, 1),  # Base case
    (1, 1),  # First recursive case
    (2, 2),  # Second recursive case
    (3, 6),  # Third recursive case
    (4, 24),  # Fourth recursive case
    (5, 120),  # Fifth recursive case
    (6, 720),  # Sixth recursive case
    (7, 5040),  # Seventh recursive case
    (8, 40320),  # Eighth recursive case
    (9, 362880),  # Ninth recursive case
    (10, 3628800),  # Tenth recursive case
]

# Test cases for is_prime
IS_PRIME_TESTS = [
    (0, False),  # Not prime
    (1, False),  # Not prime
    (2, True),  # First prime
    (3, True),  # Second prime
    (4, False),  # Not prime
    (5, True),  # Third prime
    (6, False),  # Not prime
    (7, True),  # Fourth prime
    (8, False),  # Not prime
    (9, False),  # Not prime
    (10, False),  # Not prime
    (11, True),  # Fifth prime
    (12, False),  # Not prime
    (13, True),  # Sixth prime
    (14, False),  # Not prime
    (15, False),  # Not prime
    (16, False),  # Not prime
    (17, True),  # Seventh prime
    (18, False),  # Not prime
    (19, True),  # Eighth prime
    (20, False),  # Not prime
]

# Test cases for gcd_recursive
GCD_TESTS = [
    (48, 18, 6),  # Common case
    (54, 24, 6),  # Another common case
    (7, 13, 1),  # Coprime numbers
    (0, 5, 5),  # One number is 0
    (5, 0, 5),  # Other number is 0
    (0, 0, 0),  # Both numbers are 0
    (1, 1, 1),  # Both numbers are 1
    (2, 4, 2),  # One number is multiple of other
    (3, 9, 3),  # Another multiple case
    (10, 15, 5),  # Common factor case
    (17, 19, 1),  # Prime numbers
    (100, 75, 25),  # Large numbers
    (81, 27, 27),  # Perfect multiple
    (16, 24, 8),  # Common factor case
]

# Combine all test cases
TEST_CASES = {
    "fibonacci_recursive": FIBONACCI_TESTS,
    "sum_list_conditional": SUM_LIST_TESTS,
    "factorial_recursive": FACTORIAL_TESTS,
    "is_prime": IS_PRIME_TESTS,
    "gcd_recursive": GCD_TESTS,
} 