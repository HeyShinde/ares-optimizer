def fibonacci_recursive(n: int) -> int:
    """
    Calculate the nth Fibonacci number using recursion.
    This is intentionally inefficient for training purposes.
    """
    if n <= 1:
        return n
    return fibonacci_recursive(n-1) + fibonacci_recursive(n-2)

def fibonacci_optimized(n: int) -> int:
    """
    Calculate the nth Fibonacci number using dynamic programming.
    This is the known optimized version.
    """
    if n <= 1:
        return n
    
    a, b = 0, 1
    for _ in range(2, n + 1):
        a, b = b, a + b
    return b

# Test cases
def test_fibonacci():
    test_cases = [
        (0, 0),
        (1, 1),
        (2, 1),
        (3, 2),
        (4, 3),
        (5, 5),
        (6, 8),
        (7, 13),
        (8, 21),
        (9, 34),
        (10, 55)
    ]
    
    # Test original function
    for n, expected in test_cases:
        assert fibonacci_recursive(n) == expected, f"Original function failed for n={n}"
    
    # Test optimized function
    for n, expected in test_cases:
        assert fibonacci_optimized(n) == expected, f"Optimized function failed for n={n}"
    
    print("All test cases passed!")

if __name__ == "__main__":
    test_fibonacci() 