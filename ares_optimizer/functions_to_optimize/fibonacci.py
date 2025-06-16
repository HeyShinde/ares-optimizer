"""
Example function: Fibonacci sequence calculation.
This is a good candidate for optimization as it can be improved using memoization.
"""

def fibonacci(n):
    """
    Calculate the nth Fibonacci number.
    
    Args:
        n: Position in the Fibonacci sequence (0-based)
        
    Returns:
        The nth Fibonacci number
    """
    if n <= 1:
        return n
    
    a, b = 0, 1
    for _ in range(2, n + 1):
        a, b = b, a + b
    return b 