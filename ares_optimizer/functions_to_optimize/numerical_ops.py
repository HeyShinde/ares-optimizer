"""Numerical operations with intentionally inefficient implementations."""

def fibonacci_recursive(n: int) -> int:
    """Calculate the nth Fibonacci number using recursion.
    
    This is intentionally inefficient as it recalculates the same values multiple times.
    """
    if n <= 0:
        return 0
    elif n == 1:
        return 1
    else:
        return fibonacci_recursive(n - 1) + fibonacci_recursive(n - 2)


def sum_list_conditional(numbers: list[int], threshold: int) -> int:
    """Sum numbers in a list that are greater than the threshold.
    
    This is intentionally inefficient as it uses a list comprehension and sum() separately.
    """
    filtered_numbers = [x for x in numbers if x > threshold]
    total = 0
    for num in filtered_numbers:
        total += num
    return total


def factorial_recursive(n: int) -> int:
    """Calculate factorial using recursion.
    
    This is intentionally inefficient as it uses recursion instead of iteration.
    """
    if n < 0:
        raise ValueError("Factorial is not defined for negative numbers")
    elif n == 0 or n == 1:
        return 1
    else:
        return n * factorial_recursive(n - 1)


def is_prime(n: int) -> bool:
    """Check if a number is prime.
    
    This is intentionally inefficient as it checks all numbers up to n.
    """
    if n < 2:
        return False
    for i in range(2, n):
        if n % i == 0:
            return False
    return True


def gcd_recursive(a: int, b: int) -> int:
    """Calculate Greatest Common Divisor using recursion.
    
    This is intentionally inefficient as it uses recursion instead of iteration.
    """
    if b == 0:
        return a
    return gcd_recursive(b, a % b) 