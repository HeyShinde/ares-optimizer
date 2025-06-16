"""
Example function: List operations.
This function can be optimized using list comprehensions and built-in functions.
"""

def process_list(numbers):
    """
    Process a list of numbers by:
    1. Filtering out negative numbers
    2. Squaring the remaining numbers
    3. Summing the results
    
    Args:
        numbers: List of numbers to process
        
    Returns:
        Sum of squares of non-negative numbers
    """
    result = 0
    for num in numbers:
        if num >= 0:
            squared = num * num
            result += squared
    return result 