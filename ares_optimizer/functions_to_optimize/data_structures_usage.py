"""Data structure usage functions with intentionally inefficient implementations."""

def check_membership_large_list(items: list, target: any) -> bool:
    """Check if an item exists in a large list.
    
    This is intentionally inefficient as it uses a list for membership testing.
    """
    for item in items:
        if item == target:
            return True
    return False


def count_frequency_dict(items: list) -> dict:
    """Count frequency of items in a list using a dictionary.
    
    This is intentionally inefficient as it uses multiple dictionary operations.
    """
    freq = {}
    for item in items:
        if item in freq:
            freq[item] = freq[item] + 1
        else:
            freq[item] = 1
    return freq


def find_intersection_sets(set1: set, set2: set) -> set:
    """Find intersection of two sets.
    
    This is intentionally inefficient as it uses a loop instead of set operations.
    """
    result = set()
    for item in set1:
        if item in set2:
            result.add(item)
    return result


def stack_operations(operations: list[tuple[str, any]]) -> list:
    """Perform stack operations (push/pop) on a list.
    
    This is intentionally inefficient as it uses a list for stack operations.
    """
    stack = []
    for op, value in operations:
        if op == "push":
            stack.append(value)
        elif op == "pop":
            if stack:
                stack.pop()
    return stack


def queue_operations(operations: list[tuple[str, any]]) -> list:
    """Perform queue operations (enqueue/dequeue) on a list.
    
    This is intentionally inefficient as it uses a list for queue operations.
    """
    queue = []
    for op, value in operations:
        if op == "enqueue":
            queue.append(value)
        elif op == "dequeue":
            if queue:
                queue.pop(0)
    return queue 