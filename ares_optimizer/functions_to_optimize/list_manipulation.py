"""List manipulation functions with intentionally inefficient implementations."""

def find_duplicates(items: list) -> list:
    """Find duplicates in a list using nested loops.
    
    This is intentionally inefficient as it uses nested loops.
    """
    duplicates = []
    for i in range(len(items)):
        for j in range(i + 1, len(items)):
            if items[i] == items[j] and items[i] not in duplicates:
                duplicates.append(items[i])
    return duplicates


def reverse_in_place(items: list) -> list:
    """Reverse a list in place using a temporary list.
    
    This is intentionally inefficient as it uses a temporary list.
    """
    temp = items.copy()
    for i in range(len(items)):
        items[i] = temp[-(i + 1)]
    return items


def remove_duplicates(items: list) -> list:
    """Remove duplicates while preserving order.
    
    This is intentionally inefficient as it uses a list for membership testing.
    """
    result = []
    for item in items:
        if item not in result:
            result.append(item)
    return result


def find_common_elements(list1: list, list2: list) -> list:
    """Find common elements between two lists using nested loops.
    
    This is intentionally inefficient as it uses nested loops.
    """
    common = []
    for item1 in list1:
        for item2 in list2:
            if item1 == item2 and item1 not in common:
                common.append(item1)
    return common


def rotate_list(items: list, k: int) -> list:
    """Rotate a list by k positions to the right.
    
    This is intentionally inefficient as it uses multiple list operations.
    """
    if not items:
        return items
    k = k % len(items)
    return items[-k:] + items[:-k] 