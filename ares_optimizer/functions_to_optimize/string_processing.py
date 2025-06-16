"""String processing functions with intentionally inefficient implementations."""

def count_substring_occurrences(text: str, substring: str) -> int:
    """Count occurrences of a substring in a text using string slicing.
    
    This is intentionally inefficient as it uses string slicing in a loop.
    """
    if not substring:
        return 0
    count = 0
    for i in range(len(text) - len(substring) + 1):
        if text[i:i + len(substring)] == substring:
            count += 1
    return count


def reverse_words(text: str) -> str:
    """Reverse the order of words in a text.
    
    This is intentionally inefficient as it uses multiple string operations.
    """
    words = text.split()
    return " ".join(words[::-1])


def is_palindrome(text: str) -> bool:
    """Check if a text is a palindrome.
    
    This is intentionally inefficient as it creates a new string for comparison.
    """
    # Remove non-alphanumeric characters and convert to lowercase
    cleaned = ""
    for char in text.lower():
        if char.isalnum():
            cleaned += char
    return cleaned == cleaned[::-1]


def find_longest_common_substring(str1: str, str2: str) -> str:
    """Find the longest common substring between two strings.
    
    This is intentionally inefficient as it uses nested loops and string slicing.
    """
    if not str1 or not str2:
        return ""
    
    longest = ""
    for i in range(len(str1)):
        for j in range(i + 1, len(str1) + 1):
            substring = str1[i:j]
            if substring in str2 and len(substring) > len(longest):
                longest = substring
    return longest


def compress_string(text: str) -> str:
    """Compress a string by replacing repeated characters with their count.
    
    This is intentionally inefficient as it uses string concatenation in a loop.
    """
    if not text:
        return ""
    
    result = ""
    current_char = text[0]
    count = 1
    
    for char in text[1:]:
        if char == current_char:
            count += 1
        else:
            result += current_char + (str(count) if count > 1 else "")
            current_char = char
            count = 1
    
    result += current_char + (str(count) if count > 1 else "")
    return result 