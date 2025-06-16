"""Test cases for string processing functions."""

# Test cases for count_substring_occurrences
SUBSTRING_TESTS = [
    ("hello world", "l", 3),  # Multiple occurrences
    ("hello world", "o", 2),  # Multiple occurrences
    ("hello world", "x", 0),  # No occurrences
    ("", "a", 0),  # Empty string
    ("aaaa", "aa", 3),  # Overlapping occurrences
    ("hello world", "world", 1),  # Single occurrence
    ("hello world", "hello", 1),  # Single occurrence
    ("hello world", "", 0),  # Empty substring
]

# Test cases for reverse_words
REVERSE_WORDS_TESTS = [
    ("hello world", "world hello"),  # Basic case
    ("the quick brown fox", "fox brown quick the"),  # Multiple words
    ("a b c", "c b a"),  # Single letters
    ("", ""),  # Empty string
    ("single", "single"),  # Single word
    ("multiple   spaces", "spaces multiple"),  # Multiple spaces
]

# Test cases for is_palindrome
PALINDROME_TESTS = [
    ("", True),  # Empty string
    ("a", True),  # Single character
    ("aa", True),  # Two same characters
    ("aba", True),  # Odd length palindrome
    ("abba", True),  # Even length palindrome
    ("abc", False),  # Not a palindrome
    ("A man a plan a canal: Panama", True),  # With spaces and punctuation
    ("race a car", False),  # Not a palindrome
    ("Was it a car or a cat I saw?", True),  # With spaces and punctuation
    ("hello", False),  # Not a palindrome
]

# Test cases for find_longest_common_substring
COMMON_SUBSTRING_TESTS = [
    ("hello", "world", "l"),  # Longest common substring is 'l'
    ("hello", "help", "hel"),  # Common prefix
    ("hello", "yellow", "ello"),  # Common substring
    ("", "world", ""),  # Empty string
    ("hello", "", ""),  # Empty string
    ("hello", "hello", "hello"),  # Same string
    ("hello world", "world hello", "hello"),  # Common word
    ("abcdef", "defghi", "def"),  # Common suffix
]

# Test cases for compress_string
COMPRESS_TESTS = [
    ("", ""),  # Empty string
    ("a", "a"),  # Single character
    ("aa", "a2"),  # Two same characters
    ("aaa", "a3"),  # Three same characters
    ("aab", "a2b"),  # Two different characters
    ("aabb", "a2b2"),  # Two pairs
    ("aaabbb", "a3b3"),  # Three pairs
    ("abc", "abc"),  # No compression needed
    ("aabbcc", "a2b2c2"),  # Three pairs
    ("aaabbbccc", "a3b3c3"),  # Three triples
]

# Combine all test cases
TEST_CASES = {
    "count_substring_occurrences": SUBSTRING_TESTS,
    "reverse_words": REVERSE_WORDS_TESTS,
    "is_palindrome": PALINDROME_TESTS,
    "find_longest_common_substring": COMMON_SUBSTRING_TESTS,
    "compress_string": COMPRESS_TESTS,
} 