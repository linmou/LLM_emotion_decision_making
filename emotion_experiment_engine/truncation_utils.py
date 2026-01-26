"""
Truncation utilities for memory benchmarks.
Extracted from original adapters/truncation_utils.py during refactoring.
"""


def calculate_max_context_length(max_model_len: int, preserve_ratio: float = 0.95, prompt_overhead: int = 200) -> int:
    """
    Calculate maximum context length allowing for prompt overhead.
    
    Args:
        max_model_len: Maximum model sequence length
        preserve_ratio: Ratio of model length to preserve for context (0.0-1.0)
        prompt_overhead: Estimated tokens for prompt formatting
        
    Returns:
        Maximum context length in tokens
    """
    max_context_length = int(max_model_len * preserve_ratio) - prompt_overhead
    return max(0, max_context_length)