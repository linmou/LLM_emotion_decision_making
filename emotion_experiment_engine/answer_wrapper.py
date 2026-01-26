"""
Answer wrapper classes for transforming ground truth based on context.
Follows the same pattern as PromptWrapper but for output adaptation.
"""

from typing import Any, Optional


class AnswerWrapper:
    """Base class for answer/ground truth transformations"""

    def __init__(self, benchmark_config=None):
        self.benchmark_config = benchmark_config

    def transform_answer(self, ground_truth: Any, **context) -> Any:
        """Transform ground truth based on context"""
        return ground_truth

    def __call__(self, ground_truth: Any, **context) -> Any:
        """Main interface - matches PromptWrapper pattern"""
        return self.transform_answer(ground_truth, **context)


class IdentityAnswerWrapper(AnswerWrapper):
    """Default wrapper that passes through ground truth unchanged"""

    def transform_answer(self, ground_truth: Any, **context) -> Any:
        """Pass through unchanged - no transformation"""
        return ground_truth


class EmotionAnswerWrapper(AnswerWrapper):
    """Answer wrapper that adapts ground truth based on emotion"""
    
    def transform_answer(self, ground_truth: Any, emotion: Optional[str] = None, 
                        benchmark_name: Optional[str] = None, 
                        task_type: Optional[str] = None, **context) -> Any:
        """Transform ground truth based on emotion context"""
        # For EmotionCheck tasks, ground truth IS the activated emotion
        if benchmark_name == "emotion_check" and emotion:
            return emotion
        
        # For other tasks, ground truth doesn't change with emotion
        return ground_truth


def get_answer_wrapper(benchmark_name: str, task_type: str) -> AnswerWrapper:
    """
    Factory function to get appropriate answer wrapper for benchmark/task.
    Follows same pattern as get_benchmark_prompt_wrapper.
    """
    # EmotionCheck needs emotion-aware wrapper
    if benchmark_name == "emotion_check":
        return EmotionAnswerWrapper()

    # All other benchmarks use identity wrapper (pass-through)
    return IdentityAnswerWrapper()
