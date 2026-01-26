"""
Universal benchmark prompt wrapper factory.
Clean design for extensible benchmark integration.
"""

from typing import Union, TYPE_CHECKING
from neuro_manipulation.prompt_wrapper import PromptWrapper

if TYPE_CHECKING:
    from neuro_manipulation.prompt_formats import PromptFormat

# Import existing memory benchmark wrappers
from .memory_prompt_wrapper import (
    MemoryPromptWrapper, 
    PasskeyPromptWrapper, 
    ConversationalQAPromptWrapper, 
    LongContextQAPromptWrapper,
    LongbenchRetrievalPromptWrapper,
    EmotionCheckPromptWrapper,
)

# Import new MTBench101 wrapper
from .mtbench101_prompt_wrapper import MTBench101PromptWrapper

# Import TruthfulQA wrapper
from .truthfulqa_prompt_wrapper import TruthfulQAPromptWrapper
from .fantom_prompt_wrapper import FantomPromptWrapper
from .bfcl_prompt_wrapper import BFCLPromptWrapper
from .pubmedqa_prompt_wrapper import PubMedQAPromptWrapper


def get_benchmark_prompt_wrapper(
    benchmark_name: str, 
    task_type: str, 
    prompt_format: "PromptFormat"
) -> PromptWrapper:
    """
    Universal factory for all benchmark prompt wrappers.
    Clean design for extensibility to future benchmarks.
    
    Args:
        benchmark_name: Name of benchmark (infinitebench, longbench, mtbench101, etc.)
        task_type: Specific task within benchmark
        prompt_format: PromptFormat instance for the model
        
    Returns:
        Appropriate PromptWrapper subclass for the benchmark and task
        
    Examples:
        # MTBench101 tasks
        wrapper = get_benchmark_prompt_wrapper("mtbench101", "CM", prompt_format)
        wrapper = get_benchmark_prompt_wrapper("mtbench101", "SI", prompt_format)
        
        # Memory benchmarks
        wrapper = get_benchmark_prompt_wrapper("infinitebench", "passkey", prompt_format)
        wrapper = get_benchmark_prompt_wrapper("longbench", "qa", prompt_format)
    """
    
    benchmark_lower = benchmark_name.lower()
    task_lower = task_type.lower()

    # Emotion Check – use dedicated wrapper
    if benchmark_lower == "emotion_check":
        return EmotionCheckPromptWrapper(prompt_format, task_type)

    # TruthfulQA - multiple choice tasks
    if benchmark_lower == "truthfulqa":
        return TruthfulQAPromptWrapper(prompt_format, task_type)
    
    # PubMedQA - yes/no/maybe classification
    if benchmark_lower == "pubmed_qa":
        return PubMedQAPromptWrapper(prompt_format)
    
    # Fantom – use common Fantom wrapper
    if benchmark_lower == "fantom":
        return FantomPromptWrapper(prompt_format)
    
    # MTBench101 - all tasks use unified wrapper with task-specific configuration
    if benchmark_lower == "mtbench101":
        return MTBench101PromptWrapper(prompt_format, task_type)
    
    # BFCL - function calling tasks
    if benchmark_lower == "bfcl":
        return BFCLPromptWrapper(prompt_format, task_type)
    
    # InfiniteBench tasks
    if benchmark_lower == "infinitebench":
        if "passkey" in task_lower:
            return PasskeyPromptWrapper(prompt_format)
        elif any(keyword in task_lower for keyword in ["conversational", "conversation"]):
            return ConversationalQAPromptWrapper(prompt_format) 
        elif "qa" in task_lower:
            return LongContextQAPromptWrapper(prompt_format)
        else:
            # Default for other infinitebench tasks
            return MemoryPromptWrapper(prompt_format)
    
    # LongBench tasks
    if benchmark_lower == "longbench":
        if any(keyword in task_lower for keyword in ["retrieval", "longbench"]):
            return LongbenchRetrievalPromptWrapper(prompt_format)
        elif "qa" in task_lower:
            return LongContextQAPromptWrapper(prompt_format)
        else:
            # Default for other longbench tasks
            return MemoryPromptWrapper(prompt_format)
    
    # LoCoMo (conversational memory) tasks
    if benchmark_lower in ["locomo", "conversational"]:
        return ConversationalQAPromptWrapper(prompt_format)
    
    # Handle legacy memory task routing
    # This supports cases where benchmark_name might still refer to task types
    if any(keyword in benchmark_lower for keyword in ["passkey"]):
        return PasskeyPromptWrapper(prompt_format)
    elif any(keyword in benchmark_lower for keyword in ["conversational", "locomo"]):
        return ConversationalQAPromptWrapper(prompt_format)
    elif any(keyword in benchmark_lower for keyword in ["qa", "question"]):
        return LongContextQAPromptWrapper(prompt_format)
    elif any(keyword in benchmark_lower for keyword in ["retrieval"]):
        return LongbenchRetrievalPromptWrapper(prompt_format)
    
    # Default fallback for unknown benchmarks
    return MemoryPromptWrapper(prompt_format)


# Helper function to get all supported benchmarks
def get_supported_benchmarks():
    """
    Get list of all supported benchmark names.
    
    Returns:
        Dict mapping benchmark names to supported task types
    """
    return {
        "pubmed_qa": [
            "pqa_labeled"
        ],
        "truthfulqa": [
            "mc1", "mc2"
        ],
        "mtbench101": [
            "CM", "SI", "AR", "TS", "CC", "CR", 
            "FR", "SC", "SA", "MR", "GR", "IC", "PI"
        ],
        "infinitebench": [
            "passkey", "number_string", "kv_retrieval", 
            "longbook_sum_eng", "longbook_qa_eng", "math_calc",
            "math_find", "code_run", "code_debug"  
        ],
        "longbench": [
            "narrativeqa", "qasper", "multifieldqa_en", 
            "hotpotqa", "2wikimqa", "musique", "gov_report",
            "qmsum", "multi_news", "vcsum", "trec", 
            "triviaqa", "samsum", "passage_count",
            "passage_retrieval_en", "lcc", "repobench-p"
        ],
        "locomo": [
            "conversational_qa", "context_carryover", "memory_recall"
        ]
    }


# Helper function to check if a benchmark/task combination is supported
def is_supported_benchmark_task(benchmark_name: str, task_type: str) -> bool:
    """
    Check if a benchmark and task type combination is supported.
    
    Args:
        benchmark_name: Name of the benchmark
        task_type: Task type within the benchmark
        
    Returns:
        True if the combination is supported, False otherwise
    """
    supported = get_supported_benchmarks()
    benchmark_lower = benchmark_name.lower()
    
    if benchmark_lower in supported:
        # Case-insensitive comparison for task types
        tasks = supported[benchmark_lower]
        task_lower = task_type.lower()
        return any(t.lower() == task_lower for t in tasks)
    
    # Always return True for unknown benchmarks (will use default wrapper)
    return True
