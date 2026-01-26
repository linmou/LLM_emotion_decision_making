#!/usr/bin/env python3
"""
Component Specification Registry for Benchmark Processing

This module provides a single source of truth for the relationships between
benchmark processing components: prompt_wrapper, answer_wrapper, and dataset classes.

The registry eliminates the need to manually assemble these components in
experiment code and ensures consistency across different benchmark configurations.

Key Features:
- BenchmarkSpec: Encapsulates the three component types needed for each benchmark
- BENCHMARK_SPECS: Registry mapping (benchmark_name, task_type) to specifications
- create_benchmark_components(): Factory function that assembles all components
- Type safety and clear error messages for unknown combinations
- Easy extension for new benchmarks

Example Usage:
    from .benchmark_component_registry import create_benchmark_components

    prompt_wrapper, answer_wrapper, dataset = create_benchmark_components(
        benchmark_name="mtbench101",
        task_type="CM",
        config=config,
        prompt_format=prompt_format,
        emotion="anger"
    )
"""

from dataclasses import dataclass
from functools import partial
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, TYPE_CHECKING

from neuro_manipulation.prompt_wrapper import PromptWrapper

from .answer_wrapper import AnswerWrapper, EmotionAnswerWrapper, IdentityAnswerWrapper
from .benchmark_prompt_wrapper import get_benchmark_prompt_wrapper
from .mtbench101_prompt_wrapper import MTBench101PromptWrapper
from .truthfulqa_prompt_wrapper import TruthfulQAPromptWrapper
from .memory_prompt_wrapper import (
    MemoryPromptWrapper,
    PasskeyPromptWrapper,
    ConversationalQAPromptWrapper,
    LongContextQAPromptWrapper,
    LongbenchRetrievalPromptWrapper,
    EmotionCheckPromptWrapper,
)
from .data_models import BenchmarkConfig
from .datasets.mtbench101 import MTBench101Dataset
from .datasets.infinitebench import InfiniteBenchDataset
from .datasets.longbench import LongBenchDataset
from .datasets.bfcl import BFCLDataset
from .datasets.locomo import LoCoMoDataset
from .datasets.truthfulqa import TruthfulQADataset
from .datasets.fantom import FantomDataset
from .datasets.emotion_check import EmotionCheckDataset
from .datasets.trustllm_ethics import TrustLLMEthicsDataset
from .datasets.trustllm_fairness import TrustLLMFairnessDataset
from .datasets.trustllm_privacy import TrustLLMPrivacyDataset
from .datasets.trustllm_robustness import TrustLLMRobustnessDataset
from .datasets.trustllm_safety import TrustLLMSafetyDataset
from .datasets.trustllm_truthfulness import TrustLLMTruthfulnessDataset
from .datasets.games import GameTheoryCompletionOptionIdDataset, GameTheoryDataset
from .datasets.diplomacy_gradient import DiplomacyGradientDataset
from .diplomacy_prompts import DiplomacyOptionsPromptWrapper
def create_dataset_from_config(*args, **kwargs):  # lazy import to avoid heavy deps at import time
    from .dataset_factory import create_dataset_from_config as _real_create
    return _real_create(*args, **kwargs)
if TYPE_CHECKING:
    from .datasets.base import BaseBenchmarkDataset
from .fantom_prompt_wrapper import FantomPromptWrapper
from .bfcl_prompt_wrapper import BFCLPromptWrapper
from .game_prompt_wrapper import (
    GameBenchmarkPromptWrapper,
    GameCompletionOptionIdPromptWrapper,
    GameDecisionPromptWrapper,
)


@dataclass
class BenchmarkSpec:
    """
    Specification of all components needed to process a specific benchmark task.

    This encapsulates the knowledge of which prompt_wrapper, answer_wrapper,
    and dataset class should be used together for a given benchmark and task.
    """

    dataset_class: "Type[BaseBenchmarkDataset]"
    answer_wrapper_class: Type[AnswerWrapper]

    prompt_wrapper_class: Optional[Type[PromptWrapper]] = None

    def create_components(
        self,
        config: BenchmarkConfig,
        prompt_format: Any,
        emotion: Optional[str] = None,
        enable_thinking: bool = False,
        augmentation_config: Optional[Dict] = None,
        user_messages: str = "Please provide your answer.",
        **dataset_kwargs,
    ) -> Tuple[Callable, Callable, Any]:
        """
        Create all three components using this specification.

        Args:
            config: BenchmarkConfig for the dataset
            prompt_format: PromptFormat instance for the model
            emotion: Emotion context for processing
            enable_thinking: Whether to enable thinking mode
            augmentation_config: Configuration for prompt augmentation
            user_messages: User message template
            **dataset_kwargs: Additional arguments for dataset creation

        Returns:
            Tuple of (prompt_wrapper_partial, answer_wrapper_partial, dataset)
        """
        # Create prompt wrapper using explicit class when provided,
        # otherwise fall back to the factory selector.
        if self.prompt_wrapper_class is not None:
            try:
                # Some wrappers accept (prompt_format, task_type)
                prompt_wrapper = self.prompt_wrapper_class(
                    prompt_format, config.task_type
                )
            except TypeError:
                # Others accept only (prompt_format)
                prompt_wrapper = self.prompt_wrapper_class(prompt_format)
        else:
            prompt_wrapper = get_benchmark_prompt_wrapper(
                config.name, config.task_type, prompt_format
            )

        # Create partial function with emotion-specific parameters
        prompt_wrapper_partial = partial(
            prompt_wrapper.__call__,
            user_messages=user_messages,
            enable_thinking=enable_thinking,
            augmentation_config=augmentation_config,
            emotion=emotion,
        )

        # Create answer wrapper instance
        answer_wrapper = self.answer_wrapper_class()

        # Create partial function with emotion context
        answer_wrapper_partial = partial(
            answer_wrapper.__call__,
            emotion=emotion,
            benchmark_name=config.name,
            task_type=config.task_type,
        )

        # Create dataset using existing factory (maintains current architecture)
        dataset = create_dataset_from_config(
            config,
            prompt_wrapper=prompt_wrapper_partial,
            answer_wrapper=answer_wrapper_partial,
            **dataset_kwargs,
        )

        return prompt_wrapper_partial, answer_wrapper_partial, dataset


# Registry mapping (benchmark_name, task_type) to component specifications
# This is the single source of truth for component relationships
BENCHMARK_SPECS: Dict[Tuple[str, str], BenchmarkSpec] = {
    # MTBench101: all tasks share the same spec
    ("mtbench101", "*"): BenchmarkSpec(
        dataset_class=MTBench101Dataset,
        answer_wrapper_class=IdentityAnswerWrapper,
        prompt_wrapper_class=MTBench101PromptWrapper,
    ),
    # Memory benchmarks - InfiniteBench
    # Default for most tasks
    ("infinitebench", "*"): BenchmarkSpec(
        dataset_class=InfiniteBenchDataset,
        answer_wrapper_class=IdentityAnswerWrapper,
        prompt_wrapper_class=MemoryPromptWrapper,
    ),
    # Overrides
    ("infinitebench", "passkey"): BenchmarkSpec(
        dataset_class=InfiniteBenchDataset,
        answer_wrapper_class=IdentityAnswerWrapper,
        prompt_wrapper_class=PasskeyPromptWrapper,
    ),
    ("infinitebench", "longbook_qa_eng"): BenchmarkSpec(
        dataset_class=InfiniteBenchDataset,
        answer_wrapper_class=IdentityAnswerWrapper,
        prompt_wrapper_class=LongContextQAPromptWrapper,
    ),
    ("infinitebench", "longbook_qa_eng_121k"): BenchmarkSpec(
        dataset_class=InfiniteBenchDataset,
        answer_wrapper_class=IdentityAnswerWrapper,
        prompt_wrapper_class=LongContextQAPromptWrapper,
    ),
    ("infinitebench", "longbook_qa_chn"): BenchmarkSpec(
        dataset_class=InfiniteBenchDataset,
        answer_wrapper_class=IdentityAnswerWrapper,
        prompt_wrapper_class=LongContextQAPromptWrapper,
    ),
    # Memory benchmarks - LongBench
    ("longbench", "*"): BenchmarkSpec(
        dataset_class=LongBenchDataset,
        answer_wrapper_class=IdentityAnswerWrapper,
        prompt_wrapper_class=LongContextQAPromptWrapper,
    ),
    ("longbench", "passage_retrieval_en"): BenchmarkSpec(
        dataset_class=LongBenchDataset,
        answer_wrapper_class=IdentityAnswerWrapper,
        prompt_wrapper_class=LongbenchRetrievalPromptWrapper,
    ),
    ("longbench", "passage_retrieval_zh"): BenchmarkSpec(
        dataset_class=LongBenchDataset,
        answer_wrapper_class=IdentityAnswerWrapper,
        prompt_wrapper_class=LongbenchRetrievalPromptWrapper,
    ),
    # BFCL benchmark
    ("bfcl", "*"): BenchmarkSpec(
        dataset_class=BFCLDataset,
        answer_wrapper_class=IdentityAnswerWrapper,
        prompt_wrapper_class=BFCLPromptWrapper,
    ),
    # LoCoMo benchmark
    ("locomo", "locomo"): BenchmarkSpec(
        dataset_class=LoCoMoDataset,
        answer_wrapper_class=IdentityAnswerWrapper,
        prompt_wrapper_class=ConversationalQAPromptWrapper,
    ),
    # TruthfulQA benchmark
    ("truthfulqa", "mc1"): BenchmarkSpec(
        dataset_class=TruthfulQADataset,
        answer_wrapper_class=IdentityAnswerWrapper,
        prompt_wrapper_class=TruthfulQAPromptWrapper,
    ),
    ("truthfulqa", "mc2"): BenchmarkSpec(
        dataset_class=TruthfulQADataset,
        answer_wrapper_class=IdentityAnswerWrapper,
        prompt_wrapper_class=TruthfulQAPromptWrapper,
    ),
    # Emotion Check benchmark - all tasks use the same wrapper
    ("emotion_check", "*"): BenchmarkSpec(
        dataset_class=EmotionCheckDataset,
        answer_wrapper_class=EmotionAnswerWrapper,
        prompt_wrapper_class=EmotionCheckPromptWrapper,
    ),
    # FANToM – allow wildcard for shared wrapper
    ("fantom", "*"): BenchmarkSpec(
        dataset_class=FantomDataset,
        answer_wrapper_class=IdentityAnswerWrapper,
        prompt_wrapper_class=FantomPromptWrapper,
    ),
    # TrustLLM families (GPT-4o-mini evaluator via dataset logic)
    ("trustllm_ethics", "*"): BenchmarkSpec(
        dataset_class=TrustLLMEthicsDataset,
        answer_wrapper_class=IdentityAnswerWrapper,
    ),
    ("trustllm_fairness", "*"): BenchmarkSpec(
        dataset_class=TrustLLMFairnessDataset,
        answer_wrapper_class=IdentityAnswerWrapper,
    ),
    ("trustllm_privacy", "*"): BenchmarkSpec(
        dataset_class=TrustLLMPrivacyDataset,
        answer_wrapper_class=IdentityAnswerWrapper,
    ),
    ("trustllm_robustness", "*"): BenchmarkSpec(
        dataset_class=TrustLLMRobustnessDataset,
        answer_wrapper_class=IdentityAnswerWrapper,
    ),
    ("trustllm_safety", "*"): BenchmarkSpec(
        dataset_class=TrustLLMSafetyDataset,
        answer_wrapper_class=IdentityAnswerWrapper,
    ),
    ("trustllm_truthfulness", "*"): BenchmarkSpec(
        dataset_class=TrustLLMTruthfulnessDataset,
        answer_wrapper_class=IdentityAnswerWrapper,
    ),
    ("game_theory_decision", "*"): BenchmarkSpec(
        dataset_class=GameTheoryDataset,
        answer_wrapper_class=IdentityAnswerWrapper,
        prompt_wrapper_class=GameDecisionPromptWrapper,
    ),
    ("game_theory_completion_option_id", "*"): BenchmarkSpec(
        dataset_class=GameTheoryCompletionOptionIdDataset,
        answer_wrapper_class=IdentityAnswerWrapper,
        prompt_wrapper_class=GameCompletionOptionIdPromptWrapper,
    ),
    ("game_theory", "*"): BenchmarkSpec(
        dataset_class=GameTheoryDataset,
        answer_wrapper_class=IdentityAnswerWrapper,
        prompt_wrapper_class=GameBenchmarkPromptWrapper,
    ),
    ("diplomacy_pd", "*"): BenchmarkSpec(
        dataset_class=DiplomacyGradientDataset,
        answer_wrapper_class=IdentityAnswerWrapper,
        prompt_wrapper_class=DiplomacyOptionsPromptWrapper,
    ),
}


def create_benchmark_components(
    benchmark_name: str,
    task_type: str,
    config: BenchmarkConfig,
    prompt_format: Any,
    emotion: Optional[str] = None,
    enable_thinking: bool = False,
    augmentation_config: Optional[Dict] = None,
    user_messages: str = "Please provide your answer.",
    **dataset_kwargs,
) -> Tuple[Callable, Callable, Any]:
    """
    Factory function to create all benchmark processing components from registry.

    This is the main entry point that replaces manual component assembly
    in experiment code. It uses the BENCHMARK_SPECS registry to ensure
    consistent component relationships.

    Args:
        benchmark_name: Name of the benchmark (e.g., "mtbench101", "infinitebench")
        task_type: Specific task within the benchmark (e.g., "CM", "passkey")
        config: BenchmarkConfig for dataset creation
        prompt_format: PromptFormat instance for the model
        emotion: Optional emotion context for processing
        enable_thinking: Whether to enable thinking mode in prompts
        augmentation_config: Configuration for prompt augmentation
        user_messages: Template for user messages in prompts
        **dataset_kwargs: Additional arguments passed to dataset creation

    Returns:
        Tuple of (prompt_wrapper_partial, answer_wrapper_partial, dataset)

    Raises:
        KeyError: If the (benchmark_name, task_type) combination is not registered

    Example:
        prompt_wrapper, answer_wrapper, dataset = create_benchmark_components(
            benchmark_name="mtbench101",
            task_type="CM",
            config=config,
            prompt_format=prompt_format,
            emotion="anger"
        )
    """
    # Normalize benchmark name for consistent lookup, but preserve task_type case
    benchmark_key = (benchmark_name.lower(), task_type)

    spec: Optional[BenchmarkSpec] = BENCHMARK_SPECS.get(benchmark_key)
    if spec is None:
        # Fallback to wildcard default like (name, "*") to reduce duplication
        wildcard_key = (benchmark_name.lower(), "*")
        spec = BENCHMARK_SPECS.get(wildcard_key)
    if spec is None:
        available_combinations = list(BENCHMARK_SPECS.keys())
        raise KeyError(
            f"Unknown benchmark combination: ({benchmark_name}, {task_type}). "
            f"Available combinations: {available_combinations}"
        )

    return spec.create_components(
        config=config,
        prompt_format=prompt_format,
        emotion=emotion,
        enable_thinking=enable_thinking,
        augmentation_config=augmentation_config,
        user_messages=user_messages,
        **dataset_kwargs,
    )


def list_available_benchmarks() -> Dict[str, List[str]]:
    """
    Get a summary of all available benchmark and task combinations.

    Returns:
        Dictionary mapping benchmark names to lists of available tasks
    """
    benchmarks = {}
    for (benchmark_name, task_type), spec in BENCHMARK_SPECS.items():
        if benchmark_name not in benchmarks:
            benchmarks[benchmark_name] = []
        benchmarks[benchmark_name].append(task_type)

    return benchmarks


def get_benchmark_description(benchmark_name: str, task_type: str) -> str:
    """
    Get description for a specific benchmark and task combination.

    Args:
        benchmark_name: Name of the benchmark
        task_type: Task type within the benchmark

    Returns:
        Description string, or "Unknown combination" if not found
    """
    benchmark_key = (benchmark_name.lower(), task_type)
    # Description field removed from BenchmarkSpec. Provide a simple fallback.
    if benchmark_key in BENCHMARK_SPECS:
        return f"{benchmark_name} - {task_type}"
    return "Unknown combination"
