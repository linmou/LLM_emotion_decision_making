#!/usr/bin/env python3
"""
Registry-based Dataset Factory for eliminating if-else chains

This module implements a factory pattern using a registry to create specialized
dataset classes based on configuration. It replaces if-else branching with
polymorphic dispatch through a registry lookup table.

Key Features:
- Registry-based dataset class selection (no if-else chains)
- Case-insensitive benchmark name handling
- Dynamic registration of new dataset types
- Seamless parameter passing to dataset constructors
- Informative error messages for unknown benchmarks

Example Usage:
    config = BenchmarkConfig(name="infinitebench", task_type="passkey", ...)
    dataset = create_dataset_from_config(config, max_context_length=1000)

    # Or register a new dataset type:
    register_dataset_class("custom_benchmark", CustomDatasetClass)
"""

from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Type

from .data_models import (
    DEFAULT_VLLM_MAX_MODEL_LEN,
    BenchmarkConfig,
    ExperimentConfig,
    VLLMLoadingConfig,
)
from .datasets.base import BaseBenchmarkDataset

# Note: The authoritative mapping now lives in BENCHMARK_SPECS
# We derive the name -> dataset class map from it to avoid duplication.
_OVERRIDES: Dict[str, Type[BaseBenchmarkDataset]] = {}


def _derive_registry_from_specs() -> Dict[str, Type[BaseBenchmarkDataset]]:
    from .benchmark_component_registry import BENCHMARK_SPECS

    derived: Dict[str, Type[BaseBenchmarkDataset]] = {}
    for (name, _task), spec in BENCHMARK_SPECS.items():
        key = name.lower().strip()
        cls = spec.dataset_class
        if key in derived and derived[key] is not cls:
            raise ValueError(
                f"Conflicting dataset classes for benchmark '{name}': "
                f"{derived[key].__name__} vs {cls.__name__}"
            )
        derived[key] = cls
    return derived


# Back-compat export for tests and users; this is a view, not a hand-maintained registry
DATASET_REGISTRY: Dict[str, Type[BaseBenchmarkDataset]] = _derive_registry_from_specs()
DATASET_REGISTRY.update(_OVERRIDES)


def create_dataset_from_config(
    config: BenchmarkConfig,
    prompt_wrapper: Optional[Callable[[str, str], str]] = None,
    max_context_length: Optional[int] = None,
    tokenizer: Optional[Any] = None,
    truncation_strategy: str = "right",
    answer_wrapper: Optional[Callable] = None,
    **kwargs,
) -> BaseBenchmarkDataset:
    """
    Create a specialized dataset from configuration using registry lookup.

    This factory function eliminates if-else chains by using a registry-based
    approach to select the appropriate dataset class based on the benchmark name.

    Args:
        config: Benchmark configuration specifying which dataset to create
        prompt_wrapper: Optional function to format prompts
        max_context_length: Maximum context length for truncation
        tokenizer: Optional tokenizer for text processing
        truncation_strategy: Strategy for context truncation ("left", "right", "middle")
        **kwargs: Additional keyword arguments passed to dataset constructor

    Returns:
        Specialized dataset instance (InfiniteBenchDataset, LongBenchDataset, etc.)

    Raises:
        ValueError: If benchmark name is not recognized
        FileNotFoundError: If data file doesn't exist

    Example:
        >>> config = BenchmarkConfig(name="infinitebench", task_type="passkey", data_path="data.jsonl")
        >>> dataset = create_dataset_from_config(config, max_context_length=2048)
        >>> isinstance(dataset, InfiniteBenchDataset)
        True
    """
    # Normalize benchmark name for case-insensitive lookup
    benchmark_name = config.name.lower().strip()

    # Registry lookup derived from BENCHMARK_SPECS (no if-else chains!)
    dataset_class = DATASET_REGISTRY.get(benchmark_name)

    if dataset_class is None:
        available_benchmarks = list(DATASET_REGISTRY.keys())
        raise ValueError(
            f"Unknown benchmark: '{config.name}'. "
            f"Available benchmarks: {available_benchmarks}"
        )

    # Create dataset instance with all provided parameters
    dataset = dataset_class(
        config=config,
        prompt_wrapper=prompt_wrapper,
        max_context_length=max_context_length,
        tokenizer=tokenizer,
        truncation_strategy=truncation_strategy,
        answer_wrapper=answer_wrapper,
        **kwargs,
    )

    return dataset


def register_dataset_class(
    benchmark_name: str, dataset_class: Type[BaseBenchmarkDataset]
) -> None:
    """
    Dynamically register a new dataset class in the factory registry.

    This allows adding new benchmark datasets at runtime without modifying
    the core factory code.

    Args:
        benchmark_name: Name to use for benchmark identification (case-insensitive)
        dataset_class: Dataset class that extends BaseBenchmarkDataset

    Raises:
        TypeError: If dataset_class doesn't extend BaseBenchmarkDataset

    Example:
        >>> class MyBenchmarkDataset(BaseBenchmarkDataset):
        ...     # Implementation here
        ...     pass
        >>> register_dataset_class("my_benchmark", MyBenchmarkDataset)
        >>> "my_benchmark" in get_available_datasets()
        True
    """
    if not issubclass(dataset_class, BaseBenchmarkDataset):
        raise TypeError(
            f"Dataset class must extend BaseBenchmarkDataset, "
            f"got {dataset_class.__name__}"
        )

    # Normalize name for consistent lookup
    normalized_name = benchmark_name.lower().strip()
    _OVERRIDES[normalized_name] = dataset_class
    DATASET_REGISTRY[normalized_name] = dataset_class


def get_available_datasets() -> List[str]:
    """
    Get list of all available benchmark dataset types.

    Returns:
        List of registered benchmark names (lowercase)

    Example:
        >>> datasets = get_available_datasets()
        >>> "infinitebench" in datasets
        True
        >>> "longbench" in datasets
        True
    """
    return sorted(DATASET_REGISTRY.keys())


def unregister_dataset_class(benchmark_name: str) -> bool:
    """
    Remove a dataset class from the registry.

    Args:
        benchmark_name: Name of benchmark to remove (case-insensitive)

    Returns:
        True if benchmark was found and removed, False otherwise

    Example:
        >>> register_dataset_class("temp_benchmark", SomeTempDataset)
        >>> unregister_dataset_class("temp_benchmark")
        True
        >>> "temp_benchmark" in get_available_datasets()
        False
    """
    normalized_name = benchmark_name.lower().strip()
    removed_override = _OVERRIDES.pop(normalized_name, None)
    # Recompute the exported registry view
    base = _derive_registry_from_specs()
    base.update(_OVERRIDES)
    DATASET_REGISTRY.clear()
    DATASET_REGISTRY.update(base)
    return removed_override is not None


def get_dataset_class(benchmark_name: str) -> Optional[Type[BaseBenchmarkDataset]]:
    """
    Get the dataset class for a specific benchmark without creating an instance.

    Args:
        benchmark_name: Name of benchmark (case-insensitive)

    Returns:
        Dataset class if found, None otherwise

    Example:
        >>> cls = get_dataset_class("infinitebench")
        >>> cls.__name__
        'InfiniteBenchDataset'
    """
    normalized_name = benchmark_name.lower().strip()
    return DATASET_REGISTRY.get(normalized_name)


# Config creation functions consolidated here (no backward compatibility needed)



def create_vllm_config_from_dict(
    config_dict: Dict[str, Any], model_path: str
) -> Optional[VLLMLoadingConfig]:
    """
    [Deprecated]
    Create VLLMLoadingConfig from configuration dictionary.
    """
    if "loading_config" not in config_dict:
        return None

    loading_cfg_dict = config_dict["loading_config"]
    return VLLMLoadingConfig(
        model_path=loading_cfg_dict.get("model_path", model_path),
        gpu_memory_utilization=loading_cfg_dict.get("gpu_memory_utilization", 0.90),
        tensor_parallel_size=loading_cfg_dict.get("tensor_parallel_size"),
        max_model_len=loading_cfg_dict.get("max_model_len", DEFAULT_VLLM_MAX_MODEL_LEN),
        enforce_eager=loading_cfg_dict.get("enforce_eager", True),
        quantization=loading_cfg_dict.get("quantization"),
        trust_remote_code=loading_cfg_dict.get("trust_remote_code", True),
        dtype=loading_cfg_dict.get("dtype", "float16"),
        seed=loading_cfg_dict.get("seed", 42),
        disable_custom_all_reduce=loading_cfg_dict.get(
            "disable_custom_all_reduce", False
        ),
        additional_vllm_kwargs=loading_cfg_dict.get("additional_vllm_kwargs", {}),
    )


def create_experiment_config_from_dict(
    config_dict: Dict[str, Any],
    model_path: str,
    benchmark_config: BenchmarkConfig,
    loading_config: Optional[VLLMLoadingConfig] = None,
) -> ExperimentConfig:
    """
    [Deprecated]
    Create ExperimentConfig from configuration dictionary.
    """
    return ExperimentConfig(
        model_path=model_path,
        emotions=config_dict["emotions"],
        intensities=config_dict["intensities"],
        benchmark=benchmark_config,
        output_dir=config_dict.get("output_dir", "results/memory_experiments"),
        batch_size=config_dict.get("batch_size", 4),
        generation_config=config_dict.get("generation_config"),
        loading_config=loading_config,
        repe_eng_config=config_dict.get("repe_eng_config"),
        max_evaluation_workers=config_dict.get("max_evaluation_workers", 4),
        pipeline_queue_size=config_dict.get("pipeline_queue_size", 2),
        defer_evaluation=bool(config_dict.get("defer_evaluation", False)),
    )
