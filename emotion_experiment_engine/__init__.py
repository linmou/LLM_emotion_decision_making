"""
Emotion Memory Experiments Package

A framework for testing emotion effects on long-context memory benchmarks.
This package provides tools to run emotion manipulation experiments on various
memory benchmarks like InfiniteBench and LoCoMo.

Key Components:
- EmotionMemoryExperiment: Main experiment orchestrator
- BenchmarkAdapter: Adapters for different benchmark formats
- Data models for configuration and results
- YAML configuration loader for easy experiment setup
- Comprehensive test suite

Example Usage:
    # Using YAML configuration (recommended)
    from emotion_experiment_engine.config_loader import load_emotion_memory_config

    exp_config = load_emotion_memory_config("config/emotion_memory_passkey.yaml")
    experiment = EmotionMemoryExperiment(exp_config)
    results = experiment.run_experiment()

    # Using programmatic configuration
    from emotion_experiment_engine import EmotionMemoryExperiment
    from emotion_experiment_engine.data_models import ExperimentConfig, BenchmarkConfig

    benchmark_config = BenchmarkConfig(
        name="infinitebench",
        data_path="passkey_data.jsonl",
        task_type="passkey"
    )

    exp_config = ExperimentConfig(
        model_path="/path/to/model",
        emotions=["anger", "happiness"],
        intensities=[1.0, 1.5],
        benchmark=benchmark_config,
        output_dir="results"
    )

    experiment = EmotionMemoryExperiment(exp_config)
    results = experiment.run_experiment()
"""

# Import EmotionMemoryExperiment conditionally to avoid vllm dependency during imports
try:
    from .experiment import EmotionExperiment
except Exception:
    # vllm not available, EmotionMemoryExperiment will be None
    EmotionExperiment = None
# Adapters replaced by smart datasets in refactoring
from .config_loader import EmotionMemoryConfigLoader, load_emotion_memory_config
from .data_models import (
    DEFAULT_GENERATION_CONFIG,
    BenchmarkConfig,
    ExperimentConfig,
    ResultRecord,
)
__version__ = "1.0.0"


def __getattr__(name: str):
    if name == "EmotionExperiment":
        from .experiment import EmotionExperiment as _EmotionExperiment

        return _EmotionExperiment
    if name == "swebench_evaluation":
        from . import swebench_evaluation as _swebench_evaluation

        return _swebench_evaluation
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
__all__ = [
    "EmotionExperiment",
    "ExperimentConfig",
    "BenchmarkConfig",
    "ResultRecord",
    "DEFAULT_GENERATION_CONFIG",
    # "get_adapter" removed in smart dataset refactoring
    "load_emotion_memory_config",
    "EmotionMemoryConfigLoader",
]
