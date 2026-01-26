"""
Covers registry mapping for ("swebench", "patch").
Responsible files:
 - emotion_experiment_engine/benchmark_component_registry.py
 - emotion_experiment_engine/datasets/swebench.py
 - emotion_experiment_engine/swebench_prompt_wrapper.py (if used)
"""

import pytest

from emotion_experiment_engine.data_models import BenchmarkConfig
from emotion_experiment_engine.dataset_factory import get_dataset_class


def _bench_config() -> BenchmarkConfig:
    return BenchmarkConfig(
        name="swebench",
        task_type="patch",
        data_path=None,  # not needed for import check; dataset instantiation may still fail later
        base_data_dir=None,
        sample_limit=1,
        augmentation_config=None,
        enable_auto_truncation=False,
        truncation_strategy="right",
        preserve_ratio=1.0,
        llm_eval_config=None,
    )


def test_registry_entry_exists_for_swebench_patch():
    # Ensure the dataset class is discoverable via factory registry
    cls = get_dataset_class("swebench")
    assert cls is not None
