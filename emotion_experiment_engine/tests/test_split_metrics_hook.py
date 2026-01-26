"""
Red phase tests for split-level metrics integration.

We start by asserting that datasets expose a split-level aggregation hook and
that the experiment persistence layer writes out the aggregated metrics.

Both tests are expected to FAIL initially (TDD Red) until the production
implementations are added.
"""

import json
import logging
import sys
import types
from pathlib import Path
from typing import List

import pytest

from emotion_experiment_engine.datasets.base import BaseBenchmarkDataset
from emotion_experiment_engine.data_models import ResultRecord

# Preload lightweight stubs for heavy deps so importing EmotionExperiment does not fail
nm = types.ModuleType("neuro_manipulation")
nm_model_utils = types.ModuleType("neuro_manipulation.model_utils")
nm_model_utils.load_emotion_readers = lambda *args, **kwargs: None
nm_model_utils.setup_model_and_tokenizer = lambda *args, **kwargs: (None, None)
nm_config = types.ModuleType("neuro_manipulation.configs")
nm_experiment_config = types.ModuleType("neuro_manipulation.configs.experiment_config")
nm_experiment_config.get_repe_eng_config = lambda *args, **kwargs: None
nm_layer_detector = types.ModuleType("neuro_manipulation.model_layer_detector")
class _LD:  # minimal stub
    pass
nm_layer_detector.ModelLayerDetector = _LD
nm_repe = types.ModuleType("neuro_manipulation.repe")
nm_repe_pipelines = types.ModuleType("neuro_manipulation.repe.pipelines")
nm_repe_pipelines.get_pipeline = lambda *args, **kwargs: None

sys.modules.setdefault("neuro_manipulation", nm)
sys.modules.setdefault("neuro_manipulation.model_utils", nm_model_utils)
sys.modules.setdefault("neuro_manipulation.configs", nm_config)
sys.modules.setdefault("neuro_manipulation.configs.experiment_config", nm_experiment_config)
sys.modules.setdefault("neuro_manipulation.model_layer_detector", nm_layer_detector)
sys.modules.setdefault("neuro_manipulation.repe", nm_repe)
sys.modules.setdefault("neuro_manipulation.repe.pipelines", nm_repe_pipelines)

# Provide a lightweight stub for our internal registry to avoid importing heavy datasets
reg = types.ModuleType("emotion_experiment_engine.benchmark_component_registry")
def _stub_create_benchmark_components(**kwargs):
    # Return (prompt_wrapper_partial, answer_wrapper_partial, dataset)
    return (lambda **k: ""), (lambda x, **k: x), object()
reg.create_benchmark_components = _stub_create_benchmark_components
sys.modules.setdefault(
    "emotion_experiment_engine.benchmark_component_registry", reg
)



def test_base_dataset_exposes_split_level_hook():
    """
    Expect BaseBenchmarkDataset to define a split-level metrics hook so all
    concrete datasets can implement dataset-scope aggregation.

    This test drives the addition of a default method on the base class.
    """
    assert hasattr(
        BaseBenchmarkDataset, "compute_split_metrics"
    ), "BaseBenchmarkDataset must expose compute_split_metrics(records)"

    hook = getattr(BaseBenchmarkDataset, "compute_split_metrics")
    assert callable(hook), "compute_split_metrics must be callable"


def test_experiment_code_contains_split_metrics_persistence():
    """
    Textual contract test: ensure experiment._save_results contains logic to
    call compute_split_metrics(...) and persist to 'split_metrics.json'.

    We avoid importing the heavy module; instead verify source contains the
    two key strings, which protects against accidental removal.
    """
    path = Path("emotion_experiment_engine/experiment.py")
    src = path.read_text(encoding="utf-8")
    assert "compute_split_metrics(" in src, "Experiment must call dataset.compute_split_metrics()"
    assert "split_metrics.json" in src, "Experiment must persist split_metrics.json file"
