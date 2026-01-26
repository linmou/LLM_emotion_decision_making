#!/usr/bin/env python3
"""
API Compatibility Regression Tests

These tests ensure that public APIs remain stable across versions, preventing
breaking changes that would affect research reproducibility and user code.
"""

import inspect
import pytest
from unittest.mock import patch, MagicMock
from typing import Dict, Any, List, Callable

# Import all public APIs to test
from emotion_experiment_engine.dataset_factory import (
    create_dataset_from_config, 
    DATASET_REGISTRY,
    register_dataset_class
)
from emotion_experiment_engine.data_models import BenchmarkConfig, ExperimentConfig, BenchmarkItem
from emotion_experiment_engine.evaluation_utils import llm_evaluate_response
from emotion_experiment_engine.experiment import EmotionExperiment
from emotion_experiment_engine.datasets.base import BaseBenchmarkDataset


def make_benchmark_config(**overrides) -> BenchmarkConfig:
    base_kwargs = {
        "name": "test",
        "task_type": "task",
        "data_path": None,
        "base_data_dir": None,
        "sample_limit": None,
        "augmentation_config": None,
        "enable_auto_truncation": False,
        "truncation_strategy": "right",
        "preserve_ratio": 0.8,
        "llm_eval_config": None,
    }
    base_kwargs.update(overrides)
    return BenchmarkConfig(**base_kwargs)


def make_experiment_config(**overrides) -> ExperimentConfig:
    base_kwargs = {
        "model_path": "test_model",
        "emotions": ["anger"],
        "intensities": [1.0],
        "benchmark": make_benchmark_config(),
        "output_dir": "test_output",
        "batch_size": 1,
        "generation_config": None,
        "loading_config": None,
        "repe_eng_config": None,
        "max_evaluation_workers": 1,
        "pipeline_queue_size": 1,
    }
    base_kwargs.update(overrides)
    return ExperimentConfig(**base_kwargs)


@pytest.mark.regression
class TestDatasetFactoryAPICompatibility:
    """Ensure dataset factory API remains stable"""
    
    def test_create_dataset_from_config_signature(self):
        """Factory function signature must remain stable"""
        sig = inspect.signature(create_dataset_from_config)
        params = list(sig.parameters.keys())
        
        # Required parameters that must exist
        required_params = ['config']
        for param in required_params:
            assert param in params, f"Required parameter '{param}' missing from create_dataset_from_config"
        
        # Verify parameter types
        config_param = sig.parameters['config']
        assert config_param.annotation in [BenchmarkConfig, inspect.Parameter.empty], \
            "config parameter must accept BenchmarkConfig"
    
    def test_dataset_registry_interface_stability(self):
        """Dataset registry must maintain interface"""
        # Registry must exist and be accessible
        assert DATASET_REGISTRY is not None, "DATASET_REGISTRY must be accessible"
        assert isinstance(DATASET_REGISTRY, dict), "DATASET_REGISTRY must be a dictionary"
        
        # Core datasets must be registered
        expected_datasets = ["infinitebench", "longbench", "locomo", "emotion_check"]
        registry_keys = [k.lower() for k in DATASET_REGISTRY.keys()]
        
        for dataset in expected_datasets:
            assert dataset in registry_keys, f"Dataset '{dataset}' missing from registry"
    
    def test_register_dataset_class_signature(self):
        """Dynamic registration function must remain stable"""
        sig = inspect.signature(register_dataset_class)
        params = list(sig.parameters.keys())
        
        expected_params = ['benchmark_name', 'dataset_class']
        assert len(params) >= len(expected_params), "register_dataset_class missing required parameters"
        
        for param in expected_params:
            assert param in params, f"Parameter '{param}' missing from register_dataset_class"
    
    def test_dataset_creation_backward_compatibility(self):
        """Ensure dataset creation supports legacy-like configs via registry"""

        class StubDataset(BaseBenchmarkDataset):
            def _load_and_parse_data(self):
                return [BenchmarkItem(id="1", input_text="q", context=None, ground_truth="a", metadata=None)]

            def evaluate_response(self, response, ground_truth, task_name, prompt):
                return 1.0

            def get_task_metrics(self, task_name):
                return ["accuracy"]

        with patch.dict(DATASET_REGISTRY, {"legacy": StubDataset}, clear=False):
            legacy_configs = [
                make_benchmark_config(name="legacy", task_type="passkey"),
                make_benchmark_config(name="legacy", task_type="passkey", sample_limit=5),
                make_benchmark_config(name="legacy", task_type="passkey", llm_eval_config={"model": "gpt-4o-mini"}),
            ]

            for config in legacy_configs:
                dataset = create_dataset_from_config(config)
                assert isinstance(dataset, StubDataset), f"Unexpected dataset type for {config}"


@pytest.mark.regression
class TestDataModelCompatibility:
    """Ensure data model interfaces remain stable"""
    
    def test_benchmark_config_fields(self):
        """BenchmarkConfig must maintain required fields"""
        # Test that we can create config with minimal required fields
        config = make_benchmark_config()
        
        # Required fields must exist
        required_fields = ["name", "task_type"]
        for field in required_fields:
            assert hasattr(config, field), f"BenchmarkConfig missing field '{field}'"
            assert getattr(config, field) is not None, f"BenchmarkConfig field '{field}' is None"
    
    def test_benchmark_config_optional_fields(self):
        """Optional fields should have reasonable defaults"""
        config = make_benchmark_config()
        
        # Optional fields with defaults
        optional_fields = [
            "sample_limit",
            "augmentation_config",
            "base_data_dir",
            "llm_eval_config",
        ]

        for field in optional_fields:
            assert hasattr(config, field), f"BenchmarkConfig missing optional field '{field}'"
    
    def test_experiment_config_backward_compatibility(self):
        """ExperimentConfig must support legacy initialization"""
        # Test minimal initialization
        benchmark_config = make_benchmark_config()
        
        try:
            config = make_experiment_config(emotions=["anger", "neutral"], benchmark=benchmark_config)
            
            # Verify required fields exist
            assert hasattr(config, "model_path")
            assert hasattr(config, "emotions")
            assert hasattr(config, "benchmark")
            
        except Exception as e:
            pytest.fail(f"ExperimentConfig creation failed: {str(e)}")


@pytest.mark.regression
class TestEvaluationAPICompatibility:
    """Ensure evaluation API remains stable"""
    
    def test_llm_evaluate_response_signature(self):
        """LLM evaluation function signature must be stable"""
        sig = inspect.signature(llm_evaluate_response)
        params = list(sig.parameters.keys())
        
        # Core parameters that must exist
        expected_params = ["system_prompt", "query", "llm_eval_config"]
        for param in expected_params:
            assert param in params, f"Parameter '{param}' missing from llm_evaluate_response"
    
    @patch('emotion_experiment_engine.evaluation_utils._get_openai_client')
    def test_llm_evaluate_response_return_format(self, mock_client_factory):
        """LLM evaluation must return parsed JSON dictionaries"""
        mock_client = MagicMock()
        mock_client_factory.return_value = mock_client
        mock_client.chat.completions.create.return_value = MagicMock(
            choices=[
                MagicMock(
                    message=MagicMock(content='{"label": "neutral", "confidence": 0.8}')
                )
            ]
        )

        result = llm_evaluate_response(
            system_prompt="Test",
            query="Test query",
            llm_eval_config={"model": "gpt-4o-mini", "temperature": 0.0},
        )

        assert isinstance(result, dict), "llm_evaluate_response must return a dictionary"
        assert result["label"] == "neutral"
        assert result["confidence"] == 0.8
        mock_client.chat.completions.create.assert_called_once()
    
    def test_base_dataset_evaluate_batch_signature(self):
        """Base dataset evaluate_batch must keep prompts parameter"""
        sig = inspect.signature(BaseBenchmarkDataset.evaluate_batch)
        params = list(sig.parameters.keys())

        expected_prefix = [
            "self",
            "responses",
            "ground_truths",
            "task_names",
            "prompts",
        ]
        assert params[: len(expected_prefix)] == expected_prefix


@pytest.mark.regression
class TestBaseBenchmarkDatasetInterface:
    """Ensure base dataset interface remains stable"""
    
    def test_abstract_methods_stability(self):
        """Abstract base class must define required methods"""
        # Check that BaseBenchmarkDataset has required abstract methods
        abstract_methods = getattr(BaseBenchmarkDataset, '__abstractmethods__', set())
        
        expected_abstract = {
            "_load_and_parse_data",
            "evaluate_response", 
            "get_task_metrics"
        }
        
        for method in expected_abstract:
            assert method in abstract_methods or hasattr(BaseBenchmarkDataset, method), \
                f"BaseBenchmarkDataset missing required method '{method}'"
    
    def test_pytorch_dataset_interface(self):
        """Must maintain PyTorch Dataset interface"""
        # BaseBenchmarkDataset must have PyTorch Dataset methods
        required_methods = ["__len__", "__getitem__"]
        
        for method in required_methods:
            assert hasattr(BaseBenchmarkDataset, method), \
                f"BaseBenchmarkDataset missing PyTorch Dataset method '{method}'"
            assert callable(getattr(BaseBenchmarkDataset, method)), \
                f"Method '{method}' is not callable"
    
    def test_evaluate_response_signature_stability(self):
        """evaluate_response method signature must be stable across all datasets"""
        # Get all dataset classes
        from emotion_experiment_engine.datasets import infinitebench, longbench, locomo, emotion_check
        
        dataset_classes = [
            infinitebench.InfiniteBenchDataset,
            longbench.LongBenchDataset,
            locomo.LoCoMoDataset,
            emotion_check.EmotionCheckDataset
        ]
        
        required = {"self", "response", "ground_truth", "task_name"}

        for dataset_class in dataset_classes:
            if hasattr(dataset_class, 'evaluate_response'):
                params = set(inspect.signature(dataset_class.evaluate_response).parameters.keys())
                assert required.issubset(params), (
                    f"{dataset_class.__name__}.evaluate_response missing required parameters: "
                    f"{required - params}"
                )


@pytest.mark.regression
class TestExperimentClassCompatibility:
    """Ensure experiment orchestration API remains stable"""
    
    def test_experiment_initialization_compatibility(self):
        """EmotionExperiment must accept standard config"""
        benchmark_config = make_benchmark_config()

        experiment_config = make_experiment_config(benchmark=benchmark_config)
        
        basic_tokenizer = MagicMock(name="tokenizer")

        # Mock heavy dependencies
        with patch('emotion_experiment_engine.experiment.load_emotion_readers'), \
             patch('emotion_experiment_engine.experiment.setup_model_and_tokenizer', return_value=(MagicMock(), basic_tokenizer, MagicMock(), None)), \
             patch('emotion_experiment_engine.experiment.get_pipeline'), \
             patch('neuro_manipulation.utils.load_tokenizer_only', return_value=(basic_tokenizer, None)), \
             patch('neuro_manipulation.model_layer_detector.ModelLayerDetector.num_layers', return_value=1), \
             patch('emotion_experiment_engine.experiment.create_benchmark_components') as mock_components, \
             patch('emotion_experiment_engine.experiment.DataLoader') as mock_dataloader:

            class _StubDataset:
                def __init__(self):
                    self.eval_workers = 1

                def __len__(self):  # pragma: no cover - trivial
                    return 0

                def __iter__(self):  # pragma: no cover - trivial
                    return iter([])

                def collate_fn(self, batch):
                    return {"prompts": [], "items": [], "ground_truths": []}

                def evaluate_batch(self, *args, **kwargs):
                    return []

            mock_components.return_value = (lambda **_: None, lambda x: x, _StubDataset())
            mock_dataloader.return_value = []

            try:
                experiment = EmotionExperiment(experiment_config)
                assert experiment is not None, "Failed to create EmotionExperiment"

                # Verify key attributes exist
                assert hasattr(experiment, 'config'), "Missing config attribute"

            except Exception as e:
                pytest.fail(f"EmotionExperiment initialization failed: {str(e)}")
    
    def test_experiment_public_methods_stability(self):
        """Public methods must remain available"""
        # Key public methods that should remain stable
        expected_methods = [
            "__init__",
            "run_sanity_check",
            "run_experiment",
            "close",
        ]

        for method in expected_methods:
            assert hasattr(EmotionExperiment, method), \
                f"EmotionExperiment missing method '{method}'"
            assert callable(getattr(EmotionExperiment, method)), \
                f"Method '{method}' is not callable"


@pytest.mark.regression
class TestVersionCompatibilityMatrix:
    """Test compatibility across dependency versions"""
    
    def test_python_version_support(self):
        """Ensure Python version requirements haven't changed"""
        import sys
        
        # Minimum Python version requirement
        min_python = (3, 8)
        current_python = sys.version_info[:2]
        
        assert current_python >= min_python, \
            f"Python {current_python} < minimum required {min_python}"
    
    def test_import_stability(self):
        """All public imports must remain available"""
        # Test that key imports work without errors
        try:
            # Core functionality
            from emotion_experiment_engine import experiment, dataset_factory, data_models
            from emotion_experiment_engine.datasets import base, infinitebench, longbench, locomo
            from emotion_experiment_engine import evaluation_utils, config_loader
            
            # Test the new file mentioned by user
            from emotion_experiment_engine.tests import test_answer_wrapper_comprehensive
            
        except ImportError as e:
            pytest.fail(f"Import regression detected: {str(e)}")
    
    @patch('emotion_experiment_engine.evaluation_utils._get_openai_client')
    def test_openai_api_version_compatibility(self, mock_client_factory):
        """Test OpenAI API version compatibility"""
        mock_client = MagicMock()
        mock_client_factory.return_value = mock_client
        mock_client.chat.completions.create.return_value = MagicMock(
            choices=[
                MagicMock(
                    message=MagicMock(content='{"emotion": "test", "confidence": 1.0}')
                )
            ]
        )

        try:
            result = llm_evaluate_response(
                system_prompt="test",
                query="test",
                llm_eval_config={"model": "gpt-4o-mini"}
            )
            assert isinstance(result, dict), "OpenAI API compatibility broken"

        except Exception as e:
            pytest.fail(f"OpenAI API compatibility issue: {str(e)}")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-m", "regression"])
