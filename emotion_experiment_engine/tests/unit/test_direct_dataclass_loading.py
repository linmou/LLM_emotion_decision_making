#!/usr/bin/env python3
"""
Test file for direct dataclass loading from YAML configurations.
Ensures dataclasses can be instantiated directly from YAML structures.
"""
import unittest
from pathlib import Path

from ...data_models import BenchmarkConfig, VLLMLoadingConfig, ExperimentConfig


class TestDirectDataclassLoading(unittest.TestCase):
    """Test direct dataclass instantiation from YAML-like dictionaries"""

    def test_vllm_loading_config_direct_instantiation(self):
        """Test direct VLLMLoadingConfig instantiation from YAML data"""
        loading_config_data = {
            "model_path": "/test/model",
            "gpu_memory_utilization": 0.85,
            "tensor_parallel_size": None,
            "max_model_len": 16384,
            "enforce_eager": True,
            "quantization": None,
            "trust_remote_code": True,
            "dtype": "bfloat16",
            "seed": 42,
            "disable_custom_all_reduce": False,
            "additional_vllm_kwargs": {"rope-theta": 1000000}
        }

        vllm_config = VLLMLoadingConfig(**loading_config_data)
        
        self.assertEqual(vllm_config.model_path, "/test/model")
        self.assertEqual(vllm_config.gpu_memory_utilization, 0.85)
        self.assertEqual(vllm_config.max_model_len, 16384)
        self.assertEqual(vllm_config.dtype, "bfloat16")
        self.assertEqual(vllm_config.additional_vllm_kwargs, {"rope-theta": 1000000})

    def test_vllm_config_with_defaults(self):
        """Test VLLMLoadingConfig with only required fields"""
        minimal_config = VLLMLoadingConfig(
            model_path="/test/model",
            gpu_memory_utilization=0.90,
            tensor_parallel_size=None,
            max_model_len=32768,
            enforce_eager=True,
            quantization=None,
            trust_remote_code=True,
            dtype="float16",
            seed=42,
            disable_custom_all_reduce=False,
            additional_vllm_kwargs={}
        )
        
        self.assertEqual(minimal_config.model_path, "/test/model")
        self.assertEqual(minimal_config.gpu_memory_utilization, 0.90)
        self.assertEqual(minimal_config.dtype, "float16")

    def test_benchmark_config_direct_instantiation(self):
        """Test direct BenchmarkConfig instantiation from YAML data"""
        benchmark_data = {
            "name": "infinitebench",
            "task_type": "passkey",
            "data_path": Path("data/memory_benchmarks/infinitebench_passkey.jsonl"),
            "base_data_dir": "data/memory_benchmarks",
            "sample_limit": 100,
            "augmentation_config": None,
            "enable_auto_truncation": True,
            "truncation_strategy": "right",
            "preserve_ratio": 0.95,
            "llm_eval_config": None
        }

        benchmark_config = BenchmarkConfig(**benchmark_data)
        
        self.assertEqual(benchmark_config.name, "infinitebench")
        self.assertEqual(benchmark_config.task_type, "passkey")
        self.assertEqual(benchmark_config.sample_limit, 100)
        self.assertTrue(benchmark_config.enable_auto_truncation)
        self.assertEqual(benchmark_config.preserve_ratio, 0.95)

    def test_benchmark_config_with_auto_generated_path(self):
        """Test BenchmarkConfig with path auto-generation"""
        benchmark = BenchmarkConfig(
            name="longbench",
            task_type="narrativeqa", 
            data_path=None,
            base_data_dir="data/memory_benchmarks",
            sample_limit=None,
            augmentation_config=None,
            enable_auto_truncation=False,
            truncation_strategy="right",
            preserve_ratio=0.8,
            llm_eval_config=None
        )
        
        # Test path auto-generation
        auto_path = benchmark.get_data_path()
        expected_path = Path("data/memory_benchmarks/longbench_narrativeqa.jsonl")
        self.assertEqual(auto_path, expected_path)

    def test_experiment_config_direct_instantiation(self):
        """Test direct ExperimentConfig instantiation"""
        benchmark = BenchmarkConfig(
            name="test_benchmark",
            task_type="test_task",
            data_path=Path("test.jsonl"),
            base_data_dir="data/memory_benchmarks",
            sample_limit=10,
            augmentation_config=None,
            enable_auto_truncation=False,
            truncation_strategy="right",
            preserve_ratio=0.8,
            llm_eval_config=None
        )
        
        loading_config = VLLMLoadingConfig(
            model_path="/test/model",
            gpu_memory_utilization=0.90,
            tensor_parallel_size=None,
            max_model_len=32768,
            enforce_eager=True,
            quantization=None,
            trust_remote_code=True,
            dtype="float16",
            seed=42,
            disable_custom_all_reduce=False,
            additional_vllm_kwargs={}
        )
        
        experiment_config = ExperimentConfig(
            model_path="/test/model",
            emotions=["anger", "happiness"],
            intensities=[1.5],
            benchmark=benchmark,
            output_dir="results/test",
            batch_size=4,
            generation_config={"temperature": 0.1},
            loading_config=loading_config,
            repe_eng_config={"direction_method": "pca"},
            max_evaluation_workers=2,
            pipeline_queue_size=2,
            defer_evaluation=False,
        )
        
        self.assertEqual(experiment_config.model_path, "/test/model")
        self.assertEqual(experiment_config.emotions, ["anger", "happiness"])
        self.assertEqual(experiment_config.benchmark.name, "test_benchmark")
        self.assertEqual(experiment_config.batch_size, 4)

    def test_yaml_like_loading_pattern(self):
        """Test the YAML-to-dataclass loading pattern used in the refactored code"""
        # Simulate YAML config structure
        yaml_config = {
            "models": ["Qwen/Qwen2.5-0.5B-Instruct"],
            "emotions": ["anger", "happiness"],
            "intensities": [1.5],
            "benchmarks": [
                {
                    "name": "infinitebench",
                    "task_type": "passkey",
                    "sample_limit": 100,
                    "augmentation_config": None,
                    "enable_auto_truncation": True,
                    "truncation_strategy": "right", 
                    "preserve_ratio": 0.95
                }
            ],
            "loading_config": {
                "model_path": None,  # Set at runtime
                "gpu_memory_utilization": 0.90,
                "tensor_parallel_size": None,
                "max_model_len": 32768,
                "enforce_eager": True,
                "quantization": None,
                "trust_remote_code": True,
                "dtype": "float16", 
                "seed": 42,
                "disable_custom_all_reduce": False,
                "additional_vllm_kwargs": {"rope-theta": 1000000}
            },
            "output_dir": "results/test"
        }
        
        # Test direct loading pattern
        model_path = yaml_config["models"][0]
        benchmark_data = yaml_config["benchmarks"][0]
        loading_cfg = yaml_config["loading_config"]
        
        # Create BenchmarkConfig directly
        benchmark = BenchmarkConfig(
            name=benchmark_data["name"],
            task_type=benchmark_data["task_type"],
            data_path=None,  # Auto-generate
            base_data_dir="data/memory_benchmarks",
            sample_limit=benchmark_data.get("sample_limit"),
            augmentation_config=benchmark_data.get("augmentation_config"),
            enable_auto_truncation=benchmark_data.get("enable_auto_truncation", False),
            truncation_strategy=benchmark_data.get("truncation_strategy", "right"),
            preserve_ratio=benchmark_data.get("preserve_ratio", 0.8),
            llm_eval_config=None,
        )
        
        # Create VLLMLoadingConfig directly  
        loading_config = VLLMLoadingConfig(
            model_path=loading_cfg.get("model_path") or model_path,
            gpu_memory_utilization=loading_cfg.get("gpu_memory_utilization", 0.90),
            tensor_parallel_size=loading_cfg.get("tensor_parallel_size"),
            max_model_len=loading_cfg.get("max_model_len", 8192),
            enforce_eager=loading_cfg.get("enforce_eager", True),
            quantization=loading_cfg.get("quantization"),
            trust_remote_code=loading_cfg.get("trust_remote_code", True),
            dtype=loading_cfg.get("dtype", "float16"),
            seed=loading_cfg.get("seed", 42),
            disable_custom_all_reduce=loading_cfg.get("disable_custom_all_reduce", False),
            additional_vllm_kwargs=loading_cfg.get("additional_vllm_kwargs", {}),
        )
        
        # Verify the configuration loaded correctly
        self.assertEqual(benchmark.name, "infinitebench")
        self.assertEqual(loading_config.model_path, "Qwen/Qwen2.5-0.5B-Instruct")
        self.assertEqual(loading_config.additional_vllm_kwargs, {"rope-theta": 1000000})


if __name__ == "__main__":
    unittest.main()
