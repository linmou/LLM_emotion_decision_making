"""
Unit tests for data models.
"""
import unittest
from pathlib import Path

from ...data_models import (
    ResultRecord, BenchmarkConfig, ExperimentConfig, BenchmarkItem, DEFAULT_GENERATION_CONFIG
)


class TestDataModels(unittest.TestCase):
    """Test data model classes"""
    
    def test_result_record_creation(self):
        """Test ResultRecord creation and attributes"""
        result = ResultRecord(
            emotion="anger",
            intensity=1.0,
            item_id="test_item",
            task_name="passkey",
            prompt="What is the passkey?",
            response="12345",
            ground_truth="12345",
            score=1.0,
            repeat_id=0,
            metadata={"test": "data"}
        )
        
        self.assertEqual(result.emotion, "anger")
        self.assertEqual(result.intensity, 1.0)
        self.assertEqual(result.item_id, "test_item")
        self.assertEqual(result.task_name, "passkey")
        self.assertEqual(result.score, 1.0)
        self.assertEqual(result.metadata["test"], "data")
    
    def test_result_record_optional_metadata(self):
        """Test ResultRecord with no metadata"""
        result = ResultRecord(
            emotion="neutral",
            intensity=0.0,
            item_id=1,
            task_name="kv_retrieval",
            prompt="test prompt",
            response="test response",
            ground_truth="test ground truth",
            score=0.5,
            repeat_id=0,
        )
        
        self.assertIsNone(result.metadata)
    
    def test_benchmark_config_creation(self):
        """Test BenchmarkConfig creation"""
        config = BenchmarkConfig(
            name="infinitebench",
            data_path=Path("test_data.jsonl"),
            task_type="passkey",
            base_data_dir="data/test_benchmarks",
            sample_limit=10,
            augmentation_config=None,
            enable_auto_truncation=True,
            truncation_strategy="right",
            preserve_ratio=0.8,
            llm_eval_config=None
        )
        
        self.assertEqual(config.name, "infinitebench")
        self.assertEqual(config.task_type, "passkey")
        self.assertEqual(config.sample_limit, 10)
        self.assertIsInstance(config.data_path, Path)
        self.assertTrue(config.enable_auto_truncation)
        self.assertEqual(config.truncation_strategy, "right")
        self.assertEqual(config.preserve_ratio, 0.8)
    
    def test_benchmark_config_optional_sample_limit(self):
        """Test BenchmarkConfig with no sample limit"""
        config = BenchmarkConfig(
            name="locomo",
            data_path=Path("locomo.json"),
            task_type="conversational_qa",
            base_data_dir="data/locomo_data",
            sample_limit=None,
            augmentation_config=None,
            enable_auto_truncation=False,
            truncation_strategy="left",
            preserve_ratio=0.9,
            llm_eval_config=None
        )
        
        self.assertIsNone(config.sample_limit)
        self.assertFalse(config.enable_auto_truncation)
    
    def test_experiment_config_creation(self):
        """Test ExperimentConfig creation"""
        benchmark_config = BenchmarkConfig(
            name="infinitebench",
            data_path=Path("test.jsonl"),
            task_type="passkey",
            base_data_dir="data/test_benchmarks",
            sample_limit=None,
            augmentation_config=None,
            enable_auto_truncation=False,
            truncation_strategy="right",
            preserve_ratio=0.8,
            llm_eval_config=None
        )
        
        config = ExperimentConfig(
            model_path="/path/to/model",
            emotions=["anger", "happiness"],
            intensities=[0.5, 1.0],
            benchmark=benchmark_config,
            output_dir="test_output",
            batch_size=4,
            generation_config={"temperature": 0.7},
            loading_config=None,
            repe_eng_config=None,
            max_evaluation_workers=2,
            pipeline_queue_size=1,
            defer_evaluation=False,
        )
        
        self.assertEqual(config.model_path, "/path/to/model")
        self.assertEqual(len(config.emotions), 2)
        self.assertEqual(len(config.intensities), 2)
        self.assertEqual(config.batch_size, 4)
        self.assertEqual(config.generation_config["temperature"], 0.7)
    
    def test_experiment_config_defaults(self):
        """Test ExperimentConfig with default values"""
        benchmark_config = BenchmarkConfig(
            name="test",
            data_path=Path("test.jsonl"),
            task_type="test",
            base_data_dir="data/test_benchmarks",
            sample_limit=None,
            augmentation_config=None,
            enable_auto_truncation=False,
            truncation_strategy="right",
            preserve_ratio=0.8,
            llm_eval_config=None
        )
        
        config = ExperimentConfig(
            model_path="/path/to/model",
            emotions=["anger"],
            intensities=[1.0],
            benchmark=benchmark_config,
            output_dir="output",
            batch_size=4,
            generation_config=None,
            loading_config=None,
            repe_eng_config=None,
            max_evaluation_workers=4,
            pipeline_queue_size=2,
            defer_evaluation=False,
        )
        
        self.assertEqual(config.batch_size, 4)
        self.assertIsNone(config.generation_config)
    
    def test_benchmark_item_creation(self):
        """Test BenchmarkItem creation"""
        item = BenchmarkItem(
            id="item_1",
            input_text="What is the answer?",
            context="Some context text",
            ground_truth="42",
            metadata={"source": "test"}
        )
        
        self.assertEqual(item.id, "item_1")
        self.assertEqual(item.input_text, "What is the answer?")
        self.assertEqual(item.context, "Some context text")
        self.assertEqual(item.ground_truth, "42")
        self.assertEqual(item.metadata["source"], "test")
    
    def test_benchmark_item_optional_fields(self):
        """Test BenchmarkItem with all fields specified"""
        item = BenchmarkItem(
            id=123,
            input_text="Test question",
            context=None,
            ground_truth="answer",
            metadata=None
        )
        
        self.assertEqual(item.id, 123)
        self.assertEqual(item.input_text, "Test question")
        self.assertIsNone(item.context)
        self.assertEqual(item.ground_truth, "answer")
        self.assertIsNone(item.metadata)
    
    def test_default_generation_config(self):
        """Test default generation config values"""
        self.assertIn("temperature", DEFAULT_GENERATION_CONFIG)
        self.assertIn("max_new_tokens", DEFAULT_GENERATION_CONFIG)
        self.assertIn("do_sample", DEFAULT_GENERATION_CONFIG)
        self.assertIn("top_p", DEFAULT_GENERATION_CONFIG)
        
        # Check reasonable defaults
        self.assertEqual(DEFAULT_GENERATION_CONFIG["temperature"], 0.1)
        self.assertEqual(DEFAULT_GENERATION_CONFIG["max_new_tokens"], 100)
        self.assertFalse(DEFAULT_GENERATION_CONFIG["do_sample"])

    def test_benchmark_config_get_data_path_requires_base_data_dir(self):
        """Test BenchmarkConfig.get_data_path() properly enforces base_data_dir requirement
        
        This test ensures the assert works correctly for code quality by catching
        misconfigured datasets that don't specify where their data files are located.
        """
        # Create BenchmarkConfig with None base_data_dir (misconfiguration)
        config = BenchmarkConfig(
            name="some_benchmark",
            task_type="some_task", 
            data_path=None,  # Will trigger path generation
            base_data_dir=None,  # This should cause assertion failure
            sample_limit=None,
            augmentation_config=None,
            enable_auto_truncation=False,
            truncation_strategy="right",
            preserve_ratio=0.8,
            llm_eval_config=None
        )
        
        # This SHOULD raise an AssertionError to catch misconfiguration
        with self.assertRaises(AssertionError) as cm:
            config.get_data_path()
            
        self.assertIn("base_data_dir is required", str(cm.exception))

    def test_benchmark_config_get_data_path_with_explicit_path(self):
        """Test BenchmarkConfig.get_data_path() respects explicitly set data_path"""
        explicit_path = Path("/custom/path/to/data.jsonl")
        config = BenchmarkConfig(
            name="custom",
            task_type="test",
            data_path=explicit_path,
            base_data_dir=None,  # Should be ignored when data_path is set
            sample_limit=None,
            augmentation_config=None,
            enable_auto_truncation=False,
            truncation_strategy="right",
            preserve_ratio=0.8,
            llm_eval_config=None
        )
        
        # Should return the explicit path, not generate one
        result_path = config.get_data_path()
        self.assertEqual(result_path, explicit_path)
        self.assertEqual(str(result_path), "/custom/path/to/data.jsonl")

    def test_benchmark_config_get_data_path_priority_order(self):
        """Test get_data_path() parameter priority: provided > self.base_data_dir > default"""
        config = BenchmarkConfig(
            name="test",
            task_type="example",
            data_path=None,
            base_data_dir="config_base_dir",
            sample_limit=None,
            augmentation_config=None,
            enable_auto_truncation=False,
            truncation_strategy="right",
            preserve_ratio=0.8,
            llm_eval_config=None
        )
        
        # Test provided parameter takes precedence
        path_with_param = config.get_data_path("provided_base_dir")
        self.assertIn("provided_base_dir", str(path_with_param))
        
        # Test config base_data_dir is used when no parameter provided
        path_with_config = config.get_data_path()
        self.assertIn("config_base_dir", str(path_with_config))


if __name__ == '__main__':
    unittest.main()
