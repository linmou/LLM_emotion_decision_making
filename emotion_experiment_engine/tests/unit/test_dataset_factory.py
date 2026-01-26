#!/usr/bin/env python3
"""
Test file for: Dataset Factory Pattern (TDD Red phase)
Purpose: Test registry-based factory for eliminating if-else chains

This test suite ensures the factory pattern properly creates specialized dataset
classes based on configuration without if-else branching.
"""

import unittest
import tempfile
import json
from pathlib import Path
from typing import Any, Dict, List

import pytest

# These imports will initially fail (Red phase) - that's expected!
try:
    from emotion_experiment_engine.dataset_factory import (
        create_dataset_from_config, 
        DATASET_REGISTRY,
        register_dataset_class,
        get_available_datasets
    )
except ImportError:
    create_dataset_from_config = None
    DATASET_REGISTRY = None
    register_dataset_class = None
    get_available_datasets = None

from emotion_experiment_engine.data_models import BenchmarkConfig

# Import existing datasets (should exist from previous phases)
try:
    from emotion_experiment_engine.datasets.base import BaseBenchmarkDataset
    from emotion_experiment_engine.datasets.infinitebench import InfiniteBenchDataset
    from emotion_experiment_engine.datasets.longbench import LongBenchDataset
    from emotion_experiment_engine.datasets.locomo import LoCoMoDataset
except ImportError:
    # These should exist from previous phases, but handle gracefully
    BaseBenchmarkDataset = None
    InfiniteBenchDataset = None
    LongBenchDataset = None
    LoCoMoDataset = None


class TestDatasetFactory(unittest.TestCase):
    """Test registry-based dataset factory implementation"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.temp_files = []
    
    def tearDown(self):
        """Clean up temporary files"""
        for temp_file in self.temp_files:
            try:
                temp_file.unlink()
            except FileNotFoundError:
                pass
    
    def _create_test_data_file(self, benchmark_type: str) -> Path:
        """Create temporary test data file appropriate for benchmark type"""
        if benchmark_type == "infinitebench":
            # JSONL format
            test_data = [
                {"id": 0, "input": "Test input", "answer": "Test answer", "task_name": "passkey"}
            ]
            temp_file = Path(tempfile.mktemp(suffix='.jsonl'))
            with open(temp_file, 'w') as f:
                for item in test_data:
                    f.write(json.dumps(item) + '\n')
        
        elif benchmark_type == "longbench":
            # JSON format
            test_data = [
                {"id": "test_0", "input": "Test input", "answers": ["Test answer"], "task_name": "narrativeqa"}
            ]
            temp_file = Path(tempfile.mktemp(suffix='.json'))
            with open(temp_file, 'w') as f:
                json.dump(test_data, f)
                
        elif benchmark_type == "locomo":
            # JSON format with conversation structure
            test_data = [
                {
                    "sample_id": "sample_1",
                    "conversation": {
                        "session_1": [{"speaker": "Alice", "text": "Hello"}],
                        "session_1_date_time": "2024-01-01 10:00:00"
                    },
                    "qa": [{"question": "What did Alice say?", "answer": "Hello", "category": "greeting", "evidence": ["session_1"]}]
                }
            ]
            temp_file = Path(tempfile.mktemp(suffix='.json'))
            with open(temp_file, 'w') as f:
                json.dump(test_data, f)
        else:
            raise ValueError(f"Unknown benchmark type: {benchmark_type}")
        
        self.temp_files.append(temp_file)
        return temp_file
    
    def test_dataset_registry_exists(self):
        """Test that dataset registry is properly defined"""
        if DATASET_REGISTRY is None:
            self.skipTest("Dataset registry not implemented yet (Red phase)")
        
        # Should be a dictionary
        self.assertIsInstance(DATASET_REGISTRY, dict)
        
        # Should contain known benchmarks
        expected_benchmarks = ["infinitebench", "longbench", "locomo"]
        for benchmark in expected_benchmarks:
            self.assertIn(benchmark, DATASET_REGISTRY)
            self.assertTrue(callable(DATASET_REGISTRY[benchmark]))
    
    def test_create_infinitebench_dataset(self):
        """Test factory creates InfiniteBenchDataset correctly"""
        if create_dataset_from_config is None:
            self.skipTest("create_dataset_from_config not implemented yet (Red phase)")
        if InfiniteBenchDataset is None:
            self.skipTest("InfiniteBenchDataset not available from previous phases")
        
        test_file = self._create_test_data_file("infinitebench")
        config = BenchmarkConfig(
            name="infinitebench",
            task_type="passkey",
            data_path=test_file,
            sample_limit=None,
            augmentation_config=None,
            enable_auto_truncation=False,
            truncation_strategy="right",
            preserve_ratio=0.8,
            llm_eval_config=None
        )
        
        dataset = create_dataset_from_config(config)
        
        # Should create correct type
        self.assertIsInstance(dataset, InfiniteBenchDataset)
        self.assertEqual(len(dataset), 1)
    
    def test_create_longbench_dataset(self):
        """Test factory creates LongBenchDataset correctly"""
        if create_dataset_from_config is None:
            self.skipTest("create_dataset_from_config not implemented yet (Red phase)")
        if LongBenchDataset is None:
            self.skipTest("LongBenchDataset not available from previous phases")
        
        test_file = self._create_test_data_file("longbench")
        config = BenchmarkConfig(
            name="longbench",
            task_type="narrativeqa",
            data_path=test_file,
            sample_limit=None,
            augmentation_config=None,
            enable_auto_truncation=False,
            truncation_strategy="right",
            preserve_ratio=0.8,
            llm_eval_config=None
        )
        
        dataset = create_dataset_from_config(config)
        
        # Should create correct type
        self.assertIsInstance(dataset, LongBenchDataset)
        self.assertEqual(len(dataset), 1)
    
    def test_create_locomo_dataset(self):
        """Test factory creates LoCoMoDataset correctly"""
        if create_dataset_from_config is None:
            self.skipTest("create_dataset_from_config not implemented yet (Red phase)")
        if LoCoMoDataset is None:
            self.skipTest("LoCoMoDataset not available from previous phases")
        
        test_file = self._create_test_data_file("locomo")
        config = BenchmarkConfig(
            name="locomo",
            task_type="conversational_qa",
            data_path=test_file,
            sample_limit=None,
            augmentation_config=None,
            enable_auto_truncation=False,
            truncation_strategy="right",
            preserve_ratio=0.8,
            llm_eval_config=None
        )
        
        dataset = create_dataset_from_config(config)
        
        # Should create correct type
        self.assertIsInstance(dataset, LoCoMoDataset)
        self.assertEqual(len(dataset), 1)
    
    def test_case_insensitive_benchmark_names(self):
        """Test factory handles case-insensitive benchmark names"""
        if create_dataset_from_config is None:
            self.skipTest("create_dataset_from_config not implemented yet (Red phase)")
        if InfiniteBenchDataset is None:
            self.skipTest("InfiniteBenchDataset not available from previous phases")
        
        test_file = self._create_test_data_file("infinitebench")
        
        # Test different cases
        test_cases = ["INFINITEBENCH", "InfiniteBench", "infiniteBench", "InFinItEbEnCh"]
        
        for name_variant in test_cases:
            config = BenchmarkConfig(
                name=name_variant,
                task_type="passkey", 
                data_path=test_file
            )
            
            dataset = create_dataset_from_config(config)
            self.assertIsInstance(dataset, InfiniteBenchDataset)
    
    def test_unknown_benchmark_raises_error(self):
        """Test that unknown benchmark names raise informative error"""
        if create_dataset_from_config is None:
            self.skipTest("create_dataset_from_config not implemented yet (Red phase)")
        
        test_file = self._create_test_data_file("infinitebench")  # Any valid file
        config = BenchmarkConfig(
            name="unknown_benchmark",
            task_type="some_task",
            data_path=test_file,
            sample_limit=None,
            augmentation_config=None,
            enable_auto_truncation=False,
            truncation_strategy="right",
            preserve_ratio=0.8,
            llm_eval_config=None
        )
        
        with self.assertRaises(ValueError) as context:
            create_dataset_from_config(config)
        
        # Should have informative error message
        error_msg = str(context.exception)
        self.assertIn("unknown_benchmark", error_msg.lower())
        self.assertIn("unknown", error_msg.lower())
    
    def test_factory_passes_additional_kwargs(self):
        """Test that factory passes additional keyword arguments to dataset constructor"""
        if create_dataset_from_config is None:
            self.skipTest("create_dataset_from_config not implemented yet (Red phase)")
        if InfiniteBenchDataset is None:
            self.skipTest("InfiniteBenchDataset not available from previous phases")
        
        test_file = self._create_test_data_file("infinitebench")
        config = BenchmarkConfig(
            name="infinitebench",
            task_type="passkey",
            data_path=test_file,
            sample_limit=None,
            augmentation_config=None,
            enable_auto_truncation=False,
            truncation_strategy="right",
            preserve_ratio=0.8,
            llm_eval_config=None
        )
        
        # Mock tokenizer for testing
        mock_tokenizer = type('MockTokenizer', (), {'encode': lambda x: [1, 2, 3]})()
        
        dataset = create_dataset_from_config(
            config,
            max_context_length=200,
            tokenizer=mock_tokenizer,
            truncation_strategy="left"
        )
        
        # Should pass kwargs to dataset
        self.assertEqual(dataset.max_context_length, 200)
        self.assertEqual(dataset.tokenizer, mock_tokenizer)
        self.assertEqual(dataset.truncation_strategy, "left")
    
    def test_register_new_dataset_class(self):
        """Test dynamic registration of new dataset classes"""
        if register_dataset_class is None:
            self.skipTest("register_dataset_class not implemented yet (Red phase)")
        if BaseBenchmarkDataset is None:
            self.skipTest("BaseBenchmarkDataset not available from previous phases")
        
        # Create a mock dataset class
        class CustomBenchmarkDataset(BaseBenchmarkDataset):
            def _load_and_parse_data(self):
                return []
            def evaluate_response(self, response, ground_truth, task_name):
                return 1.0
            def get_task_metrics(self, task_name):
                return ["accuracy"]
        
        # Register the new class
        register_dataset_class("custom_benchmark", CustomBenchmarkDataset)
        
        # Should be able to create it via factory
        test_file = self._create_test_data_file("infinitebench")  # Any valid file
        config = BenchmarkConfig(
            name="custom_benchmark",
            task_type="test",
            data_path=test_file,
            sample_limit=None,
            augmentation_config=None,
            enable_auto_truncation=False,
            truncation_strategy="right",
            preserve_ratio=0.8,
            llm_eval_config=None
        )
        
        dataset = create_dataset_from_config(config)
        self.assertIsInstance(dataset, CustomBenchmarkDataset)
    
    def test_get_available_datasets(self):
        """Test listing of available dataset types"""
        if get_available_datasets is None:
            self.skipTest("get_available_datasets not implemented yet (Red phase)")
        
        available = get_available_datasets()
        
        # Should be a list of strings
        self.assertIsInstance(available, list)
        self.assertTrue(all(isinstance(name, str) for name in available))
        
        # Should include known benchmarks
        expected_benchmarks = ["infinitebench", "longbench", "locomo"]
        for benchmark in expected_benchmarks:
            self.assertIn(benchmark, available)
    
    def test_factory_eliminates_if_else_chains(self):
        """Test that factory pattern eliminates if-else branching"""
        if create_dataset_from_config is None:
            self.skipTest("create_dataset_from_config not implemented yet (Red phase)")
        
        # This is more of a design test - the factory should use registry lookup
        # instead of if-else chains. We test this by ensuring different benchmarks
        # can be created without any branching logic visible to users.
        
        test_cases = [
            ("infinitebench", "passkey"),
            ("longbench", "narrativeqa"), 
            ("locomo", "conversational_qa")
        ]
        
        for benchmark_name, task_type in test_cases:
            test_file = self._create_test_data_file(benchmark_name)
            config = BenchmarkConfig(
                name=benchmark_name,
                task_type=task_type,
                data_path=test_file
            )
            
            # Should create dataset without any errors
            dataset = create_dataset_from_config(config)
            self.assertIsNotNone(dataset)
            
            # Should be proper subclass of base dataset
            if BaseBenchmarkDataset is not None:
                self.assertIsInstance(dataset, BaseBenchmarkDataset)


if __name__ == '__main__':
    unittest.main()