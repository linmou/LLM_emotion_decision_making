#!/usr/bin/env python3
"""
Test file for: InfiniteBenchDataset specialized implementation (TDD Red phase)
Purpose: Test InfiniteBench-specific data loading and evaluation

This test suite ensures InfiniteBench dataset handles its specific data format
and routing to the correct evaluators from the comprehensive evaluation_utils.
"""

import unittest
import tempfile
import json
from pathlib import Path
from typing import Any, Dict, List

import pytest

# These imports will initially fail (Red phase) - that's expected!
try:
    from emotion_experiment_engine.datasets.infinitebench import InfiniteBenchDataset
except ImportError:
    InfiniteBenchDataset = None

from emotion_experiment_engine.data_models import BenchmarkConfig, BenchmarkItem


class TestInfiniteBenchDataset(unittest.TestCase):
    """Test InfiniteBench specialized dataset implementation"""
    
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
    
    def _create_infinitebench_test_data(self, task_type: str = "passkey") -> Path:
        """Create temporary InfiniteBench test data"""
        
        if task_type == "passkey":
            test_data = [
                {
                    "id": 0,
                    "input": "What is the passkey? The passkey is hidden in this context: random text 12345 more text",
                    "answer": "12345",
                    "task_name": "passkey"
                },
                {
                    "id": 1, 
                    "input": "Find the passkey. Here's some text with the key: xyz 67890 abc",
                    "answer": "67890",
                    "task_name": "passkey"
                }
            ]
        elif task_type == "kv_retrieval":
            test_data = [
                {
                    "id": 0,
                    "input": "apple banana cherry date elderberry",
                    "answer": "apple",
                    "task_name": "kv_retrieval"
                }
            ]
        elif task_type == "math_find":
            test_data = [
                {
                    "id": 0,
                    "input": "Calculate the result: 2 + 2 = ?",
                    "answer": "4",
                    "task_name": "math_find"
                }
            ]
        elif task_type == "longbook_qa_eng":
            test_data = [
                {
                    "id": 0,
                    "input": "Based on the book, what is the main theme?",
                    "answer": ["The main theme is friendship", "Friendship"],
                    "task_name": "longbook_qa_eng"
                }
            ]
        else:
            raise ValueError(f"Unknown task type: {task_type}")
        
        # Create temporary JSONL file
        temp_file = Path(tempfile.mktemp(suffix='.jsonl'))
        with open(temp_file, 'w') as f:
            for item in test_data:
                f.write(json.dumps(item) + '\n')
        
        self.temp_files.append(temp_file)
        return temp_file
    
    def test_infinitebench_dataset_creation(self):
        """Test InfiniteBenchDataset can be created and loads data"""
        if InfiniteBenchDataset is None:
            self.skipTest("InfiniteBenchDataset not implemented yet (Red phase)")
        
        test_file = self._create_infinitebench_test_data("passkey")
        config = BenchmarkConfig(
            name="infinitebench",
            task_type="passkey",
            data_path=test_file
        )
        
        dataset = InfiniteBenchDataset(config)
        
        # Should be a proper dataset
        self.assertEqual(len(dataset), 2)
        self.assertIsInstance(dataset.items[0], BenchmarkItem)
        
        # Should have proper item data
        item = dataset.items[0]
        self.assertEqual(item.id, 0)
        self.assertIn("passkey", item.input_text)
        self.assertEqual(item.ground_truth, "12345")
    
    def test_infinitebench_task_evaluator_mapping(self):
        """Test that InfiniteBench has correct task-to-evaluator mapping"""
        if InfiniteBenchDataset is None:
            self.skipTest("InfiniteBenchDataset not implemented yet (Red phase)")
        
        test_file = self._create_infinitebench_test_data("passkey")
        config = BenchmarkConfig(name="infinitebench", task_type="passkey", data_path=test_file)
        dataset = InfiniteBenchDataset(config)
        
        # Should have static evaluator mapping
        self.assertTrue(hasattr(dataset, 'TASK_EVALUATORS'))
        self.assertIsInstance(dataset.TASK_EVALUATORS, dict)
        
        # Should include key InfiniteBench tasks
        expected_tasks = ["passkey", "kv_retrieval", "code_run", "math_find", "longbook_qa_eng"]
        for task in expected_tasks:
            self.assertIn(task, dataset.TASK_EVALUATORS)
            self.assertIsInstance(dataset.TASK_EVALUATORS[task], str)  # Function name
    
    def test_infinitebench_passkey_evaluation(self):
        """Test passkey task evaluation works correctly"""
        if InfiniteBenchDataset is None:
            self.skipTest("InfiniteBenchDataset not implemented yet (Red phase)")
        
        test_file = self._create_infinitebench_test_data("passkey")
        config = BenchmarkConfig(name="infinitebench", task_type="passkey", data_path=test_file)
        dataset = InfiniteBenchDataset(config)
        
        # Test correct passkey extraction
        score = dataset.evaluate_response("The passkey is 12345", "12345", "passkey")
        self.assertEqual(score, 1.0)
        
        # Test incorrect answer
        score = dataset.evaluate_response("The passkey is 99999", "12345", "passkey")
        self.assertEqual(score, 0.0)
        
        # Test passkey extraction from natural language
        score = dataset.evaluate_response("Based on the text, I found the passkey: 12345", "12345", "passkey")
        self.assertEqual(score, 1.0)
    
    def test_infinitebench_kv_retrieval_evaluation(self):
        """Test key-value retrieval evaluation"""
        if InfiniteBenchDataset is None:
            self.skipTest("InfiniteBenchDataset not implemented yet (Red phase)")
        
        test_file = self._create_infinitebench_test_data("kv_retrieval")
        config = BenchmarkConfig(name="infinitebench", task_type="kv_retrieval", data_path=test_file)
        dataset = InfiniteBenchDataset(config)
        
        # Test correct retrieval
        score = dataset.evaluate_response("apple banana cherry", "apple", "kv_retrieval")
        self.assertEqual(score, 1.0)
        
        # Test incorrect retrieval
        score = dataset.evaluate_response("banana cherry date", "apple", "kv_retrieval") 
        self.assertEqual(score, 0.0)
    
    def test_infinitebench_math_evaluation(self):
        """Test math task evaluation"""
        if InfiniteBenchDataset is None:
            self.skipTest("InfiniteBenchDataset not implemented yet (Red phase)")
        
        test_file = self._create_infinitebench_test_data("math_find")
        config = BenchmarkConfig(name="infinitebench", task_type="math_find", data_path=test_file)
        dataset = InfiniteBenchDataset(config)
        
        # Test correct math result
        score = dataset.evaluate_response("The answer is 4", "4", "math_find")
        self.assertEqual(score, 1.0)
        
        # Test incorrect result
        score = dataset.evaluate_response("The answer is 5", "4", "math_find")
        self.assertEqual(score, 0.0)
    
    def test_infinitebench_qa_f1_evaluation(self):
        """Test QA task uses F1 scoring"""
        if InfiniteBenchDataset is None:
            self.skipTest("InfiniteBenchDataset not implemented yet (Red phase)")
        
        test_file = self._create_infinitebench_test_data("longbook_qa_eng")
        config = BenchmarkConfig(name="infinitebench", task_type="longbook_qa_eng", data_path=test_file)
        dataset = InfiniteBenchDataset(config)
        
        # Test partial match gets partial score
        score = dataset.evaluate_response("The theme is friendship", ["The main theme is friendship"], "longbook_qa_eng")
        self.assertGreater(score, 0.0)
        self.assertLess(score, 1.0)  # Should be partial match
        
        # Test exact match gets full score
        score = dataset.evaluate_response("The main theme is friendship", ["The main theme is friendship"], "longbook_qa_eng")
        self.assertEqual(score, 1.0)
    
    def test_infinitebench_task_metrics(self):
        """Test task metrics reporting"""
        if InfiniteBenchDataset is None:
            self.skipTest("InfiniteBenchDataset not implemented yet (Red phase)")
        
        test_file = self._create_infinitebench_test_data("passkey")
        config = BenchmarkConfig(name="infinitebench", task_type="passkey", data_path=test_file)
        dataset = InfiniteBenchDataset(config)
        
        # Test exact match tasks
        metrics = dataset.get_task_metrics("passkey")
        self.assertIn("exact_match", metrics)
        
        metrics = dataset.get_task_metrics("kv_retrieval")
        self.assertIn("exact_match", metrics)
        
        # Test F1 tasks  
        metrics = dataset.get_task_metrics("longbook_qa_eng")
        self.assertIn("f1_score", metrics)
        self.assertIn("precision", metrics)
        self.assertIn("recall", metrics)
    
    def test_infinitebench_unknown_task_fallback(self):
        """Test unknown task falls back to generic evaluator"""
        if InfiniteBenchDataset is None:
            self.skipTest("InfiniteBenchDataset not implemented yet (Red phase)")
        
        test_file = self._create_infinitebench_test_data("passkey")
        config = BenchmarkConfig(name="infinitebench", task_type="passkey", data_path=test_file)
        dataset = InfiniteBenchDataset(config)
        
        # Unknown task should use generic evaluator
        score = dataset.evaluate_response("test response", "test response", "unknown_task")
        self.assertEqual(score, 1.0)  # Should fallback to exact match
        
        score = dataset.evaluate_response("different", "test response", "unknown_task")
        self.assertEqual(score, 0.0)  # Should fallback to exact match
    
    def test_infinitebench_data_parsing(self):
        """Test InfiniteBench data format parsing"""
        if InfiniteBenchDataset is None:
            self.skipTest("InfiniteBenchDataset not implemented yet (Red phase)")
        
        # Test with context field
        test_data = [
            {
                "id": 0,
                "input": "Question text",
                "context": "Context text",
                "answer": "Answer text",
                "task_name": "passkey"
            }
        ]
        
        temp_file = Path(tempfile.mktemp(suffix='.jsonl'))
        with open(temp_file, 'w') as f:
            for item in test_data:
                f.write(json.dumps(item) + '\n')
        self.temp_files.append(temp_file)
        
        config = BenchmarkConfig(name="infinitebench", task_type="passkey", data_path=temp_file)
        dataset = InfiniteBenchDataset(config)
        
        item = dataset.items[0]
        self.assertEqual(item.context, "Context text")
        self.assertEqual(item.input_text, "Question text")
        self.assertEqual(item.ground_truth, "Answer text")
    
    def test_infinitebench_sample_limit(self):
        """Test sample limiting works"""
        if InfiniteBenchDataset is None:
            self.skipTest("InfiniteBenchDataset not implemented yet (Red phase)")
        
        test_file = self._create_infinitebench_test_data("passkey")  # Has 2 items
        config = BenchmarkConfig(
            name="infinitebench", 
            task_type="passkey", 
            data_path=test_file,
            sample_limit=1
        )
        
        dataset = InfiniteBenchDataset(config)
        
        # Should only have 1 item due to limit
        self.assertEqual(len(dataset), 1)


if __name__ == '__main__':
    unittest.main()