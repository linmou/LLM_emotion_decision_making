#!/usr/bin/env python3
"""
Test file for: LongBenchDataset specialized implementation (TDD Red phase)
Purpose: Test LongBench-specific data loading and evaluation routing

This test suite ensures LongBench dataset handles its specific data format
and routes to the correct LongBench evaluators.
"""

import unittest
import tempfile
import json
from pathlib import Path
from typing import Any, Dict, List

import pytest

# These imports will initially fail (Red phase) - that's expected!
try:
    from emotion_experiment_engine.datasets.longbench import LongBenchDataset
except ImportError:
    LongBenchDataset = None

from emotion_experiment_engine.data_models import BenchmarkConfig, BenchmarkItem


class TestLongBenchDataset(unittest.TestCase):
    """Test LongBench specialized dataset implementation"""
    
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
    
    def _create_longbench_test_data(self, task_type: str = "narrativeqa") -> Path:
        """Create temporary LongBench test data"""
        
        if task_type == "narrativeqa":
            test_data = [
                {
                    "id": "narrativeqa_0",
                    "input": "Based on the story, what is the main character's motivation?",
                    "answers": ["The main character wants to find their missing brother"],
                    "task_name": "narrativeqa"
                },
                {
                    "id": "narrativeqa_1",
                    "input": "What happens at the end of the story?",
                    "answers": ["They reunite with their family", "The protagonist returns home"],
                    "task_name": "narrativeqa"
                }
            ]
        elif task_type == "gov_report":
            test_data = [
                {
                    "id": "gov_report_0", 
                    "input": "Summarize the economic findings.",
                    "answers": ["Economic growth was steady with inflation controlled"],
                    "task_name": "gov_report"
                }
            ]
        elif task_type == "trec":
            test_data = [
                {
                    "id": "trec_0",
                    "input": "What category does this question belong to?",
                    "answers": ["politics"],
                    "task_name": "trec"
                }
            ]
        elif task_type == "passage_count":
            test_data = [
                {
                    "id": "count_0",
                    "input": "How many times is the word mentioned?",
                    "answers": ["5"],
                    "task_name": "passage_count"
                }
            ]
        elif task_type == "lcc":
            test_data = [
                {
                    "id": "lcc_0", 
                    "input": "Code similarity task",
                    "answers": ["def function(): return True"],
                    "task_name": "lcc"
                }
            ]
        else:
            raise ValueError(f"Unknown task type: {task_type}")
        
        # Create temporary JSON file (LongBench uses .json, not .jsonl)
        temp_file = Path(tempfile.mktemp(suffix='.json'))
        with open(temp_file, 'w') as f:
            json.dump(test_data, f)
        
        self.temp_files.append(temp_file)
        return temp_file
    
    def test_longbench_dataset_creation(self):
        """Test LongBenchDataset can be created and loads data"""
        if LongBenchDataset is None:
            self.skipTest("LongBenchDataset not implemented yet (Red phase)")
        
        test_file = self._create_longbench_test_data("narrativeqa")
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
        
        dataset = LongBenchDataset(config)
        
        # Should be a proper dataset
        self.assertEqual(len(dataset), 2)
        self.assertIsInstance(dataset.items[0], BenchmarkItem)
        
        # Should have proper item data
        item = dataset.items[0]
        self.assertEqual(item.id, "narrativeqa_0")
        self.assertIn("motivation", item.input_text)
        self.assertIn("missing brother", str(item.ground_truth))
    
    def test_longbench_evaluator_mapping(self):
        """Test that LongBench has correct task-to-evaluator mapping"""
        if LongBenchDataset is None:
            self.skipTest("LongBenchDataset not implemented yet (Red phase)")
        
        test_file = self._create_longbench_test_data("narrativeqa")
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
        dataset = LongBenchDataset(config)
        
        # Should have static evaluator mapping
        self.assertTrue(hasattr(dataset, 'METRIC_EVALUATORS'))
        self.assertIsInstance(dataset.METRIC_EVALUATORS, dict)
        
        # Should include key LongBench tasks
        expected_tasks = ["narrativeqa", "qasper", "gov_report", "trec", "passage_count", "lcc"]
        for task in expected_tasks:
            self.assertIn(task, dataset.METRIC_EVALUATORS)
            self.assertIsInstance(dataset.METRIC_EVALUATORS[task], str)  # Function name
    
    def test_longbench_qa_f1_evaluation(self):
        """Test QA tasks use F1 scoring"""
        if LongBenchDataset is None:
            self.skipTest("LongBenchDataset not implemented yet (Red phase)")
        
        test_file = self._create_longbench_test_data("narrativeqa")
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
        dataset = LongBenchDataset(config)
        
        # Test exact match gets full score
        score = dataset.evaluate_response(
            "The main character wants to find their missing brother",
            ["The main character wants to find their missing brother"],
            "narrativeqa"
        )
        self.assertEqual(score, 1.0)
        
        # Test partial match gets partial score
        score = dataset.evaluate_response(
            "wants to find missing brother", 
            ["The main character wants to find their missing brother"],
            "narrativeqa"
        )
        self.assertGreater(score, 0.0)
        self.assertLess(score, 1.0)
    
    def test_longbench_rouge_evaluation(self):
        """Test summarization tasks use ROUGE scoring"""
        if LongBenchDataset is None:
            self.skipTest("LongBenchDataset not implemented yet (Red phase)")
        
        test_file = self._create_longbench_test_data("gov_report")
        config = BenchmarkConfig(
            name="longbench", 
            task_type="gov_report", 
            data_path=test_file,
            sample_limit=None,
            augmentation_config=None,
            enable_auto_truncation=False,
            truncation_strategy="right",
            preserve_ratio=0.8,
            llm_eval_config=None
        )
        dataset = LongBenchDataset(config)
        
        # Test ROUGE scoring (partial match should get some score)
        score = dataset.evaluate_response(
            "Economic growth steady inflation controlled",
            ["Economic growth was steady with inflation controlled"],
            "gov_report"
        )
        self.assertGreater(score, 0.0)
        self.assertLessEqual(score, 1.0)
    
    def test_longbench_classification_evaluation(self):
        """Test classification tasks use substring matching"""
        if LongBenchDataset is None:
            self.skipTest("LongBenchDataset not implemented yet (Red phase)")
        
        test_file = self._create_longbench_test_data("trec")
        config = BenchmarkConfig(
            name="longbench", 
            task_type="trec", 
            data_path=test_file,
            sample_limit=None,
            augmentation_config=None,
            enable_auto_truncation=False,
            truncation_strategy="right",
            preserve_ratio=0.8,
            llm_eval_config=None
        )
        dataset = LongBenchDataset(config)
        
        # Test correct classification
        score = dataset.evaluate_response("This is about politics", "politics", "trec")
        self.assertEqual(score, 1.0)
        
        # Test incorrect classification  
        score = dataset.evaluate_response("This is about science", "politics", "trec")
        self.assertEqual(score, 0.0)
    
    def test_longbench_count_evaluation(self):
        """Test counting tasks extract numbers"""
        if LongBenchDataset is None:
            self.skipTest("LongBenchDataset not implemented yet (Red phase)")
        
        test_file = self._create_longbench_test_data("passage_count")
        config = BenchmarkConfig(
            name="longbench", 
            task_type="passage_count", 
            data_path=test_file,
            sample_limit=None,
            augmentation_config=None,
            enable_auto_truncation=False,
            truncation_strategy="right",
            preserve_ratio=0.8,
            llm_eval_config=None
        )
        dataset = LongBenchDataset(config)
        
        # Test correct count
        score = dataset.evaluate_response("The word appears 5 times", "5", "passage_count")
        self.assertEqual(score, 1.0)
        
        # Test incorrect count
        score = dataset.evaluate_response("The word appears 3 times", "5", "passage_count")
        self.assertEqual(score, 0.0)
    
    def test_longbench_code_similarity_evaluation(self):
        """Test code tasks use code similarity"""
        if LongBenchDataset is None:
            self.skipTest("LongBenchDataset not implemented yet (Red phase)")
        
        test_file = self._create_longbench_test_data("lcc")
        config = BenchmarkConfig(
            name="longbench", 
            task_type="lcc", 
            data_path=test_file,
            sample_limit=None,
            augmentation_config=None,
            enable_auto_truncation=False,
            truncation_strategy="right",
            preserve_ratio=0.8,
            llm_eval_config=None
        )
        dataset = LongBenchDataset(config)
        
        # Test similar code gets some score
        score = dataset.evaluate_response(
            "def function(): return True",
            "def function(): return True", 
            "lcc"
        )
        self.assertGreater(score, 0.0)
        self.assertLessEqual(score, 1.0)
    
    def test_longbench_task_metrics(self):
        """Test task metrics reporting"""
        if LongBenchDataset is None:
            self.skipTest("LongBenchDataset not implemented yet (Red phase)")
        
        test_file = self._create_longbench_test_data("narrativeqa")
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
        dataset = LongBenchDataset(config)
        
        # Test QA tasks
        metrics = dataset.get_task_metrics("narrativeqa")
        self.assertIn("f1_score", metrics)
        self.assertIn("precision", metrics)
        self.assertIn("recall", metrics)
        
        # Test summarization tasks
        metrics = dataset.get_task_metrics("gov_report")
        self.assertIn("rouge_score", metrics)
        
        # Test classification tasks
        metrics = dataset.get_task_metrics("trec")
        self.assertIn("classification_accuracy", metrics)
        
        # Test count tasks
        metrics = dataset.get_task_metrics("passage_count")
        self.assertIn("count_accuracy", metrics)
    
    def test_longbench_data_parsing(self):
        """Test LongBench data format parsing"""
        if LongBenchDataset is None:
            self.skipTest("LongBenchDataset not implemented yet (Red phase)")
        
        test_data = [
            {
                "id": "test_id",
                "input": "Test question",
                "context": "Test context", 
                "answers": ["Test answer 1", "Test answer 2"],
                "task_name": "narrativeqa"
            }
        ]
        
        temp_file = Path(tempfile.mktemp(suffix='.json'))
        with open(temp_file, 'w') as f:
            json.dump(test_data, f)
        self.temp_files.append(temp_file)
        
        config = BenchmarkConfig(
            name="longbench", 
            task_type="narrativeqa", 
            data_path=temp_file,
            sample_limit=None,
            augmentation_config=None,
            enable_auto_truncation=False,
            truncation_strategy="right",
            preserve_ratio=0.8,
            llm_eval_config=None
        )
        dataset = LongBenchDataset(config)
        
        item = dataset.items[0]
        self.assertEqual(item.id, "test_id")
        self.assertEqual(item.input_text, "Test question")
        self.assertEqual(item.context, "Test context")
        self.assertEqual(item.ground_truth, "Test answer 1")  # Should take first answer
    
    def test_longbench_unknown_task_fallback(self):
        """Test unknown task falls back to generic evaluator"""
        if LongBenchDataset is None:
            self.skipTest("LongBenchDataset not implemented yet (Red phase)")
        
        test_file = self._create_longbench_test_data("narrativeqa")
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
        dataset = LongBenchDataset(config)
        
        # Unknown task should use generic evaluator
        score = dataset.evaluate_response("test response", "test response", "unknown_task")
        self.assertEqual(score, 1.0)  # Should fallback to exact match


if __name__ == '__main__':
    unittest.main()