#!/usr/bin/env python3
"""
Test file for: LoCoMoDataset specialized implementation (TDD Red phase)
Purpose: Test LoCoMo-specific data loading, conversation formatting, and F1 evaluation

This test suite ensures LoCoMo dataset handles its specific conversational data format
and uses the specialized F1 scoring with stemming.
"""

import unittest
import tempfile
import json
from pathlib import Path
from typing import Any, Dict, List

import pytest

# These imports will initially fail (Red phase) - that's expected!
try:
    from emotion_experiment_engine.datasets.locomo import LoCoMoDataset
except ImportError:
    LoCoMoDataset = None

from emotion_experiment_engine.data_models import BenchmarkConfig, BenchmarkItem


class TestLoCoMoDataset(unittest.TestCase):
    """Test LoCoMo specialized dataset implementation"""
    
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
    
    def _create_locomo_test_data(self) -> Path:
        """Create temporary LoCoMo test data"""
        
        test_data = [
            {
                "sample_id": "sample_1",
                "conversation": {
                    "session_1": [
                        {"speaker": "Alice", "text": "Hello, my budget is $30000"},
                        {"speaker": "Bob", "text": "That's a good budget for a car"}
                    ],
                    "session_1_date_time": "2024-01-01 10:00:00",
                    "session_2": [
                        {"speaker": "Alice", "text": "I'm interested in electric cars"},
                        {"speaker": "Bob", "text": "We have great electric options"}
                    ],
                    "session_2_date_time": "2024-01-02 14:00:00"
                },
                "qa": [
                    {
                        "question": "What is Alice's budget?",
                        "answer": "$30000",
                        "category": "budget",
                        "evidence": ["session_1"]
                    },
                    {
                        "question": "What type of car is Alice interested in?",
                        "answer": "electric cars",
                        "category": "preference",
                        "evidence": ["session_2"]
                    }
                ]
            },
            {
                "sample_id": "sample_2",
                "conversation": {
                    "session_1": [
                        {"speaker": "Customer", "text": "I work at Google and need reliable internet"},
                        {"speaker": "Agent", "text": "We can provide enterprise-grade service"}
                    ],
                    "session_1_date_time": "2024-01-05 09:00:00"
                },
                "qa": [
                    {
                        "question": "Where does the customer work?",
                        "answer": "Google",
                        "category": "employment",
                        "evidence": ["session_1"]
                    }
                ]
            }
        ]
        
        # Create temporary JSON file
        temp_file = Path(tempfile.mktemp(suffix='.json'))
        with open(temp_file, 'w') as f:
            json.dump(test_data, f)
        
        self.temp_files.append(temp_file)
        return temp_file
    
    def test_locomo_dataset_creation(self):
        """Test LoCoMoDataset can be created and loads data"""
        if LoCoMoDataset is None:
            self.skipTest("LoCoMoDataset not implemented yet (Red phase)")
        
        test_file = self._create_locomo_test_data()
        config = BenchmarkConfig(
            name="locomo",
            task_type="conversational_qa",
            data_path=test_file
        )
        
        dataset = LoCoMoDataset(config)
        
        # Should have 3 QA pairs total (2 from sample_1, 1 from sample_2)
        self.assertEqual(len(dataset), 3)
        self.assertIsInstance(dataset.items[0], BenchmarkItem)
        
        # Should have proper item data
        item = dataset.items[0]
        self.assertIn("Alice's budget", item.input_text)
        self.assertEqual(item.ground_truth, "$30000")
    
    def test_locomo_conversation_formatting(self):
        """Test LoCoMo conversation formatting"""
        if LoCoMoDataset is None:
            self.skipTest("LoCoMoDataset not implemented yet (Red phase)")
        
        test_file = self._create_locomo_test_data()
        config = BenchmarkConfig(name="locomo", task_type="conversational_qa", data_path=test_file)
        dataset = LoCoMoDataset(config)
        
        # Check that context is properly formatted conversation
        item = dataset.items[0]
        context = item.context
        
        # Should contain session headers
        self.assertIn("SESSION_1", context)
        self.assertIn("SESSION_2", context)
        
        # Should contain date/time info
        self.assertIn("2024-01-01 10:00:00", context)
        self.assertIn("2024-01-02 14:00:00", context)
        
        # Should contain speaker names and text
        self.assertIn("Alice:", context)
        self.assertIn("Bob:", context)
        self.assertIn("budget is $30000", context)
        self.assertIn("electric cars", context)
    
    def test_locomo_f1_scoring(self):
        """Test LoCoMo F1 scoring with stemming"""
        if LoCoMoDataset is None:
            self.skipTest("LoCoMoDataset not implemented yet (Red phase)")
        
        test_file = self._create_locomo_test_data()
        config = BenchmarkConfig(name="locomo", task_type="conversational_qa", data_path=test_file)
        dataset = LoCoMoDataset(config)
        
        # Test exact match
        score = dataset.evaluate_response("$30000", "$30000", "qa")
        self.assertEqual(score, 1.0)
        
        # Test case insensitive
        score = dataset.evaluate_response("GOOGLE", "Google", "qa")
        self.assertEqual(score, 1.0)
        
        # Test stemming (ed/ing suffix removal)
        score = dataset.evaluate_response("worked", "working", "qa")
        self.assertEqual(score, 1.0)  # Both become "work" after stemming
        
        # Test actual working stemming case
        score = dataset.evaluate_response("cars", "car", "qa")
        self.assertEqual(score, 1.0)  # Should match: car -> car
        
        # Test no match
        score = dataset.evaluate_response("completely different", "$30000", "qa")
        self.assertEqual(score, 0.0)
    
    def test_locomo_f1_detailed(self):
        """Test LoCoMo F1 calculation details"""
        if LoCoMoDataset is None:
            self.skipTest("LoCoMoDataset not implemented yet (Red phase)")
        
        test_file = self._create_locomo_test_data()
        config = BenchmarkConfig(name="locomo", task_type="conversational_qa", data_path=test_file)
        dataset = LoCoMoDataset(config)
        
        # Test with overlap
        response = "Alice works at Google company"
        ground_truth = "Google"
        
        score = dataset.evaluate_response(response, ground_truth, "qa")
        self.assertGreater(score, 0.0)
        self.assertLessEqual(score, 1.0)
        
        # Test stemming effect (s suffix removal)
        response = "cars"
        ground_truth = "car"
        
        score = dataset.evaluate_response(response, ground_truth, "qa")
        self.assertEqual(score, 1.0)  # Should match after stemming: cars->car, car->car
    
    def test_locomo_task_metrics(self):
        """Test LoCoMo task metrics"""
        if LoCoMoDataset is None:
            self.skipTest("LoCoMoDataset not implemented yet (Red phase)")
        
        test_file = self._create_locomo_test_data()
        config = BenchmarkConfig(name="locomo", task_type="conversational_qa", data_path=test_file)
        dataset = LoCoMoDataset(config)
        
        # LoCoMo uses F1 scoring
        metrics = dataset.get_task_metrics("qa")
        self.assertIn("f1_score", metrics)
        self.assertIn("precision", metrics)
        self.assertIn("recall", metrics)
    
    def test_locomo_data_parsing_metadata(self):
        """Test LoCoMo metadata parsing"""
        if LoCoMoDataset is None:
            self.skipTest("LoCoMoDataset not implemented yet (Red phase)")
        
        test_file = self._create_locomo_test_data()
        config = BenchmarkConfig(name="locomo", task_type="conversational_qa", data_path=test_file)
        dataset = LoCoMoDataset(config)
        
        # Check metadata is preserved
        item = dataset.items[0]
        metadata = item.metadata
        
        self.assertEqual(metadata["sample_id"], "sample_1")
        self.assertEqual(metadata["category"], "budget")
        self.assertIn("evidence", metadata)
        self.assertEqual(metadata["evidence"], ["session_1"])
    
    def test_locomo_multiple_sessions_formatting(self):
        """Test conversation with multiple sessions is formatted correctly"""
        if LoCoMoDataset is None:
            self.skipTest("LoCoMoDataset not implemented yet (Red phase)")
        
        test_file = self._create_locomo_test_data()
        config = BenchmarkConfig(name="locomo", task_type="conversational_qa", data_path=test_file)
        dataset = LoCoMoDataset(config)
        
        # Find an item from sample_1 which has 2 sessions
        sample_1_items = [item for item in dataset.items if "sample_1" in str(item.metadata.get("sample_id", ""))]
        self.assertGreater(len(sample_1_items), 0)
        
        item = sample_1_items[0]
        context = item.context
        
        # Should have both sessions formatted
        session_count = context.count("SESSION_")
        self.assertEqual(session_count, 2)
        
        # Should be in chronological order (session_1 before session_2)
        session_1_pos = context.find("SESSION_1")
        session_2_pos = context.find("SESSION_2")
        self.assertLess(session_1_pos, session_2_pos)
    
    def test_locomo_sample_limit(self):
        """Test sample limiting works with QA expansion"""
        if LoCoMoDataset is None:
            self.skipTest("LoCoMoDataset not implemented yet (Red phase)")
        
        test_file = self._create_locomo_test_data()  # Has 3 QA pairs total
        config = BenchmarkConfig(
            name="locomo", 
            task_type="conversational_qa", 
            data_path=test_file,
            sample_limit=2
        )
        
        dataset = LoCoMoDataset(config)
        
        # Should only have 2 items due to limit
        self.assertEqual(len(dataset), 2)
    
    def test_locomo_evaluate_all_task_types(self):
        """Test LoCoMo evaluation works for all tasks (they all use F1)"""
        if LoCoMoDataset is None:
            self.skipTest("LoCoMoDataset not implemented yet (Red phase)")
        
        test_file = self._create_locomo_test_data()
        config = BenchmarkConfig(name="locomo", task_type="conversational_qa", data_path=test_file)
        dataset = LoCoMoDataset(config)
        
        # All tasks should use the same F1 evaluation
        tasks = ["qa", "conversational_qa", "memory_qa", "any_task"]
        
        for task in tasks:
            score = dataset.evaluate_response("Google", "Google", task)
            self.assertEqual(score, 1.0)
            
            score = dataset.evaluate_response("different", "Google", task)
            self.assertEqual(score, 0.0)


if __name__ == '__main__':
    unittest.main()