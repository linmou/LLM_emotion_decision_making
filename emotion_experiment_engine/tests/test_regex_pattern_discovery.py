#!/usr/bin/env python3
"""
Test file for regex pattern discovery functionality in BenchmarkConfig.
Tests the discover_datasets_by_pattern method with various regex patterns.

This test specifically covers the bug reported with patterns like '.*retrieval.*'
not working correctly due to regex.match() vs regex.search() issue.
"""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock
import os

from ..data_models import BenchmarkConfig


class TestRegexPatternDiscovery(unittest.TestCase):
    """Test regex pattern discovery for dataset files"""
    
    def setUp(self):
        """Set up test data with mock files"""
        self.test_files = [
            "longbench_narrativeqa.jsonl",
            "longbench_passage_retrieval_en.jsonl",
            "longbench_document_retrieval.jsonl", 
            "longbench_kv_retrieval.jsonl",
            "longbench_multifieldqa_en.jsonl",
            "longbench_qasper.jsonl",
            "longbench_hotpotqa.jsonl",
            "infinitebench_passkey.jsonl",
            "infinitebench_kv_retrieval.jsonl",
        ]
        
    @patch('glob.glob')
    def test_regex_pattern_discovery_contains_retrieval(self, mock_glob):
        """Test that .*retrieval.* pattern finds all files containing 'retrieval'"""
        # Mock file discovery to return our test files
        mock_files = [f"data/memory_benchmarks/{filename}" for filename in self.test_files]
        mock_glob.return_value = mock_files
        
        # Create BenchmarkConfig with regex pattern
        config = BenchmarkConfig(
            name="longbench",
            task_type=".*retrieval.*",  # Should match files containing 'retrieval'
            data_path=None,
            base_data_dir="data/memory_benchmarks",
            sample_limit=None,
            augmentation_config=None,
            enable_auto_truncation=False,
            truncation_strategy="right",
            preserve_ratio=0.8,
            llm_eval_config=None
        )
        
        # Discover datasets matching the pattern
        discovered = config.discover_datasets_by_pattern()
        
        # Expected: All task types containing 'retrieval' 
        expected_retrieval_tasks = [
            "passage_retrieval_en",    # longbench_passage_retrieval_en.jsonl
            "document_retrieval",      # longbench_document_retrieval.jsonl  
            "kv_retrieval",           # longbench_kv_retrieval.jsonl
        ]
        
        print(f"Discovered tasks: {discovered}")
        print(f"Expected retrieval tasks: {expected_retrieval_tasks}")
        
        # Verify all expected retrieval tasks are found
        for expected_task in expected_retrieval_tasks:
            self.assertIn(expected_task, discovered, 
                         f"Pattern '.*retrieval.*' should match task '{expected_task}'")
            
        # Verify non-retrieval tasks are NOT found  
        non_retrieval_tasks = ["narrativeqa", "multifieldqa_en", "qasper", "hotpotqa"]
        for non_retrieval_task in non_retrieval_tasks:
            self.assertNotIn(non_retrieval_task, discovered,
                           f"Pattern '.*retrieval.*' should NOT match task '{non_retrieval_task}'")

    @patch('glob.glob')            
    def test_regex_pattern_discovery_contains_retrieval_no_leading_wildcard(self, mock_glob):
        """Test that 'retrieval' pattern (no leading .*) finds files containing 'retrieval'
        
        This test exposes the bug: match() vs search() issue.
        Pattern 'retrieval' should find task types containing 'retrieval' anywhere.
        """
        mock_files = [f"data/memory_benchmarks/{filename}" for filename in self.test_files]
        mock_glob.return_value = mock_files
        
        config = BenchmarkConfig(
            name="longbench",
            task_type="retrieval",  # No leading .* - should still match files containing 'retrieval'
            data_path=None,
            base_data_dir="data/memory_benchmarks",
            sample_limit=None,
            augmentation_config=None,
            enable_auto_truncation=False,
            truncation_strategy="right",
            preserve_ratio=0.8,
            llm_eval_config=None
        )
        
        discovered = config.discover_datasets_by_pattern()
        
        # Expected: All task types containing 'retrieval' anywhere
        expected_retrieval_tasks = [
            "passage_retrieval_en",    
            "document_retrieval",       
            "kv_retrieval",           
        ]
        
        print(f"Pattern 'retrieval' discovered: {discovered}")
        print(f"Expected retrieval tasks: {expected_retrieval_tasks}")
        
        # This will FAIL with current implementation using match()
        # because 'retrieval' pattern doesn't match from start of 'passage_retrieval_en'
        for expected_task in expected_retrieval_tasks:
            self.assertIn(expected_task, discovered, 
                         f"Pattern 'retrieval' should match task '{expected_task}' containing 'retrieval'")
    
    @patch('glob.glob')            
    def test_regex_pattern_discovery_starts_with_pass(self, mock_glob):
        """Test that pass.* pattern finds files starting with 'pass'"""
        mock_files = [f"data/memory_benchmarks/{filename}" for filename in self.test_files]
        mock_glob.return_value = mock_files
        
        config = BenchmarkConfig(
            name="infinitebench",
            task_type="pass.*",  # Should match files starting with 'pass'
            data_path=None,
            base_data_dir="data/memory_benchmarks",
            sample_limit=None,
            augmentation_config=None,
            enable_auto_truncation=False,
            truncation_strategy="right", 
            preserve_ratio=0.8,
            llm_eval_config=None
        )
        
        discovered = config.discover_datasets_by_pattern()
        
        expected_pass_tasks = ["passkey"]
        self.assertEqual(discovered, expected_pass_tasks)
        
    @patch('glob.glob')
    def test_regex_pattern_discovery_ends_with_qa(self, mock_glob):
        """Test that .*qa$ pattern finds files ending with 'qa'"""
        mock_files = [f"data/memory_benchmarks/{filename}" for filename in self.test_files]
        mock_glob.return_value = mock_files
        
        config = BenchmarkConfig(
            name="longbench",
            task_type=".*qa$",  # Should match files ending with 'qa'
            data_path=None,
            base_data_dir="data/memory_benchmarks",
            sample_limit=None,
            augmentation_config=None,
            enable_auto_truncation=False,
            truncation_strategy="right",
            preserve_ratio=0.8,
            llm_eval_config=None
        )
        
        discovered = config.discover_datasets_by_pattern()
        
        expected_qa_tasks = ["narrativeqa", "hotpotqa"]
        self.assertEqual(sorted(discovered), sorted(expected_qa_tasks))

    @patch('glob.glob') 
    def test_regex_pattern_discovery_literal_match(self, mock_glob):
        """Test literal task type (no regex) still works"""
        mock_files = [f"data/memory_benchmarks/{filename}" for filename in self.test_files]
        mock_glob.return_value = mock_files
        
        config = BenchmarkConfig(
            name="longbench",
            task_type="narrativeqa",  # Literal match
            data_path=None,
            base_data_dir="data/memory_benchmarks",
            sample_limit=None,
            augmentation_config=None,
            enable_auto_truncation=False,
            truncation_strategy="right",
            preserve_ratio=0.8,
            llm_eval_config=None
        )
        
        discovered = config.discover_datasets_by_pattern()
        
        self.assertEqual(discovered, ["narrativeqa"])

    @patch('glob.glob')
    def test_regex_pattern_discovery_no_matches(self, mock_glob):
        """Test pattern that matches no files returns empty list"""
        mock_files = [f"data/memory_benchmarks/{filename}" for filename in self.test_files]
        mock_glob.return_value = mock_files
        
        config = BenchmarkConfig(
            name="longbench",
            task_type="nonexistent.*pattern",
            data_path=None,
            base_data_dir="data/memory_benchmarks",
            sample_limit=None,
            augmentation_config=None,
            enable_auto_truncation=False,
            truncation_strategy="right",
            preserve_ratio=0.8,
            llm_eval_config=None
        )
        
        discovered = config.discover_datasets_by_pattern()
        
        self.assertEqual(discovered, [])

    @patch('glob.glob')
    def test_regex_pattern_discovery_invalid_regex(self, mock_glob):
        """Test that invalid regex patterns raise ValueError"""
        mock_files = [f"data/memory_benchmarks/{filename}" for filename in self.test_files]
        mock_glob.return_value = mock_files
        
        config = BenchmarkConfig(
            name="longbench", 
            task_type="[invalid regex",  # Invalid regex - missing closing bracket
            data_path=None,
            base_data_dir="data/memory_benchmarks",
            sample_limit=None,
            augmentation_config=None,
            enable_auto_truncation=False,
            truncation_strategy="right",
            preserve_ratio=0.8,
            llm_eval_config=None
        )
        
        with self.assertRaises(ValueError) as context:
            config.discover_datasets_by_pattern("data/memory_benchmarks")
            
        self.assertIn("Invalid regex pattern", str(context.exception))

    @patch('glob.glob')        
    def test_regex_pattern_all_files(self, mock_glob):
        """Test .* pattern matches all available files"""
        mock_files = [f"data/memory_benchmarks/{filename}" for filename in self.test_files]
        mock_glob.return_value = mock_files
        
        config = BenchmarkConfig(
            name="longbench",
            task_type=".*",  # Match everything
            data_path=None,
            base_data_dir="data/memory_benchmarks",
            sample_limit=None,
            augmentation_config=None,
            enable_auto_truncation=False,
            truncation_strategy="right",
            preserve_ratio=0.8,
            llm_eval_config=None
        )
        
        discovered = config.discover_datasets_by_pattern()
        
        expected_longbench_tasks = [
            "narrativeqa",
            "passage_retrieval_en", 
            "document_retrieval",
            "kv_retrieval",
            "multifieldqa_en",
            "qasper",
            "hotpotqa"
        ]
        
        self.assertEqual(sorted(discovered), sorted(expected_longbench_tasks))

    def test_regex_pattern_discovery_real_file_system(self):
        """Test with actual temporary files on filesystem"""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create test files
            test_files = [
                "longbench_narrativeqa.jsonl",
                "longbench_passage_retrieval_en.jsonl", 
                "longbench_document_retrieval.jsonl",
                "longbench_kv_retrieval.jsonl"
            ]
            
            # Create the files
            for filename in test_files:
                (Path(temp_dir) / filename).touch()
                
            config = BenchmarkConfig(
                name="longbench",
                task_type=".*retrieval.*",
                data_path=None,
                base_data_dir=temp_dir,
                sample_limit=None,
                augmentation_config=None,
                enable_auto_truncation=False,
                truncation_strategy="right", 
                preserve_ratio=0.8,
                llm_eval_config=None
            )
            
            discovered = config.discover_datasets_by_pattern(temp_dir)
            
            expected_retrieval_tasks = [
                "passage_retrieval_en",
                "document_retrieval", 
                "kv_retrieval"
            ]
            
            self.assertEqual(sorted(discovered), sorted(expected_retrieval_tasks))


if __name__ == "__main__":
    # Run with verbose output to see detailed failure information
    unittest.main(verbosity=2)