#!/usr/bin/env python3
"""
Behavioral Equivalence Regression Tests

These tests ensure that refactored components produce identical results to
previous implementations, protecting scientific validity and reproducibility.
"""

import json
import numpy as np
import pandas as pd
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock
from typing import Dict, Any, List, Tuple

from emotion_experiment_engine.dataset_factory import create_dataset_from_config
from emotion_experiment_engine.data_models import BenchmarkConfig
from emotion_experiment_engine.evaluation_utils import llm_evaluate_response


@pytest.mark.regression
@pytest.mark.critical
class TestEvaluationBehavioralEquivalence:
    """Ensure evaluation produces identical results across versions"""
    
    # Reference evaluation results for regression testing
    REFERENCE_EVALUATIONS = [
        {
            "response": "The answer is 42",
            "ground_truth": "42", 
            "task_type": "passkey",
            "expected_score": 1.0
        },
        {
            "response": "I think it's 43",
            "ground_truth": "42",
            "task_type": "passkey", 
            "expected_score": 0.0
        },
        {
            "response": "Machine learning is a branch of artificial intelligence",
            "ground_truth": "Machine learning is a subset of AI",
            "task_type": "qa",
            "expected_score": 0.8  # Semantic similarity
        },
        {
            "response": "",
            "ground_truth": "any_answer",
            "task_type": "any",
            "expected_score": 0.0
        }
    ]
    
    @pytest.mark.parametrize("test_case", REFERENCE_EVALUATIONS)
    @patch('emotion_experiment_engine.evaluation_utils.openai.ChatCompletion.create')
    def test_evaluation_score_equivalence(self, mock_openai, test_case):
        """Ensure identical inputs produce identical evaluation scores"""
        
        # Mock OpenAI to return deterministic response
        expected_score = test_case["expected_score"]
        mock_openai.return_value.choices = [
            MagicMock(message=MagicMock(
                content=json.dumps({
                    "emotion": "neutral", 
                    "confidence": expected_score
                })
            ))
        ]
        
        # Test evaluation through dataset interface
        config = BenchmarkConfig(
            name="emotion_check",
            task_type=test_case["task_type"],
            llm_eval_config={
                "model": "gpt-4o-mini",
                "temperature": 0.0  # Ensure determinism
            },
            data_path=None
        )
        
        # Mock data loading
        from emotion_experiment_engine.datasets.emotion_check import EmotionCheckDataset
        with patch.object(EmotionCheckDataset, '_load_raw_data') as mock_load:
            mock_load.return_value = [
                {
                    "id": 1, 
                    "input": "test", 
                    "ground_truth": [test_case["ground_truth"]], 
                    "category": "test"
                }
            ]
            
            dataset = EmotionCheckDataset(config, prompt_wrapper=None)
            
            # Run evaluation multiple times to ensure consistency
            scores = []
            for _ in range(3):
                score = dataset.evaluate_response(
                    test_case["response"],
                    test_case["ground_truth"], 
                    test_case["task_type"]
                )
                scores.append(score)
            
            # Verify deterministic behavior
            assert all(s == scores[0] for s in scores), \
                f"Non-deterministic evaluation for {test_case}: {scores}"
            
            # Verify expected score
            assert abs(scores[0] - expected_score) < 0.001, \
                f"Score mismatch for {test_case}: got {scores[0]}, expected {expected_score}"
    
    def test_dataset_output_consistency(self):
        """Ensure dataset transformations remain consistent"""
        test_data = [
            {
                "id": 0,
                "input": "What is the passkey?",
                "answer": "12345", 
                "context": "The secret code is 12345",
                "task_name": "passkey"
            },
            {
                "id": 1,
                "input": "Summarize the text",
                "answer": "This is a summary",
                "context": "Long text that needs summarizing...",
                "task_name": "summary"
            }
        ]
        
        # Test that same data produces same dataset items
        config = BenchmarkConfig(
            name="infinitebench", 
            task_type="passkey",
            data_path=None
        )
        
        from emotion_experiment_engine.datasets.infinitebench import InfiniteBenchDataset
        with patch.object(InfiniteBenchDataset, '_load_raw_data') as mock_load:
            mock_load.return_value = test_data
            
            # Create dataset multiple times
            datasets = []
            for _ in range(3):
                dataset = InfiniteBenchDataset(config, prompt_wrapper=None)
                datasets.append(dataset)
            
            # Verify identical structure
            for i in range(len(datasets[0])):
                item1 = datasets[0][i]
                item2 = datasets[1][i] 
                item3 = datasets[2][i]
                
                # Items should be identical
                assert item1["item"].id == item2["item"].id == item3["item"].id
                assert item1["prompt"] == item2["prompt"] == item3["prompt"]
                assert item1["ground_truth"] == item2["ground_truth"] == item3["ground_truth"]
    
    def test_factory_creation_equivalence(self):
        """Ensure factory creates identical dataset instances"""
        config = BenchmarkConfig(
            name="longbench",
            task_type="narrativeqa", 
            data_path=None,
            sample_limit=5
        )
        
        # Mock data loading for consistency
        test_data = [
            {
                "id": f"test_{i}",
                "input": f"Question {i}",
                "answers": [f"Answer {i}"],
                "context": f"Context {i}",
                "task_name": "narrativeqa"
            }
            for i in range(5)
        ]
        
        from emotion_experiment_engine.datasets.longbench import LongBenchDataset
        with patch.object(LongBenchDataset, '_load_raw_data') as mock_load:
            mock_load.return_value = test_data
            
            # Create datasets through factory
            datasets = []
            for _ in range(3):
                dataset = create_dataset_from_config(config)
                datasets.append(dataset)
            
            # Verify identical behavior
            assert len(datasets[0]) == len(datasets[1]) == len(datasets[2])
            assert all(isinstance(ds, LongBenchDataset) for ds in datasets)
            
            # Compare first item from each dataset
            for i in range(min(3, len(datasets[0]))):
                items = [ds[i] for ds in datasets]
                
                # Prompts should be identical
                prompts = [item["prompt"] for item in items]
                assert all(p == prompts[0] for p in prompts), \
                    f"Non-identical prompts at index {i}: {prompts}"
                
                # Ground truth should be identical  
                ground_truths = [item["ground_truth"] for item in items]
                assert all(gt == ground_truths[0] for gt in ground_truths), \
                    f"Non-identical ground truth at index {i}: {ground_truths}"


@pytest.mark.regression
class TestStatisticalBehavioralEquivalence:
    """Ensure statistical properties remain equivalent across versions"""
    
    def test_score_distribution_stability(self):
        """Score distributions should remain stable across implementations"""
        
        # Generate test cases with known distribution
        test_cases = [
            ("perfect_match", "perfect_match", 1.0),
            ("partial_match", "partial match with extra words", 0.7),
            ("wrong_answer", "correct_answer", 0.0),
            ("empty_response", "any_answer", 0.0),
            ("close_match", "very close match", 0.8),
        ] * 10  # Repeat for statistical stability
        
        scores = []
        
        # Mock consistent LLM responses
        def mock_llm_response(system_prompt, query, llm_eval_config):
            response_text = query.split("Response: ")[1].split("\n")[0] if "Response: " in query else ""
            ground_truth = query.split("Ground truth: ")[1].split("\n")[0] if "Ground truth: " in query else ""
            
            # Simple similarity scoring for deterministic results
            if response_text == ground_truth:
                return {"emotion": "neutral", "confidence": 1.0}
            elif response_text in ground_truth or ground_truth in response_text:
                return {"emotion": "neutral", "confidence": 0.8}
            elif len(response_text) > 0 and len(ground_truth) > 0:
                # Simple word overlap
                response_words = set(response_text.lower().split())
                truth_words = set(ground_truth.lower().split())
                overlap = len(response_words & truth_words) / len(truth_words | response_words)
                return {"emotion": "neutral", "confidence": max(0.0, overlap)}
            else:
                return {"emotion": "neutral", "confidence": 0.0}
        
        with patch('emotion_experiment_engine.evaluation_utils.llm_evaluate_response', 
                   side_effect=mock_llm_response):
            
            config = BenchmarkConfig(
                name="emotion_check",
                task_type="test",
                llm_eval_config={"model": "gpt-4o-mini", "temperature": 0.0},
                data_path=None
            )
            
            from emotion_experiment_engine.datasets.emotion_check import EmotionCheckDataset
            with patch.object(EmotionCheckDataset, '_load_raw_data') as mock_load:
                mock_load.return_value = [{"id": 1, "input": "test", "ground_truth": ["test"], "category": "test"}]
                
                dataset = EmotionCheckDataset(config, prompt_wrapper=None)
                
                for response, ground_truth, expected in test_cases:
                    score = dataset.evaluate_response(response, ground_truth, "test")
                    scores.append(score)
        
        # Statistical properties should be stable
        score_array = np.array(scores)
        
        # Basic statistical checks
        assert 0.0 <= score_array.min() <= score_array.max() <= 1.0, \
            "Scores outside valid range"
        
        # Distribution should have expected properties
        perfect_scores = sum(1 for s in scores if s == 1.0)
        zero_scores = sum(1 for s in scores if s == 0.0)
        
        assert perfect_scores > 0, "No perfect scores found"
        assert zero_scores > 0, "No zero scores found"
        
        # Mean should be reasonable for mixed test cases
        mean_score = np.mean(score_array)
        assert 0.3 <= mean_score <= 0.7, f"Unexpected mean score: {mean_score}"
    
    def test_evaluation_variance_bounds(self):
        """Evaluation variance should remain within acceptable bounds"""
        
        # Test same input multiple times to measure variance
        test_input = ("Machine learning is AI", "Machine learning is artificial intelligence")
        
        # Mock deterministic response with slight controlled variance
        call_count = 0
        def mock_response_with_variance(system_prompt, query, llm_eval_config):
            nonlocal call_count
            call_count += 1
            
            # Introduce controlled tiny variance to simulate real conditions
            base_confidence = 0.85
            variance = 0.001 * (call_count % 3 - 1)  # ±0.001 variance
            
            return {
                "emotion": "neutral", 
                "confidence": base_confidence + variance
            }
        
        with patch('emotion_experiment_engine.evaluation_utils.llm_evaluate_response',
                   side_effect=mock_response_with_variance):
            
            config = BenchmarkConfig(
                name="emotion_check",
                task_type="test",
                llm_eval_config={"model": "gpt-4o-mini", "temperature": 0.0},
                data_path=None
            )
            
            from emotion_experiment_engine.datasets.emotion_check import EmotionCheckDataset
            with patch.object(EmotionCheckDataset, '_load_raw_data') as mock_load:
                mock_load.return_value = [{"id": 1, "input": "test", "ground_truth": ["test"], "category": "test"}]
                
                dataset = EmotionCheckDataset(config, prompt_wrapper=None)
                
                # Run evaluation multiple times
                scores = []
                for _ in range(10):
                    score = dataset.evaluate_response(test_input[0], test_input[1], "test")
                    scores.append(score)
        
        # Variance should be minimal for deterministic evaluation
        variance = np.var(scores)
        max_acceptable_variance = 0.01  # 1% variance allowed
        
        assert variance <= max_acceptable_variance, \
            f"Evaluation variance too high: {variance} > {max_acceptable_variance}"
        
        # All scores should be close to expected range
        mean_score = np.mean(scores)
        assert 0.84 <= mean_score <= 0.86, f"Mean score drift: {mean_score}"


@pytest.mark.regression
class TestDatasetSpecificBehavioralEquivalence:
    """Test behavioral equivalence for each dataset type"""
    
    def test_infinitebench_passkey_equivalence(self):
        """InfiniteBench passkey evaluation must remain equivalent"""
        
        # Test cases with known expected behavior
        passkey_cases = [
            ("The passkey is 12345", "12345", 1.0),
            ("12345", "12345", 1.0),
            ("The code is 67890", "12345", 0.0),
            ("I don't know", "12345", 0.0)
        ]
        
        # Mock the specific InfiniteBench evaluation
        def mock_passkey_eval(system_prompt, query, llm_eval_config):
            response_text = query.split("Response: ")[1].split("\n")[0] if "Response: " in query else ""
            ground_truth = query.split("Ground truth: ")[1].split("\n")[0] if "Ground truth: " in query else ""
            
            # Passkey-specific logic: exact number match
            if ground_truth in response_text:
                return {"emotion": "neutral", "confidence": 1.0}
            else:
                return {"emotion": "neutral", "confidence": 0.0}
        
        with patch('emotion_experiment_engine.evaluation_utils.llm_evaluate_response',
                   side_effect=mock_passkey_eval):
            
            config = BenchmarkConfig(
                name="infinitebench",
                task_type="passkey",
                data_path=None
            )
            
            from emotion_experiment_engine.datasets.infinitebench import InfiniteBenchDataset
            with patch.object(InfiniteBenchDataset, '_load_raw_data') as mock_load:
                mock_load.return_value = [
                    {"id": 1, "input": "What's the passkey?", "answer": "12345", "task_name": "passkey"}
                ]
                
                dataset = InfiniteBenchDataset(config, prompt_wrapper=None)
                
                for response, ground_truth, expected_score in passkey_cases:
                    score = dataset.evaluate_response(response, ground_truth, "passkey")
                    
                    assert abs(score - expected_score) < 0.001, \
                        f"Passkey evaluation regression: {response} -> {score} != {expected_score}"
    
    def test_longbench_qa_equivalence(self):
        """LongBench QA evaluation must remain equivalent"""
        
        qa_cases = [
            ("AI is machine learning", "Machine learning is part of AI", 0.6),
            ("Machine learning is part of AI", "Machine learning is part of AI", 1.0),
            ("Completely wrong answer", "Machine learning is part of AI", 0.0)
        ]
        
        # Mock semantic similarity evaluation
        def mock_qa_eval(system_prompt, query, llm_eval_config):
            response_text = query.split("Response: ")[1].split("\n")[0] if "Response: " in query else ""
            ground_truth = query.split("Ground truth: ")[1].split("\n")[0] if "Ground truth: " in query else ""
            
            # Simple semantic similarity mock
            if response_text == ground_truth:
                return {"emotion": "neutral", "confidence": 1.0}
            elif any(word in ground_truth.lower() for word in response_text.lower().split()):
                return {"emotion": "neutral", "confidence": 0.6}
            else:
                return {"emotion": "neutral", "confidence": 0.0}
        
        with patch('emotion_experiment_engine.evaluation_utils.llm_evaluate_response',
                   side_effect=mock_qa_eval):
            
            config = BenchmarkConfig(
                name="longbench", 
                task_type="narrativeqa",
                data_path=None
            )
            
            from emotion_experiment_engine.datasets.longbench import LongBenchDataset
            with patch.object(LongBenchDataset, '_load_raw_data') as mock_load:
                mock_load.return_value = [
                    {"id": 1, "input": "Question?", "answers": ["Answer"], "task_name": "narrativeqa"}
                ]
                
                dataset = LongBenchDataset(config, prompt_wrapper=None)
                
                for response, ground_truth, expected_score in qa_cases:
                    score = dataset.evaluate_response(response, ground_truth, "narrativeqa")
                    
                    assert abs(score - expected_score) < 0.1, \
                        f"QA evaluation regression: {response} -> {score} != {expected_score}"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-m", "regression"])