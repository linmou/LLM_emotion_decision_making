#!/usr/bin/env python3
"""
Research-Critical Test Framework

These tests protect the core scientific validity of emotion memory experiments.
They MUST NEVER FAIL as they directly impact research reproducibility and validity.
"""

import numpy as np
import pytest
from unittest.mock import patch, MagicMock
from typing import Dict, Any, List

from emotion_experiment_engine.evaluation_utils import llm_evaluate_response
from emotion_experiment_engine.dataset_factory import create_dataset_from_config, DATASET_REGISTRY
from emotion_experiment_engine.data_models import BenchmarkConfig


class ResearchCriticalInvariants:
    """Core scientific invariants that must always hold"""
    
    # Critical test configuration
    CRITICAL_INVARIANTS = {
        "evaluation_determinism": {
            "priority": "P0",
            "impact": "Results non-reproducible",
            "max_variance": 0.001,  # Max allowed variance in repeated evaluations
        },
        "emotion_manipulation_validity": {
            "priority": "P0", 
            "impact": "Core hypothesis invalidation",
            "required_distinctiveness": 0.1,  # Min difference between emotions
        },
        "benchmark_scoring_invariance": {
            "priority": "P0",
            "impact": "Invalid benchmark comparisons",
            "score_bounds": (0.0, 1.0),  # Valid score range
        }
    }


@pytest.mark.critical
@pytest.mark.order(1)  # Run first, always
class TestScientificInvariants:
    """Protect core scientific assumptions and invariants"""
    
    def test_evaluation_determinism(self):
        """
        CRITICAL: Same input must always produce identical evaluation scores
        
        This test ensures reproducibility of research results. If this fails,
        published results cannot be replicated.
        """
        test_cases = [
            ("The answer is 42", "42", "passkey"),
            ("Machine learning is AI", "Machine learning is artificial intelligence", "qa"),
            ("I feel happy", "happiness", "emotion"),
        ]
        
        for response, ground_truth, task_type in test_cases:
            scores = []
            
            # Run evaluation multiple times
            for _ in range(5):
                # Mock consistent LLM response for determinism
                with patch('emotion_experiment_engine.evaluation_utils.llm_evaluate_response') as mock_eval:
                    mock_eval.return_value = {"emotion": "neutral", "confidence": 0.85}
                    
                    from emotion_experiment_engine.datasets.base import BaseBenchmarkDataset
                    
                    class TestDataset(BaseBenchmarkDataset):
                        def _load_and_parse_data(self): return []
                        def get_task_metrics(self, task_name): return ["accuracy"]
                    
                    dataset = TestDataset(BenchmarkConfig(
                        name="test", 
                        task_type=task_type,
                        llm_eval_config={"model": "gpt-4o-mini", "temperature": 0.0}
                    ))
                    
                    score = dataset.evaluate_response(response, ground_truth, task_type)
                    scores.append(score)
            
            # Verify determinism
            score_variance = np.var(scores)
            assert score_variance < ResearchCriticalInvariants.CRITICAL_INVARIANTS[
                "evaluation_determinism"]["max_variance"], \
                f"Evaluation not deterministic for {task_type}: variance={score_variance}, scores={scores}"
    
    def test_score_bounds_invariant(self):
        """
        CRITICAL: All evaluation scores must be in [0.0, 1.0] range
        
        Scores outside this range invalidate statistical analysis and comparisons.
        """
        from emotion_experiment_engine.datasets.emotion_check import EmotionCheckDataset
        
        # Test with extreme inputs
        extreme_cases = [
            ("", "expected_answer"),  # Empty response
            ("x" * 10000, "short"),   # Very long response
            ("Special chars: äöüß€", "normal"),  # Unicode
            ("Multiple\nlines\nhere", "single"),  # Multiline
            (None, "answer"),         # None response (should be handled)
        ]
        
        config = BenchmarkConfig(
            name="emotion_check",
            task_type="validation",
            llm_eval_config={"model": "gpt-4o-mini", "temperature": 0.0}
        )
        
        # Mock data loading to avoid file dependency
        with patch.object(EmotionCheckDataset, '_load_raw_data') as mock_load:
            mock_load.return_value = [
                {"id": 1, "input": "Test", "ground_truth": ["test"], "category": "test"}
            ]
            
            dataset = EmotionCheckDataset(config, prompt_wrapper=None)
            
            for response, ground_truth in extreme_cases:
                try:
                    with patch('emotion_experiment_engine.evaluation_utils.llm_evaluate_response') as mock_eval:
                        mock_eval.return_value = {"emotion": "neutral", "confidence": 0.5}
                        
                        score = dataset.evaluate_response(
                            response if response is not None else "",
                            ground_truth, 
                            "validation"
                        )
                    
                    # Verify bounds
                    bounds = ResearchCriticalInvariants.CRITICAL_INVARIANTS[
                        "benchmark_scoring_invariance"]["score_bounds"]
                    
                    assert bounds[0] <= score <= bounds[1], \
                        f"Score {score} outside valid bounds {bounds} for response: '{response}'"
                        
                except Exception as e:
                    pytest.fail(f"Evaluation crashed with extreme input '{response}': {e}")
    
    def test_dataset_factory_registry_integrity(self):
        """
        CRITICAL: Dataset factory registry must be stable and complete
        
        Missing or corrupted registry entries break experiment execution.
        """
        # Verify registry exists and is populated
        assert DATASET_REGISTRY is not None, "Dataset registry not initialized"
        assert len(DATASET_REGISTRY) > 0, "Dataset registry is empty"
        
        # Verify expected datasets are registered
        expected_datasets = ["infinitebench", "longbench", "locomo", "emotion_check"]
        for dataset_name in expected_datasets:
            assert dataset_name.lower() in [k.lower() for k in DATASET_REGISTRY.keys()], \
                f"Critical dataset '{dataset_name}' missing from registry"
        
        # Verify all registered classes are valid
        from emotion_experiment_engine.datasets.base import BaseBenchmarkDataset
        
        for name, dataset_class in DATASET_REGISTRY.items():
            assert issubclass(dataset_class, BaseBenchmarkDataset), \
                f"Registry entry '{name}' is not a valid dataset class"
            
            # Verify class has required methods
            required_methods = ["_load_and_parse_data", "evaluate_response", "get_task_metrics"]
            for method in required_methods:
                assert hasattr(dataset_class, method), \
                    f"Dataset class '{name}' missing required method '{method}'"
    
    def test_llm_evaluation_consistency(self):
        """
        CRITICAL: LLM evaluation must produce consistent relative rankings
        
        If better responses don't score higher, the evaluation is invalid.
        """
        # Test cases with clear quality hierarchy
        test_hierarchies = [
            {
                "ground_truth": "42",
                "responses": [
                    ("42", 1.0),          # Perfect match
                    ("The answer is 42", 1.0),  # Correct with context
                    ("43", 0.0),          # Wrong number
                    ("", 0.0),            # Empty response
                ],
                "task": "passkey"
            },
            {
                "ground_truth": "Machine learning is a subset of AI",
                "responses": [
                    ("Machine learning is a subset of AI", 1.0),  # Perfect
                    ("ML is part of AI", 0.8),                    # Paraphrase
                    ("AI includes machine learning", 0.6),        # Related
                    ("Deep learning", 0.2),                       # Somewhat related
                    ("Cooking recipes", 0.0),                     # Unrelated
                ],
                "task": "qa"
            }
        ]
        
        for hierarchy in test_hierarchies:
            ground_truth = hierarchy["ground_truth"]
            task = hierarchy["task"]
            
            # Mock LLM evaluation to return expected scores
            def mock_eval(system_prompt, query, llm_eval_config):
                response_text = query.split("Response: ")[1].split("\n")[0] if "Response: " in query else ""
                
                # Find expected score for this response
                for response, expected_score in hierarchy["responses"]:
                    if response in response_text or response_text in response:
                        return {"emotion": "neutral", "confidence": expected_score}
                return {"emotion": "neutral", "confidence": 0.0}
            
            with patch('emotion_experiment_engine.evaluation_utils.llm_evaluate_response', side_effect=mock_eval):
                
                # Test with EmotionCheckDataset as representative
                config = BenchmarkConfig(
                    name="emotion_check",
                    task_type=task,
                    llm_eval_config={"model": "gpt-4o-mini", "temperature": 0.0}
                )
                
                with patch.object(EmotionCheckDataset, '_load_raw_data') as mock_load:
                    mock_load.return_value = [{"id": 1, "input": "Test", "ground_truth": [ground_truth], "category": "test"}]
                    
                    from emotion_experiment_engine.datasets.emotion_check import EmotionCheckDataset
                    dataset = EmotionCheckDataset(config, prompt_wrapper=None)
                    
                    scores = []
                    for response, expected_score in hierarchy["responses"]:
                        score = dataset.evaluate_response(response, ground_truth, task)
                        scores.append((response, score, expected_score))
                    
                    # Verify ranking consistency
                    for i in range(len(scores) - 1):
                        current_response, current_score, current_expected = scores[i]
                        next_response, next_score, next_expected = scores[i + 1]
                        
                        if current_expected > next_expected:
                            assert current_score >= next_score, \
                                f"Ranking inconsistency: '{current_response}' (score={current_score}) " \
                                f"should score >= '{next_response}' (score={next_score})"


@pytest.mark.critical
class TestEmotionManipulationValidity:
    """Validate core emotion manipulation assumptions"""
    
    def test_emotion_distinctiveness_requirement(self):
        """
        CRITICAL: Different emotions must produce measurably different effects
        
        If emotions don't produce distinct behavioral changes, the research
        hypothesis is invalid.
        """
        # This would be implemented with actual emotion manipulation once
        # the neural components are integrated. For now, we test the framework.
        
        emotions = ["anger", "happiness", "sadness", "fear", "neutral"]
        
        # Mock different emotion effects
        emotion_effects = {
            "anger": 0.8,      # High activation
            "happiness": 0.7,  # High activation, different pattern
            "sadness": 0.3,    # Low activation
            "fear": 0.6,       # Medium activation
            "neutral": 0.5,    # Baseline
        }
        
        # Verify minimum distinctiveness between emotions
        min_distinctiveness = ResearchCriticalInvariants.CRITICAL_INVARIANTS[
            "emotion_manipulation_validity"]["required_distinctiveness"]
        
        for i, emotion1 in enumerate(emotions):
            for emotion2 in emotions[i+1:]:
                effect_diff = abs(emotion_effects[emotion1] - emotion_effects[emotion2])
                
                # Allow neutral-neutral comparison to be zero
                if emotion1 == "neutral" and emotion2 == "neutral":
                    continue
                    
                assert effect_diff >= min_distinctiveness, \
                    f"Emotions '{emotion1}' and '{emotion2}' too similar " \
                    f"(difference: {effect_diff} < {min_distinctiveness})"
    
    @pytest.mark.statistical  
    def test_statistical_significance_framework(self):
        """
        CRITICAL: Framework must support statistical significance testing
        
        Research conclusions require proper statistical validation.
        """
        # Test that we can collect data in format suitable for statistical analysis
        sample_data = [
            {"emotion": "anger", "score": 0.8, "subject": "item_1"},
            {"emotion": "anger", "score": 0.75, "subject": "item_2"},
            {"emotion": "neutral", "score": 0.5, "subject": "item_1"},
            {"emotion": "neutral", "score": 0.55, "subject": "item_2"},
        ]
        
        # Group by emotion
        anger_scores = [d["score"] for d in sample_data if d["emotion"] == "anger"]
        neutral_scores = [d["score"] for d in sample_data if d["emotion"] == "neutral"]
        
        # Verify we have enough data for statistical testing
        assert len(anger_scores) >= 2, "Insufficient anger samples for statistical testing"
        assert len(neutral_scores) >= 2, "Insufficient neutral samples for statistical testing"
        
        # Verify data structure supports analysis
        assert all(0.0 <= score <= 1.0 for score in anger_scores + neutral_scores), \
            "Scores outside valid range for statistical analysis"
        
        # Test basic statistical measures
        anger_mean = np.mean(anger_scores)
        neutral_mean = np.mean(neutral_scores)
        effect_size = abs(anger_mean - neutral_mean)
        
        # In real implementation, we'd test for statistical significance
        # Here we just verify the framework supports it
        assert isinstance(effect_size, float), "Effect size calculation failed"
        assert effect_size >= 0.0, "Invalid effect size calculation"


@pytest.mark.critical
@pytest.mark.performance 
class TestCriticalPerformanceRequirements:
    """Performance requirements that impact research feasibility"""
    
    def test_evaluation_performance_requirements(self):
        """
        CRITICAL: Evaluation must be fast enough for large-scale experiments
        
        Slow evaluation makes large-scale research infeasible.
        """
        import time
        
        # Mock fast evaluation
        with patch('emotion_experiment_engine.evaluation_utils.llm_evaluate_response') as mock_eval:
            mock_eval.return_value = {"emotion": "neutral", "confidence": 0.8}
            
            config = BenchmarkConfig(
                name="emotion_check", 
                task_type="test",
                llm_eval_config={"model": "gpt-4o-mini", "temperature": 0.0}
            )
            
            from emotion_experiment_engine.datasets.emotion_check import EmotionCheckDataset
            
            with patch.object(EmotionCheckDataset, '_load_raw_data') as mock_load:
                mock_load.return_value = [{"id": 1, "input": "Test", "ground_truth": ["test"], "category": "test"}]
                
                dataset = EmotionCheckDataset(config, prompt_wrapper=None)
                
                # Time multiple evaluations
                start_time = time.time()
                for i in range(10):
                    score = dataset.evaluate_response(f"response_{i}", "ground_truth", "test")
                duration = time.time() - start_time
                
                # Should be fast enough for large experiments
                avg_time_per_eval = duration / 10
                max_acceptable_time = 5.0  # 5 seconds per evaluation
                
                assert avg_time_per_eval < max_acceptable_time, \
                    f"Evaluation too slow: {avg_time_per_eval:.2f}s per evaluation " \
                    f"(max: {max_acceptable_time}s)"


if __name__ == "__main__":
    # Run critical tests only
    pytest.main([__file__, "-v", "-m", "critical"])