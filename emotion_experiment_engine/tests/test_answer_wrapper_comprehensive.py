"""
Comprehensive test suite for AnswerWrapper classes and their integration.

This file combines all AnswerWrapper tests into a single comprehensive suite:
- Basic functionality tests for all wrapper classes
- Factory function tests  
- Dataset integration tests
- End-to-end experiment integration tests
- Regression tests to ensure backward compatibility

Follows the TDD pattern established during development.
"""

import unittest
from functools import partial
from typing import Any, Dict
from unittest.mock import Mock, patch

from ..answer_wrapper import (
    AnswerWrapper,
    EmotionAnswerWrapper,
    IdentityAnswerWrapper,
    get_answer_wrapper
)
from ..data_models import BenchmarkItem, BenchmarkConfig
from ..datasets.base import BaseBenchmarkDataset


# =============================================================================
# BASIC FUNCTIONALITY TESTS
# =============================================================================

class TestAnswerWrapperBase(unittest.TestCase):
    """Test the base AnswerWrapper functionality"""
    
    def test_answer_wrapper_creation(self):
        """Test AnswerWrapper base class can be instantiated"""
        wrapper = AnswerWrapper()
        self.assertIsInstance(wrapper, AnswerWrapper)
    
    def test_answer_wrapper_call_interface(self):
        """Test AnswerWrapper follows callable interface like PromptWrapper"""
        wrapper = AnswerWrapper()
        
        # Should be callable with ground truth and context
        result = wrapper("test_answer", emotion="anger")
        
        # Base class should pass through unchanged
        self.assertEqual(result, "test_answer")
    
    def test_transform_answer_method(self):
        """Test transform_answer method signature"""
        wrapper = AnswerWrapper()
        
        # Should have transform_answer method
        self.assertTrue(hasattr(wrapper, 'transform_answer'))
        self.assertTrue(callable(wrapper.transform_answer))
        
        # Should pass through unchanged by default
        result = wrapper.transform_answer("test_answer", emotion="anger")
        self.assertEqual(result, "test_answer")


class TestIdentityAnswerWrapper(unittest.TestCase):
    """Test IdentityAnswerWrapper - default pass-through behavior"""
    
    def test_identity_wrapper_creation(self):
        """Test IdentityAnswerWrapper instantiation"""
        wrapper = IdentityAnswerWrapper()
        self.assertIsInstance(wrapper, AnswerWrapper)
    
    def test_identity_wrapper_passes_through_unchanged(self):
        """Test IdentityAnswerWrapper doesn't modify ground truth"""
        wrapper = IdentityAnswerWrapper()
        
        # Test with different data types
        test_cases = [
            "string_answer",
            ["list", "answer"],
            {"dict": "answer"},
            42,
            None
        ]
        
        for test_input in test_cases:
            with self.subTest(input=test_input):
                result = wrapper(test_input, emotion="anger", context="test")
                self.assertEqual(result, test_input)


class TestEmotionAnswerWrapper(unittest.TestCase):
    """Test EmotionAnswerWrapper - emotion-based ground truth adaptation"""
    
    def test_emotion_wrapper_creation(self):
        """Test EmotionAnswerWrapper instantiation"""
        wrapper = EmotionAnswerWrapper()
        self.assertIsInstance(wrapper, AnswerWrapper)
    
    def test_emotion_check_task_returns_emotion_as_ground_truth(self):
        """Test that EmotionAnswerWrapper returns emotion for emotion_check tasks"""
        wrapper = EmotionAnswerWrapper()
        
        # For emotion_check tasks, should return the emotion regardless of original ground truth
        result = wrapper(
            ground_truth="original_answer",
            emotion="anger",
            benchmark_name="emotion_check"
        )
        
        self.assertEqual(result, "anger")
    
    def test_emotion_check_with_different_emotions(self):
        """Test EmotionAnswerWrapper with various emotions"""
        wrapper = EmotionAnswerWrapper()
        
        emotions = ["anger", "happiness", "sadness", "fear", "disgust", "surprise"]
        
        for emotion in emotions:
            with self.subTest(emotion=emotion):
                result = wrapper(
                    ground_truth="ignored",
                    emotion=emotion,
                    benchmark_name="emotion_check"
                )
                self.assertEqual(result, emotion)
    
    def test_non_emotion_check_tasks_pass_through(self):
        """Test that non-emotion_check tasks are not modified"""
        wrapper = EmotionAnswerWrapper()
        
        # For other tasks, should pass through original ground truth
        result = wrapper(
            ground_truth="original_answer",
            emotion="anger",
            benchmark_name="passkey"
        )
        
        self.assertEqual(result, "original_answer")
    
    def test_no_emotion_provided_passes_through(self):
        """Test that missing emotion parameter passes through unchanged"""
        wrapper = EmotionAnswerWrapper()
        
        # Without emotion, should pass through even for emotion_check
        result = wrapper(
            ground_truth="original_answer",
            benchmark_name="emotion_check"
        )
        
        self.assertEqual(result, "original_answer")
    
    def test_no_benchmark_name_passes_through(self):
        """Test that missing benchmark_name parameter passes through unchanged"""
        wrapper = EmotionAnswerWrapper()
        
        # Without benchmark_name, should pass through
        result = wrapper(
            ground_truth="original_answer",
            emotion="anger"
        )
        
        self.assertEqual(result, "original_answer")


# =============================================================================
# FACTORY FUNCTION TESTS
# =============================================================================

class TestAnswerWrapperFactory(unittest.TestCase):
    """Test the answer wrapper factory function"""
    
    def test_get_answer_wrapper_function_exists(self):
        """Test that get_answer_wrapper function exists"""
        self.assertTrue(callable(get_answer_wrapper))
    
    def test_get_emotion_check_wrapper(self):
        """Test factory returns EmotionAnswerWrapper for emotion_check"""
        wrapper = get_answer_wrapper("emotion_check", "basic_validation")
        self.assertIsInstance(wrapper, EmotionAnswerWrapper)
    
    def test_get_identity_wrapper_for_other_benchmarks(self):
        """Test factory returns IdentityAnswerWrapper for other benchmarks"""
        test_cases = [
            ("infinitebench", "passkey"),
            ("longbench", "narrativeqa"),
            ("truthfulqa", "mc1"),
            ("unknown_benchmark", "unknown_task")
        ]
        
        for benchmark_name, task_type in test_cases:
            with self.subTest(benchmark=benchmark_name, task=task_type):
                wrapper = get_answer_wrapper(benchmark_name, task_type)
                self.assertIsInstance(wrapper, IdentityAnswerWrapper)


# =============================================================================
# INTEGRATION TESTS
# =============================================================================

class TestAnswerWrapperIntegration(unittest.TestCase):
    """Test integration with existing systems"""
    
    def test_wrapper_works_with_partial_pattern(self):
        """Test AnswerWrapper works with functools.partial like PromptWrapper"""
        wrapper = EmotionAnswerWrapper()
        
        # Create partial function like we do with PromptWrapper
        emotion_partial = partial(
            wrapper.__call__,
            emotion="anger",
            benchmark_name="emotion_check"
        )
        
        # Test the partial function works
        result = emotion_partial(ground_truth="ignored")
        self.assertEqual(result, "anger")
    
    def test_wrapper_handles_benchmark_item_ground_truth(self):
        """Test wrapper works with BenchmarkItem.ground_truth types"""
        wrapper = EmotionAnswerWrapper()
        
        # Create a sample BenchmarkItem
        item = BenchmarkItem(
            id="test_1",
            input_text="How are you feeling?",
            context=None,
            ground_truth=["happy", "joyful", "pleased"],
            metadata={"category": "emotion_check"}
        )
        
        # Test wrapper handles list ground truth
        result = wrapper(
            ground_truth=item.ground_truth,
            emotion="anger",
            benchmark_name="emotion_check"
        )
        
        self.assertEqual(result, "anger")


# =============================================================================
# DATASET INTEGRATION TESTS
# =============================================================================

class TestAnswerWrapperDatasetIntegration(unittest.TestCase):
    """Test AnswerWrapper integration with dataset classes"""
    
    def test_dataset_transforms_ground_truth_with_wrapper(self):
        """Test dataset uses answer wrapper to transform ground truth"""
        # Mock dataset items
        items = [
            BenchmarkItem(
                id="test_1",
                input_text="How are you feeling?",
                context=None,
                ground_truth="original_answer",
                metadata={"category": "emotion_check"}
            )
        ]
        
        # Mock dataset with answer wrapper
        class MockDatasetWithAnswerWrapper:
            def __init__(self, items, answer_wrapper=None):
                self.items = items
                self.answer_wrapper = answer_wrapper
                self.prompt_wrapper = None
            
            def __getitem__(self, idx):
                item = self.items[idx]
                
                # Transform ground truth if wrapper provided
                ground_truth = (
                    self.answer_wrapper(item.ground_truth, emotion="anger", benchmark_name="emotion_check")
                    if self.answer_wrapper 
                    else item.ground_truth
                )
                
                return {
                    "item": item,
                    "prompt": "test_prompt",
                    "ground_truth": ground_truth
                }
        
        # Test with answer wrapper
        answer_wrapper = EmotionAnswerWrapper()
        dataset = MockDatasetWithAnswerWrapper(items, answer_wrapper)
        
        result = dataset[0]
        
        # Should return transformed ground truth (emotion)
        self.assertEqual(result["ground_truth"], "anger")
        
        # Test without answer wrapper
        dataset_no_wrapper = MockDatasetWithAnswerWrapper(items, None)
        result_no_wrapper = dataset_no_wrapper[0]
        
        # Should return original ground truth
        self.assertEqual(result_no_wrapper["ground_truth"], "original_answer")


class TestAnswerWrapperExperimentIntegration(unittest.TestCase):
    """Test AnswerWrapper integration with experiment creation flow"""
    
    def test_create_dataset_with_answer_wrapper(self):
        """Test that create_dataset_from_config can accept answer_wrapper parameter"""
        from ..dataset_factory import create_dataset_from_config
        from ..datasets.emotion_check import EmotionCheckDataset
        
        # Mock config for emotion_check  
        config = Mock(spec=BenchmarkConfig)
        config.name = "emotion_check"
        config.task_type = "basic_validation"
        config.base_data_dir = "data/emotion_scales"
        config.sample_limit = None
        config.enable_auto_truncation = False
        config.truncation_strategy = "right"
        config.preserve_ratio = 0.8
        config.augmentation_config = None
        config.llm_eval_config = None
        
        answer_wrapper = EmotionAnswerWrapper()
        
        # Mock the data loading to avoid file system issues
        with patch.object(EmotionCheckDataset, '_load_raw_data', return_value=[
            {
                "id": "test_1",
                "input": "How are you feeling?",
                "ground_truth": ["happy", "joyful"],
                "category": "emotion_check"
            }
        ]):
            # This works now that we've updated the factory function
            dataset = create_dataset_from_config(
                config,
                prompt_wrapper=None,
                answer_wrapper=answer_wrapper
            )
            # Should succeed without error
            self.assertTrue(hasattr(dataset, 'answer_wrapper'))
            self.assertEqual(dataset.answer_wrapper, answer_wrapper)
    
    def test_experiment_creates_answer_wrapper_partial(self):
        """Test that _create_dataset_for_emotion creates answer wrapper partial"""
        # Simulate what should happen in _create_dataset_for_emotion
        emotion = "anger"
        benchmark_name = "emotion_check"
        task_type = "basic_validation"
        
        # Get answer wrapper from factory
        answer_wrapper = get_answer_wrapper(benchmark_name, task_type)
        
        # Create partial with emotion context (updated to use benchmark_name)
        answer_wrapper_partial = partial(
            answer_wrapper.__call__,
            emotion=emotion,
            benchmark_name=benchmark_name,
            task_type=task_type
        )
        
        # Test the partial works correctly
        result = answer_wrapper_partial(ground_truth="ignored")
        self.assertEqual(result, "anger")


# =============================================================================
# END-TO-END TESTS
# =============================================================================

class TestAnswerWrapperEndToEnd(unittest.TestCase):
    """Test complete integration of answer wrapper with experiment workflow"""
    
    def test_emotion_check_experiment_ground_truth_adaptation(self):
        """Test that EmotionCheck experiment can adapt ground truth based on emotion"""
        # Simulate what happens in _create_dataset_for_emotion method
        emotion = "anger"
        
        # 1. Create benchmark configuration for emotion check
        config = Mock(spec=BenchmarkConfig)
        config.name = "emotion_check"
        config.task_type = "basic_validation"
        config.base_data_dir = "data/emotion_scales"
        config.sample_limit = None
        config.enable_auto_truncation = True
        config.truncation_strategy = "right"
        config.preserve_ratio = 0.8
        config.augmentation_config = None
        config.llm_eval_config = None
        
        # 2. Get answer wrapper from factory
        answer_wrapper = get_answer_wrapper(config.name, config.task_type)
        self.assertIsInstance(answer_wrapper, EmotionAnswerWrapper)
        
        # 3. Create partial function with emotion context
        answer_wrapper_partial = partial(
            answer_wrapper.__call__,
            emotion=emotion,
            benchmark_name=config.name,
            task_type=config.task_type
        )
        
        # 4. Test that the partial function adapts ground truth correctly
        original_ground_truth = ["happy", "joyful", "pleased"]
        adapted_ground_truth = answer_wrapper_partial(original_ground_truth)
        
        # Should return the activated emotion instead of original ground truth
        self.assertEqual(adapted_ground_truth, "anger")
        
    def test_answer_wrapper_with_different_emotions(self):
        """Test answer wrapper works with different emotions"""
        emotions = ["happiness", "sadness", "fear", "disgust", "surprise"]
        
        answer_wrapper = EmotionAnswerWrapper()
        
        for emotion in emotions:
            with self.subTest(emotion=emotion):
                answer_wrapper_partial = partial(
                    answer_wrapper.__call__,
                    emotion=emotion,
                    benchmark_name="emotion_check"
                )
                
                result = answer_wrapper_partial("ignored")
                self.assertEqual(result, emotion)
    
    def test_identity_wrapper_for_other_benchmarks(self):
        """Test that other benchmarks get identity wrapper (no ground truth changes)"""
        other_benchmarks = [
            ("infinitebench", "passkey"),
            ("longbench", "narrativeqa"),
            ("truthfulqa", "mc1")
        ]
        
        for benchmark_name, task_type in other_benchmarks:
            with self.subTest(benchmark=benchmark_name, task=task_type):
                answer_wrapper = get_answer_wrapper(benchmark_name, task_type)
                
                # Should be identity wrapper
                self.assertIsInstance(answer_wrapper, IdentityAnswerWrapper)
                
                # Should pass through unchanged
                original_ground_truth = "test_answer"
                result = answer_wrapper(original_ground_truth, emotion="anger")
                self.assertEqual(result, original_ground_truth)
    
    def test_complete_workflow_simulation(self):
        """Test complete workflow as it would happen in _create_dataset_for_emotion"""
        emotion = "sadness"
        
        # Mock what would be passed to _create_dataset_for_emotion
        class MockExperiment:
            def __init__(self):
                self.config = Mock()
                self.config.benchmark.name = "emotion_check"
                self.config.benchmark.task_type = "basic_validation"
                self.config.benchmark.augmentation_config = None
                self.enable_thinking = False
                
            def _create_dataset_for_emotion(self, emotion: str):
                """Simulated _create_dataset_for_emotion with AnswerWrapper integration"""
                
                # Get answer wrapper (new step)
                answer_wrapper = get_answer_wrapper(
                    self.config.benchmark.name,
                    self.config.benchmark.task_type
                )
                
                # Create partial with emotion context (new step)
                answer_wrapper_partial = partial(
                    answer_wrapper.__call__,
                    emotion=emotion,
                    benchmark_name=self.config.benchmark.name,
                    task_type=self.config.benchmark.task_type
                )
                
                # Mock prompt wrapper (existing)
                prompt_wrapper_partial = Mock()
                
                return {
                    'answer_wrapper_partial': answer_wrapper_partial,
                    'prompt_wrapper_partial': prompt_wrapper_partial,
                    'emotion': emotion
                }
        
        experiment = MockExperiment()
        result = experiment._create_dataset_for_emotion(emotion)
        
        # Test the answer wrapper partial works correctly
        answer_wrapper_partial = result['answer_wrapper_partial']
        adapted_ground_truth = answer_wrapper_partial("original")
        
        self.assertEqual(adapted_ground_truth, "sadness")
        self.assertEqual(result['emotion'], "sadness")


# =============================================================================
# REGRESSION TESTS
# =============================================================================

class TestAnswerWrapperRegressionTests(unittest.TestCase):
    """Ensure AnswerWrapper doesn't break existing functionality"""
    
    def test_emotion_check_dataset_still_works_without_wrapper(self):
        """Test EmotionCheckDataset works without answer wrapper (backward compatibility)"""
        from ..datasets.emotion_check import EmotionCheckDataset
        
        # Mock minimal config
        config = Mock(spec=BenchmarkConfig)
        config.name = "emotion_check"
        config.task_type = "basic_validation"
        config.base_data_dir = "data/emotion_scales"
        config.sample_limit = None
        config.enable_auto_truncation = False
        config.truncation_strategy = "right"
        config.preserve_ratio = 0.8
        config.augmentation_config = None
        config.llm_eval_config = None
        
        # Mock the data loading
        with patch.object(EmotionCheckDataset, '_load_raw_data', return_value=[
            {
                "id": "test_1",
                "input": "How are you feeling?",
                "ground_truth": ["happy", "joyful"],
                "category": "emotion_check"
            }
        ]):
            # Should work without answer_wrapper parameter
            dataset = EmotionCheckDataset(config, prompt_wrapper=None)
            
            # Should have loaded the item
            self.assertEqual(len(dataset), 1)
            
            # __getitem__ should work (return original ground truth without wrapper)
            result = dataset[0]
            self.assertEqual(result["ground_truth"], ["happy", "joyful"])
    
    def test_emotion_check_dataset_works_with_wrapper(self):
        """Test EmotionCheckDataset works correctly with answer wrapper"""
        from ..datasets.emotion_check import EmotionCheckDataset
        
        # Mock minimal config
        config = Mock(spec=BenchmarkConfig)
        config.name = "emotion_check"
        config.task_type = "basic_validation"
        config.base_data_dir = "data/emotion_scales"
        config.sample_limit = None
        config.enable_auto_truncation = False
        config.truncation_strategy = "right"
        config.preserve_ratio = 0.8
        config.augmentation_config = None
        config.llm_eval_config = None
        
        # Create answer wrapper partial
        answer_wrapper = EmotionAnswerWrapper()
        answer_wrapper_partial = partial(
            answer_wrapper.__call__,
            emotion="anger",
            benchmark_name="emotion_check"
        )
        
        # Mock the data loading
        with patch.object(EmotionCheckDataset, '_load_raw_data', return_value=[
            {
                "id": "test_1",
                "input": "How are you feeling?",
                "ground_truth": ["happy", "joyful"],
                "category": "emotion_check"
            }
        ]):
            # Should work with answer_wrapper parameter
            dataset = EmotionCheckDataset(
                config, 
                prompt_wrapper=None, 
                answer_wrapper=answer_wrapper_partial
            )
            
            # Should have loaded the item
            self.assertEqual(len(dataset), 1)
            
            # __getitem__ should return transformed ground truth
            result = dataset[0]
            self.assertEqual(result["ground_truth"], "anger")


if __name__ == "__main__":
    unittest.main()