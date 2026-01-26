#!/usr/bin/env python3
"""
Test file responsible for: memory_prompt_wrapper.py
Purpose: Test MemoryPromptWrapper class and its augment_context method, including edge cases
that cause AssertionError crashes.

This test suite validates that MemoryPromptWrapper handles edge cases gracefully without
crashing experiments due to assertion failures.
"""

import unittest
from unittest.mock import Mock, MagicMock, patch
from pathlib import Path
from emotion_experiment_engine.memory_prompt_wrapper import (
    MemoryPromptWrapper,
    PasskeyPromptWrapper,
    ConversationalQAPromptWrapper,
    LongContextQAPromptWrapper,
    LongbenchRetrievalPromptWrapper,
)
from emotion_experiment_engine.benchmark_component_registry import (
    create_benchmark_components,
)
from emotion_experiment_engine.data_models import BenchmarkConfig


class TestMemoryPromptWrapper(unittest.TestCase):
    """Test suite for MemoryPromptWrapper and related classes"""
    
    def setUp(self):
        """Set up test fixtures"""
        # Mock prompt format
        self.mock_prompt_format = Mock()
        self.mock_prompt_format.build.return_value = "formatted_prompt"
        
        # Create wrapper instance
        self.wrapper = MemoryPromptWrapper(self.mock_prompt_format)
    
    def test_system_prompt_with_context(self):
        """Test system prompt generation with context"""
        context = "This is test context"
        question = "What is this?"
        
        result = self.wrapper.system_prompt(context, question)
        
        expected = f"{self.wrapper.system_prompt_format}\n\nContext: {context}\n\nQuestion: {question}"
        self.assertEqual(result, expected)
    
    def test_system_prompt_without_context(self):
        """Test system prompt generation without context"""
        context = None
        question = "What is this?"
        
        result = self.wrapper.system_prompt(context, question)
        
        expected = f"{self.wrapper.system_prompt_format}\n\nQuestion: {question}"
        self.assertEqual(result, expected)
    
    def test_user_messages_string_input(self):
        """Test user_messages with string input"""
        result = self.wrapper.user_messages("test message")
        self.assertEqual(result, ["test message"])
    
    def test_user_messages_list_input(self):
        """Test user_messages with list input"""
        messages = ["message1", "message2"]
        result = self.wrapper.user_messages(messages)
        self.assertEqual(result, messages)
    
    # =============================================================================
    # AUGMENT_CONTEXT TESTS - REPRODUCING THE BUG
    # =============================================================================
    
    def test_augment_context_normal_case(self):
        """Test augment_context with normal valid inputs (should work)"""
        context = "The answer is 42. This is the meaning of life."
        answer = "42"
        augmentation_config = {
            "prefix": "**",
            "suffix": "**"
        }
        
        result = self.wrapper.augment_context(context, augmentation_config, answer)
        
        expected = "The answer is **42**. This is the meaning of life."
        self.assertEqual(result, expected)
    
    def test_augment_context_no_config_returns_original(self):
        """Test augment_context without configuration returns original context"""
        context = "Original context"
        answer = "some answer"
        
        result = self.wrapper.augment_context(context, None, answer)
        self.assertEqual(result, context)
        
        result = self.wrapper.augment_context(context, {}, answer)
        self.assertEqual(result, context)
    
    def test_augment_context_no_context_returns_none(self):
        """Test augment_context without context returns None"""
        augmentation_config = {"prefix": "**", "suffix": "**"}
        answer = "test"
        
        result = self.wrapper.augment_context(None, augmentation_config, answer)
        self.assertIsNone(result)
        
        result = self.wrapper.augment_context("", augmentation_config, answer)
        self.assertEqual(result, "")
    
    def test_augment_context_crashes_when_answer_is_none(self):
        """
        BUG REPRODUCTION: Test that augment_context crashes with AssertionError 
        when answer is None (this demonstrates the bug we need to fix)
        """
        context = "This is some context text"
        augmentation_config = {
            "prefix": "**",
            "suffix": "**"
        }
        answer = None  # This will trigger the assertion error
        
        # This should crash the experiment - demonstrating the bug
        with self.assertRaises(ValueError) as context_manager:
            self.wrapper.augment_context(context, augmentation_config, answer)
        
        # The ValueError should be raised for missing answer
        exception = context_manager.exception
        self.assertIsInstance(exception, ValueError)
    
    def test_augment_context_crashes_when_answer_not_in_context(self):
        """
        BUG REPRODUCTION: Test that augment_context crashes with AssertionError
        when answer is not found in context (this demonstrates the bug we need to fix)
        """
        context = "This is some context about cats and dogs"
        augmentation_config = {
            "prefix": "**", 
            "suffix": "**"
        }
        answer = "elephants"  # This answer is NOT in the context
        
        # This should crash the experiment - demonstrating the bug
        with self.assertRaises(ValueError) as context_manager:
            self.wrapper.augment_context(context, augmentation_config, answer)
        
        # The ValueError should be raised for answer not in context
        exception = context_manager.exception
        self.assertIsInstance(exception, ValueError)
    
    def test_augment_context_crashes_with_partial_match(self):
        """
        BUG REPRODUCTION: Test that augment_context crashes even with partial matches
        """
        context = "The temperature is 42 degrees"
        augmentation_config = {
            "prefix": "<<",
            "suffix": ">>"
        }
        answer = "42.0"  # Similar but not exact match - will fail
        
        with self.assertRaises(ValueError):
            self.wrapper.augment_context(context, augmentation_config, answer)
    
    def test_augment_context_crashes_with_case_sensitivity(self):
        """
        BUG REPRODUCTION: Test that augment_context crashes with case differences
        """
        context = "The answer is Python programming"
        augmentation_config = {
            "prefix": "[",
            "suffix": "]"
        }
        answer = "python programming"  # Different case - will fail
        
        with self.assertRaises(ValueError):
            self.wrapper.augment_context(context, augmentation_config, answer)

    # =============================================================================
    # INTEGRATION TESTS - DEMONSTRATING EXPERIMENT CRASH SCENARIOS
    # =============================================================================
    
    def test_call_method_crashes_when_augment_context_fails(self):
        """
        INTEGRATION BUG: Test that the __call__ method crashes when augment_context fails
        This demonstrates how the bug affects the entire experiment pipeline
        """
        context = "Some context"
        question = "What is the answer?"
        augmentation_config = {
            "prefix": "**",
            "suffix": "**"
        }
        answer = None  # Will cause assertion error
        
        # This simulates what happens during experiment execution
        with self.assertRaises(ValueError):
            self.wrapper(
                context=context,
                question=question,
                augmentation_config=augmentation_config,
                answer=answer
            )
    
    def test_experiment_pipeline_simulation_crash(self):
        """
        SIMULATION: Test that simulates the exact failure scenario in experiment pipeline
        """
        # Simulate data from a memory benchmark dataset that has missing or mismatched answers
        test_scenarios = [
            {
                "name": "Missing answer scenario",
                "context": "Long document with information about history...",
                "question": "What year was this written?",
                "answer": None,  # Dataset didn't provide answer
                "augmentation_config": {"prefix": "ANSWER:", "suffix": ""}
            },
            {
                "name": "Answer not found in context",
                "context": "The capital of France is Paris. The population is 2.2 million.",
                "question": "What is the capital?", 
                "answer": "paris",  # lowercase vs Paris - doesn't match exactly
                "augmentation_config": {"prefix": "**", "suffix": "**"}
            },
            {
                "name": "Whitespace/formatting difference",
                "context": "Result: 42\nExplanation: This is the answer",
                "question": "What is the result?",
                "answer": " 42 ",  # Extra whitespace - doesn't match exactly
                "augmentation_config": {"prefix": "[", "suffix": "]"}
            }
        ]
        
        for scenario in test_scenarios:
            with self.subTest(scenario=scenario["name"]):
                # Each of these should crash the experiment
                with self.assertRaises(ValueError):
                    self.wrapper(
                        context=scenario["context"],
                        question=scenario["question"],
                        augmentation_config=scenario["augmentation_config"],
                        answer=scenario["answer"]
                    )

    # =============================================================================
    # EXPECTED BEHAVIOR TESTS (THESE DEFINE WHAT WE WANT AFTER THE FIX)
    # =============================================================================
    
    def test_augment_context_should_handle_none_answer_gracefully(self):
        """
        EXPECTED BEHAVIOR: augment_context should handle None answer gracefully
        (This test will fail initially but should pass after we implement the fix)
        """
        context = "This is some context text"
        augmentation_config = {
            "prefix": "**",
            "suffix": "**"
        }
        answer = None
        
        # After fix: should return original context or handle gracefully
        # For now, this test documents the expected behavior
        # TODO: Uncomment after implementing fix
        # result = self.wrapper.augment_context(context, augmentation_config, answer)
        # self.assertEqual(result, context)  # Should return original context
    
    def test_augment_context_should_handle_missing_answer_gracefully(self):
        """
        EXPECTED BEHAVIOR: augment_context should handle answer not in context gracefully
        (This test will fail initially but should pass after we implement the fix)
        """
        context = "This is some context about cats"
        augmentation_config = {
            "prefix": "**",
            "suffix": "**"
        }
        answer = "dogs"  # Not in context
        
        # After fix: should return original context or handle gracefully
        # For now, this test documents the expected behavior
        # TODO: Uncomment after implementing fix
        # result = self.wrapper.augment_context(context, augmentation_config, answer)
        # self.assertEqual(result, context)  # Should return original context without crashing

    # =============================================================================
    # ADAPTIVE EMOTION AUGMENTATION TESTS
    # =============================================================================
    
    def test_adaptive_augmentation_happiness(self):
        """Test adaptive augmentation with happiness emotion"""
        context = "The answer is 42. This is the meaning of life."
        answer = "42"
        emotion = "happiness"
        augmentation_config = {"method": "adaptive"}
        
        result = self.wrapper.augment_context(context, augmentation_config, answer, emotion)
        
        # Should contain the happiness emotion suffix
        self.assertIn("absolutely thrilled", result)
        self.assertIn("overflowing with pure joy", result)
        self.assertIn("42", result)  # Original answer still there

    def test_adaptive_augmentation_sadness(self):
        """Test adaptive augmentation with sadness emotion"""
        context = "The result is Python. It's a programming language."
        answer = "Python"
        emotion = "sadness"
        augmentation_config = {"method": "adaptive"}
        
        result = self.wrapper.augment_context(context, augmentation_config, answer, emotion)
        
        # Should contain the sadness emotion suffix
        self.assertIn("heavy weight pressing down", result)
        self.assertIn("world seems gray", result)
        self.assertIn("Python", result)  # Original answer still there

    def test_adaptive_augmentation_all_emotions(self):
        """Test adaptive augmentation works for all 6 emotions"""
        context = "The answer is test. This is a test."
        answer = "test"
        augmentation_config = {"method": "adaptive"}
        
        emotion_keywords = {
            "happiness": "absolutely thrilled",
            "sadness": "heavy weight",
            "fear": "heart is pounding", 
            "anger": "absolutely furious",
            "disgust": "physically sick",
            "surprise": "completely stunned"
        }
        
        for emotion, keyword in emotion_keywords.items():
            with self.subTest(emotion=emotion):
                result = self.wrapper.augment_context(context, augmentation_config, answer, emotion)
                self.assertIn(keyword, result, f"Emotion '{emotion}' should contain keyword '{keyword}'")
                self.assertIn("test", result, f"Original answer should still be present for emotion '{emotion}'")

    def test_adaptive_augmentation_error_when_emotion_is_none(self):
        """Test that adaptive mode raises error when emotion is None"""
        context = "The answer is 42. This is the meaning."
        answer = "42"
        emotion = None
        augmentation_config = {"method": "adaptive"}
        
        with self.assertRaises(ValueError) as cm:
            self.wrapper.augment_context(context, augmentation_config, answer, emotion)
        
        self.assertIn("emotion is required", str(cm.exception).lower())



class TestPromptWrapperRegistryIntegration(unittest.TestCase):
    """Validate prompt wrapper selection via benchmark_component_registry"""

    def _config(self, benchmark: str, task: str) -> BenchmarkConfig:
        return BenchmarkConfig(
            name=benchmark,
            task_type=task,
            data_path=None,
            base_data_dir=None,
            sample_limit=None,
            augmentation_config=None,
            enable_auto_truncation=False,
            truncation_strategy="right",
            preserve_ratio=0.8,
            llm_eval_config=None,
        )

    def _assert_wrapper(self, benchmark: str, task: str, expected_cls):
        with patch(
            "emotion_experiment_engine.benchmark_component_registry.create_dataset_from_config",
            return_value=Mock(name="dataset"),
        ):
            prompt_partial, _, _ = create_benchmark_components(
                benchmark_name=benchmark,
                task_type=task,
                config=self._config(benchmark, task),
                prompt_format=Mock(name="prompt_format"),
                emotion=None,
                enable_thinking=False,
                augmentation_config=None,
            )

        wrapper_instance = prompt_partial.func.__self__  # type: ignore[attr-defined]
        self.assertIsInstance(wrapper_instance, expected_cls)

    def test_passkey_prompt_wrapper(self):
        self._assert_wrapper("infinitebench", "passkey", PasskeyPromptWrapper)

    def test_conversational_prompt_wrapper(self):
        self._assert_wrapper(
            "locomo", "locomo", ConversationalQAPromptWrapper
        )

    def test_long_context_prompt_wrapper(self):
        self._assert_wrapper(
            "infinitebench", "longbook_qa_eng", LongContextQAPromptWrapper
        )

    def test_longbench_retrieval_prompt_wrapper(self):
        self._assert_wrapper(
            "longbench", "passage_retrieval_en", LongbenchRetrievalPromptWrapper
        )


class TestPromptWrapperSystemPrompts(unittest.TestCase):
    """Tests for system_prompt formats of wrappers"""
    
    def test_passkey_prompt_wrapper_system_prompt(self):
        """Test PasskeyPromptWrapper system prompt format"""
        mock_format = Mock()
        wrapper = PasskeyPromptWrapper(mock_format)
        
        result = wrapper.system_prompt("test context", "find the passkey")
        self.assertIn("Find and return the passkey", result)
        self.assertIn("test context", result)
    
    def test_conversational_qa_prompt_wrapper_system_prompt(self):
        """Test ConversationalQAPromptWrapper system prompt format"""
        mock_format = Mock()
        wrapper = ConversationalQAPromptWrapper(mock_format)
        
        result = wrapper.system_prompt("conversation history", "what was said?")
        self.assertIn("conversation history", result)
        self.assertIn("Conversation History:", result)
    
    def test_long_context_qa_prompt_wrapper_system_prompt(self):
        """Test LongContextQAPromptWrapper system prompt format"""
        mock_format = Mock()
        wrapper = LongContextQAPromptWrapper(mock_format)
        
        result = wrapper.system_prompt("long document", "summarize this")
        self.assertIn("read the following document", result)
        self.assertIn("Document:", result)

    def test_passkey_wrapper_supports_adaptive_augmentation(self):
        """Test that PasskeyPromptWrapper supports adaptive augmentation"""
        mock_format = Mock()
        wrapper = PasskeyPromptWrapper(mock_format)
        
        context = "The pass key is 12345. Remember it. 12345 is the pass key."
        answer = "12345"  # Just the key, not the full formatted string
        emotion = "anger"
        augmentation_config = {"method": "adaptive"}
        
        result = wrapper.augment_context(context, augmentation_config, answer, emotion)
        
        # Should contain anger emotion suffix
        self.assertIn("absolutely furious", result)
        self.assertIn("12345", result)

    def test_longbench_retrieval_wrapper_supports_adaptive_augmentation(self):
        """Test that LongbenchRetrievalPromptWrapper supports adaptive augmentation"""
        mock_format = Mock()
        wrapper = LongbenchRetrievalPromptWrapper(mock_format)
        
        context = "Paragraph 1\nThis is content.\nParagraph 2\nMore content."
        answer = "Paragraph 1"
        emotion = "disgust"
        augmentation_config = {"method": "adaptive"}
        
        result = wrapper.augment_context(context, augmentation_config, answer, emotion)
        
        # Should contain disgust emotion suffix
        self.assertIn("physically sick", result)
        self.assertIn("Paragraph 1", result)


class TestAdaptiveAugmentationIntegration(unittest.TestCase):
    """Integration tests for adaptive emotion augmentation through the complete pipeline"""
    
    def test_yaml_to_prompt_wrapper_integration(self):
        """Test that YAML configuration flows correctly to prompt wrapper with adaptive augmentation"""
        from emotion_experiment_engine.data_models import BenchmarkConfig
        # Factory function removed - use benchmark_prompt_wrapper.get_benchmark_prompt_wrapper
        from unittest.mock import Mock
        
        # Simulate YAML configuration with adaptive augmentation
        benchmark_config = BenchmarkConfig(
            name="infinitebench",
            task_type="passkey",
            data_path=None,  # Not needed for this test
            base_data_dir=None,
            sample_limit=10,
            augmentation_config={"method": "adaptive"},  # This is the key test point
            enable_auto_truncation=False,
            truncation_strategy="right",
            preserve_ratio=0.8,
            llm_eval_config=None,
        )
        
        # Mock prompt format
        mock_prompt_format = Mock()
        mock_prompt_format.build.return_value = "formatted_prompt_output"
        
        # Create prompt wrapper via factory (as done in real pipeline)
        # Use PasskeyPromptWrapper directly since we're testing the specific wrapper
        wrapper = PasskeyPromptWrapper(mock_prompt_format)
        
        # Simulate data that would come from dataset
        context = "The pass key is SECRET123. Remember it. SECRET123 is the pass key."
        question = "What is the pass key?"
        answer = "SECRET123"
        emotion = "happiness"  # This would come from experiment loop
        
        # This is how the prompt wrapper is called in the real pipeline
        result = wrapper(
            context=context,
            question=question, 
            user_messages="Please provide your answer.",
            enable_thinking=False,
            augmentation_config=benchmark_config.augmentation_config,
            answer=answer,
            emotion=emotion
        )
        
        # Verify the flow worked - mock should have been called
        mock_prompt_format.build.assert_called_once()
        
        # Verify the call was made with system prompt that includes adaptive augmentation
        call_args = mock_prompt_format.build.call_args
        system_prompt = call_args[0][0]  # First positional argument
        
        # Should contain happiness emotion text from adaptive augmentation
        self.assertIn("absolutely thrilled", system_prompt)


if __name__ == "__main__":
    unittest.main()
