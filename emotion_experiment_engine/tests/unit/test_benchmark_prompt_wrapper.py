"""
Test file for benchmark prompt wrapper factory system.
Tests the new universal factory and MTBench101 wrapper integration.
"""

import unittest
from unittest.mock import Mock
from neuro_manipulation.prompt_formats import PromptFormat

from emotion_experiment_engine.benchmark_prompt_wrapper import (
    get_benchmark_prompt_wrapper,
    get_supported_benchmarks,
    is_supported_benchmark_task
)
from emotion_experiment_engine.mtbench101_prompt_wrapper import MTBench101PromptWrapper
from emotion_experiment_engine.memory_prompt_wrapper import (
    MemoryPromptWrapper, 
    PasskeyPromptWrapper,
    ConversationalQAPromptWrapper,
    LongContextQAPromptWrapper,
    LongbenchRetrievalPromptWrapper
)


class TestBenchmarkPromptWrapperFactory(unittest.TestCase):
    """Test the universal benchmark prompt wrapper factory"""
    
    def setUp(self):
        """Set up mock prompt format for testing"""
        self.mock_prompt_format = Mock(spec=PromptFormat)
        self.mock_prompt_format.build.return_value = "test_prompt"
    
    def test_mtbench101_factory_routing(self):
        """Test factory correctly routes MTBench101 tasks to MTBench101PromptWrapper"""
        mtbench_tasks = ["CM", "SI", "AR", "TS", "CC", "CR", "FR", "SC", "SA", "MR", "GR", "IC", "PI"]
        
        for task in mtbench_tasks:
            with self.subTest(task=task):
                wrapper = get_benchmark_prompt_wrapper("mtbench101", task, self.mock_prompt_format)
                self.assertIsInstance(wrapper, MTBench101PromptWrapper)
                self.assertEqual(wrapper.task_type, task)
    
    def test_infinitebench_factory_routing(self):
        """Test factory correctly routes InfiniteBench tasks"""
        # Passkey task
        wrapper = get_benchmark_prompt_wrapper("infinitebench", "passkey", self.mock_prompt_format)
        self.assertIsInstance(wrapper, PasskeyPromptWrapper)
        
        # Conversational task
        wrapper = get_benchmark_prompt_wrapper("infinitebench", "conversational", self.mock_prompt_format)
        self.assertIsInstance(wrapper, ConversationalQAPromptWrapper)
        
        # QA task
        wrapper = get_benchmark_prompt_wrapper("infinitebench", "qa", self.mock_prompt_format)
        self.assertIsInstance(wrapper, LongContextQAPromptWrapper)
        
        # Unknown task (defaults to MemoryPromptWrapper)
        wrapper = get_benchmark_prompt_wrapper("infinitebench", "unknown_task", self.mock_prompt_format)
        self.assertIsInstance(wrapper, MemoryPromptWrapper)
        self.assertNotIsInstance(wrapper, (PasskeyPromptWrapper, ConversationalQAPromptWrapper))
    
    def test_longbench_factory_routing(self):
        """Test factory correctly routes LongBench tasks"""
        # Retrieval task
        wrapper = get_benchmark_prompt_wrapper("longbench", "retrieval", self.mock_prompt_format)
        self.assertIsInstance(wrapper, LongbenchRetrievalPromptWrapper)
        
        # QA task
        wrapper = get_benchmark_prompt_wrapper("longbench", "qa", self.mock_prompt_format)
        self.assertIsInstance(wrapper, LongContextQAPromptWrapper)
        
        # Unknown task
        wrapper = get_benchmark_prompt_wrapper("longbench", "unknown", self.mock_prompt_format)
        self.assertIsInstance(wrapper, MemoryPromptWrapper)
    
    def test_locomo_factory_routing(self):
        """Test factory correctly routes LoCoMo tasks"""
        wrapper = get_benchmark_prompt_wrapper("locomo", "conversational_qa", self.mock_prompt_format)
        self.assertIsInstance(wrapper, ConversationalQAPromptWrapper)
        
        wrapper = get_benchmark_prompt_wrapper("conversational", "any_task", self.mock_prompt_format)
        self.assertIsInstance(wrapper, ConversationalQAPromptWrapper)
    
    def test_case_insensitive_routing(self):
        """Test factory handles case-insensitive benchmark and task names"""
        # Test different case variations
        test_cases = [
            ("MTBENCH101", "cm", MTBench101PromptWrapper),
            ("mtbench101", "CM", MTBench101PromptWrapper),
            ("MTBench101", "SI", MTBench101PromptWrapper),
            ("INFINITEBENCH", "passkey", PasskeyPromptWrapper),
            ("InfiniteBench", "PASSKEY", PasskeyPromptWrapper)
        ]
        
        for benchmark, task, expected_type in test_cases:
            with self.subTest(benchmark=benchmark, task=task):
                wrapper = get_benchmark_prompt_wrapper(benchmark, task, self.mock_prompt_format)
                self.assertIsInstance(wrapper, expected_type)
    
    def test_legacy_task_routing(self):
        """Test factory handles legacy task routing for backward compatibility"""
        # When benchmark name might refer to task types (legacy usage)
        wrapper = get_benchmark_prompt_wrapper("passkey", "any", self.mock_prompt_format)
        self.assertIsInstance(wrapper, PasskeyPromptWrapper)
        
        wrapper = get_benchmark_prompt_wrapper("conversational", "any", self.mock_prompt_format)
        self.assertIsInstance(wrapper, ConversationalQAPromptWrapper)
        
        wrapper = get_benchmark_prompt_wrapper("retrieval", "any", self.mock_prompt_format)
        self.assertIsInstance(wrapper, LongbenchRetrievalPromptWrapper)
    
    def test_unknown_benchmark_fallback(self):
        """Test factory returns default MemoryPromptWrapper for unknown benchmarks"""
        wrapper = get_benchmark_prompt_wrapper("unknown_benchmark", "any_task", self.mock_prompt_format)
        self.assertIsInstance(wrapper, MemoryPromptWrapper)
        self.assertNotIsInstance(wrapper, (PasskeyPromptWrapper, MTBench101PromptWrapper))
    
    def test_get_supported_benchmarks(self):
        """Test get_supported_benchmarks returns correct structure"""
        supported = get_supported_benchmarks()
        
        # Check expected benchmarks are present
        self.assertIn("mtbench101", supported)
        self.assertIn("infinitebench", supported)
        self.assertIn("longbench", supported)
        self.assertIn("locomo", supported)
        
        # Check MTBench101 has all 13 tasks
        self.assertEqual(len(supported["mtbench101"]), 13)
        self.assertIn("CM", supported["mtbench101"])
        self.assertIn("PI", supported["mtbench101"])
        
        # Check structure is dict of lists
        for benchmark, tasks in supported.items():
            self.assertIsInstance(tasks, list)
            self.assertGreater(len(tasks), 0)
    
    def test_is_supported_benchmark_task(self):
        """Test is_supported_benchmark_task function"""
        # Test supported combinations
        self.assertTrue(is_supported_benchmark_task("mtbench101", "CM"))
        self.assertTrue(is_supported_benchmark_task("infinitebench", "passkey"))
        self.assertTrue(is_supported_benchmark_task("longbench", "narrativeqa"))
        
        # Test unsupported combinations (but should still return True for unknown benchmarks)
        self.assertTrue(is_supported_benchmark_task("unknown_benchmark", "any_task"))
        
        # Test case sensitivity
        self.assertTrue(is_supported_benchmark_task("MTBENCH101", "cm"))


class TestMTBench101PromptWrapper(unittest.TestCase):
    """Test MTBench101PromptWrapper specifically"""
    
    def setUp(self):
        """Set up mock prompt format for testing"""
        self.mock_prompt_format = Mock(spec=PromptFormat)
        self.mock_prompt_format.build.return_value = "formatted_prompt"
    
    def test_mtbench101_wrapper_initialization(self):
        """Test MTBench101PromptWrapper initialization with different task types"""
        # Test with specific task type
        wrapper = MTBench101PromptWrapper(self.mock_prompt_format, "CM")
        self.assertEqual(wrapper.task_type, "CM")
        self.assertEqual(wrapper.prompt_format, self.mock_prompt_format)
        
        # Test with no task type
        wrapper = MTBench101PromptWrapper(self.mock_prompt_format)
        self.assertIsNone(wrapper.task_type)
    
    def test_get_system_prompt(self):
        """Test get_system_prompt method"""
        wrapper = MTBench101PromptWrapper(self.mock_prompt_format, "CM")
        
        # Test default prompt for known task
        system_prompt = wrapper.get_system_prompt()
        self.assertIsInstance(system_prompt, str)
        self.assertIn("helpful AI assistant", system_prompt)
        
        # Test custom prompt override
        custom_prompt = "Custom system prompt"
        system_prompt = wrapper.get_system_prompt(custom_prompt)
        self.assertEqual(system_prompt, custom_prompt)
        
        # Test unknown task fallback
        wrapper = MTBench101PromptWrapper(self.mock_prompt_format, "UNKNOWN")
        system_prompt = wrapper.get_system_prompt()
        self.assertEqual(system_prompt, wrapper.TASK_PROMPTS["default"])
    
    def test_call_method_conversation_handling(self):
        """Test __call__ method handles conversation format correctly"""
        wrapper = MTBench101PromptWrapper(self.mock_prompt_format, "CM")
        
        user_messages = ["Hello", "How are you?", "What's the weather?"]
        assistant_messages = ["Hi there!", "I'm good.", "It's sunny!"]
        
        # Call the wrapper
        result = wrapper(user_messages, assistant_messages)
        
        # Verify prompt format was called correctly
        self.mock_prompt_format.build.assert_called_once()
        call_args = self.mock_prompt_format.build.call_args
        
        # Check arguments passed to build method
        self.assertIn("system_prompt", call_args.kwargs)
        self.assertIn("user_messages", call_args.kwargs)
        self.assertIn("assistant_messages", call_args.kwargs)
        self.assertIn("enable_thinking", call_args.kwargs)
        
        # Check that assistant messages are truncated (last one removed for generation)
        passed_assistant_messages = call_args.kwargs["assistant_messages"]
        self.assertEqual(len(passed_assistant_messages), 2)  # One less than original
        self.assertEqual(passed_assistant_messages, ["Hi there!", "I'm good."])
        
        # Check user messages are passed as-is
        passed_user_messages = call_args.kwargs["user_messages"]
        self.assertEqual(passed_user_messages, user_messages)
        
        self.assertEqual(result, "formatted_prompt")
    
    def test_call_method_empty_assistant_messages(self):
        """Test __call__ method with empty assistant messages"""
        wrapper = MTBench101PromptWrapper(self.mock_prompt_format, "SI")
        
        user_messages = ["Please help me"]
        result = wrapper(user_messages)  # No assistant messages
        
        # Verify build was called
        self.mock_prompt_format.build.assert_called_once()
        call_args = self.mock_prompt_format.build.call_args
        
        # Assistant messages should be empty list
        self.assertEqual(call_args.kwargs["assistant_messages"], [])
        self.assertEqual(call_args.kwargs["user_messages"], user_messages)
        
        self.assertEqual(result, "formatted_prompt")
    
    def test_call_method_with_enable_thinking(self):
        """Test __call__ method with enable_thinking parameter"""
        wrapper = MTBench101PromptWrapper(self.mock_prompt_format, "MR")
        
        user_messages = ["What is 2+2?"]
        result = wrapper(user_messages, enable_thinking=True)
        
        # Verify enable_thinking was passed correctly
        self.mock_prompt_format.build.assert_called_once()
        call_args = self.mock_prompt_format.build.call_args
        self.assertTrue(call_args.kwargs["enable_thinking"])
    
    def test_integration_with_factory(self):
        """Test MTBench101PromptWrapper works correctly when created via factory"""
        wrapper = get_benchmark_prompt_wrapper("mtbench101", "GR", self.mock_prompt_format)
        
        self.assertIsInstance(wrapper, MTBench101PromptWrapper)
        self.assertEqual(wrapper.task_type, "GR")
        
        # Test it can be called
        result = wrapper(["Solve this problem"], ["Here's my approach"])
        self.assertEqual(result, "formatted_prompt")


if __name__ == '__main__':
    unittest.main()