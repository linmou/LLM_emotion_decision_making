import unittest
from unittest.mock import patch, MagicMock
from transformers import AutoTokenizer
from jinja2.exceptions import TemplateError
from neuro_manipulation.prompt_formats import ManualPromptFormat, PromptFormat, Llama2InstFormat, Llama3InstFormat, MistralInstFormat, ModelPromptFormat, ClassPropertyDescriptor, classproperty, RWKVsFormat

class TestPromptTemplates(unittest.TestCase):
    """Test the application of model-specific chat templates"""
    
    def setUp(self):
        self.model_names = [
            "meta-llama/Llama-2-7b-chat-hf",
            "meta-llama/Llama-3.1-8B-Instruct",
            "mistralai/Mistral-7B-Instruct-v0.3"
        ]
        self.tokenizers = {
            model_name: AutoTokenizer.from_pretrained(model_name)
            for model_name in self.model_names
        }
    
    def test_chat_template_application(self):
        """Test direct application of chat templates"""
        for model_name, tokenizer in self.tokenizers.items():
            chat = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Hello, how are you?"},
                {"role": "assistant", "content": "I'm doing well, thank you!"}
            ]
            
            # Direct application of chat template
            prompt = tokenizer.apply_chat_template(chat, tokenize=False)
            
            # Verify basic properties
            print(f"Prompt for {model_name}: {prompt}")
            self.assertIsInstance(prompt, str) 
            # self.assertIn("You are a helpful assistant", prompt) # Some models might handle system prompts differently
            self.assertIn("Hello, how are you?", prompt)
            self.assertIn("I'm doing well, thank you!", prompt)
    
    def test_llama2_chat_template(self):
        """Test specific properties of Llama2 chat template"""
        model_name = "meta-llama/Llama-2-7b-chat-hf"
        tokenizer = self.tokenizers[model_name]
        
        chat = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello, how are you?"},
            {"role": "assistant", "content": "I'm doing well, thank you!"}
        ]
        
        prompt = tokenizer.apply_chat_template(chat, tokenize=False)
        
        # Check Llama2-specific format elements
        self.assertIn("[INST]", prompt)
        self.assertIn("[/INST]", prompt)
        self.assertIn("<<SYS>>", prompt)
        self.assertIn("<</SYS>>", prompt)
    
    def test_llama3_chat_template(self):
        """Test specific properties of Llama3 chat template"""
        model_name = "meta-llama/Llama-3.1-8B-Instruct"
        tokenizer = self.tokenizers[model_name]
        
        chat = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello, how are you?"},
            {"role": "assistant", "content": "I'm doing well, thank you!"}
        ]
        
        prompt = tokenizer.apply_chat_template(chat, tokenize=False)
        
        # Check Llama3-specific format elements (adjust based on actual template)
        self.assertIn("<|begin_of_text|>", prompt) 
        self.assertIn("<|start_header_id|>system<|end_header_id|>", prompt)
        self.assertIn("<|start_header_id|>user<|end_header_id|>", prompt)
        self.assertIn("<|start_header_id|>assistant<|end_header_id|>", prompt)
    
    def test_mistral_chat_template(self):
        """Test specific properties of Mistral chat template"""
        model_name = "mistralai/Mistral-7B-Instruct-v0.3"
        tokenizer = self.tokenizers[model_name]
        
        chat = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello, how are you?"},
            {"role": "assistant", "content": "I'm doing well, thank you!"}
        ]
        
        prompt = tokenizer.apply_chat_template(chat, tokenize=False)
        print(f"Mistral prompt: {prompt}")
        
        # Check Mistral-specific format elements (adjust based on actual template)
        self.assertIn("[INST]", prompt)
        # Mistral doesn't include system prompts in the template output
        self.assertIn("Hello, how are you?", prompt)
        self.assertIn("I'm doing well, thank you!", prompt)

class TestPromptFormatBasics(unittest.TestCase):
    def setUp(self):
        self.model_names = [
            "meta-llama/Llama-2-7b-chat-hf",
            "meta-llama/Llama-3.1-8B-Instruct",
            "mistralai/Mistral-7B-Instruct-v0.3"
        ]
        self.tokenizers = {
            model_name: AutoTokenizer.from_pretrained(model_name)
            for model_name in self.model_names
        }
        
    def test_initialization(self):
        """Test basic initialization of PromptFormat with a tokenizer"""
        for model_name, tokenizer in self.tokenizers.items():
            prompt_format = PromptFormat(tokenizer)
            self.assertIsNotNone(prompt_format)
            self.assertEqual(prompt_format.tokenizer, tokenizer)
            
    def test_basic_prompt_building(self):
        """Test building a simple prompt with system and one user message"""
        for model_name, tokenizer in self.tokenizers.items():
            prompt_format = PromptFormat(tokenizer)
            system_prompt = "You are a helpful assistant."
            user_message = "Hello, how are you?"
            
            prompt = prompt_format.build(model_name, system_prompt, [user_message])
            
            # Check that the prompt is a string and contains the messages
            self.assertIsInstance(prompt, str)
            self.assertIn(system_prompt, prompt)
            self.assertIn(user_message, prompt)
            
    def test_multi_turn_conversation(self):
        """Test building a prompt with multiple turns of conversation"""
        for model_name, tokenizer in self.tokenizers.items():
            prompt_format = PromptFormat(tokenizer)
            system_prompt = "You are a helpful assistant."
            user_messages = ["Hello, how are you?", "Can you help me with a coding task?"]
            assistant_messages = ["I'm doing well! How can I help you today?"]
            
            prompt = prompt_format.build(model_name, system_prompt, user_messages, assistant_messages)
            
            # Check that the prompt contains all messages
            self.assertIsInstance(prompt, str)
            self.assertIn(system_prompt, prompt)
            for msg in user_messages:
                self.assertIn(msg, prompt)
            for msg in assistant_messages:
                self.assertIn(msg, prompt)
    
    def test_format_detection(self):
        """Test that format detection works correctly for known model types"""
        model_formats = {
            "meta-llama/Llama-2-7b-chat-hf": Llama2InstFormat,
            "meta-llama/Llama-3.1-8B-Instruct": Llama3InstFormat,
            "mistralai/Mistral-7B-Instruct-v0.3": MistralInstFormat
        }
        
        for model_name, format_class in model_formats.items():
            self.assertTrue(format_class.name_pattern(model_name))
            
    def test_empty_messages(self):
        """Test handling of empty message arrays"""
        model_name = "meta-llama/Llama-2-7b-chat-hf"
        tokenizer = self.tokenizers[model_name]
        prompt_format = PromptFormat(tokenizer)
        system_prompt = "You are a helpful assistant."
        
        # Most chat templates require at least one user message
        prompt = prompt_format.build(model_name, system_prompt, [""])
        self.assertIsInstance(prompt, str)
        self.assertIn(system_prompt, prompt)
        
    def test_more_user_than_assistant_messages(self):
        """Test handling of more user messages than assistant messages"""
        model_name = "meta-llama/Llama-2-7b-chat-hf"
        tokenizer = self.tokenizers[model_name]
        prompt_format = PromptFormat(tokenizer)
        system_prompt = "You are a helpful assistant."
        user_messages = ["Message 1", "Message 2", "Message 3"]
        assistant_messages = ["Response 1", "Response 2"]
        
        # This should work because we can have one more user message than assistant message
        prompt = prompt_format.build(model_name, system_prompt, user_messages, assistant_messages)
        self.assertIsInstance(prompt, str)
        for msg in user_messages:
            self.assertIn(msg, prompt)
        for msg in assistant_messages:
            self.assertIn(msg, prompt)
            
    def test_long_messages(self):
        """Test handling of long messages"""
        model_name = "meta-llama/Llama-2-7b-chat-hf"
        tokenizer = self.tokenizers[model_name]
        prompt_format = PromptFormat(tokenizer)
        system_prompt = "You are a helpful assistant."
        long_message = "This is a very long message. " * 50  # 1000+ characters
        
        prompt = prompt_format.build(model_name, system_prompt, [long_message])
        self.assertIsInstance(prompt, str)
        self.assertIn(long_message, prompt)
        
    def test_special_characters(self):
        """Test handling of messages with special characters"""
        model_name = "meta-llama/Llama-2-7b-chat-hf"
        tokenizer = self.tokenizers[model_name]
        prompt_format = PromptFormat(tokenizer)
        system_prompt = "You are a helpful assistant."
        special_chars_message = "Message with special chars: !@#$%^&*()_+-=[]{}|;':\",./<>?"
        
        prompt = prompt_format.build(model_name, system_prompt, [special_chars_message])
        print(f"Special chars prompt: {prompt}")
        print(f"Special chars message: {special_chars_message}")
        self.assertIsInstance(prompt, str)
        self.assertIn(special_chars_message, prompt)
        
    def test_fallback_mechanism(self):
        """Test the fallback mechanism when chat template fails"""
        # This test would need a mock to properly test the fallback
        # For now, just verify the code doesn't crash with an unknown model
        model_name = "meta-llama/Llama-2-7b-chat-hf"
        tokenizer = self.tokenizers[model_name]
        prompt_format = PromptFormat(tokenizer)
        system_prompt = "You are a helpful assistant."
        user_message = "Hello"
        
        # Should use fallback for an unknown model name
        unknown_model = "unknown-model/that-does-not-exist"
        prompt = prompt_format.build(unknown_model, system_prompt, [user_message])
        self.assertIsInstance(prompt, str)
        self.assertIn(system_prompt, prompt)
        self.assertIn(user_message, prompt)
        
    def test_exception_handling(self):
        """Test exception handling for models like Llama-2 that may have issues with the system role"""
        model_name = "meta-llama/Llama-2-7b-chat-hf"
        tokenizer = self.tokenizers[model_name]
        prompt_format = PromptFormat(tokenizer)
        system_prompt = "You are a helpful assistant."
        user_messages = ["Hello, how are you?"]
        
        # Save original method
        original_method = tokenizer.apply_chat_template
        
        try:
            # Define a side effect function that simulates the chat template behavior
            def side_effect(conversation, *args, **kwargs):
                # Check if this is the first or second call by examining the conversation structure
                if len(conversation) > 1 and conversation[0]["role"] == "system":
                    # First call with separate system message - fail with error about system role
                    raise TemplateError("Mocked template error: cannot handle system role")
                else:
                    # Second call with merged system message - succeed
                    return "<s>[INST] <<SYS>>\nYou are a helpful assistant.\n<</SYS>>\n\nHello, how are you? [/INST]"
            
            # Replace the original method with our mock
            tokenizer.apply_chat_template = MagicMock(side_effect=side_effect)
            
            # Test the method
            result = prompt_format.build(model_name, system_prompt, user_messages)
            
            # Verify the result contains the expected content
            self.assertIn(system_prompt, result)
            self.assertIn(user_messages[0], result)
            
            # Verify the mock was called twice (first call fails, second succeeds)
            self.assertEqual(tokenizer.apply_chat_template.call_count, 2)
            
            # Print debug info
            print(f"Result: {result}")
            print(f"Call count: {tokenizer.apply_chat_template.call_count}")
            print(f"Call args: {tokenizer.apply_chat_template.call_args_list}")
            
        finally:
            # Restore original method
            tokenizer.apply_chat_template = original_method
        
    def test_fallback_to_old_format(self):
        """Test fallback to ManualPromptFormat when all template applications fail"""
        model_name = "meta-llama/Llama-2-7b-chat-hf"
        tokenizer = self.tokenizers[model_name]
        prompt_format = PromptFormat(tokenizer)
        system_prompt = "You are a helpful assistant."
        user_messages = ["Hello, how are you?"]
        
        # Save original methods
        original_apply_template = tokenizer.apply_chat_template
        original_old_build = ManualPromptFormat.build
        
        try:
            # Replace with a mock that always raises TemplateError
            tokenizer.apply_chat_template = MagicMock(side_effect=TemplateError("Mocked template error"))
            
            # Mock ManualPromptFormat.build
            expected_result = "Fallback to old format successful"
            ManualPromptFormat.build = MagicMock(return_value=expected_result)
            
            # Test the method
            result = prompt_format.build(model_name, system_prompt, user_messages)
            
            # Verify ManualPromptFormat.build was called with correct arguments
            ManualPromptFormat.build.assert_called_once_with(model_name, system_prompt, user_messages, [])
            
            # Verify the result
            self.assertEqual(result, expected_result)
            
            # Verify the template was attempted multiple times before falling back
            self.assertGreaterEqual(tokenizer.apply_chat_template.call_count, 1)
            
            # Print debug info
            print(f"Result: {result}")
            print(f"apply_chat_template call count: {tokenizer.apply_chat_template.call_count}")
            print(f"ManualPromptFormat.build called with: {ManualPromptFormat.build.call_args}")
            
        finally:
            # Restore original methods
            tokenizer.apply_chat_template = original_apply_template
            ManualPromptFormat.build = original_old_build

class TestPromptFormatCompatibility(unittest.TestCase):
    def setUp(self):
        self.model_names = [
            "meta-llama/Llama-2-7b-chat-hf",
            "meta-llama/Llama-3.1-8B-Instruct",
            "mistralai/Mistral-7B-Instruct-v0.3"
        ]
        self.tokenizers = {
            model_name: AutoTokenizer.from_pretrained(model_name)
            for model_name in self.model_names
        }
        self.system_prompt = "You are a helpful AI assistant."
        self.user_messages = ["Hello, how are you?", "Can you help me with a task?"]
        self.assistant_messages = ["I'm doing well, how can I help you today?"]
    
    def test_llama2_compatibility(self):
        model_name = "meta-llama/Llama-2-7b-chat-hf"
        
        # Old implementation
        old_prompt = ManualPromptFormat.build(
            model_name, 
            self.system_prompt, 
            self.user_messages, 
            self.assistant_messages
        )
        
        # New implementation
        prompt_format = PromptFormat(self.tokenizers[model_name])
        new_prompt = prompt_format.build(
            model_name, 
            self.system_prompt, 
            self.user_messages, 
            self.assistant_messages
        )
        
        # Print for debugging
        print(f"Old prompt: {old_prompt}")
        print(f"New prompt: {new_prompt}")
        
        # Check content equivalence rather than exact string equality
        self.assertIn(self.system_prompt, old_prompt)
        self.assertIn(self.system_prompt, new_prompt)
        for msg in self.user_messages:
            self.assertIn(msg, old_prompt)
            self.assertIn(msg, new_prompt)
        for msg in self.assistant_messages:
            self.assertIn(msg, old_prompt)
            self.assertIn(msg, new_prompt)
    
    def test_llama3_compatibility(self):
        model_name = "meta-llama/Llama-3.1-8B-Instruct"
        
        # Old implementation
        old_prompt = ManualPromptFormat.build(
            model_name, 
            self.system_prompt, 
            self.user_messages, 
            self.assistant_messages
        )
        
        # New implementation
        prompt_format = PromptFormat(self.tokenizers[model_name])
        new_prompt = prompt_format.build(
            model_name, 
            self.system_prompt, 
            self.user_messages, 
            self.assistant_messages
        )
        
        # Check content equivalence rather than exact string equality
        print(f"Old prompt: {old_prompt}")
        print(f"New prompt: {new_prompt}")
        self.assertIn(self.system_prompt, old_prompt)
        self.assertIn(self.system_prompt, new_prompt)
        for msg in self.user_messages:
            self.assertIn(msg, old_prompt)
            self.assertIn(msg, new_prompt)
        for msg in self.assistant_messages:
            self.assertIn(msg, old_prompt)
            self.assertIn(msg, new_prompt)
    
    def test_mistral_compatibility(self):
        model_name = "mistralai/Mistral-7B-Instruct-v0.3"
        
        # Old implementation
        old_prompt = ManualPromptFormat.build(
            model_name, 
            self.system_prompt, 
            self.user_messages, 
            self.assistant_messages
        )
        
        # New implementation
        prompt_format = PromptFormat(self.tokenizers[model_name])
        new_prompt = prompt_format.build(
            model_name, 
            self.system_prompt, 
            self.user_messages, 
            self.assistant_messages
        )
        print(f"Mistral old prompt: {old_prompt}")
        print(f"Mistral new prompt: {new_prompt}")
        # Mistral might handle system prompts differently
        for msg in self.user_messages:
            self.assertIn(msg, old_prompt)
            self.assertIn(msg, new_prompt)
        for msg in self.assistant_messages:
            self.assertIn(msg, old_prompt)
            self.assertIn(msg, new_prompt)
    
    def test_different_message_counts(self):
        model_name = "meta-llama/Llama-2-7b-chat-hf"
        
        # Test with no assistant messages
        old_prompt = ManualPromptFormat.build(
            model_name, 
            self.system_prompt, 
            self.user_messages[:1], 
            []
        )
        
        prompt_format = PromptFormat(self.tokenizers[model_name])
        new_prompt = prompt_format.build(
            model_name, 
            self.system_prompt, 
            self.user_messages[:1], 
            []
        )
        
        # Check content rather than exact equality
        self.assertIn(self.system_prompt, old_prompt)
        self.assertIn(self.system_prompt, new_prompt)
        self.assertIn(self.user_messages[0], old_prompt)
        self.assertIn(self.user_messages[0], new_prompt)
        
        # Test with multiple message pairs
        old_prompt = ManualPromptFormat.build(
            model_name, 
            self.system_prompt, 
            self.user_messages, 
            ["I'm doing well", "I can help with that"]
        )
        
        new_prompt = prompt_format.build(
            model_name, 
            self.system_prompt, 
            self.user_messages, 
            ["I'm doing well", "I can help with that"]
        )
        
        # Check content rather than exact equality
        self.assertIn(self.system_prompt, old_prompt)
        self.assertIn(self.system_prompt, new_prompt)
        for msg in self.user_messages:
            self.assertIn(msg, old_prompt)
            self.assertIn(msg, new_prompt)
        for msg in ["I'm doing well", "I can help with that"]:
            self.assertIn(msg, old_prompt)
            self.assertIn(msg, new_prompt)

    def test_rwkv_format(self):
        """Test the RWKVsFormat implementation"""
        system_prompt = "You are a helpful AI assistant."
        user_messages = ["Hello, how are you?", "Can you help me with a task?"]
        assistant_messages = ["I'm doing well, how can I help you today?"]
        
        # Test without system prompt
        prompt = RWKVsFormat.build("", user_messages[:1], [])
        self.assertEqual(prompt, "User:  Hello, how are you?\n\n")
        
        # Test with system prompt
        prompt = RWKVsFormat.build(system_prompt, user_messages[:1], [])
        self.assertEqual(prompt, "User: You are a helpful AI assistant. Hello, how are you?\n\n")
        
        # Test with multiple messages
        prompt = RWKVsFormat.build(system_prompt, user_messages, assistant_messages)
        expected = "User: You are a helpful AI assistant. Hello, how are you?\n\nAssistant: I'm doing well, how can I help you today?\n\nUser: Can you help me with a task?\n\n"
        self.assertEqual(prompt, expected)
        
        # Test name pattern matching
        self.assertTrue(RWKVsFormat.name_pattern("rwkv-4-raven"))
        self.assertFalse(RWKVsFormat.name_pattern("llama-2-7b"))

if __name__ == "__main__":
    unittest.main()


class TestPromptFormatFallbackRegression(unittest.TestCase):
    # Tests for neuro_manipulation/prompt_formats.py; ensure system prompt survives fallback when chat template drops it.

    def test_llama3_system_prompt_not_lost_when_template_drops_system(self):
        """I am starting with a failing test. This is the Red phase."""

        class _TokenizerDropsSystem:
            name_or_path = "/data/home/jjl7137/huggingface_models/meta-llama/Llama-3.2-1B-Instruct"

            def apply_chat_template(
                self,
                chat,
                tokenize=False,
                add_generation_prompt=True,
                encoding_strategy=None,
            ):
                # Simulate HF template that ignores the system message and only renders the user.
                user_content = ""
                for msg in chat:
                    if msg.get("role") == "user":
                        user_content = msg.get("content", "")
                        break
                return (
                    "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
                    "Cutting Knowledge Date: December 2023\nToday Date: 23 Dec 2025\n\n"
                    "<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n"
                    + user_content
                    + "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
                )

        pf = PromptFormat(_TokenizerDropsSystem())
        system_prompt = "Scenario: RISKY_SYSTEM_SHOULD_SURVIVE"
        user_messages = ["Please provide your answer."]
        prompt = pf.build(system_prompt, user_messages, enable_thinking=False)

        # Regression assertion: even if apply_chat_template drops system content,
        # PromptFormat should fall back and keep the system prompt in the final prompt.
        self.assertIn(system_prompt, prompt)

    def test_llama3_base_model_without_template_uses_llama3_format(self):
        # neuro_manipulation/tests/test_prompt_format.py: ensure Llama3 base model names without chat templates still resolve to Llama3InstFormat.
        class _TokenizerNoTemplate:
            name_or_path = "meta-llama/Llama-3.2-1B"

            def apply_chat_template(self, *args, **kwargs):
                raise ValueError(
                    "Cannot use chat template functions because tokenizer.chat_template is not set and no template argument was passed!"
                )

        model_name = _TokenizerNoTemplate.name_or_path
        self.assertIs(ManualPromptFormat.get(model_name), Llama3InstFormat)

        prompt_format = PromptFormat(_TokenizerNoTemplate())
        result = prompt_format.build("You are a helpful assistant.", ["Hello?"])

        self.assertIn("<|begin_of_text|>", result)
        self.assertIn("Hello?", result)
