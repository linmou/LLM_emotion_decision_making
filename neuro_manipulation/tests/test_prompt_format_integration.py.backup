import unittest
from transformers import AutoTokenizer
from neuro_manipulation.prompt_formats import PromptFormat
from neuro_manipulation.prompt_wrapper import PromptWrapper, GameReactPromptWrapper

class MockGameDecision:
    @staticmethod
    def example():
        return '{"decision": "choice", "rationale": "reason", "option_id": 1}'

class TestPromptFormatIntegration(unittest.TestCase):
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
        self.prompt_formats = {
            model_name: PromptFormat(tokenizer)
            for model_name, tokenizer in self.tokenizers.items()
        }
        
    def test_prompt_wrapper_integration(self):
        for model_name, prompt_format in self.prompt_formats.items():
            with self.subTest(model=model_name):
                wrapper = PromptWrapper(prompt_format)
                event = "You are at a party"
                options = ["Go talk to people", "Stay in a corner"]
                user_messages = "What should I do?"
                
                prompt = wrapper(event, options, user_messages)
                
                # Basic validation - note that 'event' may not appear directly in the prompt
                self.assertIsInstance(prompt, str)
                # Check parts of the options and user message instead
                for option in options:
                    self.assertIn(option, prompt)
                self.assertIn(user_messages, prompt)
        
    def test_game_react_prompt_wrapper_integration(self):
        for model_name, prompt_format in self.prompt_formats.items():
            with self.subTest(model=model_name):
                wrapper = GameReactPromptWrapper(prompt_format, MockGameDecision)
                event = "You are playing a game"
                options = ["Cooperate", "Defect"]
                user_messages = "What's your choice?"
                emotion = "angry"
                prompt = wrapper(event, options, emotion, user_messages)
                
                # Basic validation - the event should be in the prompt for GameReactPromptWrapper 
                # because its system_prompt_format includes the {event} directly
                self.assertIsInstance(prompt, str)
                self.assertIn(event, prompt)
                for option in options:
                    self.assertIn(option, prompt)
                self.assertIn(user_messages, prompt)
                self.assertIn('{"decision": "choice", "rationale": "reason", "option_id": 1}', prompt)

if __name__ == "__main__":
    unittest.main() 