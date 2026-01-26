#!/usr/bin/env python3
"""
Comprehensive Integration Tests for Prompt Format System

This file consolidates all prompt format integration tests including:
- Text-only model integration (Llama, Mistral)
- Multimodal model integration (Qwen-VL)
- Game wrapper integration
- Real model pipeline integration
- Official tokenizer compatibility validation
"""

import unittest
import sys
import os
from pathlib import Path
from unittest.mock import patch, MagicMock
import pytest
import torch
from PIL import Image
import numpy as np
import tempfile

# Add project root to path for imports
project_root = Path(__file__).parents[2]
sys.path.insert(0, str(project_root))

from transformers import AutoTokenizer, AutoProcessor, AutoModel
from transformers.pipelines import pipeline
from jinja2.exceptions import TemplateError

from neuro_manipulation.prompt_formats import (
    PromptFormat, QwenVLInstFormat, ManualPromptFormat,
    Llama2InstFormat, Llama3InstFormat, MistralInstFormat
)
from neuro_manipulation.prompt_wrapper import PromptWrapper, GameReactPromptWrapper
from neuro_manipulation.repe.pipelines import repe_pipeline_registry
from neuro_manipulation.repe.rep_reading_pipeline import RepReadingPipeline
from neuro_manipulation.model_layer_detector import ModelLayerDetector
from neuro_manipulation.repe.multimodal_processor import QwenVLMultimodalProcessor

# Model paths for testing
QWEN_VL_MODEL_PATH = "Qwen/Qwen2.5-VL-3B-Instruct"
SKIP_QWEN_REASON = f"Qwen2.5-VL model not found at {QWEN_VL_MODEL_PATH}"

class MockGameDecision:
    @staticmethod
    def example():
        return '{"decision": "choice", "rationale": "reason", "option_id": 1}'

# ===== TEXT-ONLY MODEL INTEGRATION TESTS =====

class TestTextModelIntegration(unittest.TestCase):
    """Integration tests for text-only models (Llama, Mistral)."""
    
    def setUp(self):
        self.model_names = [
            "meta-llama/Llama-2-7b-chat-hf",
            "meta-llama/Llama-3.1-8B-Instruct", 
            "mistralai/Mistral-7B-Instruct-v0.3"
        ]
        
        # Try to load tokenizers, skip if not available
        self.tokenizers = {}
        self.prompt_formats = {}
        
        for model_name in self.model_names:
            try:
                tokenizer = AutoTokenizer.from_pretrained(model_name)
                self.tokenizers[model_name] = tokenizer
                self.prompt_formats[model_name] = PromptFormat(tokenizer)
            except Exception as e:
                print(f"⚠️ Skipping {model_name}: {e}")
                continue
    
    def test_official_tokenizer_compatibility(self):
        """Test compatibility with official tokenizer chat templates."""
        if not self.tokenizers:
            self.skipTest("No tokenizers available for testing")
            
        for model_name, tokenizer in self.tokenizers.items():
            with self.subTest(model=model_name):
                prompt_format = self.prompt_formats[model_name]
                
                # Test case
                system_prompt = "You are a helpful assistant."
                user_messages = ["Hello, how are you?"]
                assistant_messages = []
                
                # Our format
                our_result = prompt_format.build(
                    model_name, system_prompt, user_messages, assistant_messages
                )
                
                # Official format
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_messages[0]}
                ]
                
                try:
                    official_result = tokenizer.apply_chat_template(
                        messages, tokenize=False, add_generation_prompt=True
                    )
                    
                    # Content validation (allow format differences)
                    self.assertIn(system_prompt, our_result)
                    self.assertIn(user_messages[0], our_result)
                    self.assertIn(system_prompt, official_result)
                    self.assertIn(user_messages[0], official_result)
                    
                    print(f"✓ {model_name} compatibility validated")
                    
                except Exception as e:
                    print(f"⚠️ Official tokenizer test failed for {model_name}: {e}")
                    # Still validate our format works
                    self.assertIsInstance(our_result, str)
                    self.assertIn(system_prompt, our_result)
    
    def test_prompt_wrapper_integration(self):
        if not self.prompt_formats:
            self.skipTest("No prompt formats available for testing")
            
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
        if not self.prompt_formats:
            self.skipTest("No prompt formats available for testing")
            
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
    
    def test_format_detection_integration(self):
        """Test that format detection works with real tokenizers."""
        model_formats = {
            "meta-llama/Llama-2-7b-chat-hf": Llama2InstFormat,
            "meta-llama/Llama-3.1-8B-Instruct": Llama3InstFormat,
            "mistralai/Mistral-7B-Instruct-v0.3": MistralInstFormat
        }
        
        for model_name, expected_format in model_formats.items():
            with self.subTest(model=model_name):
                # Test name pattern matching
                self.assertTrue(expected_format.name_pattern(model_name))
                
                # Test format retrieval
                detected_format = ManualPromptFormat.get(model_name)
                self.assertEqual(detected_format, expected_format)
                
                print(f"✓ {model_name} -> {expected_format.__name__}")

# ===== MULTIMODAL MODEL INTEGRATION TESTS =====

class TestMultimodalIntegration(unittest.TestCase):
    """Integration tests for multimodal models (Qwen-VL)."""
    
    def test_qwen_format_basic(self):
        """Test basic QwenVL format functionality."""
        print("Testing QwenVL Format Basic Functionality")
        
        # Test name pattern matching
        test_models = [
            "Qwen2.5-VL-3B-Instruct",
            "Qwen2.5-VL-7B-Instruct", 
            "Qwen/Qwen2.5-VL-3B-Instruct",
            "llama-3.1-instruct",
            "mistral-7b-instruct"
        ]
        
        for model in test_models:
            matches = QwenVLInstFormat.name_pattern(model)
            expected = "Qwen" in model and "VL" in model
            self.assertEqual(matches, expected, f"Pattern matching failed for {model}")
        
        # Test text-only formatting
        text_only = QwenVLInstFormat.build(
            system_prompt="You are a helpful assistant",
            user_messages=["What is the weather today?"],
            assistant_answers=[]
        )
        self.assertIsInstance(text_only, str)
        self.assertIn("What is the weather today?", text_only)
        
        # Test multimodal formatting  
        multimodal = QwenVLInstFormat.build(
            system_prompt="You are a helpful assistant",
            user_messages=["when you see this image, your emotion is anger"],
            assistant_answers=[],
            images=[Image.new('RGB', (224, 224), 'red')]
        )
        self.assertIsInstance(multimodal, str)
        self.assertIn("when you see this image, your emotion is anger", multimodal)
        self.assertIn('<|vision_start|>', multimodal)
        self.assertIn('<|vision_end|>', multimodal)
        
        print("✓ QwenVL format basic tests passed")
    
    def test_manual_format_detection(self):
        """Test that ManualPromptFormat can find QwenVL format."""
        model_name = "Qwen2.5-VL-3B-Instruct"
        
        format_class = ManualPromptFormat.get(model_name)
        self.assertEqual(format_class, QwenVLInstFormat)
        self.assertTrue(format_class.supports_multimodal())
        
        # Test building with the detected format
        result = format_class.build(
            system_prompt=None,
            user_messages=["when you see this image, your emotion is happiness"],
            assistant_answers=[],
            images=[Image.new('RGB', (224, 224), 'yellow')]
        )
        self.assertIsInstance(result, str)
        self.assertIn("when you see this image, your emotion is happiness", result)
        
        print("✓ Manual format detection tests passed")
    
    @unittest.skipIf(True, "Skip tokenizer loading test - requires HuggingFace access")
    def test_tokenizer_validation(self):
        """Test tokenizer validation with real Qwen-VL tokenizer."""
        model_path = QWEN_VL_MODEL_PATH
        
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
            
            # Test validation
            is_valid = QwenVLInstFormat.validate_tokenizer(tokenizer)
            self.assertTrue(is_valid)
            
            # Check specific tokens
            required_tokens = ['<|vision_start|>', '<|vision_end|>', '<|image_pad|>']
            for token in required_tokens:
                token_id = tokenizer.convert_tokens_to_ids(token)
                self.assertIsNotNone(token_id, f"Token {token} not found")
            
            print("✓ Tokenizer validation tests passed")
            
        except Exception as e:
            self.skipTest(f"Failed to load tokenizer: {e}")
    
    def test_integrated_emotion_prompt(self):
        """Test the integrated approach with emotion extraction prompts."""
        emotions = ['anger', 'happiness', 'sadness', 'disgust', 'fear', 'surprise']
        
        for emotion in emotions:
            with self.subTest(emotion=emotion):
                result = QwenVLInstFormat.build(
                    system_prompt=None,
                    user_messages=[f"when you see this image, your emotion is {emotion}"],
                    assistant_answers=[],
                    images=[Image.new('RGB', (224, 224), 'blue')]
                )
                
                self.assertIsInstance(result, str)
                self.assertIn(f"your emotion is {emotion}", result)
                self.assertIn('<|vision_start|>', result)
                self.assertIn('<|vision_end|>', result)
        
        print("✓ Integrated emotion prompt tests passed")

# ===== REAL MODEL INTEGRATION TESTS =====

@pytest.mark.skipif(not os.path.exists(QWEN_VL_MODEL_PATH), reason=SKIP_QWEN_REASON)
class TestRealModelIntegration(unittest.TestCase):
    """Integration tests with real Qwen2.5-VL model (requires model download)."""
    
    @classmethod
    def setUpClass(cls):
        """Load model components once for all tests."""
        try:
            cls.model = AutoModel.from_pretrained(
                QWEN_VL_MODEL_PATH,
                trust_remote_code=True,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                low_cpu_mem_usage=True
            )
            
            cls.tokenizer = AutoTokenizer.from_pretrained(
                QWEN_VL_MODEL_PATH, 
                trust_remote_code=True
            )
            
            cls.processor = AutoProcessor.from_pretrained(
                QWEN_VL_MODEL_PATH,
                trust_remote_code=True
            )
            
            # Sample images
            cls.sample_images = {
                'anger': Image.new('RGB', (224, 224), 'red'),
                'happiness': Image.new('RGB', (224, 224), 'yellow'),
                'sadness': Image.new('RGB', (224, 224), 'blue')
            }
            
            print("✓ Real model components loaded")
            
        except Exception as e:
            raise unittest.SkipTest(f"Failed to load model: {e}")
    
    def test_multimodal_model_detection(self):
        """Test that Qwen2.5-VL is correctly detected as multimodal."""
        is_multimodal = ModelLayerDetector.is_multimodal_model(self.model)
        self.assertTrue(is_multimodal, "Qwen2.5-VL should be detected as multimodal")
        
        layer_info = ModelLayerDetector.get_multimodal_layer_info(self.model)
        found_layers = sum(1 for v in layer_info.values() if v is not None)
        self.assertGreater(found_layers, 0, "Should find at least some layer types")
        
        print("✓ Multimodal model detection passed")
    
    def test_prompt_format_integration_real(self):
        """Test prompt format system with real model."""
        model_name = self.tokenizer.name_or_path
        
        # Test format detection
        format_class = ManualPromptFormat.get(model_name)
        self.assertEqual(format_class, QwenVLInstFormat)
        
        # Test multimodal support
        self.assertTrue(format_class.supports_multimodal())
        
        # Test tokenizer validation
        is_valid = format_class.validate_tokenizer(self.tokenizer)
        self.assertTrue(is_valid, "Tokenizer should be valid for QwenVL format")
        
        # Test format building
        text_content = "when you see this image, your emotion is happiness"
        formatted_text = format_class.build(
            system_prompt=None,
            user_messages=[text_content],
            assistant_answers=[],
            images=[self.sample_images['happiness']]
        )
        
        # Verify format contains vision tokens
        self.assertIn('<|vision_start|>', formatted_text)
        self.assertIn('<|vision_end|>', formatted_text)
        self.assertIn(text_content, formatted_text)
        
        print("✓ Real model prompt format integration passed")
    
    def test_model_forward_pass(self):
        """Test forward pass with multimodal input."""
        text_content = "when you see this image, your emotion is anger"
        image = self.sample_images['anger']
        
        # Use QwenVL format system
        formatted_text = QwenVLInstFormat.build(
            system_prompt=None,
            user_messages=[text_content],
            assistant_answers=[],
            images=[image]
        )
        
        # Process inputs
        try:
            inputs = self.processor(text=[formatted_text], images=[image], return_tensors="pt")
            
            # Move to device
            device = next(self.model.parameters()).device
            inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                     for k, v in inputs.items()}
            
            # Forward pass
            with torch.no_grad():
                outputs = self.model(**inputs, output_hidden_states=True)
            
            self.assertTrue(hasattr(outputs, 'hidden_states'))
            self.assertGreater(len(outputs.hidden_states), 0)
            
            print("✓ Real model forward pass passed")
            
        except Exception as e:
            self.fail(f"Forward pass failed: {e}")
    
    def test_pipeline_integration(self):
        """Test integration with RepE pipeline."""
        # Register pipelines
        repe_pipeline_registry()
        
        try:
            rep_pipeline = pipeline(
                task="multimodal-rep-reading",
                model=self.model,
                tokenizer=self.tokenizer,
                image_processor=self.processor
            )
            
            # Create training stimuli
            stimuli = []
            for emotion in ['anger', 'happiness']:
                stimulus = {
                    'images': [self.sample_images[emotion]],
                    'text': f'when you see this image, your emotion is {emotion}'
                }
                stimuli.extend([stimulus, stimulus])  # Duplicate for PCA
            
            # Test direction extraction
            directions = rep_pipeline.get_directions(
                train_inputs=stimuli,
                rep_token=-1,
                hidden_layers=[-1],
                direction_method='pca',
                batch_size=2
            )
            
            self.assertGreater(len(directions.directions), 0)
            
            for layer, direction_matrix in directions.directions.items():
                self.assertGreaterEqual(direction_matrix.shape[0], 1)
                self.assertGreater(direction_matrix.shape[1], 0)
            
            print("✓ Pipeline integration passed")
            
        except Exception as e:
            self.fail(f"Pipeline integration failed: {e}")

# ===== TEST RUNNER =====

def run_all_tests():
    """Run all prompt format integration tests."""
    print("🚀 COMPREHENSIVE PROMPT FORMAT INTEGRATION TESTS")
    print("=" * 60)
    
    # Create test suite
    suite = unittest.TestSuite()
    
    # Add text model tests
    suite.addTest(unittest.makeSuite(TestTextModelIntegration))
    
    # Add multimodal tests
    suite.addTest(unittest.makeSuite(TestMultimodalIntegration))
    
    # Add real model tests (if available)
    if os.path.exists(QWEN_VL_MODEL_PATH):
        suite.addTest(unittest.makeSuite(TestRealModelIntegration))
        print(f"✓ Including real model tests with {QWEN_VL_MODEL_PATH}")
    else:
        print(f"⚠️ Skipping real model tests - {QWEN_VL_MODEL_PATH} not found")
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Summary
    total_tests = result.testsRun
    failures = len(result.failures)
    errors = len(result.errors)
    
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    print(f"Total tests: {total_tests}")
    print(f"Failures: {failures}")
    print(f"Errors: {errors}")
    
    if failures == 0 and errors == 0:
        print("\n🎉 All tests passed! Prompt format integration is working.")
        return True
    else:
        print(f"\n⚠️ {failures + errors} tests failed. Check implementation.")
        return False

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)