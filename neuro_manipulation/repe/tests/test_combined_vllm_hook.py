import unittest
import sys
import os
import torch
import gc

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Attempt to import required modules, skipping test if they are not available
try:
    from vllm import LLM
    from transformers import AutoTokenizer
    from sequence_prob_vllm_hook import CombinedVLLMHook
    VLLM_AVAILABLE = True
except ImportError:
    VLLM_AVAILABLE = False

@unittest.skipIf(not VLLM_AVAILABLE, "vLLM, transformers is not installed")
class TestCombinedVLLMHook(unittest.TestCase):
    """
    Test suite for the new CombinedVLLMHook functionality.
    Tests sequence probability, representation control, and layer recording.
    """
    
    @classmethod
    def setUpClass(cls):
        """Initialize model and tokenizer for testing, once for the entire class."""
        cls.model_name = "gpt2"  # Use smaller model for testing
        cls.model = LLM(
            model=cls.model_name, 
            tensor_parallel_size=1, 
            gpu_memory_utilization=0.4, 
            max_num_seqs=8,
            enforce_eager=True  # Required for hooks to work
        )
        cls.tokenizer = AutoTokenizer.from_pretrained(cls.model_name)
        if cls.tokenizer.pad_token is None:
            cls.tokenizer.pad_token = cls.tokenizer.eos_token
        
        # Test configuration
        cls.test_prompt = "The capital of France is"
        cls.test_targets = ["Paris", "London"]
        cls.test_layers = [5, 8]  # GPT2 has 12 layers, so these should be valid
        cls.hidden_dim = 768  # GPT2 hidden dimension

    @classmethod
    def tearDownClass(cls):
        """Clean up resources after all tests are done."""
        del cls.model
        del cls.tokenizer
        torch.cuda.empty_cache()
        gc.collect()

    def setUp(self):
        """Set up for each test."""
        # Clear CUDA cache before each test to prevent memory issues
        torch.cuda.empty_cache()

    def test_sequence_probability_only(self):
        """Test sequence probability calculation functionality only."""
        hook = CombinedVLLMHook(
            self.model, 
            self.tokenizer,
            enable_sequence_prob=True,
            enable_rep_control=False,
            enable_layer_logit_recording=False
        )
        
        # Test basic sequence probability calculation
        results = hook.get_log_prob([self.test_prompt], self.test_targets)
        
        # Should get at least one result
        self.assertGreaterEqual(len(results), 1, "Should return at least one probability result")
        
        # Check result structure
        for result in results:
            self.assertIsInstance(result, dict, "Each result should be a dictionary")
            expected_keys = {'sequence', 'log_prob', 'prob', 'perplexity', 'num_tokens'}
            self.assertEqual(set(result.keys()), expected_keys, "Result should contain expected keys")
            self.assertIn(result['sequence'], self.test_targets, "Sequence should be one of the targets")
            self.assertIsInstance(result['log_prob'], float, "log_prob should be a float")
            self.assertIsInstance(result['prob'], float, "prob should be a float")
            self.assertIsInstance(result['perplexity'], float, "perplexity should be a float")

    def test_representation_control_only(self):
        """Test representation control functionality only."""
        hook = CombinedVLLMHook(
            self.model,
            self.tokenizer,
            layers=self.test_layers,
            block_name="decoder_block",
            enable_sequence_prob=False,
            enable_rep_control=True,
            enable_layer_logit_recording=False
        )
        
        # Create control vectors
        control_activations = {}
        for layer_id in self.test_layers:
            control_activations[layer_id] = torch.randn(self.hidden_dim, dtype=torch.float16) * 0.1
        
        # Test baseline generation (no control)
        baseline_outputs = hook.generate_with_control(
            [self.test_prompt],
            max_new_tokens=5,
            temperature=0.0
        )
        self.assertEqual(len(baseline_outputs), 1, "Should generate one output")
        baseline_text = baseline_outputs[0].outputs[0].text
        
        # Test controlled generation
        controlled_outputs = hook.generate_with_control(
            [self.test_prompt],
            activations=control_activations,
            operator='linear_comb',
            normalize=False,
            max_new_tokens=5,
            temperature=0.0
        )
        self.assertEqual(len(controlled_outputs), 1, "Should generate one controlled output")
        controlled_text = controlled_outputs[0].outputs[0].text
        
        # Both should be valid strings (may or may not be different due to small control magnitude)
        self.assertIsInstance(baseline_text, str, "Baseline output should be a string")
        self.assertIsInstance(controlled_text, str, "Controlled output should be a string")

    def test_layer_recording_only(self):
        """Test layer-wise logit recording functionality only."""
        hook = CombinedVLLMHook(
            self.model,
            self.tokenizer,
            layers=self.test_layers,
            enable_sequence_prob=False,
            enable_rep_control=False,
            enable_layer_logit_recording=True
        )
        
        # Generate with recording enabled
        outputs = hook.generate_with_control(
            [self.test_prompt],
            record_layer_logits=True,
            max_new_tokens=3,
            temperature=0.0
        )
        
        self.assertEqual(len(outputs), 1, "Should generate one output")
        
        # Retrieve layer recordings
        layer_logits = hook.get_layer_logits()
        
        # Should have recordings for each layer
        self.assertEqual(set(layer_logits.keys()), set(self.test_layers), 
                         "Should have recordings for all specified layers")
        
        # Check recording structure
        for layer_id, recordings in layer_logits.items():
            self.assertIsInstance(recordings, list, f"Layer {layer_id} recordings should be a list")
            if recordings:  # If any recordings were captured
                for recording in recordings:
                    self.assertIsInstance(recording, dict, "Each recording should be a dictionary")
                    self.assertIn('layer_id', recording, "Recording should have layer_id")
                    self.assertIn('activations', recording, "Recording should have activations")
                    self.assertIn('shape', recording, "Recording should have shape")

    def test_combined_functionality(self):
        """Test all features working together."""
        hook = CombinedVLLMHook(
            self.model,
            self.tokenizer,
            layers=self.test_layers,
            enable_sequence_prob=True,
            enable_rep_control=True,
            enable_layer_logit_recording=True
        )
        
        # Test sequence probability calculation
        prob_results = hook.get_log_prob([self.test_prompt], self.test_targets)
        self.assertGreaterEqual(len(prob_results), 1, "Should get probability results")
        
        # Create control vectors
        control_activations = {}
        for layer_id in self.test_layers:
            control_activations[layer_id] = torch.randn(self.hidden_dim, dtype=torch.float16) * 0.05
        
        # Test controlled generation with recording
        outputs = hook.generate_with_control(
            [self.test_prompt],
            activations=control_activations,
            record_layer_logits=True,
            operator='linear_comb',
            max_new_tokens=3,
            temperature=0.0
        )
        
        self.assertEqual(len(outputs), 1, "Should generate one output")
        
        # Retrieve layer recordings
        layer_logits = hook.get_layer_logits()
        self.assertEqual(set(layer_logits.keys()), set(self.test_layers), 
                         "Should have recordings for all layers")



    def test_error_handling(self):
        """Test error handling for various edge cases."""
        # Test feature disabled errors
        hook = CombinedVLLMHook(
            self.model,
            self.tokenizer,
            enable_sequence_prob=False,
            enable_rep_control=False,
            enable_layer_logit_recording=False
        )
        
        with self.assertRaises(RuntimeError):
            hook.get_log_prob([self.test_prompt], self.test_targets)
        
        with self.assertRaises(RuntimeError):
            hook.get_layer_logits()
            
        # Test using disabled features in generate_with_control
        with self.assertRaises(RuntimeError):
            hook.generate_with_control([self.test_prompt], activations={5: torch.randn(768)})
            
        with self.assertRaises(RuntimeError):
            hook.generate_with_control([self.test_prompt], record_layer_logits=True)

    def test_different_operators(self):
        """Test different representation control operators."""
        hook = CombinedVLLMHook(
            self.model,
            self.tokenizer,
            layers=self.test_layers,
            enable_rep_control=True,
            enable_sequence_prob=False,
            enable_layer_logit_recording=False
        )
        
        control_activations = {}
        for layer_id in self.test_layers:
            control_activations[layer_id] = torch.randn(self.hidden_dim, dtype=torch.float16) * 0.1
        
        # Test linear_comb operator
        outputs_linear = hook.generate_with_control(
            [self.test_prompt],
            activations=control_activations,
            operator='linear_comb',
            max_new_tokens=3
        )
        self.assertEqual(len(outputs_linear), 1, "Linear combination operator should work")
        
        # Test piecewise_linear operator
        outputs_piecewise = hook.generate_with_control(
            [self.test_prompt],
            activations=control_activations,
            operator='piecewise_linear',
            max_new_tokens=3
        )
        self.assertEqual(len(outputs_piecewise), 1, "Piecewise linear operator should work")

    def test_token_position_control(self):
        """Test different token position controls."""
        hook = CombinedVLLMHook(
            self.model,
            self.tokenizer,
            layers=self.test_layers,
            enable_rep_control=True,
            enable_sequence_prob=False,
            enable_layer_logit_recording=False
        )
        
        control_activations = {}
        for layer_id in self.test_layers:
            control_activations[layer_id] = torch.randn(self.hidden_dim, dtype=torch.float16) * 0.1
        
        # Test different token positions
        for token_pos in [None, 'start', 'end', 0]:
            with self.subTest(token_pos=token_pos):
                outputs = hook.generate_with_control(
                    [self.test_prompt],
                    activations=control_activations,
                    token_pos=token_pos,
                    max_new_tokens=2
                )
                self.assertEqual(len(outputs), 1, f"Token position {token_pos} should work")


if __name__ == '__main__':
    unittest.main() 