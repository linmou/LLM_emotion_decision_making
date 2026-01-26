import unittest
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Attempt to import required modules, skipping test if they are not available
try:
    import torch
    from vllm import LLM
    from transformers import AutoTokenizer
    from sequence_prob_vllm_hook import SequenceProbVLLMHook
    VLLM_AVAILABLE = True
except ImportError:
    VLLM_AVAILABLE = False

@unittest.skipIf(not VLLM_AVAILABLE, "vLLM, transformers is not installed")
class TestSequenceProbFix(unittest.TestCase):
    """
    Test to verify that the sequence probability fix is working correctly.
    """
    
    @classmethod
    def setUpClass(cls):
        """Initialize model and tokenizer for testing, once for the entire class."""
        cls.model_name = "gpt2"
        cls.model = LLM(model=cls.model_name, tensor_parallel_size=1, gpu_memory_utilization=0.5, max_num_seqs=16)
        cls.tokenizer = AutoTokenizer.from_pretrained(cls.model_name)
        cls.hook = SequenceProbVLLMHook(cls.model, cls.tokenizer)

    @classmethod
    def tearDownClass(cls):
        """Clean up resources after all tests are done."""
        del cls.model
        del cls.tokenizer
        del cls.hook
        torch.cuda.empty_cache()

    def test_sequence_probability_logic(self):
        """
        Comprehensive test for sequence probability logic.
        This single test case covers:
        1. Basic functionality with a single target.
        2. Multiple targets to ensure correct iteration.
        3. Token encoding variants (with/without leading space).
        """
        # --- Test 1: Basic functionality ---
        prompts = ["The capital of France is"]
        targets = ["Paris"]
        results = self.hook.get_log_prob(prompts, targets)
        
        self.assertEqual(len(results), 1, "Should return exactly one result for 'Paris'")
        result = results[0]
        self.assertIsInstance(result, dict, "Result for 'Paris' should be a dictionary")
        expected_keys = {'sequence', 'log_prob', 'prob', 'perplexity', 'num_tokens'}
        self.assertEqual(set(result.keys()), expected_keys, "Result for 'Paris' should contain expected keys")
        self.assertEqual(result['sequence'], 'Paris', "Sequence should match 'Paris'")
        self.assertIsInstance(result['log_prob'], float, "log_prob for 'Paris' should be a float")

        # --- Test 2: Multiple targets (some may not be calculable) ---
        targets_multi = ["Paris", "London", "Berlin"]
        results_multi = self.hook.get_log_prob(prompts, targets_multi)
        
        # At least one result should be returned (Paris should be calculable)
        self.assertGreaterEqual(len(results_multi), 1, "Should return at least one result")
        # All returned results should be valid
        for i, result in enumerate(results_multi):
            self.assertIn(result['sequence'], targets_multi, f"Result {i} should match one of the target cities")
            self.assertIsInstance(result['log_prob'], float, f"Result {i} log_prob should be a float")

        # --- Test 3: Token encoding variants ---
        prompts_encoding = ["The"]
        targets_encoding = ["the", "first", "second"]  # More likely words after "The"
        results_encoding = self.hook.get_log_prob(prompts_encoding, targets_encoding)
        
        self.assertGreaterEqual(len(results_encoding), 1, "Should find at least one word despite encoding variants")
        self.assertIsInstance(results_encoding[0]['log_prob'], float, "Should return a valid log_prob")

        # --- Test 4: Test with more likely sequences ---
        prompts_likely = ["The capital"]
        targets_likely = ["of", "city", "is"]  # More likely words after "The capital"
        results_likely = self.hook.get_log_prob(prompts_likely, targets_likely)
        
        # These common words should be calculable
        self.assertGreaterEqual(len(results_likely), 1, "Should return at least one result for common words")
        for result in results_likely:
            self.assertIn(result['sequence'], targets_likely, "Result should match one of the target words")
            self.assertIsInstance(result['log_prob'], float, "log_prob should be a float")


if __name__ == '__main__':
    unittest.main() 