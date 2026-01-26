import unittest
import torch
import os
import sys

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Attempt to import vllm and ray, skipping test if they are not available
try:
    import ray
    from vllm import LLM
    from transformers import AutoTokenizer
    from sequence_prob_vllm_hook import SequenceProbVLLMHook
    VLLM_AVAILABLE = True
except ImportError:
    VLLM_AVAILABLE = False

MODEL_NAME = "gpt2"

@unittest.skipIf(not VLLM_AVAILABLE, "vLLM, transformers, or ray is not installed")
@unittest.skipIf(torch.cuda.device_count() < 2, "This test requires at least 2 GPUs")
class TestTensorParallelConsistency(unittest.TestCase):
    """
    This test class verifies the numerical consistency of sequence probability calculations
    between a single-GPU setup and a tensor-parallel (multi-GPU) setup.
    """

    def tearDown(self):
        """
        Ensures Ray is shut down after each test to free up GPU resources.
        """
        if VLLM_AVAILABLE and ray.is_initialized():
            ray.shutdown()
        torch.cuda.empty_cache()

    def _run_log_prob_test(self, tensor_parallel_size: int) -> list:
        """
        Helper function to initialize a vLLM model with a specific tensor parallel size,
        run the log probability calculation, and then clean up resources.

        Args:
            tensor_parallel_size: The number of GPUs to use for tensor parallelism.

        Returns:
            The results from the get_log_prob calculation.
        """
        if ray.is_initialized():
            ray.shutdown()
        
        print(f"\n!!! DEBUG: Initializing LLM with tensor_parallel_size={tensor_parallel_size}")
        model = LLM(model=MODEL_NAME, tensor_parallel_size=tensor_parallel_size)
        
        # DEBUG: inspect llm_engine
        print("!!! DEBUG: Inspecting model.llm_engine...")
        print(f"!!! DEBUG: llm_engine type: {type(model.llm_engine)}")
        print(f"!!! DEBUG: has collective_rpc: {hasattr(model.llm_engine, 'collective_rpc')}")
        if hasattr(model.llm_engine, 'parallel_config'):
            print(f"!!! DEBUG: parallel_config: {model.llm_engine.parallel_config}")
            if hasattr(model.llm_engine.parallel_config, 'tensor_parallel_size'):
                print(f"!!! DEBUG: detected TP size: {model.llm_engine.parallel_config.tensor_parallel_size}")

        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        
        print(f"!!! DEBUG: Creating SequenceProbVLLMHook...")
        hook = SequenceProbVLLMHook(model, tokenizer)
        print(f"!!! DEBUG: Hook created, hook_registered: {hook.hook_registered}")
        
        prompts = ["The capital of France is"]
        targets = ["Paris"]
        
        print(f"!!! DEBUG: Calling get_log_prob with prompts={prompts}, targets={targets}")
        results = hook.get_log_prob(prompts, targets)
        print(f"!!! DEBUG: get_log_prob returned: {results}")
        print(f"!!! DEBUG: results type: {type(results)}, length: {len(results) if results else 'None'}")
        
        # Explicitly delete objects and shut down Ray to release GPU memory
        del hook
        del model
        if ray.is_initialized():
            ray.shutdown()
        torch.cuda.empty_cache()

        return results

    def test_tp_consistency(self):
        """
        Tests that the log probability from a tensor-parallel model (tp_size=2)
        is numerically consistent with the log probability from a single-GPU model (tp_size=1).
        """
        # Run with tensor_parallel_size=1
        results_tp1 = self._run_log_prob_test(tensor_parallel_size=1)
        
        # Run with tensor_parallel_size=2
        results_tp2 = self._run_log_prob_test(tensor_parallel_size=2)
        
        self.assertIsNotNone(results_tp1, "TP=1 run failed to produce results.")
        self.assertIsNotNone(results_tp2, "TP=2 run failed to produce results.")
        self.assertEqual(len(results_tp1), 1)
        self.assertEqual(len(results_tp2), 1)
        
        log_prob_tp1 = results_tp1[0]['log_prob']
        log_prob_tp2 = results_tp2[0]['log_prob']
        
        print(f"Log probability (tensor_parallel_size=1): {log_prob_tp1}")
        print(f"Log probability (tensor_parallel_size=2): {log_prob_tp2}")
        
        self.assertAlmostEqual(
            log_prob_tp1, 
            log_prob_tp2, 
            places=4, 
            msg="Log probabilities between tp=1 and tp=2 setups are not consistent."
        )

if __name__ == '__main__':
    # This allows running the test script directly
    unittest.main() 