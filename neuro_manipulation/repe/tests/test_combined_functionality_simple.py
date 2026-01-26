#!/usr/bin/env python3
"""
Simple integration test for CombinedVLLMHook functionality.
This test uses a minimal setup to validate core features work.
"""

import sys
import os
import unittest

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def run_simple_test():
    """Run a simple test without unittest framework for quick validation."""
    try:
        import torch
        from vllm import LLM
        from transformers import AutoTokenizer
        from sequence_prob_vllm_hook import CombinedVLLMHook
        
        print("🚀 Starting simple functionality test...")
        
        # Use smallest possible model for testing
        model_name = "gpt2"
        print(f"📦 Loading model: {model_name}")
        
        # Initialize with minimal memory usage
        llm = LLM(
            model=model_name,
            tensor_parallel_size=1,
            gpu_memory_utilization=0.3,
            max_num_seqs=4,
            enforce_eager=True
        )
        
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        print("✅ Model and tokenizer loaded successfully")
        
        # Test 1: Sequence Probability Only
        print("\n🧮 Test 1: Sequence Probability Calculation")
        seq_hook = CombinedVLLMHook(
            llm, tokenizer,
            enable_sequence_prob=True,
            enable_rep_control=False,
            enable_layer_logit_recording=False
        )
        
        results = seq_hook.get_log_prob(
            ["The capital of France is"],
            ["Paris", "London"]
        )
        
        print(f"   Found {len(results)} probability results:")
        for result in results:
            print(f"   - '{result['sequence']}': {result['prob']:.6f}")
        
        # Test 2: Representation Control Only
        print("\n🎛️  Test 2: Representation Control")
        control_hook = CombinedVLLMHook(
            llm, tokenizer,
            layers=[5, 8],  # GPT2 layers
            enable_sequence_prob=False,
            enable_rep_control=True,
            enable_layer_logit_recording=False
        )
        
        # Create small control vectors
        control_activations = {
            5: torch.randn(768, dtype=torch.float16) * 0.05,  # GPT2 hidden_dim = 768
            8: torch.randn(768, dtype=torch.float16) * 0.05
        }
        
        baseline_output = control_hook.generate_with_control(
            ["The weather today is"],
            max_new_tokens=3,
            temperature=0.0
        )
        
        controlled_output = control_hook.generate_with_control(
            ["The weather today is"],
            activations=control_activations,
            max_new_tokens=3,
            temperature=0.0
        )
        
        baseline_text = baseline_output[0].outputs[0].text
        controlled_text = controlled_output[0].outputs[0].text
        
        print(f"   Baseline:    '{baseline_text}'")
        print(f"   Controlled:  '{controlled_text}'")
        print(f"   Different:   {baseline_text != controlled_text}")
        
        # Test 3: Combined Features
        print("\n🔄 Test 3: Combined Functionality")
        combined_hook = CombinedVLLMHook(
            llm, tokenizer,
            layers=[5],
            enable_sequence_prob=True,
            enable_rep_control=True,
            enable_layer_logit_recording=True
        )
        
        # Sequence probability
        prob_results = combined_hook.get_log_prob(["Hello"], ["world", "there"])
        print(f"   Probability results: {len(prob_results)}")
        
        # Combined generation with recording
        outputs = combined_hook.generate_with_control(
            ["Hello"],
            activations={5: torch.randn(768, dtype=torch.float16) * 0.02},
            record_layer_logits=True,
            max_new_tokens=2
        )
        
        layer_data = combined_hook.get_layer_logits()
        print(f"   Generated text: '{outputs[0].outputs[0].text}'")
        print(f"   Recorded layers: {list(layer_data.keys())}")
        
        print("\n✅ All tests completed successfully!")
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    finally:
        # Cleanup
        try:
            if 'llm' in locals():
                del llm
            torch.cuda.empty_cache()
            print("🧹 Cleanup completed")
        except:
            pass


class TestCombinedFunctionalitySimple(unittest.TestCase):
    """Simple unittest wrapper for the integration test."""
    
    def test_combined_functionality(self):
        """Test that the combined functionality works end-to-end."""
        success = run_simple_test()
        self.assertTrue(success, "Combined functionality test should pass")


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == '--simple':
        # Run simple test directly
        run_simple_test()
    else:
        # Run as unittest
        unittest.main() 