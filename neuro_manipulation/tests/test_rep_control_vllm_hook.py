import unittest
import torch
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
from neuro_manipulation.repe.rep_control_vllm_hook import RepControlVLLMHook
import logging
import os
import sys
import gc
import traceback

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    stream=sys.stdout)
logger = logging.getLogger(__name__)

# --- Determine number of GPUs ---
num_gpus = 1 # Default
if torch.cuda.is_available():
    num_gpus = torch.cuda.device_count()
    logger.info(f"CUDA is available. Found {num_gpus} GPU(s). Setting tensor_parallel_size={num_gpus}")
else:
    logger.warning("CUDA not available. Test requiring GPU will be skipped.")
    num_gpus = 0 # Ensure tests requiring GPU are skipped

# --- Test Parameters ---
MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"
GPU_MEM_UTIL = 0.85
MAX_SEQS = 16
HIDDEN_DIM = 4096 # Llama 3.1 8B
LAYERS_TO_CONTROL = [10, 15]
BLOCK_TO_HOOK = "decoder_block"
CONTROL_METHOD = "reading_vec"
PROMPT = "Describe the feeling of joy in one sentence:"

@unittest.skipIf(num_gpus == 0, "Requires CUDA GPU")
class TestRepControlVLLMHook(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """Load the model and tokenizer once for all tests in this class."""
        cls.llm = None
        cls.tokenizer = None
        try:
            logger.info(f"[setUpClass] Loading tokenizer for {MODEL_NAME}...")
            cls.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
            if cls.tokenizer.pad_token is None:
                cls.tokenizer.pad_token = cls.tokenizer.eos_token
                logger.info("[setUpClass] Set pad_token to eos_token.")

            logger.info(f"[setUpClass] Loading LLM {MODEL_NAME} with tensor_parallel_size={num_gpus}...")
            cls.llm = LLM(model=MODEL_NAME,
                          tokenizer=cls.tokenizer.name_or_path,
                          enforce_eager=True, # Important for hooks
                          trust_remote_code=True,
                          tensor_parallel_size=num_gpus,
                          gpu_memory_utilization=GPU_MEM_UTIL,
                          max_num_seqs=MAX_SEQS)
            logger.info("[setUpClass] LLM loaded successfully.")

        except Exception as e:
            logger.error(f"[setUpClass] Failed to load model or tokenizer: {e}", exc_info=True)
            # If setup fails, we might want to prevent tests from running
            # Raising an exception here will stop the test suite for this class.
            raise RuntimeError(f"Failed to initialize LLM/Tokenizer in setUpClass: {e}")

    @classmethod
    def tearDownClass(cls):
        """Clean up resources after all tests in this class are done."""
        logger.info("[tearDownClass] Cleaning up LLM resources...")
        if hasattr(cls, 'llm') and cls.llm is not None:
            del cls.llm
            cls.llm = None
        torch.cuda.empty_cache()
        gc.collect()
        logger.info("[tearDownClass] LLM resources cleaned up.")

    def setUp(self):
        """Set up for each test method.
           Ensures RepControl is initialized before each test and cleaned after.
        """
        self.rep_control = None
        if self.llm and self.tokenizer: # Proceed only if class setup was successful
            try:
                logger.info(f"[setUp] Initializing RepControlVLLMHook for layers {LAYERS_TO_CONTROL}, block '{BLOCK_TO_HOOK}'...")
                self.rep_control = RepControlVLLMHook(self.llm,
                                                      self.tokenizer,
                                                      LAYERS_TO_CONTROL,
                                                      BLOCK_TO_HOOK,
                                                      CONTROL_METHOD,
                                                      tensor_parallel_size=num_gpus)
                logger.info("[setUp] RepControlVLLMHook initialized.")
            except Exception as e:
                 logger.error(f"[setUp] Failed to initialize RepControlVLLMHook: {e}", exc_info=True)
                 self.fail(f"Failed to initialize RepControlVLLMHook in setUp: {e}")
        else:
             self.skipTest("Skipping test because LLM/Tokenizer failed to load in setUpClass.")

    def tearDown(self):
         """Clean up after each test method.
            Attempts hook removal if needed (though may be complex).
         """
         if hasattr(self, 'rep_control') and self.rep_control is not None:
              logger.info("[tearDown] Cleaning up RepControlVLLMHook (hooks should be reset by __call__ finally block)...")
              # Optional: Call explicit hook removal if implemented and necessary
              # try:
              #     self.rep_control.remove_hooks()
              # except Exception as e:
              #     logger.warning(f"[tearDown] Error during experimental hook removal: {e}")
              del self.rep_control
              self.rep_control = None
         # gc.collect might be useful here too if memory is tight between tests
         # gc.collect()

    def test_baseline_vs_controlled_generation(self):
        """Tests that controlled generation output differs from baseline."""
        self.assertIsNotNone(self.rep_control, "RepControlVLLMHook not initialized in setUp.")

        try:
            # 1. Baseline Generation
            logger.info("[test_baseline_vs_controlled] Running Baseline Generation...")
            baseline_outputs = self.rep_control([PROMPT], max_new_tokens=15, temperature=0.0)
            self.assertTrue(baseline_outputs, "Baseline generation returned empty list.")
            baseline_text = baseline_outputs[0].outputs[0].text
            logger.info(f"[test_baseline_vs_controlled] Baseline Output: '{baseline_text}'")
            self.assertTrue(baseline_text, "Baseline generation failed to produce text.")

            # 2. Controlled Generation
            logger.info("[test_baseline_vs_controlled] Running Controlled Generation...")
            control_activations = {}
            for layer_id in LAYERS_TO_CONTROL:
                # Create a simple, non-zero dummy vector
                # Use float16 as vLLM often uses it internally
                vec = torch.randn(HIDDEN_DIM, dtype=torch.float16) * 0.01 # Small magnitude perturbation
                vec = torch.randn(HIDDEN_DIM, dtype=torch.float16) * 1.0 # Increased magnitude
                control_activations[layer_id] = vec

            controlled_outputs = self.rep_control(
                [PROMPT],
                activations=control_activations,
                max_new_tokens=15,
                temperature=0.0, # Keep temp 0 for comparison
                operator='linear_comb',
                normalize=False,
                token_pos=None # Apply to all tokens
            )
            self.assertTrue(controlled_outputs, "Controlled generation returned empty list.")
            controlled_text = controlled_outputs[0].outputs[0].text
            logger.info(f"[test_baseline_vs_controlled] Controlled Output: '{controlled_text}'")
            self.assertTrue(controlled_text, "Controlled generation failed to produce text.")

            # 3. Comparison
            logger.info("[test_baseline_vs_controlled] Comparing outputs...")
            self.assertNotEqual(baseline_text, controlled_text,
                                f"Baseline and controlled outputs were the same (\'{baseline_text}\'). Hook modification might not have had an effect.")
            logger.info("[test_baseline_vs_controlled] SUCCESS: Baseline and controlled outputs differ.")

        except Exception as e:
            logger.error(f"[test_baseline_vs_controlled] Test failed with exception: {e}", exc_info=True)
            self.fail(f"Test failed with exception: {e}\nTraceback: {traceback.format_exc()}")

# How to run:
# Make sure vLLM is installed and CUDA is available.
# Ensure you are in the root directory of the project.
# Set huggingface token if needed: export HF_TOKEN=your_token
# Run with: python -m unittest neuro_manipulation.tests.test_rep_control_vllm_hook

if __name__ == '__main__':
    # You can potentially add setup here if needed globally, but setUpClass/tearDownClass is preferred
    unittest.main() 