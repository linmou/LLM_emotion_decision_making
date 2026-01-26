import unittest
import torch
from vllm import LLM, SamplingParams
import logging
import os
import sys
import gc # Import gc for garbage collection
import traceback # Import traceback for printing full trace on failure

# Configure logging to output to stdout
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    stream=sys.stdout)
logger = logging.getLogger(__name__)

# --- Determine number of GPUs --- #
# Default to 1 GPU if not specified or invalid
num_gpus = 1
if "CUDA_VISIBLE_DEVICES" in os.environ:
    cuda_devices = os.environ["CUDA_VISIBLE_DEVICES"].split(",")
    if all(dev.isdigit() for dev in cuda_devices):
        num_gpus = len(cuda_devices)
    logger.info(f"CUDA_VISIBLE_DEVICES found: {os.environ['CUDA_VISIBLE_DEVICES']}. Setting tensor_parallel_size={num_gpus}")
else:
    logger.warning("CUDA_VISIBLE_DEVICES not set. Defaulting to tensor_parallel_size=1. Test may be skipped.")
# --- End Determine number of GPUs --- #

# Models to test
MODELS_TO_TEST = [ "meta-llama/Llama-2-7b-hf"]

# Use the same parameters that caused issues before, to see if loading alone fails
GPU_MEM_UTIL = 0.85 # Value from previous attempt
MAX_SEQS = 32      # Value from previous attempt

@unittest.skipIf("CUDA_VISIBLE_DEVICES" not in os.environ, "Requires GPU")
class TestVLLMBasicLoading(unittest.TestCase):

    def test_sequential_model_loading(self):
        """
        Test if multiple models can be loaded and run sequentially using vLLM
        with tensor parallelism, without involving hooks.
        This helps isolate OOM errors that might occur during loading/cleanup.
        """
        global num_gpus # Access the globally determined number of GPUs
        for model_name in MODELS_TO_TEST:
            # Initialize variables for cleanup in case of early failure
            llm = None

            with self.subTest(model=model_name, gpus=num_gpus):
                try:
                    logger.info(f"[{model_name}] Initializing LLM with tensor_parallel_size={num_gpus}, gpu_memory_utilization={GPU_MEM_UTIL}, max_num_seqs={MAX_SEQS}")
                    llm = LLM(model=model_name,
                              enforce_eager=True, # Keep eager for consistency with previous tests
                              trust_remote_code=True,
                              tensor_parallel_size=num_gpus,
                              gpu_memory_utilization=GPU_MEM_UTIL,
                              max_num_seqs=MAX_SEQS)

                    # Basic check: Run a simple generation
                    prompt = "Test prompt"
                    sampling_params = SamplingParams(max_tokens=5)
                    logger.info(f"[{model_name}] Running generation for prompt: '{prompt}'")
                    outputs = llm.generate(prompt, sampling_params)
                    self.assertTrue(len(outputs) > 0, f"[{model_name}] LLM generation returned no output.")
                    logger.info(f"[{model_name}] Generation successful.")

                    logger.info(f"[{model_name}] Subtest completed successfully.")

                except Exception as e:
                    # Catch exceptions to allow cleanup before failing the subtest
                    self.fail(f"[{model_name}] Subtest failed with exception: {e}\nTraceback: {traceback.format_exc()}")
                finally:
                    # --- Aggressive Cleanup --- #
                    logger.info(f"[{model_name}] Cleaning up resources...")
                    if llm is not None:
                        # Delete the LLM object first
                        del llm
                    torch.cuda.empty_cache() # Clear PyTorch cache
                    gc.collect() # Trigger Python garbage collection
                    logger.info(f"[{model_name}] Resource cleanup finished.")
                    # --- End Cleanup --- #

# Example of how to run this test:
# python -m unittest neuro_manipulation.tests.test_vllm_basic_loading

if __name__ == '__main__':
    unittest.main() 