import unittest
import torch
from vllm import LLM, SamplingParams
from neuro_manipulation.model_layer_detector import ModelLayerDetector
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
num_gpus = 1 # Default
if torch.cuda.is_available():
    num_gpus = torch.cuda.device_count()
    logger.info(f"CUDA is available. Found {num_gpus} GPU(s). Setting tensor_parallel_size={num_gpus}")
    # Optionally log the value of CUDA_VISIBLE_DEVICES if set
    if "CUDA_VISIBLE_DEVICES" in os.environ:
        logger.info(f"CUDA_VISIBLE_DEVICES is set to: {os.environ['CUDA_VISIBLE_DEVICES']}")
    else:
        logger.info("CUDA_VISIBLE_DEVICES is not set. PyTorch sees all available GPUs.")
else:
    logger.warning("CUDA not available. Test requiring GPU will be skipped.")
# --- End Determine number of GPUs --- #

# --- Global Hook Function (for Modification) --- #
def hook_fn_modify_output_global(module, args, output):
    """Hook function that modifies the output tensor by zeroing it out and logs details."""
    rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 'N/A'
    try:
        # Check if output is a tensor or a tuple (common in some models)
        if isinstance(output, torch.Tensor):
            target_tensor = output
        elif isinstance(output, tuple) and len(output) > 0 and isinstance(output[0], torch.Tensor):
            # Assuming the first element is the main hidden state tensor
            target_tensor = output[0]
            logger.warning(f"Rank {rank} - Module {module.__class__.__name__} output is a tuple. Modifying the first element.")
        else:
            logger.error(f"Rank {rank} - Module {module.__class__.__name__} output type is not a Tensor or Tuple[Tensor, ...]: {type(output)}. Cannot modify.")
            return output # Return original output if we can't handle it

        # Log details before modification
        logger.info(f"*** MODIFY HOOK EXECUTING on Rank {rank} for module {module.__class__.__name__} ***")
        logger.info(f"Rank {rank} - Original output tensor shape: {target_tensor.shape}, dtype: {target_tensor.dtype}, device: {target_tensor.device}")
        # Log a small slice before zeroing (optional, can be verbose)
        logger.info(f"Rank {rank} - Original output tensor slice (pre-zero): {target_tensor.flatten()[:8]}")

        # --- Modification: Zero out the tensor --- # 
        modified_tensor = torch.zeros_like(target_tensor)
        # --- End Modification --- #

        # --- Verification within the hook --- #
        are_tensors_equal = torch.equal(target_tensor, modified_tensor)
        logger.info(f"Rank {rank} - Verification: Original tensor == Modified tensor? {are_tensors_equal}")
        if are_tensors_equal and target_tensor.numel() > 0: # Avoid false positive for empty tensors
             logger.warning(f"Rank {rank} - !!! Modification might have failed: Original and modified tensors are equal. !!!")
        # --- End Verification --- #

        # Log after modification
        logger.info(f"Rank {rank} - Modified output tensor slice (post-zero): {modified_tensor.flatten()[:8]}")
        logger.info(f"*** MODIFY HOOK FINISHED on Rank {rank} ***")

        # Reconstruct the output if it was a tuple
        if isinstance(output, tuple):
            # Create a new tuple with the modified tensor as the first element
            modified_output_tuple = (modified_tensor,) + output[1:]
            return modified_output_tuple
        else:
            return modified_tensor # Return the modified tensor

    except Exception as e:
        logger.error(f"Rank {rank} - Error in modify hook for {module.__class__.__name__}: {e}", exc_info=True)
        return output # Return original output on error
# --- End Global Hook Function --- #

# --- RPC Function to Register Hook on Worker --- #
# (Keep the existing RPC function, it's reusable)
def _register_hook_on_worker_rpc(worker_self, layer_index, hook_func):
    """
    This function runs on each vLLM worker process via collective_rpc.
    It accesses the worker's model, finds layers, and registers the hook.
    'worker_self' refers to the worker instance (e.g., GpuWorker).
    """
    try:
        logger.info(f"RPC: Worker Rank {worker_self.rank} attempting to register hook {hook_func.__name__} on layer {layer_index}.")
        # Access the model on the worker
        if not hasattr(worker_self, 'model_runner') or not hasattr(worker_self.model_runner, 'model'):
             logger.error(f"RPC: Worker Rank {worker_self.rank} could not find model_runner.model")
             return False

        model = worker_self.model_runner.model

        # Detect layers on the worker's model instance
        logger.info(f"RPC: Worker Rank {worker_self.rank} detecting layers...")
        layers = ModelLayerDetector.get_model_layers(model)
        logger.info(f"RPC: Worker Rank {worker_self.rank} detected {len(layers)} layers.")

        if len(layers) <= layer_index:
            logger.warning(f"RPC: Worker Rank {worker_self.rank} - Layer index {layer_index} out of bounds ({len(layers)} layers found). Skipping hook registration.")
            return False # Indicate failure/skip

        target_layer = layers[layer_index]
        logger.info(f"RPC: Worker Rank {worker_self.rank} registering hook {hook_func.__name__} to {target_layer.__class__.__name__}")
        handle = target_layer.register_forward_hook(hook_func)

        # We can't easily return the handle, just confirm registration attempt
        if handle:
             logger.info(f"RPC: Worker Rank {worker_self.rank} hook registered successfully.")
             # Note: handle removal would need another RPC call, not handled here.
             return True
        else:
             logger.error(f"RPC: Worker Rank {worker_self.rank} hook registration failed.")
             return False
    except Exception as e:
        logger.error(f"RPC: Worker Rank {worker_self.rank} error during hook registration: {e}", exc_info=True)
        return False
# --- End RPC Function --- #

# Models to test
MODELS_TO_TEST = ["meta-llama/Llama-3.1-8B-Instruct"] # Reduced set for faster testing initially
# "meta-llama/Llama-2-7b-hf" # Can add back later

# Use the same parameters identified as working in the basic loading test
GPU_MEM_UTIL = 0.85
MAX_SEQS = 32

@unittest.skipIf(not torch.cuda.is_available(), "Requires CUDA GPU")
class TestVLLMHookModification(unittest.TestCase): # Renamed class

    def test_vllm_forward_hook_modification_via_rpc(self): # Renamed method
        """
        Test registering hooks via RPC that modify layer output and check
        if the final generation changes compared to a baseline.
        """
        global num_gpus
        for model_name in MODELS_TO_TEST:
            llm = None
            baseline_output_text = None
            modified_output_text = None
            with self.subTest(model=model_name, gpus=num_gpus):
                try:
                    logger.info(f"[{model_name}] Initializing LLM with tensor_parallel_size={num_gpus}, gpu_memory_utilization={GPU_MEM_UTIL}, max_num_seqs={MAX_SEQS}")
                    llm = LLM(model=model_name,
                              enforce_eager=True,
                              trust_remote_code=True,
                              tensor_parallel_size=num_gpus,
                              gpu_memory_utilization=GPU_MEM_UTIL,
                              max_num_seqs=MAX_SEQS)

                    prompt = "The quick brown fox jumps over the" # Use a slightly longer prompt
                    sampling_params = SamplingParams(max_tokens=10, temperature=0.0) # Use temp 0 for deterministic baseline

                    # 1. Run baseline generation (no hook)
                    logger.info(f"[{model_name}] Running baseline generation...")
                    baseline_outputs = llm.generate(prompt, sampling_params)
                    baseline_output_text = baseline_outputs[0].outputs[0].text
                    logger.info(f"[{model_name}] Baseline output: '{baseline_output_text}'")
                    self.assertTrue(baseline_output_text, "Baseline generation failed to produce text.")

                    # Define target layer index (e.g., first decoder block)
                    target_layer_index = 0
                    logger.info(f"[{model_name}] Attempting to register modification hook on layer {target_layer_index} via RPC...")

                    # 2. Register the modification hook via RPC
                    rpc_results = llm.llm_engine.collective_rpc(_register_hook_on_worker_rpc,
                                                               args=(target_layer_index, hook_fn_modify_output_global))

                    logger.info(f"[{model_name}] RPC modification hook registration results: {rpc_results}")
                    self.assertTrue(any(rpc_results), f"[{model_name}] RPC modification hook registration failed on all workers.")

                    # 3. Run generation with the hook active
                    logger.info(f"[{model_name}] Running generation with modification hook...")
                    modified_outputs = llm.generate(prompt, sampling_params) # Use same prompt and params
                    modified_output_text = modified_outputs[0].outputs[0].text
                    logger.info(f"[{model_name}] Modified output: '{modified_output_text}'")
                    self.assertTrue(modified_output_text, "Modified generation failed to produce text.")

                    # 4. Assert that the outputs are different
                    logger.info(f"[{model_name}] Comparing baseline and modified outputs...")
                    self.assertNotEqual(baseline_output_text, modified_output_text,
                                        f"[{model_name}] Baseline and modified outputs were the same. Hook modification might not have had an effect.")
                    logger.info(f"[{model_name}] Outputs are different, modification hook likely had an effect.")

                    # Hook removal is complex with RPC, skipping explicit removal test. Relying on model re-initialization.

                    logger.info(f"[{model_name}] Subtest completed successfully.")

                except Exception as e:
                    self.fail(f"[{model_name}] Subtest failed with exception: {e} Traceback: {traceback.format_exc()}")
                finally:
                    logger.info(f"[{model_name}] Cleaning up resources...")
                    if llm is not None:
                        # Destroy the vLLM engine and workers
                        # NOTE: vLLM might not have a clean public API for forced destruction.
                        # Relying on 'del' and gc for now. In practice, restarting the process might be needed for true isolation.
                        # Accessing internal engine state for shutdown is fragile:
                        # if hasattr(llm, 'llm_engine') and hasattr(llm.llm_engine, 'shutdown'):
                        #     try:
                        #         llm.llm_engine.shutdown()
                        #         logger.info(f"[{model_name}] Called llm_engine.shutdown()")
                        #     except Exception as shutdown_e:
                        #         logger.warning(f"[{model_name}] Error during llm_engine.shutdown(): {shutdown_e}")
                        del llm
                    torch.cuda.empty_cache()
                    gc.collect()
                    logger.info(f"[{model_name}] Resource cleanup finished.")

# Example of how to run this test:
# python -m unittest neuro_manipulation.tests.test_vllm_hook_modification

if __name__ == '__main__':
    # Ensure multiprocessing start method is appropriate if needed, although vLLM handles it internally.
    # import torch.multiprocessing as mp
    # try:
    #     mp.set_start_method('spawn') # Often safer with CUDA
    # except RuntimeError:
    #     pass # Already set or not applicable
    unittest.main() 