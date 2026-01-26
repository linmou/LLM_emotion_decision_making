import unittest
import torch
from vllm import LLM, SamplingParams
from neuro_manipulation.model_layer_detector import ModelLayerDetector
import logging
import os
import sys
import torch
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

# --- Global Hook Function (for RPC) --- #
def hook_fn_global(module, args, output):
    """Simple global hook function that logs execution."""
    rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 'N/A'
    logger.info(f"*** HOOK EXECUTED on Rank {rank} for module {module.__class__.__name__} ***")
    # We don't capture state here as it's hard to retrieve from workers easily
    return output
# --- End Global Hook Function --- #

# --- RPC Function to Register Hook on Worker --- #
def _register_hook_on_worker_rpc(worker_self, layer_index, hook_func):
    """
    This function runs on each vLLM worker process via collective_rpc.
    It accesses the worker's model, finds layers, and registers the hook.
    'worker_self' refers to the worker instance (e.g., GpuWorker).
    """
    try:
        logger.info(f"RPC: Worker Rank {worker_self.rank} attempting to register hook on layer {layer_index}.")
        # Access the model on the worker
        # The exact path might depend on the worker class, common is model_runner
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
MODELS_TO_TEST = ["gpt2", "facebook/opt-125m", "meta-llama/Llama-2-7b-hf"]

# Use the same parameters identified as working in the basic loading test
GPU_MEM_UTIL = 0.85
MAX_SEQS = 32

@unittest.skipIf(not torch.cuda.is_available(), "Requires CUDA GPU")
class TestVLLMHookRegistration(unittest.TestCase):

    def test_vllm_forward_hook_via_rpc(self):
        """
        Test registering hooks via RPC and check logs for execution.
        Addresses inability to access model directly in v1 engine.
        Also checks if sequential loading works with this method.
        """
        global num_gpus
        for model_name in MODELS_TO_TEST:
            llm = None
            with self.subTest(model=model_name, gpus=num_gpus):
                try:
                    logger.info(f"[{model_name}] Initializing LLM with tensor_parallel_size={num_gpus}, gpu_memory_utilization={GPU_MEM_UTIL}, max_num_seqs={MAX_SEQS}")
                    llm = LLM(model=model_name,
                              enforce_eager=True,
                              trust_remote_code=True,
                              tensor_parallel_size=num_gpus,
                              gpu_memory_utilization=GPU_MEM_UTIL,
                              max_num_seqs=MAX_SEQS)

                    # Define target layer index
                    target_layer_index = 0
                    logger.info(f"[{model_name}] Attempting to register hook on layer {target_layer_index} via RPC...")

                    # Execute the registration function on all workers via RPC
                    # Pass the global hook function directly
                    rpc_results = llm.llm_engine.collective_rpc(_register_hook_on_worker_rpc,
                                                               args=(target_layer_index, hook_fn_global))

                    logger.info(f"[{model_name}] RPC registration results: {rpc_results}")
                    # Simple check: Ensure at least one worker succeeded (especially rank 0)
                    # Note: In TP, all workers might have the layer, or only some might.
                    # Depending on layer type and TP strategy.
                    self.assertTrue(any(rpc_results), f"[{model_name}] RPC hook registration failed on all workers.")

                    prompt = "Hello, my name is"
                    sampling_params = SamplingParams(max_tokens=5)
                    logger.info(f"[{model_name}] Running generation. Check logs for '*** HOOK EXECUTED ***' message.")
                    outputs = llm.generate(prompt, sampling_params)
                    logger.info(f"[{model_name}] Generation output: {outputs[0].outputs[0].text}")

                    # Cannot easily assert hook state across processes.
                    # Manual log check is needed for '*** HOOK EXECUTED ***'
                    logger.warning(f"[{model_name}] Hook execution verification requires manual check of logs for '*** HOOK EXECUTED ***' message from workers.")

                    # Hook removal test is omitted due to complexity with RPC handles.

                    logger.info(f"[{model_name}] Subtest completed successfully (pending log check).")

                except Exception as e:
                    self.fail(f"[{model_name}] Subtest failed with exception: {e}\nTraceback: {traceback.format_exc()}")
                finally:
                    logger.info(f"[{model_name}] Cleaning up resources...")
                    # No hook handle to delete in the main process
                    if llm is not None:
                        del llm
                    torch.cuda.empty_cache()
                    gc.collect()
                    logger.info(f"[{model_name}] Resource cleanup finished.")

# Example of how to run this test:
# python -m unittest neuro_manipulation.tests.test_vllm_hook_registration

if __name__ == '__main__':
    unittest.main() 