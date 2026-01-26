# Neuro-Manipulation Tests

This directory contains tests for the utilities within the `neuro_manipulation` package.

## Tests

### `test_model_layer_detector.py`

Contains unit tests for the `ModelLayerDetector` class. These tests verify its ability to correctly identify transformer layers across various model architectures, including:

- Standard HuggingFace models (GPT-2, OPT)
- Models with different layer naming conventions (ChatGLM)
- Non-standard architectures (RWKV - requires `trust_remote_code=True`)
- Custom-built simple transformer models
- Models loaded via vLLM

### `test_vllm_hook_registration.py`

Tests the ability to register PyTorch forward hooks on transformer layers within a model served by vLLM.

- Uses `ModelLayerDetector` to find the target layers dynamically.
- Initializes a vLLM `LLM` instance with standard models (e.g., GPT-2, OPT-125m, Llama-2-7b).
- Accesses the underlying PyTorch model from the vLLM engine.
- Registers a simple forward hook on a specific layer.
- Runs inference using `llm.generate()`.
- Verifies that the hook was executed during inference.
- Removes the hook and verifies it is no longer active.

**Note:** This test requires a GPU environment with CUDA available and vLLM installed. For multi-GPU testing, set the `CUDA_VISIBLE_DEVICES` environment variable (e.g., `CUDA_VISIBLE_DEVICES=0,1`) before running the test. The test will automatically use the specified number of GPUs for tensor parallelism (`tensor_parallel_size`).

### `test_vllm_hook_modification.py`

# Test: VLLM Hook Output Modification via RPC

This test script (`test_vllm_hook_modification.py`) verifies the ability to register PyTorch forward hooks onto specific layers of a model running within vLLM worker processes and **modify** the output of those layers. This is achieved using vLLM's `collective_rpc` mechanism.

## Purpose

The primary goal is to demonstrate that:
1.  A hook function can be successfully registered onto a target layer (e.g., the first decoder block) across all tensor-parallel workers using RPC.
2.  This hook function can intercept the layer's forward pass output.
3.  The hook function can modify the output tensor (in this case, by zeroing it out).
4.  The modification to the intermediate layer's output has a tangible effect on the final generated text produced by the model.

## How it Works

1.  **Initialization:** An LLM instance is created using vLLM, potentially with tensor parallelism (`tensor_parallel_size > 1`).
2.  **Baseline Generation:** The script first generates text for a given prompt *without* any hooks active. This serves as a baseline. `temperature=0.0` is used for deterministic output.
3.  **Hook Function (`hook_fn_modify_output_global`):** A global Python function is defined. This function:
    *   Accepts the standard PyTorch hook arguments (`module`, `args`, `output`).
    *   Identifies the main output tensor (handling cases where the layer returns a tuple).
    *   Logs information about the tensor (shape, dtype, device).
    *   **Modifies** the tensor by creating a new tensor of zeros with the same properties (`torch.zeros_like`).
    *   Logs completion and returns the modified tensor (or reconstructed tuple).
4.  **RPC Registration (`_register_hook_on_worker_rpc`):**
    *   A helper function `_register_hook_on_worker_rpc` is defined to run on each worker via RPC.
    *   This function accesses the worker's local model instance (`worker_self.model_runner.model`).
    *   It uses `ModelLayerDetector` to find the layers of the model.
    *   It registers the `hook_fn_modify_output_global` to the specified `target_layer_index` (e.g., index 0).
    *   The main process calls `llm.llm_engine.collective_rpc(_register_hook_on_worker_rpc, args=(target_layer_index, hook_fn_modify_output_global))` to execute registration on all workers.
5.  **Modified Generation:** The script runs generation again with the *same* prompt and sampling parameters, but now the hook is active.
6.  **Verification:**
    *   The test asserts that the RPC call reported success on at least one worker.
    *   Crucially, it asserts that the text generated *with* the hook is **different** from the baseline text generated *without* the hook (`assertNotEqual`).
    *   Extensive logging is included in the hook and the test itself to aid debugging. Look for `*** MODIFY HOOK EXECUTING ***` messages in the logs from the workers.
7.  **Cleanup:** Resources (LLM object) are deleted, and CUDA cache is cleared between test runs for different models.

## Running the Test

Ensure you have the necessary dependencies (vLLM, PyTorch, etc.) installed in your environment (e.g., `conda activate llm`).

```bash
# Activate your environment if needed
# conda activate llm

# Run the specific test file
python -m unittest neuro_manipulation.tests.test_vllm_hook_modification
```

## Requirements

*   vLLM installed.
*   PyTorch installed.
*   Access to CUDA GPU(s). The test skips if `CUDA_VISIBLE_DEVICES` is not set.
*   The `neuro_manipulation` package (containing `ModelLayerDetector`) available in the Python path.

## Running Tests

To run all tests in this directory, navigate to the project root and use the `unittest` discovery mechanism:

```bash
python -m unittest discover neuro_manipulation/tests/
```

To run a specific test file (e.g., the vLLM hook test):

```bash
python -m unittest neuro_manipulation.tests.test_vllm_hook_registration
```

To run a specific test class within a file:

```bash
python -m unittest neuro_manipulation.tests.test_vllm_hook_registration.TestVLLMHookRegistration
```

To run a specific test method within a class:
```bash
python -m unittest neuro_manipulation.tests.test_vllm_hook_registration.TestVLLMHookRegistration.test_vllm_forward_hook
```
