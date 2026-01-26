# RepControlVLLMHook
<!-- Updated: 2025-12-20 | Commit: a5cad74 -->

This module provides the `RepControlVLLMHook` class, designed to apply representation control techniques (specifically `reading_vec` for now) to models running within the vLLM framework by leveraging forward hooks and Remote Procedure Calls (RPC).

## Overview

Instead of wrapping model layers (as in `rep_control_reading_vec.py`), this approach injects control by registering PyTorch forward hooks directly onto the target layers (or submodules like `mlp`, `self_attn`) of the model running on each vLLM worker process. This avoids modifying the model structure itself but relies on vLLM's `collective_rpc` mechanism to manage the hooks and their associated state.

## How it Works

1.  **Initialization (`__init__`)**:
    *   Takes a vLLM `LLM` instance, tokenizer, target layer indices, the name of the block/module within the layer to hook (e.g., `"decoder_block"` for the layer's main output), and the control method (`"reading_vec"`).
    *   Uses `collective_rpc` to call `_register_hook_on_worker_rpc` on each worker.
    *   `_register_hook_on_worker_rpc` finds the specified module (e.g., the Nth decoder layer) on the worker's copy of the model and registers the `hook_fn_rep_control` function as a forward hook.
    *   The hook initially does nothing, as its control state is not yet set.

2.  **Generation (`__call__`)**:
    *   Takes prompts and optional control parameters (`activations`, `token_pos`, `masks`, `normalize`, `operator`).
    *   **Set State**: If `activations` (a dictionary mapping layer indices to control tensors) are provided, it calls `_set_controller_state_on_worker_rpc` via `collective_rpc`.
        *   This RPC function finds the target module on the worker and attaches a `_rep_control_state` attribute to it. This attribute holds the control tensor, mask, operator function, and other parameters needed by the hook.
    *   **Run Inference**: It calls the standard `model.generate()` method.
        *   During the forward pass on each worker, when execution reaches a hooked module, the `hook_fn_rep_control` is triggered.
        *   The hook checks if `_rep_control_state` exists on the module.
        *   If the state exists, the hook retrieves the control parameters (controller tensor, mask, operator, etc.) and applies the modification logic (e.g., adding the controller vector to the module's output) before returning the modified output.
        *   If no state exists, the hook simply returns the original output.
    *   **Reset State**: After generation finishes (in a `finally` block to ensure cleanup), it calls `_reset_controller_state_on_worker_rpc` via `collective_rpc`.
        *   This RPC function finds the target module on the worker and deletes the `_rep_control_state` attribute, ensuring subsequent unrelated inference calls are not affected.

3.  **Hook Function (`hook_fn_rep_control`)**:
    *   This function contains the core logic for applying the representation control modification, similar to the logic within `WrappedBlock.forward` in `rep_control_reading_vec.py`.
    *   It handles accessing the correct output tensor (even if the module returns a tuple), applying masks, handling token positions, performing normalization, and using the specified operator (e.g., linear combination).

## Advantages

*   **No Model Monkey-Patching**: Doesn't require modifying the vLLM model's layer structure directly.
*   **Leverages vLLM Infrastructure**: Uses `collective_rpc` for distributed state management.

## Disadvantages/Considerations

*   **RPC Overhead**: Sending control state via RPC for every controlled generation call might introduce some overhead compared to having the logic permanently wrapped in the layer.
*   **State Management Complexity**: Relies on correctly setting and resetting state via RPC. Errors in RPC or state management could lead to inconsistent behavior.
*   **Hook Limitations**: Hooks might interact unexpectedly with vLLM's internal optimizations or execution graph. `enforce_eager=True` might be necessary when initializing the vLLM `LLM` object.
*   **Hook Removal**: Properly removing hooks registered via RPC requires careful handle management, which is currently implemented conceptually but might need refinement.

## vLLM v0.11+ Compatibility Notes (Important)

vLLM v0.11 tightened multiprocessing and RPC serialization rules. The RepControl hook workflow is designed to be compatible and safe by default.

### 1) RPC must use method names (no Python callables)

vLLM v0.11+ disallows serializing Python callables over `collective_rpc` by default. This hook therefore uses string RPC method names:

- `_nm_repcontrol_register_hook`
- `_nm_repcontrol_set_state`
- `_nm_repcontrol_reset_state`

These methods are implemented on workers via the vLLM worker extension class:

- `neuro_manipulation/repe/vllm_worker_extension.py` → `NMRepControlWorkerExtension`

### 2) Worker extension class must be importable in vLLM worker processes

vLLM spawns worker processes that may not have the repo root on `sys.path`.
The loader ensures the repo root is injected into `PYTHONPATH` before creating the vLLM `LLM`, so workers can import:

- `neuro_manipulation.repe.vllm_worker_extension.NMRepControlWorkerExtension`

### 3) Controller payload must be RPC-serializable

Do not send `torch.Tensor` directly in the RPC state. The hook converts tensors to Python lists before RPC, and workers convert lists back to tensors in the forward hook.

This avoids failures like:
- `list object has no attribute shape`
- `ValueError: too many dimensions 'str'`

### 4) FlashAttention ABI issues (torch upgrade)

If you see an import error like:

`flash_attn_2_cuda...so: undefined symbol ...`

it means the installed `flash-attn` binary is not compatible with your current `torch` build. You can force vLLM to use a different attention backend without touching the rest of the hook workflow:

- Set `VLLM_ATTENTION_BACKEND=TRITON_ATTN`, or
- Provide `attention_backend: "TRITON_ATTN"` via vLLM kwargs in your experiment config.

## Minimal YAML knobs (EmotionExperimentEngine)

When using `emotion_experiment_engine`, vLLM kwargs are passed via `loading_config.additional_vllm_kwargs`.

Example:

```yaml
loading_config:
  gpu_memory_utilization: 0.85
  max_model_len: 2048
  dtype: "bfloat16"
  additional_vllm_kwargs:
    max_num_seqs: 8
    attention_backend: "TRITON_ATTN"
```

## Usage

See the `if __name__ == "__main__":` block in `rep_control_vllm_hook.py` for an example demonstrating initialization, baseline generation, and controlled generation.  
