import logging
import sys
import traceback
from typing import Any, Dict, Optional

import numpy as np
import torch
import torch.distributed as dist
from vllm import LLM, SamplingParams

from neuro_manipulation.model_layer_detector import ModelLayerDetector

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)

def _to_rpc_serializable_tensor_payload(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, dict):
        return {k: _to_rpc_serializable_tensor_payload(v) for k, v in value.items()}
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (list, tuple)):
        return [_to_rpc_serializable_tensor_payload(v) for v in value]
    return value


# --- Hook Function ---
def hook_fn_rep_control(module, args, output):
    """
    Forward hook function that applies Representation Control modifications.
    It checks for state attached directly to the module instance.
    """
    rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else "N/A"
    if not hasattr(module, "_rep_control_state") or module._rep_control_state is None:
        # logger.debug(f"Rank {rank} - Module {module.__class__.__name__} - No RepControl state found, skipping.")
        return output  # No state set, do nothing

    state = module._rep_control_state
    mask = state.get("mask")
    token_pos = state.get("token_pos")
    normalize = state.get("normalize", False)
    operator_name = state.get("operator_name")
    tp_size = state.get("tp_size", 1)
    tp_rank = state.get("tp_rank")
    full_controller = state.get("controller")

    if full_controller is None or operator_name is None:
        logger.warning(
            f"Rank {rank} - Module {module.__class__.__name__} - RepControl state incomplete (missing controller or operator_name), skipping."
        )
        return output

    try:
        # --- Define operator function locally based on name --- Start
        if operator_name == "linear_comb":
            operator_fn = lambda current, controller: current + controller
        elif operator_name == "piecewise_linear":
            # Note: This requires controller to be compatible shape for sum
            operator_fn = lambda current, controller: current + controller * torch.sign(
                (current * controller).sum(-1, keepdim=True)
            )
        else:
            logger.error(
                f"Rank {rank} - Unknown operator_name in hook: {operator_name}. Skipping modification."
            )
            return output
        # --- Define operator function locally based on name --- End

        # Identify the target tensor (usually the first element if output is a tuple)
        if isinstance(output, torch.Tensor):
            modified = output
        elif (
            isinstance(output, tuple)
            and len(output) > 0
            and isinstance(output[0], torch.Tensor)
        ):
            modified = output[0]
        else:
            logger.error(
                f"Rank {rank} - Module {module.__class__.__name__} output type is not a Tensor or Tuple[Tensor, ...]: {type(output)}. Cannot modify."
            )
            return output

        if isinstance(full_controller, (list, tuple, np.ndarray)):
            full_controller = torch.as_tensor(full_controller)

        modified_is_packed = modified.ndim == 2
        modified_is_batched = modified.ndim == 3
        if not (modified_is_packed or modified_is_batched):
            logger.error(
                f"Rank {rank} - Unsupported activation tensor shape for RepControl: {modified.shape}"
            )
            return output

        logger.debug(
            f"Rank {rank} - Applying RepControl hook on {module.__class__.__name__}. Full Controller shape: {getattr(full_controller, 'shape', None)}, Modified shape: {modified.shape}, TP size: {tp_size}"
        )

        # --- Apply modification logic (Tensor Parallel Aware) ---
        norm_pre = torch.norm(modified, dim=-1, keepdim=True) if normalize else None

        # Ensure mask is on the correct device and dtype
        if mask is None:
            if "position_ids" in state.get("kwargs", {}):  # Check if kwargs were passed
                # Basic mask handling if position_ids available (less robust than original)
                pos = state["kwargs"]["position_ids"]
                zero_indices = (pos == 0).cumsum(1).argmax(1, keepdim=True)
                col_indices = torch.arange(pos.size(1), device=pos.device).unsqueeze(0)
                target_shape = modified.shape
                mask = (
                    (col_indices >= zero_indices)
                    .float()
                    .reshape(target_shape[0], target_shape[1], 1)
                )
                mask = mask.to(modified.dtype)
                logger.debug(f"Rank {rank} - Generated mask from position_ids.")
            else:
                mask = 1.0  # Default mask
        else:
            if isinstance(mask, (list, tuple, np.ndarray)):
                mask = torch.as_tensor(mask)
            if isinstance(mask, torch.Tensor):
                mask = mask.to(modified.device, dtype=modified.dtype)

        # --- Conditional Tensor Parallel Slicing Logic --- Start
        full_controller = full_controller.to(modified.device, dtype=modified.dtype)
        controller_ready = None  # Initialize
        modified_dim = modified.shape[-1]
        full_dim = full_controller.shape[-1]
        seq_len = modified.shape[0] if modified_is_packed else modified.shape[1]
        batch_size = 1 if modified_is_packed else modified.shape[0]

        logger.debug(
            f"Rank {rank} - Comparing shapes: modified_dim={modified_dim}, full_dim={full_dim}, tp_size={tp_size}"
        )

        # Determine if slicing is needed based on the observed output shape
        if modified_dim == full_dim:
            # Output is full tensor, use controller as is (no slicing)
            logger.debug(
                f"Rank {rank} - Output shape ({modified_dim}) matches full controller dim. Using full controller."
            )
            controller_slice = full_controller
        elif (
            tp_size > 1 and modified_dim == full_dim // tp_size
        ):  # TODO: Test if this branch is correct
            # Output shape matches expected sharded dimension, slice the controller
            logger.debug(
                f"Rank {rank} - Output shape ({modified_dim}) matches sharded dim ({full_dim // tp_size}). Slicing controller."
            )
            effective_rank = tp_rank if isinstance(tp_rank, int) else rank
            if not isinstance(effective_rank, int):
                logger.error(
                    f"Rank {rank} - Cannot slice controller for TP: no integer rank available (tp_rank={tp_rank})."
                )
                return output
            if full_dim % tp_size != 0:
                logger.error(
                    f"Rank {rank} - Full hidden dimension {full_dim} is not divisible by TP size {tp_size}. Cannot slice controller accurately."
                )
                return output  # Cannot proceed reliably
            chunk_size = full_dim // tp_size
            start_idx = effective_rank * chunk_size
            end_idx = (effective_rank + 1) * chunk_size
            controller_slice = full_controller[..., start_idx:end_idx]
            logger.debug(
                f"Rank {rank} - Sliced controller: indices [{start_idx}:{end_idx}], shape {controller_slice.shape}"
            )
        else:
            # Unexpected shape mismatch
            logger.error(
                f"Rank {rank} - Unexpected shape mismatch! Modified dim ({modified_dim}), Full controller dim ({full_dim}), TP size ({tp_size}). Cannot determine how to apply controller."
            )
            return output  # Cannot proceed

        # Reshape controller_slice to be broadcastable (same logic as before)
        if len(controller_slice.shape) == 1:
            controller_ready = controller_slice.unsqueeze(0)
        elif len(controller_slice.shape) == 2:
            if controller_slice.shape[0] in (1, seq_len):
                controller_ready = controller_slice
            else:
                logger.warning(
                    f"Rank {rank} - Controller has unexpected 2D shape: {controller_slice.shape} for seq_len={seq_len}."
                )
                return output
        elif len(controller_slice.shape) == 3:
            if controller_slice.shape[1] == 1:
                controller_ready = controller_slice.expand(-1, seq_len, -1)
            elif controller_slice.shape[1] == seq_len:
                controller_ready = controller_slice
            else:
                logger.warning(
                    f"Rank {rank} - Controller sequence length ({controller_slice.shape[1]}) does not match activation sequence length ({seq_len}) and is not 1. Trying broadcast."
                )
                controller_ready = controller_slice  # Hope broadcasting works
        else:
            logger.error(
                f"Rank {rank} - Unexpected controller_slice shape after potential slicing: {controller_slice.shape}"
            )
            return output

        logger.debug(
            f"Rank {rank} - Final controller_ready shape for operation: {controller_ready.shape}"
        )
        # --- Conditional Tensor Parallel Slicing Logic --- End

        # Make controller token-indexable for the observed activation layout.
        if modified_is_packed:
            if controller_ready.ndim == 2 and controller_ready.shape[0] == 1:
                controller_ready = controller_ready.expand(seq_len, -1)
            elif controller_ready.ndim == 3:
                if controller_ready.shape[0] == 1 and controller_ready.shape[1] == seq_len:
                    controller_ready = controller_ready[0]
                else:
                    logger.error(
                        f"Rank {rank} - Cannot align controller to packed activations: controller_ready={controller_ready.shape}, modified={modified.shape}"
                    )
                    return output
        else:
            if controller_ready.ndim == 2:
                if controller_ready.shape[0] == 1:
                    controller_ready = controller_ready.expand(seq_len, -1)
                if controller_ready.shape[0] == seq_len:
                    controller_ready = controller_ready.unsqueeze(0).expand(batch_size, -1, -1)
            elif controller_ready.ndim == 3:
                if controller_ready.shape[0] == 1:
                    controller_ready = controller_ready.expand(batch_size, -1, -1)
                elif controller_ready.shape[0] != batch_size:
                    logger.error(
                        f"Rank {rank} - Cannot align controller to batched activations: controller_ready={controller_ready.shape}, modified={modified.shape}"
                    )
                    return output

        controller_masked = controller_ready * mask

        if modified_is_packed:
            if isinstance(token_pos, int):
                modified[token_pos] = operator_fn(
                    modified[token_pos], controller_masked[token_pos]
                )
            elif isinstance(token_pos, (list, tuple, np.ndarray)):
                if isinstance(token_pos, np.ndarray):
                    token_pos = token_pos.tolist()
                modified[token_pos] = operator_fn(
                    modified[token_pos], controller_masked[token_pos]
                )
            elif isinstance(token_pos, str):
                len_token = controller_ready.shape[0] if controller_ready.ndim > 1 else 1
                if token_pos == "end":
                    modified[-len_token:] = operator_fn(
                        modified[-len_token:], controller_masked[-len_token:]
                    )
                elif token_pos == "start":
                    modified[:len_token] = operator_fn(
                        modified[:len_token], controller_masked[:len_token]
                    )
                else:
                    logger.error(f"Rank {rank} - Unknown token position string: {token_pos}")
            else:
                modified = operator_fn(modified, controller_masked)
        else:
            if isinstance(token_pos, int):
                modified[:, token_pos] = operator_fn(
                    modified[:, token_pos], controller_masked[:, token_pos]
                )
            elif isinstance(token_pos, (list, tuple, np.ndarray)):
                # Ensure token_pos is usable as index
                if isinstance(token_pos, np.ndarray):
                    token_pos = token_pos.tolist()
                modified[:, token_pos] = operator_fn(
                    modified[:, token_pos], controller_masked[:, token_pos]
                )
            elif isinstance(token_pos, str):
                len_token = (
                    controller_ready.shape[1] if len(controller_ready.shape) > 1 else 1
                )
                if token_pos == "end":
                    modified[:, -len_token:] = operator_fn(
                        modified[:, -len_token:], controller_masked[:, -len_token:]
                    )
                elif token_pos == "start":
                    modified[:, :len_token] = operator_fn(
                        modified[:, :len_token], controller_masked[:, :len_token]
                    )
                else:
                    logger.error(
                        f"Rank {rank} - Unknown token position string: {token_pos}"
                    )
            else:  # Apply to all tokens if token_pos is None or not recognized
                modified = operator_fn(modified, controller_masked)

        # Apply normalization if requested
        if normalize:
            norm_post = torch.norm(modified, dim=-1, keepdim=True)
            # Avoid division by zero
            norm_post = torch.where(
                norm_post == 0, torch.ones_like(norm_post), norm_post
            )
            modified = modified / norm_post * norm_pre

        # --- End Modification Logic ---

        # Reconstruct the output if it was a tuple
        if isinstance(output, tuple):
            final_output = (modified,) + output[1:]
        else:
            final_output = modified

        logger.debug(
            f"Rank {rank} - RepControl hook applied successfully on {module.__class__.__name__}."
        )
        return final_output

    except Exception as e:
        logger.error(
            f"Rank {rank} - Error in RepControl hook for {module.__class__.__name__}: {e}",
            exc_info=True,
        )
        # Ensure state is cleared on error to prevent broken state? Or rely on external reset?
        # For now, just return original output.
        # delattr(module, '_rep_control_state') # Maybe too aggressive
        return output


# --- RPC Functions ---
def _get_nested_module(model, module_path):
    """Helper to get a nested module by path."""
    modules = module_path.split(".")
    current_module = model
    for mod_name in modules:
        if hasattr(current_module, mod_name):
            current_module = getattr(current_module, mod_name)
        else:
            # Handle list-like access (e.g., model.layers[0])
            try:
                idx = int(mod_name)
                current_module = current_module[idx]
            except (ValueError, IndexError, TypeError):
                logger.error(
                    f"Could not find module part: {mod_name} in path {module_path}"
                )
                return None
    return current_module


def _find_target_module(worker_self, layer_index, block_name):
    """Finds the target module on the worker."""
    if not hasattr(worker_self, "model_runner") or not hasattr(
        worker_self.model_runner, "model"
    ):
        logger.error(
            f"RPC: Worker Rank {worker_self.rank} could not find model_runner.model"
        )
        return None
    model = worker_self.model_runner.model

    # Use ModelLayerDetector to find the base layer
    layers = ModelLayerDetector.get_model_layers(
        model
    )  # Expects list of top-level layers
    if not layers or layer_index >= len(layers):
        logger.warning(
            f"RPC: Worker Rank {worker_self.rank} - Layer index {layer_index} out of bounds ({len(layers)} layers found)."
        )
        return None
    target_layer = layers[layer_index]

    # If block_name is 'decoder_block' or similar top-level name, target is the layer itself
    # If block_name specifies a submodule like 'mlp' or 'self_attn', navigate to it
    if (
        block_name == "decoder_block"
    ):  # Assuming 'decoder_block' means the main layer module
        return target_layer
    elif hasattr(target_layer, block_name):
        return getattr(target_layer, block_name)
    else:
        logger.warning(
            f"RPC: Worker Rank {worker_self.rank} - Could not find block '{block_name}' within layer {layer_index} ({target_layer.__class__.__name__}). Targeting layer itself."
        )
        # Fallback to targeting the whole layer if specific block not found
        return target_layer  # Or return None if strict matching is required


def _register_hook_on_worker_rpc(worker_self, layer_index, block_name, hook_func):
    """RPC function to register a forward hook on a specific module on a worker."""
    rank = worker_self.rank
    try:
        target_module = _find_target_module(worker_self, layer_index, block_name)
        if target_module is None:
            logger.error(
                f"RPC: Worker Rank {rank} failed to find target module for layer {layer_index}, block {block_name}."
            )
            return False

        logger.debug(
            f"RPC: Worker Rank {rank} registering hook {hook_func.__name__} to {target_module.__class__.__name__} (Layer {layer_index}, Block {block_name})"
        )
        handle_key = (layer_index, block_name, hook_func.__name__)
        if hasattr(worker_self, "_hook_handles"):
            existing = worker_self._hook_handles.get(handle_key)
            if existing is not None:
                existing_module_id, _existing_handle = existing
                if existing_module_id == id(target_module):
                    return True
        handle = target_module.register_forward_hook(hook_func)

        if handle:
            logger.debug(f"RPC: Worker Rank {rank} hook registered successfully.")
            # Store the handle if we need to remove it later (more complex)
            # We might need a way to manage multiple handles per module if needed
            if not hasattr(worker_self, "_hook_handles"):
                worker_self._hook_handles = {}
            # Store handle associated with the specific module instance id to be robust
            worker_self._hook_handles[handle_key] = (id(target_module), handle)
            return True
        else:
            logger.error(f"RPC: Worker Rank {rank} hook registration failed.")
            return False
    except Exception as e:
        logger.error(
            f"RPC: Worker Rank {rank} error during hook registration: {e}",
            exc_info=True,
        )
        return False


def _set_controller_state_on_worker_rpc(worker_self, layer_index, block_name, state):
    """RPC function to set the control state on the target module on a worker."""
    rank = worker_self.rank
    try:
        target_module = _find_target_module(worker_self, layer_index, block_name)
        if target_module is None:
            logger.error(
                f"RPC: Worker Rank {rank} failed to find target module for layer {layer_index}, block {block_name} to set state."
            )
            return False

        logger.debug(
            f"RPC: Worker Rank {rank} setting RepControl state on {target_module.__class__.__name__} (Layer {layer_index}, Block {block_name}). State keys: {list(state.keys())}"
        )
        # --- Add Debug Logging Here --- Start
        received_tp_size = state.get("tp_size", "Not Found")
        logger.debug(
            f"RPC: Worker Rank {rank} - Received state dictionary contains tp_size: {received_tp_size} (Type: {type(received_tp_size)})"
        )
        # --- Add Debug Logging Here --- End
        # Attach state directly to the module instance
        if "tp_rank" not in state and isinstance(worker_self.rank, int):
            state["tp_rank"] = worker_self.rank
        target_module._rep_control_state = state
        return True
    except Exception as e:
        logger.error(
            f"RPC: Worker Rank {rank} error setting controller state: {e}",
            exc_info=True,
        )
        return False


def _reset_controller_state_on_worker_rpc(worker_self, layer_index, block_name):
    """RPC function to remove the control state from the target module on a worker."""
    rank = worker_self.rank
    try:
        target_module = _find_target_module(worker_self, layer_index, block_name)
        if target_module is None:
            # Don't error if module not found, maybe it wasn't used.
            logger.warning(
                f"RPC: Worker Rank {rank} could not find target module for layer {layer_index}, block {block_name} to reset state. Skipping."
            )
            return True  # Indicate success as state is effectively not present

        if hasattr(target_module, "_rep_control_state"):
            logger.debug(
                f"RPC: Worker Rank {rank} resetting RepControl state on {target_module.__class__.__name__} (Layer {layer_index}, Block {block_name})"
            )
            delattr(target_module, "_rep_control_state")
        else:
            logger.debug(
                f"RPC: Worker Rank {rank} - No RepControl state found on {target_module.__class__.__name__} to reset."
            )

        return True
    except Exception as e:
        logger.error(
            f"RPC: Worker Rank {rank} error resetting controller state: {e}",
            exc_info=True,
        )
        return False


# --- Main Class ---
class RepControlVLLMHook:
    def __init__(
        self,
        model: LLM,
        tokenizer,
        layers: list[int],
        block_name: str,
        control_method: str,
        tensor_parallel_size: int = 1,
    ):
        """
        Initializes RepControlVLLMHook.

        Args:
            model: The vLLM LLM instance.
            tokenizer: The tokenizer.
            layers: List of layer indices to apply control.
            block_name: Name of the block/module within the layer to hook
                      (e.g., 'decoder_block', 'mlp', 'self_attn').
                      'decoder_block' usually refers to the main layer module.
            control_method: The control method ('reading_vec' supported).
            tensor_parallel_size: The tensor parallel size used by the vLLM model.
        """
        self.model = model
        self.tokenizer = tokenizer
        self.layers = layers
        self.block_name = block_name
        self.control_method = control_method
        self.hook_handles: Dict[Any, Any] = {}
        self.tp_size = tensor_parallel_size  # Use the passed value

        if control_method != "reading_vec":
            raise ValueError(f"Control method '{control_method}' not supported yet.")
        if not hasattr(self.model, "llm_engine") or not hasattr(
            self.model.llm_engine, "collective_rpc"
        ):
            raise AttributeError(
                "Provided model does not have 'llm_engine.collective_rpc'. Is it a valid vLLM LLM object?"
            )

        # --- Get and store Tensor Parallel Size --- Start
        if hasattr(self.model.llm_engine, "parallel_config") and hasattr(
            self.model.llm_engine.parallel_config, "tensor_parallel_size"
        ):
            self.tp_size = self.model.llm_engine.parallel_config.tensor_parallel_size
            logger.info(f"Stored Tensor Parallel Size during init: {self.tp_size}")
        else:
            logger.warning(
                "Could not detect tensor_parallel_size from engine's parallel_config during init. Assuming tp_size=1."
            )
        # --- Get and store Tensor Parallel Size --- End

        logger.info(
            f"Initializing RepControlVLLMHook for layers {layers}, block '{block_name}' with TP size: {self.tp_size}."
        )

        # Register the hook on all target layers via RPC
        for layer_id in self.layers:
            logger.info(
                f"Registering hook for layer {layer_id}, block '{block_name}'..."
            )
            rpc_results = self.model.llm_engine.collective_rpc(
                "_nm_repcontrol_register_hook",
                args=(layer_id, self.block_name),
            )
            logger.info(
                f"RPC hook registration results for layer {layer_id}: {rpc_results}"
            )
            if not any(rpc_results):
                logger.warning(
                    f"Failed to register hook on any worker for layer {layer_id}. Control might not work."
                )
            # Optionally store success status or handles if needed for removal later

    def _get_operator_fn(self, operator_name="linear_comb"):
        """Returns the operator function based on name."""
        if operator_name == "linear_comb":
            return lambda current, controller: current + controller
        elif operator_name == "piecewise_linear":
            # Note: This requires controller to be compatible shape for sum
            return lambda current, controller: current + controller * torch.sign(
                (current * controller).sum(-1, keepdim=True)
            )
        # Add other operators as needed
        else:
            raise NotImplementedError(f"Operator '{operator_name}' not implemented.")

    def __call__(
        self,
        text_inputs: list[str],
        activations: Optional[Dict[int, torch.Tensor]] = None,  # Dict mapping layer_id to activation tensor
        token_pos=None,
        masks=None,  # Can be a single mask or dict mapping layer_id to mask
        normalize=False,
        operator="linear_comb",
        **kwargs,
    ):
        """
        Generates text with optional representation control.

        Args:
            text_inputs: List of input prompts.
            activations: Dictionary mapping layer indices (subset of self.layers)
                         to control tensors (e.g., reading vectors). If None or empty,
                         runs generation without control.
            token_pos: Position to apply control (int, list, 'start', 'end', None for all).
            masks: Optional masks for applying activations. Can be a single tensor
                   broadcasted, or a dict mapping layer_id to a mask tensor.
            normalize: Whether to normalize activations after applying control.
            operator: How to combine activations ('linear_comb', 'piecewise_linear').
            **kwargs: Additional arguments for vLLM SamplingParams (max_tokens, temp, etc.)
                      and potentially for the hook (e.g., 'position_ids' if needed by mask).

        Returns:
            List of vLLM RequestOutput objects.
        """
        activations = activations or {}
        control_active = len(activations) > 0

        try:
            # 1. Set controller state via RPC if activations are provided
            if control_active:
                logger.info(
                    f"Setting controller state for layers {list(activations.keys())}..."
                )
                set_results = []
                for layer_id, activation_tensor in activations.items():
                    if layer_id not in self.layers:
                        logger.warning(
                            f"Layer {layer_id} in activations not in initialized layers {self.layers}. Skipping."
                        )
                        continue

                    # Prepare state dict for this layer
                    layer_mask = (
                        masks.get(layer_id) if isinstance(masks, dict) else masks
                    )
                    state = {
                        "controller": _to_rpc_serializable_tensor_payload(activation_tensor),
                        "mask": _to_rpc_serializable_tensor_payload(layer_mask),
                        "token_pos": _to_rpc_serializable_tensor_payload(token_pos),
                        "normalize": normalize,
                        "operator_name": operator,
                        "kwargs": _to_rpc_serializable_tensor_payload(kwargs),  # Must be RPC-serializable
                        "tp_size": self.tp_size,  # Use stored tp_size
                    }
                    # Verify the tp_size being sent
                    logger.debug(
                        f"Sending state for layer {layer_id} with tp_size: {state['tp_size']}"
                    )

                    # Send state via RPC
                    rpc_results = self.model.llm_engine.collective_rpc(
                        "_nm_repcontrol_set_state",
                        args=(layer_id, self.block_name, state),
                    )
                    set_results.append(
                        all(rpc_results)
                    )  # Track if successful on all workers
                    logger.debug(
                        f"RPC set state results for layer {layer_id}: {rpc_results}"
                    )

                if not all(set_results):
                    logger.warning(
                        "Failed to set controller state on some workers/layers. Control might be inconsistent."
                    )

            # 2. Prepare inputs and sampling parameters
            # Note: Tokenization should handle padding correctly based on vLLM requirements
            # For vLLM generate, we typically pass prompt strings directly, or token IDs
            # Passing prompt strings is usually easier.
            # tokens = self.tokenizer(text_inputs, padding=True, return_tensors="pt") # Not usually needed for llm.generate
            # prompt_token_ids = tokens['input_ids']

            sampling_params = SamplingParams(
                max_tokens=kwargs.get("max_new_tokens", 40000),
                temperature=kwargs.get(
                    "temperature", 0.0
                ),  # Default to 0 for reproducibility if baseline needed
                repetition_penalty=kwargs.get("repetition_penalty", 1.0),
                top_p=kwargs.get("top_p", 1.0),
                seed=kwargs.get("seed", None),
                # Add other params from kwargs if needed
                # **{k: v for k, v in kwargs.items() if k not in ['max_new_tokens', 'temperature', 'repetition_penalty', 'top_p']}
            )

            # 3. Run generation
            logger.info(
                f"Running generation with{' control active' if control_active else 'out control'}..."
            )
            outputs = self.model.generate(text_inputs, sampling_params)
            logger.info("Generation finished.")

        finally:
            # 4. Reset controller state via RPC if it was active
            if control_active:
                logger.info(
                    f"Resetting controller state for layers {list(activations.keys())}..."
                )
                reset_results = []
                # Reset state only for layers where it was potentially set
                for layer_id in activations.keys():
                    if layer_id not in self.layers:
                        continue  # Skip layers not managed by this instance

                    rpc_results = self.model.llm_engine.collective_rpc(
                        "_nm_repcontrol_reset_state",
                        args=(layer_id, self.block_name),
                    )
                    reset_results.append(all(rpc_results))
                    logger.debug(
                        f"RPC reset state results for layer {layer_id}: {rpc_results}"
                    )
                if not all(reset_results):
                    logger.warning(
                        "Failed to reset controller state on some workers/layers."
                    )

        return outputs

    def remove_hooks(self):
        """Attempts to remove hooks - Requires storing handles or more complex RPC."""
        # This is complex because handles are on workers. Needs another RPC call.
        # Simplified version: Assume collective_rpc can trigger removal if handles were stored.
        logger.warning(
            "Hook removal is experimental and may require specific handle management."
        )
        # Example (conceptual - needs matching RPC function _remove_hook_on_worker_rpc):
        # for layer_id in self.layers:
        #     rpc_results = self.model.llm_engine.collective_rpc(
        #         _remove_hook_on_worker_rpc,
        #         args=(layer_id, self.block_name, hook_fn_rep_control.__name__) # Need to identify hook
        #     )
        #     logger.info(f"RPC hook removal results for layer {layer_id}: {rpc_results}")
        pass  # Placeholder


# --- Example Usage (requires CUDA and vLLM installation) ---
if __name__ == "__main__":
    import gc

    import torch
    from transformers import AutoTokenizer

    # --- Configuration ---
    model_name = "meta-llama/Llama-3.1-8B-Instruct"  # Make sure you have access
    num_gpus = 1  # Adjust based on your hardware
    if torch.cuda.is_available():
        detected_gpus = torch.cuda.device_count()
        logger.info(f"CUDA available. Found {detected_gpus} GPUs.")
        num_gpus = min(
            num_gpus, detected_gpus
        )  # Use specified or max available, whichever is smaller
    else:
        logger.error("CUDA not available. This example requires GPU.")
        sys.exit(1)

    layers_to_control = [10, 15]  # Example layers
    block_to_hook = "decoder_block"  # Hook the main output of these layers
    control_method = "reading_vec"
    prompt = "The capital of France is"
    dummy_vec_dim = (
        4096  # Adjust to the model's hidden size (e.g., Llama3.1-8B is 4096)
    )

    llm = None
    try:
        # --- Initialize Model and Tokenizer ---
        logger.info(f"Loading tokenizer for {model_name}...")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        # Add padding token if missing (common for Llama)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            logger.info("Set pad_token to eos_token.")

        logger.info(f"Loading LLM {model_name} with tensor_parallel_size={num_gpus}...")
        llm = LLM(
            model=model_name,
            tokenizer=tokenizer.name_or_path,  # Pass path instead of object sometimes helps vLLM
            enforce_eager=True,  # Eager mode might be necessary for hooks to work reliably
            trust_remote_code=True,
            tensor_parallel_size=num_gpus,
            gpu_memory_utilization=0.85,  # Adjust as needed
            max_num_seqs=16,
        )  # Adjust as needed
        logger.info("LLM loaded successfully.")

        # --- Initialize RepControl ---
        logger.info(
            f"Initializing RepControlVLLMHook for layers {layers_to_control}, block '{block_to_hook}'..."
        )
        rep_control = RepControlVLLMHook(
            llm, tokenizer, layers_to_control, block_to_hook, control_method
        )
        logger.info("RepControlVLLMHook initialized.")

        # --- Baseline Generation ---
        logger.info("--- Running Baseline Generation ---")
        baseline_outputs = rep_control(
            [prompt], max_new_tokens=10, temperature=0.0
        )  # No activations passed
        baseline_text = baseline_outputs[0].outputs[0].text
        logger.info(f"Baseline Output: '{baseline_text}'")

        # --- Controlled Generation ---
        logger.info("--- Running Controlled Generation ---")
        # Create dummy activation vectors (replace with actual reading vectors)
        control_activations = {}
        for layer_id in layers_to_control:
            # Simple dummy vector: tensor of ones
            control_activations[layer_id] = (
                torch.ones(dummy_vec_dim, dtype=torch.float16) * 0.1
            )  # Smaller magnitude

        controlled_outputs = rep_control(
            [prompt],
            activations=control_activations,
            max_new_tokens=10,
            temperature=0.0,  # Keep temp 0 for comparison
            operator="linear_comb",  # 'linear_comb' or 'piecewise_linear'
            normalize=False,
            token_pos=None,  # Apply to all tokens in the layer output
        )
        controlled_text = controlled_outputs[0].outputs[0].text
        logger.info(f"Controlled Output: '{controlled_text}'")

        # --- Comparison ---
        if baseline_text != controlled_text:
            logger.info("SUCCESS: Baseline and controlled outputs differ.")
        else:
            logger.warning(
                "WARNING: Baseline and controlled outputs are the same. Control might not have worked as expected."
            )

    except Exception as e:
        logger.error(f"An error occurred: {e}", exc_info=True)
        print(f"Traceback: {traceback.format_exc()}")  # Print traceback clearly

    finally:
        # --- Cleanup ---
        logger.info("Cleaning up resources...")
        if llm is not None:
            # Explicitly delete engine might help, but standard 'del' and gc should be okay
            # if hasattr(llm, 'llm_engine'): del llm.llm_engine
            del llm
            logger.info("LLM instance deleted.")
        torch.cuda.empty_cache()
        gc.collect()
        logger.info("CUDA cache cleared and garbage collected.")
