"""
This implementation may have bugs. 
Tensor parallel results are not consistent with single GPU results.
More details:
python -m unittest neuro_manipulation/repe/tests/test_sequence_prob_tp_consistency.py
"""

import torch
import numpy as np
import logging
import sys
from vllm import LLM, SamplingParams
from neuro_manipulation.model_layer_detector import ModelLayerDetector
import traceback
import torch.distributed as dist
from typing import List, Dict, Union, Optional
import torch.nn.functional as F

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    stream=sys.stdout)
logger = logging.getLogger(__name__)

# --- Combined Hook Function ---
def hook_fn_combined(module, args, output):
    """
    Combined forward hook function that:
    1. Applies Representation Control modifications (like hook_fn_rep_control)
    2. Captures logits for sequence probability calculation
    3. Optionally records layer-wise logits
    
    It checks for state attached directly to the module instance.
    """
    rank = dist.get_rank() if dist.is_initialized() else 'N/A'
    
    # Check if any state is set
    has_rep_control = hasattr(module, '_rep_control_state') and module._rep_control_state is not None
    has_sequence_prob = hasattr(module, '_sequence_prob_state') and module._sequence_prob_state is not None
    has_layer_logit_recording = hasattr(module, '_layer_logit_state') and module._layer_logit_state is not None
    
    if not (has_rep_control or has_sequence_prob or has_layer_logit_recording):
        return output
    
    # Identify the target tensor (usually the first element if output is a tuple)
    if isinstance(output, torch.Tensor):
        target_tensor = output
        is_tuple_output = False
    elif isinstance(output, tuple) and len(output) > 0 and isinstance(output[0], torch.Tensor):
        target_tensor = output[0]
        is_tuple_output = True
    else:
        logger.error(f"Rank {rank} - Module {module.__class__.__name__} output type is not a Tensor or Tuple[Tensor, ...]: {type(output)}. Cannot process.")
        return output
    
    modified_tensor = target_tensor
    
    # === 1. APPLY REPRESENTATION CONTROL (if state exists) ===
    if has_rep_control:
        try:
            state = module._rep_control_state
            controller = state.get('controller')
            mask = state.get('mask')
            token_pos = state.get('token_pos')
            normalize = state.get('normalize', False)
            operator_name = state.get('operator_name')
            tp_size = state.get('tp_size', 1)
            full_controller = state.get('controller')

            if full_controller is not None and operator_name is not None:
                logger.debug(f"Rank {rank} - Applying RepControl on {module.__class__.__name__}")
                
                # Define operator function locally based on name
                if operator_name == 'linear_comb':
                    operator_fn = lambda current, controller: current + controller
                elif operator_name == 'piecewise_linear':
                    operator_fn = lambda current, controller: current + controller * torch.sign((current * controller).sum(-1, keepdim=True))
                else:
                    logger.error(f"Rank {rank} - Unknown operator_name in hook: {operator_name}. Skipping modification.")
                    operator_fn = None
                
                if operator_fn is not None:
                    # Apply modification logic (Tensor Parallel Aware)
                    norm_pre = torch.norm(modified_tensor, dim=-1, keepdim=True) if normalize else None

                    # Ensure mask is on the correct device and dtype
                    if mask is not None and isinstance(mask, torch.Tensor):
                        mask = mask.to(modified_tensor.device, dtype=modified_tensor.dtype)
                    elif mask is None and "position_ids" in state.get('kwargs', {}):
                        # Basic mask handling if position_ids available
                        pos = state['kwargs']["position_ids"]
                        zero_indices = (pos == 0).cumsum(1).argmax(1, keepdim=True)
                        col_indices = torch.arange(pos.size(1), device=pos.device).unsqueeze(0)
                        target_shape = modified_tensor.shape
                        mask = (col_indices >= zero_indices).float().reshape(target_shape[0], target_shape[1], 1)
                        mask = mask.to(modified_tensor.dtype)
                        logger.debug(f"Rank {rank} - Generated mask from position_ids.")
                    else:
                        mask = 1.0  # Default mask

                    # Conditional Tensor Parallel Slicing Logic
                    full_controller = full_controller.to(modified_tensor.device, dtype=modified_tensor.dtype)
                    modified_dim = modified_tensor.shape[-1]
                    full_dim = full_controller.shape[-1]

                    logger.debug(f"Rank {rank} - Comparing shapes: modified_dim={modified_dim}, full_dim={full_dim}, tp_size={tp_size}")

                    # Determine if slicing is needed based on the observed output shape
                    if modified_dim == full_dim:
                        # Output is full tensor, use controller as is (no slicing)
                        logger.debug(f"Rank {rank} - Output shape ({modified_dim}) matches full controller dim. Using full controller.")
                        controller_slice = full_controller
                    elif tp_size > 1 and modified_dim == full_dim // tp_size:
                        # Output shape matches expected sharded dimension, slice the controller
                        logger.debug(f"Rank {rank} - Output shape ({modified_dim}) matches sharded dim ({full_dim // tp_size}). Slicing controller.")
                        if full_dim % tp_size != 0:
                            logger.error(f"Rank {rank} - Full hidden dimension {full_dim} is not divisible by TP size {tp_size}. Cannot slice controller accurately.")
                            controller_slice = None
                        else:
                            chunk_size = full_dim // tp_size
                            start_idx = rank * chunk_size
                            end_idx = (rank + 1) * chunk_size
                            controller_slice = full_controller[..., start_idx:end_idx]
                            logger.debug(f"Rank {rank} - Sliced controller: indices [{start_idx}:{end_idx}], shape {controller_slice.shape}")
                    else:
                        # Unexpected shape mismatch
                        logger.error(f"Rank {rank} - Unexpected shape mismatch! Modified dim ({modified_dim}), Full controller dim ({full_dim}), TP size ({tp_size}). Cannot determine how to apply controller.")
                        controller_slice = None

                    if controller_slice is not None:
                        # Reshape controller_slice to be broadcastable
                        controller_ready = None
                        if len(controller_slice.shape) == 1:
                            controller_ready = controller_slice.unsqueeze(0)
                        elif len(controller_slice.shape) == 2:
                            if controller_slice.shape[0] == 1:
                                controller_ready = controller_slice
                            else:
                                logger.warning(f"Rank {rank} - Controller has unexpected 2D shape: {controller_slice.shape}. First dimension should be 1.")
                        elif len(controller_slice.shape) == 3:
                            if controller_slice.shape[1] == 1:
                                controller_ready = controller_slice.expand(-1, modified_tensor.shape[1], -1)
                            elif controller_slice.shape[1] == modified_tensor.shape[1]:
                                controller_ready = controller_slice
                            else:
                                logger.warning(f"Rank {rank} - Controller sequence length ({controller_slice.shape[1]}) does not match activation sequence length ({modified_tensor.shape[1]}) and is not 1. Trying broadcast.")
                                controller_ready = controller_slice  # Hope broadcasting works
                        else:
                            logger.error(f"Rank {rank} - Unexpected controller_slice shape after potential slicing: {controller_slice.shape}")

                        if controller_ready is not None:
                            logger.debug(f"Rank {rank} - Final controller_ready shape for operation: {controller_ready.shape}")
                            
                            controller_masked = controller_ready * mask

                            # Apply the operator based on token position
                            if isinstance(token_pos, int):
                                modified_tensor[:, token_pos] = operator_fn(modified_tensor[:, token_pos], controller_masked[:, token_pos])
                            elif isinstance(token_pos, (list, tuple, np.ndarray)):
                                if isinstance(token_pos, np.ndarray):
                                    token_pos = token_pos.tolist()
                                modified_tensor[:, token_pos] = operator_fn(modified_tensor[:, token_pos], controller_masked[:, token_pos])
                            elif isinstance(token_pos, str):
                                len_token = controller_ready.shape[1] if len(controller_ready.shape) > 1 else 1
                                if token_pos == "end":
                                    modified_tensor[:, -len_token:] = operator_fn(modified_tensor[:, -len_token:], controller_masked[:, -len_token:])
                                elif token_pos == "start":
                                    modified_tensor[:, :len_token] = operator_fn(modified_tensor[:, :len_token], controller_masked[:, :len_token])
                                else:
                                    logger.error(f"Rank {rank} - Unknown token position string: {token_pos}")
                            else:  # Apply to all tokens if token_pos is None or not recognized
                                modified_tensor = operator_fn(modified_tensor, controller_masked)

                            # Apply normalization if requested
                            if normalize:
                                norm_post = torch.norm(modified_tensor, dim=-1, keepdim=True)
                                # Avoid division by zero
                                norm_post = torch.where(norm_post == 0, torch.ones_like(norm_post), norm_post)
                                modified_tensor = modified_tensor / norm_post * norm_pre

                            logger.debug(f"Rank {rank} - RepControl modification applied successfully.")

        except Exception as e:
            logger.error(f"Rank {rank} - Error in RepControl modification for {module.__class__.__name__}: {e}", exc_info=True)
    
    # === 2. CAPTURE LOGITS FOR SEQUENCE PROBABILITY (LM head only) ===
    if has_sequence_prob:
        try:
            state = module._sequence_prob_state
            
            # Check if this is likely the language model head (logits should be vocab_size)
            # We can identify LM head by checking if last dimension is large (vocab size)
            if len(modified_tensor.shape) >= 2 and modified_tensor.shape[-1] > 1000:  # Heuristic for vocab size
                # Store logits in state for later aggregation
                if 'captured_logits' not in state:
                    state['captured_logits'] = []
                
                # Move tensor to CPU to avoid CUDA serialization issues
                cpu_logits = modified_tensor.detach().clone().cpu()
                
                state['captured_logits'].append({
                    'logits': cpu_logits,
                    'rank': rank,
                    'shape': modified_tensor.shape
                })
                logger.debug(f"Rank {rank} - Captured logits from LM head: {modified_tensor.shape}")
            
        except Exception as e:
            logger.error(f"Rank {rank} - Error in logit capture for {module.__class__.__name__}: {e}", exc_info=True)
    
    # === 3. RECORD LAYER-WISE LOGITS (optional feature) ===
    if has_layer_logit_recording:
        try:
            state = module._layer_logit_state
            layer_id = state.get('layer_id', 'unknown')
            
            # Store layer activations/logits
            if 'recorded_logits' not in state:
                state['recorded_logits'] = []
            
            # Move tensor to CPU to avoid CUDA serialization issues
            cpu_tensor = modified_tensor.detach().clone().cpu()
            
            state['recorded_logits'].append({
                'layer_id': layer_id,
                'activations': cpu_tensor,
                'rank': rank,
                'shape': modified_tensor.shape
            })
            logger.debug(f"Rank {rank} - Recorded layer {layer_id} activations: {modified_tensor.shape}")
            
        except Exception as e:
            logger.error(f"Rank {rank} - Error in layer logit recording for {module.__class__.__name__}: {e}", exc_info=True)
    
    # === RECONSTRUCT OUTPUT ===
    if is_tuple_output:
        final_output = (modified_tensor,) + output[1:]
    else:
        final_output = modified_tensor
        
    return final_output

# --- Hook Function Aliases for Backward Compatibility ---
hook_fn_sequence_prob = hook_fn_combined  # For LM head logit capture
hook_fn_rep_control = hook_fn_combined    # For hidden state modification

# --- RPC Functions ---
def _get_nested_module(model, module_path):
    """Helper to get a nested module by path."""
    modules = module_path.split('.')
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
                  logger.error(f"Could not find module part: {mod_name} in path {module_path}")
                  return None
    return current_module

def _find_lm_head_module(worker_self):
    """Finds the language model head module on the worker."""
    rank = worker_self.rank
    
    if not hasattr(worker_self, 'model_runner') or not hasattr(worker_self.model_runner, 'model'):
        logger.error(f"RPC: Worker Rank {rank} could not find model_runner.model")
        return None
    model = worker_self.model_runner.model

    # Common language model head names
    lm_head_names = ['lm_head', 'language_model_head', 'head', 'output_projection', 'embed_out']
    
    for head_name in lm_head_names:
        if hasattr(model, head_name):
            lm_head = getattr(model, head_name)
            logger.debug(f"RPC: Worker Rank {rank} found LM head: {head_name} - {lm_head.__class__.__name__}")
            return lm_head
    
    # If not found directly, try to find it in model structure
    # For some models, it might be nested deeper
    if hasattr(model, 'model') and hasattr(model.model, 'lm_head'):
        return model.model.lm_head
    
    logger.warning(f"RPC: Worker Rank {rank} - Could not find language model head. Available attributes: {list(model.__dict__.keys())}")
    return None

def _find_target_module(worker_self, layer_index, block_name):
    """Finds the target module on the worker."""
    rank = worker_self.rank
    if not hasattr(worker_self, 'model_runner') or not hasattr(worker_self.model_runner, 'model'):
        logger.error(f"RPC: Worker Rank {rank} could not find model_runner.model")
        return None
    model = worker_self.model_runner.model

    # Use ModelLayerDetector to find the base layer
    layers = ModelLayerDetector.get_model_layers(model)
    if not layers or layer_index >= len(layers):
         logger.warning(f"RPC: Worker Rank {rank} - Layer index {layer_index} out of bounds ({len(layers)} layers found).")
         return None
    target_layer = layers[layer_index]

    # If block_name is 'decoder_block' or similar top-level name, target is the layer itself
    # If block_name specifies a submodule like 'mlp' or 'self_attn', navigate to it
    if block_name == "decoder_block":
         return target_layer
    elif hasattr(target_layer, block_name):
         return getattr(target_layer, block_name)
    else:
         logger.warning(f"RPC: Worker Rank {rank} - Could not find block '{block_name}' within layer {layer_index} ({target_layer.__class__.__name__}). Targeting layer itself.")
         return target_layer

def _register_lm_head_hook_rpc(worker_self, hook_func):
    """RPC function to register a forward hook on the language model head."""
    rank = worker_self.rank
    try:
        logger.debug(f"Worker Rank {rank} attempting to find LM head module...")
        lm_head = _find_lm_head_module(worker_self)
        if lm_head is None:
            logger.error(f"RPC: Worker Rank {rank} failed to find LM head module.")
            return False

        logger.debug(f"Worker Rank {rank} found LM head: {lm_head.__class__.__name__}")
        logger.info(f"RPC: Worker Rank {rank} registering hook {hook_func.__name__} to {lm_head.__class__.__name__}")
        handle = lm_head.register_forward_hook(hook_func)

        if handle:
            logger.info(f"RPC: Worker Rank {rank} LM head hook registered successfully.")
            # Store the handle for potential removal
            if not hasattr(worker_self, '_lm_head_hook_handle'):
                 worker_self._lm_head_hook_handle = handle
            return True
        else:
            logger.error(f"RPC: Worker Rank {rank} LM head hook registration failed.")
            return False
    except Exception as e:
        logger.error(f"RPC: Worker Rank {rank} error during LM head hook registration: {e}", exc_info=True)
        return False

def _set_sequence_prob_state_rpc(worker_self, state):
    """RPC function to set the sequence probability state on the LM head module."""
    rank = worker_self.rank
    try:
        lm_head = _find_lm_head_module(worker_self)
        if lm_head is None:
            logger.error(f"RPC: Worker Rank {rank} failed to find LM head module to set state.")
            return False

        logger.debug(f"RPC: Worker Rank {rank} setting SequenceProb state on {lm_head.__class__.__name__}")
        # Attach state directly to the module instance
        lm_head._sequence_prob_state = state
        return True
    except Exception as e:
        logger.error(f"RPC: Worker Rank {rank} error setting sequence probability state: {e}", exc_info=True)
        return False

def _reset_sequence_prob_state_rpc(worker_self):
    """RPC function to remove the sequence probability state from the LM head module."""
    rank = worker_self.rank
    try:
        lm_head = _find_lm_head_module(worker_self)
        if lm_head is None:
            logger.warning(f"RPC: Worker Rank {rank} could not find LM head module to reset state. Skipping.")
            return True

        if hasattr(lm_head, '_sequence_prob_state'):
            logger.debug(f"RPC: Worker Rank {rank} resetting SequenceProb state on {lm_head.__class__.__name__}")
            delattr(lm_head, '_sequence_prob_state')
        else:
            logger.debug(f"RPC: Worker Rank {rank} - No SequenceProb state found on {lm_head.__class__.__name__} to reset.")

        return True
    except Exception as e:
        logger.error(f"RPC: Worker Rank {rank} error resetting sequence probability state: {e}", exc_info=True)
        return False

def _get_captured_logits_rpc(worker_self):
    """RPC function to retrieve captured logits from the LM head module."""
    rank = worker_self.rank
    try:
        lm_head = _find_lm_head_module(worker_self)
        if lm_head is None or not hasattr(lm_head, '_sequence_prob_state'):
            logger.warning(f"RPC: Worker Rank {rank} - No state found for logits retrieval.")
            return None

        state = lm_head._sequence_prob_state
        captured_logits = state.get('captured_logits', [])
        
        logger.debug(f"RPC: Worker Rank {rank} - Retrieved {len(captured_logits)} captured logits")
        return captured_logits
    except Exception as e:
        logger.error(f"RPC: Worker Rank {rank} error retrieving captured logits: {e}", exc_info=True)
        return None

# --- Additional RPC Functions ---
def _register_layer_hook_rpc(worker_self, layer_index, block_name, hook_func):
    """RPC function to register a forward hook on a specific layer module."""
    rank = worker_self.rank
    try:
        target_module = _find_target_module(worker_self, layer_index, block_name)
        if target_module is None:
            logger.error(f"RPC: Worker Rank {rank} failed to find target module for layer {layer_index}, block {block_name}.")
            return False

        logger.info(f"RPC: Worker Rank {rank} registering hook {hook_func.__name__} to {target_module.__class__.__name__} (Layer {layer_index}, Block {block_name})")
        handle = target_module.register_forward_hook(hook_func)

        if handle:
            logger.info(f"RPC: Worker Rank {rank} layer hook registered successfully.")
            # Store the handle for potential removal
            if not hasattr(worker_self, '_layer_hook_handles'):
                 worker_self._layer_hook_handles = {}
            handle_key = (layer_index, block_name, hook_func.__name__)
            worker_self._layer_hook_handles[handle_key] = (id(target_module), handle)
            return True
        else:
            logger.error(f"RPC: Worker Rank {rank} layer hook registration failed.")
            return False
    except Exception as e:
        logger.error(f"RPC: Worker Rank {rank} error during layer hook registration: {e}", exc_info=True)
        return False

def _set_rep_control_state_rpc(worker_self, layer_index, block_name, state):
    """RPC function to set the representation control state on a target module."""
    rank = worker_self.rank
    try:
        target_module = _find_target_module(worker_self, layer_index, block_name)
        if target_module is None:
            logger.error(f"RPC: Worker Rank {rank} failed to find target module for layer {layer_index}, block {block_name} to set state.")
            return False

        logger.debug(f"RPC: Worker Rank {rank} setting RepControl state on {target_module.__class__.__name__} (Layer {layer_index}, Block {block_name})")
        # Attach state directly to the module instance
        target_module._rep_control_state = state
        return True
    except Exception as e:
        logger.error(f"RPC: Worker Rank {rank} error setting rep control state: {e}", exc_info=True)
        return False

def _set_layer_logit_state_rpc(worker_self, layer_index, block_name, state):
    """RPC function to set the layer logit recording state on a target module."""
    rank = worker_self.rank
    try:
        target_module = _find_target_module(worker_self, layer_index, block_name)
        if target_module is None:
            logger.error(f"RPC: Worker Rank {rank} failed to find target module for layer {layer_index}, block {block_name} to set logit state.")
            return False

        logger.debug(f"RPC: Worker Rank {rank} setting layer logit state on {target_module.__class__.__name__} (Layer {layer_index}, Block {block_name})")
        # Attach state directly to the module instance
        target_module._layer_logit_state = state
        return True
    except Exception as e:
        logger.error(f"RPC: Worker Rank {rank} error setting layer logit state: {e}", exc_info=True)
        return False

def _reset_rep_control_state_rpc(worker_self, layer_index, block_name):
    """RPC function to remove the representation control state from a target module."""
    rank = worker_self.rank
    try:
        target_module = _find_target_module(worker_self, layer_index, block_name)
        if target_module is None:
            logger.warning(f"RPC: Worker Rank {rank} could not find target module for layer {layer_index}, block {block_name} to reset state. Skipping.")
            return True

        if hasattr(target_module, '_rep_control_state'):
            logger.debug(f"RPC: Worker Rank {rank} resetting RepControl state on {target_module.__class__.__name__} (Layer {layer_index}, Block {block_name})")
            delattr(target_module, '_rep_control_state')
        else:
            logger.debug(f"RPC: Worker Rank {rank} - No RepControl state found on {target_module.__class__.__name__} to reset.")

        return True
    except Exception as e:
        logger.error(f"RPC: Worker Rank {rank} error resetting rep control state: {e}", exc_info=True)
        return False

def _reset_layer_logit_state_rpc(worker_self, layer_index, block_name):
    """RPC function to remove the layer logit recording state from a target module."""
    rank = worker_self.rank
    try:
        target_module = _find_target_module(worker_self, layer_index, block_name)
        if target_module is None:
            logger.warning(f"RPC: Worker Rank {rank} could not find target module for layer {layer_index}, block {block_name} to reset logit state. Skipping.")
            return True

        if hasattr(target_module, '_layer_logit_state'):
            logger.debug(f"RPC: Worker Rank {rank} resetting layer logit state on {target_module.__class__.__name__} (Layer {layer_index}, Block {block_name})")
            delattr(target_module, '_layer_logit_state')
        else:
            logger.debug(f"RPC: Worker Rank {rank} - No layer logit state found on {target_module.__class__.__name__} to reset.")

        return True
    except Exception as e:
        logger.error(f"RPC: Worker Rank {rank} error resetting layer logit state: {e}", exc_info=True)
        return False

def _get_layer_logits_rpc(worker_self, layer_index, block_name):
    """RPC function to retrieve recorded layer logits from a target module."""
    rank = worker_self.rank
    try:
        target_module = _find_target_module(worker_self, layer_index, block_name)
        if target_module is None or not hasattr(target_module, '_layer_logit_state'):
            logger.warning(f"RPC: Worker Rank {rank} - No logit state found for layer {layer_index}, block {block_name}.")
            return None

        state = target_module._layer_logit_state
        recorded_logits = state.get('recorded_logits', [])
        
        logger.debug(f"RPC: Worker Rank {rank} - Retrieved {len(recorded_logits)} recorded logits from layer {layer_index}")
        return recorded_logits
    except Exception as e:
        logger.error(f"RPC: Worker Rank {rank} error retrieving layer logits: {e}", exc_info=True)
        return None

# --- Main Combined Hook Class ---
class CombinedVLLMHook:
    def __init__(self, model: LLM, tokenizer, 
                 layers: Optional[List[int]] = None, 
                 block_name: str = "decoder_block",
                 tensor_parallel_size: int = 1,
                 enable_sequence_prob: bool = True,
                 enable_rep_control: bool = True,
                 enable_layer_logit_recording: bool = False):
        """
        Initializes CombinedVLLMHook for multiple functionalities.

        Args:
            model: The vLLM LLM instance.
            tokenizer: The tokenizer.
            layers: List of layer indices to apply control/recording (None for LM head only).
            block_name: Name of the block/module within the layer to hook
                      (e.g., 'decoder_block', 'mlp', 'self_attn').
            tensor_parallel_size: The tensor parallel size used by the vLLM model.
            enable_sequence_prob: Whether to enable sequence probability calculation.
            enable_rep_control: Whether to enable representation control.
            enable_layer_logit_recording: Whether to enable layer-wise logit recording.
        """
        self.model = model
        self.tokenizer = tokenizer
        self.layers = layers or []
        self.block_name = block_name
        self.hook_registered = False
        self.tp_size = tensor_parallel_size
        self.enable_sequence_prob = enable_sequence_prob
        self.enable_rep_control = enable_rep_control
        self.enable_layer_logit_recording = enable_layer_logit_recording

        if not hasattr(self.model, 'llm_engine') or not hasattr(self.model.llm_engine, 'collective_rpc'):
             raise AttributeError("Provided model does not have 'llm_engine.collective_rpc'. Is it a valid vLLM LLM object?")

        # Get tensor parallel size from engine if available
        if hasattr(self.model.llm_engine, 'parallel_config') and hasattr(self.model.llm_engine.parallel_config, 'tensor_parallel_size'):
            self.tp_size = self.model.llm_engine.parallel_config.tensor_parallel_size
            logger.info(f"Detected Tensor Parallel Size: {self.tp_size}")
        else:
            logger.warning("Could not detect tensor_parallel_size from engine's parallel_config. Using provided value.")

        logger.info(f"Initializing CombinedVLLMHook with TP size: {self.tp_size}")
        logger.info(f"Features enabled - Sequence Prob: {enable_sequence_prob}, Rep Control: {enable_rep_control}, Layer Recording: {enable_layer_logit_recording}")

        # Register hooks
        self._register_hooks()

    def _register_hooks(self):
        """Register hooks on the appropriate modules."""
        success_count = 0
        
        # Register LM head hook for sequence probability
        if self.enable_sequence_prob:
            logger.info("Registering LM head hook...")
            rpc_results = self.model.llm_engine.collective_rpc(
                _register_lm_head_hook_rpc,
                args=(hook_fn_combined,)
            )
            logger.info(f"RPC hook registration results: {rpc_results}")
            if any(rpc_results):
                logger.info("LM head hook registered successfully.")
                success_count += 1
        
        # Register layer hooks for rep control and/or layer recording
        if (self.enable_rep_control or self.enable_layer_logit_recording) and self.layers:
            for layer_id in self.layers:
                logger.info(f"Registering layer hook for layer {layer_id}, block '{self.block_name}'...")
                rpc_results = self.model.llm_engine.collective_rpc(
                    _register_layer_hook_rpc,
                    args=(layer_id, self.block_name, hook_fn_combined)
                )
                logger.info(f"RPC layer hook registration results for layer {layer_id}: {rpc_results}")
                if any(rpc_results):
                    success_count += 1
        
        if success_count == 0:
            logger.error("Failed to register any hooks. Functionality will not work.")
            self.hook_registered = False
        else:
            self.hook_registered = True
            logger.info(f"Successfully registered {success_count} hook(s).")

    def get_log_prob(self,
                     text_inputs: List[str],
                     target_sequences: List[str],
                     **kwargs) -> List[Dict[str, float]]:
        """
        Calculates log probabilities for target sequences given input prompts
        using prompt_logprobs.
        """
        if not self.enable_sequence_prob:
            raise RuntimeError("Sequence probability functionality is not enabled.")

        if len(text_inputs) != 1:
            raise NotImplementedError("get_log_prob currently only supports a single text_input.")
        
        prompt = text_inputs[0]
        results = []

        try:
            sampling_params = SamplingParams(
                max_tokens=1,
                logprobs=1, 
                prompt_logprobs=1,
                temperature=0.0,
            )

            full_texts = [prompt + seq for seq in target_sequences]
            
            prompt_token_ids = self.tokenizer.encode(prompt, add_special_tokens=False)
            len_prompt_tokens = len(prompt_token_ids)
            
            outputs = self.model.generate(full_texts, sampling_params, use_tqdm=False)
            
            for i, output in enumerate(outputs):
                sequence = target_sequences[i]
                prompt_logprobs_list = output.prompt_logprobs
                
                if prompt_logprobs_list is None:
                    logger.warning(f"Could not get prompt_logprobs for sequence '{sequence}'.")
                    continue
                
                output_prompt_tokens = output.prompt_token_ids
                
                # Align tokenizations
                if output_prompt_tokens[:len_prompt_tokens] != prompt_token_ids:
                    logger.error(f"Tokenizer mismatch for prompt in sequence '{sequence}'. Skipping.")
                    continue

                target_logprobs_list = prompt_logprobs_list[len_prompt_tokens:]
                target_token_ids = output_prompt_tokens[len_prompt_tokens:]

                if not target_logprobs_list:
                    logger.warning(f"No logprobs for target sequence '{sequence}'.")
                    continue

                total_log_prob = 0.0
                valid = True
                for j, token_id in enumerate(target_token_ids):
                    logprob_dict = target_logprobs_list[j]
                    if logprob_dict is None or token_id not in logprob_dict:
                        logger.warning(f"Logprob not found for token {token_id} at position {j} in '{sequence}'.")
                        valid = False
                        break
                    total_log_prob += logprob_dict[token_id].logprob

                if not valid:
                    continue

                prob = torch.exp(torch.tensor(total_log_prob)).item()
                perplexity = torch.exp(torch.tensor(-total_log_prob)).item()

                results.append({
                    'sequence': sequence,
                    'log_prob': total_log_prob,
                    'prob': prob,
                    'perplexity': perplexity,
                    'num_tokens': len(target_token_ids)
                })
        except Exception as e:
            logger.error(f"Error in get_log_prob: {e}", exc_info=True)
            
        return results

    def _calculate_sequence_prob_from_logprobs(self, 
                                             outputs: List,
                                             target_token_ids: torch.Tensor,
                                             sequence: str) -> Optional[torch.Tensor]:
        """[DEPRECATED]"""
        logger.warning("`_calculate_sequence_prob_from_logprobs` is deprecated.")
        return None
    
    def generate_with_control(self, prompts, activations=None, 
                             record_layer_logits=False, operator='linear_comb', 
                             normalize=False, token_pos=None, masks:Dict[int, torch.Tensor]=None, **sampling_kwargs):
        """Generate text with optional representation control."""
        from vllm import SamplingParams
        
        # Check if rep control features are being used when disabled
        if (activations is not None) and not self.enable_rep_control:
            raise RuntimeError("Representation control functionality is not enabled.")
        
        # Check if layer recording is being used when disabled  
        if record_layer_logits and not self.enable_layer_logit_recording:
            raise RuntimeError("Layer logit recording functionality is not enabled.")
        
        # Set up representation control if activations provided
        if activations and self.enable_rep_control:
            self._set_control_activations(activations, operator, normalize, token_pos, masks)
        
        # Set up layer recording if requested
        if record_layer_logits and self.enable_layer_logit_recording:
            self._set_layer_recording(True)
        
        try:
            # Default sampling parameters
            sampling_params = SamplingParams(
                max_tokens=sampling_kwargs.get('max_new_tokens', 40000),
                temperature=sampling_kwargs.get('temperature', 0.7),
                top_p=sampling_kwargs.get('top_p', 1.0)
            )
            
            # Generate
            outputs = self.model.generate(prompts, sampling_params)
            return outputs
            
        finally:
            # Clean up control states
            if activations and self.enable_rep_control:
                self._clear_control_activations()
            if record_layer_logits and self.enable_layer_logit_recording:
                self._set_layer_recording(False)

    def _set_control_activations(self, activations, operator='linear_comb', normalize=False, token_pos=None, masks:Dict[int, torch.Tensor]=None):
        """Set control activations for representation control."""
        for layer_id, control_vector in activations.items():
            if layer_id in self.layers:
                layer_mask = masks.get(layer_id) if isinstance(masks, dict) else masks
                state = {
                    'controller': control_vector,
                    'mask': layer_mask,
                    'token_pos': token_pos,
                    'normalize': normalize,
                    'operator_name': operator,
                    'tp_size': self.tp_size,
                    'kwargs': {}
                }
                
                # Set state on workers
                rpc_results = self.model.llm_engine.collective_rpc(
                    _set_rep_control_state_rpc,
                    args=(layer_id, self.block_name, state)
                )
                logger.debug(f"Set control activation for layer {layer_id}: {any(rpc_results)}")

    def _clear_control_activations(self):
        """Clear control activations."""
        for layer_id in self.layers:
            rpc_results = self.model.llm_engine.collective_rpc(
                _reset_rep_control_state_rpc,
                args=(layer_id, self.block_name)
            )
            logger.debug(f"Cleared control activation for layer {layer_id}: {any(rpc_results)}")

    def _set_layer_recording(self, enable):
        """Enable or disable layer recording."""
        for layer_id in self.layers:
            if enable:
                state = {'layer_id': layer_id}
                rpc_results = self.model.llm_engine.collective_rpc(
                    _set_layer_logit_state_rpc,
                    args=(layer_id, self.block_name, state)
                )
            else:
                # Don't reset layer state immediately - let get_layer_logits() handle cleanup
                # This prevents losing data before it can be retrieved
                logger.debug(f"Layer recording disabled for layer {layer_id} (state will be cleared on next get_layer_logits call)")
                pass
            logger.debug(f"Layer recording {'enabled' if enable else 'disabled'} for layer {layer_id}: {any(rpc_results) if enable else True}")

    def get_layer_logits(self):
        """Get recorded layer logits."""
        if not self.enable_layer_logit_recording:
            raise RuntimeError("Layer logit recording functionality is not enabled.")
            
        layer_data = {}
        for layer_id in self.layers:
            rpc_results = self.model.llm_engine.collective_rpc(
                _get_layer_logits_rpc,
                args=(layer_id, self.block_name)
            )
            
            # Aggregate results from all workers
            all_logits = []
            for result in rpc_results:
                if result:
                    all_logits.extend(result)
            
            if all_logits:
                layer_data[layer_id] = all_logits
                
        # Clear the recording state after retrieving data
        for layer_id in self.layers:
            rpc_results = self.model.llm_engine.collective_rpc(
                _reset_layer_logit_state_rpc,
                args=(layer_id, self.block_name)
            )
            logger.debug(f"Cleared layer recording state for layer {layer_id}: {any(rpc_results)}")
                
        return layer_data

    def remove_hooks(self):
        """Remove registered hooks (placeholder for future implementation)."""
        logger.warning("Hook removal not yet implemented for CombinedVLLMHook.")
        pass


if __name__ == "__main__":
    import torch
    import gc
    from transformers import AutoTokenizer

    # Configuration
    model_name = "meta-llama/Llama-3.1-8B-Instruct"
    num_gpus = 1
    
    if torch.cuda.is_available():
         detected_gpus = torch.cuda.device_count()
         logger.info(f"CUDA available. Found {detected_gpus} GPUs.")
         num_gpus = min(num_gpus, detected_gpus)
    else:
         logger.error("CUDA not available. This example requires GPU.")
         sys.exit(1)

    prompt = "The capital of France is"
    target_sequences = ["Paris", "London", "Berlin"]

    llm = None
    try:
        # Initialize Model and Tokenizer
        logger.info(f"Loading tokenizer for {model_name}...")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if tokenizer.pad_token is None:
             tokenizer.pad_token = tokenizer.eos_token

        logger.info(f"Loading LLM {model_name} with tensor_parallel_size={num_gpus}...")
        llm = LLM(model=model_name,
                  tokenizer=tokenizer.name_or_path,
                  enforce_eager=True,
                  trust_remote_code=True,
                  tensor_parallel_size=num_gpus,
                  gpu_memory_utilization=0.85,
                  max_num_seqs=16)

        # Initialize Combined Hook for sequence probability
        logger.info("Initializing CombinedVLLMHook...")
        seq_prob_hook = CombinedVLLMHook(
            llm, tokenizer,
            enable_sequence_prob=True,
            enable_rep_control=False,
            enable_layer_logit_recording=False
        )

        # Calculate sequence probabilities
        logger.info("--- Calculating Sequence Probabilities ---")
        results = seq_prob_hook.get_log_prob([prompt], target_sequences)
        
        print("\n=== SEQUENCE PROBABILITY RESULTS ===")
        for result in results:
            print(f"Sequence: '{result['sequence']}'")
            print(f"  Log Probability: {result['log_prob']:.4f}")
            print(f"  Probability: {result['prob']:.6f}")
            print(f"  Perplexity: {result['perplexity']:.4f}")
            print(f"  Tokens: {result['num_tokens']}")
            print()

    except Exception as e:
        logger.error(f"An error occurred: {e}", exc_info=True)
        print(f"Traceback: {traceback.format_exc()}")

    finally:
        # Cleanup
        logger.info("Cleaning up resources...")
        if llm is not None:
            del llm
        torch.cuda.empty_cache()
        gc.collect()
        logger.info("Cleanup completed.")
