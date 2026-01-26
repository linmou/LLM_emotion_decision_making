from typing import List, Union, Optional, Dict, Any
import inspect
from transformers import Pipeline
import torch
import numpy as np
from PIL import Image
from .rep_readers import DIRECTION_FINDERS, RepReader
from ..prompt_formats import ManualPromptFormat

class RepReadingPipeline(Pipeline):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _get_hidden_states(
            self, 
            outputs,
            rep_token: Union[str, int, list]=-1,
            hidden_layers: Union[List[int], int]=-1,
            which_hidden_states: Optional[str]=None):
        
        if hasattr(outputs, 'encoder_hidden_states') and hasattr(outputs, 'decoder_hidden_states'):
            outputs['hidden_states'] = outputs[f'{which_hidden_states}_hidden_states']
    
        hidden_states_layers = {}
        for layer in hidden_layers:
            hidden_states = outputs['hidden_states'][layer]
            
            # CRITICAL FIX: Handle multimodal sequence indexing safely
            seq_len = hidden_states.shape[1]
            
            # Convert rep_token to safe index
            if isinstance(rep_token, int):
                if rep_token < 0:
                    # Negative indexing: -1 means last token, -2 means second-to-last, etc.
                    safe_token_idx = seq_len + rep_token
                else:
                    # Positive indexing
                    safe_token_idx = rep_token
                
                # Bounds check to prevent "index X is out of bounds" errors
                if safe_token_idx < 0 or safe_token_idx >= seq_len:
                    print(f"WARNING: rep_token={rep_token} (safe_idx={safe_token_idx}) out of bounds for seq_len={seq_len}")
                    print(f"Using last available token (index {seq_len-1}) instead")
                    safe_token_idx = seq_len - 1  # Use last available token
                    
                hidden_states_extracted = hidden_states[:, safe_token_idx, :].detach()
            else:
                # Handle list/other types (fallback to original behavior)
                hidden_states_extracted = hidden_states[:, rep_token, :].detach()
            
            if hidden_states_extracted.dtype == torch.bfloat16:
                hidden_states_extracted = hidden_states_extracted.float()
            hidden_states_layers[layer] = hidden_states_extracted.detach()

        return hidden_states_layers

    def _sanitize_parameters(self, 
                             rep_reader: RepReader=None,
                             rep_token: Union[str, int]=-1,
                             hidden_layers: Union[List[int], int]=-1,
                             component_index: int=0,
                             which_hidden_states: Optional[str]=None,
                             **tokenizer_kwargs):
        preprocess_params = tokenizer_kwargs
        forward_params =  {}
        postprocess_params = {}

        forward_params['rep_token'] = rep_token

        if not isinstance(hidden_layers, list):
            hidden_layers = [hidden_layers]


        assert rep_reader is None or len(rep_reader.directions) == len(hidden_layers), f"expect total rep_reader directions ({len(rep_reader.directions)})== total hidden_layers ({len(hidden_layers)})"                 
        forward_params['rep_reader'] = rep_reader
        forward_params['hidden_layers'] = hidden_layers
        forward_params['component_index'] = component_index
        forward_params['which_hidden_states'] = which_hidden_states
        
        return preprocess_params, forward_params, postprocess_params
 
    def _is_multimodal_input(self, inputs: Any) -> bool:
        """Check if inputs contain both images and text for multimodal processing."""
        if isinstance(inputs, dict):
            return 'images' in inputs or 'image' in inputs
        if isinstance(inputs, list) and len(inputs) > 0:
            return any(isinstance(item, (Image.Image, torch.Tensor)) or 
                      (isinstance(item, dict) and ('images' in item or 'image' in item)) 
                      for item in inputs)
        return False

    def _prepare_multimodal_inputs(self, inputs: Union[Dict, List], **tokenizer_kwargs) -> Dict[str, Any]:
        """Prepare multimodal inputs using the correct Qwen2.5-VL processor format."""
        
        if isinstance(inputs, dict):
            images = inputs.get('images', inputs.get('image'))
            text = inputs.get('text', inputs.get('prompt', ''))
            
            if not isinstance(images, list) and images is not None:
                images = [images]
            
            # EXTREME VRAM OPTIMIZATION: Pre-compress images to tiny sizes
            if images:
                compressed_images = []
                for img in images:
                    if hasattr(img, 'resize'):
                        # Force resize to very small dimensions to prevent large allocations
                        compressed_img = img.resize((224, 224), Image.Resampling.LANCZOS)
                        compressed_images.append(compressed_img)
                    else:
                        compressed_images.append(img)
                images = compressed_images
            
            # Create proper message format for Qwen2.5-VL
            content = []
            
            # Add images first
            if images:
                for image in images:
                    content.append({
                        "type": "image",
                        "image": image
                    })
            
            # Add text
            content.append({
                "type": "text",
                "text": text
            })
            
            messages = [
                {
                    "role": "user",
                    "content": content
                }
            ]
            
            # CRITICAL FIX: Detect pre-formatted text to avoid double chat template application
            # This prevents token/feature mismatch by avoiding duplicate processing
            try:
                # Check if this is a Gemma 3 model - handle differently
                model_name = getattr(self.tokenizer, 'name_or_path', '')
                if 'gemma-3' in model_name.lower():
                    # Gemma 3 specific processing
                    formatted_text = self.image_processor.apply_chat_template(
                        messages, 
                        tokenize=False, 
                        add_generation_prompt=True
                    )
                    
                    # Extract images directly from messages
                    extracted_images = []
                    for message in messages:
                        content = message.get("content", [])
                        if isinstance(content, list):
                            for item in content:
                                if isinstance(item, dict) and item.get("type") == "image":
                                    extracted_images.append(item.get("image"))
                    
                    # Use the processor directly with proper format
                    # Filter out tokenizer_kwargs that might conflict with image processor
                    safe_kwargs = {k: v for k, v in tokenizer_kwargs.items() 
                                 if k not in ['size', 'do_resize', 'do_normalize', 'image_mean', 'image_std']}
                    
                    model_inputs = self.image_processor(
                        text=[formatted_text],
                        images=extracted_images if extracted_images else None,
                        padding=True,
                        return_tensors="pt",
                        **safe_kwargs
                    )
                    
                    return model_inputs
                
                # Method 1: Try with qwen_vl_utils (official and most reliable for Qwen models)
                try:
                    from qwen_vl_utils import process_vision_info
                    
                    # Check if text is already formatted (contains Qwen-VL tokens)
                    is_pre_formatted = (
                        '<|im_start|>' in text and 
                        '<|vision_start|>' in text and 
                        '<|image_pad|>' in text
                    )
                    
                    if is_pre_formatted:
                        # Text already formatted by PromptFormat.build() - use directly
                        formatted_text = text
                        # Extract vision information from messages for consistency
                        image_inputs, video_inputs = process_vision_info(messages)
                    else:
                        # Raw text - apply chat template (this handles token formatting correctly)
                        formatted_text = self.image_processor.apply_chat_template(
                            messages, 
                            tokenize=False, 
                            add_generation_prompt=True
                        )
                        # Extract vision information
                        image_inputs, video_inputs = process_vision_info(messages)
                    
                    # Use unified processor - this prevents token/feature mismatch
                    model_inputs = self.image_processor(
                        text=[formatted_text],
                        images=image_inputs,
                        videos=video_inputs,
                        padding=True,
                        return_tensors="pt",
                        **tokenizer_kwargs
                    )
                    
                    return model_inputs
                    
                except ImportError:
                    # Method 2: Fallback without qwen_vl_utils but still unified
                    print("qwen_vl_utils not available, using unified processor fallback")
                    
                    # Check if text is already formatted (contains Qwen-VL tokens)
                    is_pre_formatted = (
                        '<|im_start|>' in text and 
                        '<|vision_start|>' in text and 
                        '<|image_pad|>' in text
                    )
                    
                    if is_pre_formatted:
                        # Text already formatted by PromptFormat.build() - use directly
                        formatted_text = text
                    else:
                        # Raw text - apply chat template using the processor's tokenizer
                        formatted_text = self.image_processor.apply_chat_template(
                            messages, 
                            tokenize=False, 
                            add_generation_prompt=True
                        )
                    
                    # Use unified processor - this is the key to preventing token/feature mismatch
                    model_inputs = self.image_processor(
                        text=[formatted_text],
                        images=images if images else None,
                        padding=True,
                        return_tensors="pt",
                        **tokenizer_kwargs
                    )
                    
                    return model_inputs
                    
            except Exception as e:
                print(f"ERROR: Unified processing failed with: {e}")
                print("This indicates a configuration issue with the AutoProcessor.")
                print("Ensure you're using: AutoProcessor.from_pretrained(model_path, trust_remote_code=True)")
                
                # DO NOT fall back to separate processing as it causes token/feature mismatch
                # Instead, raise the error to help debug the processor setup
                raise ValueError(
                    f"Multimodal processing failed. This usually indicates:\n"
                    f"1. image_processor is not a proper AutoProcessor instance\n"
                    f"2. Model does not support multimodal processing\n"
                    f"3. Missing required dependencies (qwen_vl_utils recommended)\n"
                    f"Original error: {e}"
                )
                
        elif isinstance(inputs, list):
            # Handle batch processing
            if inputs:
                # For now, process the first item (TODO: implement proper batching)
                # This maintains consistency while avoiding the token mismatch
                return self._prepare_multimodal_inputs(inputs[0], **tokenizer_kwargs)
        
        return {}

    def preprocess(
            self, 
            inputs: Union[str, List[str], List[List[str]], Dict, List[Dict]],
            **tokenizer_kwargs):

        # Check if this is multimodal input
        if self._is_multimodal_input(inputs):
            return self._prepare_multimodal_inputs(inputs, **tokenizer_kwargs)
        
        # Legacy image processor path (backward compatibility)
        if self.image_processor and not isinstance(inputs, (str, list)):
            return self.image_processor(inputs, add_end_of_utterance_token=False, return_tensors="pt")
            
        # Standard text processing
        return self.tokenizer(inputs, return_tensors=self.framework, **tokenizer_kwargs)

    def postprocess(self, outputs):
        return outputs

    def _forward(self, model_inputs, rep_token, hidden_layers, rep_reader=None, component_index=0, which_hidden_states=None, pad_token_id=None):
        """
        Args:
        - which_hidden_states (str): Specifies which part of the model (encoder, decoder, or both) to compute the hidden states from. 
                        It's applicable only for encoder-decoder models. Valid values: 'encoder', 'decoder'.
        Return: 
        - hidden_states (dict): A dictionary with keys as layer numbers and values as rep_token's projection at PCA direction
        """
        # get model hidden states and optionally transform them with a RepReader
        with torch.no_grad():
            # Ensure inputs are on the same device as model
            if hasattr(self.model, 'device'):
                device = self.model.device
            else:
                device = next(self.model.parameters()).device
            
            # Move inputs to model device
            for key, value in model_inputs.items():
                if isinstance(value, torch.Tensor):
                    model_inputs[key] = value.to(device)
            
            if hasattr(self.model, "encoder") and hasattr(self.model, "decoder"):
                decoder_start_token = [self.tokenizer.pad_token] * model_inputs['input_ids'].size(0)
                decoder_input = self.tokenizer(decoder_start_token, return_tensors="pt").input_ids.to(device)
                model_inputs['decoder_input_ids'] = decoder_input
            
            model_kwargs = dict(model_inputs)
            if "use_cache" in inspect.signature(self.model.forward).parameters:
                model_kwargs["use_cache"] = False

            outputs = self.model(**model_kwargs, output_hidden_states=True)
            
            # MEMORY OPTIMIZATION: Extract hidden states immediately and clear outputs
            hidden_states = self._get_hidden_states(outputs, rep_token, hidden_layers, which_hidden_states)
            
            # SAFE MEMORY CLEANUP: Only clear outputs object, avoid torch.cuda.empty_cache()
            del outputs  # Clear large outputs object
        
        if rep_reader is None:
            return hidden_states
        
        return rep_reader.transform(hidden_states, hidden_layers, component_index)


    def _batched_string_to_hiddens(self, train_inputs, rep_token, hidden_layers, batch_size, which_hidden_states, **tokenizer_args):
        # Wrapper method to get a dictionary hidden states from a list of strings
        # VRAM OPTIMIZATION: Use smaller batch sizes for multimodal inputs
        original_batch_size = batch_size
        if isinstance(train_inputs, list) and len(train_inputs) > 0:
            # Check if this is multimodal input
            if isinstance(train_inputs[0], dict) and 'images' in str(train_inputs[0]):
                # Reduce batch size for multimodal to prevent OOM
                batch_size = 1  # Process one multimodal item at a time for stability
        
        hidden_states_outputs = self(train_inputs, rep_token=rep_token,
            hidden_layers=hidden_layers, batch_size=batch_size, rep_reader=None, which_hidden_states=which_hidden_states, **tokenizer_args)
        hidden_states = {layer: [] for layer in hidden_layers}
        
        # MEMORY OPTIMIZATION: Process and clear batches immediately
        for hidden_states_batch in hidden_states_outputs:
            for layer in hidden_states_batch:
                hidden_states[layer].extend(hidden_states_batch[layer])
            # SAFE CLEANUP: Just delete the batch object, avoid cache clearing
            del hidden_states_batch
                
        return {k: np.vstack(v) for k, v in hidden_states.items()}
    
    def _validate_params(self, n_difference, direction_method):
        # validate params for get_directions
        if direction_method == 'clustermean':
            assert n_difference == 1, "n_difference must be 1 for clustermean"

    def get_directions(
            self, 
            train_inputs: Union[str, List[str], List[List[str]]], 
            rep_token: Union[str, int]=-1, 
            hidden_layers: Union[str, int]=-1,
            n_difference: int = 1,
            batch_size: int = 8, 
            train_labels: List[int] = None,
            direction_method: str = 'pca',
            direction_finder_kwargs: dict = {},
            which_hidden_states: Optional[str]=None,
            **tokenizer_args,):
        """Train a RepReader on the training data.
        Args:
            batch_size: batch size to use when getting hidden states
            direction_method: string specifying the RepReader strategy for finding directions
            direction_finder_kwargs: kwargs to pass to RepReader constructor
        """

        if not isinstance(hidden_layers, list): 
            assert isinstance(hidden_layers, int)
            hidden_layers = [hidden_layers]
        
        self._validate_params(n_difference, direction_method)

        # initialize a DirectionFinder
        direction_finder = DIRECTION_FINDERS[direction_method](**direction_finder_kwargs)

		# if relevant, get the hidden state data for training set
        hidden_states = None
        relative_hidden_states = None
        if direction_finder.needs_hiddens:
            # get raw hidden states for the train inputs
            hidden_states = self._batched_string_to_hiddens(train_inputs, rep_token, hidden_layers, batch_size, which_hidden_states, **tokenizer_args)
            
            # get differences between pairs
            relative_hidden_states = {k: np.copy(v) for k, v in hidden_states.items()}
            
            # Track how many samples we actually have for each layer after processing
            final_sample_count = None
            
            for layer in hidden_layers:
                for _ in range(n_difference):
                    # BUGFIX: Handle odd number of samples by ensuring even/odd arrays have same length
                    even_indices = relative_hidden_states[layer][::2]
                    odd_indices = relative_hidden_states[layer][1::2]
                    
                    # Ensure both arrays have the same length by truncating the longer one
                    min_length = min(len(even_indices), len(odd_indices))
                    even_indices = even_indices[:min_length]
                    odd_indices = odd_indices[:min_length]
                    
                    relative_hidden_states[layer] = even_indices - odd_indices
                    
                    # Track the final sample count after truncation
                    if final_sample_count is None:
                        final_sample_count = min_length
            
            # CRITICAL FIX: Also truncate the original hidden_states to match relative_hidden_states
            # This ensures consistency between hidden_states used for sign calculation and processed data
            if final_sample_count is not None and final_sample_count * 2 < len(hidden_states[hidden_layers[0]]):
                for layer in hidden_layers:
                    hidden_states[layer] = hidden_states[layer][:final_sample_count * 2]
                
                # Also truncate train_labels if provided to match
                if train_labels is not None:
                    # Debug: Check the structure of train_labels
                    if isinstance(train_labels, list):
                        original_label_count = len(np.concatenate(train_labels)) if len(train_labels) > 0 and isinstance(train_labels[0], (list, np.ndarray)) else len(train_labels)
                    else:
                        original_label_count = len(train_labels)
                    
                    target_count = final_sample_count * 2
                    if original_label_count > target_count:
                        
                        if isinstance(train_labels, list) and len(train_labels) > 0 and isinstance(train_labels[0], (list, np.ndarray)):
                            # Handle nested list structure - need to ensure total concatenated count matches target_count
                            # Since each group originally had pairs, and we're processing with n_difference=1,
                            # we need to maintain the same structure but with truncated data
                            
                            # SIMPLE FIX: Just take target_count samples and reconstruct as pairs
                            # The proportional approach was overcomplicated and buggy
                            
                            # Flatten all labels, truncate to target_count, then reconstruct as pairs
                            flat_labels = np.concatenate(train_labels)
                            truncated_flat = flat_labels[:target_count]
                            
                            # Reconstruct as pairs (groups of size 2) to maintain original structure
                            train_labels = []
                            for i in range(0, len(truncated_flat), 2):
                                if i + 1 < len(truncated_flat):
                                    # Complete pair
                                    train_labels.append([truncated_flat[i], truncated_flat[i+1]])
                                else:
                                    # Odd final element - create single-element group
                                    train_labels.append([truncated_flat[i]])
                            
                            final_count = sum(len(x) for x in train_labels)
                            # This should now always equal target_count
                            if final_count != target_count:
                                # CRITICAL FALLBACK: Ensure exact target_count
                                # The key insight: PCARepReader expects len(np.concatenate(train_labels)) == target_count
                                non_empty_groups = [group for group in train_labels if len(group) > 0]
                                if non_empty_groups:
                                    flat_labels = np.concatenate(non_empty_groups)
                                    
                                    # Simply truncate to exact target_count and reconstruct groups of size 2
                                    truncated_flat = flat_labels[:target_count]
                                    
                                    # Reconstruct as pairs (groups of size 2) to maintain original structure
                                    train_labels = []
                                    for i in range(0, len(truncated_flat), 2):
                                        if i + 1 < len(truncated_flat):
                                            # Complete pair
                                            train_labels.append([truncated_flat[i], truncated_flat[i+1]])
                                        else:
                                            # Odd final element - create single-element group
                                            train_labels.append([truncated_flat[i]])
                                    
                                    # Verify the fix
                                    final_flat = [item for sublist in train_labels for item in sublist]
                                    
                                    if len(final_flat) != target_count:
                                        # Last resort: use first target_count elements directly as flat array
                                        train_labels = [[x] for x in flat_labels[:target_count]]
                                else:
                                    # No valid labels - create empty structure
                                    train_labels = []
                        else:
                            # Handle flat array
                            train_labels = train_labels[:target_count]

		# get the directions
        direction_finder.directions = direction_finder.get_rep_directions(
            self.model, self.tokenizer, relative_hidden_states, hidden_layers,
            train_choices=train_labels)
        for layer in direction_finder.directions:
            if type(direction_finder.directions[layer]) == np.ndarray:
                direction_finder.directions[layer] = direction_finder.directions[layer].astype(np.float32)

        if train_labels is not None:
            direction_finder.direction_signs = direction_finder.get_signs(
            hidden_states, train_labels, hidden_layers)
        
        return direction_finder
