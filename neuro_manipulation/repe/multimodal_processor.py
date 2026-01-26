#!/usr/bin/env python3
"""
Multimodal Processor for Qwen2.5-VL

This module provides the correct implementation for processing multimodal inputs
with Qwen2.5-VL models, ensuring proper image token expansion and alignment.
"""

from typing import List, Union, Dict, Any, Optional
import torch
from PIL import Image


class QwenVLMultimodalProcessor:
    """
    Correct multimodal processing for Qwen2.5-VL models.
    
    This class handles the proper integration of text and images using
    the unified AutoProcessor approach, eliminating token/feature mismatches.
    """
    
    def __init__(self, processor, tokenizer=None):
        """
        Initialize with AutoProcessor.
        
        Args:
            processor: AutoProcessor from transformers
            tokenizer: Optional separate tokenizer (for backward compatibility)
        """
        self.processor = processor
        self.tokenizer = tokenizer or processor.tokenizer
        
    @staticmethod
    def create_multimodal_messages(text: str, images: List[Image.Image] = None) -> List[Dict]:
        """
        Create properly formatted messages for Qwen2.5-VL.
        
        Args:
            text: Input text
            images: List of PIL Images
            
        Returns:
            List of messages in the correct format
        """
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
        
        return [
            {
                "role": "user",
                "content": content
            }
        ]
    
    def process_with_official_utils(self, messages: List[Dict]) -> Dict[str, Any]:
        """
        Process using qwen_vl_utils (preferred method).
        
        Args:
            messages: Properly formatted messages
            
        Returns:
            Model inputs ready for forward pass
        """
        try:
            from qwen_vl_utils import process_vision_info
            
            # Apply chat template
            text = self.processor.apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=True
            )
            
            # Extract vision information
            image_inputs, video_inputs = process_vision_info(messages)
            
            # Unified processing
            inputs = self.processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt"
            )
            
            return inputs
            
        except ImportError:
            raise ImportError("qwen_vl_utils not available. Install with: pip install qwen-vl-utils")
    
    def process_with_fallback(self, messages: List[Dict]) -> Dict[str, Any]:
        """
        Fallback processing without qwen_vl_utils.
        
        Args:
            messages: Properly formatted messages
            
        Returns:
            Model inputs ready for forward pass
        """
        # Extract images from messages
        images = []
        for message in messages:
            if message.get("role") == "user":
                for content_item in message.get("content", []):
                    if content_item.get("type") == "image":
                        images.append(content_item.get("image"))
        
        # Apply chat template
        text = self.processor.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
        )
        
        # Unified processing
        inputs = self.processor(
            text=[text],
            images=images if images else None,
            padding=True,
            return_tensors="pt"
        )
        
        return inputs
    
    def process_multimodal_input(self, text: str, images: List[Image.Image] = None) -> Dict[str, Any]:
        """
        Main processing method that handles multimodal inputs correctly.
        
        Args:
            text: Input text
            images: List of PIL Images
            
        Returns:
            Model inputs with properly aligned tokens and features
        """
        # Create proper message format
        messages = self.create_multimodal_messages(text, images)
        
        # Try official method first, then fallback
        try:
            return self.process_with_official_utils(messages)
        except ImportError:
            return self.process_with_fallback(messages)
    
    def process_batch(self, inputs: List[Dict]) -> Dict[str, Any]:
        """
        Process a batch of multimodal inputs.
        
        Args:
            inputs: List of dicts with 'text' and optionally 'images' keys
            
        Returns:
            Batched model inputs
        """
        # For now, process the first item (TODO: implement proper batching)
        if inputs:
            first_input = inputs[0]
            text = first_input.get('text', '')
            images = first_input.get('images', [])
            return self.process_multimodal_input(text, images)
        
        return {}


# Legacy compatibility functions (no longer needed - implementation is now integrated directly)
def patch_rep_reading_pipeline():
    """
    Legacy function for backward compatibility.
    The multimodal processing is now integrated directly into RepReadingPipeline.
    """
    print("✓ Multimodal processing is now integrated directly - no patching needed")


def create_fixed_pipeline(model, tokenizer, processor):
    """
    Legacy function for backward compatibility.
    Just use the standard pipeline creation now.
    """
    from neuro_manipulation.repe.rep_reading_pipeline import RepReadingPipeline
    
    # Create pipeline instance normally
    pipeline = RepReadingPipeline.__new__(RepReadingPipeline)
    pipeline.model = model
    pipeline.tokenizer = tokenizer
    pipeline.image_processor = processor  # Use full processor
    
    return pipeline


# Integration with existing prompt formats
class QwenVLInstFormatFixed:
    """
    Fixed version of QwenVLInstFormat that works with the correct processor.
    """
    
    @staticmethod
    def create_processor_messages(system_prompt: str = None, 
                                user_messages: List[str] = None, 
                                assistant_answers: List[str] = None, 
                                images: List[Image.Image] = None) -> List[Dict]:
        """
        Create messages in the format expected by Qwen2.5-VL AutoProcessor.
        
        This bypasses manual token formatting and uses the processor's chat template.
        """
        messages = []
        
        if system_prompt:
            messages.append({
                "role": "system", 
                "content": system_prompt
            })
        
        user_messages = user_messages or []
        assistant_answers = assistant_answers or []
        
        for i, user_msg in enumerate(user_messages):
            content = []
            
            # Add image if available for this message
            if images and i < len(images):
                content.append({
                    "type": "image",
                    "image": images[i]
                })
            
            # Add text content
            content.append({
                "type": "text",
                "text": user_msg
            })
            
            messages.append({
                "role": "user",
                "content": content
            })
            
            # Add assistant response if available
            if i < len(assistant_answers):
                messages.append({
                    "role": "assistant",
                    "content": assistant_answers[i]
                })
        
        return messages


# Example usage function
def example_usage():
    """Example of how to use the fixed multimodal processing."""
    
    # This would be your actual model loading code
    # from transformers import AutoModel, AutoProcessor
    # 
    # model = AutoModel.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct", trust_remote_code=True)
    # processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct", trust_remote_code=True)
    # tokenizer = processor.tokenizer
    
    # Create the fixed pipeline
    # pipeline = create_fixed_pipeline(model, tokenizer, processor)
    
    # Create test data
    # test_image = Image.new('RGB', (224, 224), color='red')
    # multimodal_input = {
    #     'images': [test_image],
    #     'text': 'when you see this image, your emotion is anger'
    # }
    
    # Process with the fixed method
    # inputs = pipeline._prepare_multimodal_inputs(multimodal_input)
    
    # Forward pass (should work without token/feature mismatch)
    # with torch.no_grad():
    #     outputs = model(**inputs, output_hidden_states=True)
    
    print("Example usage code is commented out - see function for details")


if __name__ == "__main__":
    print("Qwen2.5-VL Multimodal Processor Fix")
    print("="*50)
    print()
    print("This module provides the correct implementation for processing")
    print("multimodal inputs with Qwen2.5-VL models.")
    print()
    print("Key features:")
    print("- Unified AutoProcessor usage")
    print("- Proper message formatting")
    print("- Automatic token expansion")
    print("- Backward compatibility")
    print()
    print("To use, import and call patch_rep_reading_pipeline() or use")
    print("create_fixed_pipeline() for new instances.")