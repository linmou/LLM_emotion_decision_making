from vllm import LLM, SamplingParams
import os
import logging
import sys
import traceback
import torch
import json
from huggingface_hub import HfApi, hf_hub_download

# Set up logging with more detailed format
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    stream=sys.stdout
)
logger = logging.getLogger(__name__)

def is_huggingface_model_name(model_path):
    """
    Check if the given path is a HuggingFace model name.
    A HuggingFace model name typically has the format 'organization/model_name'.
    """
    return '/' in model_path and not os.path.exists(model_path)

def get_model_config(model_path):
    """
    Get model configuration from either local path or HuggingFace.
    Returns the config dictionary and the actual model path to use.
    """
    try:
        if is_huggingface_model_name(model_path):
            logger.info(f"Detected HuggingFace model name: {model_path}")
            # Download config file from HuggingFace
            config_path = hf_hub_download(
                repo_id=model_path,
                filename="config.json",
                cache_dir=None  # Use default cache
            )
            actual_path = model_path  # Use the HuggingFace model name directly
        else:
            logger.info(f"Using local model path: {model_path}")
            config_path = os.path.join(model_path, 'config.json')
            actual_path = model_path

        # Read config file
        with open(config_path, 'r') as f:
            config = json.load(f)
            
        return config, actual_path
        
    except Exception as e:
        logger.error(f"Error getting model config: {str(e)}")
        raise

def get_optimal_tensor_parallel_size(model_path):
    """
    Calculate the optimal tensor parallel size based on model architecture and available GPUs.
    Returns the maximum possible tensor parallel size that divides the number of attention heads evenly.
    """
    try:
        # Get number of available GPUs
        num_gpus = torch.cuda.device_count()
        logger.info(f"Number of available GPUs: {num_gpus}")
        
        if num_gpus == 0:
            logger.warning("No GPUs available, defaulting to tensor_parallel_size=1")
            return 1
            
        # Get model config
        config, _ = get_model_config(model_path)
            
        # Get number of attention heads from config
        # Different models might store this in different keys
        num_heads = None
        possible_keys = ['num_attention_heads', 'n_head', 'num_heads', 'n_heads']
        for key in possible_keys:
            if key in config:
                num_heads = config[key]
                break
                
        if num_heads is None:
            logger.warning("Could not determine number of attention heads, defaulting to tensor_parallel_size=1")
            return 1
            
        logger.info(f"Number of attention heads in model: {num_heads}")
        
        # Find the largest divisor of num_heads that is <= num_gpus
        optimal_size = 1
        for i in range(2, min(num_heads + 1, num_gpus + 1), 2):
            if num_heads % i == 0:
                optimal_size = i
                
        logger.info(f"Optimal tensor parallel size: {optimal_size}")
        return optimal_size
        
    except Exception as e:
        logger.error(f"Error calculating optimal tensor parallel size: {str(e)}")
        logger.error("Defaulting to tensor_parallel_size=1")
        return 1

def check_model_directory(model_path):
    """Check the model directory structure and permissions"""
    logger.info(f"Checking model path: {model_path}")
    
    if is_huggingface_model_name(model_path):
        logger.info("Using HuggingFace model, skipping local directory check")
        return True
        
    if not os.path.exists(model_path):
        logger.error(f"Model directory does not exist: {model_path}")
        return False
        
    try:
        files = os.listdir(model_path)
        logger.info(f"Found {len(files)} files in model directory")
        logger.info(f"Files: {files}")
        
        # Check for essential files
        essential_files = ['config.json', 'model.safetensors', 'tokenizer.json']
        missing_files = [f for f in essential_files if f not in files]
        
        if missing_files:
            logger.error(f"Missing essential files: {missing_files}")
            return False
            
        # Check permissions
        for file in essential_files:
            file_path = os.path.join(model_path, file)
            if not os.access(file_path, os.R_OK):
                logger.error(f"No read permission for file: {file}")
                return False
                
        return True
    except Exception as e:
        logger.error(f"Error checking model directory: {str(e)}")
        return False

# --- Configuration ---
# Model path on your system - can be either HuggingFace model name or local path
# MODEL_PATH = "Qwen/Qwen2.5-0.5B-Instruct"  # Example HuggingFace model name
MODEL_PATH = "/data/home/jjl7137/huggingface_models/Qwen/Qwen2.5-1.5B-Instruct"  # Example local path

# Calculate optimal tensor parallel size
TENSOR_PARALLEL_SIZE = get_optimal_tensor_parallel_size(MODEL_PATH)
assert TENSOR_PARALLEL_SIZE == 2, f"Tensor parallel size is {TENSOR_PARALLEL_SIZE}, but should be 2"

# Set to True if the model requires custom code from Hugging Face Hub
TRUST_REMOTE_CODE = True

# --- Sample Prompts ---
# You can modify these or add your own
prompts = [
    "Hello! Can you tell me a short story?",
    "What is the capital of France?",
    "Write a python function to calculate the factorial of a number.",
    "Translate the following English sentence to Chinese: 'The weather is nice today.'",
]

# --- Sampling Parameters ---
# These parameters control the generation process.
# Adjust them as needed for your use case.
# For more details, see: https://vllm.readthedocs.io/en/latest/dev/sampling_params.html
sampling_params = SamplingParams(
    n=1,  # Number of output sequences to return
    temperature=0.7, # Controls randomness. Higher is more random.
    top_p=0.95, # Nucleus sampling.
    max_tokens=200, # Maximum number of tokens to generate per output.
    # stop=["<|im_end|>", "<|endoftext|>"] # Add model-specific stop tokens if known
                                        # Qwen models often use <|im_end|>
)

def run_vllm_inference():
    """
    Initializes the vLLM engine, generates text for the sample prompts,
    and prints the outputs.
    """
    logger.info("Starting vLLM inference process")
    
    # First check the model directory/path
    if not check_model_directory(MODEL_PATH):
        logger.error("Model path check failed. Exiting.")
        return

    try:
        logger.info(f"Initializing LLM with model: {MODEL_PATH}")
        logger.info(f"Parameters: tensor_parallel_size={TENSOR_PARALLEL_SIZE}, trust_remote_code={TRUST_REMOTE_CODE}")
        
        llm = LLM(
            model=MODEL_PATH,
            tensor_parallel_size=TENSOR_PARALLEL_SIZE,
            trust_remote_code=TRUST_REMOTE_CODE,
        )
        logger.info("Model loaded successfully")

        logger.info("Starting inference with sample prompts")
        outputs = llm.generate(prompts, sampling_params)
        
        logger.info("Inference completed successfully")
        print("\n--- Generated Outputs ---")
        for i, output in enumerate(outputs):
            prompt = output.prompt
            generated_text = output.outputs[0].text
            print(f"\nPrompt {i+1}: {prompt}")
            print(f"Generated Text: {generated_text}")
            print("-" * 30)

    except ImportError as e:
        logger.error(f"vLLM import error: {str(e)}")
        logger.error("Please ensure vLLM is installed correctly: pip install vllm")
    except RuntimeError as e:
        logger.error(f"Runtime error during vLLM initialization: {str(e)}")
        logger.error("Full traceback:")
        logger.error(traceback.format_exc())
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        logger.error("Full traceback:")
        logger.error(traceback.format_exc())

if __name__ == "__main__":
    try:
        run_vllm_inference()
    except Exception as e:
        logger.error(f"Fatal error in main: {str(e)}")
        logger.error("Full traceback:")
        logger.error(traceback.format_exc())
