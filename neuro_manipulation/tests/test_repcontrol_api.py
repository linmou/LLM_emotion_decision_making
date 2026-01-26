#!/usr/bin/env python3
"""
Test client for vLLM API Server with RepControl support

This script demonstrates how to use the API endpoints with and without
representation control.
"""

import json
from typing import Dict, List, Optional

import numpy as np
import requests


class RepControlAPIClient:
    def __init__(self, base_url: str = "http://localhost:8000"):
        """
        Initialize the API client.

        Args:
            base_url: Base URL of the vLLM API server
        """
        self.base_url = base_url.rstrip("/")

    def health_check(self) -> Dict:
        """Check server health"""
        response = requests.get(f"{self.base_url}/health")
        response.raise_for_status()
        return response.json()

    def list_models(self) -> Dict:
        """List available models"""
        response = requests.get(f"{self.base_url}/v1/models")
        response.raise_for_status()
        return response.json()

    def get_repcontrol_info(self) -> Dict:
        """Get RepControl capabilities information"""
        response = requests.get(f"{self.base_url}/v1/repcontrol/info")
        response.raise_for_status()
        return response.json()

    def completion(
        self,
        prompt: str,
        model: str = "meta-llama/Llama-3.1-8B-Instruct",
        max_tokens: int = 100,
        temperature: float = 0.0,
        rep_control: Optional[Dict] = None,
    ) -> Dict:
        """
        Create a completion request.

        Args:
            prompt: Input prompt
            model: Model name
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            rep_control: RepControl configuration dictionary
        """
        payload = {
            "model": model,
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }

        if rep_control:
            payload["rep_control"] = rep_control

        response = requests.post(
            f"{self.base_url}/v1/completions",
            json=payload,
            headers={"Content-Type": "application/json"},
        )
        response.raise_for_status()
        return response.json()

    def chat_completion(
        self,
        messages: List[Dict[str, str]],
        model: str = "meta-llama/Llama-3.1-8B-Instruct",
        max_tokens: int = 100,
        temperature: float = 0.0,
        rep_control: Optional[Dict] = None,
    ) -> Dict:
        """
        Create a chat completion request.

        Args:
            messages: List of message dictionaries with 'role' and 'content'
            model: Model name
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            rep_control: RepControl configuration dictionary
        """
        payload = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }

        if rep_control:
            payload["rep_control"] = rep_control

        response = requests.post(
            f"{self.base_url}/v1/chat/completions",
            json=payload,
            headers={"Content-Type": "application/json"},
        )
        response.raise_for_status()
        return response.json()


def create_dummy_activation_vector(
    hidden_dim: int = 4096, magnitude: float = 0.1
) -> List[float]:
    """Create a dummy activation vector for testing"""
    # Create a simple pattern - you would replace this with actual reading vectors
    vector = np.random.normal(0, magnitude, hidden_dim).astype(np.float32)
    return vector.tolist()


def demo_basic_usage():
    """Demonstrate basic API usage without RepControl"""
    print("=== Basic API Usage Demo ===")

    client = RepControlAPIClient()

    # Health check
    print("1. Health Check:")
    health = client.health_check()
    print(f"   Status: {health}")

    # List models
    print("\n2. Available Models:")
    models = client.list_models()
    print(f"   Models: {[model['id'] for model in models['data']]}")

    # Basic completion
    print("\n3. Basic Completion:")
    prompt = "The capital of France is"
    result = client.completion(prompt=prompt, max_tokens=10, temperature=0.0)
    print(f"   Prompt: {prompt}")
    print(f"   Response: {result['choices'][0]['text']}")

    # Basic chat completion
    print("\n4. Basic Chat Completion:")
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is the capital of France?"},
    ]
    result = client.chat_completion(messages=messages, max_tokens=20, temperature=0.0)
    print(f"   Messages: {messages}")
    print(f"   Response: {result['choices'][0]['message']['content']}")


def demo_repcontrol_usage():
    """Demonstrate RepControl API usage"""
    print("\n=== RepControl API Usage Demo ===")

    client = RepControlAPIClient()

    # Get RepControl info
    print("1. RepControl Info:")
    info = client.get_repcontrol_info()
    print(f"   Supported operators: {info['supported_operators']}")
    print(f"   Supported blocks: {info['supported_blocks']}")
    print(f"   Tensor parallel size: {info['tensor_parallel_size']}")

    # Create RepControl configuration
    print("\n2. RepControl Configuration:")

    # Example: Apply control to layers 10 and 15
    layers_to_control = [10, 15]
    hidden_dim = 4096  # Adjust based on your model

    # Create dummy activation vectors for each layer
    activations = {}
    for layer_id in layers_to_control:
        activations[layer_id] = create_dummy_activation_vector(
            hidden_dim, magnitude=0.05
        )

    rep_control_config = {
        "layers": layers_to_control,
        "block_name": "decoder_block",
        "activations": activations,
        "token_pos": None,  # Apply to all tokens
        "normalize": False,
        "operator": "linear_comb",
    }

    print(f"   Controlling layers: {layers_to_control}")
    print(f"   Block: {rep_control_config['block_name']}")
    print(f"   Operator: {rep_control_config['operator']}")

    # Test with completion
    print("\n3. RepControl Completion:")
    prompt = "The capital of France is"

    # Baseline without control
    baseline_result = client.completion(prompt=prompt, max_tokens=10, temperature=0.0)
    baseline_text = baseline_result["choices"][0]["text"]

    # With RepControl
    controlled_result = client.completion(
        prompt=prompt, max_tokens=10, temperature=0.0, rep_control=rep_control_config
    )
    controlled_text = controlled_result["choices"][0]["text"]

    print(f"   Prompt: {prompt}")
    print(f"   Baseline: {baseline_text}")
    print(f"   Controlled: {controlled_text}")
    print(f"   Difference: {'Yes' if baseline_text != controlled_text else 'No'}")

    # Test with chat completion
    print("\n4. RepControl Chat Completion:")
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Tell me about artificial intelligence."},
    ]

    # Baseline without control
    baseline_result = client.chat_completion(
        messages=messages, max_tokens=30, temperature=0.0
    )
    baseline_text = baseline_result["choices"][0]["message"]["content"]

    # With RepControl
    controlled_result = client.chat_completion(
        messages=messages,
        max_tokens=30,
        temperature=0.0,
        rep_control=rep_control_config,
    )
    controlled_text = controlled_result["choices"][0]["message"]["content"]

    print(f"   Messages: {messages[-1]['content']}")
    print(f"   Baseline: {baseline_text[:100]}...")
    print(f"   Controlled: {controlled_text[:100]}...")
    print(f"   Difference: {'Yes' if baseline_text != controlled_text else 'No'}")


def demo_advanced_repcontrol():
    """Demonstrate advanced RepControl features"""
    print("\n=== Advanced RepControl Demo ===")

    client = RepControlAPIClient()

    # Test different operators
    print("1. Testing Different Operators:")
    prompt = "The weather today is"

    operators = ["linear_comb", "piecewise_linear"]
    for operator in operators:
        rep_control_config = {
            "layers": [12],
            "block_name": "decoder_block",
            "activations": {12: create_dummy_activation_vector(4096, 0.1)},
            "operator": operator,
            "normalize": False,
        }

        try:
            result = client.completion(
                prompt=prompt,
                max_tokens=15,
                temperature=0.0,
                rep_control=rep_control_config,
            )
            text = result["choices"][0]["text"]
            print(f"   {operator}: {text}")
        except Exception as e:
            print(f"   {operator}: Error - {e}")

    # Test different token positions
    print("\n2. Testing Different Token Positions:")
    token_positions = [None, "start", "end", 0, -1]

    for token_pos in token_positions:
        rep_control_config = {
            "layers": [10],
            "block_name": "decoder_block",
            "activations": {10: create_dummy_activation_vector(4096, 0.05)},
            "token_pos": token_pos,
            "operator": "linear_comb",
        }

        try:
            result = client.completion(
                prompt=prompt,
                max_tokens=10,
                temperature=0.0,
                rep_control=rep_control_config,
            )
            text = result["choices"][0]["text"]
            print(f"   token_pos={token_pos}: {text}")
        except Exception as e:
            print(f"   token_pos={token_pos}: Error - {e}")


def demo_error_handling():
    """Demonstrate error handling"""
    print("\n=== Error Handling Demo ===")

    client = RepControlAPIClient()

    # Test invalid layer
    print("1. Testing Invalid Layer:")
    try:
        rep_control_config = {
            "layers": [999],  # Invalid layer
            "activations": {999: create_dummy_activation_vector(4096)},
        }

        result = client.completion(prompt="Test", rep_control=rep_control_config)
        print("   Unexpected success")
    except Exception as e:
        print(f"   Expected error: {e}")

    # Test mismatched activation dimension
    print("\n2. Testing Mismatched Activation Dimension:")
    try:
        rep_control_config = {
            "layers": [10],
            "activations": {10: create_dummy_activation_vector(128)},  # Wrong dimension
        }

        result = client.completion(prompt="Test", rep_control=rep_control_config)
        print("   Unexpected success")
    except Exception as e:
        print(f"   Expected error: {e}")


if __name__ == "__main__":
    print("vLLM RepControl API Test Client")
    print("================================")

    try:
        # Basic usage demo
        demo_basic_usage()

        # RepControl demo
        demo_repcontrol_usage()

        # Advanced features
        demo_advanced_repcontrol()

        # Error handling
        demo_error_handling()

    except requests.exceptions.ConnectionError:
        print("\nError: Could not connect to the API server.")
        print("Make sure the server is running on http://localhost:8000")
        print("Start the server with: python vllm_api_server.py")
    except Exception as e:
        print(f"\nUnexpected error: {e}")
        import traceback

        traceback.print_exc()
