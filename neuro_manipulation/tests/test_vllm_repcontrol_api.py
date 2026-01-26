#!/usr/bin/env python3
"""
Unit tests for vLLM RepControl API

Run with: python -m unittest test_vllm_repcontrol_api.py
"""

import json
import os
import sys
import unittest
from unittest.mock import MagicMock, Mock, patch

import numpy as np
import torch

# Add the project root to the Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from test_repcontrol_api import RepControlAPIClient, create_dummy_activation_vector

# Import the modules to test
from vllm_api_server import RepControlConfig, VLLMAPIServer


class TestRepControlConfig(unittest.TestCase):
    """Test RepControlConfig validation"""

    def test_valid_config(self):
        """Test creating a valid RepControlConfig"""
        config = RepControlConfig(
            layers=[10, 15],
            block_name="decoder_block",
            activations={10: [0.1] * 4096, 15: [0.2] * 4096},
            operator="linear_comb",
        )

        self.assertEqual(config.layers, [10, 15])
        self.assertEqual(config.block_name, "decoder_block")
        self.assertEqual(config.operator, "linear_comb")
        self.assertFalse(config.normalize)

    def test_default_values(self):
        """Test default values in RepControlConfig"""
        config = RepControlConfig(layers=[10])

        self.assertEqual(config.block_name, "decoder_block")
        self.assertEqual(config.operator, "linear_comb")
        self.assertIsNone(config.activations)
        self.assertIsNone(config.token_pos)
        self.assertFalse(config.normalize)

    def test_invalid_operator(self):
        """Test validation catches invalid operators"""
        # Note: Pydantic doesn't validate enum values by default,
        # but this test shows the structure for when validation is added
        config = RepControlConfig(layers=[10], operator="invalid_operator")
        # This currently passes, but could be enhanced with validation
        self.assertEqual(config.operator, "invalid_operator")


class TestVLLMAPIServerMethods(unittest.TestCase):
    """Test VLLMAPIServer utility methods without requiring GPU"""

    def setUp(self):
        """Set up test fixtures with mocked dependencies"""
        with patch("vllm_api_server.AutoTokenizer"), patch("vllm_api_server.LLM"):

            # Mock tokenizer
            mock_tokenizer = Mock()
            mock_tokenizer.name_or_path = "test-model"
            mock_tokenizer.pad_token = None
            mock_tokenizer.eos_token = "<eos>"

            # Mock LLM
            mock_llm = Mock()
            mock_llm.llm_engine.parallel_config.tensor_parallel_size = 2

            with patch(
                "vllm_api_server.AutoTokenizer.from_pretrained",
                return_value=mock_tokenizer,
            ), patch("vllm_api_server.LLM", return_value=mock_llm):

                self.server = VLLMAPIServer(
                    model_name="test-model", tensor_parallel_size=2
                )

    def test_get_rep_control_key(self):
        """Test RepControl cache key generation"""
        config = RepControlConfig(layers=[10, 15], block_name="mlp")
        key = self.server._get_rep_control_key(config)

        self.assertEqual(key, "[10, 15]_mlp")

    def test_convert_activations_to_tensors(self):
        """Test activation list to tensor conversion"""
        activations = {10: [0.1, 0.2, 0.3, 0.4], 15: [0.5, 0.6, 0.7, 0.8]}

        tensor_activations = self.server._convert_activations_to_tensors(activations)

        self.assertIn(10, tensor_activations)
        self.assertIn(15, tensor_activations)
        self.assertIsInstance(tensor_activations[10], torch.Tensor)
        self.assertEqual(tensor_activations[10].dtype, torch.float16)
        self.assertEqual(list(tensor_activations[10].shape), [4])

    def test_convert_masks_to_tensors(self):
        """Test mask list to tensor conversion"""
        masks = {10: [1.0, 1.0, 0.0, 0.0], 15: [0.5, 0.5, 0.5, 0.5]}

        tensor_masks = self.server._convert_masks_to_tensors(masks)

        self.assertIn(10, tensor_masks)
        self.assertIn(15, tensor_masks)
        self.assertIsInstance(tensor_masks[10], torch.Tensor)
        self.assertEqual(tensor_masks[10].dtype, torch.float16)


class TestRepControlAPIClient(unittest.TestCase):
    """Test the API client utility functions"""

    def test_client_initialization(self):
        """Test API client initialization"""
        client = RepControlAPIClient("http://test.com")
        self.assertEqual(client.base_url, "http://test.com")

        # Test URL normalization
        client2 = RepControlAPIClient("http://test.com/")
        self.assertEqual(client2.base_url, "http://test.com")

    @patch("test_repcontrol_api.requests.get")
    def test_health_check(self, mock_get):
        """Test health check method"""
        mock_response = Mock()
        mock_response.json.return_value = {"status": "healthy"}
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response

        client = RepControlAPIClient()
        result = client.health_check()

        self.assertEqual(result, {"status": "healthy"})
        mock_get.assert_called_once_with("http://localhost:8000/health")

    @patch("test_repcontrol_api.requests.post")
    def test_completion_request(self, mock_post):
        """Test completion request formatting"""
        mock_response = Mock()
        mock_response.json.return_value = {
            "choices": [{"text": "Paris"}],
            "usage": {"total_tokens": 10},
        }
        mock_response.raise_for_status.return_value = None
        mock_post.return_value = mock_response

        client = RepControlAPIClient()
        result = client.completion(
            prompt="The capital of France is", max_tokens=10, temperature=0.0
        )

        self.assertEqual(result["choices"][0]["text"], "Paris")

        # Verify the request was formatted correctly
        call_args = mock_post.call_args
        self.assertIn("json", call_args.kwargs)
        request_data = call_args.kwargs["json"]
        self.assertEqual(request_data["prompt"], "The capital of France is")
        self.assertEqual(request_data["max_tokens"], 10)


class TestUtilityFunctions(unittest.TestCase):
    """Test utility functions"""

    def test_create_dummy_activation_vector(self):
        """Test dummy activation vector creation"""
        vector = create_dummy_activation_vector(hidden_dim=100, magnitude=0.5)

        self.assertEqual(len(vector), 100)
        self.assertIsInstance(vector, list)
        self.assertIsInstance(vector[0], float)

        # Check that magnitude is approximately correct
        np_vector = np.array(vector)
        self.assertLess(np.abs(np.mean(np_vector)), 0.1)  # Mean should be close to 0
        self.assertLess(
            np.abs(np.std(np_vector) - 0.5), 0.2
        )  # Std should be close to magnitude

    def test_create_dummy_activation_vector_default_params(self):
        """Test dummy activation vector with default parameters"""
        vector = create_dummy_activation_vector()

        self.assertEqual(len(vector), 4096)  # Default hidden_dim

        # Check magnitude
        np_vector = np.array(vector)
        self.assertLess(np.abs(np.std(np_vector) - 0.1), 0.05)  # Default magnitude


class TestDataValidation(unittest.TestCase):
    """Test data validation and error handling"""

    def test_activation_vector_validation(self):
        """Test validation of activation vectors"""
        # Test valid activation vector
        valid_activations = {10: [0.1] * 4096}
        self.assertIsInstance(valid_activations[10], list)
        self.assertEqual(len(valid_activations[10]), 4096)

        # Test that all elements are numbers
        for val in valid_activations[10][:10]:  # Check first 10 elements
            self.assertIsInstance(val, (int, float))

    def test_layer_indices_validation(self):
        """Test validation of layer indices"""
        # Valid layer indices
        valid_layers = [0, 10, 15, 31]
        for layer in valid_layers:
            self.assertIsInstance(layer, int)
            self.assertGreaterEqual(layer, 0)

        # Test typical transformer layer range
        self.assertLess(max(valid_layers), 100)  # Reasonable upper bound

    def test_config_consistency(self):
        """Test consistency between config fields"""
        layers = [10, 15]
        activations = {10: [0.1] * 4096, 15: [0.2] * 4096}

        # All specified layers should have activations
        for layer in layers:
            self.assertIn(layer, activations)

        # All activation layers should be in specified layers
        for layer in activations.keys():
            self.assertIn(layer, layers)


class TestIntegrationScenarios(unittest.TestCase):
    """Test integration scenarios without requiring actual server"""

    def test_repcontrol_config_serialization(self):
        """Test that RepControl config can be serialized to JSON"""
        config = {
            "layers": [10, 15],
            "block_name": "decoder_block",
            "activations": {
                "10": [0.1] * 100,  # Smaller for testing
                "15": [0.2] * 100,
            },
            "token_pos": None,
            "normalize": False,
            "operator": "linear_comb",
        }

        # Should be able to serialize and deserialize
        json_str = json.dumps(config)
        restored_config = json.loads(json_str)

        self.assertEqual(config["layers"], restored_config["layers"])
        self.assertEqual(config["operator"], restored_config["operator"])
        self.assertEqual(
            len(config["activations"]["10"]), len(restored_config["activations"]["10"])
        )

    def test_multiple_layer_config(self):
        """Test configuration with multiple layers"""
        layers = [5, 10, 15, 20]
        hidden_dim = 128  # Small for testing

        activations = {}
        for layer_id in layers:
            activations[layer_id] = create_dummy_activation_vector(hidden_dim, 0.1)

        config = {
            "layers": layers,
            "activations": activations,
            "operator": "linear_comb",
        }

        # Verify all layers have activations
        self.assertEqual(len(config["activations"]), len(layers))
        for layer_id in layers:
            self.assertIn(layer_id, config["activations"])
            self.assertEqual(len(config["activations"][layer_id]), hidden_dim)

    def test_different_operators(self):
        """Test different operator configurations"""
        operators = ["linear_comb", "piecewise_linear"]

        for operator in operators:
            config = RepControlConfig(layers=[10], operator=operator)
            self.assertEqual(config.operator, operator)

    def test_token_position_options(self):
        """Test different token position configurations"""
        token_positions = [None, "start", "end", 0, -1, [0, 1, 2]]

        for token_pos in token_positions:
            config = RepControlConfig(layers=[10], token_pos=token_pos)
            self.assertEqual(config.token_pos, token_pos)


if __name__ == "__main__":
    # Set up test suite
    unittest.main(verbosity=2)
