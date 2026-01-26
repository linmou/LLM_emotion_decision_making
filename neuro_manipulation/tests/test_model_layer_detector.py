import unittest
import torch
import os
import sys
import logging

# Add the parent directory to the path to import the module
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from model_layer_detector import ModelLayerDetector

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TestModelLayerDetector(unittest.TestCase):
    """Test the ModelLayerDetector class with various model architectures"""
    
    @classmethod
    def setUpClass(cls):
        """Set up environment variables and check for GPU"""
        # Set environment variables for models that require them
        os.environ["HF_ENDPOINT"] = "https://huggingface.co"
        
        # Skip tests if CUDA is not available
        cls.skip_gpu_tests = not torch.cuda.is_available()
        if cls.skip_gpu_tests:
            logger.warning("CUDA not available, skipping GPU tests")
    
    def test_layer_detection_small_models(self):
        """Test layer detection with small models"""
        # Skip if running on CI without proper credentials
        if os.environ.get('CI') == 'true':
            self.skipTest("Skipping test on CI environment")
            
        # List of small models to test
        models_to_test = [
            "gpt2",
            "facebook/opt-125m",
        ]
        
        for model_name in models_to_test:
            with self.subTest(model=model_name):
                try:
                    logger.info(f"Testing layer detection on {model_name}")
                    from transformers import AutoModelForCausalLM
                    
                    # Load model in half precision for efficiency
                    model = AutoModelForCausalLM.from_pretrained(
                        model_name, 
                        torch_dtype=torch.float16,
                        trust_remote_code=True
                    )
                    
                    # Test layer detection
                    layers = ModelLayerDetector.get_model_layers(model)
                    
                    # Verify the result is a ModuleList
                    self.assertIsInstance(layers, torch.nn.ModuleList)
                    
                    # Verify it has more than one layer
                    self.assertGreater(len(layers), 0)
                    
                    logger.info(f"✓ Successfully detected layers for {model_name}: Found {len(layers)} layers")
                except Exception as e:
                    logger.error(f"Error with model {model_name}: {str(e)}")
                    raise
    
    @unittest.skipIf(False, "Testing ChatGLM model specifically")
    def test_chatglm_model(self):
        """Test layer detection with ChatGLM models"""
        if self.skip_gpu_tests:
            self.skipTest("CUDA not available, skipping ChatGLM test")
            
        try:
            logger.info("Testing layer detection on ChatGLM model")
            from transformers import AutoModel
            
            model = AutoModel.from_pretrained(
                "THUDM/glm-4-9b", 
                torch_dtype=torch.float16, 
                device_map="auto", 
                trust_remote_code=True
            )
            
            # Print model structure to help with debugging
            logger.info("Model structure:")
            ModelLayerDetector.print_model_structure(model)
            
            # Test layer detection
            layers = ModelLayerDetector.get_model_layers(model)
            
            # Verify the result is a ModuleList
            self.assertIsInstance(layers, torch.nn.ModuleList)
            
            # Verify it has more than one layer
            self.assertGreater(len(layers), 0)
            
            logger.info(f"✓ Successfully detected layers for ChatGLM model: Found {len(layers)} layers")
        except Exception as e:
            logger.error(f"Error with ChatGLM model: {str(e)}")
            raise
    
    @unittest.skipIf(False, "Testing RWKV model specifically")
    def test_rwkv_model(self):
        """Test layer detection with RWKV models"""
        if self.skip_gpu_tests:
            self.skipTest("CUDA not available, skipping RWKV test")
            
        try:
            logger.info("Testing layer detection on RWKV model")
            from transformers import AutoModelForCausalLM
            from transformers.models.rwkv.modeling_rwkv import RwkvBlock
            model = AutoModelForCausalLM.from_pretrained(
                "RWKV/rwkv-4-world-169m", 
                torch_dtype=torch.float16,
                trust_remote_code=True
            )
            
            # Print model structure to help with debugging
            logger.info("RWKV model structure:")
            ModelLayerDetector.print_model_structure(model)
            
            # Test layer detection
            layers = ModelLayerDetector.get_model_layers(model)
            
            # Verify the result is a ModuleList
            self.assertIsInstance(layers, torch.nn.ModuleList)
            self.assertIsInstance(layers[0], RwkvBlock)
             
            # Verify it has more than one layer
            self.assertGreater(len(layers), 0)
            
            logger.info(f"✓ Successfully detected layers for RWKV model: Found {len(layers)} layers")
        except Exception as e:
            logger.error(f"Error with RWKV model: {str(e)}")
            raise
    
    def test_with_custom_model(self):
        """Test with a custom model structure to ensure robustness"""
        class CustomTransformerLayer(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.self_attn = torch.nn.Linear(128, 128)
                self.mlp = torch.nn.Sequential(
                    torch.nn.Linear(128, 512),
                    torch.nn.ReLU(),
                    torch.nn.Linear(512, 128)
                )
                self.norm1 = torch.nn.LayerNorm(128)
                self.norm2 = torch.nn.LayerNorm(128)
            
            def forward(self, x):
                return x
        
        class NestedModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.embedding = torch.nn.Embedding(1000, 128)
                self.deeply = torch.nn.Module()
                self.deeply.nested = torch.nn.Module()
                self.deeply.nested.model = torch.nn.Module()
                self.deeply.nested.model.layers = torch.nn.ModuleList([
                    CustomTransformerLayer() for _ in range(6)
                ])
            
            def forward(self, x):
                return x
        
        model = NestedModel()
        
        # Test layer detection
        layers = ModelLayerDetector.get_model_layers(model)
        
        # Verify the result is a ModuleList
        self.assertIsInstance(layers, torch.nn.ModuleList)
        
        # Verify it found the nested layers
        self.assertEqual(len(layers), 6)

    def test_layer_index_access(self):
        """Test accessing specific layer indices"""
        class CustomTransformerLayer(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.self_attn = torch.nn.Linear(128, 128)
                self.mlp = torch.nn.Sequential(
                    torch.nn.Linear(128, 512),
                    torch.nn.ReLU(),
                    torch.nn.Linear(512, 128)
                )
                self.norm1 = torch.nn.LayerNorm(128)
                self.norm2 = torch.nn.LayerNorm(128)
            
            def forward(self, x):
                return x

        class NestedModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.embedding = torch.nn.Embedding(1000, 128)
                self.deeply = torch.nn.Module()
                self.deeply.nested = torch.nn.Module()
                self.deeply.nested.model = torch.nn.Module()
                self.deeply.nested.model.layers = torch.nn.ModuleList([
                    CustomTransformerLayer() for _ in range(6)
                ])
            
            def forward(self, x):
                return x

        model = NestedModel()
        layers = ModelLayerDetector.get_model_layers(model)

        # Test positive indexing
        first_layer = layers[0]
        self.assertIsInstance(first_layer, CustomTransformerLayer)
        self.assertTrue(hasattr(first_layer, 'self_attn'))

        # Test negative indexing
        last_layer = layers[-1]
        self.assertIsInstance(last_layer, CustomTransformerLayer)
        self.assertTrue(hasattr(last_layer, 'self_attn'))

        # Test slicing
        middle_layers = layers[2:4]
        self.assertEqual(len(middle_layers), 2)
        for layer in middle_layers:
            self.assertIsInstance(layer, CustomTransformerLayer)

    def test_layer_iteration(self):
        """Test iterating over layers and performing operations"""
        class CustomTransformerLayer(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.self_attn = torch.nn.Linear(128, 128)
                self.mlp = torch.nn.Sequential(
                    torch.nn.Linear(128, 512),
                    torch.nn.ReLU(),
                    torch.nn.Linear(512, 128)
                )
                self.norm1 = torch.nn.LayerNorm(128)
                self.norm2 = torch.nn.LayerNorm(128)
            
            def forward(self, x):
                return x

        class NestedModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.embedding = torch.nn.Embedding(1000, 128)
                self.deeply = torch.nn.Module()
                self.deeply.nested = torch.nn.Module()
                self.deeply.nested.model = torch.nn.Module()
                self.deeply.nested.model.layers = torch.nn.ModuleList([
                    CustomTransformerLayer() for _ in range(6)
                ])
            
            def forward(self, x):
                return x

        model = NestedModel()
        layers = ModelLayerDetector.get_model_layers(model)

        # Test enumeration
        for idx, layer in enumerate(layers):
            self.assertIsInstance(layer, CustomTransformerLayer)
            self.assertEqual(layer, layers[idx])

        # Test range-based iteration
        layer_count = len(layers)
        for i in range(layer_count):
            self.assertIsInstance(layers[i], CustomTransformerLayer)

        # Test list operations
        layer_list = list(layers)
        self.assertEqual(len(layer_list), 6)
        self.assertIsInstance(layer_list[0], CustomTransformerLayer)

        # Test reverse iteration
        reversed_layers = reversed(layers)
        for idx, layer in enumerate(reversed_layers):
            self.assertIsInstance(layer, CustomTransformerLayer)
            self.assertEqual(layer, layers[-(idx + 1)])

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA not available, skipping vLLM test")
    def test_vllm_model_layers(self):
        """Test layer detection with vLLM models, specifically Llama-3.1-8B-Instruct"""
        try:
            logger.info("Testing layer detection on vLLM model: meta-llama/Llama-3.1-8B-Instruct")
            
            # Import vLLM
            try:
                from vllm import LLM
            except ImportError:
                self.skipTest("vLLM not installed, skipping test")
            
            # Initialize vLLM model with minimal resources for testing
            llm = LLM(
                model="meta-llama/Llama-3.1-8B-Instruct",
                trust_remote_code=True,
                max_model_len=40000,
                tensor_parallel_size=1,
                gpu_memory_utilization=0.9
            )
            
            # Get the actual model from vLLM's nested structure
            model = llm.llm_engine.model_executor.driver_worker.model_runner.model
            
            # Print the model structure to help with debugging
            logger.info("vLLM model structure:")
            ModelLayerDetector.print_model_structure(model)
            
            # Test layer detection
            layers = ModelLayerDetector.get_model_layers(model)
            
            # Verify the result is a ModuleList
            self.assertIsInstance(layers, torch.nn.ModuleList)
            
            # Verify it has more than one layer
            self.assertGreater(len(layers), 0)
            
            # Check if the detected layers match the expected path
            # Access a specific layer to verify the structure
            try:
                # This is the expected path in vLLM
                expected_layers = model.model.layers
                
                # Verify they're the same layers
                for i in range(len(layers)):
                    self.assertIs(layers[i], expected_layers[i])
                
                # Verify MLP access works as expected (used in rep_control_vllm.py)
                test_layer = layers[0].mlp
                self.assertIsNotNone(test_layer, "Could not access MLP in the layer")
                
                logger.info(f"✓ Successfully detected layers for vLLM model: Found {len(layers)} layers")
                logger.info(f"✓ Verified correct layer path: model.model.layers")
                
            except AttributeError as e:
                logger.error(f"Expected layer structure not found: {str(e)}")
                raise
                
        except Exception as e:
            logger.error(f"Error with vLLM model: {str(e)}")
            raise

if __name__ == "__main__":
    unittest.main() 
