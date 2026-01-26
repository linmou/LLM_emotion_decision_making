#!/usr/bin/env python3
"""
Test file for import coverage - ensures all modules can be imported.

This test suite prevents import regressions during refactoring by:
1. Testing that all Python modules can be imported without errors
2. Verifying that data model exports match the expected API
3. Catching cases where imports reference non-existent classes

This would have caught the LoadingConfig→VLLMLoadingConfig refactoring issues immediately.
"""
import unittest
import importlib
import sys
from pathlib import Path


class TestImportCoverage(unittest.TestCase):
    """Test that all Python files in the project can be imported without errors"""
    
    def test_emotion_experiment_series_runner_import(self):
        """Test that emotion_experiment_series_runner can be imported"""
        # Note: Full test coverage for emotion_experiment_series_runner is now in 
        # test_emotion_experiment_series_runner.py - this is just basic import test
        try:
            import emotion_experiment_engine.emotion_experiment_series_runner
            self.assertTrue(True, "Import should succeed after fix")
        except AttributeError as e:
            if "LoadingConfig" in str(e):
                self.fail(f"LoadingConfig import error (expected during Red phase): {e}")
            else:
                raise
        except ImportError as e:
            self.fail(f"Unexpected import error: {e}")
    
    def test_run_emotion_memory_experiment_import(self):
        """Test that run_emotion_memory_experiment can be imported"""
        try:
            import emotion_experiment_engine.run_emotion_memory_experiment
            self.assertTrue(True, "Import should succeed after fix")
        except AttributeError as e:
            if "LoadingConfig" in str(e):
                self.fail(f"LoadingConfig import error (expected during Red phase): {e}")
            else:
                raise
        except ImportError as e:
            self.fail(f"Unexpected import error: {e}")
    
    def test_data_models_exports(self):
        """Test that data_models exports the correct classes after factory removal"""
        from emotion_experiment_engine.data_models import (
            VLLMLoadingConfig,
            BenchmarkConfig, 
            ExperimentConfig,
        )
        
        # These dataclasses should exist
        self.assertTrue(hasattr(VLLMLoadingConfig, 'to_vllm_kwargs'))
        self.assertTrue(hasattr(BenchmarkConfig, 'get_data_path'))
        self.assertTrue(hasattr(ExperimentConfig, '__init__'))
        
        # Old LoadingConfig should not exist
        try:
            from emotion_experiment_engine.data_models import LoadingConfig
            self.fail("LoadingConfig should not exist after refactoring")
        except ImportError:
            pass  # Expected - LoadingConfig shouldn't exist
    
    def test_direct_dataclass_instantiation_coverage(self):
        """Test that dataclasses can be instantiated directly without factory functions"""
        from emotion_experiment_engine.data_models import VLLMLoadingConfig
        
        # Test direct instantiation works (replaces factory functions)
        config = VLLMLoadingConfig(
            model_path="/test/model",
            gpu_memory_utilization=0.85,
            tensor_parallel_size=None,
            max_model_len=16384,
            enforce_eager=True,
            quantization=None,
            trust_remote_code=True,
            dtype="float16",
            seed=42,
            disable_custom_all_reduce=False,
            additional_vllm_kwargs={}
        )
        
        self.assertEqual(config.model_path, "/test/model")
        self.assertEqual(config.gpu_memory_utilization, 0.85)
        self.assertEqual(config.max_model_len, 16384)
        
        # Test to_vllm_kwargs works
        kwargs = config.to_vllm_kwargs()
        self.assertIn("model", kwargs)
        self.assertEqual(kwargs["model"], "/test/model")


if __name__ == "__main__":
    unittest.main()