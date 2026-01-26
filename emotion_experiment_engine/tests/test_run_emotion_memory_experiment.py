"""
Test for run_emotion_memory_experiment.py config parsing bug.
This test file is responsible for testing the standalone experiment runner.
Purpose: Expose the bug where runner expects dict but config provides list for benchmarks
"""
import unittest
import tempfile
from pathlib import Path
import shutil
import yaml

from ..run_emotion_memory_experiment import create_experiment_config, validate_config


class TestRunEmotionMemoryExperiment(unittest.TestCase):
    """Test the standalone experiment runner with actual config parsing"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.temp_dirs = []
        self.temp_files = []
    
    def tearDown(self):
        """Clean up temporary files and directories"""
        for temp_dir in self.temp_dirs:
            try:
                shutil.rmtree(temp_dir)
            except FileNotFoundError:
                pass
        
        for temp_file in self.temp_files:
            try:
                temp_file.unlink()
            except FileNotFoundError:
                pass
    
    def create_list_format_config(self) -> dict:
        """Create config with benchmarks as LIST (like memory_experiment_series.yaml)"""
        return {
            "experiment_name": "test_experiment",
            "models": ["test/model"],
            "benchmarks": [  # This is a LIST - will cause the bug
                {
                    "name": "infinitebench", 
                    "task_type": "passkey",
                    "sample_limit": 10,
                    "augmentation_config": None,
                    "enable_auto_truncation": True,
                    "truncation_strategy": "right", 
                    "preserve_ratio": 0.95
                }
            ],
            "emotions": ["anger"],
            "intensities": [1.0],
            "loading_config": {
                "model_path": "test/model",
                "gpu_memory_utilization": 0.9,
                "max_model_len": 1000
            },
            "generation_config": {"temperature": 0.1},
            "batch_size": 4,
            "output_dir": "test_output"
        }
    
    def create_dict_format_config(self) -> dict:
        """Create config with benchmarks as DICT (expected by runner)"""
        return {
            "model": {"model_path": "test/model"},
            "emotions": {"target_emotions": ["anger"], "intensities": [1.0]},
            "benchmarks": {  # This is a DICT - what runner expects
                "infinitebench_passkey": {
                    "name": "infinitebench",
                    "task_type": "passkey", 
                    "data_path": "test_data.jsonl",
                    "sample_limit": 10
                }
            },
            "execution": {"batch_size": 4},
            "output": {"results_dir": "test_output"}
        }
    
    def test_list_format_benchmarks_fails_with_items_error(self):
        """
        RED PHASE: Test that list format benchmarks cause 'list' object has no attribute 'items'
        
        This test exposes the bug where create_experiment_config tries to call .items()
        on a list instead of a dictionary.
        """
        config_dict = self.create_list_format_config()
        
        # This should fail with AttributeError: 'list' object has no attribute 'items'
        # because benchmarks is a list but code calls benchmarks.items()
        with self.assertRaises(AttributeError) as context:
            create_experiment_config(config_dict)
        
        # Verify it's specifically the 'items' attribute missing from list
        self.assertIn("'list' object has no attribute 'items'", str(context.exception))
    
    def test_validate_config_fails_with_list_format(self):
        """
        RED PHASE: Test that validation also fails with list format benchmarks
        
        The validation function also tries to iterate over benchmarks as dict.
        """
        config_dict = self.create_list_format_config()
        
        # Add required sections for basic validation
        config_dict["model"] = {"path": "/mock/model/path"}
        config_dict["execution"] = {"batch_size": 4}
        config_dict["output"] = {"results_dir": "test_output"}
        
        # This should fail because validation tries to access benchmarks as dict
        with self.assertRaises(AttributeError) as context:
            validate_config(config_dict)
        
        self.assertIn("'list' object has no attribute 'items'", str(context.exception))
    
    def test_dict_format_benchmarks_works(self):
        """
        Test that dictionary format works (for comparison)
        
        This test shows what the expected format should be.
        """
        config_dict = self.create_dict_format_config()
        
        # Create temporary data file
        temp_file = Path(tempfile.mktemp(suffix='.jsonl'))
        temp_file.write_text('{"test": "data"}\n')
        self.temp_files.append(temp_file)
        
        # Update config to use real file path
        benchmark_config = list(config_dict["benchmarks"].values())[0]
        benchmark_config["data_path"] = str(temp_file)
        
        # This should work without errors (though may fail for other reasons)
        try:
            result = create_experiment_config(config_dict)
            # If we get here, the .items() bug is fixed
            self.assertIsNotNone(result)
        except KeyError:
            # May fail due to missing keys, but NOT due to .items() 
            pass  # Expected - we're only testing the .items() bug
        except FileNotFoundError:
            # May fail due to missing model files, but NOT due to .items()
            pass  # Expected - we're only testing the .items() bug


if __name__ == '__main__':
    unittest.main()