"""
Unit tests for model existence checking in ExperimentSeriesRunner
"""
import unittest
import os
import shutil
import io
from pathlib import Path
from unittest.mock import patch, MagicMock

from neuro_manipulation.experiment_series_runner import ExperimentSeriesRunner
import neuro_manipulation.experiment_series_runner as experiment_series_module

class TestModelCheck(unittest.TestCase):
    """Tests for the model checking functionality in ExperimentSeriesRunner"""
    
    def setUp(self):
        """Set up the test environment"""
        # Create a mock config for testing
        self.mock_config = {
            'experiment': {
                'output': {
                    'base_dir': 'results/test'
                },
                'name': 'test_experiment'
            }
        }
        
        # Create test directories to simulate cache and alternative locations
        self.home_dir = Path("test_home")
        self.cache_path = self.home_dir / ".cache" / "huggingface" / "hub"
        self.alt_path = Path("test_huggingface_models")
        
        # Create test directories
        os.makedirs(self.home_dir, exist_ok=True)
        os.makedirs(self.cache_path, exist_ok=True)
        os.makedirs(self.alt_path, exist_ok=True)
        
    def tearDown(self):
        """Clean up after tests"""
        # Remove test directories
        if os.path.exists(self.home_dir):
            shutil.rmtree(self.home_dir)
        if os.path.exists(self.alt_path):
            shutil.rmtree(self.alt_path)
        if os.path.exists("results"):
            shutil.rmtree("results")
    
    @patch('neuro_manipulation.experiment_series_runner.get_exp_config')
    @patch('neuro_manipulation.experiment_series_runner.os.path.expanduser')
    @patch('neuro_manipulation.experiment_series_runner.AutoConfig.from_pretrained')
    @patch('neuro_manipulation.experiment_series_runner.subprocess.Popen')
    def test_local_model_path(self, mock_popen, mock_config, mock_expanduser, mock_get_config):
        """Test that local model paths are skipped and return True"""
        # Configure mocks
        mock_get_config.return_value = self.mock_config
        mock_expanduser.return_value = str(self.home_dir)
        
        # Create runner instance
        runner = ExperimentSeriesRunner("dummy_path")
        
        # Test with a local path
        result = runner._check_model_existence("/data/home/models/my_model")
        
        # Assert that the resolved path is returned and no download was attempted
        self.assertEqual(result, "/data/home/models/my_model")
        mock_config.assert_not_called()
        mock_popen.assert_not_called()
    
    @patch('neuro_manipulation.experiment_series_runner.get_exp_config')
    @patch('neuro_manipulation.experiment_series_runner.os.path.expanduser')
    @patch('neuro_manipulation.experiment_series_runner.os.path.exists')
    def test_model_exists_in_alt_path_returns_local_dir(self, mock_exists, mock_expanduser, mock_get_config):
        """Return the local alt directory when the mirror already exists"""
        mock_get_config.return_value = self.mock_config
        mock_expanduser.return_value = str(self.home_dir)

        module_parent_dir = os.path.abspath(
            os.path.join(os.path.dirname(experiment_series_module.__file__), "..", "..")
        )
        alt_model_path = os.path.join(
            module_parent_dir, "huggingface_models", "facebook", "MobileLLM-R1-950M"
        )

        def exists_side_effect(path):
            if os.path.normpath(path) == os.path.normpath(alt_model_path):
                return True
            return False

        mock_exists.side_effect = exists_side_effect

        runner = ExperimentSeriesRunner("dummy_path")

        result = runner._check_model_existence("facebook/MobileLLM-R1-950M")

        self.assertEqual(result, alt_model_path)

    @patch('neuro_manipulation.experiment_series_runner.get_exp_config')
    @patch('neuro_manipulation.experiment_series_runner.os.path.expanduser')
    @patch('neuro_manipulation.experiment_series_runner.os.path.exists')
    def test_model_exists_in_cache(self, mock_exists, mock_expanduser, mock_get_config):
        """Test that the function returns True when the model exists in cache"""
        # Configure mocks
        mock_get_config.return_value = self.mock_config
        mock_expanduser.return_value = str(self.home_dir)
        
        # Make os.path.exists return True for the cache path
        def exists_side_effect(path):
            if "models--meta-llama--Llama-3.1-8B-Instruct" in path:
                return True
            return False
            
        mock_exists.side_effect = exists_side_effect
        
        # Create runner instance
        runner = ExperimentSeriesRunner("dummy_path")
        
        # Test with a model that "exists" in cache
        result = runner._check_model_existence("meta-llama/Llama-3.1-8B-Instruct")
        
        # Assert that we keep the repo id when relying on HF cache
        self.assertEqual(result, "meta-llama/Llama-3.1-8B-Instruct")
    
    @patch('neuro_manipulation.experiment_series_runner.get_exp_config')
    @patch('neuro_manipulation.experiment_series_runner.os.path.expanduser')
    @patch('neuro_manipulation.experiment_series_runner.os.path.exists')
    @patch('neuro_manipulation.experiment_series_runner.AutoConfig.from_pretrained')
    @patch('neuro_manipulation.experiment_series_runner.subprocess.Popen')
    def test_model_download(self, mock_popen, mock_config, mock_exists, 
                           mock_expanduser, mock_get_config):
        """Test model download functionality using huggingface-cli"""
        # Configure mocks
        mock_get_config.return_value = self.mock_config
        mock_expanduser.return_value = str(self.home_dir)
        mock_exists.return_value = False  # Model doesn't exist
        
        # Create realistic stdout with IO stream
        stdout_lines = ["Downloading...", "Download complete", ""]
        stdout = MagicMock()
        stdout.readline.side_effect = stdout_lines
        
        # Set up the process mock with proper return values
        process_mock = MagicMock()
        process_mock.stdout = stdout
        process_mock.poll.return_value = 0  # Success return code
        process_mock.stderr = MagicMock()
        process_mock.stderr.read.return_value = ""
        
        # Configure the mock Popen to return our process mock
        mock_popen.return_value = process_mock
        
        # Create runner instance
        runner = ExperimentSeriesRunner("dummy_path")
        
        # Test with a model that doesn't exist
        result = runner._check_model_existence("meta-llama/Llama-3.1-8B-Instruct")
        
        # Assert that the download was attempted and alt path returned
        module_parent_dir = os.path.abspath(
            os.path.join(os.path.dirname(experiment_series_module.__file__), "..", "..")
        )
        expected_alt = os.path.join(
            module_parent_dir, "huggingface_models", "meta-llama", "Llama-3.1-8B-Instruct"
        )
        self.assertEqual(result, expected_alt)
        mock_config.assert_called_once()
        mock_popen.assert_called_once()
        
        # Verify the command was correct
        cmd_args = mock_popen.call_args[0][0]
        self.assertEqual(cmd_args[0], "huggingface-cli")
        self.assertEqual(cmd_args[1], "download")
        self.assertEqual(cmd_args[2], "meta-llama/Llama-3.1-8B-Instruct")
        
        # Verify HF_HUB_ENABLE_HF_TRANSFER was set
        env_args = mock_popen.call_args[1]["env"]
        self.assertEqual(env_args["HF_HUB_ENABLE_HF_TRANSFER"], "1")
    
    @patch('neuro_manipulation.experiment_series_runner.get_exp_config')
    @patch('neuro_manipulation.experiment_series_runner.os.path.expanduser')
    @patch('neuro_manipulation.experiment_series_runner.os.path.exists')
    @patch('neuro_manipulation.experiment_series_runner.AutoConfig.from_pretrained')
    def test_model_not_on_huggingface(self, mock_config, mock_exists, 
                                     mock_expanduser, mock_get_config):
        """Test handling when model verification fails"""
        # Configure mocks
        mock_get_config.return_value = self.mock_config
        mock_expanduser.return_value = str(self.home_dir)
        mock_exists.return_value = False  # Model doesn't exist
        mock_config.side_effect = Exception("Model not found")
        
        # Create runner instance
        runner = ExperimentSeriesRunner("dummy_path")
        
        # Test with a model that will fail to verify
        result = runner._check_model_existence("nonexistent-model/not-real")
        
        # Assert that the result is None
        self.assertIsNone(result)
        mock_config.assert_called_once()
    
    @patch('neuro_manipulation.experiment_series_runner.get_exp_config')
    @patch('neuro_manipulation.experiment_series_runner.os.path.expanduser')
    @patch('neuro_manipulation.experiment_series_runner.os.path.exists')
    @patch('neuro_manipulation.experiment_series_runner.AutoConfig.from_pretrained')
    @patch('neuro_manipulation.experiment_series_runner.subprocess.Popen')
    def test_model_download_failure(self, mock_popen, mock_config, mock_exists, 
                                   mock_expanduser, mock_get_config):
        """Test handling of download failures"""
        # Configure mocks
        mock_get_config.return_value = self.mock_config
        mock_expanduser.return_value = str(self.home_dir)
        mock_exists.return_value = False  # Model doesn't exist
        
        # Create realistic stdout with IO stream
        stdout_lines = ["Downloading...", "Error occurred", ""]
        stdout = MagicMock()
        stdout.readline.side_effect = stdout_lines
        
        # Set up the process mock with proper return values
        process_mock = MagicMock()
        process_mock.stdout = stdout
        process_mock.poll.return_value = 1  # Error return code
        process_mock.stderr = MagicMock()
        process_mock.stderr.read.return_value = "Failed to download model: connection error"
        
        # Configure the mock Popen to return our process mock
        mock_popen.return_value = process_mock
        
        # Create runner instance
        runner = ExperimentSeriesRunner("dummy_path")
        
        # Test with a model that will fail to download
        result = runner._check_model_existence("meta-llama/Llama-3.1-8B-Instruct")
        
        # Assert that the result is None
        self.assertIsNone(result)
        mock_config.assert_called_once()
        mock_popen.assert_called_once()

if __name__ == '__main__':
    unittest.main() 
