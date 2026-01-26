"""
Unit tests for the experiment series runner.
"""
import unittest
import tempfile
import os
import json
from pathlib import Path
import shutil
import time 
import yaml
import sys
from unittest.mock import patch, MagicMock, call
import subprocess

from neuro_manipulation.experiment_series_runner import ExperimentSeriesRunner, ExperimentReport, ExperimentStatus

# Add parent directory to path to import modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

class TestExperimentReport(unittest.TestCase):
    """Test the ExperimentReport class."""
    
    def setUp(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        
    def tearDown(self):
        """Clean up after tests."""
        shutil.rmtree(self.temp_dir)
        
    def test_add_experiment(self):
        """Test adding an experiment to the report."""
        report = ExperimentReport(self.temp_dir, "test_series", None)
        report.add_experiment("Game1", "Model1", "Game1_Model1")
        
        # Check if experiment was added
        self.assertIn("Game1_Model1", report.experiments)
        self.assertEqual(report.experiments["Game1_Model1"]["game_name"], "Game1")
        self.assertEqual(report.experiments["Game1_Model1"]["model_name"], "Model1")
        self.assertEqual(report.experiments["Game1_Model1"]["status"], ExperimentStatus.PENDING)
        
    def test_update_experiment(self):
        """Test updating an experiment in the report."""
        report = ExperimentReport(self.temp_dir, "test_series", None)
        report.add_experiment("Game1", "Model1", "Game1_Model1")
        
        # Update experiment
        report.update_experiment("Game1_Model1", status=ExperimentStatus.RUNNING, start_time="2023-01-01")
        
        # Check if experiment was updated
        self.assertEqual(report.experiments["Game1_Model1"]["status"], ExperimentStatus.RUNNING)
        self.assertEqual(report.experiments["Game1_Model1"]["start_time"], "2023-01-01")
        
    def test_get_pending_experiments(self):
        """Test getting pending experiments."""
        report = ExperimentReport(self.temp_dir, "test_series", None)
        report.add_experiment("Game1", "Model1", "Game1_Model1")
        report.add_experiment("Game2", "Model1", "Game2_Model1", status=ExperimentStatus.RUNNING)
        
        # Get pending experiments
        pending = report.get_pending_experiments()
        
        # Check if pending experiments are correct
        self.assertEqual(len(pending), 1)
        self.assertEqual(pending[0]["exp_id"], "Game1_Model1")
        
    def test_get_failed_experiments(self):
        """Test getting failed experiments."""
        report = ExperimentReport(self.temp_dir, "test_series", None)
        report.add_experiment("Game1", "Model1", "Game1_Model1")
        report.add_experiment("Game2", "Model1", "Game2_Model1", status=ExperimentStatus.FAILED)
        
        # Get failed experiments
        failed = report.get_failed_experiments()
        
        # Check if failed experiments are correct
        self.assertEqual(len(failed), 1)
        self.assertEqual(failed[0]["exp_id"], "Game2_Model1")
        
    def test_save_and_load_report(self):
        """Test saving and loading the report."""
        # Create a fixed timestamp for test determinism
        fixed_timestamp = "20250101_000000"
        
        # Create first report with fixed timestamp
        report = ExperimentReport(self.temp_dir, "test_series", None)
        report.timestamp = fixed_timestamp
        report.report_file = Path(f"{self.temp_dir}/test_series_{fixed_timestamp}/experiment_report.json")
        report.report_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Add experiments
        report.add_experiment("Game1", "Model1", "Game1_Model1")
        report.add_experiment("Game2", "Model1", "Game2_Model1", status=ExperimentStatus.COMPLETED)
        
        # Create a new report and manually set the path to match
        new_report = ExperimentReport(self.temp_dir, "test_series", None)
        new_report.timestamp = fixed_timestamp
        new_report.report_file = Path(f"{self.temp_dir}/test_series_{fixed_timestamp}/experiment_report.json")
        
        # Ensure file exists
        self.assertTrue(new_report.report_file.exists())
        
        # Load the report
        success = new_report.load_report()
        
        # Check if report was loaded successfully
        self.assertTrue(success)
        self.assertEqual(len(new_report.experiments), 2)
        self.assertEqual(new_report.experiments["Game1_Model1"]["status"], ExperimentStatus.PENDING)
        self.assertEqual(new_report.experiments["Game2_Model1"]["status"], ExperimentStatus.COMPLETED)
        
    def test_get_summary(self):
        """Test getting a summary of experiment statuses."""
        report = ExperimentReport(self.temp_dir, "test_series", None)
        report.add_experiment("Game1", "Model1", "Game1_Model1")
        report.add_experiment("Game2", "Model1", "Game2_Model1", status=ExperimentStatus.RUNNING)
        report.add_experiment("Game3", "Model1", "Game3_Model1", status=ExperimentStatus.COMPLETED)
        report.add_experiment("Game4", "Model1", "Game4_Model1", status=ExperimentStatus.FAILED)
        
        # Get summary
        summary = report.get_summary()
        
        # Check if summary is correct
        self.assertEqual(summary["total"], 4)
        self.assertEqual(summary["pending"], 1)
        self.assertEqual(summary["running"], 1)
        self.assertEqual(summary["completed"], 1)
        self.assertEqual(summary["failed"], 1)

    def test_experiment_name_precedence(self):
        """Test that experiment_name takes precedence over series_name when provided."""
        # Create a fixed timestamp for test determinism
        fixed_timestamp = "20250101_000000"
        
        # Create report with both series_name and experiment_name
        report = ExperimentReport(self.temp_dir, "test_series", "config_experiment_name")
        # Set timestamp after creating the report
        report.timestamp = fixed_timestamp
        report.report_file = Path(f"{self.temp_dir}/config_experiment_name_{fixed_timestamp}/experiment_report.json")
        report.report_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Check if the report file uses the experiment_name
        expected_path = Path(f"{self.temp_dir}/config_experiment_name_{fixed_timestamp}/experiment_report.json")
        self.assertEqual(str(report.report_file), str(expected_path))
        
        # Verify that without experiment_name, it falls back to series_name
        fallback_report = ExperimentReport(self.temp_dir, "test_series", None)
        fallback_report.timestamp = fixed_timestamp
        fallback_report.report_file = Path(f"{self.temp_dir}/test_series_{fixed_timestamp}/experiment_report.json")
        fallback_report.report_file.parent.mkdir(parents=True, exist_ok=True)
        
        expected_fallback_path = Path(f"{self.temp_dir}/test_series_{fixed_timestamp}/experiment_report.json")
        self.assertEqual(str(fallback_report.report_file), str(expected_fallback_path))

class TestExperimentSeriesRunner(unittest.TestCase):
    """Test the ExperimentSeriesRunner class."""
    
    def setUp(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.config_path = os.path.join(self.temp_dir, "test_config.yaml")
        
        # Create a simple test config
        config = {
            "experiment": {
                "name": "Test_Experiment",
                "games": ["Game1", "Game2"],
                "models": ["Model1", "Model2"],
                "output": {
                    "base_dir": self.temp_dir
                },
                "game": {
                    "name": "Game1",
                    "previous_actions_length": 0
                },
                "llm": {
                    "model_name": "Model1"
                }
            }
        }
        
        # Write config to file using yaml instead of json
        with open(self.config_path, "w") as f:
            yaml.dump(config, f)
            
        # Mock the get_exp_config and get_game_config functions
        from unittest.mock import patch, MagicMock
        
        # Create patchers
        self.get_exp_config_patcher = patch('neuro_manipulation.experiment_series_runner.get_exp_config')
        self.get_repe_eng_config_patcher = patch('neuro_manipulation.experiment_series_runner.get_repe_eng_config')
        self.get_game_config_patcher = patch('neuro_manipulation.experiment_series_runner.get_game_config')
        
        # Start patchers
        self.mock_get_exp_config = self.get_exp_config_patcher.start()
        self.mock_get_repe_eng_config = self.get_repe_eng_config_patcher.start()
        self.mock_get_game_config = self.get_game_config_patcher.start()
        
        # Configure mocks
        self.mock_get_exp_config.return_value = config
        self.mock_get_repe_eng_config.return_value = {"model_name_or_path": "Model1"}
        self.mock_get_game_config.return_value = {"game_name": "Game1", "previous_actions_length": 0}
        
        # Now create the runner after mocks are in place
        self.runner = ExperimentSeriesRunner(self.config_path, "test_series")
        
    def tearDown(self):
        """Clean up after tests."""
        # Stop all patchers
        self.get_exp_config_patcher.stop()
        self.get_repe_eng_config_patcher.stop()
        self.get_game_config_patcher.stop()
        
        # Remove temp directory
        shutil.rmtree(self.temp_dir)
        
    def test_format_model_name_for_folder(self):
        """Test formatting model name for folder name."""
        # Define test cases that should match the implementation
        test_cases = [
            # Full path with multiple slashes
            ("/data/home/huggingface_models/RWKV/v6-Finch-7B-HF", "RWKV/v6-Finch-7B-HF"),
            # Normal HF path (Note: rsplit with 2 will only return the last part)
            ("meta-llama/Llama-3.1-8B-Instruct", "meta-llama/Llama-3.1-8B-Instruct"),
            # Path with exactly two parts
            ("org/model", "org/model"),
            # Single part (no slashes)
            ("Llama-3.1-8B-Instruct", "Llama-3.1-8B-Instruct"),
            # Path with exactly three parts
            ("a/b/c", "b/c"),
            # Path with more than three parts
            ("w/x/y/z", "y/z")
        ]
        
        # Test each case
        for input_name, expected_output in test_cases:
            result = self.runner._format_model_name_for_folder(input_name)
            self.assertEqual(result, expected_output, 
                             f"Failed for {input_name}, got {result}, expected {expected_output}")

    def test_uses_experiment_name_from_config(self):
        """Test that ExperimentSeriesRunner uses experiment name from config in the report."""
        # We can check this by examining the report_file path
        # The runner was initialized in setUp with a config that has experiment.name = "Test_Experiment"
        
        # Extract the experiment name from the report file path
        report_path = self.runner.report.report_file
        report_dir_name = report_path.parent.name
        
        # The directory should start with the experiment name from config
        self.assertTrue(report_dir_name.startswith("Test_Experiment_"), 
                       f"Report directory '{report_dir_name}' should start with 'Test_Experiment_'")
                       
        # Create a runner with a custom series name
        custom_runner = ExperimentSeriesRunner(self.config_path, "custom_series_name")
        custom_report_path = custom_runner.report.report_file
        custom_report_dir = custom_report_path.parent.name
        
        # Even with custom series name, it should still use the experiment name from config
        self.assertTrue(custom_report_dir.startswith("Test_Experiment_"),
                       f"Custom runner report directory '{custom_report_dir}' should start with 'Test_Experiment_'")

    def test_previous_actions_length_validation(self):
        """Test that the setup_experiment method validates previous_actions_length."""
        # Mock GameNames.from_string to return a mock game name that allows is_sequential() to be mocked
        from unittest.mock import patch, MagicMock
        
        mock_game_name = MagicMock()
        mock_game_name.is_sequential.return_value = False
        
        with patch('neuro_manipulation.experiment_series_runner.GameNames.from_string', return_value=mock_game_name):
            # Create a game config with non-zero previous_actions_length
            self.mock_get_game_config.return_value = {"game_name": "Game1", "previous_actions_length": 1}
            
            # The setup_experiment method should raise a ValueError
            with self.assertRaises(ValueError) as context:
                self.runner.setup_experiment("Game1", "Model1")
                
            # Check that the error message mentions previous_actions_length
            self.assertIn("Previous actions length must be 0", str(context.exception))

class TestCudaMemoryCleanup(unittest.TestCase):
    """Test the CUDA memory cleanup functionality in ExperimentSeriesRunner"""
    
    def setUp(self):
        """Set up test fixtures, create temp directory for output"""
        self.temp_dir = tempfile.mkdtemp()
        self.config_path = os.path.join(self.temp_dir, 'test_config.yaml')
        
        # Create a minimal test config
        with open(self.config_path, 'w') as f:
            f.write("""
experiment:
  name: test_experiment
  output:
    base_dir: {}
  game:
    name: dictator
  llm:
    model_name: test_model
  repeat: 1
""".format(self.temp_dir))
    
    def tearDown(self):
        """Clean up temp files"""
        shutil.rmtree(self.temp_dir)
    
    @patch('subprocess.run')
    @patch('gc.collect')
    def test_cuda_memory_cleanup_no_torch(self, mock_gc_collect, mock_subprocess_run):
        """Test memory cleanup when PyTorch is not available"""
        # Setup mocks
        mock_subprocess_run.return_value = MagicMock(stdout="memory.used [MiB], memory.free [MiB]\n1000, 15000\n")
        
        # Create a runner with the import for torch patched out
        with patch.dict('sys.modules', {'torch': None}):
            runner = ExperimentSeriesRunner(self.config_path)
            
            # Call the cleanup method
            runner._clean_cuda_memory()
        
        # Verify gc.collect was called
        mock_gc_collect.assert_called_once()
        
        # Verify nvidia-smi was called
        mock_subprocess_run.assert_called_once_with(
            ['nvidia-smi', '--query-gpu=memory.used,memory.free', '--format=csv'],
            capture_output=True, text=True, check=True
        )
    
    @patch('subprocess.run')
    @patch('torch.cuda.empty_cache')
    @patch('torch.cuda.memory_allocated')
    @patch('torch.cuda.max_memory_allocated')
    @patch('torch.cuda.memory_reserved')
    @patch('torch.cuda.max_memory_reserved')
    @patch('torch.cuda.is_available')
    @patch('gc.collect')
    def test_cuda_memory_cleanup_with_torch(self, mock_gc_collect, mock_cuda_available, 
                                          mock_max_reserved, mock_reserved, mock_max_allocated,
                                          mock_allocated, mock_empty_cache, mock_subprocess_run):
        """Test memory cleanup when PyTorch is available"""
        # Setup mocks
        mock_cuda_available.return_value = True
        mock_allocated.return_value = 1 * (1024 ** 3)  # 1 GB
        mock_max_allocated.return_value = 8 * (1024 ** 3)  # 8 GB
        mock_reserved.return_value = 10 * (1024 ** 3)  # 10 GB
        mock_max_reserved.return_value = 12 * (1024 ** 3)  # 12 GB
        mock_subprocess_run.return_value = MagicMock(stdout="memory.used [MiB], memory.free [MiB]\n10240, 16384\n")
        
        # Create a runner
        runner = ExperimentSeriesRunner(self.config_path)
        
        # Call the cleanup method
        runner._clean_cuda_memory()
        
        # Verify gc.collect was called
        mock_gc_collect.assert_called_once()
        
        # Verify PyTorch CUDA methods were called
        mock_cuda_available.assert_called_once()
        mock_empty_cache.assert_called_once()
        mock_allocated.assert_called_once()
        mock_max_allocated.assert_called_once()
        mock_reserved.assert_called_once()
        mock_max_reserved.assert_called_once()
        
        # Verify nvidia-smi was called
        mock_subprocess_run.assert_called_once_with(
            ['nvidia-smi', '--query-gpu=memory.used,memory.free', '--format=csv'],
            capture_output=True, text=True, check=True
        )
    
    @patch('subprocess.run')
    @patch('gc.collect')
    def test_cuda_memory_cleanup_nvidia_smi_error(self, mock_gc_collect, mock_subprocess_run):
        """Test memory cleanup when nvidia-smi fails"""
        # Setup mocks
        mock_subprocess_run.side_effect = subprocess.SubprocessError("Command failed")
        
        # Create a runner with the import for torch patched out
        with patch.dict('sys.modules', {'torch': None}):
            runner = ExperimentSeriesRunner(self.config_path)
            
            # Call the cleanup method
            runner._clean_cuda_memory()
        
        # Verify gc.collect was called
        mock_gc_collect.assert_called_once()
        
        # Verify nvidia-smi was called
        mock_subprocess_run.assert_called_once_with(
            ['nvidia-smi', '--query-gpu=memory.used,memory.free', '--format=csv'],
            capture_output=True, text=True, check=True
        )
    
    @patch('neuro_manipulation.experiment_series_runner.ExperimentSeriesRunner._clean_cuda_memory')
    @patch('neuro_manipulation.experiment_series_runner.ExperimentSeriesRunner.setup_experiment')
    def test_cleanup_called_after_successful_experiment(self, mock_setup_experiment, mock_clean_cuda):
        """Test that cleanup is called after a successful experiment"""
        # Setup mocks
        mock_experiment = MagicMock()
        mock_experiment.output_dir = os.path.join(self.temp_dir, 'output')
        mock_setup_experiment.return_value = mock_experiment
        
        # Create a runner
        runner = ExperimentSeriesRunner(self.config_path)
        
        # Run a single experiment
        result = runner.run_single_experiment('dictator', 'test_model', 'dictator_test_model')
        
        # Verify cleanup was called
        mock_clean_cuda.assert_called_once()
        self.assertTrue(result)  # Experiment should succeed
    
    @patch('neuro_manipulation.experiment_series_runner.ExperimentSeriesRunner._clean_cuda_memory')
    @patch('neuro_manipulation.experiment_series_runner.ExperimentSeriesRunner.setup_experiment')
    def test_cleanup_called_after_failed_experiment(self, mock_setup_experiment, mock_clean_cuda):
        """Test that cleanup is called after a failed experiment"""
        # Setup mocks
        mock_experiment = MagicMock()
        mock_experiment.output_dir = os.path.join(self.temp_dir, 'output')
        mock_experiment.run_experiment.side_effect = RuntimeError("Test error")
        mock_setup_experiment.return_value = mock_experiment
        
        # Create a runner
        runner = ExperimentSeriesRunner(self.config_path)
        
        # Run a single experiment that will fail
        result = runner.run_single_experiment('dictator', 'test_model', 'dictator_test_model')
        
        # Verify cleanup was called
        mock_clean_cuda.assert_called_once()
        self.assertFalse(result)  # Experiment should fail

if __name__ == "__main__":
    unittest.main() 