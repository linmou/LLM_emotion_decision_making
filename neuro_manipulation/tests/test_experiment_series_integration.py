"""
Integration tests for the experiment series runner.
Tests the complete workflow including stopping and resuming experiments.
"""
import unittest
import tempfile
import os
import json
import yaml
import time
import signal
import threading
from unittest.mock import patch, MagicMock, Mock
from pathlib import Path
import shutil
from datetime import datetime

from neuro_manipulation.experiment_series_runner import ExperimentSeriesRunner, ExperimentReport, ExperimentStatus

class MockEmotionGameExperiment:
    """Mock class for EmotionGameExperiment to simulate experiment running."""
    
    def __init__(self, *args, **kwargs):
        self.output_dir = kwargs.get('output_dir', 'mock_output_dir')
        self.exp_config = kwargs.get('exp_config', {})
        self.should_fail = kwargs.get('should_fail', False)
        self.sleep_time = kwargs.get('sleep_time', 0.1)
    
    def run_experiment(self):
        """Simulate running an experiment."""
        time.sleep(self.sleep_time)
        if self.should_fail:
            raise Exception("Mock experiment failure")
        return {}
    
    def run_sanity_check(self):
        """Simulate running a sanity check."""
        time.sleep(self.sleep_time)
        if self.should_fail:
            raise Exception("Mock sanity check failure")
        return {}

class TestExperimentSeriesIntegration(unittest.TestCase):
    """Integration tests for the experiment series runner."""
    
    def setUp(self):
        """Set up test environment with a mock config."""
        self.temp_dir = tempfile.mkdtemp()
        self.config_path = os.path.join(self.temp_dir, "test_config.yaml")
        
        # Create a test config with multiple games and models
        config = {
            "experiment": {
                "name": "Test_Experiment_Series",
                "games": ["Game1", "Game2", "Game3"],
                "models": ["Model1", "Model2"],
                "output": {
                    "base_dir": self.temp_dir
                },
                "game": {
                    "name": "Game1"  # Default game
                },
                "llm": {
                    "model_name": "Model1"  # Default model
                },
                "run_sanity_check": False,
                "repeat": 1,  # Add required repeat field
                "batch_size": 4,  # Add batch size for safety
                # Additional required fields
                "emotions": ["anger", "happiness"],
                "intensity": [1.0],
                "system_message_template": "Test message template"
            }
        }
        
        with open(self.config_path, 'w') as f:
            yaml.dump(config, f)
        
    def tearDown(self):
        """Clean up after tests."""
        shutil.rmtree(self.temp_dir)
    
    @patch('neuro_manipulation.experiment_series_runner.EmotionGameExperiment')
    @patch('neuro_manipulation.experiment_series_runner.get_repe_eng_config')
    @patch('neuro_manipulation.experiment_series_runner.get_game_config')
    @patch('neuro_manipulation.experiment_series_runner.GameNames')
    @patch('neuro_manipulation.experiment_series_runner.repe_pipeline_registry')
    def test_full_experiment_series(self, mock_registry, mock_game_names, mock_game_config,
                                   mock_repe_eng_config, mock_emotion_game_exp):
        """Test running a full experiment series."""
        # Setup mocks
        mock_game_instance = MagicMock()
        mock_game_names.from_string.return_value = mock_game_instance
        mock_game_instance.is_sequential.return_value = False
        
        mock_repe_eng_config.return_value = {}
        mock_game_config.return_value = {}
        
        # Mock the EmotionGameExperiment to return a controlled instance
        def create_mock_exp(*args, **kwargs):
            game_name = args[1]['experiment']['game']['name']
            model_name = args[1]['experiment']['llm']['model_name']
            output_dir = os.path.join(self.temp_dir, f"{game_name}_{model_name}_{int(time.time())}")
            kwargs['output_dir'] = output_dir
            kwargs['exp_config'] = args[1]
            return MockEmotionGameExperiment(*args, **kwargs)
        
        mock_emotion_game_exp.side_effect = create_mock_exp
        
        # Create and run the experiment series
        runner = ExperimentSeriesRunner(self.config_path, "test_series")
        runner.run_experiment_series()
        
        # Verify all experiments were attempted
        self.assertEqual(mock_emotion_game_exp.call_count, 6)  # 3 games x 2 models
        
        # Verify report contains all experiments
        self.assertEqual(len(runner.report.experiments), 6)
        
        # Verify all experiments are marked as completed
        completed_count = sum(1 for exp in runner.report.experiments.values() 
                             if exp["status"] == ExperimentStatus.COMPLETED)
        self.assertEqual(completed_count, 6)
        
        # Verify report file was created
        self.assertTrue(runner.report.report_file.exists())
        
        # Load the report file and verify its contents
        with open(runner.report.report_file, 'r') as f:
            report_data = json.load(f)
        
        self.assertEqual(len(report_data['experiments']), 6)
    
    @patch('neuro_manipulation.experiment_series_runner.EmotionGameExperiment')
    @patch('neuro_manipulation.experiment_series_runner.get_repe_eng_config')
    @patch('neuro_manipulation.experiment_series_runner.get_game_config')
    @patch('neuro_manipulation.experiment_series_runner.GameNames')
    @patch('neuro_manipulation.experiment_series_runner.repe_pipeline_registry')
    def test_experiment_with_failures(self, mock_registry, mock_game_names, mock_game_config,
                                    mock_repe_eng_config, mock_emotion_game_exp):
        """Test running experiments with some failures."""
        # Setup mocks
        mock_game_instance = MagicMock()
        mock_game_names.from_string.return_value = mock_game_instance
        mock_game_instance.is_sequential.return_value = False
        
        mock_repe_eng_config.return_value = {}
        mock_game_config.return_value = {}
        
        # Mock to make specific experiments fail
        def create_mock_exp(*args, **kwargs):
            game_name = args[1]['experiment']['game']['name']
            model_name = args[1]['experiment']['llm']['model_name']
            
            # Make Game2 with Model1 fail
            should_fail = (game_name == "Game2" and model_name == "Model1")
            
            output_dir = os.path.join(self.temp_dir, f"{game_name}_{model_name}_{int(time.time())}")
            kwargs['output_dir'] = output_dir
            kwargs['exp_config'] = args[1]
            kwargs['should_fail'] = should_fail
            
            return MockEmotionGameExperiment(*args, **kwargs)
        
        mock_emotion_game_exp.side_effect = create_mock_exp
        
        # Create and run the experiment series
        runner = ExperimentSeriesRunner(self.config_path, "test_series")
        runner.run_experiment_series()
        
        # Verify all experiments were attempted
        self.assertEqual(mock_emotion_game_exp.call_count, 6)  # 3 games x 2 models
        
        # Verify report contains all experiments
        self.assertEqual(len(runner.report.experiments), 6)
        
        # Verify correct experiment failed
        failed_experiments = runner.report.get_failed_experiments()
        self.assertEqual(len(failed_experiments), 1)
        self.assertEqual(failed_experiments[0]["game_name"], "Game2")
        self.assertEqual(failed_experiments[0]["model_name"], "Model1")
        
        # Verify other experiments completed
        completed_count = sum(1 for exp in runner.report.experiments.values() 
                             if exp["status"] == ExperimentStatus.COMPLETED)
        self.assertEqual(completed_count, 5)
    
    @patch('neuro_manipulation.experiment_series_runner.EmotionGameExperiment')
    @patch('neuro_manipulation.experiment_series_runner.get_repe_eng_config')
    @patch('neuro_manipulation.experiment_series_runner.get_game_config')
    @patch('neuro_manipulation.experiment_series_runner.GameNames')
    @patch('neuro_manipulation.experiment_series_runner.repe_pipeline_registry')
    def test_experiment_resume(self, mock_registry, mock_game_names, mock_game_config,
                             mock_repe_eng_config, mock_emotion_game_exp):
        """Test resuming an interrupted experiment series."""
        # Setup mocks
        mock_game_instance = MagicMock()
        mock_game_names.from_string.return_value = mock_game_instance
        mock_game_instance.is_sequential.return_value = False
        
        mock_repe_eng_config.return_value = {}
        mock_game_config.return_value = {}

        # Helper to format model name like the runner does
        def format_model_name(model_name: str) -> str:
            return model_name.rsplit('/', 1)[-1]

        # Create the runner first to access its report object and flag
        runner = ExperimentSeriesRunner(self.config_path, "test_series")
        
        # Counter to track experiment setup calls in the first run
        exp_setup_count = 0
        
        # First run: Run first 4 experiments, then simulate interrupt before the 5th
        def create_mock_exp_first_run(*args, **kwargs):
            nonlocal exp_setup_count
            exp_setup_count += 1
            
            exp_config_dict = args[1]
            game_name = exp_config_dict['experiment']['game']['name'] 
            model_name = exp_config_dict['experiment']['llm']['model_name']
            model_folder_name = format_model_name(model_name)
            exp_id = f"{game_name}_{model_folder_name}"

            output_dir = os.path.join(self.temp_dir, f"{game_name}_{model_folder_name}_{int(time.time())}")
            kwargs['output_dir'] = output_dir
            kwargs['exp_config'] = exp_config_dict

            # Simulate successful run side effect by updating the report
            # This happens *before* run_experiment is called, so mark as running first
            runner.report.update_experiment(exp_id, 
                                             status=ExperimentStatus.RUNNING, # Mark as running
                                             start_time=datetime.now().isoformat(),
                                             output_dir=output_dir)

            mock_exp = MockEmotionGameExperiment(*args, **kwargs)
            
            # In the Mock's run_experiment, we'll mark it completed
            original_run_exp = mock_exp.run_experiment
            def run_and_complete():
                 original_run_exp()
                 runner.report.update_experiment(exp_id, 
                                                  status=ExperimentStatus.COMPLETED, 
                                                  end_time=datetime.now().isoformat())
            mock_exp.run_experiment = run_and_complete

            # Request shutdown *after* setting up the 4th experiment
            # The runner loop will check the flag *after* this 4th experiment runs
            if exp_setup_count == 4: # Game2/Model2 is the 4th combination
                runner.shutdown_requested = True
                
            return mock_exp # Return the mock instance
        
        mock_emotion_game_exp.side_effect = create_mock_exp_first_run
        
        # First run - no custom exception needed
        runner.run_experiment_series()
        
        # Verify shutdown was requested
        self.assertTrue(runner.shutdown_requested)

        # Store the report file path for resuming
        report_file_path = runner.report.report_file
        series_name = runner.series_name
        timestamp = runner.report.timestamp
        
        # Verify first run state: 4 completed, 2 pending
        runner.report.load_report() 
        completed_count = sum(1 for exp in runner.report.experiments.values() 
                             if exp["status"] == ExperimentStatus.COMPLETED)
        self.assertEqual(completed_count, 4, f"Expected 4 completed, found {completed_count}. Report: {runner.report.experiments}")
        
        pending_count = sum(1 for exp in runner.report.experiments.values() 
                           if exp["status"] == ExperimentStatus.PENDING)
        self.assertEqual(pending_count, 2, f"Expected 2 pending, found {pending_count}. Report: {runner.report.experiments}")
        
        # Reset the mock for second run
        mock_emotion_game_exp.reset_mock()
        mock_emotion_game_exp.side_effect = None 

        # Second run: Complete the remaining experiments
        def create_mock_exp_second_run(*args, **kwargs):
            exp_config_dict = args[1]
            game_name = exp_config_dict['experiment']['game']['name']
            model_name = exp_config_dict['experiment']['llm']['model_name']
            model_folder_name = format_model_name(model_name)
            output_dir = os.path.join(self.temp_dir, f"{game_name}_{model_folder_name}_{int(time.time())}")
            kwargs['output_dir'] = output_dir
            kwargs['exp_config'] = exp_config_dict
            # Return a simple mock for the resume run
            return MockEmotionGameExperiment(*args, **kwargs) 
        
        mock_emotion_game_exp.side_effect = create_mock_exp_second_run
        
        # Create a new runner with resume=True and the same parameters
        resume_runner = ExperimentSeriesRunner(self.config_path, series_name, resume=True)
        resume_runner.report.timestamp = timestamp 
        resume_runner.report.report_file = report_file_path 
        if not resume_runner.report.load_report():
             self.fail("Failed to load report for resume run")
        
        # Run the remaining experiments
        resume_runner.run_experiment_series()
        
        # Verify second run processed only pending experiments
        # Expecting 2 calls for the remaining experiments
        self.assertEqual(mock_emotion_game_exp.call_count, 2, f"Expected 2 calls in resume, got {mock_emotion_game_exp.call_count}")  # Only the 2 pending

        # Verify all experiments are now completed
        resume_runner.report.load_report() 
        completed_count = sum(1 for exp in resume_runner.report.experiments.values() 
                             if exp["status"] == ExperimentStatus.COMPLETED)
        self.assertEqual(completed_count, 6, f"Expected 6 completed, found {completed_count}. Report: {resume_runner.report.experiments}")
        failed_count = sum(1 for exp in resume_runner.report.experiments.values() if exp["status"] == ExperimentStatus.FAILED)
        self.assertEqual(failed_count, 0, f"Expected 0 failed, found {failed_count}. Report: {resume_runner.report.experiments}")
    
    @patch('neuro_manipulation.experiment_series_runner.EmotionGameExperiment')
    @patch('neuro_manipulation.experiment_series_runner.get_repe_eng_config')
    @patch('neuro_manipulation.experiment_series_runner.get_game_config')
    @patch('neuro_manipulation.experiment_series_runner.GameNames')
    @patch('neuro_manipulation.experiment_series_runner.repe_pipeline_registry')
    def test_shutdown_handler(self, mock_registry, mock_game_names, mock_game_config,
                             mock_repe_eng_config, mock_emotion_game_exp):
        """Test the shutdown handler correctly processes SIGINT."""
        # Setup mocks
        mock_game_instance = MagicMock()
        mock_game_names.from_string.return_value = mock_game_instance
        mock_game_instance.is_sequential.return_value = False
        
        mock_repe_eng_config.return_value = {}
        mock_game_config.return_value = {}
        
        # Make experiments take a bit longer so we can interrupt
        def create_mock_exp(*args, **kwargs):
            output_dir = os.path.join(self.temp_dir, f"{args[1]['experiment']['game']['name']}_{args[1]['experiment']['llm']['model_name']}_{int(time.time())}")
            kwargs['output_dir'] = output_dir
            kwargs['exp_config'] = args[1]
            kwargs['sleep_time'] = 0.1  # Make it sleep a bit
            return MockEmotionGameExperiment(*args, **kwargs)
        
        mock_emotion_game_exp.side_effect = create_mock_exp
        
        # Create the runner
        runner = ExperimentSeriesRunner(self.config_path, "test_series")
        
        # Mock out sys.exit to prevent actual exit
        with patch('sys.exit') as mock_exit:
            # Start the experiment in a thread so we can send signals
            thread = threading.Thread(target=runner.run_experiment_series)
            thread.daemon = True
            thread.start()
            
            # Let it run a bit
            time.sleep(0.3)
            
            # Send SIGINT to simulate Ctrl+C
            runner._handle_shutdown(signal.SIGINT, None)
            
            # Wait for completion
            thread.join(timeout=1.0)
            
            # Verify shutdown was requested
            self.assertTrue(runner.shutdown_requested)
            
            # Verify exit wasn't called (first SIGINT)
            mock_exit.assert_not_called()
            
            # Send second SIGINT to simulate force quit
            runner._handle_shutdown(signal.SIGINT, None)
            
            # Verify sys.exit was called (second SIGINT)
            mock_exit.assert_called_once()
    
    @patch('neuro_manipulation.experiment_series_runner.EmotionGameExperiment')
    @patch('neuro_manipulation.experiment_series_runner.get_repe_eng_config')
    @patch('neuro_manipulation.experiment_series_runner.get_game_config')
    @patch('neuro_manipulation.experiment_series_runner.GameNames')
    @patch('neuro_manipulation.experiment_series_runner.repe_pipeline_registry')
    def test_report_file_updates(self, mock_registry, mock_game_names, mock_game_config,
                               mock_repe_eng_config, mock_emotion_game_exp):
        """Test that report file is updated after each experiment."""
        # Setup mocks
        mock_game_instance = MagicMock()
        mock_game_names.from_string.return_value = mock_game_instance
        mock_game_instance.is_sequential.return_value = False
        
        mock_repe_eng_config.return_value = {}
        mock_game_config.return_value = {}
        
        # Create a file modification time tracker
        report_mod_times = []
        
        # Replace _save_report to track calls
        original_save_report = ExperimentReport._save_report
        
        def tracked_save_report(self_report):
            original_save_report(self_report)
            if self_report.report_file.exists():
                report_mod_times.append(os.path.getmtime(self_report.report_file))
        
        ExperimentReport._save_report = tracked_save_report
        
        # Setup mock experiments
        def create_mock_exp(*args, **kwargs):
            output_dir = os.path.join(self.temp_dir, f"{args[1]['experiment']['game']['name']}_{args[1]['experiment']['llm']['model_name']}_{int(time.time())}")
            kwargs['output_dir'] = output_dir
            kwargs['exp_config'] = args[1]
            return MockEmotionGameExperiment(*args, **kwargs)
        
        mock_emotion_game_exp.side_effect = create_mock_exp
        
        # Create and run the runner
        runner = ExperimentSeriesRunner(self.config_path, "test_series")
        runner.run_experiment_series()
        
        # Restore original _save_report
        ExperimentReport._save_report = original_save_report
        
        # Verify report file was updated multiple times
        # Initial save + experiment adds + status updates
        self.assertGreater(len(report_mod_times), 6)  # At least once per experiment
        
        # Verify timestamps are increasing (file is being updated)
        for i in range(1, len(report_mod_times)):
            self.assertGreaterEqual(report_mod_times[i], report_mod_times[i-1])

if __name__ == "__main__":
    unittest.main() 