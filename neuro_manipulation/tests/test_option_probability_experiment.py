import unittest
import sys
import os
import tempfile
import json
import shutil
from unittest.mock import Mock, patch, MagicMock, call
from pathlib import Path
import pandas as pd
import numpy as np
from vllm import LLM

# Add the project root to sys.path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from neuro_manipulation.experiments.option_probability_experiment import (
    OptionProbabilityExperiment,
    ExperimentCondition
)


class TestExperimentCondition(unittest.TestCase):
    """Test cases for ExperimentCondition dataclass."""
    
    def test_condition_creation(self):
        """Test creating ExperimentCondition instances."""
        condition = ExperimentCondition(
            emotion="angry",
            context="with_description",
            emotion_intensity=1.0,
            include_description=True
        )
        
        self.assertEqual(condition.emotion, "angry")
        self.assertEqual(condition.context, "with_description")
        self.assertEqual(condition.emotion_intensity, 1.0)
        self.assertTrue(condition.include_description)
        
    def test_condition_defaults(self):
        """Test default values in ExperimentCondition."""
        condition = ExperimentCondition(
            emotion="neutral",
            context="without_description"
        )
        
        self.assertEqual(condition.emotion_intensity, 0.0)
        self.assertTrue(condition.include_description)


class TestOptionProbabilityExperiment(unittest.TestCase):
    """Test cases for OptionProbabilityExperiment class."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create temporary directory for test outputs
        self.temp_dir = tempfile.mkdtemp()
        
        # Mock configurations
        self.repe_eng_config = {
            'model_name_or_path': 'test-model',
            'tensor_parallel_size': 1,
            'coeffs': [0.0, 1.0],
            'block_name': 'model.layers.{}.self_attn',
            'control_method': 'reading_vec'
        }
        
        self.exp_config = {
            'experiment': {
                'name': 'test_emotion_context',
                'emotions': ['neutral', 'angry'],
                'emotion_intensities': [0.0, 1.0],
                'output': {
                    'base_dir': self.temp_dir
                }
            }
        }
        
        self.game_config = {
            'game_name': 'prisoners_dilemma',
            'decision_class': Mock(),
            'data_path': '/mock/path.json'
        }
        
        # Create logs directory
        os.makedirs('logs', exist_ok=True)
        
    def tearDown(self):
        """Clean up test fixtures."""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
            
    @patch('neuro_manipulation.experiments.option_probability_experiment.GameReactPromptWrapper')
    @patch('neuro_manipulation.experiments.option_probability_experiment.CombinedVLLMHook')
    @patch('neuro_manipulation.experiments.option_probability_experiment.setup_model_and_tokenizer')
    def test_initialization(self, mock_setup, mock_hook, mock_prompt_wrapper):
        """Test experiment initialization."""
        mock_setup.return_value = (Mock(spec=LLM), Mock(), Mock())
        
        # Initialize experiment
        experiment = OptionProbabilityExperiment(
            repe_eng_config=self.repe_eng_config,
            exp_config=self.exp_config,
            game_config=self.game_config,
            batch_size=2,
            sample_num=10
        )
        
        # Check initialization
        self.assertEqual(experiment.batch_size, 2)
        self.assertEqual(experiment.sample_num, 10)
        self.assertEqual(experiment.emotions, ['neutral', 'angry'])
        self.assertTrue(os.path.exists(experiment.output_dir))
        
        # Check mocks were called
        mock_setup.assert_called_once()
        mock_hook.assert_called_once()
        mock_prompt_wrapper.assert_called_once()
        
    @patch('neuro_manipulation.experiments.option_probability_experiment.GameReactPromptWrapper')
    @patch('neuro_manipulation.experiments.option_probability_experiment.CombinedVLLMHook')
    @patch('neuro_manipulation.experiments.option_probability_experiment.setup_model_and_tokenizer')
    def test_initialization_non_vllm_error(self, mock_setup, mock_hook, mock_prompt_wrapper):
        """Test that initialization fails with non-vLLM model."""
        mock_setup.return_value = (Mock(), Mock(), Mock()) # A generic mock is not an instance of LLM
        
        with self.assertRaises(ValueError) as context:
            OptionProbabilityExperiment(
                repe_eng_config=self.repe_eng_config,
                exp_config=self.exp_config,
                game_config=self.game_config
            )
            
        self.assertIn("requires vLLM model", str(context.exception))
        
    @patch('neuro_manipulation.experiments.option_probability_experiment.GameReactPromptWrapper')
    @patch('neuro_manipulation.experiments.option_probability_experiment.CombinedVLLMHook')
    @patch('neuro_manipulation.experiments.option_probability_experiment.setup_model_and_tokenizer')
    def test_generate_experimental_conditions(self, mock_setup, mock_hook, mock_prompt_wrapper):
        """Test generation of experimental conditions."""
        mock_setup.return_value = (Mock(spec=LLM), Mock(), Mock())
        
        experiment = OptionProbabilityExperiment(
            repe_eng_config=self.repe_eng_config,
            exp_config=self.exp_config,
            game_config=self.game_config
        )
        
        conditions = experiment._generate_experimental_conditions()
        
        # Should have 4 conditions: 2 emotions × 2 context conditions
        self.assertEqual(len(conditions), 4)
        
        # Check all combinations are present
        expected_combinations = [
            ('neutral', 'with_description'),
            ('neutral', 'without_description'),
            ('angry', 'with_description'),
            ('angry', 'without_description')
        ]
        
        actual_combinations = [(c.emotion, c.context) for c in conditions]
        for expected in expected_combinations:
            self.assertIn(expected, actual_combinations)
            
        # Check emotion intensities
        for condition in conditions:
            if condition.emotion == 'neutral':
                self.assertEqual(condition.emotion_intensity, 0.0)
            else:
                self.assertEqual(condition.emotion_intensity, 1.0)
                
    @patch('neuro_manipulation.experiments.option_probability_experiment.ContextManipulationDataset')
    @patch('neuro_manipulation.experiments.option_probability_experiment.GameReactPromptWrapper')
    @patch('neuro_manipulation.experiments.option_probability_experiment.CombinedVLLMHook')
    @patch('neuro_manipulation.experiments.option_probability_experiment.setup_model_and_tokenizer')
    def test_create_dataset_for_condition(self, mock_setup, mock_hook, mock_prompt_wrapper, mock_dataset):
        """Test dataset creation for specific conditions."""
        mock_setup.return_value = (Mock(spec=LLM), Mock(), Mock())
        
        experiment = OptionProbabilityExperiment(
            repe_eng_config=self.repe_eng_config,
            exp_config=self.exp_config,
            game_config=self.game_config
        )
        
        # Test condition with description
        condition_with_desc = ExperimentCondition(
            emotion="angry",
            context="with_description",
            emotion_intensity=1.0,
            include_description=True
        )
        
        dataset = experiment._create_dataset_for_condition(condition_with_desc)
        
        # Check that ContextManipulationDataset was called with correct parameters
        mock_dataset.assert_called_once()
        call_args = mock_dataset.call_args
        self.assertEqual(call_args[1]['include_description'], True)
        self.assertEqual(call_args[1]['game_config'], self.game_config)
        
    @patch('neuro_manipulation.experiments.option_probability_experiment.GameReactPromptWrapper')
    @patch('neuro_manipulation.experiments.option_probability_experiment.CombinedVLLMHook')
    @patch('neuro_manipulation.experiments.option_probability_experiment.setup_model_and_tokenizer')
    def test_measure_option_probabilities(self, mock_setup, mock_hook, mock_prompt_wrapper):
        """Test option probability measurement."""
        mock_setup.return_value = (Mock(spec=LLM), Mock(), Mock())
        
        # Mock CombinedVLLMHook
        mock_hook_instance = Mock()
        mock_hook_instance.get_log_prob.return_value = [
            {
                'sequence': 'Cooperate',
                'log_prob': -0.5,
                'prob': 0.606,
                'perplexity': 1.649,
                'num_tokens': 1
            },
            {
                'sequence': 'Defect',
                'log_prob': -1.0,
                'prob': 0.368,
                'perplexity': 2.718,
                'num_tokens': 1
            }
        ]
        mock_hook.return_value = mock_hook_instance
        
        experiment = OptionProbabilityExperiment(
            repe_eng_config=self.repe_eng_config,
            exp_config=self.exp_config,
            game_config=self.game_config
        )
        
        # Create test batch
        test_batch = {
            'prompt': ['Test prompt'],
            'options': [('Option 1. Cooperate', 'Option 2. Defect')],
            'scenario': ['test_scenario'],
            'behavior_choices': ['Cooperate, Defect']
        }
        
        test_condition = ExperimentCondition(
            emotion="angry",
            context="with_description",
            emotion_intensity=1.0,
            include_description=True
        )
        
        results = experiment._measure_option_probabilities(test_batch, test_condition)
        
        # Check results structure
        self.assertEqual(len(results), 1)
        result = results[0]
        
        self.assertEqual(result['condition_emotion'], 'angry')
        self.assertEqual(result['condition_context'], 'with_description')
        self.assertIn('option_probabilities', result)
        self.assertIn('log_probabilities', result)
        
        # Check that probabilities were normalized
        probs = result['option_probabilities']
        total_prob = sum(probs.values())
        self.assertAlmostEqual(total_prob, 1.0, places=5)
        
    @patch('neuro_manipulation.experiments.option_probability_experiment.GameReactPromptWrapper')
    @patch('neuro_manipulation.experiments.option_probability_experiment.CombinedVLLMHook')
    @patch('neuro_manipulation.experiments.option_probability_experiment.setup_model_and_tokenizer')
    def test_save_results(self, mock_setup, mock_hook, mock_prompt_wrapper):
        """Test results saving functionality."""
        mock_setup.return_value = (Mock(spec=LLM), Mock(), Mock())
        
        experiment = OptionProbabilityExperiment(
            repe_eng_config=self.repe_eng_config,
            exp_config=self.exp_config,
            game_config=self.game_config
        )
        
        # Add mock results
        experiment.results = [
            {
                'condition_emotion': 'angry',
                'condition_context': 'with_description',
                'emotion_intensity': 1.0,
                'include_description': True,
                'scenario': 'test_scenario',
                'behavior_choices': 'Cooperate, Defect',
                'option_probabilities': {'Cooperate': 0.6, 'Defect': 0.4},
                'log_probabilities': {'Cooperate': -0.5, 'Defect': -0.9}
            }
        ]
        
        csv_file = experiment._save_results()
        
        # Check files were created
        self.assertTrue(os.path.exists(csv_file))
        json_file = csv_file.replace('.csv', '.json')
        self.assertTrue(os.path.exists(json_file))
        
        # Check CSV content
        df = pd.read_csv(csv_file)
        self.assertEqual(len(df), 2)  # Two options
        self.assertIn('condition_emotion', df.columns)
        self.assertIn('option_id', df.columns)
        self.assertIn('probability', df.columns)
        
    @patch('neuro_manipulation.experiments.option_probability_experiment.GameReactPromptWrapper')
    @patch('neuro_manipulation.experiments.option_probability_experiment.CombinedVLLMHook')
    @patch('neuro_manipulation.experiments.option_probability_experiment.setup_model_and_tokenizer')
    def test_statistical_analysis(self, mock_setup, mock_hook, mock_prompt_wrapper):
        """Test statistical analysis functionality."""
        mock_setup.return_value = (Mock(spec=LLM), Mock(), Mock())
        
        experiment = OptionProbabilityExperiment(
            repe_eng_config=self.repe_eng_config,
            exp_config=self.exp_config,
            game_config=self.game_config
        )
        
        # Create test CSV data
        test_data = {
            'condition_emotion': ['angry', 'angry', 'neutral', 'neutral'],
            'condition_context': ['with_description', 'without_description', 'with_description', 'without_description'],
            'option_id': [1, 1, 1, 1],
            'option_text': ['Cooperate', 'Cooperate', 'Cooperate', 'Cooperate'],
            'probability': [0.6, 0.7, 0.5, 0.4],
            'log_probability': [-0.5, -0.4, -0.7, -0.9]
        }
        
        df = pd.DataFrame(test_data)
        csv_file = os.path.join(experiment.output_dir, 'test_results.csv')
        df.to_csv(csv_file, index=False)
        
        # Run analysis
        experiment._run_statistical_analysis(csv_file)
        
        # Check analysis file was created
        analysis_file = os.path.join(experiment.output_dir, 'statistical_analysis.json')
        self.assertTrue(os.path.exists(analysis_file))
        
        # Check report file was created
        report_file = os.path.join(experiment.output_dir, 'experiment_summary_report.txt')
        self.assertTrue(os.path.exists(report_file))
        
        # Check analysis content
        with open(analysis_file, 'r') as f:
            analysis = json.load(f)
            
        self.assertIn('summary_statistics', analysis)
        self.assertIn('interaction_effects', analysis)
        self.assertIn('pairwise_comparisons', analysis)
        
    @patch('neuro_manipulation.experiments.option_probability_experiment.ContextManipulationDataset')
    @patch('neuro_manipulation.experiments.option_probability_experiment.DataLoader')
    @patch('neuro_manipulation.experiments.option_probability_experiment.GameReactPromptWrapper')
    @patch('neuro_manipulation.experiments.option_probability_experiment.CombinedVLLMHook')
    @patch('neuro_manipulation.experiments.option_probability_experiment.setup_model_and_tokenizer')
    def test_run_experiment_integration(self, mock_setup, mock_hook, mock_prompt_wrapper, mock_dataloader, mock_dataset):
        """Test full experiment run integration."""
        mock_setup.return_value = (Mock(spec=LLM), Mock(), Mock())
        
        # Mock CombinedVLLMHook
        mock_hook_instance = mock_hook.return_value
        mock_hook_instance.get_log_prob.return_value = [
            {
                'sequence': 'Cooperate',
                'log_prob': -0.5,
                'prob': 0.606,
                'perplexity': 1.649,
                'num_tokens': 1
            },
            {
                'sequence': 'Defect',
                'log_prob': -1.0,
                'prob': 0.368,
                'perplexity': 2.718,
                'num_tokens': 1
            }
        ]
        
        # Mock dataset and dataloader
        mock_dataset_instance = mock_dataset.return_value
        mock_dataset_instance.__len__.return_value = 2
        
        mock_dataloader.return_value = iter([
            {
                'prompt': ['Test prompt'],
                'options': [('Option 1. Cooperate', 'Option 2. Defect')],
                'scenario': ['test_scenario'],
                'behavior_choices': ['Cooperate, Defect']
            }
        ])
        
        experiment = OptionProbabilityExperiment(
            repe_eng_config=self.repe_eng_config,
            exp_config=self.exp_config,
            game_config=self.game_config,
            sample_num=2
        )
        
        # Run experiment
        results_file = experiment.run_experiment()
        
        # Check that results file was created and returned
        self.assertTrue(os.path.exists(results_file))
        self.assertTrue(results_file.endswith('.csv'))
        
        # Check that all 4 conditions were processed (2 emotions × 2 contexts)
        self.assertEqual(mock_dataset.call_count, 4)
        self.assertEqual(mock_dataloader.call_count, 4)


if __name__ == '__main__':
    # Create logs directory if it doesn't exist
    os.makedirs('logs', exist_ok=True)
    
    # Run tests
    unittest.main() 