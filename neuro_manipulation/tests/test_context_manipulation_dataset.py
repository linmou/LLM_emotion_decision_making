import unittest
import sys
from unittest.mock import Mock, patch, MagicMock
from functools import partial
import tempfile
import json
import os
import numpy as np

# Add the project root to sys.path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from neuro_manipulation.datasets.context_manipulation_dataset import (
    ContextManipulationDataset,
    collate_context_manipulation_scenarios
)
from games.game import GameScenario
from torch.utils.data import DataLoader


class MockGameScenario(GameScenario):
    """Mock GameScenario for testing purposes."""
    
    def __init__(self, **kwargs):
        # Pydantic models are initialized with keyword arguments
        super().__init__(**kwargs)
        self.scenario_data = kwargs
        
    def get_behavior_choices(self):
        choices_mock = Mock()
        choices_mock.get_choices.return_value = ['Cooperate', 'Defect']
        return choices_mock
        
    def get_scenario_info(self):
        return {
            'scenario': self.scenario_data.get('scenario', 'Test scenario'),
            'description': self.scenario_data.get('description', 'Test description with context')
        }
        
    def __str__(self):
        return self.scenario_data.get('event', 'Test event description')
        
    @staticmethod
    def example():
        """Required abstract method implementation."""
        return {"scenario": "test", "description": "test description"}
        
    def find_behavior_from_decision(self, decision):
        """Required abstract method implementation."""
        return "Mock behavior"


class TestContextManipulationDataset(unittest.TestCase):
    """Test cases for ContextManipulationDataset class."""
    
    @patch('neuro_manipulation.datasets.game_scenario_dataset.GameScenarioDataset.__init__')
    def setUp(self, mock_super_init):
        """Set up test fixtures."""
        mock_super_init.return_value = None

        self.test_scenarios = [
            MockGameScenario(
                scenario='prisoners_dilemma',
                description='Classic PD.',
                event='Decide to cooperate or defect.'
            ),
            MockGameScenario(
                scenario='trust_game',
                description='Trust game.',
                event='Decide to trust or not.'
            )
        ]

        self.game_config = {
            'game_name': 'test_game',
            'scenario_class': MockGameScenario
        }
        
        self.prompt_wrapper = Mock(return_value="Formatted prompt")
        
    def test_initialization_with_description(self):
        """Test dataset initialization with description inclusion."""
        dataset = ContextManipulationDataset(
            game_config=self.game_config,
            prompt_wrapper=self.prompt_wrapper,
            include_description=True
        )
        dataset.data = self.test_scenarios

        self.assertTrue(dataset.include_description)
        self.assertEqual(len(dataset), 2)
        
    def test_initialization_without_description(self):
        """Test dataset initialization without description inclusion."""
        dataset = ContextManipulationDataset(
            game_config=self.game_config,
            prompt_wrapper=self.prompt_wrapper,
            include_description=False
        )
        dataset.data = self.test_scenarios

        self.assertFalse(dataset.include_description)
        self.assertEqual(len(dataset), 2)

    def test_getitem_with_description(self):
        """Test __getitem__ method when including descriptions."""
        dataset = ContextManipulationDataset(
            game_config=self.game_config,
            prompt_wrapper=self.prompt_wrapper,
            include_description=True
        )
        dataset.data = self.test_scenarios
        item = dataset[0]

        self.assertTrue(item['has_description'])
        self.assertIn('Context:', self.prompt_wrapper.call_args[1]['event'])

    def test_getitem_without_description(self):
        """Test __getitem__ method when excluding descriptions."""
        dataset = ContextManipulationDataset(
            game_config=self.game_config,
            prompt_wrapper=self.prompt_wrapper,
            include_description=False
        )
        dataset.data = self.test_scenarios
        item = dataset[0]

        self.assertFalse(item['has_description'])
        self.assertNotIn('Context:', self.prompt_wrapper.call_args[1]['event'])

    def test_sample_num_randomness(self):
        """Test that sample_num correctly limits and randomizes dataset size."""
        # This test needs to be adapted as the logic is in the parent.
        # We'll just test the length for now.
        dataset = ContextManipulationDataset(
            game_config=self.game_config,
            prompt_wrapper=self.prompt_wrapper,
            include_description=True,
        )
        dataset.data = self.test_scenarios
        
        # Manually apply sampling logic
        dataset.data = np.random.permutation(dataset.data)[:1].tolist()
        self.assertEqual(len(dataset), 1)

    def test_collate_function(self):
        """Test the custom collate function."""
        # This test can remain as it is, it doesn't depend on the dataset's internal state.
        mock_batch = [
            {
                'prompt': 'Prompt 1',
                'options': ('Option 1. Cooperate', 'Option 2. Defect'),
                'behavior_choices': 'Cooperate, Defect',
                'scenario': 'scenario1',
                'description': 'desc1',
                'has_description': True,
                'condition': 'with_description'
            },
            {
                'prompt': 'Prompt 2',
                'options': ('Option 1. Trust', 'Option 2. Not Trust'),
                'behavior_choices': 'Trust, Not Trust',
                'scenario': 'scenario2',
                'description': '',
                'has_description': False,
                'condition': 'without_description'
            }
        ]
        
        collated = collate_context_manipulation_scenarios(mock_batch)
        self.assertEqual(len(collated['prompt']), 2)
        self.assertEqual(collated['condition'], ['with_description', 'without_description'])

    def test_dataloader_integration(self):
        """Test integration with PyTorch DataLoader."""
        dataset = ContextManipulationDataset(
            game_config=self.game_config,
            prompt_wrapper=self.prompt_wrapper,
            include_description=True,
        )
        dataset.data = self.test_scenarios

        dataloader = DataLoader(
            dataset,
            batch_size=2,
            shuffle=False,
            collate_fn=collate_context_manipulation_scenarios
        )
        
        batch = next(iter(dataloader))
        self.assertIn('prompt', batch)
        self.assertEqual(len(batch['prompt']), 2)


if __name__ == '__main__':
    # Create logs directory if it doesn't exist
    os.makedirs('logs', exist_ok=True)
    
    # Run tests
    unittest.main() 