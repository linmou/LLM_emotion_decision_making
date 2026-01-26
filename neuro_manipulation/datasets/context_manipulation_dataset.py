import json
import os
import numpy as np
from typing import Optional, Dict, Any

from torch.utils.data import Dataset

from games.game import GameScenario
from merge_data_samples import merge_data_samples
from .game_scenario_dataset import GameScenarioDataset


class ContextManipulationDataset(GameScenarioDataset):
    """
    Dataset for context manipulation experiments that conditionally includes scenario descriptions.
    
    This dataset extends GameScenarioDataset to support 2x2 factorial design experiments
    where one factor is emotion intervention and the other is context description presence.
    
    Args:
        game_config: Configuration dictionary for the game
        prompt_wrapper: Function to wrap prompts with appropriate formatting
        include_description: Whether to include scenario descriptions in the prompt
        sample_num: Number of samples to use (optional)
        
    The dataset modifies the __getitem__ method to conditionally include scenario
    descriptions based on the include_description parameter, enabling controlled
    context manipulation studies.
    """
    
    def __init__(self, game_config, prompt_wrapper, include_description: bool = True, sample_num: Optional[int] = None):
        # Initialize parent class
        super().__init__(game_config, prompt_wrapper, sample_num)
        self.include_description = include_description
        
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Get a single item from the dataset with conditional description inclusion.
        
        Args:
            idx: Index of the item to retrieve
            
        Returns:
            Dictionary containing:
            - prompt: Formatted prompt (with or without description)
            - options: Tuple of available behavioral choices
            - behavior_choices: String representation of choices
            - scenario: Scenario information
            - description: Description (may be empty if include_description=False)
            - has_description: Boolean indicating if description was included
        """
        item: GameScenario = self.data[idx]
        options = tuple(f'Option {i+1}. {opt}' for i, opt in enumerate(item.get_behavior_choices().get_choices()))
        
        # Get scenario info
        scenario_info = item.get_scenario_info()
        event = str(item)
        
        # Conditionally include description based on experimental condition
        if self.include_description:
            # Include full scenario description
            description = scenario_info['description']
            # Create event string that includes the description
            event_with_context = f"{event}\n\nContext: {description}"
        else:
            # Exclude description - use only basic scenario information
            description = ""
            event_with_context = event
        
        return {
            "prompt": self.prompt_wrapper(event=event_with_context, options=options),
            "options": options,
            'behavior_choices': str(item.get_behavior_choices()),
            'scenario': scenario_info['scenario'],
            'description': description,
            'has_description': self.include_description,
            'condition': 'with_description' if self.include_description else 'without_description'
        }


def collate_context_manipulation_scenarios(batch):
    """
    Custom collate function for ContextManipulationDataset that properly handles 
    option tuples and context manipulation features.
    
    Args:
        batch: List of dictionaries containing dataset items
        
    Returns:
        Collated batch with all features properly grouped including context conditions
    """
    return {
        'prompt': [item['prompt'] for item in batch],
        'options': [item['options'] for item in batch],
        'behavior_choices': [item['behavior_choices'] for item in batch],
        'scenario': [item['scenario'] for item in batch],
        'description': [item['description'] for item in batch],
        'has_description': [item['has_description'] for item in batch],
        'condition': [item['condition'] for item in batch],
    }


if __name__ == "__main__":
    # Test the ContextManipulationDataset
    from games.game_configs import get_game_config
    from constants import GameNames
    from transformers import AutoTokenizer
    from neuro_manipulation.prompt_formats import PromptFormat
    from neuro_manipulation.prompt_wrapper import GameReactPromptWrapper
    from games.game import GameDecision
    from functools import partial
    from torch.utils.data import DataLoader
    
    # Setup test configuration
    game_name = GameNames.PRISONERS_DILEMMA
    game_config = get_game_config(game_name)
    if game_name.is_sequential():
        game_config['previous_actions_length'] = 2
    
    # Setup prompt wrapper
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
    prompt_format = PromptFormat(tokenizer)
    prompt_wrapper = GameReactPromptWrapper(prompt_format, GameDecision)
    
    # Test with description
    dataset_with_desc = ContextManipulationDataset(
        game_config, 
        prompt_wrapper=partial(prompt_wrapper.__call__, user_messages='You are Amy'), 
        include_description=True,
        sample_num=10
    )
    
    # Test without description
    dataset_without_desc = ContextManipulationDataset(
        game_config, 
        prompt_wrapper=partial(prompt_wrapper.__call__, user_messages='You are Amy'), 
        include_description=False,
        sample_num=10
    )
    
    print("=== WITH DESCRIPTION ===")
    print(f"Sample item: {dataset_with_desc[0]}")
    print(f"Has description: {dataset_with_desc[0]['has_description']}")
    print(f"Condition: {dataset_with_desc[0]['condition']}")
    
    print("\n=== WITHOUT DESCRIPTION ===")
    print(f"Sample item: {dataset_without_desc[0]}")
    print(f"Has description: {dataset_without_desc[0]['has_description']}")
    print(f"Condition: {dataset_without_desc[0]['condition']}")
    
    # Test DataLoader
    data_loader = DataLoader(
        dataset_with_desc, 
        batch_size=2, 
        shuffle=True,
        collate_fn=collate_context_manipulation_scenarios
    )
    
    print("\n=== DATALOADER TEST ===")
    for batch in data_loader:
        print(f"Batch keys: {batch.keys()}")
        print(f"Conditions: {batch['condition']}")
        break 