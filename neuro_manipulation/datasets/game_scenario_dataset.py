import json
import os
import numpy as np

from torch.utils.data import Dataset

from games.game import GameScenario
from merge_data_samples import merge_data_samples

class GameScenarioDataset(Dataset):
    def __init__(self, game_config, prompt_wrapper, sample_num=None): 
        self.game_config = game_config
        data_path = game_config['data_path']
        
        if not os.path.exists(data_path):
            data_folder = game_config.get('data_folder')
            if data_folder:
                self.raw_data = merge_data_samples(data_folder, game_config['game_name'])
            else:
                raise ValueError(f'data_path: {data_path} does not exist, and data_folder is not provided')
        
        else:
            assert data_path.endswith('.json'), f'data_path: {data_path} should be a csv file'
            with open(data_path, 'r') as f:
                self.raw_data = json.load(f)
        
        assert len(self.raw_data) > 0, f'raw_data is empty, you are reading {data_path}'
        scenario_class: GameScenario = self.game_config['scenario_class']
        self.data: list[GameScenario] = []
        for item in self.raw_data:
            if 'payoff_matrix' not in item:
                item['payoff_matrix'] = self.game_config['payoff_matrix']
                
                # Universal field injection: inject ALL config fields that the schema expects
                for field_name in scenario_class.model_fields:
                    if field_name in self.game_config and field_name not in item:
                        item[field_name] = self.game_config[field_name]
            try:
                item = scenario_class(**item)
                self.data.append(item)
            except Exception as e:
                print(f'Skip item for error: {e}')
                continue
            
        if sample_num is not None:
            self.data = np.random.permutation(self.data)[:sample_num]
        
        self.prompt_wrapper = prompt_wrapper    
                
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item: GameScenario = self.data[idx]
        options = tuple(f'Option {i+1}. {opt}' for i, opt in enumerate(item.get_behavior_choices().get_choices()))
        event = str(item)
        return {
            "prompt": self.prompt_wrapper(event=event, options=options),
            "options": options,
            'behavior_choices': str(item.get_behavior_choices()),
            'scenario': item.get_scenario_info()['scenario'],
            'description': item.get_scenario_info()['description'],
        }
        
def collate_game_scenarios(batch):
    """
    Custom collate function for GameScenarioDataset that properly handles option tuples.
    Args:
        batch: List of dictionaries containing dataset items
    Returns:
        Collated batch with options properly grouped
    """
    return {
        'prompt': [item['prompt'] for item in batch],
        'options': [item['options'] for item in batch],
        'behavior_choices': [item['behavior_choices'] for item in batch],
        'scenario': [item['scenario'] for item in batch],
        'description': [item['description'] for item in batch],
    }

if __name__ == "__main__":
    from games.game_configs import get_game_config
    from constants import GameNames
    game_name = GameNames.PRISONERS_DILEMMA
    game_config = get_game_config(game_name)
    if game_name.is_sequential():
        game_config['previous_actions_length'] = 2
    
    def prompt_wrapper(event, options):
        return f"Scenario: {event}\nOptions: {options}"
   
    from transformers import AutoTokenizer
    from neuro_manipulation.prompt_formats import PromptFormat
    from neuro_manipulation.prompt_wrapper import GameReactPromptWrapper
    from games.game import GameDecision
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
    prompt_format = PromptFormat(tokenizer)
    prompt_wrapper = GameReactPromptWrapper(prompt_format, GameDecision)
    from functools import partial
    emo_dataset = GameScenarioDataset(game_config, 
                                    prompt_wrapper=partial(prompt_wrapper.__call__,
                                                            # emotion='angry',
                                                user_messages='You are Amy ,you are angry'), 
                                    sample_num=200)
    print(emo_dataset[0])
    
    from torch.utils.data import DataLoader
    data_loader = DataLoader(
        emo_dataset, 
        batch_size=2, 
        shuffle=True,
        collate_fn=collate_game_scenarios
    )
    for batch in data_loader:
        print(batch)
        break