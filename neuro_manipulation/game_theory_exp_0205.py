'''
This script is used to generate the reactions of different LLMs on the synthesis data.
'''
import time
import argparse
from constants import GameNames
from neuro_manipulation.repe import repe_pipeline_registry
from neuro_manipulation.configs.experiment_config import get_exp_config, get_repe_eng_config, get_model_config
from neuro_manipulation.experiments.emotion_game_experiment import EmotionGameExperiment
from games.game_configs import get_game_config

def setup_experiment(config_path='config/escalGame_repEng_experiment_config.yaml'):
    repe_pipeline_registry()
    
    exp_config = get_exp_config(config_path)
    game_name = GameNames.from_string(exp_config['experiment']['game']['name'])
    model_name = exp_config['experiment']['llm']['model_name'] #  "meta-llama/Meta-Llama-3.1-8B-Instruct"
    repe_eng_config = get_repe_eng_config(model_name, yaml_config=exp_config)
    game_config = get_game_config(game_name)
    if game_name.is_sequential():
        game_config['previous_actions_length'] = exp_config['experiment']['game']['previous_actions_length']
    
    # The EmotionGameExperiment class now handles batch size optimization internally
    experiment = EmotionGameExperiment(
        repe_eng_config, 
        exp_config, 
        game_config,
        repeat=exp_config['experiment']['repeat'],
        batch_size=exp_config['experiment'].get('batch_size', 300)
    )
    
    return experiment

def main():
    parser = argparse.ArgumentParser(description='Run an emotion game experiment')
    parser.add_argument('--config', type=str, default='config/priDeli_repEng_experiment_config.yaml',
                        help='Path to experiment config file')
    parser.add_argument('--series', action='store_true',
                        help='Run a series of experiments (uses multiple games and models)')
    parser.add_argument('--series-name', type=str, default=None,
                        help='Custom name for the experiment series')
    parser.add_argument('--resume', action='store_true',
                        help='Resume interrupted experiment series')
    
    args = parser.parse_args()
    
    time_start = time.time()
    
    if args.series:
        # Run a series of experiments
        from neuro_manipulation.experiment_series_runner import ExperimentSeriesRunner
        
        print(f"Running experiment series with config: {args.config}")
        runner = ExperimentSeriesRunner(
            config_path=args.config,
            series_name=args.series_name,
            resume=args.resume
        )
        runner.run_experiment_series()
    else:
        # Run a single experiment
        experiment = setup_experiment(args.config)
        
        if experiment.exp_config['experiment'].get('run_sanity_check', False):
            experiment.run_sanity_check()
        else:
            experiment.run_experiment()
        
    time_end = time.time()
    print(f"Time taken: {time_end - time_start} seconds")

def run_sanity_check():
    config_path = 'config/escalGame_repEng_experiment_config.yaml'
    
    experiment = setup_experiment(config_path)
    
    time_start = time.time()
    experiment.run_sanity_check()
    time_end = time.time()
    print(f"Time taken: {time_end - time_start} seconds")

if __name__ == "__main__":
    main()
