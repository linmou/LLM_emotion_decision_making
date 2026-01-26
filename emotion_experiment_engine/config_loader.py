"""
YAML configuration loader for emotion memory experiments.
Allows users to define experiment configurations in YAML format.

DEPRECATED: This module is deprecated and only used for tests.
Production code uses direct yaml.safe_load() in emotion_experiment_series_runner.py.
"""
import warnings
import yaml
from pathlib import Path
from typing import Dict, Any, List, Union
from .data_models import ExperimentConfig, BenchmarkConfig


class EmotionMemoryConfigLoader:
    """Loads emotion memory experiment configurations from YAML files.
    
    DEPRECATED: This class is deprecated and only used for tests.
    Production code uses direct yaml.safe_load() in emotion_experiment_series_runner.py.
    """
    
    @staticmethod
    def load_from_yaml(config_path: Union[str, Path]) -> ExperimentConfig:
        """Load experiment configuration from YAML file.
        
        DEPRECATED: This method is deprecated and only used for tests.
        """
        warnings.warn(
            "EmotionMemoryConfigLoader.load_from_yaml is deprecated and only used for tests. "
            "Production code uses direct yaml.safe_load() in emotion_experiment_series_runner.py.",
            DeprecationWarning,
            stacklevel=2
        )
        config_path = Path(config_path)
        
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        
        with open(config_path, 'r') as f:
            yaml_config = yaml.safe_load(f)
        
        return EmotionMemoryConfigLoader._parse_config(yaml_config)
    
    @staticmethod
    def _parse_config(yaml_config: Dict[str, Any]) -> ExperimentConfig:
        """Parse YAML config dict into ExperimentConfig object."""
        config = yaml_config.get('experiment', {})
        
        # Parse benchmark configuration
        benchmark_section = config.get('benchmark', {})
        benchmark_config = BenchmarkConfig(
            name=benchmark_section.get('name', 'infinitebench'),
            data_path=Path(benchmark_section.get('data_path', '')),
            task_type=benchmark_section.get('task_type', 'passkey'),
            sample_limit=benchmark_section.get('sample_limit', None),
            augmentation_config=benchmark_section.get('augmentation_config', None),
            enable_auto_truncation=benchmark_section.get('enable_auto_truncation', True),
            truncation_strategy=benchmark_section.get('truncation_strategy', 'right'),
            preserve_ratio=benchmark_section.get('preserve_ratio', 0.8),
            llm_eval_config=benchmark_section.get('llm_eval_config', None)
        )
        
        # Parse model and emotions
        model_path = config.get('model_path', config.get('model', ''))
        emotions = config.get('emotions', ['anger', 'happiness'])
        intensities = config.get('intensities', config.get('intensity', [1.0]))
        
        # Parse generation config
        generation_config = config.get('generation_config', {})
        if not generation_config:
            # Check for llm section (legacy support)
            llm_config = config.get('llm', {})
            generation_config = llm_config.get('generation_config', {})
        
        # Parse output configuration
        output_section = config.get('output', {})
        output_dir = output_section.get('base_dir', output_section.get('dir', 'results/emotion_memory'))
        
        # Parse experimental parameters
        batch_size = config.get('batch_size', config.get('data', {}).get('batch_size', 4))
        
        return ExperimentConfig(
            model_path=model_path,
            emotions=emotions,
            intensities=intensities,
            benchmark=benchmark_config,
            output_dir=output_dir,
            batch_size=batch_size,
            generation_config=generation_config if generation_config else None,
            loading_config=None,  # Add missing required fields
            repe_eng_config=None,
            max_evaluation_workers=1,
            pipeline_queue_size=10,
            defer_evaluation=False,
        )
    
    @staticmethod
    def create_sample_config(output_path: Union[str, Path]) -> Path:
        """Create a sample YAML configuration file.
        
        DEPRECATED: This method is deprecated and only used for tests.
        """
        warnings.warn(
            "EmotionMemoryConfigLoader.create_sample_config is deprecated and only used for tests.",
            DeprecationWarning,
            stacklevel=2
        )
        output_path = Path(output_path)
        
        sample_config = {
            'experiment': {
                'name': 'Emotion_Memory_Benchmark_Experiment',
                'model_path': '/data/home/jjl7137/huggingface_models/Qwen/Qwen2.5-0.5B-Instruct',
                
                'emotions': [
                    'anger',
                    'happiness', 
                    'sadness',
                    'fear',
                    'disgust',
                    'surprise'
                ],
                
                'intensities': [0.5, 1.0, 1.5],
                
                'benchmark': {
                    'name': 'infinitebench',
                    'data_path': '/data/home/jjl7137/memory_benchmarks/InfiniteBench/data/passkey.jsonl',
                    'task_type': 'passkey',
                    'evaluation_method': 'get_score_one_passkey',
                    'sample_limit': 100,
                    'metadata': {
                        'description': 'Passkey retrieval from long context',
                        'context_length': '100k+'
                    }
                },
                
                'generation_config': {
                    'temperature': 0.1,
                    'max_new_tokens': 50,
                    'do_sample': False,
                    'top_p': 0.9
                },
                
                'batch_size': 4,
                
                'output': {
                    'base_dir': 'results/emotion_experiment_engine',
                    'save_plots': True,
                    'plot_format': 'png'
                }
            }
        }
        
        with open(output_path, 'w') as f:
            yaml.dump(sample_config, f, default_flow_style=False, indent=2)
        
        return output_path


def load_emotion_memory_config(config_path: Union[str, Path]) -> ExperimentConfig:
    """Convenience function to load emotion memory experiment config from YAML.
    
    DEPRECATED: This function is deprecated and only used for tests.
    Production code uses direct yaml.safe_load() in emotion_experiment_series_runner.py.
    """
    warnings.warn(
        "load_emotion_memory_config is deprecated and only used for tests. "
        "Production code uses direct yaml.safe_load() in emotion_experiment_series_runner.py.",
        DeprecationWarning,
        stacklevel=2
    )
    return EmotionMemoryConfigLoader.load_from_yaml(config_path)
