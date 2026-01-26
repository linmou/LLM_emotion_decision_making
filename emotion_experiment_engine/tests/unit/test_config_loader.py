"""
Tests for YAML configuration loader.
"""
import pytest
import tempfile
import yaml
from pathlib import Path

from emotion_experiment_engine.config_loader import (
    EmotionMemoryConfigLoader, 
    load_emotion_memory_config
)
from emotion_experiment_engine.data_models import ExperimentConfig, BenchmarkConfig


class TestEmotionMemoryConfigLoader:
    """Test the YAML configuration loader"""
    
    def test_create_sample_config(self):
        """Test creating a sample configuration file"""
        with tempfile.NamedTemporaryFile(suffix='.yaml', delete=False) as tmp:
            config_path = Path(tmp.name)
        
        try:
            result_path = EmotionMemoryConfigLoader.create_sample_config(config_path)
            
            assert result_path == config_path
            assert config_path.exists()
            
            # Verify it's valid YAML
            with open(config_path, 'r') as f:
                config_data = yaml.safe_load(f)
            
            assert 'experiment' in config_data
            assert 'model_path' in config_data['experiment']
            assert 'emotions' in config_data['experiment']
            assert 'benchmark' in config_data['experiment']
            
        finally:
            config_path.unlink(missing_ok=True)
    
    def test_load_basic_config(self):
        """Test loading a basic configuration"""
        config_data = {
            'experiment': {
                'model_path': '/test/model/path',
                'emotions': ['anger', 'happiness'],
                'intensities': [1.0, 1.5],
                'benchmark': {
                    'name': 'infinitebench',
                    'data_path': '/test/data.jsonl',
                    'task_type': 'passkey',
                    'evaluation_method': 'get_score_one_passkey',
                    'sample_limit': 50
                },
                'batch_size': 4,
                'output': {
                    'base_dir': 'results/test'
                }
            }
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as tmp:
            yaml.dump(config_data, tmp, default_flow_style=False)
            config_path = Path(tmp.name)
        
        try:
            exp_config = EmotionMemoryConfigLoader.load_from_yaml(config_path)
            
            assert isinstance(exp_config, ExperimentConfig)
            assert exp_config.model_path == '/test/model/path'
            assert exp_config.emotions == ['anger', 'happiness']
            assert exp_config.intensities == [1.0, 1.5]
            assert exp_config.batch_size == 4
            assert exp_config.output_dir == 'results/test'
            
            # Check benchmark config
            assert isinstance(exp_config.benchmark, BenchmarkConfig)
            assert exp_config.benchmark.name == 'infinitebench'
            assert exp_config.benchmark.data_path == Path('/test/data.jsonl')
            assert exp_config.benchmark.task_type == 'passkey'
            assert exp_config.benchmark.sample_limit == 50
            assert exp_config.benchmark.evaluation_method == 'get_score_one_passkey'
            
        finally:
            config_path.unlink(missing_ok=True)
    
    def test_load_config_with_generation_config(self):
        """Test loading configuration with generation parameters"""
        config_data = {
            'experiment': {
                'model_path': '/test/model',
                'emotions': ['anger'],
                'intensities': [1.0],
                'benchmark': {
                    'name': 'infinitebench',
                    'data_path': '/test/data.jsonl',
                    'task_type': 'passkey'
                },
                'generation_config': {
                    'temperature': 0.5,
                    'max_new_tokens': 100,
                    'do_sample': True,
                    'top_p': 0.9
                }
            }
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as tmp:
            yaml.dump(config_data, tmp, default_flow_style=False)
            config_path = Path(tmp.name)
        
        try:
            exp_config = EmotionMemoryConfigLoader.load_from_yaml(config_path)
            
            assert exp_config.generation_config is not None
            assert exp_config.generation_config['temperature'] == 0.5
            assert exp_config.generation_config['max_new_tokens'] == 100
            assert exp_config.generation_config['do_sample'] is True
            assert exp_config.generation_config['top_p'] == 0.9
            
        finally:
            config_path.unlink(missing_ok=True)
    
    def test_load_config_legacy_llm_section(self):
        """Test loading configuration with legacy llm section"""
        config_data = {
            'experiment': {
                'model_path': '/test/model',
                'emotions': ['anger'],
                'intensities': [1.0],
                'benchmark': {
                    'name': 'infinitebench',
                    'data_path': '/test/data.jsonl',
                    'task_type': 'passkey'
                },
                'llm': {
                    'generation_config': {
                        'temperature': 0.3,
                        'max_new_tokens': 150
                    }
                }
            }
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as tmp:
            yaml.dump(config_data, tmp, default_flow_style=False)
            config_path = Path(tmp.name)
        
        try:
            exp_config = EmotionMemoryConfigLoader.load_from_yaml(config_path)
            
            assert exp_config.generation_config is not None
            assert exp_config.generation_config['temperature'] == 0.3
            assert exp_config.generation_config['max_new_tokens'] == 150
            
        finally:
            config_path.unlink(missing_ok=True)
    
    def test_convenience_function(self):
        """Test the convenience function"""
        config_data = {
            'experiment': {
                'model_path': '/test/model',
                'emotions': ['anger'],
                'intensities': [1.0],
                'benchmark': {
                    'name': 'infinitebench',
                    'data_path': '/test/data.jsonl',
                    'task_type': 'passkey'
                }
            }
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as tmp:
            yaml.dump(config_data, tmp, default_flow_style=False)
            config_path = Path(tmp.name)
        
        try:
            exp_config = load_emotion_memory_config(config_path)
            
            assert isinstance(exp_config, ExperimentConfig)
            assert exp_config.model_path == '/test/model'
            
        finally:
            config_path.unlink(missing_ok=True)
    
    def test_file_not_found_error(self):
        """Test error handling for missing files"""
        with pytest.raises(FileNotFoundError):
            EmotionMemoryConfigLoader.load_from_yaml("/nonexistent/config.yaml")
    
    def test_minimal_config(self):
        """Test loading minimal configuration with defaults"""
        config_data = {
            'experiment': {
                'model_path': '/test/model',
                'benchmark': {
                    'data_path': '/test/data.jsonl'
                }
            }
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as tmp:
            yaml.dump(config_data, tmp, default_flow_style=False)
            config_path = Path(tmp.name)
        
        try:
            exp_config = EmotionMemoryConfigLoader.load_from_yaml(config_path)
            
            # Check defaults are applied
            assert exp_config.emotions == ['anger', 'happiness']  # Default
            assert exp_config.intensities == [1.0]  # Default
            assert exp_config.benchmark.name == 'infinitebench'  # Default
            assert exp_config.benchmark.task_type == 'passkey'  # Default
            assert exp_config.batch_size == 4  # Default
            
        finally:
            config_path.unlink(missing_ok=True)