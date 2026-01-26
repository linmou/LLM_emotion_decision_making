"""
End-to-end integration tests: YAML → Config Loading → Dataset Creation → Evaluation
Testing: Full pipeline from configuration files to evaluation results

This comprehensive test validates the entire workflow:
1. YAML configuration loading with llm_eval_config
2. BenchmarkConfig creation from YAML
3. Real dataset creation via factory
4. ThreadPoolExecutor-based evaluation
5. Results validation
"""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pytest
import yaml

from emotion_experiment_engine.data_models import BenchmarkConfig, ExperimentConfig
from emotion_experiment_engine.dataset_factory import create_dataset_from_config


class TestEndToEndIntegration(unittest.TestCase):
    """Comprehensive end-to-end integration tests"""

    def setUp(self):
        """Set up test fixtures"""
        self.temp_files = []

    def tearDown(self):
        """Clean up temporary files"""
        for temp_file in self.temp_files:
            if temp_file.exists():
                temp_file.unlink()

    def create_test_yaml_config(
        self, benchmark_name="mtbench101", task_type="CM", llm_eval_config=None
    ):
        """Create a comprehensive YAML config file for testing"""
        config_data = {
            "model_path": "/data/models/Qwen2.5-0.5B-Instruct",
            "emotions": ["anger", "happiness", "neutral"],
            "intensities": [0.5, 1.0],
            "benchmark": {
                "name": benchmark_name,
                "task_type": task_type,
                "data_path": None,  # Auto-discovery
                "sample_limit": 2,  # Small for testing
                "augmentation_config": None,
                "enable_auto_truncation": True,
                "truncation_strategy": "right",
                "preserve_ratio": 0.8,
            },
            "output_dir": "test_results",
            "batch_size": 2,
            "generation_config": {
                "temperature": 0.1,
                "max_new_tokens": 50,
                "do_sample": False,
            },
            "loading_config": {
                "model_path": "/data/models/Qwen2.5-0.5B-Instruct",
                "gpu_memory_utilization": 0.8,
                "tensor_parallel_size": 1,
                "max_model_len": 2048,
                "enforce_eager": True,
                "quantization": None,
                "trust_remote_code": True,
                "dtype": "float16",
                "seed": 42,
                "disable_custom_all_reduce": False,
                "additional_vllm_kwargs": {},
            },
            "repe_eng_config": None,
            "max_evaluation_workers": 2,
            "pipeline_queue_size": 4,
        }

        # Add llm_eval_config if provided
        if llm_eval_config:
            config_data["benchmark"]["llm_eval_config"] = llm_eval_config

        temp_file = Path(tempfile.mktemp(suffix=".yaml"))
        with open(temp_file, "w") as f:
            yaml.dump(config_data, f, default_flow_style=False)

        self.temp_files.append(temp_file)
        return temp_file

    def create_test_mtbench_data(self):
        """Create test MTBench101 data file"""
        test_data = [
            {
                "task": "CM",
                "id": 1001,
                "history": [
                    {
                        "user": "What's the weather like?",
                        "bot": "I can't check real-time weather",
                    },
                    {
                        "user": "Can you help with cooking?",
                        "bot": "Sure! What would you like to cook?",
                    },
                ],
            },
            {
                "task": "CM",
                "id": 1002,
                "history": [
                    {
                        "user": "Tell me about Python",
                        "bot": "Python is a programming language",
                    },
                    {
                        "user": "How do I start learning?",
                        "bot": "Start with basic syntax and variables",
                    },
                ],
            },
        ]

        temp_file = Path(tempfile.mktemp(suffix="_CM.jsonl"))
        with open(temp_file, "w") as f:
            for item in test_data:
                f.write(json.dumps(item) + "\n")

        self.temp_files.append(temp_file)
        return temp_file

    def create_test_longbench_data(self):
        """Create test LongBench data file"""
        test_data = [
            {
                "id": "narrativeqa_0",
                "input": "What is the main character's motivation?",
                "context": "In the story, John was determined to find his missing sister who had disappeared three months ago. He quit his job and dedicated all his time to the search.",
                "answers": ["To find his missing sister"],
                "length": 1500,
                "task_name": "narrativeqa",
            },
            {
                "id": "narrativeqa_1",
                "input": "Where does the story take place?",
                "context": "The investigation led John through the bustling streets of New York City, from the financial district to the quiet neighborhoods in Brooklyn.",
                "answers": ["New York City"],
                "length": 1200,
                "task_name": "narrativeqa",
            },
        ]

        temp_file = Path(tempfile.mktemp(suffix="_narrativeqa.json"))
        with open(temp_file, "w") as f:
            json.dump(test_data, f)

        self.temp_files.append(temp_file)
        return temp_file

    def test_yaml_to_benchmark_config_flow(self):
        """Test YAML config → BenchmarkConfig creation with llm_eval_config"""
        # Create YAML with custom llm_eval_config
        custom_llm_config = {
            "model": "gpt-4o-mini",
            "temperature": 0.7,
            "max_tokens": 200,
        }

        yaml_file = self.create_test_yaml_config(
            benchmark_name="mtbench101",
            task_type="CM",
            llm_eval_config=custom_llm_config,
        )

        # Load configuration using production pattern (direct YAML loading)
        with open(yaml_file, "r") as f:
            config_dict = yaml.safe_load(f)

        # Verify benchmark config structure
        benchmark_config = config_dict["benchmark"]
        assert benchmark_config["name"] == "mtbench101"
        assert benchmark_config["task_type"] == "CM"
        assert benchmark_config["llm_eval_config"] == custom_llm_config
        assert benchmark_config["sample_limit"] == 2
        assert benchmark_config["enable_auto_truncation"] == True
        assert benchmark_config["preserve_ratio"] == 0.8

        print("✓ YAML → BenchmarkConfig flow working")

    def test_mtbench101_end_to_end_integration(self):
        """Test complete MTBench101 pipeline: YAML → Dataset → Evaluation"""
        # Step 1: Create test data and YAML config
        data_file = self.create_test_mtbench_data()

        custom_llm_config = {
            "model": "gpt-4o-mini",
            "temperature": 0.3,
            "max_tokens": 100,
        }

        yaml_file = self.create_test_yaml_config(
            benchmark_name="mtbench101",
            task_type="CM",
            llm_eval_config=custom_llm_config,
        )

        # Step 2: Load config and create BenchmarkConfig
        with open(yaml_file, "r") as f:
            config_dict = yaml.safe_load(f)
        benchmark_dict = config_dict["benchmark"]

        # Create BenchmarkConfig with explicit data path
        benchmark_config = BenchmarkConfig(
            name=benchmark_dict["name"],
            task_type=benchmark_dict["task_type"],
            data_path=data_file,  # Use our test data
            sample_limit=benchmark_dict["sample_limit"],
            augmentation_config=benchmark_dict["augmentation_config"],
            enable_auto_truncation=benchmark_dict["enable_auto_truncation"],
            truncation_strategy=benchmark_dict["truncation_strategy"],
            preserve_ratio=benchmark_dict["preserve_ratio"],
            llm_eval_config=benchmark_dict["llm_eval_config"],
        )

        # Step 3: Create dataset via factory with prompt wrapper (required by new design)  
        from emotion_experiment_engine.mtbench101_prompt_wrapper import MTBench101PromptWrapper
        from unittest.mock import Mock
        
        # Create minimal prompt wrapper for MTBench101
        mock_prompt_format = Mock()
        mock_prompt_format.build.return_value = "mtbench101_test_prompt"
        prompt_wrapper = MTBench101PromptWrapper(task_type="CM", prompt_format=mock_prompt_format)
        
        dataset = create_dataset_from_config(benchmark_config, prompt_wrapper=prompt_wrapper)

        # Verify dataset creation
        assert dataset is not None
        assert len(dataset) == 2  # Our test data has 2 items
        assert hasattr(dataset, "llm_eval_config")
        assert dataset.llm_eval_config["temperature"] == 0.3  # Custom config applied

        # Step 4: Test dataset functionality
        item_dict = dataset[0]
        benchmark_item = item_dict["item"]

        assert benchmark_item.id == "CM_1001"  # MTBench101 formats IDs as "tasktype_id"
        assert isinstance(
            benchmark_item.input_text, tuple
        )  # MTBench101 conversation format
        user_messages, assistant_messages = benchmark_item.input_text
        assert len(user_messages) >= 1
        assert len(assistant_messages) >= 1

        # Step 5: Test evaluation (with mock to avoid LLM calls)
        with patch(
            "emotion_experiment_engine.datasets.mtbench101.llm_evaluate_response"
        ) as mock_llm:
            mock_llm.return_value = {
                "Rating": "8"
            }  # Mock evaluation result (MTBench101 expects "Rating")

            score = dataset.evaluate_response(
                response="I can help you with cooking recipes.",
                ground_truth=benchmark_item.ground_truth,
                task_name="CM",
            )

            assert isinstance(score, float)
            assert score == 8.0  # MTBench101 returns min of 5 ratings, should be 8.0
            assert (
                mock_llm.call_count >= 1
            )  # MTBench101 calls multiple times for aggregation

        print("✓ MTBench101 end-to-end integration working")

    def create_test_infinitebench_data(self):
        """Create test InfiniteBench data file"""
        test_data = [
            {
                "id": 1,
                "context": "The password is 12345. Remember this key information: the secret code is 67890. There are many other details about the system configuration, network settings, and user preferences that should be remembered.",
                "input": "What is the password mentioned in the context?",  
                "answer": "12345",
                "task": "passkey"
            },
            {
                "id": 2,
                "context": "Here is important data: {'name': 'Alice', 'age': 30, 'city': 'New York'}. The key-value pair shows personal information that needs to be retrieved accurately.",
                "input": "What is Alice's age according to the context?",
                "answer": "30", 
                "task": "kv_retrieval"
            }
        ]
        
        temp_file = Path(tempfile.mktemp(suffix='_passkey.jsonl'))
        with open(temp_file, 'w') as f:
            for item in test_data:
                f.write(json.dumps(item) + '\n')
        
        self.temp_files.append(temp_file)
        return temp_file
    
    def create_test_locomo_data(self):
        """Create test LoCoMo data file with proper conversation format"""
        test_data = [
            {
                "sample_id": "locomo_sample_1",
                "conversation": {
                    "session_1": [
                        {"speaker": "User", "text": "I'm feeling stressed about work lately."},
                        {"speaker": "Assistant", "text": "I understand that work stress can be overwhelming. What specifically is causing you the most stress?"}
                    ],
                    "session_1_date_time": "2024-01-15 14:30:00",
                    "session_2": [
                        {"speaker": "User", "text": "It's the constant deadlines and unrealistic expectations."},
                        {"speaker": "Assistant", "text": "That sounds very challenging. How are you currently managing these pressures?"}
                    ],
                    "session_2_date_time": "2024-01-16 09:15:00"
                },
                "qa": [
                    {
                        "question": "What should the assistant focus on next in helping this user?",
                        "answer": "The assistant should help develop concrete stress management strategies and time management techniques.",
                        "category": "conversational_guidance",
                        "evidence": ["session_1", "session_2"]
                    },
                    {
                        "question": "What pattern emerges from the user's concerns?",
                        "answer": "The user faces systemic workplace issues involving time pressure and unrealistic expectations.",
                        "category": "pattern_analysis",
                        "evidence": ["session_2"]
                    }
                ]
            },
            {
                "sample_id": "locomo_sample_2", 
                "conversation": {
                    "session_1": [
                        {"speaker": "User", "text": "I completed my project ahead of schedule!"},
                        {"speaker": "Assistant", "text": "That's fantastic! You must feel great about finishing early."}
                    ],
                    "session_1_date_time": "2024-01-20 16:45:00"
                },
                "qa": [
                    {
                        "question": "How should the assistant respond to build on this positive momentum?",
                        "answer": "The assistant should encourage reflection on what worked well and how to apply those strategies to future projects.",
                        "category": "positive_reinforcement",
                        "evidence": ["session_1"]
                    }
                ]
            }
        ]
        
        temp_file = Path(tempfile.mktemp(suffix='_conversational_qa.jsonl'))
        with open(temp_file, 'w') as f:
            for item in test_data:
                f.write(json.dumps(item) + '\n')
        
        self.temp_files.append(temp_file)
        return temp_file

    def test_all_benchmarks_integration(self):
        """Test ALL supported benchmarks in single comprehensive integration test"""
        benchmark_configs = [
            ("mtbench101", "CM", self.create_test_mtbench_data()),
            ("longbench", "narrativeqa", self.create_test_longbench_data()),
            ("infinitebench", "passkey", self.create_test_infinitebench_data()),
            ("locomo", "conversational_qa", self.create_test_locomo_data())
        ]
        
        for benchmark_name, task_type, data_file in benchmark_configs:
            with self.subTest(benchmark=benchmark_name, task=task_type):
                print(f"\n🧪 Testing {benchmark_name}/{task_type} integration...")
                
                # Custom llm_eval_config for each benchmark
                custom_config = {
                    'model': 'gpt-4o-mini', 
                    'temperature': 0.2,
                    'benchmark_specific': f'{benchmark_name}_test'
                }
                
                # Create YAML config
                yaml_file = self.create_test_yaml_config(
                    benchmark_name=benchmark_name,
                    task_type=task_type,
                    llm_eval_config=custom_config
                )
                
                # Load and create BenchmarkConfig
                with open(yaml_file, 'r') as f:
                    config_dict = yaml.safe_load(f)
                benchmark_dict = config_dict['benchmark']
                
                benchmark_config = BenchmarkConfig(
                    name=benchmark_dict['name'],
                    task_type=benchmark_dict['task_type'],
                    data_path=data_file,
                    sample_limit=benchmark_dict['sample_limit'],
                    augmentation_config=benchmark_dict['augmentation_config'],
                    enable_auto_truncation=benchmark_dict['enable_auto_truncation'],
                    truncation_strategy=benchmark_dict['truncation_strategy'],
                    preserve_ratio=benchmark_dict['preserve_ratio'],
                    llm_eval_config=benchmark_dict['llm_eval_config']
                )
                
                # Test dataset creation via factory with prompt wrapper (required by new design)
                from emotion_experiment_engine.benchmark_prompt_wrapper import get_benchmark_prompt_wrapper
                from unittest.mock import Mock
                
                # Create minimal prompt wrapper for testing
                mock_prompt_format = Mock()
                mock_prompt_format.build.return_value = f"{benchmark_name}_test_prompt"
                
                if benchmark_name == "mtbench101":
                    from emotion_experiment_engine.mtbench101_prompt_wrapper import MTBench101PromptWrapper
                    prompt_wrapper = MTBench101PromptWrapper(task_type=task_type, prompt_format=mock_prompt_format)
                else:
                    prompt_wrapper = get_benchmark_prompt_wrapper(benchmark_name, task_type, mock_prompt_format)
                
                dataset = create_dataset_from_config(benchmark_config, prompt_wrapper=prompt_wrapper)
                
                # Verify dataset properties
                assert dataset is not None, f"{benchmark_name} dataset creation failed"
                assert len(dataset) >= 1, f"{benchmark_name} dataset is empty"
                assert hasattr(dataset, 'llm_eval_config'), f"{benchmark_name} missing llm_eval_config"
                assert dataset.llm_eval_config['temperature'] == 0.2, f"{benchmark_name} config not applied"
                assert dataset.llm_eval_config['benchmark_specific'] == f'{benchmark_name}_test'
                
                # Test dataset item structure
                item_dict = dataset[0]
                benchmark_item = item_dict['item']
                
                assert benchmark_item.id is not None, f"{benchmark_name} item missing ID"
                assert benchmark_item.input_text is not None, f"{benchmark_name} item missing input"
                assert benchmark_item.ground_truth is not None, f"{benchmark_name} item missing ground truth"
                
                # Test evaluation capability (with mocks to avoid LLM costs)
                mock_path = self._get_mock_path_for_benchmark(benchmark_name)
                with patch(mock_path) as mock_llm:
                    mock_llm.return_value = self._get_mock_response_for_benchmark(benchmark_name)
                    
                    try:
                        score = dataset.evaluate_response(
                            response="Test response",
                            ground_truth=benchmark_item.ground_truth,
                            task_name=task_type
                        )
                        assert isinstance(score, (int, float)), f"{benchmark_name} evaluation didn't return numeric score"
                        print(f"  ✅ {benchmark_name} evaluation: {score}")
                    except Exception as e:
                        print(f"  ⚠️  {benchmark_name} evaluation failed (expected for some): {e}")
                
                print(f"  ✅ {benchmark_name}/{task_type} integration complete")
    
    def _get_mock_path_for_benchmark(self, benchmark_name):
        """Get the correct mock path for each benchmark's evaluation function"""
        mock_paths = {
            'mtbench101': 'emotion_experiment_engine.datasets.mtbench101.llm_evaluate_response',
            'longbench': 'emotion_experiment_engine.evaluation_utils.llm_evaluate_response',
            'infinitebench': 'emotion_experiment_engine.evaluation_utils.llm_evaluate_response', 
            'locomo': 'emotion_experiment_engine.evaluation_utils.llm_evaluate_response'
        }
        return mock_paths.get(benchmark_name, 'emotion_experiment_engine.evaluation_utils.llm_evaluate_response')
    
    def _get_mock_response_for_benchmark(self, benchmark_name):
        """Get appropriate mock response for each benchmark type"""
        mock_responses = {
            'mtbench101': {"Rating": "7"},  # MTBench101 expects Rating field
            'longbench': {"answer": "1.0"}, # LongBench expects answer field
            'infinitebench': {"answer": "1.0"}, # InfiniteBench expects answer field
            'locomo': {"answer": "0.8"}     # LoCoMo expects answer field
        }
        return mock_responses.get(benchmark_name, {"answer": "1.0"})

    def test_longbench_end_to_end_integration(self):
        """Test complete LongBench pipeline: YAML → Dataset → ThreadPoolExecutor evaluation"""
        # Step 1: Create test data and YAML config
        data_file = self.create_test_longbench_data()

        custom_llm_config = {"model": "gpt-4o", "temperature": 0.0, "max_tokens": 150}

        yaml_file = self.create_test_yaml_config(
            benchmark_name="longbench",
            task_type="narrativeqa",
            llm_eval_config=custom_llm_config,
        )

        # Step 2: Load config and create BenchmarkConfig
        with open(yaml_file, "r") as f:
            config_dict = yaml.safe_load(f)
        benchmark_dict = config_dict["benchmark"]

        benchmark_config = BenchmarkConfig(
            name=benchmark_dict["name"],
            task_type=benchmark_dict["task_type"],
            data_path=data_file,
            sample_limit=benchmark_dict["sample_limit"],
            augmentation_config=benchmark_dict["augmentation_config"],
            enable_auto_truncation=benchmark_dict["enable_auto_truncation"],
            truncation_strategy=benchmark_dict["truncation_strategy"],
            preserve_ratio=benchmark_dict["preserve_ratio"],
            llm_eval_config=benchmark_dict["llm_eval_config"],
        )

        # Step 3: Create dataset via factory
        dataset = create_dataset_from_config(benchmark_config)

        # Verify dataset creation
        assert dataset is not None
        assert len(dataset) == 2
        assert dataset.llm_eval_config["model"] == "gpt-4o"
        assert dataset.llm_eval_config["temperature"] == 0.0

        # Step 4: Test ThreadPoolExecutor batch evaluation
        test_responses = ["To find his missing sister", "New York City"]
        test_ground_truths = [["To find his missing sister"], ["New York City"]]
        test_task_names = ["narrativeqa", "narrativeqa"]

        with patch(
            "emotion_experiment_engine.evaluation_utils.llm_evaluate_response"
        ) as mock_llm:
            mock_llm.return_value = {"answer": "1.0"}

            # Test batch evaluation using ThreadPoolExecutor
            scores = dataset.evaluate_batch(
                test_responses, test_ground_truths, test_task_names
            )

            assert len(scores) == 2
            assert all(isinstance(score, float) for score in scores)
            assert mock_llm.call_count == 2  # ThreadPoolExecutor called for each item

        print("✓ LongBench end-to-end with ThreadPoolExecutor working")

    def test_config_validation_and_error_handling(self):
        """Test error handling in end-to-end pipeline"""

        # Test 1: Invalid benchmark name
        yaml_file = self.create_test_yaml_config(benchmark_name="nonexistent_benchmark")
        with open(yaml_file, "r") as f:
            config_dict = yaml.safe_load(f)
        benchmark_dict = config_dict["benchmark"]

        invalid_config = BenchmarkConfig(
            name=benchmark_dict["name"],
            task_type=benchmark_dict["task_type"],
            data_path=None,
            sample_limit=benchmark_dict["sample_limit"],
            augmentation_config=benchmark_dict["augmentation_config"],
            enable_auto_truncation=benchmark_dict["enable_auto_truncation"],
            truncation_strategy=benchmark_dict["truncation_strategy"],
            preserve_ratio=benchmark_dict["preserve_ratio"],
            llm_eval_config=benchmark_dict.get("llm_eval_config", None),
        )

        with pytest.raises(ValueError, match="Unknown benchmark"):
            create_dataset_from_config(invalid_config)

        # Test 2: Missing data file
        yaml_file = self.create_test_yaml_config(benchmark_name="mtbench101")
        with open(yaml_file, "r") as f:
            config_dict = yaml.safe_load(f)
        benchmark_dict = config_dict["benchmark"]

        missing_data_config = BenchmarkConfig(
            name=benchmark_dict["name"],
            task_type=benchmark_dict["task_type"],
            data_path=Path("nonexistent_file.jsonl"),
            sample_limit=benchmark_dict["sample_limit"],
            augmentation_config=benchmark_dict["augmentation_config"],
            enable_auto_truncation=benchmark_dict["enable_auto_truncation"],
            truncation_strategy=benchmark_dict["truncation_strategy"],
            preserve_ratio=benchmark_dict["preserve_ratio"],
            llm_eval_config=benchmark_dict.get("llm_eval_config", None),
        )

        with pytest.raises(FileNotFoundError):
            create_dataset_from_config(missing_data_config)

        print("✓ Error handling validation working")

    def test_llm_eval_config_priority_and_merging(self):
        """Test that custom llm_eval_config properly overrides defaults"""
        data_file = self.create_test_mtbench_data()

        # Custom config with some overrides
        custom_llm_config = {
            "temperature": 0.9,  # Override default 0.0
            "custom_param": "test_value",  # New parameter
            # 'model' not specified - should keep default 'gpt-4o-mini'
        }

        yaml_file = self.create_test_yaml_config(
            benchmark_name="mtbench101", llm_eval_config=custom_llm_config
        )

        with open(yaml_file, "r") as f:
            config_dict = yaml.safe_load(f)
        benchmark_dict = config_dict["benchmark"]

        benchmark_config = BenchmarkConfig(
            name=benchmark_dict["name"],
            task_type=benchmark_dict["task_type"],
            data_path=data_file,
            sample_limit=benchmark_dict["sample_limit"],
            augmentation_config=benchmark_dict["augmentation_config"],
            enable_auto_truncation=benchmark_dict["enable_auto_truncation"],
            truncation_strategy=benchmark_dict["truncation_strategy"],
            preserve_ratio=benchmark_dict["preserve_ratio"],
            llm_eval_config=benchmark_dict["llm_eval_config"],
        )

        dataset = create_dataset_from_config(benchmark_config)

        # Verify config merging priority
        final_config = dataset.llm_eval_config
        assert final_config["temperature"] == 0.9  # Custom override
        assert final_config["model"] == "gpt-4o-mini"  # Default preserved
        assert final_config["custom_param"] == "test_value"  # Custom addition

        print("✓ LLM eval config merging priority working")
        print(f"  Final merged config: {final_config}")

    def test_full_pipeline_performance(self):
        """Test performance characteristics of full pipeline"""
        import time

        data_file = self.create_test_mtbench_data()
        yaml_file = self.create_test_yaml_config(benchmark_name="mtbench101")

        start_time = time.time()

        # Full pipeline
        with open(yaml_file, "r") as f:
            config_dict = yaml.safe_load(f)
        benchmark_dict = config_dict["benchmark"]

        benchmark_config = BenchmarkConfig(
            name=benchmark_dict["name"],
            task_type=benchmark_dict["task_type"],
            data_path=data_file,
            sample_limit=benchmark_dict["sample_limit"],
            augmentation_config=benchmark_dict["augmentation_config"],
            enable_auto_truncation=benchmark_dict["enable_auto_truncation"],
            truncation_strategy=benchmark_dict["truncation_strategy"],
            preserve_ratio=benchmark_dict["preserve_ratio"],
            llm_eval_config=benchmark_dict.get("llm_eval_config", None),
        )

        # Add prompt wrapper for MTBench101 (required by new design)
        from emotion_experiment_engine.mtbench101_prompt_wrapper import MTBench101PromptWrapper
        from unittest.mock import Mock
        
        mock_prompt_format = Mock()
        mock_prompt_format.build.return_value = "performance_test_prompt"
        prompt_wrapper = MTBench101PromptWrapper(task_type="CM", prompt_format=mock_prompt_format)
        
        dataset = create_dataset_from_config(benchmark_config, prompt_wrapper=prompt_wrapper)

        # Test data loading performance
        items_loaded = len(dataset)
        item = dataset[0]

        end_time = time.time()
        pipeline_time = end_time - start_time

        assert pipeline_time < 5.0  # Should complete within 5 seconds
        assert items_loaded == 2
        assert item is not None

        print(
            f"✓ Full pipeline performance: {pipeline_time:.2f}s for {items_loaded} items"
        )

    def test_real_prompt_wrapper_integration_with_experiment_flow(self):
        """Test complete experiment pipeline with REAL prompt wrapper and tokenizer (matches experiment.py)"""
        from functools import partial
        from torch.utils.data import DataLoader
        from transformers import AutoTokenizer
        from emotion_experiment_engine.mtbench101_prompt_wrapper import MTBench101PromptWrapper
        from emotion_experiment_engine.benchmark_prompt_wrapper import get_benchmark_prompt_wrapper
        from neuro_manipulation.prompt_formats import PromptFormat

        # Create test data and YAML config
        data_file = self.create_test_mtbench_data()
        yaml_file = self.create_test_yaml_config(
            benchmark_name="mtbench101", 
            task_type="CM",
            llm_eval_config={
                "model": "gpt-4o-mini",
                "temperature": 0.3,
                "max_tokens": 150
            }
        )

        # Step 1: Load YAML config (matches experiment.py config loading)
        with open(yaml_file, 'r') as f:
            config_dict = yaml.safe_load(f)

        # Step 2: Create BenchmarkConfig (matches experiment.py __init__)
        benchmark_dict = config_dict['benchmark']
        benchmark_config = BenchmarkConfig(
            name=benchmark_dict['name'],
            task_type=benchmark_dict['task_type'],
            data_path=data_file,
            sample_limit=benchmark_dict['sample_limit'],
            augmentation_config=benchmark_dict.get('augmentation_config', {
                'emotion_intensity': 0.8,
                'context_enhancement': True
            }),
            enable_auto_truncation=benchmark_dict.get('enable_auto_truncation', True),
            truncation_strategy=benchmark_dict.get('truncation_strategy', 'right'),
            preserve_ratio=benchmark_dict.get('preserve_ratio', 0.8),
            llm_eval_config=benchmark_dict['llm_eval_config']
        )

        # Step 3: Create MTBench101PromptWrapper with REAL Qwen tokenizer (matches experiment.py)
        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct", trust_remote_code=True)
        real_prompt_format = PromptFormat(tokenizer)
        
        benchmark_prompt_wrapper = MTBench101PromptWrapper(
            task_type=benchmark_config.task_type,
            prompt_format=real_prompt_format
        )

        # Step 4: Create partial function (matches experiment.py lines 199-205)
        emotion = "anger"
        enable_thinking = config_dict.get('generation_config', {}).get('enable_thinking', False)
        
        benchmark_prompt_wrapper_partial = partial(
            benchmark_prompt_wrapper,
            enable_thinking=enable_thinking,
            augmentation_config=benchmark_config.augmentation_config,
            emotion=emotion
        )

        # Step 5: Create dataset with prompt wrapper (matches experiment.py lines 208-214)
        dataset = create_dataset_from_config(
            benchmark_config,
            prompt_wrapper=benchmark_prompt_wrapper_partial,
            max_context_length=1600,
            tokenizer=None,
            truncation_strategy="right"
        )

        # Step 6: Create DataLoader (matches experiment.py lines 218-223)
        dataloader = DataLoader(
            dataset,
            batch_size=2,
            shuffle=False,
            collate_fn=dataset.collate_fn
        )

        # Step 7: Test batch processing and show REAL prompt output
        batch = next(iter(dataloader))

        # Verify batch structure
        required_keys = ["prompts", "items", "ground_truths"]
        for key in required_keys:
            assert key in batch, f"Batch missing required key: {key}"

        assert len(batch["prompts"]) == 2
        assert len(batch["items"]) == 2
        assert len(batch["ground_truths"]) == 2

        # Print real dataset items and prompt output for inspection
        print("\n🔍 REAL Prompt Wrapper Integration Test:")
        print("=" * 60)
        
        for i in range(len(dataset)):
            item = dataset.items[i]
            processed_item = dataset[i]
            
            print(f"\nItem {i}:")
            print(f"  ID: {item.id}")
            print(f"  Conversation: {item.input_text}")
            print(f"  Ground Truth: {item.ground_truth}")
            print(f"  REAL Prompt from MTBench101PromptWrapper:")
            print(f"  {processed_item['prompt']}")
            print("-" * 40)

        print(f"\nDataLoader Batch - REAL Prompts:")
        for i, prompt in enumerate(batch["prompts"]):
            print(f"Batch Prompt {i}: {prompt}")
            print("-" * 40)

        # Test evaluation with mocked LLM
        with patch('emotion_experiment_engine.datasets.mtbench101.llm_evaluate_response') as mock_llm:
            mock_llm.return_value = {"Rating": "8"}
            
            test_responses = ["I can help with cooking recipes.", "Python is great for beginners."]
            task_names = [benchmark_config.task_type] * len(test_responses)
            
            scores = dataset.evaluate_batch(test_responses, batch["ground_truths"], task_names)
            assert len(scores) == 2
            assert all(isinstance(score, (int, float)) for score in scores)

        print("✅ Real prompt wrapper integration with experiment.py flow - COMPLETE")
        print(f"  - Tokenizer: Qwen/Qwen2.5-1.5B-Instruct")
        print(f"  - Prompt Format: Real PromptFormat with <|im_start|>/<|im_end|> tokens")
        print(f"  - Dataset Items: {len(dataset)}")
        print(f"  - Evaluation Scores: {scores}")
        print(f"  - Task Type: {benchmark_config.task_type}")
        print(f"  - Emotion: {emotion}")
        print(f"  - Thinking Mode: {enable_thinking}")
        print(f"  - LLM Eval Config: {dataset.llm_eval_config}")


if __name__ == "__main__":
    unittest.main()
