"""
Test utilities for emotion memory experiments.
Provides mock data generators and testing helpers.
"""
import json
import tempfile
from pathlib import Path
from typing import List, Dict, Any
import random

from ..data_models import BenchmarkConfig, ExperimentConfig, BenchmarkItem


def create_mock_passkey_data(num_items: int = 10) -> List[Dict[str, Any]]:
    """Create mock passkey retrieval data"""
    data = []
    for i in range(num_items):
        # Generate noise text
        noise = "The grass is green. The sky is blue. The sun is yellow. Here we go. There and back again. " * 50
        
        # Generate random passkey
        passkey = f"12345{i:03d}"
        
        # Embed passkey in noise
        context = noise + f"The passkey is {passkey}. Remember it. " + noise
        
        data.append({
            'id': i,
            'context': context,
            'input': 'What is the passkey?',
            'answer': passkey
        })
    
    return data


def create_mock_kv_retrieval_data(num_items: int = 10) -> List[Dict[str, Any]]:
    """Create mock key-value retrieval data"""
    data = []
    for i in range(num_items):
        # Generate random key-value pairs
        key = f"key_{i}"
        value = f"value_{random.randint(1000, 9999)}"
        
        # Generate noise with embedded key-value pairs
        noise_pairs = []
        for j in range(20):
            noise_key = f"noise_key_{j}"
            noise_value = f"noise_value_{random.randint(100, 999)}"
            noise_pairs.append(f"{noise_key}: {noise_value}")
        
        # Insert target pair randomly
        insert_pos = random.randint(5, 15)
        noise_pairs.insert(insert_pos, f"{key}: {value}")
        
        context = "Here are some key-value pairs:\n" + "\n".join(noise_pairs)
        
        data.append({
            'id': i,
            'context': context,
            'input': f'What is the value for {key}?',
            'answer': value
        })
    
    return data


def create_mock_qa_data(num_items: int = 10) -> List[Dict[str, Any]]:
    """Create mock reading comprehension data"""
    data = []
    stories = [
        "Alice went to the store. She bought apples and oranges. The apples were red and the oranges were sweet.",
        "Bob studied for his math test. He practiced algebra and geometry. His favorite subject was geometry.",
        "Carol walked her dog in the park. The dog's name was Max. Max liked to chase squirrels.",
        "David cooked dinner for his family. He made pasta with tomato sauce. Everyone loved the meal.",
        "Emma read a book about space. She learned about planets and stars. Her favorite planet was Jupiter."
    ]
    
    questions = [
        ("What did Alice buy?", "apples and oranges"),
        ("What was Bob's favorite subject?", "geometry"),
        ("What was the dog's name?", "Max"),
        ("What did David cook?", "pasta with tomato sauce"),
        ("What was Emma's favorite planet?", "Jupiter")
    ]
    
    for i in range(num_items):
        story_idx = i % len(stories)
        story = stories[story_idx] * 10  # Repeat to make it longer
        question, answer = questions[story_idx]
        
        data.append({
            'id': i,
            'context': story,
            'input': question,
            'answer': answer
        })
    
    return data


def create_mock_locomo_data(num_items: int = 5) -> List[Dict[str, Any]]:
    """Create mock LoCoMo conversational data"""
    data = []
    
    for i in range(num_items):
        # Create mock conversation sessions
        conversation = {
            'speaker_a': f'Alice_{i}',
            'speaker_b': f'Bob_{i}',
            'session_1': [
                {'speaker': f'Alice_{i}', 'text': f'Hi Bob, how was your day? I went to the {["park", "mall", "beach"][i % 3]} today.'},
                {'speaker': f'Bob_{i}', 'text': f'Hi Alice! My day was good. I had a meeting about the {["project", "budget", "schedule"][i % 3]}.'}
            ],
            'session_1_date_time': '2024-01-01 10:00:00',
            'session_2': [
                {'speaker': f'Alice_{i}', 'text': f'Remember when we talked about the {["vacation", "party", "dinner"][i % 3]} yesterday?'},
                {'speaker': f'Bob_{i}', 'text': f'Yes, I think we should plan it for {["summer", "winter", "spring"][i % 3]}.'}
            ],
            'session_2_date_time': '2024-01-02 15:30:00'
        }
        
        # Create QA pairs
        qa_pairs = [
            {
                'question': f'Where did Alice_{i} go?',
                'answer': ["park", "mall", "beach"][i % 3],
                'category': 'factual'
            },
            {
                'question': f'What did Bob_{i} have a meeting about?',
                'answer': ["project", "budget", "schedule"][i % 3],
                'category': 'factual'
            }
        ]
        
        data.append({
            'sample_id': f'sample_{i}',
            'conversation': conversation,
            'qa': qa_pairs
        })
    
    return data


def create_temp_data_file(data: List[Dict[str, Any]], file_format: str = 'jsonl') -> Path:
    """Create a temporary file with test data"""
    suffix = f'.{file_format}'
    with tempfile.NamedTemporaryFile(mode='w', suffix=suffix, delete=False) as f:
        if file_format == 'jsonl':
            for item in data:
                f.write(json.dumps(item) + '\n')
        else:  # json
            json.dump(data, f, indent=2)
        
        temp_path = Path(f.name)
    
    return temp_path


def create_mock_benchmark_config(task_type: str = 'passkey', num_items: int = 10) -> BenchmarkConfig:
    """Create a mock benchmark config with temporary data file"""
    if task_type == 'passkey':
        data = create_mock_passkey_data(num_items)
    elif task_type == 'kv_retrieval':
        data = create_mock_kv_retrieval_data(num_items)
    elif task_type == 'qa':
        data = create_mock_qa_data(num_items)
    elif task_type == 'locomo':
        data = create_mock_locomo_data(num_items)
    else:
        raise ValueError(f"Unknown task type: {task_type}")
    
    temp_file = create_temp_data_file(data)
    
    return BenchmarkConfig(
        name='infinitebench' if task_type != 'locomo' else 'locomo',
        data_path=temp_file,
        task_type=task_type,
        base_data_dir='test_data',
        sample_limit=num_items,
        augmentation_config=None,
        enable_auto_truncation=False,
        truncation_strategy="right",
        preserve_ratio=0.8,
        llm_eval_config=None
    )


def create_mock_experiment_config(
    task_type: str = 'passkey',
    num_items: int = 5,
    *,
    defer_evaluation: bool = False,
) -> ExperimentConfig:
    """Create a mock experiment config for testing"""
    benchmark_config = create_mock_benchmark_config(task_type, num_items)
    
    return ExperimentConfig(
        model_path="/fake/model/path",  # Will be mocked in tests
        emotions=["anger", "happiness"],
        intensities=[0.5, 1.0],
        benchmark=benchmark_config,
        output_dir="test_output",
        batch_size=2,
        generation_config={
            "temperature": 0.1,
            "max_new_tokens": 50,
            "do_sample": False,
            "top_p": 0.9
        },
        loading_config=None,
        repe_eng_config=None,
        max_evaluation_workers=2,
        pipeline_queue_size=1,
        defer_evaluation=defer_evaluation,
    )


class MockOutput:
    """Mock vLLM output for testing"""
    def __init__(self, text: str):
        self.outputs = [MockRequestOutput(text)]


class MockRequestOutput:
    """Mock vLLM request output for testing"""
    def __init__(self, text: str):
        self.text = text


class MockRepControlPipeline:
    """Mock RepE control pipeline for testing"""
    def __init__(self, responses: List[str]):
        self.responses = responses
        self.call_count = 0
    
    def __call__(self, prompts, activations=None, **kwargs):
        self.call_count += 1
        outputs = []
        for i, prompt in enumerate(prompts):
            if i < len(self.responses):
                response_text = prompt + " " + self.responses[i]
            else:
                response_text = prompt + " mock_response"
            outputs.append(MockOutput(response_text))
        return outputs


def cleanup_temp_files(configs: List[BenchmarkConfig]):
    """Clean up temporary data files"""
    for config in configs:
        try:
            Path(config.data_path).unlink()
        except FileNotFoundError:
            pass
