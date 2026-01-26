import unittest
import torch
import yaml
from pathlib import Path
from neuro_manipulation.experiments.emotion_game_experiment import EmotionGameExperiment
from neuro_manipulation.game_theory_exp_0205 import repe_pipeline_registry
from neuro_manipulation.configs.experiment_config import get_exp_config, get_repe_eng_config, get_game_config
from constants import GameNames
import logging
from unittest.mock import patch, MagicMock, call
import sys
import pandas as pd # Import pandas for potential DataFrame checks later
from threading import Thread, current_thread
import time

# Mock vLLM classes if vllm is not installed or for isolation
try:
    from vllm.outputs import RequestOutput, CompletionOutput
except ImportError:
    # Create dummy classes if vllm is not available
    class CompletionOutput:
        # Update dummy class to accept required args
        def __init__(self, text, index=0, token_ids=None, cumulative_logprob=0.0, logprobs=None, finish_reason=None):
            self.text = text
            self.index = index
            self.token_ids = token_ids if token_ids is not None else []
            self.cumulative_logprob = cumulative_logprob
            self.logprobs = logprobs
            self.finish_reason = finish_reason

    class RequestOutput:
        # Update dummy class to accept required args
        def __init__(self, outputs, request_id="", prompt="", prompt_token_ids=None, prompt_logprobs=None, finished=True):
            self.outputs = outputs
            self.request_id = request_id
            self.prompt = prompt
            self.prompt_token_ids = prompt_token_ids
            self.prompt_logprobs = prompt_logprobs # Add missing arg
            self.finished = finished

# Make sure the experiment class is importable
# Adjust the path as necessary based on your project structure
sys.path.insert(0, '/data/home/jjl7137/LLM_EmoBehav_game_theory') # Add project root to path
from neuro_manipulation.experiments.emotion_game_experiment import EmotionGameExperiment, ExtractedResult

# Configure basic logging for tests
logging.basicConfig(level=logging.DEBUG)


class TestEmotionGameExperiment(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Setup test environment once for all tests"""
        # Register pipeline
        repe_pipeline_registry()
        
        # Load test configuration
        cls.exp_config = get_exp_config('config/test_emotion_game_config.yaml')
        cls.game_name = GameNames.from_string(cls.exp_config['experiment']['game']['name'])
        cls.model_name = cls.exp_config['experiment']['llm']['model_name']
        cls.repe_eng_config = get_repe_eng_config(cls.model_name)
        cls.game_config = get_game_config(cls.game_name)
        
        if cls.game_name.is_sequential():
            cls.game_config['previous_actions_length'] = cls.exp_config['experiment']['game']['previous_actions_length']
    
    def setUp(self):
        """Setup for each test"""
        self.experiment = EmotionGameExperiment(
            self.repe_eng_config,
            self.exp_config,
            self.game_config,
            batch_size=2,
            repeat=self.exp_config['experiment']['repeat']
        )
        
    def test_post_process_batch_alignment(self):
        """Test if the batch processing maintains proper alignment of inputs and outputs"""
        # Create a mock batch with known values
        batch = {
            'prompt': ['prompt1', 'prompt2'],
            'scenario': ['scenario1', 'scenario2'],
            'description': ['desc1', 'desc2'],
            'options': [('Cooperate1', 'Defect1'), ('Cooperate2', 'Defect2')]  # Using tuples for options
        }
        
        # Create mock control outputs that include the prompts
        control_outputs = [
            [{'generated_text': 'prompt1 {"decision": "Cooperate", "rationale": "why1", "option_id": 1}'}],
            [{'generated_text': 'prompt2 {"decision": "Defect", "rationale": "why2", "option_id": 2}'}]
        ]
        
        self.experiment.cur_emotion = 'happy'
        self.experiment.cur_coeff = 0.5
        
        # Process batch
        results = self.experiment._post_process_batch(batch, control_outputs)
        
        # Verify alignments
        self.assertEqual(len(results), 2, "Should have same number of results as inputs")
        
        # Check first result
        self.assertEqual(results[0]['scenario'], 'scenario1')
        self.assertEqual(results[0]['description'], 'desc1')
        self.assertEqual(results[0]['input'], 'prompt1')
        self.assertEqual(results[0]['decision'], 'Cooperate')
        self.assertEqual(results[0]['rationale'], 'why1')
        self.assertEqual(results[0]['category'], 1)
        
        # Check second result
        self.assertEqual(results[1]['scenario'], 'scenario2')
        self.assertEqual(results[1]['description'], 'desc2')
        self.assertEqual(results[1]['input'], 'prompt2')
        self.assertEqual(results[1]['decision'], 'Defect')
        self.assertEqual(results[1]['rationale'], 'why2')
        self.assertEqual(results[1]['category'], 2)
        
    def test_post_process_batch_with_repeats(self):
        """Test if the batch processing handles repeats correctly"""
        # Create a mock batch with known values
        batch = {
            'prompt': ['prompt1'],
            'scenario': ['scenario1'],
            'description': ['desc1'],
            'options': [['Cooperate', 'Defect']]  # Using actual game options
        }
        
        # Create mock control outputs with repeated data
        control_outputs = [
            [{'generated_text': 'prompt1 {"decision": "Cooperate", "rationale": "why1", "option_id": 1}'}],
            [{'generated_text': 'prompt1 {"decision": "Defect", "rationale": "why1_repeat", "option_id": 2}'}]
        ]
        
        self.experiment.cur_emotion = 'happy'
        self.experiment.cur_coeff = 0.5
        
        # Process batch
        results = self.experiment._post_process_batch(batch, control_outputs)
        
        # Verify repeats
        self.assertEqual(len(results), 2, "Should have number of results equal to repeats")
        
        # Check first result
        self.assertEqual(results[0]['scenario'], 'scenario1')
        self.assertEqual(results[0]['repeat_num'], 0)
        self.assertEqual(results[0]['decision'], 'Cooperate')
        
        # Check second result (repeated)
        self.assertEqual(results[1]['scenario'], 'scenario1')
        self.assertEqual(results[1]['repeat_num'], 1)
        self.assertEqual(results[1]['decision'], 'Defect')


class TestPostProcessBatch(unittest.TestCase):

    def setUp(self):
        """Set up common test resources."""
        self.mock_experiment = MagicMock(spec=EmotionGameExperiment)
        # Mock logger to avoid actual logging during tests
        self.mock_experiment.logger = MagicMock(spec=logging.Logger)
        # Mock attributes needed by the method
        self.mock_experiment.cur_emotion = "TestEmotion"
        self.mock_experiment.cur_coeff = 0.5
        self.mock_experiment.repeat = 2
        self.mock_experiment.batch_size = 2 # FIX 1: Add mocked batch_size

        # Example batch data (adjust scenarios, descriptions etc. as needed)
        self.batch_data = {
            'prompt': ["prompt1", "prompt2"] * self.mock_experiment.repeat, # Repeated prompts
            'options': [["optA1", "optB1"], ["optA2", "optB2"]] * self.mock_experiment.repeat,
            'scenario': ["scenario1", "scenario2"] * self.mock_experiment.repeat,
            'description': ["desc1", "desc2"] * self.mock_experiment.repeat
        }
        self.total_items = len(self.batch_data['prompt']) # 2 scenarios * 2 repeats = 4 items

        # Mock the return value of _post_process_single_output
        # We assume it returns ExtractedResult objects based on input
        self.mock_extracted_results = [
            ExtractedResult(option_id=1, rationale="rational1_rep1", decision="decisionA1_rep1"),
            ExtractedResult(option_id=2, rationale="rational2_rep1", decision="decisionB2_rep1"),
            ExtractedResult(option_id=1, rationale="rational1_rep2", decision="decisionA1_rep2"),
            ExtractedResult(option_id=2, rationale="rational2_rep2", decision="decisionB2_rep2"),
        ]

    @patch('neuro_manipulation.experiments.emotion_game_experiment.ThreadPoolExecutor')
    def test_post_process_batch_vllm(self, MockExecutor):
        """Test _post_process_batch with is_vllm=True."""
        self.mock_experiment.is_vllm = True

        # Mock the executor's map to return our predefined extracted results
        mock_map = MagicMock()
        mock_map.return_value = self.mock_extracted_results
        MockExecutor.return_value.__enter__.return_value.map = mock_map

        # Mock vLLM style control_outputs
        # FIX 4: Add required args to RequestOutput mocks
        mock_control_outputs = [
            RequestOutput(outputs=[CompletionOutput(text=self.batch_data['prompt'][0] + " gen_text1_rep1", index=0, token_ids=[1,2], cumulative_logprob=0.0, logprobs=None)],
                          request_id="req-0", prompt=self.batch_data['prompt'][0], prompt_token_ids=[10], prompt_logprobs=None, finished=True),
            RequestOutput(outputs=[CompletionOutput(text=self.batch_data['prompt'][1] + " gen_text2_rep1", index=0, token_ids=[3,4], cumulative_logprob=0.0, logprobs=None)],
                          request_id="req-1", prompt=self.batch_data['prompt'][1], prompt_token_ids=[11], prompt_logprobs=None, finished=True),
            RequestOutput(outputs=[CompletionOutput(text=self.batch_data['prompt'][2] + " gen_text1_rep2", index=0, token_ids=[1,2], cumulative_logprob=0.0, logprobs=None)],
                          request_id="req-2", prompt=self.batch_data['prompt'][2], prompt_token_ids=[10], prompt_logprobs=None, finished=True),
            RequestOutput(outputs=[CompletionOutput(text=self.batch_data['prompt'][3] + " gen_text2_rep2", index=0, token_ids=[3,4], cumulative_logprob=0.0, logprobs=None)],
                          request_id="req-3", prompt=self.batch_data['prompt'][3], prompt_token_ids=[11], prompt_logprobs=None, finished=True),
        ]

        # Bind the method to the mock instance for testing
        bound_method = EmotionGameExperiment._post_process_batch.__get__(self.mock_experiment, EmotionGameExperiment)

        # --- Call the method under test ---
        results = bound_method(self.batch_data, mock_control_outputs)
        # --- End Call ---

        # Assertions
        self.assertEqual(len(results), self.total_items)

        # Check the first result (scenario1, repeat 1)
        self.assertEqual(results[0]['emotion'], "TestEmotion")
        self.assertEqual(results[0]['intensity'], 0.5)
        self.assertEqual(results[0]['scenario'], "scenario1")
        self.assertEqual(results[0]['description'], "desc1")
        self.assertEqual(results[0]['input'], "prompt1")
        self.assertEqual(results[0]['output'], " gen_text1_rep1") # Note the space if replace works like that
        self.assertEqual(results[0]['rationale'], "rational1_rep1")
        self.assertEqual(results[0]['decision'], "decisionA1_rep1")
        self.assertEqual(results[0]['category'], 1)
        self.assertEqual(results[0]['repeat_num'], 0) # First repeat

        # Check the third result (scenario1, repeat 2)
        self.assertEqual(results[2]['repeat_num'], 1) # Second repeat
        self.assertEqual(results[2]['scenario'], "scenario1")
        self.assertEqual(results[2]['output'], " gen_text1_rep2")
        self.assertEqual(results[2]['rationale'], "rational1_rep2")
        self.assertEqual(results[2]['decision'], "decisionA1_rep2")

        # Verify _post_process_single_output was called correctly via the executor map
        expected_map_calls = [
            ((" gen_text1_rep1", ["optA1", "optB1"])),
            ((" gen_text2_rep1", ["optA2", "optB2"])),
            ((" gen_text1_rep2", ["optA1", "optB1"])),
            ((" gen_text2_rep2", ["optA2", "optB2"])),
        ]
        # Check the arguments passed to map
        actual_args = list(mock_map.call_args[0][1]) # Get the iterable passed to map
        self.assertEqual(len(actual_args), len(expected_map_calls))
        for actual, expected in zip(actual_args, expected_map_calls):
             self.assertEqual(actual[0], expected[0]) # Compare generated text
             self.assertListEqual(actual[1], expected[1]) # Compare options list

    @patch('neuro_manipulation.experiments.emotion_game_experiment.ThreadPoolExecutor')
    def test_post_process_batch_hf(self, MockExecutor):
        """Test _post_process_batch with is_vllm=False."""
        self.mock_experiment.is_vllm = False

        # Mock the executor's map
        mock_map = MagicMock()
        mock_map.return_value = self.mock_extracted_results
        MockExecutor.return_value.__enter__.return_value.map = mock_map

        # Mock standard Hugging Face pipeline style control_outputs
        mock_control_outputs = [
            [{'generated_text': self.batch_data['prompt'][0] + " hf_gen_text1_rep1"}],
            [{'generated_text': self.batch_data['prompt'][1] + " hf_gen_text2_rep1"}],
            [{'generated_text': self.batch_data['prompt'][2] + " hf_gen_text1_rep2"}],
            [{'generated_text': self.batch_data['prompt'][3] + " hf_gen_text2_rep2"}],
        ]

        # Bind the method
        bound_method = EmotionGameExperiment._post_process_batch.__get__(self.mock_experiment, EmotionGameExperiment)

        # --- Call the method ---
        results = bound_method(self.batch_data, mock_control_outputs)
        # --- End Call ---

        # Assertions
        self.assertEqual(len(results), self.total_items)

        # Check the second result (scenario2, repeat 1)
        self.assertEqual(results[1]['scenario'], "scenario2")
        self.assertEqual(results[1]['output'], " hf_gen_text2_rep1")
        self.assertEqual(results[1]['rationale'], "rational2_rep1")
        self.assertEqual(results[1]['decision'], "decisionB2_rep1")
        self.assertEqual(results[1]['category'], 2)
        self.assertEqual(results[1]['repeat_num'], 0)

        # Check the fourth result (scenario2, repeat 2)
        self.assertEqual(results[3]['repeat_num'], 1)
        self.assertEqual(results[3]['scenario'], "scenario2")
        self.assertEqual(results[3]['output'], " hf_gen_text2_rep2")
        self.assertEqual(results[3]['rationale'], "rational2_rep2")
        self.assertEqual(results[3]['decision'], "decisionB2_rep2")

        # Verify calls to map
        expected_map_calls = [
             ((" hf_gen_text1_rep1", ["optA1", "optB1"])),
             ((" hf_gen_text2_rep1", ["optA2", "optB2"])),
             ((" hf_gen_text1_rep2", ["optA1", "optB1"])),
             ((" hf_gen_text2_rep2", ["optA2", "optB2"])),
        ]
        actual_args = list(mock_map.call_args[0][1])
        self.assertEqual(len(actual_args), len(expected_map_calls))
        for actual, expected in zip(actual_args, expected_map_calls):
             self.assertEqual(actual[0], expected[0])
             self.assertListEqual(actual[1], expected[1])


class TestForwardDataloaderParallelism(unittest.TestCase):

    def setUp(self):
        """Set up mocks for testing _forward_dataloader."""
        self.mock_experiment = MagicMock(spec=EmotionGameExperiment)
        self.mock_experiment.logger = logging.getLogger("TestForwardDataloader")
        self.mock_experiment.logger.setLevel(logging.DEBUG)
        # Ensure logs are visible during test runs
        if not self.mock_experiment.logger.handlers:
             self.mock_experiment.logger.addHandler(logging.StreamHandler(sys.stdout))

        self.mock_experiment.repeat = 1 # Simplify: no repeats for this test
        self.mock_experiment.batch_size = 1 # Simplify: batch size of 1
        self.mock_experiment.generation_config = {
            'temperature': 0.1,
            'max_new_tokens': 50,
            'do_sample': False,
            'top_p': 1.0
        }
        self.mock_experiment.is_vllm = False # Assume HF pipeline for simplicity

        # Mock time-consuming methods
        self.pipeline_call_times = {}
        self.post_process_call_times = {}
        self.post_process_finish_times = {}

        def mock_pipeline_side_effect(prompts, *args, **kwargs):
            batch_idx = len(self.pipeline_call_times)
            start_time = time.time()
            self.pipeline_call_times[batch_idx] = start_time
            self.mock_experiment.logger.info(f"MockPipeline: Start Batch {batch_idx} at {start_time:.2f}")
            time.sleep(0.2) # Simulate pipeline work (shorter)
            self.mock_experiment.logger.info(f"MockPipeline: Finish Batch {batch_idx} at {time.time():.2f}")
            # Return dummy output structure matching expected format
            # Based on _post_process_batch, HF expects list of lists of dicts
            return [[{'generated_text': f"output_for_{p}"}] for p in prompts]

        def mock_post_process_side_effect(batch, control_outputs, batch_idx):
            start_time = time.time()
            self.post_process_call_times[batch_idx] = start_time
            self.mock_experiment.logger.info(f"MockPostProcess: Start Batch {batch_idx} at {start_time:.2f} on {current_thread().name}")
            time.sleep(0.5) # Simulate post-processing work (longer)
            finish_time = time.time()
            self.post_process_finish_times[batch_idx] = finish_time
            self.mock_experiment.logger.info(f"MockPostProcess: Finish Batch {batch_idx} at {finish_time:.2f}")
            # Return dummy results structure
            return [{ 'result': f"processed_{batch_idx}_{i}"} for i in range(len(batch['prompt'])) ]

        self.mock_experiment.rep_control_pipeline = MagicMock(side_effect=mock_pipeline_side_effect)
        # Directly assign the method for testing instance methods
        self.mock_experiment._post_process_batch = MagicMock(side_effect=mock_post_process_side_effect)

        # Mock dataloader (simple list of batches)
        self.mock_dataloader = [
            {'prompt': ['p1'], 'other': ['o1']}, # Batch 0
            {'prompt': ['p2'], 'other': ['o2']}, # Batch 1
            {'prompt': ['p3'], 'other': ['o3']}, # Batch 2
        ]

        # Mock activations
        self.mock_activations = {'layer_key': torch.tensor([0.1])}

    def test_parallel_execution_evidence(self):
        """Test if pipeline and post-processing show signs of parallel execution."""
        # Bind the method from the class to the instance
        bound_forward_dataloader = EmotionGameExperiment._forward_dataloader.__get__(self.mock_experiment, EmotionGameExperiment)

        # Run the dataloader processing
        # Use assertLogs to capture logs specifically from our test logger
        with self.assertLogs(self.mock_experiment.logger, level='INFO') as cm:
            results = bound_forward_dataloader(self.mock_dataloader, self.mock_activations)

        # --- Analysis --- 
        # Check if expected number of results are returned
        self.assertEqual(len(results), 3)
        # Check if methods were called expected number of times
        self.assertEqual(self.mock_experiment.rep_control_pipeline.call_count, 3)
        self.assertEqual(self.mock_experiment._post_process_batch.call_count, 3)

        # Verify parallel execution: 
        # Check if pipeline for batch N+1 starts before post-processing for batch N finishes.
        # Requires at least 2 batches
        self.assertTrue(len(self.mock_dataloader) >= 2, "Test needs at least 2 batches for parallelism check")

        pipeline_start_1 = self.pipeline_call_times.get(1, float('inf'))
        post_process_finish_0 = self.post_process_finish_times.get(0, float('-inf'))

        self.mock_experiment.logger.info(f"Check 1: Pipeline Start B1 ({pipeline_start_1:.2f}) vs PostProcess Finish B0 ({post_process_finish_0:.2f})")
        self.assertLess(pipeline_start_1, post_process_finish_0,
                        f"Pipeline for batch 1 (started {pipeline_start_1:.2f}) should start before post-processing for batch 0 finishes (finished {post_process_finish_0:.2f}). Execution might be sequential.")

        # Check if post-processing for batch N+1 starts before post-processing for batch N finishes (ThreadPoolExecutor working)
        post_process_start_1 = self.post_process_call_times.get(1, float('inf'))
        post_process_finish_0 = self.post_process_finish_times.get(0, float('-inf')) # Reuse from above

        self.mock_experiment.logger.info(f"Check 2: PostProcess Start B1 ({post_process_start_1:.2f}) vs PostProcess Finish B0 ({post_process_finish_0:.2f})")
        self.assertLess(post_process_start_1, post_process_finish_0,
                       f"Post-processing for batch 1 (started {post_process_start_1:.2f}) should start before post-processing for batch 0 finishes (finished {post_process_finish_0:.2f}). ThreadPoolExecutor might not be concurrent.")

        # Optional: Print captured logs for manual inspection if needed
        # print("\nCaptured Logs:")
        # for record in cm.records:
        #     print(record.getMessage())


if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)

# To run this test specifically:
# python -m unittest neuro_manipulation.tests.test_emotion_game_experiment.TestPostProcessBatch 