"""
Integration tests for the emotion memory experiment framework.
Tests the full workflow with mock data.
"""

import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd

from ...data_models import BenchmarkConfig, ExperimentConfig
from ...experiment import EmotionExperiment
from ..test_utils import (
    MockRepControlPipeline,
    cleanup_temp_files,
    create_mock_passkey_data,
    create_temp_data_file,
)


class TestIntegration(unittest.TestCase):
    """Integration tests for full workflow"""

    def setUp(self):
        """Set up test fixtures"""
        self.temp_dirs = []
        self.temp_files = []
        self.configs = []

    def tearDown(self):
        """Clean up temporary files"""
        cleanup_temp_files(self.configs)

        for temp_dir in self.temp_dirs:
            try:
                shutil.rmtree(temp_dir)
            except FileNotFoundError:
                pass

        for temp_file in self.temp_files:
            try:
                temp_file.unlink()
            except FileNotFoundError:
                pass

    @patch("emotion_experiment_engine.experiment.setup_model_and_tokenizer")
    @patch("emotion_experiment_engine.experiment.ModelLayerDetector")
    @patch("emotion_experiment_engine.experiment.load_emotion_readers")
    @patch("emotion_experiment_engine.experiment.get_pipeline")
    def test_full_passkey_experiment_workflow(
        self,
        mock_get_pipeline,
        mock_load_emotion_readers,
        mock_model_detector,
        mock_setup_model,
    ):
        """Test complete passkey experiment workflow"""
        # Create test data
        test_data = create_mock_passkey_data(5)
        temp_file = create_temp_data_file(test_data, "jsonl")
        self.temp_files.append(temp_file)

        # Create output directory
        temp_dir = Path(tempfile.mkdtemp())
        self.temp_dirs.append(temp_dir)

        # Setup experiment config
        benchmark_config = BenchmarkConfig(
            name="infinitebench",
            data_path=temp_file,
            task_type="passkey",
            evaluation_method="get_score_one_passkey",
        )
        self.configs.append(benchmark_config)

        exp_config = ExperimentConfig(
            model_path="/fake/model/path",
            emotions=["anger", "happiness"],
            intensities=[0.5, 1.0],
            benchmark=benchmark_config,
            output_dir=str(temp_dir),
            batch_size=2,
            generation_config=None,
            loading_config=None,
            repe_eng_config=None,
            max_evaluation_workers=1,
            pipeline_queue_size=1,
            defer_evaluation=False,
        )

        # Setup mocks
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        mock_setup_model.return_value = (mock_model, mock_tokenizer, "chat")
        mock_model_detector.num_layers.return_value = 8

        # Mock emotion readers
        import numpy as np

        mock_anger_reader = MagicMock()
        # Create directions and signs for all hidden layers (-1 to -8)
        mock_anger_reader.directions = {
            layer: np.array([0.1, 0.2]) for layer in range(-1, -9, -1)
        }
        mock_anger_reader.direction_signs = {
            layer: np.array([1, -1]) for layer in range(-1, -9, -1)
        }

        mock_happiness_reader = MagicMock()
        mock_happiness_reader.directions = {
            layer: np.array([0.2, 0.3]) for layer in range(-1, -9, -1)
        }
        mock_happiness_reader.direction_signs = {
            layer: np.array([-1, 1]) for layer in range(-1, -9, -1)
        }

        mock_load_emotion_readers.return_value = {
            "anger": mock_anger_reader,
            "happiness": mock_happiness_reader,
        }

        # Mock pipeline responses (correct passkeys for some, incorrect for others)
        expected_responses = []
        for i, item in enumerate(test_data):
            if i % 2 == 0:
                expected_responses.append(item["answer"])  # Correct answer
            else:
                expected_responses.append("wrong_answer")  # Wrong answer

        # Repeat for each condition (2 emotions × 2 intensities + 1 neutral = 5 conditions)
        all_responses = expected_responses * 5
        mock_pipeline = MockRepControlPipeline(all_responses)
        mock_get_pipeline.return_value = mock_pipeline

        # Run experiment
        experiment = EmotionExperiment(exp_config)

        # Mock is_vllm to True since we're using MockOutput
        experiment.is_vllm = True

        results_df = experiment.run_experiment()

        # Verify results structure
        self.assertIsInstance(results_df, pd.DataFrame)

        # Should have 5 items × (2 emotions × 2 intensities + 1 neutral) = 25 results
        expected_total = 5 * (2 * 2 + 1)
        self.assertEqual(len(results_df), expected_total)

        # Verify all required columns exist
        required_columns = [
            "emotion",
            "intensity",
            "item_id",
            "task_name",
            "score",
            "benchmark",
        ]
        for col in required_columns:
            self.assertIn(col, results_df.columns)

        # Verify emotion conditions
        emotions_in_results = set(results_df["emotion"].unique())
        expected_emotions = {"anger", "happiness", "neutral"}
        self.assertEqual(emotions_in_results, expected_emotions)

        # Verify intensity conditions
        intensities_in_results = set(results_df["intensity"].unique())
        expected_intensities = {0.0, 0.5, 1.0}
        self.assertEqual(intensities_in_results, expected_intensities)

        # Verify scores are reasonable (0.0 to 1.0)
        self.assertTrue(all(0.0 <= score <= 1.0 for score in results_df["score"]))

        # Verify output files were created
        output_files = list(experiment.output_dir.glob("*"))
        output_names = [f.name for f in output_files]

        self.assertIn("detailed_results.csv", output_names)
        self.assertIn("summary_results.csv", output_names)
        self.assertIn("raw_results.json", output_names)

        # Verify pipeline was called correct number of times
        # Should be called once per emotion condition (2 emotions × 2 intensities + 1 neutral = 5 times)
        self.assertEqual(mock_pipeline.call_count, 5)

    @patch("emotion_experiment_engine.experiment.setup_model_and_tokenizer")
    @patch("emotion_experiment_engine.experiment.ModelLayerDetector")
    @patch("emotion_experiment_engine.experiment.load_emotion_readers")
    @patch("emotion_experiment_engine.experiment.get_pipeline")
    def test_sanity_check_workflow(
        self,
        mock_get_pipeline,
        mock_load_emotion_readers,
        mock_model_detector,
        mock_setup_model,
    ):
        """Test sanity check workflow"""
        # Create test data
        test_data = create_mock_passkey_data(10)
        temp_file = create_temp_data_file(test_data, "jsonl")
        self.temp_files.append(temp_file)

        temp_dir = Path(tempfile.mkdtemp())
        self.temp_dirs.append(temp_dir)

        # Setup config
        benchmark_config = BenchmarkConfig(
            name="infinitebench",
            data_path=temp_file,
            task_type="passkey",
            evaluation_method="get_score_one_passkey",
        )
        self.configs.append(benchmark_config)

        exp_config = ExperimentConfig(
            model_path="/fake/model/path",
            emotions=["anger"],
            intensities=[1.0],
            benchmark=benchmark_config,
            output_dir=str(temp_dir),
            batch_size=5,
            generation_config=None,
            loading_config=None,
            repe_eng_config=None,
            max_evaluation_workers=1,
            pipeline_queue_size=1,
            defer_evaluation=False,
        )

        # Setup mocks
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        mock_setup_model.return_value = (mock_model, mock_tokenizer, "chat")
        mock_model_detector.num_layers.return_value = 6

        import numpy as np

        mock_emotion_reader = MagicMock()
        # Create directions and signs for all hidden layers (-1 to -6)
        mock_emotion_reader.directions = {
            layer: np.array([0.1]) for layer in range(-1, -7, -1)
        }
        mock_emotion_reader.direction_signs = {
            layer: np.array([1]) for layer in range(-1, -7, -1)
        }
        mock_load_emotion_readers.return_value = {"anger": mock_emotion_reader}

        # Mock responses for sanity check (3 items × 2 conditions = 6 responses)
        responses = ["12345000", "12345001", "12345002"] * 2
        mock_pipeline = MockRepControlPipeline(responses)
        mock_get_pipeline.return_value = mock_pipeline

        # Run sanity check
        experiment = EmotionExperiment(exp_config)

        # Mock is_vllm to True since we're using MockOutput
        experiment.is_vllm = True

        results_df = experiment.run_sanity_check(sample_limit=3)

        # Verify sanity check results
        self.assertIsInstance(results_df, pd.DataFrame)

        # Should have 3 items × (1 emotion × 1 intensity + 1 neutral) = 6 results
        expected_total = 3 * (1 * 1 + 1)
        self.assertEqual(len(results_df), expected_total)

        # Verify original config wasn't permanently modified
        # (The adapter will be recreated, so we can't easily test this)

    @patch("emotion_experiment_engine.experiment.setup_model_and_tokenizer")
    @patch("emotion_experiment_engine.experiment.ModelLayerDetector")
    @patch("emotion_experiment_engine.experiment.load_emotion_readers")
    @patch("emotion_experiment_engine.experiment.get_pipeline")
    def test_different_batch_sizes(
        self,
        mock_get_pipeline,
        mock_load_emotion_readers,
        mock_model_detector,
        mock_setup_model,
    ):
        """Test experiment with different batch sizes"""
        # Create test data
        test_data = create_mock_passkey_data(7)  # Odd number to test batching
        temp_file = create_temp_data_file(test_data, "jsonl")
        self.temp_files.append(temp_file)

        temp_dir = Path(tempfile.mkdtemp())
        self.temp_dirs.append(temp_dir)

        # Setup config with small batch size
        benchmark_config = BenchmarkConfig(
            name="infinitebench",
            data_path=temp_file,
            task_type="passkey",
            evaluation_method="get_score_one_passkey",
        )
        self.configs.append(benchmark_config)

        exp_config = ExperimentConfig(
            model_path="/fake/model/path",
            emotions=["anger"],
            intensities=[1.0],
            benchmark=benchmark_config,
            output_dir=str(temp_dir),
            batch_size=3,  # Smaller than data size
            generation_config=None,
            loading_config=None,
            repe_eng_config=None,
            max_evaluation_workers=1,
            pipeline_queue_size=1,
            defer_evaluation=False,
        )

        # Setup mocks
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        mock_setup_model.return_value = (mock_model, mock_tokenizer, "chat")
        mock_model_detector.num_layers.return_value = 4

        import numpy as np

        mock_emotion_reader = MagicMock()
        # Create directions and signs for all hidden layers (-1 to -4)
        mock_emotion_reader.directions = {
            layer: np.array([0.1]) for layer in range(-1, -5, -1)
        }
        mock_emotion_reader.direction_signs = {
            layer: np.array([1]) for layer in range(-1, -5, -1)
        }
        mock_load_emotion_readers.return_value = {"anger": mock_emotion_reader}

        # Mock responses (7 items × 2 conditions = 14 responses)
        responses = [f"response_{i}" for i in range(14)]
        mock_pipeline = MockRepControlPipeline(responses)
        mock_get_pipeline.return_value = mock_pipeline

        # Run experiment
        experiment = EmotionExperiment(exp_config)

        # Mock is_vllm to True since we're using MockOutput
        experiment.is_vllm = True

        results_df = experiment.run_experiment()

        # Verify all items were processed despite odd batch size
        expected_total = 7 * (1 * 1 + 1)  # 7 items × 2 conditions
        self.assertEqual(len(results_df), expected_total)

        # Verify all items have unique responses
        responses_in_results = results_df["response"].tolist()
        # Each response should contain one of our mock responses
        for response in responses_in_results:
            self.assertTrue(any(mock_resp in response for mock_resp in responses))


if __name__ == "__main__":
    unittest.main()
