"""
Unit tests for the main experiment class.
"""

import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch
import os

import pandas as pd

from ..data_models import BenchmarkConfig, ExperimentConfig
from ..datasets.base import BaseBenchmarkDataset
from ..experiment import EmotionExperiment
from .test_utils import (
    MockOutput,
    MockRepControlPipeline,
    cleanup_temp_files,
    create_mock_experiment_config,
)


class TestEmotionMemoryExperiment(unittest.TestCase):
    """Test the main experiment class"""

    def setUp(self):
        """Set up test fixtures"""
        self.temp_dirs = []
        self.temp_files = []
        self.configs = []

    def tearDown(self):
        """Clean up temporary files and directories"""
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
    @patch("neuro_manipulation.utils.load_tokenizer_only", return_value=(MagicMock(), None))
    @patch("emotion_experiment_engine.experiment.EmotionExperiment._assert_tokenizers_equivalent", return_value=None)
    def test_experiment_initialization(
        self,
        _mock_assert,
        _mock_tok,
        mock_get_pipeline,
        mock_load_emotion_readers,
        mock_model_detector,
        mock_setup_model,
    ):
        """Test experiment initialization"""
        # Setup mocks
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        mock_processor = MagicMock()
        mock_setup_model.return_value = (
            mock_model,
            mock_tokenizer,
            "chat",
            mock_processor,
        )
        mock_model_detector.num_layers.return_value = 12
        mock_load_emotion_readers.return_value = {
            "anger": MagicMock(),
            "happiness": MagicMock(),
        }
        mock_get_pipeline.return_value = MockRepControlPipeline(["test response"])

        # Create test config
        config = create_mock_experiment_config("passkey", 3)
        self.configs.append(config.benchmark)

        # Initialize experiment
        experiment = EmotionExperiment(config)

        # Verify initialization
        self.assertEqual(experiment.config, config)
        self.assertIsNotNone(experiment.logger)
        self.assertIsNotNone(experiment.emotion_rep_readers)
        self.assertIsNotNone(experiment.rep_control_pipeline)

        # Verify model setup was called correctly
        self.assertEqual(mock_setup_model.call_count, 2)  # Once for HF, once for vLLM
        mock_load_emotion_readers.assert_called_once()
        mock_get_pipeline.assert_called_once()

    @patch("emotion_experiment_engine.experiment.setup_model_and_tokenizer")
    @patch("emotion_experiment_engine.experiment.ModelLayerDetector")
    @patch("emotion_experiment_engine.experiment.load_emotion_readers")
    @patch("emotion_experiment_engine.experiment.get_pipeline")
    @patch("neuro_manipulation.utils.load_tokenizer_only", return_value=(MagicMock(), None))
    @patch("emotion_experiment_engine.experiment.EmotionExperiment._assert_tokenizers_equivalent", return_value=None)
    def test_process_emotion_condition_with_emotion(
        self,
        _mock_assert,
        _mock_tok,
        mock_get_pipeline,
        mock_load_emotion_readers,
        mock_model_detector,
        mock_setup_model,
    ):
        """Test processing an emotion condition"""
        # Setup mocks
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        mock_setup_model.return_value = (
            mock_model,
            mock_tokenizer,
            "chat",
            MagicMock(),
        )
        mock_model_detector.num_layers.return_value = 12

        # Mock emotion reader
        import numpy as np

        mock_rep_reader = MagicMock()
        # Create directions and signs for all hidden layers (-1 to -12)
        mock_rep_reader.directions = {
            layer: np.array([0.1, 0.2, 0.3]) for layer in range(-1, -13, -1)
        }
        mock_rep_reader.direction_signs = {
            layer: np.array([1, -1, 1]) for layer in range(-1, -13, -1)
        }
        mock_load_emotion_readers.return_value = {"anger": mock_rep_reader}

        # Mock pipeline with specific responses
        responses = ["12345", "67890", "11111"]
        mock_pipeline = MockRepControlPipeline(responses)
        mock_get_pipeline.return_value = mock_pipeline

        # Create test config
        config = create_mock_experiment_config("passkey", 3)
        self.configs.append(config.benchmark)

        # Initialize experiment
        experiment = EmotionExperiment(config)

        # Mock is_vllm to True since we're using MockOutput
        experiment.is_vllm = True

        # Build dataloader instead of using benchmark_adapter
        dataloader = experiment.build_dataloader("anger")

        # Process emotion condition using dataloader
        # Set current condition state
        experiment.cur_emotion = "anger"
        experiment.cur_intensity = 1.0
        results = experiment._infer_with_activation(mock_rep_reader, dataloader)

        # Verify results
        self.assertEqual(len(results), 3)
        for result in results:
            self.assertEqual(result.emotion, "anger")
            self.assertEqual(result.intensity, 1.0)
            self.assertEqual(result.task_name, "passkey")
            self.assertIsInstance(result.score, float)
        # Check that each response matches one from our mock list
        resp_texts = [r.response for r in results]
        for t in resp_texts:
            self.assertTrue(any(expected in t for expected in responses))

    @patch("emotion_experiment_engine.experiment.setup_model_and_tokenizer")
    @patch("emotion_experiment_engine.experiment.ModelLayerDetector")
    @patch("emotion_experiment_engine.experiment.load_emotion_readers")
    @patch("emotion_experiment_engine.experiment.get_pipeline")
    @patch("neuro_manipulation.utils.load_tokenizer_only", return_value=(MagicMock(), None))
    @patch("emotion_experiment_engine.experiment.EmotionExperiment._assert_tokenizers_equivalent", return_value=None)
    def test_process_neutral_condition(
        self,
        _mock_assert,
        _mock_tok,
        mock_get_pipeline,
        mock_load_emotion_readers,
        mock_model_detector,
        mock_setup_model,
    ):
        """Test processing neutral baseline condition"""
        # Setup mocks (same as above)
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        mock_processor = MagicMock()
        mock_setup_model.return_value = (
            mock_model,
            mock_tokenizer,
            "chat",
            mock_processor,
        )
        mock_model_detector.num_layers.return_value = 12
        mock_load_emotion_readers.return_value = {
            "anger": MagicMock(),
            "happiness": MagicMock(),
        }

        responses = ["neutral_response"]
        mock_pipeline = MockRepControlPipeline(responses)
        mock_get_pipeline.return_value = mock_pipeline

        # Create test config
        config = create_mock_experiment_config("passkey", 1)
        self.configs.append(config.benchmark)

        # Initialize experiment
        experiment = EmotionExperiment(config)

        # Mock is_vllm to True since we're using MockOutput
        experiment.is_vllm = True

        # Build dataloader instead of using benchmark_adapter
        dataloader = experiment.build_dataloader("neutral")

        # Prepare dummy reader and set state for neutral (intensity 0)
        import numpy as np
        dummy_reader = MagicMock()
        dummy_reader.directions = {layer: np.array([0.0]) for layer in range(-1, -13, -1)}
        dummy_reader.direction_signs = {layer: np.array([1.0]) for layer in range(-1, -13, -1)}
        experiment.cur_emotion = "neutral"
        experiment.cur_intensity = 0.0
        # Process neutral condition using dataloader
        results = experiment._infer_with_activation(dummy_reader, dataloader)

        # Verify results
        self.assertEqual(len(results), 1)
        result = results[0]
        self.assertEqual(result.emotion, "neutral")
        self.assertEqual(result.intensity, 0.0)
        self.assertIn("neutral_response", result.response)

    @patch("emotion_experiment_engine.experiment.setup_model_and_tokenizer")
    @patch("emotion_experiment_engine.experiment.ModelLayerDetector")
    @patch("emotion_experiment_engine.experiment.load_emotion_readers")
    @patch("emotion_experiment_engine.experiment.get_pipeline")
    @patch("neuro_manipulation.utils.load_tokenizer_only", return_value=(MagicMock(), None))
    @patch("emotion_experiment_engine.experiment.EmotionExperiment._assert_tokenizers_equivalent", return_value=None)
    def test_save_results(
        self,
        _mock_assert,
        _mock_tok,
        mock_get_pipeline,
        mock_load_emotion_readers,
        mock_model_detector,
        mock_setup_model,
    ):
        """Test saving experiment results"""
        # Setup mocks
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        mock_setup_model.return_value = (
            mock_model,
            mock_tokenizer,
            "chat",
            MagicMock(),
        )
        mock_model_detector.num_layers.return_value = 12
        mock_load_emotion_readers.return_value = {
            "anger": MagicMock(),
            "happiness": MagicMock(),
        }
        mock_get_pipeline.return_value = MockRepControlPipeline(["test"])

        # Create test config with temp output dir
        config = create_mock_experiment_config("passkey", 1)
        temp_dir = Path(tempfile.mkdtemp())
        config.output_dir = str(temp_dir)
        self.temp_dirs.append(temp_dir)
        self.configs.append(config.benchmark)

        # Initialize experiment
        experiment = EmotionExperiment(config)

        # Create mock results
        from ..data_models import ResultRecord

        results = [
            ResultRecord(
                emotion="anger",
                intensity=1.0,
                item_id="test_1",
                task_name="passkey",
                prompt="Test prompt",
                response="Test response",
                ground_truth="Test ground truth",
                score=0.8,
                repeat_id=0,
                metadata={"benchmark": "infinitebench"},
            ),
            ResultRecord(
                emotion="neutral",
                intensity=0.0,
                item_id="test_2",
                task_name="passkey",
                prompt="Test prompt 2",
                response="Test response 2",
                ground_truth="Test ground truth 2",
                score=0.6,
                repeat_id=0,
                metadata={"benchmark": "infinitebench"},
            ),
        ]

        # Save results
        df = experiment._save_results(results)

        # Verify DataFrame
        self.assertIsInstance(df, pd.DataFrame)
        self.assertEqual(len(df), 2)
        self.assertIn("emotion", df.columns)
        self.assertIn("score", df.columns)

        # Verify files were created
        self.assertTrue((experiment.output_dir / "detailed_results.csv").exists())
        self.assertTrue((experiment.output_dir / "summary_results.csv").exists())
        self.assertTrue((experiment.output_dir / "raw_results.json").exists())

    @patch("emotion_experiment_engine.experiment.setup_model_and_tokenizer")
    @patch("emotion_experiment_engine.experiment.ModelLayerDetector")
    @patch("emotion_experiment_engine.experiment.load_emotion_readers")
    @patch("emotion_experiment_engine.experiment.get_pipeline")
    @patch("neuro_manipulation.utils.load_tokenizer_only", return_value=(MagicMock(), None))
    @patch("emotion_experiment_engine.experiment.EmotionExperiment._assert_tokenizers_equivalent", return_value=None)
    def test_run_sanity_check(
        self,
        _mock_assert,
        _mock_tok,
        mock_get_pipeline,
        mock_load_emotion_readers,
        mock_model_detector,
        mock_setup_model,
    ):
        """Test sanity check functionality"""
        # Setup mocks
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        mock_setup_model.return_value = (
            mock_model,
            mock_tokenizer,
            "chat",
            MagicMock(),
        )
        mock_model_detector.num_layers.return_value = 12

        # Mock emotion reader
        import numpy as np

        mock_rep_reader = MagicMock()
        # Create directions and signs for all hidden layers (-1 to -12)
        mock_rep_reader.directions = {
            layer: np.array([0.1, 0.2]) for layer in range(-1, -13, -1)
        }
        mock_rep_reader.direction_signs = {
            layer: np.array([1, -1]) for layer in range(-1, -13, -1)
        }
        mock_load_emotion_readers.return_value = {
            "anger": mock_rep_reader,
            "happiness": mock_rep_reader,
        }

        mock_pipeline = MockRepControlPipeline(["response1", "response2", "response3"])
        mock_get_pipeline.return_value = mock_pipeline

        # Create test config
        config = create_mock_experiment_config("passkey", 10)  # Start with 10 items
        temp_dir = Path(tempfile.mkdtemp())
        config.output_dir = str(temp_dir)
        self.temp_dirs.append(temp_dir)
        self.configs.append(config.benchmark)

        # Initialize experiment
        experiment = EmotionExperiment(config)

        # Mock is_vllm to True since we're using MockOutput
        experiment.is_vllm = True

        # Run sanity check with 2 samples
        df = experiment.run_sanity_check(sample_limit=2)

        # Verify results
        self.assertIsInstance(df, pd.DataFrame)
        # Should have 2 items × 2 emotions × 2 intensities + 2 items × 1 neutral = 10 results
        expected_results = 2 * 2 * 2 + 2 * 1
        self.assertEqual(len(df), expected_results)

        # Verify original config was restored
        self.assertEqual(config.benchmark.sample_limit, 10)

    @patch("emotion_experiment_engine.experiment.setup_model_and_tokenizer")
    @patch("emotion_experiment_engine.experiment.ModelLayerDetector")
    @patch("emotion_experiment_engine.experiment.load_emotion_readers")
    @patch("emotion_experiment_engine.experiment.get_pipeline")
    @patch("neuro_manipulation.utils.load_tokenizer_only", return_value=(MagicMock(), None))
    @patch("emotion_experiment_engine.experiment.EmotionExperiment._assert_tokenizers_equivalent", return_value=None)
    def test_evaluation_error_handling(
        self,
        _mock_assert,
        _mock_tok,
        mock_get_pipeline,
        mock_load_emotion_readers,
        mock_model_detector,
        mock_setup_model,
    ):
        """Test handling of evaluation errors"""
        # Setup mocks
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        mock_setup_model.return_value = (
            mock_model,
            mock_tokenizer,
            "chat",
            MagicMock(),
        )
        mock_model_detector.num_layers.return_value = 12
        mock_load_emotion_readers.return_value = {
            "anger": MagicMock(),
            "happiness": MagicMock(),
        }
        mock_get_pipeline.return_value = MockRepControlPipeline(["test"])

        # Create test config
        config = create_mock_experiment_config("passkey", 1)
        self.configs.append(config.benchmark)

        # Initialize experiment
        experiment = EmotionExperiment(config)

        # Mock is_vllm to True since we're using MockOutput
        experiment.is_vllm = True

        # Mock dataset evaluation to raise error
        dataloader = experiment.build_dataloader("anger")
        with patch.object(experiment, "dataset") as mock_dataset:
            mock_dataset.evaluate_batch.side_effect = Exception("Evaluation failed")
            # Prepare dummy reader and set state
            import numpy as np
            dummy_reader = MagicMock()
            dummy_reader.directions = {layer: np.array([0.1]) for layer in range(-1, -13, -1)}
            dummy_reader.direction_signs = {layer: np.array([1.0]) for layer in range(-1, -13, -1)}
            experiment.cur_emotion = "anger"
            experiment.cur_intensity = 1.0
            results = experiment._infer_with_activation(dummy_reader, dataloader)

            # Should still return results with score 0.0
            self.assertEqual(len(results), 1)
            self.assertEqual(results[0].score, 0.0)

    @patch("emotion_experiment_engine.experiment.setup_model_and_tokenizer")
    @patch("emotion_experiment_engine.experiment.ModelLayerDetector")
    @patch("emotion_experiment_engine.experiment.load_emotion_readers")
    @patch("emotion_experiment_engine.experiment.get_pipeline")
    @patch("neuro_manipulation.utils.load_tokenizer_only", return_value=(MagicMock(), None))
    @patch("emotion_experiment_engine.experiment.EmotionExperiment._assert_tokenizers_equivalent", return_value=None)
    def test_run_experiment_defer_evaluation_skips_inline_judging(
        self,
        _mock_assert,
        _mock_tok,
        mock_get_pipeline,
        mock_load_emotion_readers,
        mock_model_detector,
        mock_setup_model,
    ):
        """Deferred evaluation should skip inline scoring and emit log hints."""

        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        mock_setup_model.return_value = (
            mock_model,
            mock_tokenizer,
            "chat",
            MagicMock(),
        )
        mock_model_detector.num_layers.return_value = 12
        mock_load_emotion_readers.return_value = {
            "anger": MagicMock(),
            "happiness": MagicMock(),
        }
        mock_get_pipeline.return_value = MockRepControlPipeline(["test"])

        config = create_mock_experiment_config("passkey", 1, defer_evaluation=True)
        temp_dir = Path(tempfile.mkdtemp())
        config.output_dir = str(temp_dir)
        self.temp_dirs.append(temp_dir)
        self.configs.append(config.benchmark)

        experiment = EmotionExperiment(config)
        experiment.is_vllm = True

        with patch.object(
            BaseBenchmarkDataset,
            "evaluate_batch",
            side_effect=AssertionError("evaluate_batch should not run when deferred"),
        ):
            with patch.object(experiment.logger, "info") as mock_info:
                df = experiment.run_experiment()

        raw_path = experiment.output_dir / "raw_results.json"
        detailed_path = experiment.output_dir / "detailed_results.csv"
        self.assertTrue(raw_path.exists())
        self.assertFalse(detailed_path.exists())

        logged_messages = [str(call.args[0]) for call in mock_info.call_args_list if call.args]
        self.assertTrue(
            any("deferred evaluation" in message.lower() for message in logged_messages),
            "Expected deferred evaluation log message",
        )

        self.assertIn("score", df.columns)
        self.assertTrue(df["score"].isna().all())

    @patch("emotion_experiment_engine.experiment.setup_model_and_tokenizer")
    @patch("emotion_experiment_engine.experiment.ModelLayerDetector")
    @patch("emotion_experiment_engine.experiment.load_emotion_readers")
    @patch("emotion_experiment_engine.experiment.get_pipeline")
    @patch("neuro_manipulation.utils.load_tokenizer_only", return_value=(MagicMock(), None))
    @patch("emotion_experiment_engine.experiment.EmotionExperiment._assert_tokenizers_equivalent", return_value=None)
    def test_run_sanity_check_forces_inline_evaluation_even_when_deferred(
        self,
        _mock_assert,
        _mock_tok,
        mock_get_pipeline,
        mock_load_emotion_readers,
        mock_model_detector,
        mock_setup_model,
    ):
        """Sanity checks override deferred mode to keep quick feedback."""

        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        mock_setup_model.return_value = (
            mock_model,
            mock_tokenizer,
            "chat",
            MagicMock(),
        )
        mock_model_detector.num_layers.return_value = 12
        mock_load_emotion_readers.return_value = {
            "anger": MagicMock(),
            "happiness": MagicMock(),
        }
        mock_get_pipeline.return_value = MockRepControlPipeline(["test"])

        config = create_mock_experiment_config("passkey", 2, defer_evaluation=True)
        temp_dir = Path(tempfile.mkdtemp())
        config.output_dir = str(temp_dir)
        self.temp_dirs.append(temp_dir)
        self.configs.append(config.benchmark)

        experiment = EmotionExperiment(config)
        experiment.is_vllm = True

        eval_calls: list[int] = []

        def _fake_eval(self, responses, ground_truths, task_names, prompts):
            eval_calls.append(len(responses))
            return [1.0] * len(responses)

        with patch.object(BaseBenchmarkDataset, "evaluate_batch", _fake_eval):
            with patch.object(experiment.logger, "info") as mock_info:
                df = experiment.run_sanity_check(sample_limit=1)

        self.assertTrue(eval_calls, "Sanity check should evaluate inline")
        self.assertFalse(df["score"].isna().any())
        logged_messages = [str(call.args[0]) for call in mock_info.call_args_list if call.args]
        self.assertTrue(
            any("forcing inline evaluation" in message.lower() for message in logged_messages),
            "Expected sanity-check override log message",
        )


    def test_close_releases_pipeline_and_model_resources(self):
        """Test file: experiment.py | Purpose: ensure close() frees resources"""

        experiment = EmotionExperiment.__new__(EmotionExperiment)
        experiment.logger = MagicMock()
        experiment.is_vllm = True

        pipeline = MagicMock()
        pipeline.close = MagicMock()
        experiment.rep_control_pipeline = pipeline

        model = MagicMock()
        llm_engine = MagicMock()
        llm_engine.shutdown = MagicMock()
        model.llm_engine = llm_engine
        model.shutdown = MagicMock()
        experiment.model = model

        experiment.emotion_rep_readers = {"anger": MagicMock()}
        experiment.dataset = MagicMock()
        experiment.benchmark_prompt_wrapper_partial = MagicMock()
        experiment.emotion_datasets = {"anger": [MagicMock()]}

        experiment.close()

        pipeline.close.assert_called_once()
        llm_engine.shutdown.assert_called_once()
        model.shutdown.assert_called_once()

        self.assertIsNone(experiment.rep_control_pipeline)
        self.assertIsNone(experiment.model)
        self.assertIsNone(experiment.emotion_rep_readers)
        self.assertIsNone(experiment.dataset)
        self.assertIsNone(experiment.benchmark_prompt_wrapper_partial)
        self.assertIsNone(experiment.emotion_datasets)

    def test_close_handles_llm_engine_without_shutdown(self):
        """Close should fall back when llm_engine lacks shutdown()"""

        experiment = EmotionExperiment.__new__(EmotionExperiment)
        experiment.logger = MagicMock()
        experiment.is_vllm = True

        experiment.rep_control_pipeline = None

        engine_executor = MagicMock()
        engine_executor.shutdown = MagicMock()

        llm_engine = MagicMock()
        del llm_engine.shutdown  # Simulate absence
        llm_engine.engine_executor = engine_executor

        model = MagicMock()
        model.llm_engine = llm_engine
        model.shutdown = MagicMock()
        experiment.model = model

        experiment.emotion_rep_readers = None
        experiment.dataset = None
        experiment.benchmark_prompt_wrapper_partial = None
        experiment.emotion_datasets = None

        experiment.close()

        engine_executor.shutdown.assert_called_once_with(wait=False)
        model.shutdown.assert_called_once()

    def test_close_llm_engine_shutdown_times_out_but_does_not_block(self):
        """Close should not block indefinitely when engine shutdown hangs"""

        experiment = EmotionExperiment.__new__(EmotionExperiment)
        experiment.logger = MagicMock()
        experiment.is_vllm = True

        # Force very short timeout for this test
        os.environ["EMO_SHUTDOWN_TIMEOUT_SEC"] = "0.01"

        # Simulate a hanging llm_engine.shutdown by waiting on an Event that never sets
        import threading

        hang_event = threading.Event()

        def hang_shutdown():
            hang_event.wait(timeout=60.0)  # would block without timeout wrapper

        llm_engine = MagicMock()
        llm_engine.shutdown = hang_shutdown
        engine_executor = MagicMock()
        engine_executor.shutdown = MagicMock()
        llm_engine.engine_executor = engine_executor

        model = MagicMock()
        model.llm_engine = llm_engine
        model.shutdown = MagicMock()
        experiment.model = model

        experiment.rep_control_pipeline = None
        experiment.emotion_rep_readers = None
        experiment.dataset = None
        experiment.benchmark_prompt_wrapper_partial = None
        experiment.emotion_datasets = None

        # Should return quickly despite hanging shutdown
        experiment.close()

        # Executor shutdown should have been called non-blocking
        engine_executor.shutdown.assert_called_once_with(wait=False)
        # Model shutdown still called
        model.shutdown.assert_called_once()

        # Cleanup env var
        del os.environ["EMO_SHUTDOWN_TIMEOUT_SEC"]


if __name__ == "__main__":
    unittest.main()
