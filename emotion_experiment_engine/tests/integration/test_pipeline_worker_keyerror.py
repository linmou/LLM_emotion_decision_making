"""
Test for pipeline_worker KeyError bug in experiment.py.
This test file is responsible for testing experiment.py pipeline_worker functionality.
Purpose: Expose the KeyError when accessing batch["prompt"] instead of batch["prompts"]
"""

import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np

from ...experiment import EmotionExperiment
from ..test_utils import cleanup_temp_files, create_mock_experiment_config


class TestPipelineWorkerKeyError(unittest.TestCase):
    """Test pipeline worker with real dataset collate behavior to expose KeyError bug"""

    def setUp(self):
        """Set up test fixtures"""
        self.temp_dirs = []
        self.configs = []

    def tearDown(self):
        """Clean up temporary files and directories"""
        cleanup_temp_files(self.configs)

        for temp_dir in self.temp_dirs:
            try:
                shutil.rmtree(temp_dir)
            except FileNotFoundError:
                pass

    @patch("emotion_experiment_engine.experiment.setup_model_and_tokenizer")
    @patch("emotion_experiment_engine.experiment.ModelLayerDetector")
    @patch("emotion_experiment_engine.experiment.load_emotion_readers")
    @patch("emotion_experiment_engine.experiment.get_pipeline")
    def test_pipeline_worker_keyerror_with_real_collate_fn(
        self,
        mock_get_pipeline,
        mock_load_emotion_readers,
        mock_model_detector,
        mock_setup_model,
    ):
        """
        RED PHASE: Test that pipeline_worker fails with KeyError when accessing batch['prompt']

        This test exposes the bug where experiment.py:321 tries to access batch["prompt"]
        but the real collate_fn returns batch["prompts"] (plural).
        """
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

        # Mock emotion reader with proper structure
        mock_rep_reader = MagicMock()
        mock_rep_reader.directions = {
            layer: np.array([0.1, 0.2, 0.3]) for layer in range(-1, -13, -1)
        }
        mock_rep_reader.direction_signs = {
            layer: np.array([1, -1, 1]) for layer in range(-1, -13, -1)
        }
        mock_load_emotion_readers.return_value = {"anger": mock_rep_reader}

        # Create a REAL pipeline that will be called with the wrong batch key
        # This pipeline expects 'prompts' parameter but will receive batch["prompt"] which doesn't exist
        real_pipeline = MagicMock()
        real_pipeline.side_effect = lambda prompts, **kwargs: [
            MagicMock(outputs=[MagicMock(text="test response")]) for _ in prompts
        ]
        mock_get_pipeline.return_value = real_pipeline

        # Create test config
        config = create_mock_experiment_config("passkey", 1)
        temp_dir = Path(tempfile.mkdtemp())
        config.output_dir = str(temp_dir)
        self.temp_dirs.append(temp_dir)
        self.configs.append(config.benchmark)

        # Initialize experiment
        experiment = EmotionExperiment(config)
        experiment.is_vllm = True

        # Build dataloader with REAL collate_fn
        # The collate_fn in base.py returns: {'prompts': [...], 'items': [...], 'ground_truths': [...]}
        dataloader = experiment.build_dataloader()

        # This should fail with KeyError: 'prompt' because:
        # 1. Real collate_fn returns batch with "prompts" key
        # 2. experiment.py:321 tries to access batch["prompt"] (singular)
        # 3. KeyError is raised since "prompt" key doesn't exist
        with self.assertRaises(KeyError) as context:
            experiment._infer_with_activation(mock_rep_reader, dataloader)

        # Verify it's specifically the 'prompt' key that's missing
        self.assertIn("'prompt'", str(context.exception))

    @patch("emotion_experiment_engine.experiment.setup_model_and_tokenizer")
    @patch("emotion_experiment_engine.experiment.ModelLayerDetector")
    @patch("emotion_experiment_engine.experiment.load_emotion_readers")
    @patch("emotion_experiment_engine.experiment.get_pipeline")
    def test_post_process_batch_keyerror(
        self,
        mock_get_pipeline,
        mock_load_emotion_readers,
        mock_model_detector,
        mock_setup_model,
    ):
        """
        RED PHASE: Test that _post_process_memory_batch fails with KeyError

        This test exposes the second bug where experiment.py:429 tries to access batch["prompt"]
        but should access batch["prompts"].
        """
        # Setup mocks (minimal setup for this focused test)
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        mock_setup_model.return_value = (
            mock_model,
            mock_tokenizer,
            "chat",
            MagicMock(),
        )
        mock_model_detector.num_layers.return_value = 12
        mock_load_emotion_readers.return_value = {"anger": MagicMock()}
        mock_get_pipeline.return_value = MagicMock()

        config = create_mock_experiment_config("passkey", 1)
        self.configs.append(config.benchmark)

        experiment = EmotionExperiment(config)
        experiment.is_vllm = True

        # Create a batch with the CORRECT structure (as returned by real collate_fn)
        batch = {
            "prompts": ["test prompt"],  # Note: "prompts" plural
            "items": [MagicMock(id="test_1")],
            "ground_truths": ["test_answer"],
        }

        control_outputs = [MagicMock(outputs=[MagicMock(text="response")])]

        # This should fail with KeyError: 'prompt' because:
        # experiment.py:429 tries batch["prompt"] but batch has "prompts" key
        with self.assertRaises(KeyError) as context:
            experiment._post_process_batch(batch, control_outputs, 0)

        # Verify it's specifically the 'prompt' key that's missing
        self.assertIn("'prompt'", str(context.exception))


if __name__ == "__main__":
    unittest.main()
