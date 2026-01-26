"""
Tests for offline evaluation of deferred experiment runs.
"""

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd

from emotion_experiment_engine.evaluate_saved import evaluate_saved_run
from emotion_experiment_engine.experiment import EmotionExperiment
from emotion_experiment_engine.tests.test_utils import (
    MockRepControlPipeline,
    create_mock_experiment_config,
)


@patch("emotion_experiment_engine.experiment.setup_model_and_tokenizer")
@patch("emotion_experiment_engine.experiment.ModelLayerDetector")
@patch("emotion_experiment_engine.experiment.load_emotion_readers")
@patch("emotion_experiment_engine.experiment.get_pipeline")
@patch(
    "neuro_manipulation.utils.load_tokenizer_only",
    return_value=(MagicMock(), None),
)
@patch(
    "emotion_experiment_engine.experiment.EmotionExperiment._assert_tokenizers_equivalent",
    return_value=None,
)
def test_evaluate_saved_run_produces_scored_outputs(
    _mock_assert,
    _mock_tok,
    mock_get_pipeline,
    mock_load_emotion_readers,
    mock_model_detector,
    mock_setup_model,
):
    """Deferred runs should be evaluable via the standalone helper."""

    mock_model = MagicMock()
    mock_tokenizer = MagicMock()
    mock_setup_model.return_value = (
        mock_model,
        mock_tokenizer,
        "chat",
        MagicMock(),
    )
    mock_model_detector.num_layers.return_value = 8
    mock_load_emotion_readers.return_value = {
        "anger": MagicMock(),
        "happiness": MagicMock(),
    }
    mock_get_pipeline.return_value = MockRepControlPipeline(["resp_a", "resp_b"])

    config = create_mock_experiment_config("passkey", 2, defer_evaluation=True)
    temp_dir = Path(tempfile.mkdtemp())
    config.output_dir = str(temp_dir)

    experiment = EmotionExperiment(config)
    experiment.is_vllm = True

    raw_df = experiment.run_experiment()
    assert raw_df["score"].isna().all()
    run_path = Path(experiment.output_dir)
    assert (run_path / "detailed_results.csv").exists() is False
    assert (run_path / "raw_results.json").exists()

    scored_df = evaluate_saved_run(run_path, max_workers=2)

    assert not scored_df["score"].isna().any()
    assert (run_path / "detailed_results.csv").exists()
    assert (run_path / "summary_results.csv").exists()
    with open(run_path / "raw_results.json", "r", encoding="utf-8") as fh:
        persisted = json.load(fh)
        assert all("score" in row for row in persisted)

    split_metrics = run_path / "split_metrics.json"
    assert split_metrics.exists()
    payload = json.loads(split_metrics.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
