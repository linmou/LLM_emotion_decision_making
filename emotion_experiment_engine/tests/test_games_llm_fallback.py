"""Tests for emotion_experiment_engine/datasets/games.py LLM fallback (Gemini client support)."""

import math
import unittest
from typing import Any, Dict
from unittest.mock import MagicMock, patch

from emotion_experiment_engine.data_models import BenchmarkConfig, BenchmarkItem
from emotion_experiment_engine.datasets.games import GameTheoryDataset


class TestGameTheoryDatasetGeminiFallback(unittest.TestCase):
    def _make_config(self) -> BenchmarkConfig:
        return BenchmarkConfig(
            name="game_theory",
            task_type="Diplomacy_Escalation_Game",
            data_path=None,
            base_data_dir=None,
            sample_limit=None,
            augmentation_config=None,
            enable_auto_truncation=False,
            truncation_strategy="right",
            preserve_ratio=1.0,
            llm_eval_config={"client": "gemini", "model": "gemini-test"},
        )

    def test_llm_fallback_uses_gemini_client_when_configured(self) -> None:
        # I am starting with a failing test. This is the Red phase.
        config = self._make_config()

        with patch(
            "emotion_experiment_engine.evaluation_utils.llm_evaluate_response"
        ) as mock_llm_eval, patch.object(
            GameTheoryDataset,
            "_ensure_llm_client",
            side_effect=AssertionError("OpenAI client should not be used for gemini"),
        ) as mock_ensure_llm_client, patch.object(
            GameTheoryDataset,
            "_load_and_parse_data",
            return_value=[
                BenchmarkItem(
                    id="stub",
                    input_text="stub scenario",
                    context=None,
                    ground_truth=None,
                    metadata={"options": [{"id": 1, "text": "Cooperate"}, {"id": 2, "text": "Defect"}]},
                )
            ],
        ):
            mock_llm_eval.return_value = {"option_id": 3}

            dataset = GameTheoryDataset(
                config=config,
                prompt_wrapper=None,
                max_context_length=None,
                tokenizer=None,
                truncation_strategy="right",
                answer_wrapper=None,
            )

            # Force path where simple string matching fails so LLM fallback is used.
            prompt = "Scenario\nOption 1: Cooperate\nOption 2: Defect\nAnswer:"
            response = "This answer does not explicitly name any option."

            score = dataset.evaluate_response(
                response=response,
                ground_truth=None,
                task_name="Diplomacy_Escalation_Game",
                prompt=prompt,
            )

            self.assertEqual(score, 3.0)
            mock_llm_eval.assert_called_once()
            mock_ensure_llm_client.assert_not_called()


if __name__ == "__main__":
    unittest.main()
