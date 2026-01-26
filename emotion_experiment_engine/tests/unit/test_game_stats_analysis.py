# Tests for emotion_experiment_engine/datasets/games.py
"""Statistical analysis from GameTheoryDataset.compute_split_metrics."""

from __future__ import annotations

from typing import Dict, List

import pytest

from emotion_experiment_engine.data_models import BenchmarkConfig, ResultRecord
from emotion_experiment_engine.datasets.games import GameTheoryDataset


def _stub_game_config() -> Dict[str, object]:
    return {
        "scenarios": [
            {"id": "s1", "event": "Pick A or B", "options": [{"id": 1, "text": "A"}, {"id": 2, "text": "B"}]}
        ]
    }


def _records_emotion_shift() -> List[ResultRecord]:
    # Anger prefers option 2; Happy prefers option 1
    recs: List[ResultRecord] = []
    for _ in range(15):
        recs.append(ResultRecord("anger", 0.1, "s1", "pd", "", "", None, 2.0, 0))
    for _ in range(5):
        recs.append(ResultRecord("anger", 0.1, "s1", "pd", "", "", None, 1.0, 0))
    for _ in range(14):
        recs.append(ResultRecord("happy", 0.1, "s1", "pd", "", "", None, 1.0, 0))
    for _ in range(6):
        recs.append(ResultRecord("happy", 0.1, "s1", "pd", "", "", None, 2.0, 0))
    return recs


def _records_intensity_shift() -> List[ResultRecord]:
    # Within anger: low intensity prefers 1, high prefers 2
    recs: List[ResultRecord] = []
    for _ in range(14):
        recs.append(ResultRecord("anger", 0.1, "s1", "pd", "", "", None, 1.0, 0))
    for _ in range(6):
        recs.append(ResultRecord("anger", 0.1, "s1", "pd", "", "", None, 2.0, 0))
    for _ in range(16):
        recs.append(ResultRecord("anger", 0.9, "s1", "pd", "", "", None, 2.0, 0))
    for _ in range(4):
        recs.append(ResultRecord("anger", 0.9, "s1", "pd", "", "", None, 1.0, 0))
    return recs


@pytest.fixture()
def cfg(monkeypatch: pytest.MonkeyPatch) -> BenchmarkConfig:
    monkeypatch.setattr(
        "emotion_experiment_engine.datasets.games.get_game_config",
        lambda task_type: _stub_game_config(),
    )
    # Bypass scenario_class loading by stubbing the loader
    from emotion_experiment_engine.datasets.games import GameTheoryDataset
    from emotion_experiment_engine.data_models import BenchmarkItem
    monkeypatch.setattr(
        GameTheoryDataset,
        "_load_and_parse_data",
        lambda self: [
            BenchmarkItem(id="s1", input_text="Pick A or B", context=None, ground_truth=None, metadata={"options": [{"id":1,"text":"A"},{"id":2,"text":"B"}]})
        ],
        raising=False,
    )
    return BenchmarkConfig(
        name="game_theory",
        task_type="Prisoners_Dilemma",
        data_path=None,
        base_data_dir=None,
        sample_limit=None,
        augmentation_config=None,
        enable_auto_truncation=False,
        truncation_strategy="right",
        preserve_ratio=1.0,
        llm_eval_config=None,
    )


def test_emotion_effect_is_significant(cfg: BenchmarkConfig) -> None:
    ds = GameTheoryDataset(cfg, prompt_wrapper=None, answer_wrapper=None)
    metrics = ds.compute_split_metrics(_records_emotion_shift())
    assert "stats" in metrics
    st = metrics["stats"]
    assert "emotion_effect" in st
    ee = st["emotion_effect"]
    assert ee["p_value"] < 0.05
    assert ee["significant"] is True
    assert "pairwise" in ee
    assert any("anger_vs_happy" in k or "happy_vs_anger" in k for k in ee["pairwise"].keys())


def test_intensity_effect_is_significant(cfg: BenchmarkConfig) -> None:
    ds = GameTheoryDataset(cfg, prompt_wrapper=None, answer_wrapper=None)
    metrics = ds.compute_split_metrics(_records_intensity_shift())
    st = metrics["stats"]
    assert "intensity_effect" in st
    ie = st["intensity_effect"]
    assert "anger" in ie
    assert ie["anger"]["p_value"] < 0.05
    assert ie["anger"]["significant"] is True
