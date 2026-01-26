"""
Unit tests: backfill chosen_behavior into existing detailed_results.csv.

Covers: emotion_experiment_engine/scripts/post_process_scripts/add_chosen_behavior.py
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd
import pytest


def _write_run_dir(run_dir: Path, *, score: float) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    raw_rows = [
        {
            "emotion": "anger",
            "intensity": 0.1,
            "item_id": "pd-1",
            "task_name": "Prisoners_Dilemma",
            "prompt": "",
            "response": "",
            "ground_truth": None,
            "score": score,
            "repeat_id": 0,
            "metadata": {
                "benchmark": "game_theory",
                "item_metadata": {
                    "options": [
                        {"id": 1, "text": "Cooperate", "behavior": "cooperate"},
                        {"id": 2, "text": "Defect", "behavior": "defect"},
                    ]
                },
            },
            "error": None,
        }
    ]
    (run_dir / "raw_results.json").write_text(json.dumps(raw_rows), encoding="utf-8")

    detailed = pd.DataFrame(
        [
            {
                "emotion": "anger",
                "intensity": 0.1,
                "item_id": "pd-1",
                "task_name": "Prisoners_Dilemma",
                "response": "",
                "ground_truth": "None",
                "score": score,
                "benchmark": "game_theory",
                "repeat_id": 0,
                "error": None,
            }
        ]
    )
    detailed.to_csv(run_dir / "detailed_results.csv", index=False)


def test_script_adds_column_recursively(tmp_path: Path) -> None:
    # Responsible for: verify recursion + correct mapping from option_id -> behavior.
    from emotion_experiment_engine.scripts.post_process_scripts.add_chosen_behavior import (
        add_chosen_behavior_under_root,
    )

    _write_run_dir(tmp_path / "a" / "run1", score=1.0)
    _write_run_dir(tmp_path / "b" / "run2", score=2.0)

    updated = add_chosen_behavior_under_root(tmp_path, strict=True)
    assert updated == 2

    df1 = pd.read_csv(tmp_path / "a" / "run1" / "detailed_results.csv")
    assert df1["chosen_behavior"].tolist() == ["cooperate"]

    df2 = pd.read_csv(tmp_path / "b" / "run2" / "detailed_results.csv")
    assert df2["chosen_behavior"].tolist() == ["defect"]


def test_script_supports_parallel_jobs(tmp_path: Path) -> None:
    # Responsible for: parallel execution should still update all runs correctly.
    from emotion_experiment_engine.scripts.post_process_scripts.add_chosen_behavior import (
        add_chosen_behavior_under_root,
    )

    _write_run_dir(tmp_path / "a" / "run1", score=1.0)
    _write_run_dir(tmp_path / "b" / "run2", score=2.0)

    updated = add_chosen_behavior_under_root(tmp_path, strict=True, jobs=2)
    assert updated == 2

    df1 = pd.read_csv(tmp_path / "a" / "run1" / "detailed_results.csv")
    assert df1["chosen_behavior"].tolist() == ["cooperate"]

    df2 = pd.read_csv(tmp_path / "b" / "run2" / "detailed_results.csv")
    assert df2["chosen_behavior"].tolist() == ["defect"]


def test_script_can_resume_by_skipping_finished_files(tmp_path: Path) -> None:
    # Responsible for: resume runs should skip already-filled CSVs.
    from emotion_experiment_engine.scripts.post_process_scripts.add_chosen_behavior import (
        add_chosen_behavior_under_root,
    )

    _write_run_dir(tmp_path / "run", score=1.0)
    assert add_chosen_behavior_under_root(tmp_path, strict=True) == 1

    # Second pass: nothing left to fill, should do no work and not fail.
    assert add_chosen_behavior_under_root(tmp_path, strict=True) == 0


def test_script_strict_rejects_missing_option_id(tmp_path: Path) -> None:
    # Responsible for: verify strict assertion catches inconsistent score/options.
    from emotion_experiment_engine.scripts.post_process_scripts.add_chosen_behavior import (
        add_chosen_behavior_under_root,
    )

    _write_run_dir(tmp_path / "run", score=3.0)
    with pytest.raises(ValueError, match="option_id"):
        add_chosen_behavior_under_root(tmp_path, strict=True, skip_missing_raw=False)


def test_script_missing_option_id_error_includes_context(tmp_path: Path) -> None:
    # Responsible for: error should include paths + available option ids for debugging.
    from emotion_experiment_engine.scripts.post_process_scripts.add_chosen_behavior import (
        add_chosen_behavior_under_root,
    )

    run_dir = tmp_path / "run"
    _write_run_dir(run_dir, score=3.0)
    with pytest.raises(ValueError) as excinfo:
        add_chosen_behavior_under_root(tmp_path, strict=True, skip_missing_raw=False)

    msg = str(excinfo.value)
    assert "detailed_results.csv" in msg
    assert "raw_results.json" in msg
    assert "option_id=3" in msg
    assert "available_option_ids=[1, 2]" in msg


def test_script_strict_rejects_mismatched_existing_value(tmp_path: Path) -> None:
    # Responsible for: strict mode must catch existing chosen_behavior that disagrees.
    from emotion_experiment_engine.scripts.post_process_scripts.add_chosen_behavior import (
        add_chosen_behavior_under_root,
    )

    run_dir = tmp_path / "run"
    _write_run_dir(run_dir, score=1.0)
    df = pd.read_csv(run_dir / "detailed_results.csv")
    df["chosen_behavior"] = "defect"
    df.to_csv(run_dir / "detailed_results.csv", index=False)

    with pytest.raises(ValueError, match="chosen_behavior mismatch"):
        add_chosen_behavior_under_root(
            tmp_path, strict=True, overwrite=False, skip_finished=False
        )

    # Ensure the failing file is included for quick debugging.
    with pytest.raises(ValueError) as excinfo:
        add_chosen_behavior_under_root(
            tmp_path, strict=True, overwrite=False, skip_finished=False
        )
    assert "detailed_results.csv" in str(excinfo.value)


def test_script_handles_negative_score_as_unknown(tmp_path: Path) -> None:
    # Responsible for: skip option_id <= 0 (e.g., -1) without failing.
    from emotion_experiment_engine.scripts.post_process_scripts.add_chosen_behavior import (
        add_chosen_behavior_under_root,
    )

    run_dir = tmp_path / "run"
    _write_run_dir(run_dir, score=-1.0)
    updated = add_chosen_behavior_under_root(tmp_path, strict=True)
    assert updated == 1

    df = pd.read_csv(run_dir / "detailed_results.csv")
    assert df["chosen_behavior"].isna().all()


def test_script_defaults_skip_missing_raw_and_non_strict(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    # Responsible for: default CLI behavior should be safe for bulk postprocessing:
    # non-strict + skip missing raw, logging warnings instead of failing.
    from emotion_experiment_engine.scripts.post_process_scripts.add_chosen_behavior import (
        add_chosen_behavior_under_root,
    )

    _write_run_dir(tmp_path / "good" / "run", score=1.0)

    missing_raw_dir = tmp_path / "missing_raw" / "run"
    missing_raw_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {
                "emotion": "anger",
                "intensity": 0.1,
                "item_id": "pd-1",
                "task_name": "Prisoners_Dilemma",
                "response": "",
                "ground_truth": "None",
                "score": 1.0,
                "benchmark": "game_theory",
                "repeat_id": 0,
                "error": None,
            }
        ]
    ).to_csv(missing_raw_dir / "detailed_results.csv", index=False)

    caplog.set_level(logging.WARNING)
    updated = add_chosen_behavior_under_root(tmp_path)
    assert updated == 1

    df = pd.read_csv(tmp_path / "good" / "run" / "detailed_results.csv")
    assert df["chosen_behavior"].tolist() == ["cooperate"]
    assert any("Missing" in r.message and "raw_results.json" in r.message for r in caplog.records)


def test_script_non_strict_skips_invalid_raw_json(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    # Responsible for: corrupted raw_results.json should not crash bulk runs in non-strict mode.
    from emotion_experiment_engine.scripts.post_process_scripts.add_chosen_behavior import (
        add_chosen_behavior_under_root,
    )

    _write_run_dir(tmp_path / "good" / "run", score=1.0)

    bad_run = tmp_path / "bad" / "run"
    _write_run_dir(bad_run, score=1.0)
    (bad_run / "raw_results.json").write_text("{ this is not valid json", encoding="utf-8")

    caplog.set_level(logging.WARNING)
    updated = add_chosen_behavior_under_root(tmp_path, strict=False, skip_missing_raw=True, jobs=2)
    assert updated == 1
    assert any("raw_results.json" in r.message and "JSON" in r.message for r in caplog.records)


def test_script_normalizes_item_id_numeric_string(tmp_path: Path) -> None:
    # Responsible for: raw_results.json may store item_id as int, while detailed_results.csv
    # may store item_id as float/string like "137.0". Normalization should still map.
    from emotion_experiment_engine.scripts.post_process_scripts.add_chosen_behavior import (
        add_chosen_behavior_under_root,
    )

    run_dir = tmp_path / "run"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "raw_results.json").write_text(
        json.dumps(
            [
                {
                    "emotion": "anger",
                    "intensity": 0.1,
                    "item_id": 137,
                    "task_name": "Prisoners_Dilemma",
                    "prompt": "",
                    "response": "",
                    "ground_truth": None,
                    "score": 1.0,
                    "repeat_id": 0,
                    "metadata": {
                        "benchmark": "game_theory",
                        "item_metadata": {
                            "options": [
                                {"id": 1, "text": "Cooperate", "behavior": "cooperate"},
                                {"id": 2, "text": "Defect", "behavior": "defect"},
                            ]
                        },
                    },
                    "error": None,
                }
            ]
        ),
        encoding="utf-8",
    )

    pd.DataFrame(
        [
            {
                "emotion": "anger",
                "intensity": 0.1,
                "item_id": "137.0",
                "task_name": "Prisoners_Dilemma",
                "response": "",
                "ground_truth": "None",
                "score": 1.0,
                "benchmark": "game_theory",
                "repeat_id": 0,
                "error": None,
            }
        ]
    ).to_csv(run_dir / "detailed_results.csv", index=False)

    updated = add_chosen_behavior_under_root(tmp_path, strict=True, skip_missing_raw=False)
    assert updated == 1

    df2 = pd.read_csv(run_dir / "detailed_results.csv")
    assert df2["chosen_behavior"].tolist() == ["cooperate"]


def test_strict_skip_missing_raw_skips_incomplete_mapping(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    # Responsible for: in bulk mode you may want strict semantics, but still skip runs
    # where raw_results.json is incomplete vs detailed_results.csv.
    from emotion_experiment_engine.scripts.post_process_scripts.add_chosen_behavior import (
        add_chosen_behavior_under_root,
    )

    run_dir = tmp_path / "run"
    _write_run_dir(run_dir, score=1.0)

    # Add a detailed row that raw_results.json doesn't contain.
    df = pd.read_csv(run_dir / "detailed_results.csv")
    df2 = pd.concat(
        [
            df,
            pd.DataFrame(
                [
                    {
                        "emotion": "neutral",
                        "intensity": 0.0,
                        "item_id": "pd-missing",
                        "task_name": "Prisoners_Dilemma",
                        "response": "",
                        "ground_truth": "None",
                        "score": 1.0,
                        "benchmark": "game_theory",
                        "repeat_id": 0,
                        "error": None,
                    }
                ]
            ),
        ],
        ignore_index=True,
    )
    df2.to_csv(run_dir / "detailed_results.csv", index=False)

    caplog.set_level(logging.WARNING)
    updated = add_chosen_behavior_under_root(tmp_path, strict=True, skip_missing_raw=True)
    assert updated == 0
    assert any("incomplete" in r.message.lower() and "raw_results.json" in r.message for r in caplog.records)
