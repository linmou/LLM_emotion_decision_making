"""
Responsible file: emotion_experiment_engine/swebench_evaluation.py
Purpose: Verify deferred harness evaluation for SWE-bench runs.
"""

import json
from pathlib import Path
from types import SimpleNamespace
from typing import List

import pytest


@pytest.fixture
def sample_run_dir(tmp_path: Path) -> Path:
    run_dir = tmp_path / "results" / "swebench" / "Qwen2.5-0.5B-Instruct_swebench_patch_20250101_000000"
    predictions_dir = run_dir / "predictions"
    predictions_dir.mkdir(parents=True, exist_ok=True)

    sample_patch = {
        "instance_id": "django__django-12345",
        "model_patch": "---\n+print('hello')",
    }
    (predictions_dir / "anger_i05_r00.jsonl").write_text(
        json.dumps(sample_patch) + "\n",
        encoding="utf-8",
    )

    raw_results = [
        {
            "emotion": "anger",
            "intensity": 0.5,
            "item_id": "django__django-12345",
            "task_name": "patch",
            "response": "---\n+print('hello')",
            "ground_truth": None,
            "score": None,
            "repeat_id": 0,
            "metadata": {
                "benchmark": "swebench",
                "predictions_path": str(predictions_dir / "anger_i05_r00.jsonl"),
                "run_id": "anger_i05_r00",
            },
            "error": None,
        }
    ]
    (run_dir / "raw_results.json").write_text(
        json.dumps(raw_results, indent=2),
        encoding="utf-8",
    )

    experiment_config = {
        "model_path": "/models/Qwen2.5-0.5B-Instruct",
        "emotions": ["anger"],
        "intensities": [0.5],
        "benchmark": {
            "name": "swebench",
            "task_type": "patch",
            "data_path": "./cache/datasets/SWE-bench_Lite_text_inputs_dataset",
        },
        "output_dir": str(run_dir.parent),
    }
    (run_dir / "experiment_config.json").write_text(
        json.dumps(experiment_config, indent=2),
        encoding="utf-8",
    )
    return run_dir


def test_swebench_evaluation_invokes_harness_and_writes_manifest(
    monkeypatch: pytest.MonkeyPatch, sample_run_dir: Path, tmp_path: Path
) -> None:
    commands: List[List[str]] = []

    swebench_repo = tmp_path / "SWE-bench"
    swebench_repo.mkdir(parents=True, exist_ok=True)

    # Emulate harness report output inside provided report_dir
    def _fake_run(cmd, *, check=False, cwd=None, env=None):
        commands.append(cmd)
        assert cwd == str(swebench_repo)
        report_dir = Path(cmd[cmd.index("--report_dir") + 1])
        report_dir.mkdir(parents=True, exist_ok=True)
        model_safe = "/models/Qwen2.5-0.5B-Instruct".replace("/", "__")
        report_path = report_dir / f"{model_safe}.anger_i05_r00.json"
        report_payload = {
            "total_instances": 1,
            "resolved_instances": 1,
            "unresolved_instances": 0,
            "empty_patch_instances": 0,
            "error_instances": 0,
            "completed_instances": 1,
        }
        report_path.write_text(json.dumps(report_payload, indent=2), encoding="utf-8")
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr("subprocess.run", _fake_run)

    from emotion_experiment_engine import swebench_evaluation

    manifest = swebench_evaluation.evaluate_swebench_run(
        run_dir=sample_run_dir,
        swebench_repo=swebench_repo,
        dataset_name="SWE-bench/SWE-bench_Lite",
        split="test",
        results_root=tmp_path / "final_results",
        python_executable="python",
        max_workers=1,
    )

    assert len(commands) == 1
    cmd = commands[0]
    assert cmd[:3] == ["python", "-m", "swebench.harness.run_evaluation"]
    assert "--predictions_path" in cmd
    assert "--run_id" in cmd and cmd[cmd.index("--run_id") + 1] == "anger_i05_r00"

    prepared_path = Path(cmd[cmd.index("--predictions_path") + 1])
    contents = [json.loads(line) for line in prepared_path.read_text(encoding="utf-8").splitlines()]
    assert contents[0]["model_name_or_path"] == "/models/Qwen2.5-0.5B-Instruct"

    assert manifest["runs"][0]["pass_rate"] == 1.0
    manifest_path = tmp_path / "final_results" / "Qwen2.5-0.5B-Instruct" / (
        sample_run_dir.name + "_evaluation.json"
    )
    assert manifest_path.exists()
    stored = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert stored["runs"][0]["resolved_instances"] == 1
    assert stored["runs"][0]["harness_report_path"].endswith("anger_i05_r00.json")
