# Tests for emotion_experiment_engine.evaluate_saved_series wrapper to batch deferred evaluations
import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


def _ensure_hf_stub() -> None:
    import sys
    import types

    hub = sys.modules.get("huggingface_hub")
    if hub is None:
        hub = types.ModuleType("huggingface_hub")
        sys.modules["huggingface_hub"] = hub
    if not hasattr(hub, "HfFileSystem"):
        class _DummyFileSystem:  # pragma: no cover - simple stub
            pass

        hub.HfFileSystem = _DummyFileSystem
    if not hasattr(hub, "hf_hub_download"):
        def _hf_hub_download(*_args, **_kwargs):  # pragma: no cover - simple stub
            raise RuntimeError("hf_hub_download is stubbed in unit tests")

        hub.hf_hub_download = _hf_hub_download

    if "huggingface_hub.hf_file_system" not in sys.modules:
        sys.modules["huggingface_hub.hf_file_system"] = types.ModuleType(
            "huggingface_hub.hf_file_system"
        )


_ensure_hf_stub()


def _make_run_dir(base: Path, name: str, evaluated: bool) -> Path:
    run_dir = base / name
    run_dir.mkdir()
    readme = run_dir / "README.md"
    if evaluated:
        readme.write_text("# Experiment Results Files\n", encoding="utf-8")
        (run_dir / "evaluation_summary.json").write_text(
            json.dumps({"status": "complete"}), encoding="utf-8"
        )
    else:
        readme.write_text("# Evaluation Deferred\n", encoding="utf-8")
    (run_dir / "experiment_config.json").write_text("{}", encoding="utf-8")
    (run_dir / "raw_results.json").write_text("[]", encoding="utf-8")
    return run_dir


@pytest.fixture
def series_report(tmp_path: Path) -> Path:
    evaluated = _make_run_dir(tmp_path, "evaluated_run", True)
    pending = _make_run_dir(tmp_path, "pending_run", False)
    report = tmp_path / "series_report.json"
    payload = {
        "experiments": {
            "exp_a": {"status": "completed", "output_dir": str(evaluated)},
            "exp_b": {"status": "completed", "output_dir": str(pending)},
        }
    }
    report.write_text(json.dumps(payload), encoding="utf-8")
    return report


@pytest.mark.parametrize("dry_run", [True, False])
def test_evaluate_saved_series_filters_pending_runs(series_report: Path, dry_run: bool) -> None:
    report_dir = series_report.parent
    pending_dir = report_dir / "pending_run"

    with patch(
        "emotion_experiment_engine.evaluate_saved_series._evaluate_saved_run"
    ) as mock_eval:
        if dry_run:
            mock_eval.return_value = MagicMock()
        else:
            def _fake_eval(run_dir: Path, max_workers: int = 8) -> MagicMock:
                (run_dir / "summary_results.csv").write_text("score\n", encoding="utf-8")
                (run_dir / "README.md").write_text("# Evaluation Completed\n", encoding="utf-8")
                return MagicMock()

            mock_eval.side_effect = _fake_eval

        import emotion_experiment_engine.evaluate_saved_series as evaluate_saved_series

        result = evaluate_saved_series.process_report(series_report, dry_run=dry_run)

    assert pending_dir.resolve() in result.pending_dirs
    assert report_dir / "evaluated_run" not in result.pending_dirs

    if dry_run:
        mock_eval.assert_not_called()
    else:
        mock_eval.assert_called_once_with(pending_dir.resolve(), max_workers=8)


def test_evaluate_saved_series_updates_readme(series_report: Path) -> None:
    pending_dir = series_report.parent / "pending_run"

    with patch(
        "emotion_experiment_engine.evaluate_saved_series._evaluate_saved_run"
    ) as mock_eval:
        def _fake_eval(run_dir: Path, max_workers: int = 8) -> MagicMock:
            (run_dir / "summary_results.csv").write_text("score\n", encoding="utf-8")
            (run_dir / "README.md").write_text("# Evaluation Completed\n", encoding="utf-8")
            return MagicMock()

        mock_eval.side_effect = _fake_eval

        import emotion_experiment_engine.evaluate_saved_series as evaluate_saved_series

        evaluate_saved_series.process_report(series_report, dry_run=False)

    updated = pending_dir / "README.md"
    content = updated.read_text(encoding="utf-8")
    assert "Evaluation Deferred" not in content
    assert "Evaluation Completed" in content


def test_evaluate_saved_series_reprocesses_when_continue_false(series_report: Path) -> None:
    pending_dir = series_report.parent / "pending_run"
    evaluated_dir = series_report.parent / "evaluated_run"

    with patch(
        "emotion_experiment_engine.evaluate_saved_series._evaluate_saved_run"
    ) as mock_eval:
        def _fake_eval(run_dir: Path, max_workers: int = 8) -> MagicMock:
            (run_dir / "summary_results.csv").write_text("score\n", encoding="utf-8")
            (run_dir / "README.md").write_text("# Evaluation Completed\n", encoding="utf-8")
            return MagicMock()

        mock_eval.side_effect = _fake_eval

        import emotion_experiment_engine.evaluate_saved_series as evaluate_saved_series

        evaluate_saved_series.process_report(
            series_report,
            dry_run=False,
            continue_completed=False,
        )

    called_dirs = {call.args[0] for call in mock_eval.call_args_list}
    assert pending_dir.resolve() in called_dirs
    assert evaluated_dir.resolve() in called_dirs


def test_evaluate_saved_series_watch_mode_polls_until_complete(tmp_path: Path) -> None:
    # Tests for emotion_experiment_engine.evaluate_saved_series watch mode polling until all runs finish.
    report = tmp_path / "series_report.json"
    run_a = _make_run_dir(tmp_path, "run_a", evaluated=False)
    run_b = _make_run_dir(tmp_path, "run_b", evaluated=False)

    payload = {
        "experiments": {
            "exp_a": {"status": "completed", "output_dir": str(run_a)},
            "exp_b": {"status": "pending", "output_dir": str(run_b)},
        }
    }
    report.write_text(json.dumps(payload), encoding="utf-8")

    with patch(
        "emotion_experiment_engine.evaluate_saved_series._evaluate_saved_run"
    ) as mock_eval, patch(
        "emotion_experiment_engine.evaluate_saved_series.time.sleep"
    ) as mock_sleep:
        def _fake_eval(run_dir: Path, max_workers: int = 8) -> MagicMock:
            (run_dir / "summary_results.csv").write_text("score\n", encoding="utf-8")
            (run_dir / "README.md").write_text("# Evaluation Completed\n", encoding="utf-8")
            return MagicMock()

        mock_eval.side_effect = _fake_eval

        def _sleep_and_update(_: float) -> None:
            updated = json.loads(report.read_text(encoding="utf-8"))
            updated["experiments"]["exp_b"]["status"] = "completed"
            report.write_text(json.dumps(updated), encoding="utf-8")

        mock_sleep.side_effect = _sleep_and_update

        import emotion_experiment_engine.evaluate_saved_series as evaluate_saved_series

        evaluate_saved_series.watch_report(report, poll_interval_seconds=0.0, max_workers=8)

    called_dirs = {call.args[0] for call in mock_eval.call_args_list}
    assert run_a.resolve() in called_dirs
    assert run_b.resolve() in called_dirs


def test_evaluate_saved_series_watch_mode_keeps_watching_on_failure(tmp_path: Path) -> None:
    # Tests for emotion_experiment_engine.evaluate_saved_series watch mode continues after evaluation errors.
    report = tmp_path / "series_report.json"
    run_a = _make_run_dir(tmp_path, "run_a", evaluated=False)
    run_b = _make_run_dir(tmp_path, "run_b", evaluated=False)

    payload = {
        "experiments": {
            "exp_a": {"status": "completed", "output_dir": str(run_a)},
            "exp_b": {"status": "completed", "output_dir": str(run_b)},
        }
    }
    report.write_text(json.dumps(payload), encoding="utf-8")

    calls: list[Path] = []

    with patch(
        "emotion_experiment_engine.evaluate_saved_series._evaluate_saved_run"
    ) as mock_eval, patch(
        "emotion_experiment_engine.evaluate_saved_series.time.sleep"
    ) as mock_sleep:
        def _fake_eval(run_dir: Path, max_workers: int = 8) -> MagicMock:
            calls.append(run_dir)
            if len(calls) == 1:
                raise RuntimeError("boom")
            (run_dir / "summary_results.csv").write_text("score\n", encoding="utf-8")
            (run_dir / "README.md").write_text("# Evaluation Completed\n", encoding="utf-8")
            return MagicMock()

        mock_eval.side_effect = _fake_eval
        mock_sleep.side_effect = lambda _: None

        import emotion_experiment_engine.evaluate_saved_series as evaluate_saved_series

        evaluate_saved_series.watch_report(report, poll_interval_seconds=0.0, max_workers=8)

    # One run fails, but watch still evaluates the others and then exits (terminal report).
    assert len(calls) == 2


def test_evaluate_saved_series_skips_error_and_records_failure(tmp_path: Path) -> None:
    # Tests for emotion_experiment_engine.evaluate_saved_series recording failed eval runs instead of getting stuck.
    report = tmp_path / "series_report.json"
    bad_run = _make_run_dir(tmp_path, "bad_run", evaluated=False)
    good_run = _make_run_dir(tmp_path, "good_run", evaluated=False)

    payload = {
        "experiments": {
            "exp_bad": {"status": "completed", "output_dir": str(bad_run)},
            "exp_good": {"status": "completed", "output_dir": str(good_run)},
        }
    }
    report.write_text(json.dumps(payload), encoding="utf-8")

    with patch(
        "emotion_experiment_engine.evaluate_saved_series._evaluate_saved_run"
    ) as mock_eval:
        def _fake_eval(run_dir: Path, max_workers: int = 8) -> MagicMock:
            if run_dir.name == "bad_run":
                raise ValueError("no raw rows")
            (run_dir / "summary_results.csv").write_text("score\n", encoding="utf-8")
            (run_dir / "README.md").write_text("# Evaluation Completed\n", encoding="utf-8")
            return MagicMock()

        mock_eval.side_effect = _fake_eval

        import emotion_experiment_engine.evaluate_saved_series as evaluate_saved_series

        result = evaluate_saved_series.process_report(report, dry_run=False)

    assert good_run.resolve() in result.evaluated_dirs
    assert bad_run.resolve() in result.failed_dirs


def test_evaluate_saved_series_process_folder_recursively_filters_pending_runs(tmp_path: Path) -> None:
    # Tests for emotion_experiment_engine.evaluate_saved_series.process_folder recursively locating run dirs.
    nested = tmp_path / "nested" / "runs"
    nested.mkdir(parents=True)

    evaluated = _make_run_dir(nested, "evaluated_run", evaluated=True)
    pending = _make_run_dir(nested, "pending_run", evaluated=False)

    with patch(
        "emotion_experiment_engine.evaluate_saved_series._evaluate_saved_run"
    ) as mock_eval:
        def _fake_eval(run_dir: Path, max_workers: int = 8) -> MagicMock:
            (run_dir / "summary_results.csv").write_text("score\n", encoding="utf-8")
            (run_dir / "README.md").write_text("# Evaluation Completed\n", encoding="utf-8")
            return MagicMock()

        mock_eval.side_effect = _fake_eval

        import emotion_experiment_engine.evaluate_saved_series as evaluate_saved_series

        result = evaluate_saved_series.process_folder(tmp_path, dry_run=False)

    assert pending.resolve() in result.pending_dirs
    assert evaluated.resolve() not in result.pending_dirs
    mock_eval.assert_called_once_with(pending.resolve(), max_workers=8)


def test_evaluate_saved_series_cli_accepts_folder_argument(tmp_path: Path) -> None:
    # Tests for emotion_experiment_engine.evaluate_saved_series CLI wiring for --folder.
    pending = _make_run_dir(tmp_path, "pending_run", evaluated=False)

    with patch(
        "emotion_experiment_engine.evaluate_saved_series._evaluate_saved_run"
    ) as mock_eval:
        def _fake_eval(run_dir: Path, max_workers: int = 8) -> MagicMock:
            (run_dir / "summary_results.csv").write_text("score\n", encoding="utf-8")
            (run_dir / "README.md").write_text("# Evaluation Completed\n", encoding="utf-8")
            return MagicMock()

        mock_eval.side_effect = _fake_eval

        import emotion_experiment_engine.evaluate_saved_series as evaluate_saved_series

        evaluate_saved_series._main(["--folder", str(tmp_path)])

    mock_eval.assert_called_once_with(pending.resolve(), max_workers=8)
