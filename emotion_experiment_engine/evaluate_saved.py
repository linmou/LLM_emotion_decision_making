"""Offline scorer for deferred emotion experiment runs.

This module gives you a clean separation between GPU generation and judge
evaluation. When an experiment is executed with ``defer_evaluation=True`` only
``raw_results.json`` is written; the judge is never called inline. Running this
module later will:

* Load the recorded configuration from ``experiment_config.json``.
* Reconstruct the benchmark dataset so it can replay the heuristics and GPT
  prompts deterministically.
* Execute the batched evaluators (using ``max_workers`` to control concurrency).
* Regenerate the usual artifacts – ``detailed_results.csv``,
  ``summary_results.csv``, and ``split_metrics.json`` – in place.

Typical CLI usage::

    python -m emotion_experiment_engine.evaluate_saved \
        --input results/TrustLLM_fairness/path_Qwen3-1.7B_20240912 \
        --max-workers 16

You can also import :func:`evaluate_saved_run` programmatically if you want to
batch-score many directories or integrate the deferred workflow into other
tooling.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
from datetime import datetime
from pathlib import Path
from typing import Iterable, List, Tuple

import pandas as pd

from .data_models import BenchmarkConfig, ExperimentConfig, ResultRecord
from .dataset_factory import create_dataset_from_config
from .experiment import EmotionExperiment

LOGGER = logging.getLogger(__name__)


def _load_manifest(run_dir: Path) -> dict:
    manifest_path = run_dir / "experiment_config.json"
    if not manifest_path.exists():
        LOGGER.warning(f"Cannot locate experiment_config.json in {run_dir}. Did you point to a valid run directory?")
        return FileNotFoundError(
            f"Cannot locate experiment_config.json in {run_dir}. Did you point to a valid run directory?"
        )
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def _reconstruct_configs(manifest: dict) -> Tuple[ExperimentConfig, BenchmarkConfig]:
    bench_payload = manifest.get("benchmark", {})
    data_path = bench_payload.get("data_path")
    benchmark_config = BenchmarkConfig(
        name=bench_payload["name"],
        task_type=bench_payload["task_type"],
        data_path=Path(data_path) if data_path else None,
        base_data_dir=bench_payload.get("base_data_dir"),
        sample_limit=bench_payload.get("sample_limit"),
        augmentation_config=bench_payload.get("augmentation_config"),
        enable_auto_truncation=bench_payload.get("enable_auto_truncation", False),
        truncation_strategy=bench_payload.get("truncation_strategy", "right"),
        preserve_ratio=bench_payload.get("preserve_ratio", 0.8),
        llm_eval_config=bench_payload.get("llm_eval_config"),
    )

    experiment_config = ExperimentConfig(
        model_path=manifest["model_path"],
        emotions=manifest["emotions"],
        intensities=manifest["intensities"],
        benchmark=benchmark_config,
        output_dir=manifest["output_dir"],
        batch_size=manifest["batch_size"],
        generation_config=manifest.get("generation_config"),
        loading_config=None,
        repe_eng_config=manifest.get("repe_eng_config"),
        max_evaluation_workers=manifest.get("max_evaluation_workers", 4),
        pipeline_queue_size=manifest.get("pipeline_queue_size", 2),
        defer_evaluation=bool(manifest.get("defer_evaluation", False)),
    )

    return experiment_config, benchmark_config


def _load_raw_results(run_dir: Path) -> List[dict]:
    raw_path = run_dir / "raw_results.json"
    if not raw_path.exists():
        raise FileNotFoundError(
            f"Missing raw_results.json in {run_dir}. Run the experiment with defer_evaluation enabled first."
        )
    return json.loads(raw_path.read_text(encoding="utf-8"))


def _update_raw_results(run_dir: Path, records: Iterable[ResultRecord]) -> None:
    raw_path = run_dir / "raw_results.json"
    with open(raw_path, "w") as fh:
        json.dump([record.__dict__ for record in records], fh, indent=2, default=str)


def evaluate_saved_run(run_dir: Path | str, max_workers: int = 8) -> pd.DataFrame:
    """Score a deferred experiment run and regenerate summary artifacts."""

    run_path = Path(run_dir)
    manifest = _load_manifest(run_path)
    if type(manifest) is not dict: return manifest
    experiment_config, benchmark_config = _reconstruct_configs(manifest)

    dataset = create_dataset_from_config(
        benchmark_config,
        prompt_wrapper=None,
        max_context_length=None,
        tokenizer=None,
        truncation_strategy=benchmark_config.truncation_strategy,
        answer_wrapper=None,
    )
    setattr(dataset, "eval_workers", max_workers)

    raw_rows = _load_raw_results(run_path)
    result_records: List[ResultRecord] = []

    for row in raw_rows:
        record = ResultRecord(
            emotion=row["emotion"],
            intensity=row["intensity"],
            item_id=row["item_id"],
            task_name=row["task_name"],
            prompt=row["prompt"],
            response=row["response"],
            ground_truth=row.get("ground_truth"),
            score=None,
            repeat_id=row.get("repeat_id", 0),
            metadata=row.get("metadata"),
            error=None,
        )
        result_records.append(record)

    total_items = len(result_records)
    if total_items == 0:
        raise ValueError(
            "No raw rows found in run directory; cannot evaluate. "
            f"Check {run_path}/raw_results.json was written and is non-empty."
        )

    chunk_size_attr = getattr(dataset, "offline_eval_chunk_size", None)
    if isinstance(chunk_size_attr, int) and chunk_size_attr > 0:
        chunk_size = chunk_size_attr
    else:
        configured = experiment_config.batch_size or max_workers or 32
        chunk_size = configured
    chunk_size = max(1, min(total_items, chunk_size)) if total_items else 1

    evaluated = 0
    for start in range(0, total_items, chunk_size):
        block = result_records[start : start + chunk_size]
        responses = [record.response for record in block]
        ground_truths = [record.ground_truth for record in block]
        task_names = [record.task_name for record in block]
        prompts = [record.prompt for record in block]

        try:
            scores = dataset.evaluate_batch(responses, ground_truths, task_names, prompts)
            eval_errors = getattr(dataset, "_last_eval_errors", None)
        except Exception as exc:  # pragma: no cover - defensive logging
            LOGGER.error(
                "Batch evaluation failed for chunk starting at %d: %s; falling back to per-item scoring",
                start,
                exc,
            )
            for record in block:
                try:
                    score = dataset.evaluate_response(
                        record.response,
                        record.ground_truth,
                        record.task_name,
                        record.prompt,
                    )
                except Exception as item_exc:  # pragma: no cover - defensive logging
                    LOGGER.error("Failed to evaluate response for %s: %s", record.item_id, item_exc)
                    record.score = None
                    record.error = str(item_exc)
                else:
                    if isinstance(score, float) and (math.isnan(score) or math.isinf(score)):
                        score = None
                    record.score = score
                    record.error = None
            evaluated += len(block)
            percent = 0.0 if total_items == 0 else (evaluated / total_items) * 100.0
            LOGGER.info(
                "[%s] Deferred evaluation progress: %d/%d (%.1f%%)",
                datetime.utcnow().isoformat(timespec="seconds") + "Z",
                evaluated,
                total_items,
                percent,
            )
            continue

        for idx, record in enumerate(block):
            score = None
            if idx < len(scores):
                score = scores[idx]
                if isinstance(score, float) and (math.isnan(score) or math.isinf(score)):
                    score = None
            record.score = score

            if eval_errors and idx < len(eval_errors):
                error_val = eval_errors[idx]
                record.error = str(error_val) if error_val is not None else None
            elif record.error:
                # Reset any lingering error when batch succeeded and dataset provided no details.
                record.error = None

        evaluated += len(block)
        percent = 0.0 if total_items == 0 else (evaluated / total_items) * 100.0
        LOGGER.info(
            "[%s] Deferred evaluation progress: %d/%d (%.1f%%)",
            datetime.utcnow().isoformat(timespec="seconds") + "Z",
            evaluated,
            total_items,
            percent,
        )

    _update_raw_results(run_path, result_records)

    error_count = sum(1 for record in result_records if record.error)

    # Reuse experiment saving utilities without loading vLLM
    stub_experiment = EmotionExperiment.__new__(EmotionExperiment)
    stub_experiment.output_dir = run_path
    stub_experiment.config = experiment_config
    stub_experiment.logger = LOGGER
    stub_experiment.dataset = dataset
    stub_experiment.generation_config = experiment_config.generation_config or {}
    stub_experiment.repe_config = experiment_config.repe_eng_config
    stub_experiment.loading_config = None
    stub_experiment.enable_thinking = False
    stub_experiment.max_context_length = None
    stub_experiment.hidden_layers = []
    stub_experiment.is_vllm = False
    stub_experiment.sample_num = experiment_config.benchmark.sample_limit
    stub_experiment.repeat_runs = 1
    stub_experiment.repeat_seed_base = None
    stub_experiment.defer_evaluation = False
    stub_experiment.truncation_strategy = experiment_config.benchmark.truncation_strategy

    df = stub_experiment._save_results(result_records)
    LOGGER.info(
        "[%s] Deferred evaluation completed for %s (items=%d, errors=%d)",
        datetime.utcnow().isoformat(timespec="seconds") + "Z",
        run_path,
        total_items,
        error_count,
    )
    return df


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate a deferred emotion experiment run in-place."
    )
    parser.add_argument("--input", required=True, help="Path to run output directory")
    parser.add_argument(
        "--max-workers",
        type=int,
        default=8,
        help="Maximum number of evaluation worker threads",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    evaluate_saved_run(args.input, max_workers=args.max_workers)


if __name__ == "__main__":
    main()
