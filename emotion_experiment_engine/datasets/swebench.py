"""
Offline adapter for SWE-bench text datasets (HF save_to_disk).

This dataset reads pre-materialized prompts ("text_inputs") produced by
SWE-bench's create_text_dataset script and exposes them to the engine.

Generation-only: evaluation is deferred to the official SWE-bench harness.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from datasets import load_from_disk

from .base import BaseBenchmarkDataset
from ..data_models import BenchmarkConfig, BenchmarkItem


class SWEbenchDataset(BaseBenchmarkDataset):
    """
    Minimal dataset for SWE-bench generation parity.

    - Loads HF save_to_disk dataset from `config.data_path`
    - Expects columns: `instance_id` (str), `text_inputs` (str)
    - Returns prompts equal to `text_inputs` (no extra formatting)
    - Leaves evaluation to external SWE-bench harness (deferred)
    """

    def __init__(
        self,
        config: BenchmarkConfig,
        prompt_wrapper: Optional[Any] = None,  # not used; we pass through text_inputs
        max_context_length: Optional[int] = None,
        tokenizer: Any = None,
        truncation_strategy: str = "right",
        answer_wrapper: Optional[Any] = None,
    ) -> None:
        super().__init__(
            config=config,
            prompt_wrapper=None,  # ensure base class __getitem__ is not used
            max_context_length=max_context_length,
            tokenizer=tokenizer,
            truncation_strategy=truncation_strategy,
            answer_wrapper=answer_wrapper,
        )
        self._prediction_cache: Dict[Tuple[str, float, int], List[Dict[str, str]]] = defaultdict(list)
        self._prediction_paths: Dict[Tuple[str, float, int], Path] = {}

    # Required abstract methods
    def _load_and_parse_data(self) -> List[BenchmarkItem]:
        dp = self.config.data_path
        if dp is None:
            raise FileNotFoundError("SWE-bench requires config.data_path to point to HF save_to_disk dataset")
        path = Path(dp)
        if not path.exists():
            raise FileNotFoundError(f"SWE-bench dataset not found at {path}")

        ds = load_from_disk(str(path))

        # Handle Dataset or DatasetDict by selecting the first available split
        if hasattr(ds, "column_names"):
            dataset = ds
        else:
            # DatasetDict: pick one split deterministically
            first_split = next(iter(ds.keys()))
            dataset = ds[first_split]

        required = {"instance_id", "text_inputs"}
        missing = [c for c in required if c not in set(dataset.column_names)]
        if missing:
            raise ValueError(
                f"SWE-bench text dataset missing required columns: {missing}. "
                f"Found: {dataset.column_names}"
            )

        items: List[BenchmarkItem] = []
        for i in range(len(dataset)):
            row = dataset[i]
            inst_id = str(row["instance_id"]) if row.get("instance_id") is not None else str(i)
            prompt_text = str(row["text_inputs"]) if row.get("text_inputs") is not None else ""
            items.append(
                BenchmarkItem(
                    id=inst_id,
                    input_text=prompt_text,
                    context=None,
                    ground_truth=None,
                    metadata={
                        "benchmark": "swebench",
                        "instance_id": inst_id,
                    },
                )
            )
        return items

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        # Bypass parent prompt wrapping; return text_inputs directly as prompt
        item = self.items[idx]
        return {"item": item, "prompt": item.input_text, "ground_truth": None}

    def collate_fn(self, batch_items: List[Dict[str, Any]]) -> Dict[str, Any]:
        return {
            "prompts": [bi["prompt"] for bi in batch_items],
            "items": [bi["item"] for bi in batch_items],
            "ground_truths": [bi.get("ground_truth") for bi in batch_items],
        }

    def evaluate_response(
        self, response: str, ground_truth: Any, task_name: str, prompt: str | None = None
    ) -> float:
        # Generation-only; SWE-bench harness does the scoring
        return 0.0

    def get_task_metrics(self, task_name: str) -> List[str]:
        # No inline metrics in generation phase
        return []

    # Prediction capture helpers -------------------------------------------------

    @staticmethod
    def _run_key(emotion: str, intensity: float, repeat_id: int) -> Tuple[str, float, int]:
        return emotion.lower(), float(intensity), int(repeat_id)

    @staticmethod
    def _format_intensity_token(intensity: float) -> str:
        if float(intensity).is_integer():
            return f"{int(intensity):02d}"
        return str(intensity).replace("-", "neg").replace(".", "p")

    def _format_run_id(self, emotion: str, intensity: float, repeat_id: int) -> str:
        token = self._format_intensity_token(intensity)
        return f"{emotion.lower()}_i{token}_r{int(repeat_id):02d}"

    def record_model_patch(
        self,
        *,
        item_id: Union[str, int],
        model_patch: str,
        emotion: str,
        intensity: float,
        repeat_id: int,
        output_dir: Path,
    ) -> Tuple[Optional[Path], str]:
        patch = (model_patch or "").strip()
        run_key = self._run_key(emotion, intensity, repeat_id)
        run_id = self._format_run_id(emotion, intensity, repeat_id)
        path: Optional[Path] = None

        if patch:
            predictions_dir = Path(output_dir) / "predictions"
            predictions_dir.mkdir(parents=True, exist_ok=True)
            path = predictions_dir / f"{run_id}.jsonl"
            self._prediction_paths[run_key] = path
            self._prediction_cache[run_key].append(
                {
                    "instance_id": str(item_id),
                    "model_patch": patch,
                }
            )

        return path, run_id

    def flush_predictions(self, output_dir: Path) -> None:
        if not self._prediction_cache:
            return
        for run_key, records in list(self._prediction_cache.items()):
            if not records:
                continue
            path = self._prediction_paths.get(run_key)
            if path is None:
                continue
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("w", encoding="utf-8") as fp:
                for record in records:
                    fp.write(json.dumps(record, ensure_ascii=False))
                    fp.write("\n")
            self._prediction_cache[run_key] = []
