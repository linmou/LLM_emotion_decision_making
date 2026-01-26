"""
Responsible file: emotion_experiment_engine/datasets/swebench.py
Purpose: Validate offline adapter behavior without invoking heavy HF dependencies.
"""

import json
from pathlib import Path
from typing import Dict, List

import pytest

from emotion_experiment_engine.data_models import BenchmarkConfig
from emotion_experiment_engine.benchmark_component_registry import (
    create_benchmark_components,
)


class _FakeDataset:
    def __init__(self, rows: List[Dict[str, str]]):
        self._rows = rows
        self.column_names = list(rows[0].keys()) if rows else []

    def __len__(self) -> int:
        return len(self._rows)

    def __getitem__(self, idx: int) -> Dict[str, str]:
        return self._rows[idx]


def _make_hf_disk_dataset(tmp_dir: Path, name: str, n: int = 5) -> tuple[Path, List[Dict[str, str]]]:
    base = tmp_dir / "datasets" / name
    base.mkdir(parents=True, exist_ok=True)
    records = [
        {"instance_id": f"inst_{i}", "text_inputs": f"<STYLE-3 PROMPT #{i}>"}
        for i in range(n)
    ]
    with open(base / "data.json", "w", encoding="utf-8") as fp:
        json.dump(records, fp)
    return base, records


def _bench_config(data_dir: Path) -> BenchmarkConfig:
    # Fill all required fields explicitly (no dataclass defaults by policy)
    return BenchmarkConfig(
        name="swebench",
        task_type="patch",
        data_path=data_dir,  # points to HF save_to_disk dataset directory
        base_data_dir=None,
        sample_limit=None,
        augmentation_config=None,
        enable_auto_truncation=False,
        truncation_strategy="right",
        preserve_ratio=1.0,
        llm_eval_config=None,
    )


@pytest.mark.order(1)
def test_swebench_dataset_basic_load(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    dataset_path, records = _make_hf_disk_dataset(tmp_path, name="swebench_dummy", n=7)
    config = _bench_config(dataset_path)

    monkeypatch.setattr(
        "emotion_experiment_engine.datasets.swebench.load_from_disk",
        lambda p: _FakeDataset(records),
    )

    # Act: build dataset via registry
    prompt_wrapper, answer_wrapper, dataset = create_benchmark_components(
        benchmark_name="swebench",
        task_type="patch",
        config=config,
        prompt_format=None,
    )

    # Assert: length and item structure
    assert len(dataset) == 7
    item = dataset.__getitem__(0)
    assert isinstance(item, dict)
    assert set(item.keys()) >= {"item", "prompt", "ground_truth"}
    assert isinstance(item["prompt"], str)


@pytest.mark.order(2)
def test_swebench_collate_and_parity(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    dataset_path, records = _make_hf_disk_dataset(tmp_path, name="swebench_parity", n=5)
    config = _bench_config(dataset_path)

    monkeypatch.setattr(
        "emotion_experiment_engine.datasets.swebench.load_from_disk",
        lambda p: _FakeDataset(records),
    )

    _, _, dataset = create_benchmark_components(
        benchmark_name="swebench",
        task_type="patch",
        config=config,
        prompt_format=None,
    )

    # Collect first 5 prompts via __getitem__ and collate
    batch_items = [dataset.__getitem__(i) for i in range(5)]
    batch = dataset.collate_fn(batch_items)
    assert "prompts" in batch and isinstance(batch["prompts"], list)
    assert len(batch["prompts"]) == 5

    expected = [row["text_inputs"] for row in records[:5]]
    assert batch["prompts"] == expected
