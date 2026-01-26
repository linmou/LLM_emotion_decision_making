"""
GPQADataset - Specialized dataset for GPQA multiple choice evaluation.

Parses GPQA CSV files (main/extended/diamond) and exposes items with:
- input_text: question text
- context: same as question
- ground_truth: list with the single correct answer
- metadata: includes 'options' (correct + 3 incorrect), domain info

Evaluation: strict text equality (case-insensitive) against the correct answer.
"""

import csv
from pathlib import Path
from typing import Any, List

from ..data_models import BenchmarkConfig, BenchmarkItem
from .base import BaseBenchmarkDataset


class GPQADataset(BaseBenchmarkDataset):
    """
    Dataset adapter for GPQA CSVs. Treats GPQA as single-answer MC (MC1-style).

    Supported task_type values (used for file naming only):
    - main, extended, diamond
    """

    REQUIRED_COLUMNS = [
        "Question",
        "Correct Answer",
        "Incorrect Answer 1",
        "Incorrect Answer 2",
        "Incorrect Answer 3",
    ]

    def __init__(
        self,
        config: BenchmarkConfig,
        prompt_wrapper,
        max_context_length=None,
        tokenizer=None,
        truncation_strategy: str = "right",
        answer_wrapper=None,
        shuffle_options_seed: int = None,
    ):
        # Capture optional parity control
        self.shuffle_options_seed = shuffle_options_seed
        self._rnd = None
        if shuffle_options_seed is not None:
            import random
            self._rnd = random.Random(shuffle_options_seed)
        super().__init__(
            config=config,
            prompt_wrapper=prompt_wrapper,
            max_context_length=max_context_length,
            tokenizer=tokenizer,
            truncation_strategy=truncation_strategy,
            answer_wrapper=answer_wrapper,
        )

    def _resolve_data_path(self) -> Path:
        # Prefer explicit data_path when provided
        if self.config.data_path is not None:
            return Path(self.config.data_path)

        # Otherwise infer from base_data_dir and task_type
        assert (
            self.config.base_data_dir is not None
        ), "base_data_dir is required when data_path is None"
        filename = f"gpqa_{self.config.task_type}.csv"
        return Path(self.config.base_data_dir) / filename

    def _load_and_parse_data(self) -> List[BenchmarkItem]:
        data_path = self._resolve_data_path()
        if not data_path.exists():
            raise FileNotFoundError(f"GPQA data file not found: {data_path}")

        items: List[BenchmarkItem] = []
        with open(data_path, "r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            # Validate required columns
            for col in self.REQUIRED_COLUMNS:
                if col not in (reader.fieldnames or []):
                    raise KeyError(f"Missing required column '{col}' in {data_path}")

            for idx, row in enumerate(reader):
                q = (row.get("Question") or "").strip()
                ca = (row.get("Correct Answer") or "").strip()
                i1 = (row.get("Incorrect Answer 1") or "").strip()
                i2 = (row.get("Incorrect Answer 2") or "").strip()
                i3 = (row.get("Incorrect Answer 3") or "").strip()

                if not q:
                    raise ValueError(f"Empty Question at row {idx+2} in {data_path}")
                if not ca:
                    raise ValueError(
                        f"Empty Correct Answer at row {idx+2} in {data_path}"
                    )

                options = [ca, i1, i2, i3]
                # Optional parity with GPQA repo: shuffle options with a fixed seed
                if self._rnd is not None:
                    # GPQA shuffles a list composed as [inc1, inc2, inc3, correct]
                    baseline_choices = [i1, i2, i3, ca]
                    self._rnd.shuffle(baseline_choices)
                    options = list(baseline_choices)
                record_id = row.get("Record ID") or f"gpqa_{idx}"
                subdomain = row.get("Subdomain")
                high_domain = row.get("High-level domain") or row.get(
                    "High-level Domain"
                )

                items.append(
                    BenchmarkItem(
                        id=record_id,
                        context=q,
                        input_text=q,
                        ground_truth=[ca],
                        metadata={
                            "options": options,
                            "subdomain": subdomain,
                            "high_level_domain": high_domain,
                            "task_type": self.config.task_type,
                        },
                    )
                )

        if not items:
            raise ValueError(f"No valid items found in {data_path}")

        return items

    def evaluate_response(
        self, response: str, ground_truth: Any, task_name: str, prompt: str
    ) -> float:
        if not isinstance(ground_truth, list) or len(ground_truth) != 1:
            raise TypeError(
                f"ground_truth must be List[str] of length 1, got {ground_truth}"
            )

        return 1.0 if (response or "").strip().lower() == ground_truth[0].strip().lower() else 0.0

    def get_task_metrics(self, task_name: str) -> List[str]:
        return ["accuracy"]
