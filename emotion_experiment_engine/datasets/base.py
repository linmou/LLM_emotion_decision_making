"""
Abstract base class for all benchmark datasets.
Provides common functionality while enforcing specialized implementation requirements.
"""

import json
from abc import ABC, abstractmethod
from copy import deepcopy
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

# NOTE: Do not import torch at module import time.
# Some environments ship a torch build that aborts the interpreter on import.
# Datasets only need the duck-typed Dataset interface (__len__/__getitem__).
class Dataset:  # type: ignore
    pass

from ..data_models import BenchmarkConfig, BenchmarkItem


class BaseBenchmarkDataset(Dataset, ABC):
    """
    Abstract base class for all benchmark datasets.

    Provides common functionality:
    - Data loading infrastructure
    - Context truncation
    - PyTorch Dataset interface (__len__, __getitem__)
    - Batch collation

    Requires specialized implementation:
    - _load_and_parse_data(): Benchmark-specific data parsing
    - evaluate_response(): Benchmark-specific evaluation logic
    - get_task_metrics(): Available metrics for tasks
    """

    LLM_EVAL_CONFIG = {
        "model": "gpt-4o-mini",
        "temperature": 0.0,
    }

    def __init__(
        self,
        config: BenchmarkConfig,
        prompt_wrapper: Optional[Callable],
        max_context_length: Optional[int] = None,
        tokenizer: Any = None,
        truncation_strategy: str = "right",
        answer_wrapper: Optional[Callable] = None,
    ):
        self.config = config
        self.prompt_wrapper = prompt_wrapper
        self.max_context_length = max_context_length
        self.tokenizer = tokenizer
        self.truncation_strategy = truncation_strategy
        self.answer_wrapper = answer_wrapper
        if hasattr(self, "LLM_EVAL_CONFIG"):
            self.llm_eval_config = deepcopy(self.LLM_EVAL_CONFIG)
            if self.config.llm_eval_config:
                self.llm_eval_config.update(self.config.llm_eval_config)
        else:
            self.llm_eval_config = self.config.llm_eval_config

        # Load data using child class implementation
        self.items: List[BenchmarkItem] = self._load_and_parse_data()

        # Apply sample limit with random shuffling if specified
        if self.config.sample_limit:
            self.items = self.items[: self.config.sample_limit]

        # Apply truncation if parameters provided
        if self.max_context_length and self.tokenizer:
            self.items = self._apply_truncation(self.items)

    @abstractmethod
    def _load_and_parse_data(self) -> List[BenchmarkItem]:
        """
        Load and parse data for this specific benchmark type.
        Child classes must implement their own data loading logic.

        Returns:
            List of BenchmarkItem objects
        """
        pass

    @abstractmethod
    def evaluate_response(
        self, response: str, ground_truth: Any, task_name: str, prompt: str
    ) -> float:
        """
        Evaluate a response against ground truth for this benchmark.
        Child classes implement benchmark-specific evaluation logic.

        Args:
            response: Model's response text
            ground_truth: Expected answer(s)
            task_name: Specific task type within benchmark

        Returns:
            Score between 0.0 and 1.0
        """
        pass

    @abstractmethod
    def get_task_metrics(self, task_name: str) -> List[str]:
        """
        Get available metrics for a specific task type.

        Args:
            task_name: Task type within this benchmark

        Returns:
            List of available metric names (e.g., ["accuracy", "f1_score"])
        """
        pass

    # Split-level aggregation hook (optional)
    def compute_split_metrics(self, records: List["ResultRecord"]) -> Dict[str, float]:
        """
        Compute dataset-scope (split-level) metrics from item-level results.

        Default implementation returns an empty mapping so existing benchmarks
        are unaffected. Concrete TrustLLM datasets will override this to produce
        section-specific aggregates (e.g., RtA, Pearson, macro-F1).

        Args:
            records: List of ResultRecord objects produced by the experiment

        Returns:
            Mapping of metric_name -> value (floats preferred)
        """
        return {}

    # Common functionality implementation

    def __len__(self) -> int:
        """Return dataset size"""
        return len(self.items)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get item by index with prompt formatting"""
        item = self.items[idx]

        # Extract options from metadata for multiple choice questions
        options = None
        if item.metadata and "options" in item.metadata:
            options = item.metadata["options"]

        # Create prompt using wrapper or default format
        if self.prompt_wrapper:
            prompt = self.prompt_wrapper(
                context=item.context if item.context else "",
                question=item.input_text,
                answer=item.ground_truth,
                options=options,
            )

        else:
            # Default prompt format
            if item.context:
                prompt = (
                    f"Context: {item.context}\nQuestion: {item.input_text}\nAnswer:"
                )
            else:
                prompt = f"{item.input_text}\nAnswer:"

        # Apply answer wrapper if provided to adapt ground truth
        adapted_ground_truth = (
            self.answer_wrapper(item.ground_truth)
            if self.answer_wrapper is not None
            else item.ground_truth
        )

        return {"item": item, "prompt": prompt, "ground_truth": adapted_ground_truth}

    def collate_fn(self, batch_items: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Collate function for DataLoader"""
        return {
            "prompts": [item["prompt"] for item in batch_items],
            "items": [item["item"] for item in batch_items],
            "ground_truths": [item["ground_truth"] for item in batch_items],
        }


    def _apply_truncation(self, items: List[BenchmarkItem]) -> List[BenchmarkItem]:
        """
        Faster truncation:
        - Uses return_offsets_mapping to slice the original string (no batch_decode).
        - Only constructs truncated strings for items that actually exceed the limit.
        - Falls back to batch_decode if the tokenizer isn't "fast".
        """
        batch_size = getattr(self, "batch_size", 32)
        self.tokenizer.truncation_side = self.truncation_strategy  # "left" or "right"
        max_len = self.max_context_length
        use_fast = getattr(self.tokenizer, "is_fast", False)

        all_truncated_items: List[BenchmarkItem] = []

        for i in range(0, len(items), batch_size):
            batch_items = items[i : i + batch_size]

            # Collect contexts to process (preserve indices to rebuild)
            idxs, contexts = [], []
            for j, it in enumerate(batch_items):
                if it.context is not None:
                    idxs.append(j)
                    contexts.append(it.context or "")

            truncated_texts = []
            if contexts:
                if use_fast:
                    # Single pass: get full offsets; decide slice by strategy
                    enc = self.tokenizer(
                        contexts,
                        add_special_tokens=False,
                        padding=False,
                        truncation=False,  # get full tokenization (no trunc yet)
                        return_offsets_mapping=True,
                        return_attention_mask=False,
                        return_token_type_ids=False,
                    )
                    offsets_batch = enc["offset_mapping"]

                    if self.truncation_strategy == "right":
                        for text, offsets in zip(contexts, offsets_batch):
                            if len(offsets) <= max_len:
                                truncated_texts.append(text)
                                continue
                            # first max_len tokens
                            window = offsets[:max_len]
                            # find the furthest valid end char
                            end_char = 0
                            for s, e in window:
                                if e and e > end_char:
                                    end_char = e
                            truncated_texts.append(text[:end_char])
                    else:  # "left"
                        for text, offsets in zip(contexts, offsets_batch):
                            if len(offsets) <= max_len:
                                truncated_texts.append(text)
                                continue
                            # last max_len tokens
                            window = offsets[-max_len:]
                            # find the earliest valid start char in the kept window
                            start_char = None
                            for s, e in window:
                                if s is not None and (start_char is None or s < start_char):
                                    start_char = s
                            truncated_texts.append(text[start_char:] if start_char else text)
                else:
                    # Fallback for slow tokenizers (no offsets available)
                    enc = self.tokenizer(
                        contexts,
                        add_special_tokens=False,
                        truncation=True,
                        max_length=max_len,
                        return_tensors=None,
                        padding=False,
                    )
                    truncated_texts = self.tokenizer.batch_decode(
                        enc["input_ids"], skip_special_tokens=True
                    )

            # Rebuild results for this batch
            k = 0
            for j, item in enumerate(batch_items):
                if item.context is None:
                    all_truncated_items.append(
                        BenchmarkItem(
                            id=item.id,
                            context=None,
                            input_text=item.input_text,
                            ground_truth=item.ground_truth,
                            metadata=deepcopy(item.metadata) if item.metadata else None,
                        )
                    )
                    continue

                original_context = item.context or ""
                truncated_context = truncated_texts[k]
                k += 1

                md = deepcopy(item.metadata) if item.metadata else {}
                md["truncation_info"] = {
                    "original_length": len(original_context),
                    "truncated_length": len(truncated_context),
                    "strategy": self.truncation_strategy,
                    "was_truncated": len(truncated_context) < len(original_context),
                }

                all_truncated_items.append(
                    BenchmarkItem(
                        id=item.id,
                        context=truncated_context,
                        input_text=item.input_text,
                        ground_truth=item.ground_truth,
                        metadata=md,
                    )
                )

        return all_truncated_items


    def _load_raw_data(self) -> List[Dict[str, Any]]:
        """
        Load raw JSON data from file.
        Common loading logic used by all benchmark types.
        """
        data_path = self.config.get_data_path()
        if not data_path.exists():
            raise FileNotFoundError(f"Benchmark data not found: {data_path}")

        # Load based on file extension
        if data_path.suffix == ".jsonl":
            with open(data_path, "r", encoding="utf-8") as f:
                raw_data = [json.loads(line) for line in f if line.strip()]
        else:
            with open(data_path, "r", encoding="utf-8") as f:
                raw_data = json.load(f)

        return raw_data

    def evaluate_batch(
        self,
        responses: List[str],
        ground_truths: List[Any],
        task_names: List[str],
        prompts: List[str],
    ) -> List[float]:
        """
        ThreadPoolExecutor-based batch evaluation using individual evaluate_response calls.

        Captures per-item errors and exposes them via `self._last_eval_errors` so callers
        can persist error details alongside scores.
        """
        from concurrent.futures import ThreadPoolExecutor

        self._last_eval_errors = [None] * len(responses)  # type: ignore[attr-defined]
        scores: List[float] = []
        # Allow datasets/experiment to control evaluation parallelism.
        # Default to 8 workers if not explicitly provided.
        max_workers = getattr(self, "eval_workers", 64)
        max_workers = max(1, min(int(max_workers), len(responses)))
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(self.evaluate_response, resp, gt, task, prompt)
                for resp, gt, task, prompt in zip(
                    responses, ground_truths, task_names, prompts
                )
            ]
            for i, future in enumerate(futures):
                try:
                    scores.append(future.result())
                except Exception as e:  # Capture per-item error and continue
                    self._last_eval_errors[i] = str(e)  # type: ignore[index]
                    scores.append(float("nan"))
        return scores

    def evaluate_with_detailed_metrics(
        self, response: str, ground_truth: Any, task_name: str
    ) -> Dict[str, float]:
        """
        Return detailed metrics for evaluation.
        Default implementation returns basic score - child classes can extend.
        """
        base_score = self.evaluate_response(response, ground_truth, task_name)
        return {"overall_score": base_score, "accuracy": base_score}
