"""
MBPPDataset

Modes
- task_type == "default": Base tests only (EvalPlus MbppPlus with base_only)
- task_type == "plus": Full MbppPlus (base + extra tests)
- task_type == "*": Combined view emitting both default & plus logical items

Offline-first: requires local MbppPlus.jsonl.gz path. We rely on EvalPlus for
parsing/parity and evaluation (base_only toggles behaviour).
"""

import gzip
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from ..data_models import BenchmarkConfig, BenchmarkItem
from .base import BaseBenchmarkDataset


def _strip_code_fences(completion: str) -> str:
    if not completion:
        return completion
    trimmed = completion.strip()
    if not trimmed.startswith("```"):
        return completion
    body = trimmed[3:]
    newline_idx = body.find("\n")
    if newline_idx == -1:
        return ""
    body = body[newline_idx + 1 :]
    body = body.rstrip()
    if body.endswith("```"):
        body = body[:-3]
    return body.strip("\r\n")


def _import_evalplus() -> Any:
    try:
        import evalplus  # type: ignore
        return evalplus
    except Exception:
        ep_path = "/data/home/jjl7137/evalplus"
        if ep_path not in sys.path:
            sys.path.insert(0, ep_path)
        import evalplus  # type: ignore
        return evalplus


class MBPPDataset(BaseBenchmarkDataset):
    def __init__(
        self,
        config: BenchmarkConfig,
        prompt_wrapper: Optional[Callable],
        max_context_length: Optional[int] = None,
        tokenizer: Any = None,
        truncation_strategy: str = "right",
        answer_wrapper: Optional[Callable] = None,
        eval_timeout: float = 3.0,
        **kwargs: Any,
    ):
        self.eval_timeout = float(eval_timeout)
        self.test_details = bool(kwargs.pop("test_details", False))
        self.min_time_limit = float(kwargs.pop("min_time_limit", 2.0))
        self.gt_time_limit_factor = float(kwargs.pop("gt_time_limit_factor", 3.0))
        self._expected_outputs: Optional[Dict[str, Any]] = None
        self._dataset_hash: Optional[str] = None
        super().__init__(
            config,
            prompt_wrapper=prompt_wrapper,
            max_context_length=max_context_length,
            tokenizer=tokenizer,
            truncation_strategy=truncation_strategy,
            answer_wrapper=answer_wrapper,
        )

    def _load_and_parse_data(self) -> List[BenchmarkItem]:
        mode = (self.config.task_type or "default").strip().lower()
        data_path = self.config.get_data_path()
        if not data_path.exists():
            raise FileNotFoundError(f"MBPP data not found: {data_path}")

        items: List[BenchmarkItem] = []

        def _iter_rows():
            if str(data_path).endswith(".gz"):
                fp = gzip.open(str(data_path), "rt")
            else:
                fp = open(data_path, "r", encoding="utf-8")
            with fp:
                for line in fp:
                    if not line.strip():
                        continue
                    yield json.loads(line)

        for row in _iter_rows():
            if mode in ("default", "*"):
                default_row = dict(row)
                default_row["__mode"] = "default"
                items.append(
                    BenchmarkItem(
                        id=str(row["task_id"]) + ("::default" if mode == "*" else ""),
                        input_text=row["prompt"],
                        context=None,
                        ground_truth=default_row,
                        metadata={
                            "entry_point": row.get("entry_point"),
                            "source": "mbpp+",
                            "mode": "default",
                        },
                    )
                )
            if mode in ("plus", "*"):
                plus_row = dict(row)
                plus_row["__mode"] = "plus"
                items.append(
                    BenchmarkItem(
                        id=str(row["task_id"]),
                        input_text=row["prompt"],
                        context=None,
                        ground_truth=plus_row,
                        metadata={
                            "entry_point": row.get("entry_point"),
                            "source": "mbpp+",
                            "mode": "plus",
                        },
                    )
                )
        if not items:
            raise ValueError("No MBPP items loaded; ensure MbppPlus jsonl.gz path is correct")
        return items

    def _ensure_expected_outputs(self) -> None:
        if self._expected_outputs is not None:
            return
        _import_evalplus()
        from evalplus.evaluate import get_groundtruth  # type: ignore
        from evalplus.eval._special_oracle import MBPP_OUTPUT_NOT_NONE_TASKS  # type: ignore

        problems: Dict[str, Any] = {}
        for item in self.items:
            gt = item.ground_truth
            if isinstance(gt, dict):
                problems[str(gt["task_id"])] = gt

        if self._dataset_hash is None:
            m = hashlib.md5()
            with open(self.config.get_data_path(), "rb") as f:
                m.update(f.read())
            self._dataset_hash = m.hexdigest()

        exp_all = get_groundtruth(problems, self._dataset_hash, MBPP_OUTPUT_NOT_NONE_TASKS)

        if len(exp_all) != len(problems):
            from evalplus.data.utils import CACHE_DIR  # type: ignore

            cache_file = Path(CACHE_DIR) / f"{self._dataset_hash}.pkl"
            if cache_file.exists():
                cache_file.unlink()
            exp_all = get_groundtruth(
                problems, self._dataset_hash, MBPP_OUTPUT_NOT_NONE_TASKS
            )

        self._expected_outputs = exp_all

    def evaluate_response(
        self, response: str, ground_truth: Any, task_name: str, prompt: str
    ) -> float:
        mode = ground_truth.get("__mode", (self.config.task_type or "default").strip().lower())

        try:
            _import_evalplus()
            from evalplus.evaluate import check_correctness as ep_check  # type: ignore
        except ModuleNotFoundError as exc:  # pragma: no cover - depends on optional deps
            raise RuntimeError(
                "evalplus requires optional dependency 'tree_sitter_python'. "
                "Install evalplus[test] or tree_sitter_python to enable MBPP evaluation."
            ) from exc

        self._ensure_expected_outputs()

        cleaned = _strip_code_fences(response)
        base_only = True if mode == "default" else False
        res = ep_check(
            dataset="mbpp",
            completion_id=0,
            problem=ground_truth,
            solution=cleaned,
            expected_output=self._expected_outputs[str(ground_truth["task_id"])],
            base_only=base_only,
            fast_check=not self.test_details,
            identifier=None,
            min_time_limit=self.min_time_limit,
            gt_time_limit_factor=self.gt_time_limit_factor,
        )
        base_stat = res["base"][0]
        if base_only:
            return 1.0 if base_stat == "pass" else 0.0
        plus_stat = res.get("plus", (None, None))[0]
        return 1.0 if (base_stat == plus_stat == "pass") else 0.0

    def get_task_metrics(self, task_name: str) -> List[str]:
        return ["accuracy"]

    def compute_split_metrics(self, records: List["ResultRecord"]) -> Dict[str, float]:
        total = len(records)
        if total == 0:
            return {"pass_rate": 0.0}
        ok = 0
        for r in records:
            try:
                ok += 1 if r.score and float(r.score) >= 1.0 else 0
            except Exception:
                pass
        return {"pass_rate": ok / float(total)}
