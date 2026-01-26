"""
HumanEvalDataset (rewritten)

Modes
- task_type == "default": Original HumanEval problems and checker
- task_type == "plus": HumanEvalPlus (EvalPlus oracle; base + plus tests)
- task_type == "*": Combined view; emit two logical items per task (default & plus)

KISS: One dataset class branches by `config.task_type`. We lazy-import heavy
deps (human_eval, evalplus) and cache EvalPlus expected outputs once.
"""

import gzip
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional

if TYPE_CHECKING:
    from ..result_record import ResultRecord

from ..data_models import BenchmarkConfig, BenchmarkItem
from .base import BaseBenchmarkDataset


def _strip_code_fences(completion: str) -> str:
    """Remove leading/trailing ``` fences (optionally with language tags)."""
    if not completion:
        return completion

    trimmed = completion.strip()
    if not trimmed.startswith("```"):
        return completion

    body = trimmed[3:]
    newline_idx = body.find("\n")
    if newline_idx == -1:
        return ""
    # Drop optional language identifier (e.g. python)
    body = body[newline_idx + 1 :]

    body = body.rstrip()
    if body.endswith("```"):
        body = body[:-3]
    return body.strip("\r\n")


def _import_humaneval() -> Any:
    """Import upstream human_eval module, adding local path if needed.

    We try normal import; if it fails, add the known checkout path
    '/home/jjl7137/human-eval' to sys.path and retry.
    """
    try:
        import human_eval  # type: ignore
        return human_eval
    except Exception:
        he_path = "~/human-eval"
        if he_path not in sys.path:
            sys.path.insert(0, he_path)
        import human_eval  # type: ignore
        return human_eval


def _import_evalplus() -> Any:
    """Import local evalplus checkout if available."""
    try:
        import evalplus  # type: ignore
        return evalplus
    except Exception:
        ep_path = Path.home() / "evalplus"
        if ep_path.exists() and str(ep_path) not in sys.path:
            sys.path.insert(0, str(ep_path))
            try:
                import evalplus  # type: ignore
                return evalplus
            except Exception:
                pass
        raise RuntimeError(
            "EvalPlus is required for HumanEval plus-mode. Install it into the active env "
            "(e.g. `pip install -e /path/to/evalplus`) or export PYTHONPATH=/path/to/evalplus:$PYTHONPATH "
            "before running."
        )


@dataclass
class _EvalPlusState:
    problems: Optional[Dict[str, Any]] = None
    expected_outputs: Optional[Dict[str, Any]] = None
    dataset_hash: Optional[str] = None


class HumanEvalDataset(BaseBenchmarkDataset):
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
        # EvalPlus runtime knobs (used in plus / star modes)
        self.base_only = bool(kwargs.pop("base_only", False))
        self.test_details = bool(kwargs.pop("test_details", False))
        self.min_time_limit = float(kwargs.pop("min_time_limit", 2.0))
        self.gt_time_limit_factor = float(kwargs.pop("gt_time_limit_factor", 3.0))
        self._ep_state = _EvalPlusState()
        super().__init__(
            config,
            prompt_wrapper=prompt_wrapper,
            max_context_length=max_context_length,
            tokenizer=tokenizer,
            truncation_strategy=truncation_strategy,
            answer_wrapper=answer_wrapper,
        )

    def _ensure_evalplus_state(self, ground_truth: Dict[str, Any]) -> Dict[str, Any]:
        """Initialize cached EvalPlus state (problems, hash, expected outputs)."""

        # Ensure EvalPlus modules are importable before we rely on them.
        _import_evalplus()

        if self._ep_state.problems is None:
            problems: Dict[str, Any] = {}
            for item in self.items:
                gt = item.ground_truth
                if isinstance(gt, dict) and gt.get("__mode") == "plus":
                    problems[gt["task_id"]] = gt
            self._ep_state.problems = problems

        if (
            isinstance(ground_truth, dict)
            and ground_truth.get("__mode") == "plus"
            and ground_truth.get("task_id")
        ):
            task_id = ground_truth["task_id"]
            if self._ep_state.problems is None:
                self._ep_state.problems = {task_id: ground_truth}
            elif task_id not in self._ep_state.problems:
                self._ep_state.problems[task_id] = ground_truth

        if not self._ep_state.problems:
            raise ValueError("No HumanEval+ problems loaded for EvalPlus evaluation")

        if self._ep_state.dataset_hash is None:
            plus_path = self.config.get_data_path()
            import hashlib

            m = hashlib.md5()
            with open(plus_path, "rb") as f:
                m.update(f.read())
            self._ep_state.dataset_hash = m.hexdigest()

        if self._ep_state.expected_outputs is None:
            from evalplus.evaluate import get_groundtruth  # type: ignore
            from evalplus.data.utils import CACHE_DIR  # type: ignore

            expected_outputs = get_groundtruth(
                self._ep_state.problems,
                self._ep_state.dataset_hash,
                [],
            )

            if len(expected_outputs) != len(self._ep_state.problems):
                cache_file = Path(CACHE_DIR) / f"{self._ep_state.dataset_hash}.pkl"
                if cache_file.exists():
                    cache_file.unlink()
                expected_outputs = get_groundtruth(
                    self._ep_state.problems,
                    self._ep_state.dataset_hash,
                    [],
                )

            self._ep_state.expected_outputs = expected_outputs

        return self._ep_state.expected_outputs

    def _load_and_parse_data(self) -> List[BenchmarkItem]:
        task_mode = (self.config.task_type or "default").strip().lower()
        data_path = self.config.get_data_path()
        if not data_path.exists():
            raise FileNotFoundError(f"HumanEval data not found: {data_path}")

        if task_mode == "default":
            return self._load_humaneval_original(data_path)
        elif task_mode == "plus":
            return self._load_humaneval_plus(data_path, emit_star=False)
        elif task_mode == "*":
            return self._load_humaneval_plus(data_path, emit_star=True)
        else:
            raise ValueError(f"Unknown HumanEval task mode: {task_mode}")

    def _load_humaneval_original(self, data_path: Path) -> List[BenchmarkItem]:
        items: List[BenchmarkItem] = []
        # gz jsonl or plain jsonl
        def _iter_lines(fp):
            for line in fp:
                if line and line.strip():
                    yield json.loads(line)

        if str(data_path).endswith(".gz"):
            with gzip.open(str(data_path), "rt") as fp:
                for row in _iter_lines(fp):
                    row["__mode"] = "default"
                    items.append(
                        BenchmarkItem(
                            id=row["task_id"],
                            input_text=row["prompt"],
                            context=None,
                            ground_truth=row,  # contains "test" and "entry_point"
                            metadata={
                                "entry_point": row.get("entry_point"),
                                "source": "humaneval",
                                "mode": "default",
                            },
                        )
                    )
        else:
            with open(data_path, "r", encoding="utf-8") as fp:
                for row in _iter_lines(fp):
                    row["__mode"] = "default"
                    items.append(
                        BenchmarkItem(
                            id=row["task_id"],
                            input_text=row["prompt"],
                            context=None,
                            ground_truth=row,
                            metadata={
                                "entry_point": row.get("entry_point"),
                                "source": "humaneval",
                                "mode": "default",
                            },
                        )
                    )
        return items

    def _load_humaneval_plus(self, data_path: Path, emit_star: bool) -> List[BenchmarkItem]:
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
            plus_row = dict(row)
            plus_row["__mode"] = "plus"
            base_item = BenchmarkItem(
                id=plus_row["task_id"],
                input_text=plus_row["prompt"],
                context=None,
                ground_truth=plus_row,
                metadata={
                    "entry_point": plus_row.get("entry_point"),
                    "source": "humaneval+",
                    "mode": "plus",
                },
            )
            if emit_star:
                default_row = dict(row)
                default_row["__mode"] = "default"
                default_item = BenchmarkItem(
                    id=default_row["task_id"] + "::default",
                    input_text=default_row["prompt"],
                    context=None,
                    ground_truth=default_row,
                    metadata={
                        "entry_point": default_row.get("entry_point"),
                        "source": "humaneval+",
                        "mode": "default",
                    },
                )
                items.append(default_item)
            items.append(base_item)
        return items

    def evaluate_response(
        self, response: str, ground_truth: Any, task_name: str, prompt: str
    ) -> float:
        mode = None
        if isinstance(ground_truth, dict):
            mode = ground_truth.get("__mode")
        # Prefer config.task_type when not in star mode (unless overridden)
        task_mode = (self.config.task_type or "default").strip().lower()
        if mode is None:
            if task_mode == "*":
                mode = "plus" if "plus_input" in ground_truth else "default"
            else:
                mode = task_mode

        cleaned_response = _strip_code_fences(response)

        if mode == "default" and "test" in ground_truth:
            # Original HumanEval checker path
            _ = _import_humaneval()
            from human_eval.execution import check_correctness  # type: ignore

            result = check_correctness(
                problem=ground_truth,
                completion=cleaned_response,
                timeout=self.eval_timeout,
                completion_id=None,
            )
            return 1.0 if bool(result.get("passed")) else 0.0

        # Else: EvalPlus path (plus, or default-with-plus-records)
        _import_evalplus()
        from evalplus.evaluate import check_correctness as ep_check  # type: ignore

        expected_outputs = self._ensure_evalplus_state(ground_truth)
        task_id = ground_truth["task_id"]
        expected = expected_outputs[task_id]

        # Call evalplus checker; base_only determines whether to skip plus
        # For default mode, force base_only=True; for plus mode, base_only=False
        base_only = True if mode == "default" else False
        solution_text = ground_truth.get("prompt", "") + cleaned_response
        res = ep_check(
            dataset="humaneval",
            completion_id=0,
            problem=ground_truth,
            solution=solution_text,
            expected_output=expected,
            base_only=base_only,
            fast_check=not self.test_details,
            identifier=None,
            min_time_limit=self.min_time_limit,
            gt_time_limit_factor=self.gt_time_limit_factor,
        )
        # EvalPlus returns dict with keys base/plus -> (status, details)
        base_stat = res["base"][0]
        if base_only:
            return 1.0 if base_stat == "pass" else 0.0
        plus_stat = res.get("plus", (None, None))[0]
        return 1.0 if (base_stat == plus_stat == "pass") else 0.0

    def get_task_metrics(self, task_name: str) -> List[str]:
        return ["accuracy"]

    def compute_split_metrics(self, records: List["ResultRecord"]) -> Dict[str, float]:
        # Basic pass rate (pass@1)
        total = len(records)
        if total == 0:
            return {"pass_rate": 0.0}
        ok = 0
        for r in records:
            s = r.score
            try:
                ok += 1 if s and float(s) >= 1.0 else 0
            except Exception:
                pass
        return {"pass_rate": ok / float(total)}
