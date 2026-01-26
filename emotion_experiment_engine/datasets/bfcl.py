"""
BFCLDataset - Specialized dataset for Berkeley Function Call Leaderboard style tasks.

Supports initial categories:
- live_simple: single function call
- live_multiple: multiple function calls

Data format (JSONL per line):
- id: string
- question: list[{role, content}]  (single-turn for live_simple; may generalize)
- function: {name, parameters{JSON-Schema}}  OR  functions: [ ... ]
- ground_truth: list of {func_name: {param: [allowed_values...]}} (inline to fit our interface)

This dataset constructs context by embedding the tool schema JSON directly,
so the prompt wrapper can instruct JSON-only function calls with the schema included.
"""

import json
import re
from typing import Any, Dict, List, Optional, Tuple

from ..data_models import BenchmarkConfig, BenchmarkItem
from .base import BaseBenchmarkDataset


class BFCLDataset(BaseBenchmarkDataset):
    """Dataset for BFCL-style tool/function-calling tasks."""

    SUPPORTED_TASKS = {"live_simple", "live_multiple"}

    def _load_and_parse_data(self) -> List[BenchmarkItem]:
        raw = self._load_raw_data()
        items: List[BenchmarkItem] = []

        for i, row in enumerate(raw):
            item_id = row.get("id", i)

            # Extract question text from chat turns
            # Support both [ {dict}, ... ] and [[ {dict}, ... ], ...] formats
            question_turns = row.get("question", [])
            flat_turns: List[Dict[str, Any]] = []
            for t in question_turns:
                if isinstance(t, dict):
                    flat_turns.append(t)
                elif isinstance(t, list):
                    for tt in t:
                        if isinstance(tt, dict):
                            flat_turns.append(tt)
            question_text = " ".join([t.get("content", "") for t in flat_turns]).strip()

            # Normalize tool schema to a list for context embedding
            tools: Optional[List[Dict[str, Any]]] = None
            if "functions" in row and isinstance(row["functions"], list):
                tools = row["functions"]
            elif "function" in row and isinstance(row["function"], dict):
                tools = [row["function"]]
            else:
                tools = []

            # Embed tool schema JSON into context for the prompt wrapper
            context = json.dumps({"tools": tools}, ensure_ascii=False)

            ground_truth = row.get("ground_truth")

            items.append(
                BenchmarkItem(
                    id=item_id,
                    input_text=question_text,
                    context=context,
                    ground_truth=ground_truth,
                    metadata={"tools": tools, "raw": row},
                )
            )

        return items

    # ------------------------
    # Evaluation (AST-based)
    # ------------------------

    @staticmethod
    def _strip_code_fences(text: str) -> str:
        return re.sub(r"^\s*```[\w]*\n|\n```\s*$", "", text.strip())

    @staticmethod
    def _extract_first_json_array(text: str) -> Optional[List[Any]]:
        """Try to extract the first top-level JSON array from text."""
        text = BFCLDataset._strip_code_fences(text)
        # Fast path
        try:
            obj = json.loads(text)
            if isinstance(obj, dict):
                return [obj]
            if isinstance(obj, list):
                return obj
        except Exception:
            pass

        # Fallback: find the first [ ... ] block
        start = text.find("[")
        end = text.rfind("]")
        if start != -1 and end != -1 and end > start:
            candidate = text[start : end + 1]
            try:
                obj = json.loads(candidate)
                if isinstance(obj, dict):
                    return [obj]
                if isinstance(obj, list):
                    return obj
            except Exception:
                return None
        return None

    @staticmethod
    def _normalize_string(s: str) -> str:
        s = s.lower().strip()
        # Collapse whitespace
        s = re.sub(r"\s+", " ", s)
        # Remove punctuation
        s = re.sub(r"[\.,!?:;\-_'\"\(\)\[\]\{\}]", "", s)
        return s

    @classmethod
    def _normalize_value(cls, v: Any) -> Any:
        if isinstance(v, str):
            return cls._normalize_string(v)
        return v

    @classmethod
    def _value_matches_allowed(cls, value: Any, allowed_list: List[Any]) -> bool:
        # Ignore "" which encodes optional omission
        cleaned_allowed = [a for a in allowed_list if not (isinstance(a, str) and a == "")]
        if not cleaned_allowed:
            # If only optional marker present, any present value is acceptable
            return True

        nv = cls._normalize_value(value)
        for a in cleaned_allowed:
            if cls._normalize_value(a) == nv:
                return True
        return False

    @classmethod
    def _check_single_call(
        cls, call_obj: Dict[str, Any], gt_obj: Dict[str, Any]
    ) -> Tuple[bool, str]:
        # call_obj: { func_name: { param: value, ... } }
        # gt_obj: { func_name: { param: [allowed_values...] } }
        if not isinstance(call_obj, dict) or len(call_obj) != 1:
            return False, "call must be dict with single function"

        call_func = next(iter(call_obj.keys()))
        call_args = call_obj[call_func]
        if not isinstance(call_args, dict):
            return False, "function args must be object"

        if not isinstance(gt_obj, dict) or len(gt_obj) != 1:
            return False, "invalid ground truth format"

        exp_func = next(iter(gt_obj.keys()))
        exp_params: Dict[str, List[Any]] = gt_obj[exp_func]
        if not isinstance(exp_params, dict):
            return False, "invalid ground truth params"

        # Function name match (case-insensitive)
        if cls._normalize_string(call_func) != cls._normalize_string(exp_func):
            return False, f"function mismatch: {call_func} != {exp_func}"

        # No extra params beyond those specified in GT
        allowed_keys = set(exp_params.keys())
        extra = set(call_args.keys()) - allowed_keys
        if extra:
            return False, f"extra params present: {sorted(extra)}"

        # Check each expected param
        for p, allowed in exp_params.items():
            if not isinstance(allowed, list):
                return False, f"ground truth for param {p} must be list"

            if p not in call_args:
                # Missing parameter is ok only if "" allowed (optional)
                if "" in allowed:
                    continue
                else:
                    return False, f"missing required param: {p}"

            # Present -> must match allowed values
            if not cls._value_matches_allowed(call_args[p], allowed):
                return False, f"param {p} value not in allowed set"

        return True, "ok"

    def evaluate_response(
        self, response: str, ground_truth: Any, task_name: str, prompt: str
    ) -> float:
        arr = self._extract_first_json_array(response)
        if arr is None:
            return 0.0

        # Determine single vs multiple from ground_truth if not via task_name
        is_multiple = False
        task_lower = (task_name or self.config.task_type or "").lower()
        if task_lower == "live_multiple":
            is_multiple = True
        elif task_lower == "live_simple":
            is_multiple = False
        else:
            # Fallback: infer from ground truth length
            try:
                is_multiple = isinstance(ground_truth, list) and len(ground_truth) > 1
            except Exception:
                is_multiple = False

        try:
            if not isinstance(ground_truth, list) or not ground_truth:
                return 0.0

            if not is_multiple:
                # Expect exactly one call
                if len(arr) != 1:
                    return 0.0
                ok, _ = self._check_single_call(arr[0], ground_truth[0])
                return 1.0 if ok else 0.0

            # Multiple calls: order-sensitive matching against GT list
            if len(arr) != len(ground_truth):
                return 0.0
            for call_obj, gt_obj in zip(arr, ground_truth):
                ok, _ = self._check_single_call(call_obj, gt_obj)
                if not ok:
                    return 0.0
            return 1.0
        except Exception:
            return 0.0

    def get_task_metrics(self, task_name: str) -> List[str]:
        return ["accuracy"]
