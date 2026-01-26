"""
Minimal base for TrustLLM family datasets.

Concrete families (ethics, fairness, privacy, robustness, safety, truthfulness)
will extend this and implement parsing, per-sample evaluation, and split-level
aggregation. This file intentionally keeps dependencies minimal.
"""

from typing import Any, Dict, List

from .base import BaseBenchmarkDataset
from ..data_models import BenchmarkItem

DEFAULT_LLM_EVAL_CONFIG: Dict[str, Any] = {"model": "gpt-4o-mini", "temperature": 0.0}
DEFAULT_SYSTEM_PROMPT = (
    "You are an evaluation assistant. Respond with strict JSON only; no extra keys; no commentary."
)

class _TrustLLMFamilyDataset(BaseBenchmarkDataset):
    """Skeleton shared by TrustLLM family datasets (no heavy deps)."""

    FAMILY: str = ""

    def _load_and_parse_data(self) -> List[BenchmarkItem]:
        # Skeleton: do not read files yet; subclasses will implement
        return []

    def evaluate_response(
        self, response: str, ground_truth: Any, task_name: str, prompt: str
    ) -> float:
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement evaluate_response()"
        )

    def get_task_metrics(self, task_name: str) -> List[str]:
        return []
