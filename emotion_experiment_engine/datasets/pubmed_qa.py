"""
PubMedQADataset

Load PubMedQA (pqa_labeled) via HuggingFace datasets and expose a simple
yes/no/maybe classification task under the standardized BenchmarkItem schema.

Dataset loading:
    ds = load_dataset("pubmed_qa", "pqa_labeled")
    split chosen via ctor kwarg 'split' (default: 'test')

Ground truth:
    Prefer 'final_decision' if available, else fall back to 'label'.

Evaluation:
    Exact normalized match on {yes,no,maybe} (case-insensitive, stripped).
"""

from typing import Any, Dict, List, Optional, Tuple

try:
    from datasets import load_dataset  # type: ignore
except Exception:  # pragma: no cover - tests won't execute loading
    load_dataset = None  # type: ignore

from ..data_models import BenchmarkConfig, BenchmarkItem
from .base import BaseBenchmarkDataset


class PubMedQADataset(BaseBenchmarkDataset):
    def __init__(
        self,
        config: BenchmarkConfig,
        prompt_wrapper: Optional[Any],
        max_context_length: Optional[int] = None,
        tokenizer: Any = None,
        truncation_strategy: str = "right",
        answer_wrapper: Optional[Any] = None,
        split: str = "test",
        **kwargs: Any,
    ) -> None:
        self.split = split or "test"
        super().__init__(
            config,
            prompt_wrapper=prompt_wrapper,
            max_context_length=max_context_length,
            tokenizer=tokenizer,
            truncation_strategy=truncation_strategy,
            answer_wrapper=answer_wrapper,
        )

    def _load_and_parse_data(self) -> List[BenchmarkItem]:
        if load_dataset is None:
            raise RuntimeError(
                "datasets library not available; install 'datasets' to use PubMedQADataset"
            )

        ds = load_dataset("pubmed_qa", "pqa_labeled")
        if self.split not in ds:
            # Fallback to validation if requested split missing
            split_name = "validation" if "validation" in ds else list(ds.keys())[0]
        else:
            split_name = self.split
        table = ds[split_name]

        items: List[BenchmarkItem] = []
        for i, rec in enumerate(table):
            q = rec.get("question", "")
            ctx_text, ctx_meta = self._extract_context_text(rec.get("context"))

            gt = rec.get("final_decision")
            if gt is None:
                gt = rec.get("label")
            if isinstance(gt, list) and len(gt) == 1:
                gt = gt[0]
            if not isinstance(gt, str):
                gt = str(gt) if gt is not None else ""

            item_id = rec.get("pubid") or rec.get("id") or str(i)

            metadata: Dict[str, Any] = {
                "options": ["yes", "no", "maybe"],
                "split": self.split,
            }
            if ctx_meta:
                metadata["context_metadata"] = ctx_meta

            items.append(
                BenchmarkItem(
                    id=item_id,
                    input_text=q,
                    context=ctx_text,
                    ground_truth=gt,
                    metadata=metadata,
                )
            )

        if self.config.sample_limit:
            items = items[: self.config.sample_limit]
        return items

    def _extract_context_text(self, ctx: Any) -> Tuple[str, Optional[Dict[str, Any]]]:
        """Normalize context into a plain string and return extra metadata."""
        extra_metadata: Optional[Dict[str, Any]] = None

        if isinstance(ctx, dict):
            pieces: List[str] = []
            contexts = ctx.get("contexts")
            if isinstance(contexts, list):
                pieces.extend(str(part) for part in contexts if part)
            elif isinstance(contexts, str):
                pieces.append(contexts)

            # Preserve remaining metadata (labels, meshes, etc.) without contexts list
            extra_metadata = {
                key: value for key, value in ctx.items() if key != "contexts"
            }

            text = " ".join(pieces).strip()
            if text:
                return text, extra_metadata

            # Fallback: join any string fields from metadata
            fallback = [
                str(value)
                for value in ctx.values()
                if isinstance(value, str) and value
            ]
            if fallback:
                return " ".join(fallback), extra_metadata

            return str(ctx), extra_metadata

        if isinstance(ctx, list):
            text = " ".join(str(part) for part in ctx if part)
            return text, None

        if ctx is None:
            return "", None

        return str(ctx), None

    def evaluate_response(
        self, response: str, ground_truth: Any, task_name: str, prompt: str
    ) -> float:
        import re

        def norm(x: Any) -> str:
            return (str(x) if x is not None else "").strip().lower()

        r = norm(response)
        gt = norm(ground_truth)

        # Prefer explicit canonical labels anywhere in response (whole word)
        # Attempt to pull structured answers first (e.g., {"answer": "yes"})
        structured = re.search(
            r"\banswer[s]?\b\s*[:=]\s*[\"'`]?\s*(yes|no|maybe)\b",
            response,
            flags=re.IGNORECASE,
        )
        if structured:
            r = structured.group(1).lower()

        m_text = re.search(r"\b(yes|no|maybe)\b", r)
        if m_text:
            r = m_text.group(1)
        else:
            # Map letter choices from common patterns anywhere in response
            # Accept: A, A., A), A: (case-insensitive) and phrases like 'choose C'
            m_letter = re.search(r"\b([a-c])\s*(?:[\.)\:]\s*)?\b", r)
            if not m_letter:
                m_letter = re.search(r"\b(?:choose|option|answer|ans)\s*[:]?\s*([a-c])\b", r)
            if m_letter:
                letter = m_letter.group(1)
                mapping = {"a": "yes", "b": "no", "c": "maybe"}
                r = mapping.get(letter, r)

        if r in {"yes", "no", "maybe"} and r == gt:
            return 1.0
        return 0.0

    def get_task_metrics(self, task_name: str) -> List[str]:
        return ["accuracy"]
