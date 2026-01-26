"""
LongBenchDataset - Specialized dataset for LongBench evaluation.
Handles LongBench-specific data format and routes to appropriate metric evaluators.
"""

import inspect
from typing import Any, Dict, List

# Import all LongBench evaluators
from .. import evaluation_utils
from ..data_models import BenchmarkConfig, BenchmarkItem
from ..evaluation_utils import get_score_one
from .base import BaseBenchmarkDataset


class LongBenchDataset(BaseBenchmarkDataset):
    """
    Specialized dataset for LongBench tasks.

    Routes evaluation to metric-specific evaluators:
    - narrativeqa/qasper: F1 score with token normalization
    - gov_report/qmsum: ROUGE-L scoring
    - trec/lsht: Classification accuracy with substring matching
    - passage_count: Count accuracy with number extraction
    - lcc/repobench: Code similarity scoring
    - Chinese tasks: Character-level evaluation
    """

    # Static mapping eliminates if-else chains
    # Maps task names to evaluator function names in evaluation_utils
    # All tasks now use unified LLM evaluation
    METRIC_EVALUATORS = {
        "narrativeqa": "llm_evaluate_response",
        "qasper": "llm_evaluate_response",
        "multifieldqa_en": "llm_evaluate_response",
        "multifieldqa_zh": "llm_evaluate_response",
        "hotpotqa": "llm_evaluate_response",
        "2wikimqa": "llm_evaluate_response",
        "musique": "llm_evaluate_response",
        "dureader": "llm_evaluate_response",
        "gov_report": "llm_evaluate_response",
        "qmsum": "llm_evaluate_response",
        "multi_news": "llm_evaluate_response",
        "vcsum": "llm_evaluate_response",
        "trec": "llm_evaluate_response",
        "triviaqa": "llm_evaluate_response",
        "samsum": "llm_evaluate_response",
        "lsht": "llm_evaluate_response",
        "passage_retrieval_en": "llm_evaluate_response",
        "passage_count": "llm_evaluate_response",
        "passage_retrieval_zh": "llm_evaluate_response",
        "lcc": "llm_evaluate_response",
        "repobench-p": "llm_evaluate_response",
    }

    PROMPT_FORMAT = """Evaluate if the response correctly answers what was expected. Consider semantic meaning, not just exact words.

Examples:
Response: "The passkey is 42"
Expected: "42"  
Answer: CORRECT

Response: "The capital is Paris"
Expected: "Paris"
Answer: CORRECT

Response: "I don't know"
Expected: "Tokyo"
Answer: INCORRECT

Response: "The answer is B"
Expected: ["B"]
Answer: CORRECT

Now evaluate:
Response: {response}
Expected: {ground_truth}

response in json format: {{"answer": "1.0 for correct, 0.0 for incorrect"}}"""

    LLM_EVAL_CONFIG = {
        "model": "gpt-4o-mini",
        "temperature": 0.0,
    }

    def _load_and_parse_data(self) -> List[BenchmarkItem]:
        """Load LongBench-specific data format"""
        raw_data = self._load_raw_data()
        return self._parse_longbench_data(raw_data)

    def _parse_longbench_data(
        self, raw_data: List[Dict[str, Any]]
    ) -> List[BenchmarkItem]:
        """Parse LongBench format to BenchmarkItem objects"""
        items = []
        for i, item_data in enumerate(raw_data):
            item_id = item_data.get("id", f"longbench_{i}")
            input_text = item_data.get("input", "")
            context = item_data.get("context", "")

            # LongBench may have answers as list - take first one as primary
            ground_truth = item_data.get("answers", item_data.get("answer", None))
            if isinstance(ground_truth, list) and ground_truth:
                ground_truth = ground_truth[0]

            items.append(
                BenchmarkItem(
                    id=item_id,
                    input_text=input_text,
                    context=context,
                    ground_truth=ground_truth,
                    metadata=item_data,
                )
            )

        return items

    def evaluate_response(
        self, response: str, ground_truth: Any, task_name: str, prompt: str
    ) -> float:
        """Route to metric-specific LongBench evaluator"""
        return self._route_to_evaluator(task_name, response, ground_truth, prompt)

    def _route_to_evaluator(
        self, task_name: str, response: str, ground_truth: Any, prompt: str
    ) -> float:
        """Route to specific evaluator without if-else chains"""
        evaluator_name = self.METRIC_EVALUATORS.get(task_name)

        if not evaluator_name:
            # Fallback to generic evaluator for unknown tasks
            return float(
                get_score_one(response, ground_truth, task_name, "emotion_model")
            )

        # Handle LLM evaluation with prompt construction
        if evaluator_name == "llm_evaluate_response":
            # Construct evaluation prompt using our template
            query = self.PROMPT_FORMAT.format(
                response=response, ground_truth=ground_truth
            )

            result = evaluation_utils.llm_evaluate_response(
                system_prompt="You are an expert evaluator.",
                query=query,
                llm_eval_config=self.llm_eval_config,
            )
            return float(result.get("answer", 0.0))

        else:
            # Handle non-LLM evaluators
            evaluator_func = getattr(evaluation_utils, evaluator_name)

            func_signature = inspect.signature(evaluator_func)
            func_params = func_signature.parameters.keys()
            all_available_args = {
                "response": response,
                "ground_truth": ground_truth,
                "task_name": task_name,
                "llm_eval_config": self.llm_eval_config,
            }

            args_to_pass = {
                name: all_available_args[name]
                for name in func_params
                if name in all_available_args
            }

            result = evaluator_func(**args_to_pass)
            return float(result)

    def get_task_metrics(self, task_name: str) -> List[str]:
        """Return metrics available for LongBench task"""

        # QA F1 tasks
        if task_name in [
            "narrativeqa",
            "qasper",
            "multifieldqa_en",
            "multifieldqa_zh",
            "hotpotqa",
            "2wikimqa",
            "musique",
            "triviaqa",
        ]:
            return ["f1_score", "precision", "recall"]

        # ROUGE scoring tasks
        elif task_name in [
            "gov_report",
            "qmsum",
            "multi_news",
            "samsum",
            "dureader",
            "vcsum",
        ]:
            return ["rouge_score"]

        # Classification tasks
        elif task_name in ["trec", "lsht"]:
            return ["classification_accuracy"]

        # Retrieval tasks
        elif task_name in ["passage_retrieval_en", "passage_retrieval_zh"]:
            return ["retrieval_accuracy"]

        # Count tasks
        elif task_name == "passage_count":
            return ["count_accuracy"]

        # Code tasks
        elif task_name in ["lcc", "repobench-p"]:
            return ["code_similarity"]

        # Default
        else:
            return ["accuracy"]

    def evaluate_by_length(
        self,
        responses: List[str],
        ground_truths: List[Any],
        context_lengths: List[int],
        task_names: List[str],
    ) -> Dict[str, float]:
        """
        Evaluate responses grouped by context length ranges.
        LongBench-specific functionality for analyzing performance vs context length.
        """
        length_ranges = [(0, 4000), (4000, 8000), (8000, 16000), (16000, float("inf"))]
        results = {}

        for min_len, max_len in length_ranges:
            range_name = f"{min_len}-{max_len if max_len != float('inf') else 'inf'}"

            # Filter items in this length range
            range_indices = [
                i
                for i, length in enumerate(context_lengths)
                if min_len <= length < max_len
            ]

            if not range_indices:
                results[range_name] = 0.0
                continue

            # Evaluate items in this range
            range_scores = []
            for i in range_indices:
                score = self.evaluate_response(
                    responses[i], ground_truths[i], task_names[i], prompts[i]
                )
                range_scores.append(score)

            results[range_name] = sum(range_scores) / len(range_scores)

        return results

    def evaluate_with_detailed_metrics(
        self, response: str, ground_truth: Any, task_name: str
    ) -> Dict[str, float]:
        """Return detailed metrics for LongBench evaluation"""
        base_score = self.evaluate_response(response, ground_truth, task_name)

        metrics = {"overall_score": base_score}

        # Add task-specific metric names
        task_metrics = self.get_task_metrics(task_name)
        for metric in task_metrics:
            metrics[metric] = base_score

        return metrics
