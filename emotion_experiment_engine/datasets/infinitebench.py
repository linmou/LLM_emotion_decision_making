"""
InfiniteBenchDataset - Specialized dataset for InfiniteBench evaluation.
Handles InfiniteBench-specific data format and routes to appropriate evaluators.
"""

import inspect
from typing import Any, Dict, List

from .. import evaluation_utils
from ..data_models import BenchmarkConfig, BenchmarkItem
from ..evaluation_utils import get_score_one
from .base import BaseBenchmarkDataset


class InfiniteBenchDataset(BaseBenchmarkDataset):
    """
    Specialized dataset for InfiniteBench tasks.

    Routes evaluation to task-specific evaluators:
    - passkey: Extract first integer from response
    - kv_retrieval: Check if label is in word-split response
    - math_find: Extract numerical result with tolerance
    - longbook_qa_eng: F1 score with token normalization
    - And all other InfiniteBench tasks...
    """

    # Static mapping eliminates if-else chains
    # Maps task names to evaluator function names in evaluation_utils
    # All tasks now use unified LLM evaluation
    TASK_EVALUATORS = {
        "passkey": "llm_evaluate_response",
        "kv_retrieval": "llm_evaluate_response",
        "number_string": "llm_evaluate_response",
        "code_run": "llm_evaluate_response",
        "code_debug": "llm_evaluate_response",
        "math_find": "llm_evaluate_response",
        "math_calc": "llm_evaluate_response",
        "longdialogue_qa_eng": "llm_evaluate_response",
        "longbook_choice_eng": "llm_evaluate_response",
        "longbook_qa_eng": "llm_evaluate_response",
        "longbook_sum_eng": "llm_evaluate_response",
        "longbook_qa_chn": "llm_evaluate_response",
    }

    def _load_and_parse_data(self) -> List[BenchmarkItem]:
        """Load InfiniteBench-specific data format"""
        raw_data = self._load_raw_data()
        return self._parse_infinitebench_data(raw_data)

    def _parse_infinitebench_data(
        self, raw_data: List[Dict[str, Any]]
    ) -> List[BenchmarkItem]:
        """Parse InfiniteBench format to BenchmarkItem objects"""
        items = []
        for i, item_data in enumerate(raw_data):
            item_id = item_data.get("id", i)
            input_text = item_data.get("input", "")
            context = item_data.get("context", "")

            # InfiniteBench sometimes puts context in input field
            if not input_text and context:
                input_text = context
                context = None

            # Extract ground truth from various possible fields
            ground_truth = item_data.get(
                "answer", item_data.get("label", item_data.get("ground_truth", None))
            )
            if isinstance(ground_truth, list) and len(ground_truth) == 1:
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
        """Route to task-specific InfiniteBench evaluator"""
        return self._route_to_evaluator(task_name, response, ground_truth)

    def _route_to_evaluator(
        self, task_name: str, response: str, ground_truth: Any
    ) -> float:
        """Route to specific evaluator without if-else chains"""
        evaluator_name = self.TASK_EVALUATORS.get(task_name)

        if not evaluator_name:
            # Fallback to generic evaluator for unknown tasks
            return float(
                get_score_one(response, ground_truth, task_name, "emotion_model")
            )

        # Handle LLM evaluation with prompt construction
        if evaluator_name == "llm_evaluate_response":
            # Use the longbench prompt format as a generic template
            prompt_format = """Evaluate if the response correctly answers what was expected. Consider semantic meaning, not just exact words.

Now evaluate:
Response: {response}
Expected: {ground_truth}

response in json format: {{"answer": "1.0 for correct, 0.0 for incorrect"}}"""

            query = prompt_format.format(response=response, ground_truth=ground_truth)

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
        """Return metrics available for InfiniteBench task"""

        # Exact match tasks
        if task_name in [
            "passkey",
            "kv_retrieval",
            "number_string",
            "code_run",
            "code_debug",
            "math_find",
            "math_calc",
        ]:
            return ["exact_match"]

        # F1 scoring tasks
        elif task_name in ["longbook_qa_eng", "longbook_qa_chn"]:
            return ["f1_score", "precision", "recall"]

        # ROUGE scoring tasks
        elif task_name == "longbook_sum_eng":
            return ["rouge_score"]

        # Choice/classification tasks
        elif task_name in ["longbook_choice_eng"]:
            return ["choice_accuracy"]

        # Substring matching tasks
        elif task_name in ["longdialogue_qa_eng"]:
            return ["substring_match"]

        # Default
        else:
            return ["accuracy"]

    def evaluate_with_detailed_metrics(
        self, response: str, ground_truth: Any, task_name: str
    ) -> Dict[str, float]:
        """Return detailed metrics for InfiniteBench evaluation"""
        base_score = self.evaluate_response(response, ground_truth, task_name, prompt)

        metrics = {"overall_score": base_score}

        # Add task-specific metric names
        task_metrics = self.get_task_metrics(task_name)
        for metric in task_metrics:
            metrics[metric] = base_score

        return metrics
