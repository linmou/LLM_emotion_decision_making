"""
TruthfulQADataset - Specialized dataset for TruthfulQA multiple choice evaluation.
Handles MC1 (single correct answer) and MC2 (multiple correct answers) variants.
"""

import json
from pathlib import Path
from typing import Any, List

from ..data_models import BenchmarkConfig, BenchmarkItem
from .base import BaseBenchmarkDataset


class TruthfulQADataset(BaseBenchmarkDataset):
    """
    Specialized dataset for TruthfulQA multiple choice tasks.

    Handles both MC1 (single correct answer) and MC2 (multiple correct answers)
    variants with strict evaluation and clear error reporting.
    """

    # Task type to evaluation strategy mapping
    TASK_EVALUATORS = {"mc1": "_evaluate_mc1", "mc2": "_evaluate_mc2"}

    def _load_and_parse_data(self) -> List[BenchmarkItem]:
        """
        Load and parse TruthfulQA JSONL data into BenchmarkItem objects.

        Expected JSONL format:
        {"question": "...", "options": ["...", "..."], "answers": ["..."]}

        Returns:
            List of BenchmarkItem objects

        Raises:
            FileNotFoundError: If data file doesn't exist
            ValueError: If data format is invalid
            KeyError: If required fields are missing
        """
        # Use get_data_path() to handle None data_path and auto-generation
        # TruthfulQA uses custom data directory matching config: base_data_dir: "data/TruthfulQA"
        # Expected path format: data/TruthfulQA/{name}_{task_type}.jsonl
        # e.g., data/TruthfulQA/truthfulqa_mc1.jsonl, data/TruthfulQA/truthfulqa_mc2.jsonl
        data_path = self.config.get_data_path()
        if not data_path.exists():
            raise FileNotFoundError(f"TruthfulQA data file not found: {data_path}")

        items = []
        with open(data_path, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                try:
                    data = json.loads(line.strip())
                except json.JSONDecodeError as e:
                    raise ValueError(f"Invalid JSON on line {line_num}: {e}")

                # Strict validation of required fields
                if "question" not in data:
                    raise KeyError(f"Missing 'question' field on line {line_num}")
                if "options" not in data:
                    raise KeyError(f"Missing 'options' field on line {line_num}")
                if "answers" not in data:
                    raise KeyError(f"Missing 'answers' field on line {line_num}")

                # Validate data types
                if not isinstance(data["question"], str):
                    raise ValueError(f"'question' must be string on line {line_num}")
                if not isinstance(data["options"], list):
                    raise ValueError(f"'options' must be list on line {line_num}")
                if not isinstance(data["answers"], list):
                    raise ValueError(f"'answers' must be list on line {line_num}")

                # Validate non-empty
                if not data["question"].strip():
                    raise ValueError(f"Empty question on line {line_num}")
                if len(data["options"]) == 0:
                    raise ValueError(f"Empty options list on line {line_num}")
                if len(data["answers"]) == 0:
                    raise ValueError(f"Empty answers list on line {line_num}")

                # Validate reasonable number of options
                if len(data["options"]) > 26:
                    raise ValueError(
                        f"Too many options ({len(data['options'])}) on line {line_num}, max supported: 26"
                    )

                # Store raw data - let prompt wrapper handle formatting
                item = BenchmarkItem(
                    id=f"truthfulqa_{line_num-1}",
                    context=data["question"],  # Raw question for system prompt
                    input_text=data["question"],  # Raw question for prompt wrapper
                    ground_truth=data["answers"],
                    metadata={
                        "options": data["options"],
                        "line_number": line_num,
                        "task_type": self.config.task_type,
                    },
                )
                items.append(item)

        if len(items) == 0:
            raise ValueError(f"No valid items found in {data_path}")

        return items


    def evaluate_response(
        self, response: str, ground_truth: Any, task_name: str, prompt: str
    ) -> float:
        """
        Evaluate model response against ground truth for TruthfulQA.

        Args:
            response: Model's response text
            ground_truth: List of correct answers
            task_name: Task type (mc1 or mc2)
            prompt: The full prompt (contains item for metadata access)

        Returns:
            Score between 0.0 and 1.0

        Raises:
            ValueError: If task_name is not supported
            TypeError: If ground_truth is not a list
        """
        if not isinstance(ground_truth, list):
            raise TypeError(f"ground_truth must be list, got {type(ground_truth)}")

        if task_name not in self.TASK_EVALUATORS:
            raise ValueError(
                f"Unsupported task type: {task_name}. Supported: {list(self.TASK_EVALUATORS.keys())}"
            )

        # Route to appropriate evaluator
        evaluator_method = getattr(self, self.TASK_EVALUATORS[task_name])
        return evaluator_method(response, ground_truth)

    def _evaluate_mc1(self, response: str, ground_truth: List[str]) -> float:
        """
        Evaluate MC1 response (single correct answer).

        Args:
            response: Model response (should be option text)
            ground_truth: List containing single correct answer

        Returns:
            1.0 if correct, 0.0 if incorrect

        Raises:
            ValueError: If ground_truth doesn't contain exactly one answer
        """
        if len(ground_truth) != 1:
            raise ValueError(
                f"MC1 requires exactly 1 correct answer, got {len(ground_truth)}"
            )

        correct_answer = ground_truth[0].strip()
        response = response.strip()

        # Direct text match evaluation
        if response.lower() == correct_answer.lower():
            return 1.0

        # No match found - strict evaluation
        return 0.0

    def _evaluate_mc2(self, response: str, ground_truth: List[str]) -> float:
        """
        Evaluate MC2 response (multiple correct answers possible).

        Args:
            response: Model response (should be option text(s))
            ground_truth: List of all correct answers

        Returns:
            Partial credit score based on correct matches

        Raises:
            ValueError: If ground_truth is empty
        """
        if len(ground_truth) == 0:
            raise ValueError("MC2 requires at least 1 correct answer")

        response = response.strip()
        correct_answers = [ans.strip() for ans in ground_truth]

        # Handle text response - look for matches with correct answers
        matches = sum(1 for ans in correct_answers if ans.lower() in response.lower())
        return matches / len(correct_answers)

    def get_task_metrics(self, task_name: str) -> List[str]:
        """
        Return available metrics for TruthfulQA task types.

        Args:
            task_name: Task type (mc1 or mc2)

        Returns:
            List of available metric names

        Raises:
            ValueError: If task_name is not supported
        """
        if task_name not in self.TASK_EVALUATORS:
            raise ValueError(
                f"Unsupported task type: {task_name}. Supported: {list(self.TASK_EVALUATORS.keys())}"
            )

        if task_name == "mc1":
            return ["accuracy", "exact_match"]
        elif task_name == "mc2":
            return ["accuracy", "partial_credit", "f1_score"]

        return ["accuracy"]
