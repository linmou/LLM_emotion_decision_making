"""
E2E parity vs upstream GPQA repo (zero-shot CoT):
- Ensure __getitem__ semantics match (question, shuffled options, correct)
- Ensure prompt text equals upstream zero_shot_chain_of_thought_prompt when
  providing the same CoT reasoning string via augmentation_config.
"""

import csv
import tempfile
from pathlib import Path

from emotion_experiment_engine.data_models import BenchmarkConfig
from emotion_experiment_engine.benchmark_component_registry import create_benchmark_components


def _make_csv(rows=3) -> Path:
    fd, path_str = tempfile.mkstemp(suffix=".csv")
    path = Path(path_str)
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "Question",
            "Correct Answer",
            "Incorrect Answer 1",
            "Incorrect Answer 2",
            "Incorrect Answer 3",
        ])
        for i in range(rows):
            w.writerow([
                f"Q{i}?",
                f"Correct{i}",
                f"WrongA{i}",
                f"WrongB{i}",
                f"WrongC{i}",
            ])
    return path


def test_gpqa_parity_against_repo_shuffle_and_cot(tmp_path):
    # Arrange: create a small CSV and load both systems
    csv_path = _make_csv(rows=3)
    try:
        # Load baseline repo helper (shuffles with provided seed)
        import sys
        sys.path.insert(0, "/data/home/jjl7137/gpqa/baselines")
        import utils as gpqa_utils  # type: ignore

        seed = 1234
        examples = gpqa_utils.load_examples(str(csv_path), seed=seed)

        # Monkeypatch model call to avoid network and produce deterministic CoT
        class _StubResp:
            class _Choice:
                class _Message:
                    def __init__(self, content):
                        self.content = content
                def __init__(self, content):
                    self.message = self._Message(content)
            def __init__(self, content):
                self.choices = [self._Choice(content)]

        gpqa_utils.call_model_with_retries = lambda prompt, model_name, call_type='sample': _StubResp(
            content="Because of X, Y, Z, the best answer is A."
        )
        expected_prompt = gpqa_utils.zero_shot_chain_of_thought_prompt(
            0, examples[0], model_name="gpt-4"
        )

        # Load our dataset with parity seed so option order matches
        config = BenchmarkConfig(
            name="gpqa",
            task_type="main",
            data_path=csv_path,
            base_data_dir=str(csv_path.parent),
            sample_limit=None,
            augmentation_config=None,
            enable_auto_truncation=False,
            truncation_strategy="right",
            preserve_ratio=1.0,
            llm_eval_config=None,
        )

        prompt_wrapper, _, dataset = create_benchmark_components(
            benchmark_name="gpqa",
            task_type="main",
            config=config,
            prompt_format=None,
            shuffle_options_seed=seed,
            augmentation_config={
                "gpqa_cot_reasoning": "Because of X, Y, Z, the best answer is A."
            },
        )

        # Assert: question and shuffled options are identical across repos
        # Item parity
        item0 = dataset[0]["item"]
        ex0 = examples[0]
        assert item0.input_text == ex0.question
        assert item0.metadata["options"] == [ex0.choice1, ex0.choice2, ex0.choice3, ex0.choice4]
        assert item0.ground_truth == [[ex0.choice1, ex0.choice2, ex0.choice3, ex0.choice4][ex0.correct_index]]

        # Prompt parity (zero-shot CoT)
        prompt_ours = prompt_wrapper(
            context=item0.context,
            question=item0.input_text,
            options=item0.metadata["options"],
        )
        assert prompt_ours == expected_prompt
    finally:
        try:
            csv_path.unlink()
        except Exception:
            pass
