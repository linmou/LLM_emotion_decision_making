"""
GPQAPromptWrapper - Single wrapper with CoT as an augmentation mode (default).

Modes (controlled via augmentation_config):
- Default (or gpqa_mode='cot'): zero-shot chain-of-thought format matching
  gpqa/baselines/utils.zero_shot_chain_of_thought_prompt structure.
- gpqa_mode='zero_shot': plain zero-shot format matching
  gpqa/baselines/utils.zero_shot_prompt structure.

This keeps KISS: one class, augmentation toggles behavior.
"""

from typing import List, Optional, Union, TYPE_CHECKING
from neuro_manipulation.prompt_wrapper import PromptWrapper

if TYPE_CHECKING:
    from neuro_manipulation.prompt_formats import PromptFormat


class GPQAPromptWrapper(PromptWrapper):
    def __init__(self, prompt_format: "PromptFormat", task_type: str = "main"):
        # Accept PromptFormat for interface parity (unused for exact string parity)
        super().__init__(prompt_format)
        self.task_type = task_type

    def __call__(
        self,
        context: str,
        question: str,
        answer: Optional[str] = None,
        options: Optional[List[str]] = None,
        user_messages: Optional[Union[str, List[str]]] = None,
        augmentation_config: Optional[dict] = None,
        **kwargs,
    ) -> str:
        if not options or len(options) != 4:
            raise ValueError("GPQA requires exactly 4 options")

        # Determine mode; default to CoT as requested
        mode = "cot"
        if augmentation_config and isinstance(augmentation_config, dict):
            mode = augmentation_config.get("gpqa_mode", "cot").lower()

        # Base prompt identical for both modes
        lines = []
        lines.append(f"What is the correct answer to this question: {question}")
        lines.append("")
        lines.append("Choices:")
        letters = ["A", "B", "C", "D"]
        for i, opt in enumerate(options):
            lines.append(f"({letters[i]}) {opt}")

        if mode == "cot":
            # Insert CoT block per upstream utils.zero_shot_chain_of_thought_prompt
            cot_reasoning = None
            if augmentation_config and isinstance(augmentation_config, dict):
                cot_reasoning = augmentation_config.get("gpqa_cot_reasoning")
                provider = augmentation_config.get("gpqa_cot_provider")
                if cot_reasoning is None and callable(provider):
                    try:
                        cot_reasoning = provider(question, options)
                    except Exception:
                        cot_reasoning = None
            if cot_reasoning is None:
                cot_reasoning = ""
            lines.append("Let's think step by step: " + str(cot_reasoning))
            lines.append("")
            lines.append(
                "Based on the above, what is the single, most likely answer choice? "
                "Answer in the format \"The correct answer is (insert answer here)\"."
            )
        else:
            # Plain zero-shot formatting per upstream utils.zero_shot_prompt
            lines.append("")
            lines.append(
                "Format your response as follows: \"The correct answer is (insert answer here)\""
            )

        return "\n".join(lines)
