"""
MBPPPromptWrapper - minimal chat-style code-only instruction

Parallels HumanEval wrapper to keep outputs clean (no backticks, no prose).
"""

from typing import List, Optional, Union, TYPE_CHECKING, Any

from neuro_manipulation.prompt_wrapper import PromptWrapper

if TYPE_CHECKING:
    from neuro_manipulation.prompt_formats import PromptFormat


class MBPPPromptWrapper(PromptWrapper):
    def __init__(self, prompt_format: "PromptFormat", task_type: str = "main"):
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
        enable_thinking: bool = False,
        emotion: Optional[str] = None,
        **kwargs: Any,
    ) -> str:
        system_prompt = (
            "You are a Python assistant. Complete the function strictly as valid Python code. "
            "Do not include explanations, comments, or backticks."
        )
        user_msg = question
        return self.prompt_format.build(
            system_prompt=system_prompt,
            user_messages=[user_msg],
            assistant_messages=[],
            images=None,
            enable_thinking=False,
        )

