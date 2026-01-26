"""
HumanEvalPromptWrapper - default raw pass-through of the upstream prompt.

KISS: by default, return the exact HumanEval prompt string (no chat framing),
ensuring parity with the upstream evaluation harness. Optionally, a "chat"
mode can be enabled via augmentation_config["humaneval_mode"] == "chat",
which builds a minimal chat-style prompt using the provided PromptFormat.
"""

from typing import List, Optional, Union, TYPE_CHECKING, Any

from neuro_manipulation.prompt_wrapper import PromptWrapper

if TYPE_CHECKING:
    from neuro_manipulation.prompt_formats import PromptFormat


class HumanEvalPromptWrapper(PromptWrapper):
    def __init__(self, prompt_format: "PromptFormat", task_type: str = "main"):
        # Keep interface parity with other wrappers; task_type unused.
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
        # Always use chat-style formatting for HumanEval.
        # Instruction is strict to reduce risk of prose or fenced output.
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
