"""
FantomPromptWrapper - Common prompt wrapper for FANToM and similar tasks.

Design:
- Put the question and options in the user message.
- Define the response format in the system prompt as strict JSON:
  {"reational": <reasoning>, "answer": <answer>}

Notes:
- The key name "reational" follows the user’s explicit specification.
"""

from typing import List, Optional, Union, TYPE_CHECKING

from neuro_manipulation.prompt_wrapper import PromptWrapper

if TYPE_CHECKING:
    from neuro_manipulation.prompt_formats import PromptFormat


class FantomPromptWrapper(PromptWrapper):
    """General-purpose wrapper for Fantom-style Q&A with options in user message."""

    system_prompt_format = (
        "You are a helpful AI assistant. Read the context if provided, then answer the question with option content, not option index like A, B, C, 1, 2, 3, etc. "
        "Respond strictly as a single JSON object with exactly these keys: "
        "{'reational': <brief reasoning>, 'answer': <final answer, not option index, but option content>}. "
        "Return only the JSON object, no extra text."
    )

    def __init__(self, prompt_format: "PromptFormat"):
        super().__init__(prompt_format)

    def _format_options(self, options: Optional[List[str]]) -> str:
        if not options:
            return ""
        lines = [f"{chr(65+i)}. {str(opt).strip()}" for i, opt in enumerate(options)]
        return "\n".join(["Options:", *lines])

    def __call__(
        self,
        context: Optional[str] = None,
        question: Optional[str] = None,
        user_messages: Union[str, List[str]] = "Please provide your answer.",
        enable_thinking: bool = False,
        augmentation_config: Optional[dict] = None,
        answer: Optional[str] = None,
        emotion: Optional[str] = None,
        options: Optional[List[str]] = None,
        **kwargs,
    ) -> str:
        # Build user content: context (if any), then question and options
        parts: List[str] = []
        if context:
            parts.append(f"Context:\n{context.strip()}")
        if question:
            parts.append(f"Question: {question.strip()}")
        opts = self._format_options(options)
        if opts:
            parts.append(opts)

        # Optional extra user messages are appended last
        if isinstance(user_messages, list):
            extra = "\n\n".join(str(m).strip() for m in user_messages if str(m).strip())
        else:
            extra = str(user_messages).strip()
        if extra:
            parts.append(extra)

        user_content = "\n\n".join(parts).strip()

        return self.prompt_format.build(
            system_prompt=self.system_prompt_format,
            user_messages=[user_content],
            assistant_messages=[],
            images=None,
            enable_thinking=enable_thinking,
        )
