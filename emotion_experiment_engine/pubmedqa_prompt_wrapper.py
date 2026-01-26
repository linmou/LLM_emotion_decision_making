"""PubMedQA-specific prompt wrapper with optional CoT instructions."""

from typing import Any, List, Optional, Union, TYPE_CHECKING

from neuro_manipulation.prompt_wrapper import PromptWrapper

if TYPE_CHECKING:
    from neuro_manipulation.prompt_formats import PromptFormat


class PubMedQAPromptWrapper(PromptWrapper):
    system_prompt_format = (
        "You are a helpful AI assistant. Please answer the following question based on the given context."
    )

    def __init__(self, prompt_format: "PromptFormat") -> None:
        super().__init__(prompt_format)

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
    ) -> str:
        system_prompt = self._build_system_prompt(context, question, options)
        messages = self._prepare_user_messages(user_messages, augmentation_config)
        return self.prompt_format.build(
            system_prompt,
            messages,
            enable_thinking=enable_thinking,
        )

    def _build_system_prompt(
        self,
        context: Optional[str],
        question: Optional[str],
        options: Optional[List[str]],
    ) -> str:
        if context:
            base = (
                f"{self.system_prompt_format}\n\nContext: {context}\n\nQuestion: {question}"
            )
        else:
            base = f"{self.system_prompt_format}\n\nQuestion: {question}"
        return base + self._format_options(options)

    def _format_options(self, options: Optional[List[str]]) -> str:
        opts = options or ["yes", "no", "maybe"]
        letters = ["A", "B", "C"]
        lines = []
        for idx, opt in enumerate(opts[:3]):
            lines.append(f"{letters[idx]}. {opt}")
        body = "\n".join(lines)
        return f"\n\nOptions:\n{body}\n\nPlease select the correct answer (A, B, or C)."

    def _prepare_user_messages(
        self,
        base_user_messages: Union[str, List[str]],
        augmentation_config: Optional[dict],
    ) -> List[str]:
        messages = self.user_messages(base_user_messages)
        cot_instruction = self._extract_cot_instruction(augmentation_config)
        if cot_instruction:
            messages = list(messages) + [cot_instruction]
        return list(messages)

    def user_messages(self, user_messages: Union[str, List[str]]) -> List[str]:
        if isinstance(user_messages, list):
            return user_messages
        return [user_messages]

    def _extract_cot_instruction(
        self, augmentation_config: Optional[dict]
    ) -> Optional[str]:
        if not augmentation_config or not isinstance(augmentation_config, dict):
            return None

        if augmentation_config.get("method") == "cot":
            instr = (
                augmentation_config.get("instruction")
                or augmentation_config.get("cot_instruction")
                or augmentation_config.get("prompt")
                or augmentation_config.get("text")
            )
            if instr:
                text = str(instr).strip()
                return text or None
            if augmentation_config.get("enabled", True):
                return (
                    "Before answering, briefly reason about the evidence, then reply with the final answer."
                )
            return None

        # Backwards compatibility with older configs that used a top-level 'cot' key
        cot_cfg = augmentation_config.get("cot")
        if cot_cfg is None:
            return None

        if isinstance(cot_cfg, bool):
            return (
                "Before answering, briefly reason about the evidence, then reply with the final answer."
                if cot_cfg
                else None
            )
        if isinstance(cot_cfg, str):
            text = cot_cfg.strip()
            return text or None
        if isinstance(cot_cfg, dict):
            text = (
                cot_cfg.get("instruction")
                or cot_cfg.get("prompt")
                or cot_cfg.get("text")
            )
            if text:
                return str(text).strip()
            if cot_cfg.get("enabled"):
                return (
                    "Before answering, briefly reason about the evidence, then reply with the final answer."
                )
            return None

        return str(cot_cfg).strip() or None
