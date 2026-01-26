"""Prompt wrapper for Diplomacy PD gradient decisions."""

from __future__ import annotations

from typing import Any, List, Optional, Sequence

from neuro_manipulation.prompt_wrapper import PromptWrapper


class DiplomacyOptionsPromptWrapper(PromptWrapper):
    """Render Diplomacy decision scenarios with explicit background header."""

    def __init__(self, prompt_format: Any):
        super().__init__(prompt_format)

    @staticmethod
    def _normalize_options(options: Optional[Sequence[Any]]) -> List[str]:
        if not options:
            return []
        normalized: List[str] = []
        for option in options:
            if isinstance(option, dict):
                text = option.get("text") or option.get("value")
                normalized.append(str(text) if text is not None else str(option))
            else:
                normalized.append(str(option))
        return normalized

    @staticmethod
    def _render_header(context: Optional[str]) -> str:
        if not context:
            return ""
        cleaned = str(context).strip()
        return f"{cleaned}\n\n" if cleaned else ""

    @staticmethod
    def _render_event_options(event: str, options: Sequence[str]) -> str:
        lines = [str(event).strip()]
        for idx, option in enumerate(options, start=1):
            lines.append(f"Option {idx}. {option}")
        lines.append("Respond with the option text.")
        return "\n".join(lines)

    def __call__(
        self,
        *,
        context: Optional[str],
        question: str,
        user_messages: Sequence[str] | str | None,
        enable_thinking: bool,
        augmentation_config: Optional[dict],
        answer: Any,
        emotion: Optional[str],
        options: Optional[Sequence[Any]],
    ) -> str:
        del augmentation_config, answer, emotion
        normalized_options = self._normalize_options(options)

        if user_messages is None:
            user_messages_list: List[str] = ["Please provide your answer."]
        elif isinstance(user_messages, str):
            user_messages_list = [user_messages]
        else:
            user_messages_list = list(user_messages)

        header = self._render_header(context)
        body = self._render_event_options(question, normalized_options)
        prompt_text = header + body

        builder = getattr(self.prompt_format, "build", None)
        if callable(builder):
            return builder(prompt_text, user_messages_list, enable_thinking=enable_thinking)
        return prompt_text


__all__ = ["DiplomacyOptionsPromptWrapper"]
