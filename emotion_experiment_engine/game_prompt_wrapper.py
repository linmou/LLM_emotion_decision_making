"""Prompt wrapper adapter for game theory benchmarks."""

from __future__ import annotations

import re
from typing import Any, List, Optional, Sequence

from games.game_configs import get_game_config
from neuro_manipulation.prompt_wrapper import GameReactPromptWrapper


class GameBenchmarkPromptWrapper:
    """Adapts game theory prompts to the benchmark wrapper signature."""

    def __init__(self, prompt_format: Any | None, task_type: str) -> None:
        self.prompt_format = prompt_format
        self.task_type = task_type
        self._config = get_game_config(task_type)
        self._decision_class = self._config["decision_class"]
        self._react_wrapper: Optional[GameReactPromptWrapper] = None

        if self.prompt_format is not None:
            self._react_wrapper = GameReactPromptWrapper(
                self.prompt_format, self._decision_class
            )

    def _ensure_wrapper(self) -> GameReactPromptWrapper | None:
        if self._react_wrapper is None and self.prompt_format is not None:
            self._react_wrapper = GameReactPromptWrapper(
                self.prompt_format, self._decision_class
            )
        return self._react_wrapper

    @staticmethod
    def _normalize_options(options: Optional[Sequence[Any]]) -> List[str]:
        if not options:
            return []
        normalized: List[str] = []
        for opt in options:
            if isinstance(opt, dict):
                text = opt.get("text") or opt.get("value")
                if text is None:
                    text = str({k: v for k, v in opt.items() if k != "id"})
                normalized.append(str(text))
            else:
                normalized.append(str(opt))
        return normalized

    @staticmethod
    def _fallback_prompt(event: str, options: Sequence[str]) -> str:
        prompt_lines = [event]
        for idx, option in enumerate(options, start=1):
            prompt_lines.append(f"Option {idx}. {option}")
        prompt_lines.append("Respond with the option text.")
        return "\n".join(prompt_lines)

    def __call__(
        self,
        *,
        context: str | None,
        question: str,
        user_messages: Sequence[str] | str | None,
        enable_thinking: bool,
        augmentation_config: Optional[dict],
        answer: Any,
        emotion: Optional[str],
        options: Optional[Sequence[Any]],
    ) -> str:
        del context, augmentation_config, answer, emotion  # unused in adapter
        normalized_options = self._normalize_options(options)

        wrapper = self._ensure_wrapper()
        if wrapper is None:
            return self._fallback_prompt(question, normalized_options)

        if user_messages is None:
            user_messages = ["Please provide your answer."]
        elif isinstance(user_messages, str):
            user_messages = [user_messages]

        prompt_text = wrapper(
            event=question,
            options=normalized_options,
            user_messages=list(user_messages),
            enable_thinking=enable_thinking,
        )

        if not prompt_text:
            return self._fallback_prompt(question, normalized_options)

        return prompt_text


class GameDecisionPromptWrapper(GameBenchmarkPromptWrapper):
    """Prompt variant that demands an immediate JSON decision."""

    DECIDE_INSTRUCTION = (
        'You must make a decision now. Do not wait for additional information or refuse '
        'to decide. Respond only with this JSON object: {"decision": "<copy one option text exactly>"}'
    )
    _scenario_re = re.compile(r"\bScenario:\s*(\S+)")

    def __call__(
        self,
        *,
        context: str | None,
        question: str,
        user_messages: Sequence[str] | str | None,
        enable_thinking: bool,
        augmentation_config: Optional[dict],
        answer: Any,
        emotion: Optional[str],
        options: Optional[Sequence[Any]],
    ) -> str:
        del context, augmentation_config, answer, emotion  # unused in adapter
        normalized_options = self._normalize_options(options)

        persona = "Alice"
        m = self._scenario_re.search(question)
        if m and m.group(1).startswith("Diplomacy_"):
            persona = "a commander who needs to make the decision"

        system_lines = [
            f"You are {persona}. You are in the following situation:",
            question,
        ]
        for idx, option in enumerate(normalized_options, start=1):
            system_lines.append(f"Option {idx}. {option}")
        system_lines.append(self.DECIDE_INSTRUCTION)
        system_lines.append('Do not output anything else.')

        system_prompt = "\n".join(system_lines)

        if user_messages is None:
            user_messages = ["Please provide your answer."]
        elif isinstance(user_messages, str):
            user_messages = [user_messages]

        if self.prompt_format is None:
            return system_prompt

        return self.prompt_format.build(
            system_prompt,
            list(user_messages),
            enable_thinking=enable_thinking,
        )

class GameCompletionOptionIdPromptWrapper(GameBenchmarkPromptWrapper):
    """Completion-style prompt for base LMs that should emit an option id."""

    def __call__(
        self,
        *,
        context: str | None,
        question: str,
        user_messages: Sequence[str] | str | None,
        enable_thinking: bool,
        augmentation_config: Optional[dict],
        answer: Any,
        emotion: Optional[str],
        options: Optional[Sequence[Any]],
    ) -> str:
        del context, user_messages, enable_thinking, augmentation_config, answer, emotion  # unused
        normalized_options = self._normalize_options(options)
        n_options = len(normalized_options)
        if n_options <= 0:
            raise ValueError("Expected non-empty options for completion_option_id prompt")

        if n_options <= 9:
            if n_options == 1:
                choice_hint = "1"
            elif n_options == 2:
                choice_hint = "1 or 2"
            else:
                choice_hint = f"{', '.join(str(i) for i in range(1, n_options))}, or {n_options}"
            instruction = f"Output exactly one character: {choice_hint}."
        else:
            instruction = f"Output only the option number (1-{n_options})."

        system_lines = [
            "You are an average American.",
            "",
            question,
        ]
        for idx, option in enumerate(normalized_options, start=1):
            system_lines.append(f"Option {idx}. {option}")
        system_lines.append("")
        system_lines.append(instruction)
        system_lines.append("Answer:")
        return "\n".join(system_lines)


__all__ = [
    "GameBenchmarkPromptWrapper",
    "GameDecisionPromptWrapper",
    "GameCompletionOptionIdPromptWrapper",
]
