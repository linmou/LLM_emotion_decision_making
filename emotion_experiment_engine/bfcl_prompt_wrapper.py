"""
BFCLPromptWrapper - Prompt wrapper for BFCL function-calling tasks.

Emits strict instructions to output JSON-only function call(s), with the provided
tool schema included in the system prompt. Supports:
- live_simple: exactly one call in a one-element JSON array
- live_multiple: multiple calls in a JSON array in execution order
"""

import json
from typing import Any, Dict, List, Optional, Union, TYPE_CHECKING

from neuro_manipulation.prompt_wrapper import PromptWrapper

if TYPE_CHECKING:
    from neuro_manipulation.prompt_formats import PromptFormat


class BFCLPromptWrapper(PromptWrapper):
    def __init__(self, prompt_format: "PromptFormat", task_type: str) -> None:
        super().__init__(prompt_format)
        self.task_type = task_type

    def _format_tools_block(self, context: Optional[str]) -> str:
        if not context:
            return ""
        try:
            ctx = json.loads(context)
            tools: List[Dict[str, Any]] = ctx.get("tools", []) if isinstance(ctx, dict) else []
        except Exception:
            # Fallback: raw context
            return f"\n\nTools (JSON Schema):\n{context}"

        if not isinstance(tools, list) or not tools:
            return ""

        lines = ["\n\nTools (JSON Schema):"]
        for tool in tools:
            name = tool.get("name", "<unknown>")
            params = tool.get("parameters", {})
            pretty_params = json.dumps(params, ensure_ascii=False, indent=2)
            lines.append(f"- name: {name}\n  parameters: {pretty_params}")
        return "\n".join(lines)

    def system_prompt(
        self,
        context: Optional[str],
        question: Optional[str],
        options: Optional[list] = None,
    ) -> str:
        tools_block = self._format_tools_block(context)
        task_lower = (self.task_type or "").lower()

        if task_lower == "live_multiple":
            format_rule = (
                "Return only a JSON array of function calls in execution order. "
                "Each element must be an object like {\"<function_name>\": {<param>: <value>, ...}}. "
                "Begin response with '[' and end with ']'. No extra text."
            )
            example_rule = (
                "Example: [{\"tool_a\": {\"arg1\": \"val\"}}, {\"tool_b\": {\"x\": 1}}]"
            )
        else:
            format_rule = (
                "Return only a JSON array with a single object representing one function call, "
                "formatted exactly as [{\"<function_name>\": {<param>: <value>, ...}}]. "
                "Begin response with '[' and end with ']'. No extra text."
            )
            example_rule = (
                "Example: [{\"get_current_weather\": {\"location\": \"Tel Aviv, Israel\"}}]"
            )

        instructions = [
            "You are a function-calling assistant.",
            "Use the provided tool schema to construct valid JSON function call(s).",
            format_rule,
            "Rules:",
            "- Return valid JSON only (no prose, no code fences/backticks).",
            "- Use exact function name(s) and parameter names from the schema.",
            "- Do NOT include wrapper keys like 'name' or 'parameters' in the output.",
            "- Do not include extra keys beyond those in the schema.",
            "- Omit optional parameters only if appropriate.",
            example_rule,
        ]

        prompt = " ".join(instructions)
        if tools_block:
            prompt += tools_block
        # Question will be provided as the user turn; don't duplicate here
        return prompt

    def user_messages(self, user_messages):
        if isinstance(user_messages, str):
            return [user_messages]
        return user_messages

    def __call__(
        self,
        context: Optional[str] = None,
        question: Optional[str] = None,
        user_messages: Union[str, list] = "Please provide your answer.",
        enable_thinking: bool = False,
        augmentation_config: Optional[Dict[str, Any]] = None,
        answer: Optional[str] = None,
        emotion: Optional[str] = None,
        options: Optional[list] = None,
    ) -> str:
        # Route the actual question into the user turn for better adherence
        user_msgs: List[str]
        if question and isinstance(question, str) and question.strip():
            user_msgs = [question]
        else:
            user_msgs = self.user_messages(user_messages)

        return self.prompt_format.build(
            self.system_prompt(context, None, options),
            user_msgs,
            enable_thinking=enable_thinking,
        )
