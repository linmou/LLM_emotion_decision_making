# Tests for neuro_manipulation/prompt_wrapper.py: ensure GameReactPromptWrapper seeds
# an incomplete assistant response stub to elicit immediate choices in game prompts.

from neuro_manipulation.prompt_wrapper import GameReactPromptWrapper


class _StubPromptFormat:
    def __init__(self):
        self.calls = []

    def build(
        self,
        system_prompt,
        user_messages,
        assistant_messages=None,
        enable_thinking=False,
        add_generation_prompt=True,
        images=None,
    ):
        self.calls.append(
            {
                "system": system_prompt,
                "user_messages": user_messages,
                "assistant_messages": assistant_messages,
                "enable_thinking": enable_thinking,
                "add_generation_prompt": add_generation_prompt,
                "images": images,
            }
        )
        return "PROMPT"


class _DummyDecision:
    @staticmethod
    def example():
        return '{"decision": "Option 1"}'


def test_game_react_wrapper_prefills_incomplete_assistant_response():
    prompt_format = _StubPromptFormat()
    wrapper = GameReactPromptWrapper(prompt_format, _DummyDecision)

    result = wrapper(
        "A tense negotiation is underway.",
        ["Cooperate", "Defect"],
        "Pick now",
        enable_thinking=False,
    )

    assert result == "PROMPT"
    call = prompt_format.calls[-1]
    assert call["assistant_messages"] in (None, [], [])
    assert call["user_messages"] == ["Pick now\nassistant: I choose option "]
    assert call["add_generation_prompt"] is True
