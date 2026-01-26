"""
neuro_manipulation/prompt_wrapper.py: Ensure GameReactPromptWrapper selects persona for Diplomacy scenarios.
"""

import unittest

from neuro_manipulation.prompt_wrapper import GameReactPromptWrapper


class _DummyPromptFormat:
    def build(self, system_prompt, user_messages, enable_thinking=False):
        if isinstance(user_messages, str):
            user_messages = [user_messages]
        return system_prompt + "\n" + "\n".join(user_messages)


class _MockGameDecision:
    @staticmethod
    def example():
        return '{"decision": "choice", "rationale": "reason", "option_id": 1}'


class TestGameReactPromptWrapperPersonaSelection(unittest.TestCase):
    def test_defaults_to_alice(self):
        wrapper = GameReactPromptWrapper(_DummyPromptFormat(), _MockGameDecision)
        prompt = wrapper.system_prompt("Scenario: Prisoners_Dilemma_01", ["Cooperate", "Defect"])
        self.assertIn("You are Alice.", prompt)

    def test_diplomacy_scenario_uses_commander_persona(self):
        wrapper = GameReactPromptWrapper(_DummyPromptFormat(), _MockGameDecision)
        prompt = wrapper.system_prompt("Scenario: Diplomacy_Test_01", ["Attack", "Hold"])
        self.assertIn("You are a commander who needs to make the decision.", prompt)


if __name__ == "__main__":
    unittest.main()
