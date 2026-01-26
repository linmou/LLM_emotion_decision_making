import copy
import json
from pathlib import Path
from typing import Any, ClassVar, Dict, List, Optional

from pydantic import Field

from games.game import BehaviorChoices, GameDecision, GameScenario


class BOSBehaviorChoices(BehaviorChoices):
    option_a: str
    option_b: str

    def get_choices(self):
        return [self.option_a, self.option_b]

    def is_valid_choice(self, choice: str) -> bool:
        return choice in self.get_choices()

    def __str__(self):
        return f"Behavior Choices: {self.get_choices()}"

    @staticmethod
    def example():
        return {"option_a": "Opera", "option_b": "Movie"}


class BattleOfSexesScenario(GameScenario):
    scenario: str
    description: str
    participants: List[Dict[str, Any]]
    behavior_choices: BOSBehaviorChoices
    payoff_matrix: Dict[str, Any]
    game_name: str = "Battle_Of_Sexes"

    def find_behavior_from_decision(self, decision: str) -> str:
        if decision == self.behavior_choices.option_a:
            return "option_a"
        elif decision == self.behavior_choices.option_b:
            return "option_b"
        else:
            raise ValueError(
                f"Decision must be one of {self.behavior_choices.get_choices()}"
            )

    def get_scenario_info(self) -> dict:
        return {"scenario": self.scenario, "description": self.description}

    def get_behavior_choices(self) -> BOSBehaviorChoices:
        return self.behavior_choices

    @staticmethod
    def example():
        return {
            "scenario": "Evening Entertainment",
            "description": "A couple needs to decide on evening entertainment. One prefers the opera while the other prefers a movie. They both prefer being together over being apart, but each has a different preferred activity.",
            "participants": [
                {"name": "participant1", "profile": "Opera enthusiast"},
                {"name": "participant2", "profile": "Movie buff"},
            ],
            "behavior_choices": BOSBehaviorChoices.example(),
        }


class BattleOfSexesDecision(GameDecision):
    scenario: ClassVar[Optional[BattleOfSexesScenario]] = None
    decision: str = Field(..., description="The decision made in the scenario")
    rational: str = Field(..., description="The rationale for the decision")

    @classmethod
    def set_scenario(cls, scenario: GameScenario):
        if not isinstance(scenario, BattleOfSexesScenario):
            raise ValueError("Scenario must be a BattleOfSexesScenario")
        cls.scenario = scenario
        cls.model_fields["decision"].json_schema_extra = {
            "choices": scenario.get_behavior_choices().get_choices()
        }

    def validate_decision(self, decision: str) -> bool:
        if not self.scenario:
            raise ValueError(
                "Scenario must be set using Decision.set_scenario() before validating"
            )
        return self.scenario.get_behavior_choices().is_valid_choice(decision)


if __name__ == "__main__":
    from autogen import config_list_from_json

    config_path = "config/OAI_CONFIG_LIST"
    config_list = config_list_from_json(config_path, filter_dict={"model": ["gpt-4o"]})
    cfg_ls_cp = copy.deepcopy(config_list)
    user = UserProxyAgent(
        name="User",
        human_input_mode="NEVER",
        code_execution_config={"use_docker": False},
    )

    from games.payoff_matrix import battle_of_sexes as payoff_matrix

    for file in Path("groupchat/scenarios/Battle_Of_Sexes").glob("*.json"):
        print(f" === begin: {file.name} ===\n")
        with open(file, "r") as f:
            data = json.load(f)
            data["payoff_matrix"] = payoff_matrix
            scenario = BattleOfSexesScenario(**data)

            BattleOfSexesDecision.set_scenario(scenario)

            for config in cfg_ls_cp:
                config["response_format"] = BattleOfSexesDecision

            assistant = AssistantAgent(
                name="Alice",
                llm_config={
                    "config_list": cfg_ls_cp,
                    "temperature": 0.9,
                },
                system_message="You are Alice, trying to coordinate evening plans with your partner. Consider both your preferences and the desire to spend time together.",
            )

            message = f"Please analyze the following scenario: {scenario} and make your decision."
            while True:
                try:
                    res = user.initiate_chat(assistant, message=message, max_turns=1)
                    decision = BattleOfSexesDecision.model_validate_json(res.summary)
                    break
                except Exception as e:
                    print(f" === error: {e} ===")
                    message = (
                        f"Error: {e}\nPlease analyze the scenario again: {scenario}"
                    )

            behavior = scenario.find_behavior_from_decision(decision.decision)
            assert behavior is not None, f"Invalid decision: {decision.decision}"
            print(f" === behavior: {behavior} ===")
