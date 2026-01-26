import copy
import json
from pathlib import Path
from typing import Any, ClassVar, Dict, List, Optional

from pydantic import Field

from games.game import BehaviorChoices, GameDecision, GameScenario
from games.payoff_matrices import PayoffMatrix


class StagHuntBehaviors(BehaviorChoices):
    cooperate: str
    defect: str

    def get_choices(self):
        return [self.cooperate, self.defect]

    def is_valid_choice(self, choice: str) -> bool:
        return choice in self.get_choices()

    @staticmethod
    def example():
        return {
            "cooperate": "Collaborate intensely on the joint project",
            "defect": "Focus solely on individual assignments",
        }


class StagHuntScenario(GameScenario):
    scenario: str
    description: str
    participants: List[Dict[str, Any]]
    behavior_choices: StagHuntBehaviors
    payoff_matrix: PayoffMatrix
    game_name: str = "Stag_Hunt"

    def find_behavior_from_decision(self, decision: str) -> str:
        if decision == self.behavior_choices.cooperate:
            return "cooperate"
        elif decision == self.behavior_choices.defect:
            return "defect"
        else:
            raise ValueError(
                f"Decision must be one of {self.behavior_choices.get_choices()}"
            )

    def get_scenario_info(self) -> dict:
        return {"scenario": self.scenario, "description": self.description}

    def get_behavior_choices(self) -> StagHuntBehaviors:
        return self.behavior_choices

    @staticmethod
    def example():
        return {
            "scenario": "Project Partnership: The High-Stakes Deadline",
            "participants": [
                {"name": "You", "profile": "Software Engineer"},
                {"name": "Bob", "profile": "Software Engineer"},
            ],
            "behavior_choices": StagHuntBehaviors.example(),
            "payoff_matrix_description": {
                f"player 1: cooperate , player 2: cooperate": [
                    "player 1 gets 3: Major project success, potential promotion",
                    "player 2 gets 3: Major project success, potential promotion",
                ],
                f"player 1: cooperate , player 2: defect": [
                    "player 1 gets 0: Collaboration effort wasted; player 2 gets 1: Individual tasks completed successfully",
                    "player 2 gets 1: Individual tasks completed successfully",
                ],
                f"player 1: defect , player 2: cooperate": [
                    "player 1 gets 1: Individual tasks completed successfully",
                    "player 2 gets 0: Collaboration effort wasted",
                ],
                f"player 1: defect , player 2: defect": [
                    "player 1 gets 1: Basic requirements met",
                    "player 2 gets 1: Basic requirements met",
                ],
            },
            "description": "Two colleagues, You and Bob, are tasked with a critical project with a tight deadline. They can either collaborate intensely to achieve a major success potentially leading to promotions, or focus on their individual assignments ensuring they meet their basic requirements. If one commits to collaboration while the other focuses individually, the collaborator's efforts are largely wasted, while the individual worker secures their moderate success.",
        }


class StagHuntDecision(GameDecision):
    scenario: ClassVar[Optional[StagHuntScenario]] = None
    decision: str = Field(..., description="The decision made in the scenario")

    @classmethod
    def set_scenario(cls, scenario: GameScenario):
        if not isinstance(scenario, StagHuntScenario):
            raise ValueError("Scenario must be a StagHuntScenario")
        cls.scenario = scenario
        cls.model_fields["decision"].json_schema_extra = {
            "choices": scenario.behavior_choices.get_choices()
        }

    def validate_decision(self, decision: str) -> bool:
        if not self.scenario:
            raise ValueError(
                "Scenario must be set using Decision.set_scenario() before validating"
            )
        return self.scenario.behavior_choices.is_valid_choice(decision)

    # def __init__(self, **data):
    #     if not self.scenario:
    #         raise ValueError("Scenario must be set using Decision.set_scenario() before creating instances")
    #     if not self.validate_decision(data.get('decision')):
    #         raise ValueError(f"Decision must be one of {self.scenario.behavior_choices.get_choices()}")
    #     super().__init__(**data)

    @property
    def rational(self) -> str:
        return ""


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

    from games.payoff_matrix import stag_hunt as payoff_matrix

    for file in Path("groupchat/scenarios/Stag_Hunt").glob("*.json"):
        print(f" === begin: {file.name} ===\n")
        with open(file, "r") as f:
            data = json.load(f)
            data["payoff_matrix"] = payoff_matrix
            scenario = StagHuntScenario(**data)

            StagHuntDecision.set_scenario(scenario)

            for config in cfg_ls_cp:
                config["response_format"] = StagHuntDecision

            assistant = AssistantAgent(
                name="Alice",
                llm_config={
                    "config_list": cfg_ls_cp,
                    "temperature": 0.9,
                },
                system_message="You are Alice, an average American. Consider your partner's reliability when making decisions.",
            )

            message = f"Please analyze the following scenario: {scenario} and make your decision."
            while True:
                try:
                    res = user.initiate_chat(assistant, message=message, max_turns=1)
                    decision = StagHuntDecision.model_validate_json(res.summary)
                    break
                except Exception as e:
                    print(f" === error: {e} ===")
                    message = (
                        f"Error: {e}\nPlease analyze the scenario again: {scenario}"
                    )

            behavior = scenario.find_behavior_from_decision(decision.decision)
            assert behavior is not None, f"Invalid decision: {decision.decision}"
            print(f" === behavior: {behavior} ===")
