import copy
import json
from pathlib import Path
from typing import Any, ClassVar, Dict, List, Optional

from pydantic import Field

from games.game import BehaviorChoices, GameDecision, GameScenario


class WaitGoBehaviorChoices(BehaviorChoices):
    wait: str
    go: str

    def get_choices(self):
        return [self.wait, self.go]

    def is_valid_choice(self, choice: str) -> bool:
        return choice in self.get_choices()

    def __str__(self):
        return f"Behavior Choices: {self.get_choices()}"

    @staticmethod
    def example():
        return {"wait": "Wait", "go": "Go"}


class WaitGoScenario(GameScenario):
    scenario: str
    description: str
    participants: List[Dict[str, Any]]
    behavior_choices: WaitGoBehaviorChoices
    payoff_matrix: Dict[str, Any]
    game_name: str = "Wait_Go"

    def find_behavior_from_decision(self, decision: str) -> str:
        if decision == self.behavior_choices.wait:
            return "wait"
        elif decision == self.behavior_choices.go:
            return "go"
        else:
            raise ValueError(
                f"Decision must be one of {self.behavior_choices.get_choices()}"
            )

    def get_scenario_info(self) -> dict:
        return {"scenario": self.scenario, "description": self.description}

    def get_behavior_choices(self) -> WaitGoBehaviorChoices:
        return self.behavior_choices

    @staticmethod
    def example():
        return {
            "scenario": "Traffic Intersection",
            "description": "Two drivers arrive at an intersection simultaneously. Each must decide whether to wait or go. If both wait, there's a minor delay. If both go, there's a collision. If one waits while the other goes, traffic flows smoothly.",
            "participants": [
                {
                    "name": "participant1",
                    "profile": "Driver approaching from the north",
                },
                {"name": "participant2", "profile": "Driver approaching from the east"},
            ],
            "behavior_choices": WaitGoBehaviorChoices.example(),
        }


class WaitGoDecision(GameDecision):
    scenario: ClassVar[Optional[WaitGoScenario]] = None
    decision: str = Field(..., description="The decision made in the scenario")
    rational: str = Field(..., description="The rationale for the decision")

    @classmethod
    def set_scenario(cls, scenario: GameScenario):
        if not isinstance(scenario, WaitGoScenario):
            raise ValueError("Scenario must be a WaitGoScenario")
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

    from games.payoff_matrix import wait_go as payoff_matrix

    for file in Path("groupchat/scenarios/Wait_Go").glob("*.json"):
        print(f" === begin: {file.name} ===\n")
        with open(file, "r") as f:
            data = json.load(f)
            data["payoff_matrix"] = payoff_matrix
            scenario = WaitGoScenario(**data)

            WaitGoDecision.set_scenario(scenario)

            for config in cfg_ls_cp:
                config["response_format"] = WaitGoDecision

            assistant = AssistantAgent(
                name="Alice",
                llm_config={
                    "config_list": cfg_ls_cp,
                    "temperature": 0.9,
                },
                system_message="You are Alice, a driver approaching an intersection. Consider safety and efficiency in your decision.",
            )

            message = f"Please analyze the following scenario: {scenario} and make your decision."
            while True:
                try:
                    res = user.initiate_chat(assistant, message=message, max_turns=1)
                    decision = WaitGoDecision.model_validate_json(res.summary)
                    break
                except Exception as e:
                    print(f" === error: {e} ===")
                    message = (
                        f"Error: {e}\nPlease analyze the scenario again: {scenario}"
                    )

            behavior = scenario.find_behavior_from_decision(decision.decision)
            assert behavior is not None, f"Invalid decision: {decision.decision}"
            print(f" === behavior: {behavior} ===")
