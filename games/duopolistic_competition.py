import copy
import json
from pathlib import Path
from typing import Any, ClassVar, Dict, List, Optional

from pydantic import Field

from games.game import BehaviorChoices, GameDecision, GameScenario


class DuopolyBehaviorChoices(BehaviorChoices):
    high_price: str
    low_price: str

    def get_choices(self):
        return [self.high_price, self.low_price]

    def is_valid_choice(self, choice: str) -> bool:
        return choice in self.get_choices()

    def __str__(self):
        return f"Behavior Choices: {self.get_choices()}"

    @staticmethod
    def example():
        return {"high_price": "Premium Pricing", "low_price": "Competitive Pricing"}


class DuopolisticCompetitionScenario(GameScenario):
    scenario: str
    description: str
    participants: List[Dict[str, Any]]
    behavior_choices: DuopolyBehaviorChoices
    payoff_matrix: Dict[str, Any]
    game_name: str = "Duopolistic_Competition"

    def find_behavior_from_decision(self, decision: str) -> str:
        if decision == self.behavior_choices.high_price:
            return "high_price"
        elif decision == self.behavior_choices.low_price:
            return "low_price"
        else:
            raise ValueError(
                f"Decision must be one of {self.behavior_choices.get_choices()}"
            )

    def get_scenario_info(self) -> dict:
        return {"scenario": self.scenario, "description": self.description}

    def get_behavior_choices(self) -> DuopolyBehaviorChoices:
        return self.behavior_choices

    @staticmethod
    def example():
        return {
            "scenario": "Market Competition",
            "description": "Two companies dominate a market and must set their prices. Each can choose premium or competitive pricing. If both choose premium pricing, they share high profits. If one chooses competitive while the other stays premium, the competitive pricer gains market share. If both choose competitive pricing, they both earn lower profits.",
            "participants": [
                {"name": "participant1", "profile": "Company A CEO"},
                {"name": "participant2", "profile": "Company B CEO"},
            ],
            "behavior_choices": DuopolyBehaviorChoices.example(),
        }


class DuopolisticCompetitionDecision(GameDecision):
    scenario: ClassVar[Optional[DuopolisticCompetitionScenario]] = None
    decision: str = Field(..., description="The decision made in the scenario")
    rational: str = Field(..., description="The rationale for the decision")

    @classmethod
    def set_scenario(cls, scenario: GameScenario):
        if not isinstance(scenario, DuopolisticCompetitionScenario):
            raise ValueError("Scenario must be a DuopolisticCompetitionScenario")
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

    from games.payoff_matrix import duopolistic_competition as payoff_matrix

    for file in Path("groupchat/scenarios/Duopolistic_Competition").glob("*.json"):
        print(f" === begin: {file.name} ===\n")
        with open(file, "r") as f:
            data = json.load(f)
            data["payoff_matrix"] = payoff_matrix
            scenario = DuopolisticCompetitionScenario(**data)

            DuopolisticCompetitionDecision.set_scenario(scenario)

            for config in cfg_ls_cp:
                config["response_format"] = DuopolisticCompetitionDecision

            assistant = AssistantAgent(
                name="Alice",
                llm_config={
                    "config_list": cfg_ls_cp,
                    "temperature": 0.9,
                },
                system_message="You are Alice, a CEO making pricing decisions. Consider market dynamics and competitor behavior.",
            )

            message = f"Please analyze the following scenario: {scenario} and make your decision."
            while True:
                try:
                    res = user.initiate_chat(assistant, message=message, max_turns=1)
                    decision = DuopolisticCompetitionDecision.model_validate_json(
                        res.summary
                    )
                    break
                except Exception as e:
                    print(f" === error: {e} ===")
                    message = (
                        f"Error: {e}\nPlease analyze the scenario again: {scenario}"
                    )

            behavior = scenario.find_behavior_from_decision(decision.decision)
            assert behavior is not None, f"Invalid decision: {decision.decision}"
            print(f" === behavior: {behavior} ===")
