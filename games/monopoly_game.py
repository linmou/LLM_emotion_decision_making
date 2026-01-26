from typing import Any, ClassVar, Dict, Optional

from pydantic import Field

from games.game import (
    BehaviorChoices,
    GameDecision,
    GameScenario,
    SequentialGameScenario,
)


class MGBehaviorChoices(BehaviorChoices):
    choice_1: str  # For Company E: Stay out, For Company I: Accommodate
    choice_2: str  # For Company E: Enter, For Company I: Fight

    def get_choices(self):
        return [self.choice_1, self.choice_2]

    def is_valid_choice(self, choice: str) -> bool:
        return choice in self.get_choices()

    def __str__(self):
        return f"Behavior Choices: {self.get_choices()}"

    @staticmethod
    def example():
        return {"choice_1": "<first choice>", "choice_2": "<second choice>"}


class MonopolyGameScenario(SequentialGameScenario):
    scenario: str
    description: str
    participants: list[dict]
    behavior_choices: MGBehaviorChoices
    previous_actions_length: int
    payoff_matrix: Dict[str, Any]
    game_name: str = "Monopoly_Game"

    def get_scenario_info(self) -> dict:
        return {"scenario": self.scenario, "description": self.description}

    def get_participants(self) -> list[dict]:
        return self.participants

    def get_payoff_matrix(self) -> Dict[str, Any]:
        return self.payoff_matrix

    def get_participant_names(self) -> list[str]:
        return [participant["name"] for participant in self.participants]

    def get_behavior_choices(self) -> MGBehaviorChoices:
        return self.behavior_choices

    def find_behavior_from_decision(self, decision: str) -> str:
        if decision == self.behavior_choices.choice_1:
            return "choice_1"
        elif decision == self.behavior_choices.choice_2:
            return "choice_2"
        else:
            raise ValueError(f"Invalid decision: {decision}")

    @staticmethod
    def example():
        return {
            "scenario": "Market Entry Decision",
            "description": "You are Company E considering entering a market currently dominated by Company I (the incumbent). You must decide whether to enter the market or stay out. If you enter, Company I can choose to accommodate your entry or fight against it.",
            "participants": [
                {"name": "Company E", "role": "Entrant"},
                {"name": "Company I", "role": "Incumbent"},
            ],
            "behavior_choices": {
                "choice_1": "stay out of the market",
                "choice_2": "enter the market",
            },
        }

    def __str__(self):
        info = self.get_scenario_info()
        return f"""
        Scenario: {info.get('scenario', 'Unnamed')}
        Description: {info.get('description', 'No description')}
        Participants: {self.get_participants()}
        Behavior Choices: {self.get_behavior_choices().get_choices()}
        Previous Actions: {self.previous_actions}
        """


class MonopolyGameDecision(GameDecision):
    scenario: ClassVar[Optional[MonopolyGameScenario]] = None
    decision: str = Field(..., description="The decision made in the scenario")
    rational: str = Field(..., description="The rationale for the decision")

    @classmethod
    def set_scenario(cls, scenario: GameScenario):
        if not isinstance(scenario, MonopolyGameScenario):
            raise ValueError("Scenario must be a MonopolyGameScenario")
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
    import copy
    import json
    from pathlib import Path

    from autogen import AssistantAgent, UserProxyAgent

    # Example usage
    data_json = "groupchat/scenarios/Monopoly_Game/Market_Entry_Scenario.json"
    with open(data_json, "r") as f:
        data = json.load(f)
    from games.payoff_matrix import monopoly_game

    data["payoff_matrix"] = monopoly_game
    data["previous_actions_length"] = 0
    scenario = MonopolyGameScenario.model_validate(data)
    print(scenario)

    from autogen import config_list_from_json

    config_path = "config/OAI_CONFIG_LIST"
    config_list = config_list_from_json(config_path, filter_dict={"model": ["gpt-4"]})
    cfg_ls_cp = copy.deepcopy(config_list)
    user = UserProxyAgent(
        name="User",
        human_input_mode="NEVER",
        code_execution_config={"use_docker": False},
    )

    for file in Path("groupchat/scenarios/Monopoly_Game").glob("*.json"):
        print(f" === begin: {file.name} ===\n")
        with open(file, "r") as f:
            data = json.load(f)
            data["payoff_matrix"] = monopoly_game
            data["previous_actions_length"] = 1
            scenario = MonopolyGameScenario(**data)

            MonopolyGameDecision.set_scenario(scenario)

            for config in cfg_ls_cp:
                config["response_format"] = MonopolyGameDecision

            assistant = AssistantAgent(
                name="Company E",
                llm_config={
                    "config_list": cfg_ls_cp,
                    "temperature": 0.7,
                },
                system_message="You are Company E, a potential market entrant considering whether to enter a market dominated by Company I.",
            )

            message = f"Please analyze the following scenario: {scenario} and make your decision."
            while True:
                try:
                    res = user.initiate_chat(assistant, message=message, max_turns=1)
                    decision = MonopolyGameDecision.model_validate_json(res.summary)
                    break
                except Exception as e:
                    print(f" === error: {e} ===")
                    message = f" === Please note that in previous attempt, you made the following error: {e} ===\nPlease analyze the following scenario: {scenario} and make your decision."

            behavior = scenario.find_behavior_from_decision(decision.decision)
            assert (
                behavior is not None
            ), f"decision: {decision.decision} is not in the behavior choices"
            print(f" === behavior: {behavior} ===")
