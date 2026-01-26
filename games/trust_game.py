import random
from typing import Any, ClassVar, Dict, Optional, Union, cast

from pydantic import Field, model_validator

from games.game import (
    BehaviorChoices,
    GameDecision,
    GameScenario,
    SequentialGameScenario,
)
from games.payoff_matrices import PayoffMatrix


class TGTrustorChoices(BehaviorChoices):
    trust_none: str
    trust_low: str
    trust_high: str

    def get_choices(self):
        return [self.trust_none, self.trust_low, self.trust_high]

    def is_valid_choice(self, choice: str) -> bool:
        return choice in self.get_choices()

    def __str__(self):
        return f"Behavior Choices: {self.get_choices()}"

    @staticmethod
    def example():
        return {
            "trust_none": "<trust_none action>",
            "trust_low": "<trust_low action>",
            "trust_high": "<trust_high action>",
        }


class TGTrusteeChoices(BehaviorChoices):
    return_none: str
    return_medium: str
    return_high: str

    def get_choices(self):
        return [self.return_none, self.return_medium, self.return_high]

    def is_valid_choice(self, choice: str) -> bool:
        return choice in self.get_choices()

    def __str__(self):
        return f"Behavior Choices: {self.get_choices()}"

    @staticmethod
    def example():
        return {
            "return_none": "<return_none action>",
            "return_medium": "<return_medium action>",
            "return_high": "<return_high action>",
        }


class TrustGameScenario(SequentialGameScenario):
    scenario: str
    description: str
    participants: list[dict]
    trustor_behavior_choices: TGTrustorChoices
    trustee_behavior_choices: TGTrusteeChoices
    previous_actions_length: int
    payoff_matrix: Dict[tuple[str, str], Any]
    game_name: str = "Trust_Game"

    def get_scenario_info(self) -> dict:
        return {"scenario": self.scenario, "description": self.description}

    def get_participants(self) -> list[dict]:
        return self.participants

    def get_payoff_matrix(self) -> Dict[str, Any]:
        return cast(Dict[str, Any], self.payoff_matrix)

    def find_behavior_from_decision(self, decision: str) -> str:
        for attr, value in self.get_behavior_choices().__dict__.items():
            if value == decision:
                return attr
        raise ValueError(f"Invalid decision: {decision}")

    @model_validator(mode="after")
    def _validate_participants(self) -> "TrustGameScenario":
        if not isinstance(self.participants, list) or not self.participants:
            raise ValueError("Participants must be a non-empty list")

        role_map: Dict[str, dict] = {}
        for participant in self.participants:
            if not isinstance(participant, dict):
                continue
            role = participant.get("role")
            if role:
                role_map.setdefault(role, participant)

        missing_roles = [role for role in ("Trustor", "Trustee") if role not in role_map]
        if missing_roles:
            raise ValueError(
                "Missing participant role(s): " + ", ".join(sorted(missing_roles))
            )

        for role in ("Trustor", "Trustee"):
            participant = role_map.get(role, {})
            name = participant.get("name")
            if not name:
                raise ValueError(f"Participant role '{role}' missing required 'name'")

        return self

    @property
    def previous_actions(self) -> list:  # type: ignore[override]
        assert (
            self.previous_actions_length == 0
        ), "Currently trust game does not have previous actions"
        return []

    @property
    def trustor_name(self) -> str:
        return next(
            participant["name"]
            for participant in self.participants
            if participant["role"] == "Trustor"
        )

    @property
    def trustee_name(self) -> str:
        return next(
            participant["name"]
            for participant in self.participants
            if participant["role"] == "Trustee"
        )

    def __str__(self):
        info = self.get_scenario_info()
        previous_actions = self.previous_actions
        previous_actions_line = (
            f"\n        Previous Actions: {previous_actions}" if previous_actions else ""
        )
        return f"""
        Scenario: {info.get('scenario', 'Unnamed')}
        Description: {info.get('description', 'No description')}
        Participants: {self.get_participants()}
        Behavior Choices: {self.get_behavior_choices().get_choices()}
        {previous_actions_line}
        """

    @staticmethod
    def example():
        return {
            "scenario": "Seed_Sharing_Harvest",
            "description": "Alice has 10 seeds to plant. Alice may give some/all seeds to her neighbor Bob to grow in his fertile field. Planted seeds yield triple the harvest (e.g., 1 seed → 3 crops). Seed and crop are at similar value. Bob then decides how much of the harvest to give back to Alice.",
            "participants": [
                {"name": "Alice", "profile": "Seed_Sharer", "role": "Trustor"},
                {"name": "Bob", "profile": "Crop_Partner", "role": "Trustee"},
            ],
            "trustor_behavior_choices": {
                "trust_none": "share 0 seeds with neighbor",
                "trust_low": "share 30% seeds with neighbor",
                "trust_high": "share more that 80% seeds with neighbor",
            },
            "trustee_behavior_choices": {
                "return_none": "return 0 corps to seed sharer",
                "return_medium": "return 40-50% of the harvest to seed sharer",
                "return_high": "return more than 80% of the harvest to seed sharer",
            },
        }

    @staticmethod
    def specific_prompt() -> str:
        """
        return a prompt for scenario generation
        """

        return """
        When generating the choices, use specific percetage to describe the number of give and return.
        """


class TrustGameTrustorScenario(TrustGameScenario):

    def get_scenario_info(self) -> Dict:
        return {
            "scenario": self.scenario,
            "description": self.description.replace(self.trustor_name, "You"),
        }

    def get_behavior_choices(self) -> TGTrustorChoices:
        return self.trustor_behavior_choices

    def get_participants(self) -> list[Any]:  # type: ignore[override]
        return [
            (
                ("You", participant["profile"])
                if participant["role"] == "Trustor"
                else (participant["name"], participant["profile"])
            )
            for participant in self.participants
        ]


class TrustGameTrusteeScenario(TrustGameScenario):
    previous_trust_level: int = Field(ge=0, le=2)

    def get_scenario_info(self) -> Dict:
        return {
            "scenario": self.scenario,
            "description": self.description.replace(self.trustee_name, "You"),
        }

    def get_behavior_choices(self) -> TGTrusteeChoices:
        return self.trustee_behavior_choices

    def get_participants(self) -> list[Any]:  # type: ignore[override]
        return [
            (
                ("You", participant["profile"])
                if participant["role"] == "Trustee"
                else (participant["name"], participant["profile"])
            )
            for participant in self.participants
        ]

    @property
    def previous_actions(self) -> list:  # type: ignore[override]
        if self.previous_trust_level == 0:
            return [(self.trustor_name, self.trustor_behavior_choices.trust_low)]
        elif self.previous_trust_level == 1:
            return [(self.trustor_name, self.trustor_behavior_choices.trust_high)]
        else:
            raise ValueError(
                f"Invalid previous trust level: {self.previous_trust_level}"
            )


class TrustGameDecision(GameDecision):
    scenario: ClassVar[Optional[TrustGameScenario]] = None
    decision: str = Field(..., description="The decision made in the scenario")
    # rational: str = Field(..., description="The rationale for the decision")

    @classmethod
    def set_scenario(cls, scenario: GameScenario):
        if not isinstance(scenario, TrustGameScenario):
            raise ValueError("Scenario must be a TrustGameScenario")
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

    @property
    def rational(self) -> str:
        return ""


if __name__ == "__main__":
    import copy
    import json
    from pathlib import Path

    # from autogen import AssistantAgent, UserProxyAgent

    # Example usage
    data_json = "data_creation/scenario_creation/langgraph_creation/Trust_Game_Trustor_all_data_samples.json"
    with open(data_json, "r") as f:
        data = json.load(f)[1]

    # Import your trust game payoff matrix
    from games.game_configs import get_game_config
    game_config = get_game_config('Trust_Game_Trustor')
    data["payoff_matrix"] = game_config["payoff_matrix"]
    data["previous_actions_length"] = 0
    data["previous_trust_level"] = 0

    scenario = TrustGameTrusteeScenario.model_validate(data)
    print(scenario)
    
    scenario = TrustGameTrustorScenario.model_validate(data)
    print(scenario)

    # from autogen import config_list_from_json

    # config_path = "config/OAI_CONFIG_LIST"
    # config_list = config_list_from_json(config_path, filter_dict={"model": ["gpt-4o"]})
    # cfg_ls_cp = copy.deepcopy(config_list)

    # user = UserProxyAgent(
    #     name="User",
    #     human_input_mode="NEVER",
    #     code_execution_config={"use_docker": False},
    # )

    # # Process all scenario files
    # for file in Path("groupchat/scenarios/Trust_Game_Trustor").glob("*.json"):
    #     print(f" === begin: {file.name} ===\n")
    #     with open(file, "r") as f:
    #         data = json.load(f)
    #         data["payoff_matrix"] = trust_game
    #         data["previous_actions_length"] = 0
    #         data["previous_trust_level"] = 1
    #         scenario = TrustGameTrusteeScenario(**data)

    #         TrustGameDecision.set_scenario(scenario)

    #         for config in cfg_ls_cp:
    #             config["response_format"] = TrustGameDecision

    #         assistant = AssistantAgent(
    #             name="Alice",
    #             llm_config={
    #                 "config_list": cfg_ls_cp,
    #                 "temperature": 0.7,
    #             },
    #             system_message="You are Alice, a rational decision-maker in a trust-based scenario.",
    #         )

    #         message = f"Please analyze the following scenario: {scenario} and make your decision."
    #         while True:
    #             try:
    #                 res = user.initiate_chat(assistant, message=message, max_turns=1)
    #                 decision = TrustGameDecision.model_validate_json(res.summary)
    #                 break
    #             except Exception as e:
    #                 print(f" === error: {e} ===")
    #                 message = f" === Please note that in previous attempt, you made the following error: {e} ===\nPlease analyze the following scenario: {scenario} and make your decision."

    #         behavior = scenario.find_behavior_from_decision(decision.decision)
    #         assert (
    #             behavior is not None
    #         ), f"decision: {decision.decision} is not in the behavior choices"
    #         print(f" === behavior: {behavior} ===")
