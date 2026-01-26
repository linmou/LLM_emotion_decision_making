from typing import Any, ClassVar, Dict, List, Optional

from pydantic import Field

from games.game import BehaviorChoices, GameDecision, GameScenario


class SealedAuctionBehaviorChoices(BehaviorChoices):
    devote_low: str
    devote_medium: str
    devote_high: str

    def get_choices(self) -> list[str]:
        return [self.devote_low, self.devote_medium, self.devote_high]

    def is_valid_choice(self, choice: str) -> bool:
        return choice in self.get_choices()

    @staticmethod
    def example() -> dict:
        return {
            "devote_low": "Devote low resources.",
            "devote_medium": "Devote medium resources.",
            "devote_high": "Devote high resources.",
        }


class SealedAuctionScenario(GameScenario):
    scenario: str
    description: str
    participants: List[Dict[str, Any]]
    behavior_choices: SealedAuctionBehaviorChoices
    payoff_matrix: Dict[str, Any]
    game_category: str
    game_name: str = "Sealed_Auction"

    def find_behavior_from_decision(self, decision: str) -> str:
        if decision == self.behavior_choices.devote_low:
            return "devote_low"
        if decision == self.behavior_choices.devote_medium:
            return "devote_medium"
        if decision == self.behavior_choices.devote_high:
            return "devote_high"
        raise ValueError(f"Decision must be one of {self.behavior_choices.get_choices()}")

    def get_scenario_info(self) -> dict:
        return {"scenario": self.scenario, "description": self.description}

    def get_participants(self) -> list[dict]:
        return self.participants

    def get_behavior_choices(self) -> SealedAuctionBehaviorChoices:
        return self.behavior_choices

    @staticmethod
    def example() -> dict:
        return {
            "scenario": "The Belgian Scramble",
            "description": (
                "Belgium is contested and multiple powers are converging to bid via support "
                "chains. Each commander commits resources without seeing the others' bids. "
                "Too little and the bid fails; too much and other fronts weaken."
            ),
            "participants": [
                {"name": "You (Commander of France)"},
                {"name": "Commander of England"},
                {"name": "Commander of Germany"},
                {"name": "Commander of a minor opportunist power"},
            ],
            "behavior_choices": SealedAuctionBehaviorChoices.example(),
            "game_category": "SEALED_BID_AUCTION_MULTIPARTY",
        }


class SealedAuctionDecision(GameDecision):
    scenario: ClassVar[Optional[SealedAuctionScenario]] = None
    decision: str = Field(..., description="The decision made in the scenario")

    @classmethod
    def set_scenario(cls, scenario: GameScenario):
        if not isinstance(scenario, SealedAuctionScenario):
            raise ValueError("Scenario must be a SealedAuctionScenario")
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
