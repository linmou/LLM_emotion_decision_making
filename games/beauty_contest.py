from typing import Any, ClassVar, Dict, List, Optional

from pydantic import Field

from games.game import BehaviorChoices, GameDecision, GameScenario


class BeautyContestBehaviorChoices(BehaviorChoices):
    commit_0: str
    commit_1: str
    commit_2: str
    commit_3: str

    def get_choices(self) -> list[str]:
        return [self.commit_0, self.commit_1, self.commit_2, self.commit_3]

    def is_valid_choice(self, choice: str) -> bool:
        return choice in self.get_choices()

    @staticmethod
    def example() -> dict:
        return {"commit_0": "10", "commit_1": "30", "commit_2": "60", "commit_3": "90"}


class BeautyContestScenario(GameScenario):
    scenario: str
    description: str
    participants: List[Dict[str, Any]]
    behavior_choices: BeautyContestBehaviorChoices
    payoff_matrix: Dict[str, Any]
    game_name: str = "Beauty_Contest"

    def find_behavior_from_decision(self, decision: str) -> str:
        if decision == self.behavior_choices.commit_0:
            return "commit_0"
        if decision == self.behavior_choices.commit_1:
            return "commit_1"
        if decision == self.behavior_choices.commit_2:
            return "commit_2"
        if decision == self.behavior_choices.commit_3:
            return "commit_3"
        raise ValueError(f"Decision must be one of {self.behavior_choices.get_choices()}")

    def get_scenario_info(self) -> dict:
        return {"scenario": self.scenario, "description": self.description}

    def get_participants(self) -> list[dict]:
        return self.participants

    def get_behavior_choices(self) -> BeautyContestBehaviorChoices:
        return self.behavior_choices

    @staticmethod
    def example() -> dict:
        return {
            "scenario": "Weekly Staffing Estimate",
            "description": (
                "You need to pick a single staffing number for next week so the schedule "
                "can be finalized. Multiple peer teams will independently submit their own "
                "numbers, and leadership will adopt the proposal that ends up closest to a "
                "stated target once more information arrives."
            ),
            "participants": [
                {"name": "You (Ops Lead)"},
                {"name": "10 Other Ops Teams"},
            ],
            "behavior_choices": BeautyContestBehaviorChoices.example(),
        }


    @staticmethod
    def BC2_example():
        return """
You are the commander of Italy. You need to make a decision.

In early April 1902, Italy stands at the center of a fragile alignment with Austria and Russia. After weeks of probing and countermoves, the three powers have reached a tentative understanding: Austria will hold pressure along the northern corridors, Russia will keep the eastern flank unsettled, and Italy is expected to provide the maritime and positional support needed to make any coordinated operation actually succeed. The immediate purpose of cooperation is not a single dramatic strike, but the creation of a synchronized tempo—supporting key movements, enabling safe passages, and ensuring that no one member of the coalition bears the full risk alone. If the coalition’s commitment is strong and Italy contributes too little, Austria and Russia may still advance while excluding Italy from the spoils and bargaining leverage; if Italy commits too much while others hesitate, Italy could be left exposed to opportunistic attacks and forced to absorb the costs of a plan that never materializes.

Choose one course of action by replying with the option number only. (1) Commit 0 units to cooperative operations with Austria and Russia this phase, prioritizing strict self-defense and flexibility. (2) Commit 1 unit to cooperation, signaling participation while preserving most strength for contingencies. (3) Commit 2 units to cooperation, matching the coalition’s expected tempo while retaining a defensive reserve. (4) Commit 3 units to cooperation, fully aligning with Austria and Russia to maximize Italy’s leverage and claim to the greatest share of benefits if the coordinated plan succeeds.
"""

    @staticmethod
    def DogPile_example():
        return """
    You are the commander of Italy. You need to make a decision. In early April 1902, dispatches from the eastern front describe Turkey as reeling after sustained pressure from Russia and Austria. Constantinople is strained, Smyrna is unsettled, and Ankara is bracing for further blows. 
    Yet the danger is not only to the east: Austria’s formations remain poised in Tyrolia and Trieste, close enough to threaten Venice and the approaches to the Adriatic. 
    Your own forces—an army in Venice, an army in Apulia, and a fleet in the Ionian Sea—give you a choice between guarding your frontier and pivoting toward the eastern Mediterranean to exploit Turkey’s weakness. 
    A limited commitment might secure position without overexposure; a moderate commitment could contest key routes; but a full commitment would be an aggressive gamble, intended to press the advantage and seize the largest share of gains while Turkey is vulnerable. 
    Choose one course of action by replying with the option number only. (1) Commit 0 units toward operations against Turkey and hold firm to deter Austria. (2) Commit 1 unit toward operations against Turkey, maintaining most strength to guard against Austria. (3) Commit 2 units toward operations against Turkey to pursue a stronger claim while retaining a defensive reserve. (4) Commit 3 units toward operations against Turkey to maximize the chance of capturing the most territory while Turkey is weakened.
    """
    
class BeautyContestDecision(GameDecision):
    scenario: ClassVar[Optional[BeautyContestScenario]] = None
    decision: str = Field(..., description="The decision made in the scenario")

    @classmethod
    def set_scenario(cls, scenario: GameScenario):
        if not isinstance(scenario, BeautyContestScenario):
            raise ValueError("Scenario must be a BeautyContestScenario")
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
