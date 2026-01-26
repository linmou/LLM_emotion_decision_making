from abc import ABC, abstractmethod
from typing import Any, ClassVar, Dict, List, Optional, Type, Union

from pydantic import BaseModel, ConfigDict, Field

from constants import GameNames, GameType

from .payoff_matrices import PayoffMatrix


class BehaviorChoices(BaseModel, ABC):
    """Abstract base class for behavior choices in a game"""

    @abstractmethod
    def is_valid_choice(self, choice: str) -> bool:
        """Check if a choice is valid"""
        pass

    @abstractmethod
    def get_choices(self) -> list[str]:
        """Get the choices"""
        pass

    @staticmethod
    @abstractmethod
    def example() -> dict:
        """Provide an example of behavior choices"""
        pass

    def __str__(self):
        return ", ".join(self.get_choices())


class GameScenario(BaseModel, ABC):
    """Abstract base class for game scenarios"""

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")

    payoff_matrix: Union[Dict[str, Dict[str, Dict[str, float]]], PayoffMatrix] = Field(
        default=None
    )

    @abstractmethod
    def get_scenario_info(self) -> dict:
        """Get the scenario information"""
        pass

    def get_participants(self) -> list[dict]:
        return self.participants

    @abstractmethod
    def get_behavior_choices(self) -> BehaviorChoices:
        """Get the behavior choices"""
        pass

    @abstractmethod
    def find_behavior_from_decision(self, decision: str) -> str:
        """Convert a decision string to a behavior identifier"""
        pass

    @staticmethod
    @abstractmethod
    def example() -> dict:
        """Provide an example scenario"""
        pass

    def __str__(self):
        info = self.get_scenario_info()
        return f"""
        Scenario: {info.get('scenario', 'Unnamed')}
        Description: {info.get('description', 'No description')}
        Participants: {self.get_participants()}
        Behavior Choices: {self.get_behavior_choices().get_choices()}
        """


class SequentialGameScenario(GameScenario, ABC):
    """Base class for sequential game scenarios"""

    previous_actions_length: int

    @property
    @abstractmethod
    def previous_actions(self) -> list:
        """Get the previous actions"""
        pass


class GameDecision(BaseModel, ABC):
    """Abstract base class for game decisions"""

    @abstractmethod
    def validate_decision(self, decision: str) -> bool:
        """Validate if a decision is valid for the current scenario"""
        pass

    @staticmethod
    def example() -> dict:
        return {"rational": "<rational for the decision>", "decision": "<decision>"}


class Game:
    """Main class to handle different types of games"""

    def __init__(
        self,
        name: str,
        scenario_class: Union[GameScenario, SequentialGameScenario],
        decision_class: Type[GameDecision],
        payoff_matrix: Union[Dict[str, Any], PayoffMatrix],
        extra_attrs: Dict[str, Any] = {},
        data_path: str = None,
        data_folder: str = None,
    ):
        self.name = name
        self.scenario_class = scenario_class
        self.decision_class = decision_class
        self.payoff_matrix = payoff_matrix
        self.data_path = data_path
        self.data_folder = data_folder if data_folder else self.folder_path
        self.extra_attrs = extra_attrs

    def add_extra_attr(self, key: str, value: Any):
        self.extra_attrs.update({key: value})

    @property
    def folder_path(self) -> str:
        """Get the default folder path for scenario files"""
        return f"groupchat/scenarios/{self.name}"

    @property
    def game_type(self) -> GameType:
        """Get the game type"""
        return GameNames.from_string(self.name).game_type

    def create_scenario(self, data: dict) -> GameScenario:
        """Create a new scenario instance.

        This method creates a new scenario instance with the provided data, filtering out any
        invalid attributes that are not defined in the scenario class. The method will:
        1. Copy the input data to avoid modifications
        2. Add the game's payoff matrix and extra attributes
        3. Filter out any keys that are not valid fields in the scenario class
        4. Create and return the scenario instance

        Args:
            data (dict): Initial data for creating the scenario. Invalid keys will be ignored.

        Returns:
            GameScenario: A new instance of the game's scenario class.
        """
        # Create a copy of data to avoid modifying the original
        scenario_data = data.copy()

        # Add payoff matrix and extra attributes
        scenario_data["payoff_matrix"] = self.payoff_matrix
        scenario_data.update(self.extra_attrs)

        # Get valid field names from the scenario class
        valid_fields = set(self.scenario_class.model_fields.keys())

        # Filter out invalid keys
        filtered_data = {k: v for k, v in scenario_data.items() if k in valid_fields}

        # Create scenario with filtered data
        scenario = self.scenario_class(**filtered_data)
        self.decision_class.set_scenario(scenario)
        return scenario

    def create_decision(self, **data) -> GameDecision:
        """Create a new decision instance"""
        return self.decision_class(**data)

    @property
    def example_scenario(self) -> dict:
        """Get an example scenario for this game type"""
        return self.scenario_class.example()
