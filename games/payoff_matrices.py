"""
Store payoff matrices in a unified data structure for both simultaneous and sequential games.
"""

import os
import sys
from dataclasses import field
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, model_validator

# Add the project root to the path so we can import constants
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from constants import GameNames, GameType


class PayoffLeaf(BaseModel):
    """
    A leaf of the game tree representing a payoff.
    The data is a tuple of two elements, the first element is for the first player, and the second element is for the second player.
    for example:
    PayoffLeaf(actions=("cooperate", "cooperate"), payoffs=(3, 3), ranks=(3, 3))
    means that p1's action is "cooperate", p2's action is "cooperate", p1's payoff is 3, and p2's payoff is 3.
    """

    actions: Tuple[str, ...]
    payoffs: Optional[Tuple[int, int]]  # payoffs for each player
    ranks: Optional[Tuple[int, int]] = (
        None  # ranks of this payoff for each player (will be calculated)
    )


def _parse_sequential_game_tree(
    tree: Dict[str, Any], path: List[str] = []
) -> List[PayoffLeaf]:
    """
    Recursively parse a sequential game tree to generate payoff leaves.

    Args:
        tree: The game tree as a nested dictionary.
        path: The current path of actions taken.

    Returns:
        A list of PayoffLeaf objects representing terminal nodes.
    """
    leaves = []
    for action, outcome in tree.items():
        current_path = path + [action]
        if isinstance(outcome, dict):
            leaves.extend(_parse_sequential_game_tree(outcome, current_path))
        elif isinstance(outcome, list) and all(
            isinstance(p, (int, float)) for p in outcome
        ):
            leaves.append(
                PayoffLeaf(actions=tuple(current_path), payoffs=tuple(outcome))
            )
    return leaves


class PayoffMatrix(BaseModel):
    """
    A matrix of payoffs for a game.
    """

    player_num: Optional[int] = field(default=None)
    game_type: GameType = GameType.SIMULTANEOUS
    payoff_leaves: List[PayoffLeaf]
    ordered_payoff_leaves: Optional[Dict[int, List[Tuple[str, ...]]]] = field(
        default_factory=dict
    )

    @model_validator(mode="before")
    def build_ranks(self):
        """
        Build ranks for payoffs from each player's perspective.
        For each player, sort the payoff leaves by their payoffs and assign ranks.
        """
        # Handle dict input case during validation
        if isinstance(self, dict):
            payoffs: List[PayoffLeaf] = self["payoff_leaves"]
            game_type = self.get("game_type", GameType.SIMULTANEOUS)
            player_num: int | None = self.get("player_num")

            if game_type == GameType.SIMULTANEOUS and payoffs:
                player_num_from_leaves = len(payoffs[0].actions)
                if player_num is None:
                    player_num = player_num_from_leaves
                else:
                    assert (
                        player_num == player_num_from_leaves
                    ), f"Player number mismatch, you set player_num to {player_num} but the payoff leaves have {player_num_from_leaves} players"
            elif player_num is None:
                raise ValueError("player_num must be provided for sequential games")

            # Initialize ordered_payoff_leaves if it doesn't exist
            if "ordered_payoff_leaves" not in self:
                self["ordered_payoff_leaves"] = {}
        else:
            payoffs: List[PayoffLeaf] = getattr(self, "payoff_leaves", [])
            game_type = self.game_type
            player_num = self.player_num
            if game_type == GameType.SIMULTANEOUS and payoffs:
                player_num_from_leaves = len(payoffs[0].actions)
                if player_num is None:
                    player_num = player_num_from_leaves
                else:
                    assert (
                        player_num == player_num_from_leaves
                    ), f"Player number mismatch, you set player_num to {player_num} but the payoff leaves have {player_num_from_leaves} players"
            elif player_num is None:
                raise ValueError("player_num must be provided for sequential games")

            # Initialize ordered_payoff_leaves if it's None
            if self.ordered_payoff_leaves is None:
                self.ordered_payoff_leaves = {}

        for i in range(player_num):
            payoffs_sorted = sorted(
                payoffs,
                key=lambda x: x.payoffs[i],
                reverse=True,
            )
            if isinstance(self, dict):
                self["ordered_payoff_leaves"][i] = [
                    leaf.actions for leaf in payoffs_sorted
                ]
            else:
                self.ordered_payoff_leaves[i] = [
                    leaf.actions for leaf in payoffs_sorted
                ]

        return self

    def __str__(self):
        """
        Returns a human-readable text description of the payoff matrix.
        Describes what happens when different players make different choices.
        """
        return self.get_natural_language_description(["player 1", "player 2"])

    def get_natural_language_description(self, players: List[str]):
        """
        a refined description of the payoff matrix, with customized player names
        """
        if self.game_type == GameType.SIMULTANEOUS:
            return self._get_simultaneous_description(players)
        else:
            return self._get_sequential_description(players)

    def _get_simultaneous_description(self, players: List[str]):
        result = []

        # First add basic description of each payoff
        result.append("Game Payoffs:")
        for leaf in self.payoff_leaves:
            p1_action, p2_action = leaf.actions
            p1_payoff, p2_payoff = leaf.payoffs

            description = f"When {players[0]} chooses '{p1_action}' and {players[1]} chooses '{p2_action}', "
            description += (
                f"{players[0]} gets {p1_payoff} and {players[1]} gets {p2_payoff}."
            )
            result.append(description)

        return "\n".join(result)

    def _get_sequential_description(self, players: List[str]):
        result = []
        result.append("Game Payoffs:")
        for leaf in self.payoff_leaves:
            path_description = []
            for i, action in enumerate(leaf.actions):
                player_index = i % self.player_num
                player_name = players[player_index]
                path_description.append(f"{player_name} chooses '{action}'")

            description = f"If {' then '.join(path_description)}, "
            description += f"the outcome is a payoff of {leaf.payoffs[0]} for {players[0]} and {leaf.payoffs[1]} for {players[1]}."
            result.append(description)

        return "\n".join(result)


# SIMULTANEOUS GAMES

# Prisoner's Dilemma
# | 3,3 | 0,5 |
# | 5,0 | 1,1 |
prisoners_dilemma = [
    PayoffLeaf(actions=("cooperate", "cooperate"), payoffs=(3, 3)),
    PayoffLeaf(actions=("cooperate", "defect"), payoffs=(0, 5)),
    PayoffLeaf(actions=("defect", "cooperate"), payoffs=(5, 0)),
    PayoffLeaf(actions=("defect", "defect"), payoffs=(1, 1)),
]

# Stag Hunt
# | 3,3 | 0,1 |
# | 1,0 | 1,1 |
stag_hunt = [
    PayoffLeaf(actions=("cooperate", "cooperate"), payoffs=(3, 3)),
    PayoffLeaf(actions=("cooperate", "defect"), payoffs=(0, 1)),
    PayoffLeaf(actions=("defect", "cooperate"), payoffs=(1, 0)),
    PayoffLeaf(actions=("defect", "defect"), payoffs=(1, 1)),
]

# Battle of the Sexes
# | 2,1 | 0,0 |
# | 0,0 | 1,2 |
battle_of_sexes = [
    PayoffLeaf(actions=("opera", "opera"), payoffs=(2, 1)),
    PayoffLeaf(actions=("opera", "football"), payoffs=(0, 0)),
    PayoffLeaf(actions=("football", "opera"), payoffs=(0, 0)),
    PayoffLeaf(actions=("football", "football"), payoffs=(1, 2)),
]

# Duopolistic Competition
duopolistic_competition = [
    PayoffLeaf(actions=("choice_1", "choice_1"), payoffs=(0, 0)),
    PayoffLeaf(actions=("choice_1", "choice_2"), payoffs=(0, 9)),
    PayoffLeaf(actions=("choice_1", "choice_3"), payoffs=(0, 14)),
    PayoffLeaf(actions=("choice_1", "choice_4"), payoffs=(0, 15)),
    PayoffLeaf(actions=("choice_1", "choice_5"), payoffs=(0, 12)),
    PayoffLeaf(actions=("choice_1", "choice_6"), payoffs=(0, 5)),
    PayoffLeaf(actions=("choice_2", "choice_1"), payoffs=(9, 0)),
    PayoffLeaf(actions=("choice_2", "choice_2"), payoffs=(7, 7)),
    PayoffLeaf(actions=("choice_2", "choice_3"), payoffs=(5, 10)),
    PayoffLeaf(actions=("choice_2", "choice_4"), payoffs=(3, 9)),
    PayoffLeaf(actions=("choice_2", "choice_5"), payoffs=(1, 4)),
    PayoffLeaf(actions=("choice_2", "choice_6"), payoffs=(-1, -5)),
    PayoffLeaf(actions=("choice_3", "choice_1"), payoffs=(14, 0)),
    PayoffLeaf(actions=("choice_3", "choice_2"), payoffs=(10, 5)),
    PayoffLeaf(actions=("choice_3", "choice_3"), payoffs=(6, 6)),
    PayoffLeaf(actions=("choice_3", "choice_4"), payoffs=(2, 3)),
    PayoffLeaf(actions=("choice_3", "choice_5"), payoffs=(-2, -4)),
    PayoffLeaf(actions=("choice_3", "choice_6"), payoffs=(-2, -5)),
    PayoffLeaf(actions=("choice_4", "choice_1"), payoffs=(15, 0)),
    PayoffLeaf(actions=("choice_4", "choice_2"), payoffs=(9, 3)),
    PayoffLeaf(actions=("choice_4", "choice_3"), payoffs=(3, 2)),
    PayoffLeaf(actions=("choice_4", "choice_4"), payoffs=(-3, -3)),
    PayoffLeaf(actions=("choice_4", "choice_5"), payoffs=(-3, -4)),
    PayoffLeaf(actions=("choice_4", "choice_6"), payoffs=(-3, -5)),
    PayoffLeaf(actions=("choice_5", "choice_1"), payoffs=(12, 0)),
    PayoffLeaf(actions=("choice_5", "choice_2"), payoffs=(4, 1)),
    PayoffLeaf(actions=("choice_5", "choice_3"), payoffs=(-4, -2)),
    PayoffLeaf(actions=("choice_5", "choice_4"), payoffs=(-4, -3)),
    PayoffLeaf(actions=("choice_5", "choice_5"), payoffs=(-4, -4)),
    PayoffLeaf(actions=("choice_5", "choice_6"), payoffs=(-4, -5)),
    PayoffLeaf(actions=("choice_6", "choice_1"), payoffs=(5, 0)),
    PayoffLeaf(actions=("choice_6", "choice_2"), payoffs=(-5, -1)),
    PayoffLeaf(actions=("choice_6", "choice_3"), payoffs=(-5, -2)),
    PayoffLeaf(actions=("choice_6", "choice_4"), payoffs=(-5, -3)),
    PayoffLeaf(actions=("choice_6", "choice_5"), payoffs=(-5, -4)),
    PayoffLeaf(actions=("choice_6", "choice_6"), payoffs=(-5, -5)),
]

# Wait Go Game
wait_go = [
    PayoffLeaf(actions=("choice_1", "choice_1"), payoffs=(0, 0)),
    PayoffLeaf(actions=("choice_1", "choice_2"), payoffs=(0, 2)),
    PayoffLeaf(actions=("choice_2", "choice_1"), payoffs=(2, 0)),
    PayoffLeaf(actions=("choice_2", "choice_2"), payoffs=(-4, -4)),
]

# SEQUENTIAL GAMES

# Escalation Game
# Nash equilibrium: Player 1 chooses "withdraw"
escalation_game = {
    "withdraw": [0, 0],
    "escalate": {
        "withdraw": [1, -2],
        "escalate": {
            "withdraw": [-2, 1],
            "escalate": [-1, -1],
        },
    },
}

# Monopoly Game
# Nash equilibrium: Player 1 chooses "choice_2", Player 2 chooses "choice_1"
monopoly_game = {
    "choice_1": [0, 2],
    "choice_2": {
        "choice_1": [2, 1],
        "choice_2": [-1, -1],
    },
}

# Ultimatum Game (simplified)
# Player 1 proposes a split (fair/unfair), Player 2 accepts or rejects
ultimatum_game = {
    "fair_split": {
        "accept": [5, 5],
        "reject": [0, 0],
    },
    "unfair_split": {
        "accept": [8, 2],
        "reject": [0, 0],
    },
}

# Hot Cold Game
hot_cold_game = {
    "Alice_choice_1": {"Bob_choice_1": [3, 2], "Bob_choice_2": [2, 3]},
    "Alice_choice_2": {"Bob_choice_1": [1, 4], "Bob_choice_2": [4, 1]},
}

# Draco Game
draco_game = {
    "Alice_choice_1": {
        "Bob_choice_1": [5, 5],
        "Bob_choice_2": {"Alice_choice_1": [2, 2], "Alice_choice_2": [3, 4]},
    },
    "Alice_choice_2": {
        "Bob_choice_1": [4, 5],
        "Bob_choice_2": {"Alice_choice_1": [5, 3], "Alice_choice_2": [2, 2]},
    },
}

# Trigame
trigame = {
    "Alice_choice_1": {
        "Bob_choice_1": {"Alice_choice_1": [20, 3], "Alice_choice_2": [0, 4]},
        "Bob_choice_2": {"Alice_choice_1": [2, 5], "Alice_choice_2": [3, 4]},
    },
    "Alice_choice_2": {
        "Bob_choice_1": {"Alice_choice_1": [1, 5], "Alice_choice_2": [4, 10]},
        "Bob_choice_2": {"Alice_choice_1": [2, 1], "Alice_choice_2": [3, 2]},
    },
}

# Create PayoffMatrix objects using GameNames enum as keys
SIMULTANEOUS_GAMES = {
    GameNames.PRISONERS_DILEMMA: PayoffMatrix(
        player_num=2,
        payoff_leaves=prisoners_dilemma,
    ),
    GameNames.STAG_HUNT: PayoffMatrix(
        player_num=2,
        payoff_leaves=stag_hunt,
    ),
    GameNames.BATTLE_OF_SEXES: PayoffMatrix(
        player_num=2,
        payoff_leaves=battle_of_sexes,
    ),
    GameNames.DUOPOLISTIC_COMPETITION: PayoffMatrix(
        player_num=2,
        payoff_leaves=duopolistic_competition,
    ),
    GameNames.WAIT_GO: PayoffMatrix(
        player_num=2,
        payoff_leaves=wait_go,
    ),
    # Assuming game_of_chicken and rock_paper_scissors have corresponding GameNames entries
    # If not, they need to be added to constants.py or handled differently
    # GameNames.GAME_OF_CHICKEN: PayoffMatrix.from_simultaneous_dict(
    #     game_of_chicken, name="Game of Chicken"
    # ),
    # GameNames.ROCK_PAPER_SCISSORS: PayoffMatrix.from_simultaneous_dict(
    #     rock_paper_scissors, name="Rock-Paper-Scissors"
    # ),
}

_escalation_matrix = PayoffMatrix(
    player_num=2,
    game_type=GameType.SEQUENTIAL,
    payoff_leaves=_parse_sequential_game_tree(escalation_game),
)

SEQUENTIAL_GAMES = {
    GameNames.ESCALATION_GAME: _escalation_matrix,
    GameNames.DIPLOMACY_ESCALATION_GAME: _escalation_matrix,
    GameNames.MONOPOLY_GAME: PayoffMatrix(
        player_num=2,
        game_type=GameType.SEQUENTIAL,
        payoff_leaves=_parse_sequential_game_tree(monopoly_game),
    ),
    GameNames.HOT_COLD_GAME: PayoffMatrix(
        player_num=2,
        game_type=GameType.SEQUENTIAL,
        payoff_leaves=_parse_sequential_game_tree(hot_cold_game),
    ),
    GameNames.DRACO_GAME: PayoffMatrix(
        player_num=2,
        game_type=GameType.SEQUENTIAL,
        payoff_leaves=_parse_sequential_game_tree(draco_game),
    ),
    GameNames.TRI_GAME: PayoffMatrix(
        player_num=2,
        game_type=GameType.SEQUENTIAL,
        payoff_leaves=_parse_sequential_game_tree(trigame),
    ),
}


# # Combined dictionary of all games
ALL_GAME_PAYOFF = {
    **SIMULTANEOUS_GAMES,
    **SEQUENTIAL_GAMES,
}


if __name__ == "__main__":
    pd_matrix = PayoffMatrix(
        player_num=2,
        payoff_leaves=prisoners_dilemma,
    )
    print("Ordered payoff leaves for Prisoner's Dilemma:")
    print(pd_matrix.ordered_payoff_leaves)
    print("\nHuman readable description for Prisoner's Dilemma:")
    print(pd_matrix)
    print("\nPayoff leaves for Prisoner's Dilemma:")
    print(pd_matrix.payoff_leaves)

    print("-" * 20)

    escalation_matrix = SEQUENTIAL_GAMES[GameNames.ESCALATION_GAME]
    print("\nHuman readable description for Escalation Game:")
    print(escalation_matrix)
    print("\nOrdered payoff leaves for Escalation Game:")
    print(escalation_matrix.ordered_payoff_leaves)
    print("\nPayoff leaves for Escalation Game:")
    print(escalation_matrix.payoff_leaves)
