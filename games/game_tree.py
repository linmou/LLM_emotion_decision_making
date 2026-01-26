from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union


@dataclass
class TerminalNode:
    """A leaf of the game tree representing terminal payoffs."""

    payoffs: Tuple[float, ...]  # e.g., (3, 3) for (p1, p2)


@dataclass
class DecisionNode:
    """A sequential move where one player chooses an action."""

    player: str  # e.g., "p1" or "Alice"
    actions: Dict[str, "GameNode"] = field(default_factory=dict)
    # maps an action -> next node


@dataclass
class SimultaneousNode:
    """A simultaneous move where multiple players choose actions at once."""

    players: Tuple[str, ...]  # e.g., ("p1", "p2") or ("Alice", "Bob")
    actions: Dict[Tuple[str, ...], "GameNode"] = field(default_factory=dict)
    # maps tuple of actions (one per player, in order) -> next node


# A GameNode can be any of the above types
GameNode = Union[TerminalNode, DecisionNode, SimultaneousNode]


class OldPayoffMatrix:
    """
    Class to represent payoff matrices for both simultaneous and sequential games.

    Deprecated: Use payoff_matrices.PayoffMatrix instead.
    """

    def __init__(self, game_tree: GameNode, name: str = "", description: str = ""):
        self.game_tree = game_tree
        self.name = name
        self.description = description
        self._player_identifiers = None  # Cache for player identifiers

    def get_player_identifiers(self) -> List[str]:
        """Return a list of unique player identifiers found in the game tree."""
        if self._player_identifiers is not None:
            return self._player_identifiers

        identifiers = set()

        queue = [self.game_tree]
        visited = set()

        while queue:
            node = queue.pop(0)
            node_id = id(node)  # Use id to handle cycles if they were possible
            if node_id in visited:
                continue
            visited.add(node_id)

            if isinstance(node, DecisionNode):
                identifiers.add(node.player)
                for child in node.actions.values():
                    if id(child) not in visited:
                        queue.append(child)
            elif isinstance(node, SimultaneousNode):
                for player in node.players:
                    identifiers.add(player)
                for child in node.actions.values():
                    if id(child) not in visited:
                        queue.append(child)
            # TerminalNode has no players directly

        # Return identifiers in a consistent (though arbitrary) order
        self._player_identifiers = sorted(list(identifiers))
        return self._player_identifiers

    @staticmethod
    def from_simultaneous_dict(
        payoff_dict: Dict[str, Dict[str, Dict[str, float]]],
        name: str = "",
        description: str = "",
    ) -> "PayoffMatrix":
        """
        Create a PayoffMatrix from a simultaneous game dictionary format.

        Example format:
        {
            "p1": {
                "cooperate": {"p2_cooperate": 3, "p2_defect": 0},
                "defect": {"p2_cooperate": 5, "p2_defect": 1}
            },
            "p2": {
                "cooperate": {"p1_cooperate": 3, "p1_defect": 0},
                "defect": {"p1_cooperate": 5, "p1_defect": 1}
            }
        }
        """
        # Extract players and their choices
        players = tuple(payoff_dict.keys())
        player_choices = {
            player: list(payoff_dict[player].keys()) for player in players
        }

        # Create a simultaneous node
        sim_node = SimultaneousNode(players=players)

        # Generate all possible action combinations
        for p1_choice in player_choices[players[0]]:
            for p2_choice in player_choices[players[1]]:
                # Get payoffs for this combination
                p1_payoff = payoff_dict[players[0]][p1_choice][
                    f"{players[1]}_{p2_choice}"
                ]
                p2_payoff = payoff_dict[players[1]][p2_choice][
                    f"{players[0]}_{p1_choice}"
                ]

                # Create terminal node with payoffs
                terminal = TerminalNode(payoffs=(p1_payoff, p2_payoff))

                # Add to the simultaneous node's actions
                sim_node.actions[(p1_choice, p2_choice)] = terminal

        matrix = PayoffMatrix(game_tree=sim_node, name=name, description=description)
        # Pre-calculate identifiers based on dict keys
        matrix._player_identifiers = sorted(list(players))
        return matrix

    @staticmethod
    def from_sequential_dict(
        seq_dict: Dict[str, Any],
        players: List[str] = None,
        name: str = "",
        description: str = "",
    ) -> "PayoffMatrix":
        """
        Create a PayoffMatrix from a sequential game dictionary format.

        This method handles nested dictionaries representing sequential decision trees.
        """

        def build_tree(node_dict, current_player_idx=0, players=players):
            # If it's a list, it's a terminal node with payoffs
            if isinstance(node_dict, list):
                return TerminalNode(tuple(node_dict))

            # If it's a dictionary, it's a decision node
            player = (
                players[current_player_idx]
                if players
                else f"Player {current_player_idx+1}"
            )
            decision_node = DecisionNode(player=player)

            # Process actions
            for action, next_node_dict in node_dict.items():
                # Remove player prefix if present in action name
                action_name = action
                if action.startswith(f"{player}_"):
                    action_name = action[len(f"{player}_") :]

                # Recursively build the next node
                next_player_idx = (
                    (current_player_idx + 1) % len(players)
                    if players
                    else current_player_idx + 1
                )
                decision_node.actions[action_name] = build_tree(
                    next_node_dict, next_player_idx, players
                )

            return decision_node

        # Start building from the root
        root_node = build_tree(seq_dict, players=players)
        matrix = PayoffMatrix(game_tree=root_node, name=name, description=description)
        # Identifiers calculated lazily via get_player_identifiers if needed
        return matrix

    def describe_game(self, participant_mapping: Dict[str, str] = None) -> List[str]:
        """
        Generate a human-readable description of the game.
        Returns a list of strings describing different paths through the game.
        Uses participant_mapping to replace internal player IDs if provided.
        """
        participant_mapping = participant_mapping or {}

        def get_display_name(internal_name: str) -> str:
            return participant_mapping.get(internal_name, internal_name)

        def describe_node(
            node: GameNode, history: List[Tuple[str, str]] = None
        ) -> List[str]:
            history = history or []
            lines = []

            if isinstance(node, TerminalNode):
                # Use display names from history for the path description
                path = " → ".join(f"{pl}={act}" for pl, act in history)
                lines.append(f"{path}: payoffs = {node.payoffs}")

            elif isinstance(node, DecisionNode):
                display_player = get_display_name(node.player)
                for action, child in node.actions.items():
                    # Add the display name to the history
                    new_hist = history + [(display_player, action)]
                    lines.extend(describe_node(child, new_hist))

            elif isinstance(node, SimultaneousNode):
                # Map internal player names to display names for this step
                display_players = [get_display_name(p) for p in node.players]
                for acts, child in node.actions.items():
                    # acts is a tuple of actions aligned with node.players
                    # Create history step using display names
                    step = list(zip(display_players, acts))
                    new_hist = history + step
                    lines.extend(describe_node(child, new_hist))

            return lines

        return describe_node(self.game_tree)

    def get_natural_language_description(
        self, participants: Optional[List[str]] = None
    ) -> str:
        """
        Convert the payoff matrix to a natural language description.
        Uses participant names if provided, assuming the order matches the
        internal player identifiers.
        """
        participant_mapping = None
        if participants:
            internal_players = self.get_player_identifiers()
            if len(internal_players) == len(participants):
                participant_mapping = dict(zip(internal_players, participants))
            else:
                # Fallback if lengths don't match - use internal names
                # TODO: Log a warning or raise an error for mismatch?
                participant_mapping = {p: p for p in internal_players}

        descriptions = self.describe_game(participant_mapping=participant_mapping)
        return "\n".join(descriptions)
