from enum import Enum


class Emotions(Enum):
    ANGER = "anger"
    HAPPINESS = "happiness"
    SADNESS = "sadness"
    DISGUST = "disgust"
    FEAR = "fear"
    SURPRISE = "surprise"

    def __init__(self, *args):
        super().__init__()
        # Check for prefix overlaps during enum initialization
        for other in self.__class__:
            if other == self:
                continue
            min_len = min(len(self.value), len(other.value))
            for i in range(3, min_len + 1):
                if self.value[:i] == other.value[:i]:
                    raise ValueError(
                        f"Prefix overlap detected between {self.value} and {other.value}: "
                        f"both share prefix '{self.value[:i]}'"
                    )

    @classmethod
    def get_emotions(cls) -> list[str]:
        return [emotion.value for emotion in cls]

    @classmethod
    def from_string(cls, value: str) -> "Emotions":
        """
        Get an Emotions enum member from its string value.

        Args:
            value: String representation of the emotion

        Returns:
            Corresponding Emotions enum member

        Raises:
            ValueError: If no matching emotion is found
        """
        value = value.lower().strip()
        for emotion in cls:
            emotion_value = emotion.value.lower()
            # Case 1: Exact match
            if emotion_value == value:
                return emotion
            # Case 2: Input is a prefix of emotion value (e.g., "hap" -> "happiness")
            if len(value) > 2 and emotion_value.startswith(value):
                return emotion
            # Case 3: Emotion value is a prefix of input (e.g., "happy" matches "happiness")
            if len(value) > 2 and value.startswith(emotion_value):
                return emotion

        raise ValueError(f"No emotion found matching '{value}'")


class GameType(Enum):
    SIMULTANEOUS = "simultaneous"
    SEQUENTIAL = "sequential"


class SymmetryType(Enum):
    SYMMETRIC = "symmetric"
    ASYMMETRIC = "asymmetric"


class GameNames(Enum):
    # Simultaneous Games
    PRISONERS_DILEMMA = (
        "Prisoners_Dilemma",
        GameType.SIMULTANEOUS,
        SymmetryType.SYMMETRIC,
    )
    BATTLE_OF_SEXES = ("Battle_Of_Sexes", GameType.SIMULTANEOUS, SymmetryType.SYMMETRIC)
    WAIT_GO = ("Wait_Go", GameType.SIMULTANEOUS, SymmetryType.SYMMETRIC)
    DUOPOLISTIC_COMPETITION = (
        "Duopolistic_Competition",
        GameType.SIMULTANEOUS,
        SymmetryType.SYMMETRIC,
    )
    STAG_HUNT = ("Stag_Hunt", GameType.SIMULTANEOUS, SymmetryType.SYMMETRIC)
    BEAUTY_CONTEST = ("Beauty_Contest", GameType.SIMULTANEOUS, SymmetryType.SYMMETRIC)
    SEALED_AUCTION = (
        "Sealed_Auction",
        GameType.SIMULTANEOUS,
        SymmetryType.SYMMETRIC,
    )

    # Sequential Games
    ESCALATION_GAME = ("Escalation_Game", GameType.SEQUENTIAL, SymmetryType.SYMMETRIC)
    MONOPOLY_GAME = ("Monopoly_Game", GameType.SEQUENTIAL, SymmetryType.SYMMETRIC)
    HOT_COLD_GAME = ("Hot_Cold_Game", GameType.SEQUENTIAL, SymmetryType.SYMMETRIC)
    DRACO_GAME = ("Draco_Game", GameType.SEQUENTIAL, SymmetryType.SYMMETRIC)
    TRI_GAME = ("Tri_Game", GameType.SEQUENTIAL, SymmetryType.SYMMETRIC)
    DIPLOMACY_ESCALATION_GAME = (
        "Diplomacy_Escalation_Game",
        GameType.SEQUENTIAL,
        SymmetryType.SYMMETRIC,
    )

    # Asymmetric Games
    TRUST_GAME_TRUSTOR = (
        "Trust_Game_Trustor",
        GameType.SEQUENTIAL,
        SymmetryType.ASYMMETRIC,
    )
    TRUST_GAME_TRUSTEE = (
        "Trust_Game_Trustee",
        GameType.SEQUENTIAL,
        SymmetryType.ASYMMETRIC,
    )
    ULTIMATUM_GAME_PROPOSER = (
        "Ultimatum_Game_Proposer",
        GameType.SEQUENTIAL,
        SymmetryType.ASYMMETRIC,
    )
    ULTIMATUM_GAME_RESPONDER = (
        "Ultimatum_Game_Responder",
        GameType.SEQUENTIAL,
        SymmetryType.ASYMMETRIC,
    )

    def __init__(self, value: str, game_type: GameType, symmetry_type: SymmetryType):
        self._value_ = value
        self.game_type = game_type
        self.symmetry_type = symmetry_type

    @property
    def value(self) -> str:
        return self._value_

    @classmethod
    def from_string(cls, value: str) -> "GameNames":
        """
        Get a GameNames enum member from its string value.

        Args:
            value: String representation of the game name

        Returns:
            Corresponding GameNames enum member

        Raises:
            ValueError: If no matching game is found
        """
        value = value.strip().replace(" ", "_")

        # Try exact match first
        for game in cls:
            if game.value.lower() == value.lower():
                return game

        # Try prefix match if exact match fails
        for game in cls:
            if len(value) > 2 and game.value.lower().startswith(value.lower()):
                return game

        raise ValueError(f"No game found matching '{value}'")

    @classmethod
    def get_game_type(cls, game_name: str) -> GameType:
        """
        Get the game type (simultaneous/sequential) from a game name string.

        Args:
            game_name: String representation of the game name

        Returns:
            GameType enum member

        Raises:
            ValueError: If no matching game is found
        """
        game = cls.from_string(game_name)
        return game.game_type

    @classmethod
    def get_symmetry_type(cls, game_name: str) -> SymmetryType:
        """
        Get the symmetry type (symmetric/asymmetric) from a game name string.

        Args:
            game_name: String representation of the game name

        Returns:
            SymmetryType enum member

        Raises:
            ValueError: If no matching game is found
        """
        game = cls.from_string(game_name)
        return game.symmetry_type

    @classmethod
    def get_simultaneous_games(cls) -> list["GameNames"]:
        """Returns list of all simultaneous games"""
        return [game for game in cls if game.game_type == GameType.SIMULTANEOUS]

    @classmethod
    def get_sequential_games(cls) -> list["GameNames"]:
        """Returns list of all sequential games"""
        return [game for game in cls if game.game_type == GameType.SEQUENTIAL]

    @classmethod
    def get_symmetric_games(cls) -> list["GameNames"]:
        """Returns list of all symmetric games"""
        return [game for game in cls if game.symmetry_type == SymmetryType.SYMMETRIC]

    @classmethod
    def get_asymmetric_games(cls) -> list["GameNames"]:
        """Returns list of all asymmetric games"""
        return [game for game in cls if game.symmetry_type == SymmetryType.ASYMMETRIC]

    @classmethod
    def get_games_by_type(
        cls, game_type: GameType = None, symmetry_type: SymmetryType = None
    ) -> list["GameNames"]:
        """
        Get games filtered by game type and/or symmetry type.

        Args:
            game_type: Filter by GameType (optional)
            symmetry_type: Filter by SymmetryType (optional)

        Returns:
            List of GameNames matching the criteria
        """
        games = list(cls)

        if game_type is not None:
            games = [game for game in games if game.game_type == game_type]

        if symmetry_type is not None:
            games = [game for game in games if game.symmetry_type == symmetry_type]

        return games

    def is_sequential(self) -> bool:
        """Returns True if the game is sequential, False otherwise"""
        return self.game_type == GameType.SEQUENTIAL

    def is_simultaneous(self) -> bool:
        """Returns True if the game is simultaneous, False otherwise"""
        return self.game_type == GameType.SIMULTANEOUS

    def is_symmetric(self) -> bool:
        """Returns True if the game is symmetric, False otherwise"""
        return self.symmetry_type == SymmetryType.SYMMETRIC

    def is_asymmetric(self) -> bool:
        """Returns True if the game is asymmetric, False otherwise"""
        return self.symmetry_type == SymmetryType.ASYMMETRIC


if __name__ == "__main__":
    print(Emotions.from_string("ang"))
    print(Emotions.from_string("happiness"))
    print(Emotions.from_string("sadness"))
    print(Emotions.from_string("SAD"))

    # Test game type functionality
    print("\nTesting game types:")
    print(
        "Simultaneous games:",
        [game.value for game in GameNames.get_simultaneous_games()],
    )
    print(
        "Sequential games:", [game.value for game in GameNames.get_sequential_games()]
    )
    print("PRISONERS_DILEMMA type:", GameNames.PRISONERS_DILEMMA.game_type)
    print("ESCALATION_GAME type:", GameNames.ESCALATION_GAME.game_type)

    # Test symmetry functionality
    print("\nTesting symmetry types:")
    print(
        "Symmetric games:",
        [game.value for game in GameNames.get_symmetric_games()],
    )
    print(
        "Asymmetric games:",
        [game.value for game in GameNames.get_asymmetric_games()],
    )
    print("PRISONERS_DILEMMA symmetry:", GameNames.PRISONERS_DILEMMA.symmetry_type)
    print("TRUST_GAME_TRUSTOR symmetry:", GameNames.TRUST_GAME_TRUSTOR.symmetry_type)

    # Test combined filtering
    print("\nTesting combined filtering:")
    print(
        "Simultaneous + Symmetric games:",
        [
            game.value
            for game in GameNames.get_games_by_type(
                GameType.SIMULTANEOUS, SymmetryType.SYMMETRIC
            )
        ],
    )
    print(
        "Sequential + Asymmetric games:",
        [
            game.value
            for game in GameNames.get_games_by_type(
                GameType.SEQUENTIAL, SymmetryType.ASYMMETRIC
            )
        ],
    )

    # Test from_string and get_game_type functionality
    print("\nTesting string conversion:")
    print("From 'prisoners':", GameNames.from_string("prisoners").value)
    print("From 'ESCALATION_GAME':", GameNames.from_string("ESCALATION_GAME").value)

    print("\nTesting game type lookup:")
    print("Type of 'prisoners':", GameNames.get_game_type("prisoners"))
    print("Type of 'escalation':", GameNames.get_game_type("escalation"))

    print("\nTesting symmetry type lookup:")
    print("Symmetry of 'prisoners':", GameNames.get_symmetry_type("prisoners"))
    print(
        "Symmetry of 'trust_game_trustor':",
        GameNames.get_symmetry_type("trust_game_trustor"),
    )

    # Test boolean methods
    print("\nTesting boolean methods:")
    print(
        "PRISONERS_DILEMMA.is_symmetric():", GameNames.PRISONERS_DILEMMA.is_symmetric()
    )
    print(
        "TRUST_GAME_TRUSTOR.is_asymmetric():",
        GameNames.TRUST_GAME_TRUSTOR.is_asymmetric(),
    )
    print("ESCALATION_GAME.is_sequential():", GameNames.ESCALATION_GAME.is_sequential())
    print(
        "BATTLE_OF_SEXES.is_simultaneous():",
        GameNames.BATTLE_OF_SEXES.is_simultaneous(),
    )

    # Test error handling
    try:
        GameNames.from_string("invalid_game")
    except ValueError as e:
        print("\nExpected error:", e)
