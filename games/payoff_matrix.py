# Description: This file contains the configuration for the games.
# payoff matrix for prisoner's dilemma game
# 3,3  | 0,5
# 5,0  | 1,1
prisoner_dilemma = {
    "p1": {
        "cooperate": {"p2_cooperate": 3, "p2_defect": 0},
        "defect": {"p2_cooperate": 5, "p2_defect": 1},
    },
    "p2": {
        "cooperate": {"p1_cooperate": 3, "p1_defect": 0},
        "defect": {"p1_cooperate": 5, "p1_defect": 1},
    },
}


# payoff matrix for prisoner's dilemma game with large payoff
# 300,300  | 0,301
# 301,0  | 1,1
prisoner_dilemma_large = {
    "p1": {
        "cooperate": {"p2_cooperate": 300, "p2_defect": 0},
        "defect": {"p2_cooperate": 301, "p2_defect": 1},
    },
    "p2": {
        "cooperate": {"p1_cooperate": 300, "p1_defect": 0},
        "defect": {"p1_cooperate": 301, "p1_defect": 1},
    },
}


# payoff matrix for prisoner's dilemma game with large payoff
# 1000,1000  | 0,1001
# 1001,0  | 1,1
prisoner_dilemma_very_large = {
    "p1": {
        "cooperate": {"p2_cooperate": 1000, "p2_defect": 0},
        "defect": {"p2_cooperate": 1001, "p2_defect": 1},
    },
    "p2": {
        "cooperate": {"p1_cooperate": 1000, "p1_defect": 0},
        "defect": {"p1_cooperate": 1001, "p1_defect": 1},
    },
}


# payoff matrix for prisoner's dilemma game with large payoff
# 3,3  | -300,5
# 5,-300  | -299,-299
prisoner_dilemma_small = {
    "p1": {
        "cooperate": {"p2_cooperate": 3, "p2_defect": -300},
        "defect": {"p2_cooperate": 5, "p2_defect": -299},
    },
    "p2": {
        "cooperate": {"p1_cooperate": 3, "p1_defect": -300},
        "defect": {"p1_cooperate": 5, "p1_defect": -299},
    },
}

# payoff matrix for prisoner's dilemma game with large payoff
# 3,3  | -1000,5
# 5,-1000  | -999,-999
prisoner_dilemma_very_small = {
    "p1": {
        "cooperate": {"p2_cooperate": 3, "p2_defect": -1000},
        "defect": {"p2_cooperate": 5, "p2_defect": -999},
    },
    "p2": {
        "cooperate": {"p1_cooperate": 3, "p1_defect": -1000},
        "defect": {"p1_cooperate": 5, "p1_defect": -999},
    },
}

# payoff matrix for battle of the sexes game
# 2,1  | 0,0
# 0,0  | 1,2
battle_of_sexes = {
    "Alice": {
        "choice_1": {"Bob_choice_1": 2, "Bob_choice_2": 0},
        "choice_2": {"Bob_choice_1": 0, "Bob_choice_2": 1},
    },
    "Bob": {
        "choice_1": {"Alice_choice_1": 1, "Alice_choice_2": 0},
        "choice_2": {"Alice_choice_1": 0, "Alice_choice_2": 2},
    },
}

# payoff matrix for game_of_chicken
# -10,-10|  1, -1
# -1, 1  |  0, 0
game_of_chicken = {
    "Alice": {
        "choice_1": {"Bob_choice_1": -10, "Bob_choice_2": 1},
        "choice_2": {"Bob_choice_1": -1, "Bob_choice_2": 0},
    },
    "Bob": {
        "choice_1": {"Alice_choice_1": -10, "Alice_choice_2": 1},
        "choice_2": {"Alice_choice_1": -1, "Alice_choice_2": 0},
    },
}


# payoff matrix for stag hunt
# 3,3  |  0,1
# 1,0  |  1,1
stag_hunt = {
    "Alice": {
        "choice_1": {"Bob_choice_1": 3, "Bob_choice_2": 0},
        "choice_2": {"Bob_choice_1": 1, "Bob_choice_2": 1},
    },
    "Bob": {
        "choice_1": {"Alice_choice_1": 3, "Alice_choice_2": 0},
        "choice_2": {"Alice_choice_1": 1, "Alice_choice_2": 1},
    },
}


# payoff matrix for stag hunt large
# 300,300  |  0,1
# 1,0  |  1,1
stag_hunt_large = {
    "Alice": {
        "choice_1": {"Bob_choice_1": 300, "Bob_choice_2": 0},
        "choice_2": {"Bob_choice_1": 1, "Bob_choice_2": 1},
    },
    "Bob": {
        "choice_1": {"Alice_choice_1": 300, "Alice_choice_2": 0},
        "choice_2": {"Alice_choice_1": 1, "Alice_choice_2": 1},
    },
}

# payoff matrix for stag hunt large
# 1000,1000  |  0,1
# 1,0  |  1,1
stag_hunt_very_large = {
    "Alice": {
        "choice_1": {"Bob_choice_1": 1000, "Bob_choice_2": 0},
        "choice_2": {"Bob_choice_1": 1, "Bob_choice_2": 1},
    },
    "Bob": {
        "choice_1": {"Alice_choice_1": 1000, "Alice_choice_2": 0},
        "choice_2": {"Alice_choice_1": 1, "Alice_choice_2": 1},
    },
}

# payoff matrix for stag hunt small
# 3,3  |  -100,-99
# -99,-100  |  -99,-99
stag_hunt_small = {
    "Alice": {
        "choice_1": {"Bob_choice_1": 3, "Bob_choice_2": -100},
        "choice_2": {"Bob_choice_1": -99, "Bob_choice_2": -99},
    },
    "Bob": {
        "choice_1": {"Alice_choice_1": 3, "Alice_choice_2": -100},
        "choice_2": {"Alice_choice_1": -99, "Alice_choice_2": -99},
    },
}

# payoff matrix for stag hunt small
# 3,3  |  -1000,-999
# -999,-1000  |  -999,-999
stag_hunt_very_small = {
    "Alice": {
        "choice_1": {"Bob_choice_1": 3, "Bob_choice_2": -1000},
        "choice_2": {"Bob_choice_1": -999, "Bob_choice_2": -999},
    },
    "Bob": {
        "choice_1": {"Alice_choice_1": 3, "Alice_choice_2": -1000},
        "choice_2": {"Alice_choice_1": -999, "Alice_choice_2": -999},
    },
}


# payoff matrix for radio_station
# 25,25| 50,30 | 50,20
# 30,50| 15,15 | 30,20
# 20,50| 20,30 | 10,10
radio_station = {
    "Alice": {
        "choice_1": {"Bob_choice_1": 25, "Bob_choice_2": 50, "Bob_choice_3": 50},
        "choice_2": {"Bob_choice_1": 30, "Bob_choice_2": 15, "Bob_choice_3": 30},
        "choice_3": {"Bob_choice_1": 20, "Bob_choice_2": 20, "Bob_choice_3": 10},
    },
    "Bob": {
        "choice_1": {"Alice_choice_1": 25, "Alice_choice_2": 50, "Alice_choice_3": 50},
        "choice_2": {"Alice_choice_1": 30, "Alice_choice_2": 15, "Alice_choice_3": 30},
        "choice_3": {"Alice_choice_1": 20, "Alice_choice_2": 20, "Alice_choice_3": 10},
    },
}

# payoff matrix for rock-paper-scissors
# 0,0  |  -1,1 |  1,-1
# 1,-1 |  0,0  | -1,1
# -1,1 |  1,-1 |  0,0
rock_paper_scissors = {
    "Alice": {
        "choice_1": {"Bob_choice_1": 0, "Bob_choice_2": -1, "Bob_choice_3": 1},
        "choice_2": {"Bob_choice_1": 1, "Bob_choice_2": 0, "Bob_choice_3": -1},
        "choice_3": {"Bob_choice_1": -1, "Bob_choice_2": 1, "Bob_choice_3": 0},
    },
    "Bob": {
        "choice_1": {"Alice_choice_1": 0, "Alice_choice_2": -1, "Alice_choice_3": 1},
        "choice_2": {"Alice_choice_1": 1, "Alice_choice_2": 0, "Alice_choice_3": -1},
        "choice_3": {"Alice_choice_1": -1, "Alice_choice_2": 1, "Alice_choice_3": 0},
    },
}

# payoff matrix for IESDS
# 13, 3 | 1, 4 | 7,3
# 4, 1  | 3, 3 | 6,2
# -1, 2 | 2, 3 | 8, -1


# (a1,b2) (a2, b2), (a3, b2)
# (b1,a1) (b2, a2), (b3, a3)
IESDS = {
    "Alice": {
        "choice_1": {"Bob_choice_1": 13, "Bob_choice_2": 1, "Bob_choice_3": 7},
        "choice_2": {"Bob_choice_1": 4, "Bob_choice_2": 3, "Bob_choice_3": 6},
        "choice_3": {"Bob_choice_1": -1, "Bob_choice_2": 2, "Bob_choice_3": 8},
    },
    "Bob": {
        "choice_1": {"Alice_choice_1": 3, "Alice_choice_2": 1, "Alice_choice_3": 9},
        "choice_2": {"Alice_choice_1": 4, "Alice_choice_2": 3, "Alice_choice_3": 8},
        "choice_3": {"Alice_choice_1": 3, "Alice_choice_2": 2, "Alice_choice_3": -1},
    },
}

# 1,1 | -1, 2 | 5, 0 | 1, 1
# 2,3 | 1, 2  | 3, 0 | 5, 1
# 1,1 | 0, 5  | 1, 7 | 0, 1
imbalanced_actions = {
    "Alice": {
        "choice_1": {
            "Bob_choice_1": 1,
            "Bob_choice_2": -1,
            "Bob_choice_3": 5,
            "Bob_choice_4": 1,
        },
        "choice_2": {
            "Bob_choice_1": 2,
            "Bob_choice_2": 1,
            "Bob_choice_3": 3,
            "Bob_choice_4": 5,
        },
        "choice_3": {
            "Bob_choice_1": 1,
            "Bob_choice_2": 0,
            "Bob_choice_3": 1,
            "Bob_choice_4": 0,
        },
    },
    "Bob": {
        "choice_1": {"Alice_choice_1": 1, "Alice_choice_2": 3, "Alice_choice_3": 1},
        "choice_2": {"Alice_choice_1": 2, "Alice_choice_2": 2, "Alice_choice_3": 5},
        "choice_3": {"Alice_choice_1": 0, "Alice_choice_2": 0, "Alice_choice_3": 7},
        "choice_4": {"Alice_choice_1": 1, "Alice_choice_2": 1, "Alice_choice_3": 1},
    },
}

# payoff matrix for weakly dominated game
# 5, 1 | 4, 0
# 6, 0 | 3, 1
# 6, 4 | 4, 1
weakly_dominated = {
    "Alice": {
        "choice_1": {
            "Bob_choice_1": 5,
            "Bob_choice_2": 4,
        },
        "choice_2": {
            "Bob_choice_1": 6,
            "Bob_choice_2": 3,
        },
        "choice_3": {
            "Bob_choice_1": 6,
            "Bob_choice_2": 4,
        },
    },
    "Bob": {
        "choice_1": {"Alice_choice_1": 1, "Alice_choice_2": 0, "Alice_choice_3": 4},
        "choice_2": {"Alice_choice_1": 0, "Alice_choice_2": 1, "Alice_choice_3": 4},
    },
}

# payoff matrix for duo-polistic competition: Simple Cournot Competition
# 0,0 | 0,9 | 0,14 | 0,15 | 0,12 | 0,5
# 9,0 | 7,7 | 5,10 | 3,9  | 1,4  | -1,-5
# 14,0| 10,5| 6,6  | 2,3  | -2,-4| -2,-5
# 15,0| 9,3 | 3,2  | -3,-3| -3,-4| -3,-5
# 12,0| 4,1 | -4,-2| -4,-3| -4,-4| -4,-5
# 5,0 |-5,-1| -5,-2| -5,-3| -5,-4| -5,-5
duopolistic_competition = {
    "Alice": {
        "choice_1": {
            "Bob_choice_1": 0,
            "Bob_choice_2": 0,
            "Bob_choice_3": 0,
            "Bob_choice_4": 0,
            "Bob_choice_5": 0,
            "Bob_choice_6": 0,
        },
        "choice_2": {
            "Bob_choice_1": 9,
            "Bob_choice_2": 7,
            "Bob_choice_3": 5,
            "Bob_choice_4": 3,
            "Bob_choice_5": 1,
            "Bob_choice_6": -1,
        },
        "choice_3": {
            "Bob_choice_1": 14,
            "Bob_choice_2": 10,
            "Bob_choice_3": 6,
            "Bob_choice_4": 2,
            "Bob_choice_5": -2,
            "Bob_choice_6": -2,
        },
        "choice_4": {
            "Bob_choice_1": 15,
            "Bob_choice_2": 9,
            "Bob_choice_3": 3,
            "Bob_choice_4": -3,
            "Bob_choice_5": -3,
            "Bob_choice_6": -3,
        },
        "choice_5": {
            "Bob_choice_1": 12,
            "Bob_choice_2": 4,
            "Bob_choice_3": -4,
            "Bob_choice_4": -4,
            "Bob_choice_5": -4,
            "Bob_choice_6": -4,
        },
        "choice_6": {
            "Bob_choice_1": 5,
            "Bob_choice_2": -5,
            "Bob_choice_3": -5,
            "Bob_choice_4": -5,
            "Bob_choice_5": -5,
            "Bob_choice_6": -5,
        },
    },
    "Bob": {
        "choice_1": {
            "Alice_choice_1": 0,
            "Alice_choice_2": 0,
            "Alice_choice_3": 0,
            "Alice_choice_4": 0,
            "Alice_choice_5": 0,
            "Alice_choice_6": 0,
        },
        "choice_2": {
            "Alice_choice_1": 9,
            "Alice_choice_2": 7,
            "Alice_choice_3": 5,
            "Alice_choice_4": 3,
            "Alice_choice_5": 1,
            "Alice_choice_6": -1,
        },
        "choice_3": {
            "Alice_choice_1": 14,
            "Alice_choice_2": 10,
            "Alice_choice_3": 6,
            "Alice_choice_4": 2,
            "Alice_choice_5": -2,
            "Alice_choice_6": -2,
        },
        "choice_4": {
            "Alice_choice_1": 15,
            "Alice_choice_2": 9,
            "Alice_choice_3": 3,
            "Alice_choice_4": -3,
            "Alice_choice_5": -3,
            "Alice_choice_6": -3,
        },
        "choice_5": {
            "Alice_choice_1": 12,
            "Alice_choice_2": 4,
            "Alice_choice_3": -4,
            "Alice_choice_4": -4,
            "Alice_choice_5": -4,
            "Alice_choice_6": -4,
        },
        "choice_6": {
            "Alice_choice_1": 5,
            "Alice_choice_2": -5,
            "Alice_choice_3": -5,
            "Alice_choice_4": -5,
            "Alice_choice_5": -5,
            "Alice_choice_6": -5,
        },
    },
}

odd_even_game = {
    "Alice": {
        "choice_1": {"Bob_choice_1": -2, "Bob_choice_2": 3},
        "choice_2": {"Bob_choice_1": 3, "Bob_choice_2": -4},
    },
    "Bob": {
        "choice_1": {"Alice_choice_1": 2, "Alice_choice_2": -3},
        "choice_2": {"Alice_choice_1": -3, "Alice_choice_2": 4},
    },
}

wait_go = {
    "Alice": {
        "choice_1": {"Bob_choice_1": 0, "Bob_choice_2": 0},
        "choice_2": {"Bob_choice_1": 2, "Bob_choice_2": -4},
    },
    "Bob": {
        "choice_1": {"Alice_choice_1": 0, "Alice_choice_2": 0},
        "choice_2": {"Alice_choice_1": 2, "Alice_choice_2": -4},
    },
}

##### simultaneous games
payoff_matrix = {
    "prisoner_dilemma": prisoner_dilemma,  ## single NE
    "prisoner_dilemma_small": prisoner_dilemma_small,
    "prisoner_dilemma_large": prisoner_dilemma_large,
    "prisoner_dilemma_very_small": prisoner_dilemma_very_small,
    "prisoner_dilemma_very_large": prisoner_dilemma_very_large,
    "rock_paper_scissors": rock_paper_scissors,
    "IESDS": IESDS,
    "imbalanced_actions": imbalanced_actions,  # not run
    "weakly_dominated": weakly_dominated,  # not run
    "duopolistic_competition": duopolistic_competition,
    "radio_station": radio_station,  ## multiple NE
    "battle_of_sexes": battle_of_sexes,
    "stag_hunt": stag_hunt,
    "stag_hunt_small": stag_hunt_small,
    "stag_hunt_large": stag_hunt_large,
    "stag_hunt_very_small": stag_hunt_very_small,
    "stag_hunt_very_large": stag_hunt_very_large,
    "game_of_chicken": game_of_chicken,
    "wait_go": wait_go,
}


second_price_auction = {
    "action_type": list(range(101)),
    "game_description": "This is a bidding game, you and other players are bidding an object.\nEach player will submit a bid, and the player with the highest bid will win the game. The winner will pay the second highest bid.",
}

guess_2_3 = {
    "action_type": list(range(101)),
    "game_description": "This is a number-guessing game.\nGuess a number in the range of [0,100] you think is the 2/3 of the average of all numbers that you and other players guess. You will win the game if what you guess is the closest to 2/3 of the average than the numbers that others guessed.",
}


##### sequential games
ultimatum_game = {
    "player_1_action_type": list(range(101)),
    "player_2_action_type": ["accept", "decline"],
    "game_description": "This is a money splitting game involving two players where you want to maximize the amount of money you obtain.\nPlayer 1 will propose a split of $100, and Player 2 will decide whether to accept the proposal. If Player 2 accepts, the money will be split according to the proposal. If Player 2 rejects, neither player will receive any money.",
}

# payoff matrix for escalation game
# nash equilibrium: 1
escalation_game = {
    ("Alice", "withdraw"): [0, 0],
    ("Alice", "escalation"): {
        ("Bob", "withdraw"): [1, -2],
        ("Bob", "escalation"): {
            ("Alice", "withdraw"): [-2, 1],
            ("Alice", "escalation"): [-1, -1],
        },
    },
}

# nash equilibrium: 2 1
monopoly_game = {
    "Alice_choice_1": [0, 2],
    "Alice_choice_2": {"Bob_choice_1": [2, 1], "Bob_choice_2": [-1, -1]},
}

# https://www.youtube.com/watch?v=KUChf4OjFwk&t=300s
# nash equilibrium: 2 1 2
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

# nash equilibrium: 1, 2
hot_cold_game = {
    "Alice_choice_1": {"Bob_choice_1": [3, 2], "Bob_choice_2": [2, 3]},
    "Alice_choice_2": {"Bob_choice_1": [1, 4], "Bob_choice_2": [4, 1]},
}

# https://www.youtube.com/watch?v=K523s8iQA2M&t=15s
# nash equilibrium: 1,1
draco = {
    "Alice_choice_1": {
        "Bob_choice_1": [5, 5],
        "Bob_choice_2": {"Alice_choice_1": [2, 2], "Alice_choice_2": [3, 4]},
    },
    "Alice_choice_2": {
        "Bob_choice_1": [4, 5],
        "Bob_choice_2": {"Alice_choice_1": [5, 3], "Alice_choice_2": [2, 2]},
    },
}

sequential_payoff_matrix = {
    "escalation_game": escalation_game,  ## single NE
    "monopoly_game": monopoly_game,
    "trigame": trigame,
    "hot_cold_game": hot_cold_game,
    "draco": draco,
}
