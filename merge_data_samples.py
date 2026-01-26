import json
import os
from pathlib import Path

from constants import GameNames
from games.game_configs import GAME_CONFIGS

output_file_format = "data_creation/scenario_creation/langgraph_creation/diplomacy_{game_name_str}_all_data_samples.json"


def merge_data_samples(data_folder, game_name_str):
    data_ls = []
    for file in Path(data_folder).glob("*.json"):
        with open(file, "r") as f:
            data = json.load(f)
            data["game_name"] = game_name_str
            data_ls.append(data)

    with open(output_file_format.format(game_name_str=game_name_str), "w") as f:
        json.dump(data_ls, f, indent=4)
    print(f"Merged {game_name_str} data samples, in total {len(data_ls)} samples")

    return data_ls


game_name = GameNames.ESCALATION_GAME
game_name_str = game_name.value
if not Path(output_file_format.format(game_name_str=game_name_str)).exists():
    data_folder = "data_creation/diplomacy_scenario_creation/behave_verify/scenarios/Escalation_Game_20251127"
    merge_data_samples(data_folder, game_name_str)
else:
    print(f"File {output_file_format.format(game_name_str=game_name_str)} already exists")