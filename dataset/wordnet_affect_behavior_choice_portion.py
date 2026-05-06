from __future__ import annotations

import argparse
import json
import re
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from typing import Callable, Iterable, Iterator
import sys

import nltk
from nltk.corpus import wordnet as wn
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from constants import Emotions

WORDNET_AFFECT_URLS = {
    "a-synsets.xml": "https://raw.githubusercontent.com/shivanipods/wordnet-affect/master/wn-affect-1.1/a-synsets.xml",
    "a-hierarchy.xml": "https://raw.githubusercontent.com/shivanipods/wordnet-affect/master/wn-affect-1.1/a-hierarchy.xml",
}

BASIC_EMOTION_ALIASES = {
    "anger": "anger",
    "fear": "fear",
    "sadness": "sadness",
    "disgust": "disgust",
    "surprise": "surprise",
    "happiness": "happiness",
    "joy": "happiness",
}

CATEGORY_STOPLIST = {
    "root",
    "mental-state",
    "physical-state",
    "behaviour",
    "situation",
    "signal",
    "trait",
    "sensation",
    "cognitive-state",
    "affective-state",
    "cognitive-affective-state",
    "mood",
    "emotion",
    "emotion-eliciting-situation",
    "edonic-signal",
    "positive-emotion",
    "negative-emotion",
}


def ensure_nltk_resource(resource_id: str, download_name: str) -> None:
    try:
        nltk.data.find(resource_id)
    except LookupError:
        nltk.download(download_name, quiet=True)


def download_wordnet_affect(target_dir: Path) -> None:
    target_dir.mkdir(parents=True, exist_ok=True)
    for filename, url in WORDNET_AFFECT_URLS.items():
        target_path = target_dir / filename
        if target_path.exists():
            continue
        with urllib.request.urlopen(url) as response:
            target_path.write_bytes(response.read())


def parse_affect_hierarchy(hierarchy_path: Path) -> dict[str, str]:
    tree = ET.parse(hierarchy_path)
    parent_map: dict[str, str] = {}
    for node in tree.findall(".//categ"):
        name = node.attrib.get("name")
        parent = node.attrib.get("isa")
        if name and parent:
            parent_map[name] = parent
    return parent_map


def parse_affect_categories(hierarchy_path: Path) -> set[str]:
    tree = ET.parse(hierarchy_path)
    categories = set()
    for node in tree.findall(".//categ"):
        name = node.attrib.get("name")
        if name:
            categories.add(name)
    return categories


def map_category_to_basic_emotion(
    category: str, parent_map: dict[str, str]
) -> str | None:
    current = category
    seen: set[str] = set()
    while current and current not in seen:
        mapped = BASIC_EMOTION_ALIASES.get(current)
        if mapped:
            return mapped
        for emotion in Emotions.get_emotions():
            if emotion in current:
                return emotion
        seen.add(current)
        current = parent_map.get(current, "")
    return None


def synset_lemmas_from_wordnet(synset_id: str) -> list[str]:
    if "#" not in synset_id:
        return []
    pos, offset_str = synset_id.split("#", 1)
    if not pos or not offset_str.isdigit():
        return []
    try:
        synset = wn.synset_from_pos_and_offset(pos, int(offset_str))
    except Exception:
        return []
    if synset is None:
        return []
    return [lemma.lower() for lemma in synset.lemma_names()]


def build_emotion_lexicon(
    synsets_xml_path: Path,
    parent_map: dict[str, str],
    synset_resolver: Callable[[str], list[str]] | None = None,
) -> dict[str, set[str]]:
    resolver = synset_resolver or synset_lemmas_from_wordnet
    lexicon = {emotion: set() for emotion in Emotions.get_emotions()}

    tree = ET.parse(synsets_xml_path)
    for node in tree.findall(".//*[@id][@categ]"):
        synset_id = node.attrib.get("id", "")
        category = node.attrib.get("categ", "")
        if not synset_id or not category:
            continue
        basic_emotion = map_category_to_basic_emotion(category, parent_map)
        if not basic_emotion:
            continue
        for lemma in resolver(synset_id):
            lexicon[basic_emotion].add(lemma)
            if "_" in lemma:
                for part in lemma.split("_"):
                    if part:
                        lexicon[basic_emotion].add(part)
    categories = set(parent_map.keys()) | set(parent_map.values())
    for category in categories:
        if category in CATEGORY_STOPLIST:
            continue
        basic_emotion = map_category_to_basic_emotion(category, parent_map)
        if not basic_emotion:
            continue
        lexicon[basic_emotion].add(category)
        for part in re.split(r"[-_]", category):
            if part and part.isalpha():
                lexicon[basic_emotion].add(part)
    return lexicon


def tokenize_and_lemmatize(text: str) -> list[str]:
    lemmatizer = WordNetLemmatizer()
    tokens = []
    for token in word_tokenize(text):
        token = token.lower()
        if not re.fullmatch(r"[a-z]+", token):
            continue
        tokens.append(token)
        tokens.append(lemmatizer.lemmatize(token))
    return tokens


def analyze_choice(choice: str, lexicon: dict[str, set[str]]) -> dict[str, object]:
    tokens = set(tokenize_and_lemmatize(choice))
    emotions = []
    token_matches: dict[str, list[str]] = {}
    for emotion, words in lexicon.items():
        matched = sorted(tokens & words)
        if matched:
            emotions.append(emotion)
            token_matches[emotion] = matched
    return {
        "choice": choice,
        "emotions": sorted(emotions),
        "tokens": token_matches,
    }


def progress_iter(
    items: Iterable[str], enabled: bool, desc: str
) -> Iterator[str]:
    if not enabled:
        return iter(items)
    return tqdm(items, desc=desc, unit="choice")


def compute_choice_emotion_details(
    choices: Iterable[str],
    lexicon: dict[str, set[str]],
    max_workers: int = 1,
    show_progress: bool = False,
) -> list[dict[str, object]]:
    choices_list = [choice for choice in choices if isinstance(choice, str)]
    if max_workers <= 1 or len(choices_list) < 2:
        iterator = progress_iter(choices_list, show_progress, "Choices")
        return [analyze_choice(choice, lexicon) for choice in iterator]
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(analyze_choice, choice, lexicon)
            for choice in choices_list
        ]
        results = []
        iterator = progress_iter(futures, show_progress, "Choices")
        for future in iterator:
            results.append(future.result())
        return results


def compute_choice_emotion_portions(
    choices: Iterable[str], lexicon: dict[str, set[str]]
) -> dict[str, float]:
    details = compute_choice_emotion_details(
        choices, lexicon, max_workers=1, show_progress=False
    )
    total = len(details)
    counts = {emotion: 0 for emotion in Emotions.get_emotions()}
    any_emotion = 0

    for item in details:
        emotions = item["emotions"]
        if emotions:
            any_emotion += 1
            for emotion in emotions:
                counts[emotion] += 1

    portions = {
        emotion: (counts[emotion] / total if total else 0.0)
        for emotion in Emotions.get_emotions()
    }
    portions["any_emotion"] = any_emotion / total if total else 0.0
    portions["total_choices"] = float(total)
    return portions


def extract_behavior_choices(records: list[dict]) -> list[str]:
    choices: list[str] = []
    for record in records:
        behavior_choices = record.get("behavior_choices")
        if isinstance(behavior_choices, dict):
            for value in behavior_choices.values():
                if isinstance(value, str):
                    choices.append(value)
    return choices


def load_records(json_path: Path) -> list[dict]:
    data = json.loads(json_path.read_text())
    if not isinstance(data, list):
        raise ValueError(f"Expected list in {json_path}")
    return [record for record in data if isinstance(record, dict)]


def find_game_names(data_dir: Path) -> list[str]:
    names = []
    for path in data_dir.glob("*_all_data_samples.json"):
        name = path.name.replace("_all_data_samples.json", "")
        if name:
            names.append(name)
    return sorted(set(names))


def run(
    game_name: str,
    data_dir: Path,
    wna_dir: Path,
    max_workers: int,
    show_progress: bool,
) -> dict[str, object]:
    ensure_nltk_resource("corpora/wordnet", "wordnet")
    ensure_nltk_resource("corpora/omw-1.4", "omw-1.4")
    ensure_nltk_resource("tokenizers/punkt", "punkt")

    download_wordnet_affect(wna_dir)
    parent_map = parse_affect_hierarchy(wna_dir / "a-hierarchy.xml")
    lexicon = build_emotion_lexicon(wna_dir / "a-synsets.xml", parent_map)

    json_path = data_dir / f"{game_name}_all_data_samples.json"
    if not json_path.exists():
        raise FileNotFoundError(f"Missing data file: {json_path}")
    records = load_records(json_path)
    choices = extract_behavior_choices(records)
    portions = compute_choice_emotion_portions(choices, lexicon)
    details = compute_choice_emotion_details(
        choices,
        lexicon,
        max_workers=max_workers,
        show_progress=show_progress,
    )
    return {
        "game_name": game_name,
        "portions": portions,
        "choice_matches": details,
    }


def run_all_games(
    data_dir: Path,
    wna_dir: Path,
    max_workers: int,
    show_progress: bool,
) -> dict[str, object]:
    results = {}
    game_names = find_game_names(data_dir)
    iterator = progress_iter(game_names, show_progress, "Games")
    for game_name in iterator:
        results[game_name] = run(
            game_name,
            data_dir,
            wna_dir,
            max_workers,
            show_progress,
        )
    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compute portion of behavior choices containing WordNet-Affect "
            "words for the six basic emotions."
        )
    )
    parser.add_argument("game_name", help="Game name prefix for *_all_data_samples.json")
    parser.add_argument(
        "--data-dir",
        default="data_creation/scenario_creation/langgraph_creation",
        help="Directory containing *_all_data_samples.json files",
    )
    parser.add_argument(
        "--wordnet-affect-dir",
        default="data_creation/scenario_creation/langgraph_creation/wordnet_affect_data",
        help="Directory to cache WordNet-Affect XML files",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=1,
        help="Number of worker processes for choice analysis",
    )
    parser.add_argument(
        "--all-games",
        action="store_true",
        help="Analyze all *_all_data_samples.json files in data dir",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable progress bars for games/choices",
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    wna_dir = Path(args.wordnet_affect_dir)
    show_progress = not args.no_progress
    if args.all_games:
        results = run_all_games(
            data_dir, wna_dir, args.max_workers, show_progress
        )
    else:
        results = run(
            args.game_name,
            data_dir,
            wna_dir,
            args.max_workers,
            show_progress,
        )
    print(json.dumps(results, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
