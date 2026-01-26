"""
Backfill chosen_behavior into existing detailed_results.csv files.

Usage:
  python -m emotion_experiment_engine.scripts.post_process_scripts.add_chosen_behavior --root results/new_game_theory_decision/shuffle_choices
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, Iterable, List, Optional, Tuple

import pandas as pd

logger = logging.getLogger(__name__)

def _normalize_item_id(item_id: Any) -> str:
    if item_id is None:
        return ""
    if isinstance(item_id, bool):
        return str(item_id)
    if isinstance(item_id, int):
        return str(item_id)
    if isinstance(item_id, float):
        if item_id != item_id:  # NaN
            return ""
        as_int = int(item_id)
        return str(as_int) if float(as_int) == item_id else str(item_id)
    s = str(item_id).strip()
    if not s:
        return ""
    try:
        f = float(s)
    except Exception:
        return s
    if f != f:  # NaN
        return ""
    as_int = int(f)
    return str(as_int) if float(as_int) == f else s


@dataclass(frozen=True)
class _RowKey:
    emotion: str
    intensity: float
    item_id: str
    repeat_id: int
    option_id: int


@dataclass(frozen=True)
class _RowKeyNoOption:
    emotion: str
    intensity: float
    item_id: str
    repeat_id: int


def _is_int_like_score(score: Any) -> Optional[int]:
    if score is None:
        return None
    try:
        score_float = float(score)
    except Exception:
        return None
    if score_float != score_float:  # NaN
        return None
    option_id = int(score_float)
    if float(option_id) != score_float:
        return None
    return option_id


def _chosen_behavior_from_options(options: Any, option_id: int) -> Optional[str]:
    if not isinstance(options, list):
        return None
    for opt in options:
        if not isinstance(opt, dict):
            continue
        raw_id = opt.get("id")
        if raw_id is None:
            continue
        try:
            opt_id = int(raw_id)
        except Exception:
            continue
        if opt_id != option_id:
            continue
        behavior = opt.get("behavior")
        if isinstance(behavior, str) and behavior.strip():
            return behavior
        text = opt.get("text")
        return text if isinstance(text, str) and text.strip() else None
    return None


def _load_raw_mapping(
    raw_path: Path, *, strict: bool
) -> Tuple[Dict[_RowKey, Optional[str]], Dict[_RowKeyNoOption, List[int]]]:
    raw_rows = json.loads(raw_path.read_text(encoding="utf-8"))
    if not isinstance(raw_rows, list):
        raise ValueError(f"{raw_path} must be a JSON list")

    mapping: Dict[_RowKey, Optional[str]] = {}
    available_option_ids_by_item: Dict[_RowKeyNoOption, List[int]] = {}
    for row in raw_rows:
        if not isinstance(row, dict):
            continue
        raw_emotion = row.get("emotion")
        raw_intensity = row.get("intensity")
        raw_item_id = row.get("item_id")
        if raw_emotion is None or raw_intensity is None or raw_item_id is None:
            continue
        option_id = _is_int_like_score(row.get("score"))
        if option_id is None:
            continue
        if option_id <= 0:
            # Non-choice sentinel (e.g., -1): treat as unknown and skip mapping.
            continue
        md = row.get("metadata") or {}
        item_md = md.get("item_metadata") or {}
        options = item_md.get("options")
        try:
            opt_ids_set = set()
            for opt in options or []:
                if not isinstance(opt, dict):
                    continue
                raw_id = opt.get("id")
                if raw_id is None:
                    continue
                opt_ids_set.add(int(raw_id))
            opt_ids = sorted(opt_ids_set)
        except Exception:
            opt_ids = []
        item_key = _RowKeyNoOption(
            emotion=str(raw_emotion),
            intensity=float(raw_intensity),
            item_id=_normalize_item_id(raw_item_id),
            repeat_id=int(row.get("repeat_id", 0) or 0),
        )
        if opt_ids:
            available_option_ids_by_item[item_key] = opt_ids
        chosen = _chosen_behavior_from_options(options, option_id)
        key = _RowKey(
            emotion=str(raw_emotion),
            intensity=float(raw_intensity),
            item_id=_normalize_item_id(raw_item_id),
            repeat_id=int(row.get("repeat_id", 0) or 0),
            option_id=option_id,
        )
        if strict and key in mapping:
            raise ValueError(f"Duplicate raw mapping key found: {key}")
        mapping[key] = chosen
    return mapping, available_option_ids_by_item


def _iter_detailed_csv_paths(root: Path) -> Iterable[Path]:
    yield from root.rglob("detailed_results.csv")


def update_detailed_results_csv(
    detailed_csv: Path,
    *,
    strict: bool,
    overwrite: bool,
    skip_missing_raw: bool,
    skip_finished: bool,
) -> bool:
    run_dir = detailed_csv.parent
    raw_path = run_dir / "raw_results.json"
    df = pd.read_csv(detailed_csv)

    required = {"emotion", "intensity", "item_id", "score"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{detailed_csv} missing required columns: {sorted(missing)}")

    if "repeat_id" not in df.columns:
        df["repeat_id"] = 0

    # Resume-friendly fast path: if the column exists and has no missing values,
    # do not re-validate against raw_results.json and do not rewrite the file.
    if (
        skip_finished
        and (not overwrite)
        and "chosen_behavior" in df.columns
        and not df["chosen_behavior"].isna().any()
    ):
        return False

    if not raw_path.exists():
        if skip_missing_raw:
            logger.warning("Missing %s; skipping %s", raw_path, detailed_csv)
            return False
        raise FileNotFoundError(
            f"Missing {raw_path} required to infer chosen_behavior ({detailed_csv})"
        )

    try:
        raw_mapping, available_option_ids_by_item = _load_raw_mapping(
            raw_path, strict=strict
        )
    except json.JSONDecodeError as e:
        if strict and (not skip_missing_raw):
            raise ValueError(f"Invalid JSON in {raw_path} ({detailed_csv}): {e}") from e
        logger.warning("Invalid JSON in %s; skipping %s (%s)", raw_path, detailed_csv, e)
        return False
    except Exception as e:
        if strict and (not skip_missing_raw):
            raise RuntimeError(f"Failed processing {detailed_csv}: {e}") from e
        logger.warning("Failed processing %s; skipping (%s)", detailed_csv, e)
        return False

    if "chosen_behavior" not in df.columns:
        df["chosen_behavior"] = pd.NA

    if not overwrite:
        to_fill_mask = df["chosen_behavior"].isna()
    else:
        to_fill_mask = pd.Series([True] * len(df))

    missing_option_ids: List[Tuple[_RowKeyNoOption, int]] = []
    missing_keys: List[_RowKey] = []
    mismatched_existing: List[Tuple[_RowKey, str, str]] = []

    if strict and not overwrite:
        for _, row in df[~df["chosen_behavior"].isna()].iterrows():
            option_id = _is_int_like_score(row.get("score"))
            if option_id is None:
                continue
            key = _RowKey(
                emotion=str(row.get("emotion")),
                intensity=float(row.get("intensity")),
                item_id=_normalize_item_id(row.get("item_id")),
                repeat_id=int(row.get("repeat_id", 0) or 0),
                option_id=option_id,
            )
            expected = raw_mapping.get(key)
            actual = row.get("chosen_behavior")
            if expected is None:
                continue
            if str(actual) != str(expected):
                mismatched_existing.append((key, str(actual), str(expected)))

    for idx, row in df[to_fill_mask].iterrows():
        option_id = _is_int_like_score(row.get("score"))
        if option_id is None:
            continue
        if option_id <= 0:
            continue
        key = _RowKey(
            emotion=str(row.get("emotion")),
            intensity=float(row.get("intensity")),
            item_id=_normalize_item_id(row.get("item_id")),
            repeat_id=int(row.get("repeat_id", 0) or 0),
            option_id=option_id,
        )
        if key not in raw_mapping:
            missing_keys.append(key)
            continue
        chosen = raw_mapping[key]
        if chosen is None:
            item_key = _RowKeyNoOption(
                emotion=str(row.get("emotion")),
                intensity=float(row.get("intensity")),
                item_id=_normalize_item_id(row.get("item_id")),
                repeat_id=int(row.get("repeat_id", 0) or 0),
            )
            missing_option_ids.append((item_key, option_id))
            continue
        df.at[idx, "chosen_behavior"] = chosen

    if strict:
        if skip_missing_raw and (missing_keys or missing_option_ids):
            first = missing_keys[0] if missing_keys else missing_option_ids[0][0]
            logger.warning(
                "Incomplete raw_results.json mapping; skipping %s (missing_keys=%d missing_option_ids=%d first=%s raw=%s)",
                detailed_csv,
                len(missing_keys),
                len(missing_option_ids),
                first,
                raw_path,
            )
            return False
        if mismatched_existing:
            key, actual, expected = mismatched_existing[0]
            raise ValueError(
                f"chosen_behavior mismatch for {key}: existing={actual!r} expected={expected!r} ({detailed_csv})"
            )
        if missing_keys:
            raise ValueError(
                f"Missing raw_results.json mapping for {len(missing_keys)} rows; first={missing_keys[0]} ({detailed_csv})"
            )
        if missing_option_ids:
            item_key, option_id = missing_option_ids[0]
            available = available_option_ids_by_item.get(item_key, [])
            raise ValueError(
                "Missing option_id in raw metadata options: "
                f"detailed_csv={detailed_csv} raw_json={raw_path} "
                f"option_id={option_id} item_id={item_key.item_id} "
                f"emotion={item_key.emotion} intensity={item_key.intensity} repeat_id={item_key.repeat_id} "
                f"available_option_ids={available}"
            )
    else:
        if mismatched_existing:
            key, actual, expected = mismatched_existing[0]
            logger.warning(
                "chosen_behavior mismatch (kept existing) for %s: existing=%r expected=%r (%s)",
                key,
                actual,
                expected,
                detailed_csv,
            )
        if missing_keys:
            logger.warning(
                "Missing raw mapping for %d rows in %s; first=%s",
                len(missing_keys),
                detailed_csv,
                missing_keys[0],
            )
        if missing_option_ids:
            item_key, option_id = missing_option_ids[0]
            available = available_option_ids_by_item.get(item_key, [])
            logger.warning(
                "Missing option_id in raw metadata options (left NA): detailed_csv=%s raw_json=%s option_id=%s item=%s available=%s",
                detailed_csv,
                raw_path,
                option_id,
                item_key,
                available,
            )

    tmp_path = detailed_csv.with_suffix(".csv.tmp")
    df.to_csv(tmp_path, index=False)
    tmp_path.replace(detailed_csv)
    return True


def _render_progress(done: int, total: int) -> None:
    width = 24
    filled = int(width * done / max(total, 1))
    bar = "#" * filled + "-" * (width - filled)
    sys.stderr.write(f"\r[{bar}] {done}/{total}")
    if done >= total:
        sys.stderr.write("\n")
    sys.stderr.flush()


def add_chosen_behavior_under_root(
    root: Path,
    *,
    strict: bool = False,
    overwrite: bool = False,
    skip_missing_raw: bool = True,
    jobs: int = 1,
    skip_finished: bool = True,
    progress: bool = True,
) -> int:
    root = Path(root)
    paths = list(_iter_detailed_csv_paths(root))
    show_progress = bool(progress and sys.stderr.isatty())

    def _worker(path: Path) -> bool:
        try:
            return update_detailed_results_csv(
                path,
                strict=strict,
                overwrite=overwrite,
                skip_missing_raw=skip_missing_raw,
                skip_finished=skip_finished,
            )
        except Exception as e:
            if strict:
                raise
            logger.warning("Unhandled error in %s; skipping (%s)", path, e)
            return False

    if jobs <= 1 or len(paths) <= 1:
        changed = 0
        for i, p in enumerate(paths, start=1):
            if _worker(p):
                changed += 1
            if show_progress:
                _render_progress(i, len(paths))
        return changed

    with ThreadPoolExecutor(max_workers=jobs) as pool:
        futures = {pool.submit(_worker, p): p for p in paths}
        changed = 0
        done = 0
        for fut in as_completed(futures):
            try:
                if fut.result():
                    changed += 1
            except Exception as e:
                if strict:
                    raise
                logger.warning("Unhandled error in %s; skipping (%s)", futures[fut], e)
            done += 1
            if show_progress:
                _render_progress(done, len(paths))
        return changed


def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Recursively add chosen_behavior to detailed_results.csv (uses raw_results.json)."
    )
    p.add_argument("--root", required=True, help="Root folder to scan recursively")
    p.add_argument(
        "--strict",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Fail on missing option_id / missing raw mappings",
    )
    p.add_argument(
        "--overwrite",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Overwrite existing chosen_behavior values",
    )
    p.add_argument(
        "--skip-missing-raw",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Skip directories missing raw_results.json instead of failing",
    )
    p.add_argument(
        "--jobs",
        type=int,
        default=1,
        help="Number of worker threads to use (IO-bound; 1 disables parallelism)",
    )
    p.add_argument(
        "--skip-finished",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Skip CSVs where chosen_behavior is already fully populated",
    )
    p.add_argument(
        "--progress",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Show a simple progress bar (TTY only)",
    )
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> None:
    args = _parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    updated = add_chosen_behavior_under_root(
        Path(args.root),
        strict=bool(args.strict),
        overwrite=bool(args.overwrite),
        skip_missing_raw=bool(args.skip_missing_raw),
        jobs=int(args.jobs),
        skip_finished=bool(args.skip_finished),
        progress=bool(args.progress),
    )
    print(updated)


if __name__ == "__main__":
    main()
