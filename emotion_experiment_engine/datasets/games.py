"""Dataset adapter for game theory benchmarks."""

from __future__ import annotations

import json
import logging
import math
import os
import random
import re
from collections import defaultdict
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple
import numpy as np
from scipy import stats  # type: ignore[import-untyped]

from games.game import SequentialGameScenario
from games.game_configs import get_game_config
from pydantic import BaseModel

from .. import evaluation_utils
from ..data_models import BenchmarkItem, ResultRecord
from .base import BaseBenchmarkDataset

logger = logging.getLogger(__name__)
REPO_ROOT = Path(__file__).resolve().parents[2]
_DECISION_PATTERN = re.compile(r'"decision"\s*[:=]\s*"([^"]+)"', re.IGNORECASE)
_SINGLE_QUOTE_PATTERN = re.compile(r"'decision'\s*[:=]\s*'([^']+)'", re.IGNORECASE)
_OPTION_LINE_PATTERN = re.compile(r"\s*Option\s*(\d+)[\.:\)]\s*(.+)", re.IGNORECASE)


class GameTheoryDataset(BaseBenchmarkDataset):
    """Benchmark dataset that exposes game theory scenarios as BenchmarkItems."""

    LLM_EVAL_CONFIG = {
        "model": "gpt-4o-mini",
        "temperature": 0.0,
        "client": "openai",
    }

    def __init__(
        self,
        config,
        prompt_wrapper: Optional[Any] = None,
        max_context_length: Optional[int] = None,
        tokenizer: Any = None,
        truncation_strategy: str = "right",
        answer_wrapper: Optional[Any] = None,
        **kwargs: Any,
    ) -> None:
        base_config = deepcopy(get_game_config(config.task_type))
        if config.augmentation_config:
            base_config.update(config.augmentation_config)
        self._game_config = base_config
        super().__init__(
            config=config,
            prompt_wrapper=prompt_wrapper,
            max_context_length=max_context_length,
            tokenizer=tokenizer,
            truncation_strategy=truncation_strategy,
            answer_wrapper=answer_wrapper,
            **kwargs,
        )
        self._llm_client = None  # Lazily constructed for fallback parsing

    # ---------------------------------------------------------------------
    # Data loading
    # ---------------------------------------------------------------------
    def _load_and_parse_data(self) -> List[BenchmarkItem]:
        raw_items = self._load_raw_scenarios()

        # For configs without a structured scenario_class, fall back to raw items.
        if "scenario_class" not in self._game_config:
            return self._build_items_from_raw(raw_items)

        scenario_class = self._game_config["scenario_class"]
        payoff_matrix = self._game_config["payoff_matrix"]
        augmentation = self.config.augmentation_config or {}
        scenario_fields = getattr(scenario_class, "model_fields", {})
        config_fields = self._game_config
        # Always shuffle options; allow deterministic control via behavior_ratio.
        behavior_ratio = self._game_config.get("behavior_ratio")
        shuffle_rng = (
            random.Random(behavior_ratio)
            if behavior_ratio is not None
            else random
        )
        shuffle_options = bool(self._game_config.get("shuffle_options", True))

        items: List[BenchmarkItem] = []
        for idx, record in enumerate(raw_items):
            enriched = dict(record)
            if "payoff_matrix" not in enriched:
                enriched["payoff_matrix"] = payoff_matrix

            for field_name in scenario_fields:
                if field_name in augmentation and field_name not in enriched:
                    enriched[field_name] = augmentation[field_name]
                elif field_name in config_fields and field_name not in enriched:
                    enriched[field_name] = config_fields[field_name]

            if "previous_actions_length" in scenario_fields and "previous_actions_length" not in enriched:
                previous_actions = enriched.get("previous_actions") or []
                enriched["previous_actions_length"] = len(previous_actions) if isinstance(previous_actions, list) else 0

            try:
                scenario = scenario_class(**enriched)
            except Exception as exc:  # pragma: no cover - defensive logging
                logger.warning(
                    "Skipping scenario %s due to parse error: %r", idx, exc
                )
                continue

            # Build behavior-category options, then shuffle and reindex.
            options: List[Dict[str, Any]] = []
            item_id = enriched.get("id", idx)
            use_bins_key = self._game_config.get("use_augmented_bins")
            if isinstance(use_bins_key, str) and use_bins_key:
                persisted_bins = record.get("augmented_bins") or {}
                selected = persisted_bins.get(use_bins_key)
                if isinstance(selected, list) and selected:
                    for opt_idx, opt in enumerate(selected):
                        if not isinstance(opt, dict):
                            continue
                        text = opt.get("text")
                        behavior = opt.get("behavior")
                        if not isinstance(text, str) or not text.strip():
                            continue
                        if not isinstance(behavior, str) or not behavior.strip():
                            behavior = "interpolated"
                        options.append(
                            {"id": opt_idx + 1, "text": text.strip(), "behavior": behavior.strip()}
                        )

            augmented_options_enabled = bool(self._game_config.get("use_augmented_options"))
            if not options and augmented_options_enabled:
                field_name = self._game_config.get(
                    "augmented_options_field", "augmented_options_v1"
                )
                expected_bins = self._game_config.get("augmented_options_bins")
                expected_len: int | None = None
                if expected_bins is not None:
                    try:
                        expected_len = int(expected_bins)
                    except (TypeError, ValueError):
                        expected_len = None

                candidate = record.get(field_name)
                if not isinstance(candidate, list) or not candidate:
                    raise ValueError(
                        f"Missing augmented options field '{field_name}' for item '{item_id}'"
                    )
                if expected_len is not None and len(candidate) != expected_len:
                    raise ValueError(
                        f"Expected {expected_len} augmented options for item '{item_id}', "
                        f"found {len(candidate)}"
                    )
                for opt_idx, choice in enumerate(candidate):
                    if not isinstance(choice, str) or not choice.strip():
                        continue
                    try:
                        behavior = scenario.find_behavior_from_decision(choice)
                    except Exception:
                        behavior = "interpolated"
                    options.append(
                        {
                            "id": opt_idx + 1,
                            "text": choice.strip(),
                            "behavior": behavior,
                        }
                    )
                if not options:
                    raise ValueError(
                        f"No valid augmented options could be parsed from '{field_name}' "
                        f"for item '{item_id}'"
                    )

            if not options and not augmented_options_enabled:
                raw_choices = scenario.get_behavior_choices().get_choices()
                for opt_idx, choice in enumerate(raw_choices):
                    try:
                        behavior = scenario.find_behavior_from_decision(choice)
                    except Exception:  # pragma: no cover - defensive guard
                        behavior = ""
                    options.append(
                        {
                            "id": opt_idx + 1,
                            "text": choice,
                            "behavior": behavior,
                        }
                    )

            # Shuffle in-place and reassign ids to reflect presented order.
            if shuffle_options:
                shuffle_rng.shuffle(options)
            for new_idx, opt in enumerate(options, start=1):
                opt["id"] = new_idx

            metadata: Dict[str, Any] = {"options": options}
            try:
                scenario_info = scenario.get_scenario_info()
            except Exception:  # pragma: no cover - keep dataset load resilient
                scenario_info = {}
            if isinstance(scenario_info, dict):
                metadata.update(scenario_info)
            if shuffle_options and behavior_ratio is not None:
                metadata["behavior_ratio_used"] = behavior_ratio

            if isinstance(scenario, SequentialGameScenario):
                previous_attr = getattr(scenario, "previous_actions", None)
                try:
                    resolved = previous_attr() if callable(previous_attr) else previous_attr
                except Exception as exc:  # pragma: no cover - defensive logging
                    logger.debug(
                        "Failed to resolve previous actions for scenario %s: %r",
                        item_id,
                        exc,
                    )
                    resolved = None

                if resolved:
                    previous_list = list(resolved)
                    metadata["previous_actions"] = previous_list
                    metadata["previous_actions_length"] = len(previous_list)

            items.append(
                BenchmarkItem(
                    id=item_id,
                    input_text=str(scenario),
                    context=None,
                    ground_truth=None,
                    metadata=metadata,
                )
            )

        if not items:
            raise ValueError(
                f"No scenarios could be loaded for task '{self.config.task_type}'"
            )

        return items

    def _build_items_from_raw(
        self, raw_items: Sequence[Dict[str, Any]]
    ) -> List[BenchmarkItem]:

        items: List[BenchmarkItem] = []
        for idx, record in enumerate(raw_items):
            event = record.get("event") or record.get("scenario") or ""
            option_entries = record.get("options") or []

            normalized_options: List[Dict[str, Any]] = []
            for opt_idx, opt in enumerate(option_entries):
                if isinstance(opt, dict):
                    text = opt.get("text") or opt.get("value") or str(opt)
                    opt_id = opt.get("id") or opt_idx + 1
                else:
                    text = str(opt)
                    opt_id = opt_idx + 1
                normalized_options.append({"id": opt_id, "text": text})

            items.append(
                BenchmarkItem(
                    id=record.get("id", idx),
                    input_text=str(event),
                    context=None,
                    ground_truth=None,
                    metadata=self._compact_metadata(
                        {
                            "options": normalized_options,
                            "scenario": record.get("scenario"),
                            "description": record.get("description"),
                            "participants": record.get("participants"),
                            "game_name": record.get("game_name"),
                            "payoff_description": record.get("payoff_description"),
                        }
                    ),
                )
            )

        if not items:
            raise ValueError("Raw scenario list was empty")

        return items

    @staticmethod
    def _compact_metadata(metadata: Dict[str, Any]) -> Dict[str, Any]:
        return {key: value for key, value in metadata.items() if value is not None}

    class _ExtractionSchema(BaseModel):
        option_id: int
        rationale: str
        decision: str

    def _resolve_data_path(self) -> Path:
        explicit = getattr(self.config, "data_path", None)
        if explicit:
            explicit_path = Path(explicit)
            if not explicit_path.is_absolute():
                explicit_path = REPO_ROOT / explicit_path
            if explicit_path.exists():
                self.config.data_path = explicit_path
                return explicit_path

        candidate = Path(self._game_config["data_path"])
        if not candidate.is_absolute():
            candidate = REPO_ROOT / candidate

        if candidate.exists():
            self.config.data_path = candidate
            return candidate

        if self.config.base_data_dir is not None:
            fallback = Path(self.config.base_data_dir) / candidate.name
            if fallback.exists():
                self.config.data_path = fallback
                return fallback

        raise FileNotFoundError(
            f"Game data file not found for task '{self.config.task_type}'. "
            f"Expected at {candidate}"
        )

    def _load_raw_scenarios(self) -> List[Dict[str, Any]]:
        scenarios = self._game_config.get("scenarios")
        if isinstance(scenarios, list) and scenarios:
            return [dict(item) for item in scenarios]

        path = self._resolve_data_path()
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)

        if not isinstance(data, list):
            raise ValueError(
                f"Game data file {path} must contain a list of scenarios"
            )
        return data

    # ---------------------------------------------------------------------
    # Evaluation
    # ---------------------------------------------------------------------
    def evaluate_response(
        self, response: str, ground_truth: Any, task_name: str, prompt: str
    ) -> float:
        del ground_truth, task_name  # Accuracy not meaningful; return option id

        options = self._extract_options_from_prompt(prompt)
        choice_id = self._extract_option_from_response(response, options)

        if choice_id is not None:
            return float(choice_id)

        choice_id = self._fallback_option_via_llm(response, options)
        if choice_id is not None:
            return float(choice_id)

        logger.warning("Failed to extract option id for response: %s", response)
        return -1.0

    def get_task_metrics(self, task_name: str) -> List[str]:
        del task_name
        return ["option_id"]

    def compute_split_metrics(self, records: List[ResultRecord]) -> Dict[str, Any]:
        base_metrics = super().compute_split_metrics(records)
        overall_rows = self._choice_ratio_rows(records, include_repeat=False)
        repeat_rows = self._choice_ratio_rows(records, include_repeat=True)

        if not overall_rows and not repeat_rows:
            return base_metrics

        metrics = dict(base_metrics) if isinstance(base_metrics, dict) else {}
        metrics["choice_ratio"] = {
            "overall": overall_rows,
            "by_repeat": repeat_rows,
        }

        # Behavior-level ratios derived from item metadata + numeric scores
        behavior_payload = self._behavior_choice_ratios(records)
        if behavior_payload:
            metrics["behavior_choice_ratio"] = behavior_payload

        # Add statistical analysis over categorical choices
        stats_payload = self._compute_stats(records)
        if stats_payload:
            metrics["stats"] = stats_payload
        return metrics

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _extract_options_from_prompt(prompt: str) -> List[str]:
        options: List[str] = []
        for line in prompt.splitlines():
            match = _OPTION_LINE_PATTERN.match(line)
            if match:
                options.append(match.group(2).strip())
        return options

    @staticmethod
    def _extract_option_from_response(
        response: str, options: Sequence[str]
    ) -> Optional[int]:
        # Content-first: if the raw response contains a single option text,
        # prefer that before parsing any ids.
        response_norm = response.lower()
        matched_ids: List[int] = []
        for idx, option in enumerate(options, start=1):
            opt_norm = option.lower().strip()
            if opt_norm and opt_norm in response_norm:
                matched_ids.append(idx)
        if len(matched_ids) == 1:
            return matched_ids[0]

        candidates = []
        for pattern in (_DECISION_PATTERN, _SINGLE_QUOTE_PATTERN):
            match = pattern.search(response)
            if match:
                candidates.append(match.group(1).strip())
        if not candidates:
            # Handle bare "decision: value" cases
            match = re.search(
                r"decision\s*[:=]\s*([^\n\r]+)", response, re.IGNORECASE
            )
            if match:
                candidates.append(match.group(1).strip())

        for candidate in candidates:
            matched = GameTheoryDataset._match_option(candidate, options)
            if matched is not None:
                return matched

            candidate_stripped = candidate.strip()
            if candidate_stripped.isdigit():
                option_id = int(candidate_stripped)
                if 1 <= option_id <= len(options):
                    return option_id

            match = re.match(r"option\s*(\d+)(?:\s*[:\.\)\-]|$)", candidate_stripped, re.IGNORECASE)
            if match:
                option_id = int(match.group(1))
                if 1 <= option_id <= len(options):
                    return option_id
        return None

    @staticmethod
    def _match_option(candidate: str, options: Sequence[str]) -> Optional[int]:
        normalized = candidate.lower().strip()
        for idx, option in enumerate(options, start=1):
            opt_norm = option.lower().strip()
            if normalized == opt_norm:
                return idx
        for idx, option in enumerate(options, start=1):
            opt_norm = option.lower().strip()
            if normalized in opt_norm or opt_norm in normalized:
                return idx
        return None


    def _choice_ratio_rows(
        self, records: List[ResultRecord], *, include_repeat: bool
    ) -> List[Dict[str, Any]]:
        option_counts: Dict[Tuple[Any, ...], Dict[int, int]] = defaultdict(
            lambda: defaultdict(int)
        )
        total_counts: Dict[Tuple[Any, ...], int] = defaultdict(int)

        # If any record has behavior options metadata, restrict choice ratios
        # to that same subset so id-level and behavior-level ratios share the
        # same underlying decisions (FR-006).
        has_behavior_metadata = False
        for record in records:
            meta = getattr(record, "metadata", None)
            if isinstance(meta, dict):
                item_md = meta.get("item_metadata") or {}
                opts = item_md.get("options")
                if isinstance(opts, list) and opts:
                    has_behavior_metadata = True
                    break

        for record in records:
            if has_behavior_metadata:
                meta = getattr(record, "metadata", None)
                if not isinstance(meta, dict):
                    continue
                item_md = meta.get("item_metadata") or {}
                opts = item_md.get("options")
                if not isinstance(opts, list) or not opts:
                    continue

            score = record.score
            if score is None:
                continue

            try:
                option_val = float(score)
            except (TypeError, ValueError):
                continue

            if math.isnan(option_val):
                continue

            option_id = int(option_val)
            key_parts: List[Any] = [record.emotion, record.intensity]
            if include_repeat:
                key_parts.append(record.repeat_id)

            key = tuple(key_parts)
            option_counts[key][option_id] += 1
            total_counts[key] += 1

        rows: List[Dict[str, Any]] = []
        for key in sorted(option_counts.keys()):
            total = total_counts[key]
            if not total:
                continue

            for option_id in sorted(option_counts[key].keys()):
                row = {
                    "emotion": key[0],
                    "intensity": key[1],
                    "option_id": option_id,
                    "ratio": option_counts[key][option_id] / total,
                }
                if include_repeat:
                    row["repeat_id"] = key[2]
                rows.append(row)

        return rows

    def _behavior_choice_ratios(self, records: List[ResultRecord]) -> Dict[str, List[Dict[str, Any]]]:
        """Aggregate behavior-level counts/ratios from numeric scores and metadata.

        FR-004/FR-006/FR-007: derive behavior categories from per-item options and
        ensure that every chosen option_id has a non-empty behavior category.
        """
        def _get_options_for(meta: Dict[str, Any] | None) -> List[Dict[str, Any]]:
            if not meta:
                return []
            item_md = meta.get("item_metadata") or {}
            opts = item_md.get("options")
            if not isinstance(opts, list) or not opts:
                return []
            return opts

        # Counts keyed by (emotion, intensity[, repeat_id], behavior)
        counts_overall: Dict[Tuple[Any, Any, str], int] = defaultdict(int)
        counts_by_repeat: Dict[Tuple[Any, Any, Any, str], int] = defaultdict(int)
        totals_overall: Dict[Tuple[Any, Any], int] = defaultdict(int)
        totals_by_repeat: Dict[Tuple[Any, Any, Any], int] = defaultdict(int)

        for record in records:
            score = record.score
            if score is None:
                continue
            try:
                option_val = float(score)
            except (TypeError, ValueError):
                continue
            if math.isnan(option_val):
                continue

            option_id = int(option_val)
            # Look up behavior category from metadata
            opts = _get_options_for(record.metadata)
            if not opts:
                # No behavior metadata for this item; skip for behavior-level ratios.
                continue
            behavior: Optional[str] = None
            matched_opt: Optional[Dict[str, Any]] = None
            for opt in opts:
                if int(opt.get("id", -1)) == option_id:
                    matched_opt = opt
                    behavior = opt.get("behavior") or None
                    break

            if matched_opt is None:
                # Unmappable option id: surface in an explicit unknown bucket.
                behavior = "unknown"
            elif not behavior:
                # Matched option with missing/empty behavior is still an error.
                raise ValueError(
                    f"Missing behavior category for option_id {option_id} "
                    f"(item_id={record.item_id!r}) while computing behavior-level ratios"
                )

            key_overall = (record.emotion, record.intensity)
            counts_overall[(record.emotion, record.intensity, behavior)] += 1
            totals_overall[key_overall] += 1

            if hasattr(record, "repeat_id"):
                key_rep = (record.emotion, record.intensity, record.repeat_id)
                counts_by_repeat[(record.emotion, record.intensity, record.repeat_id, behavior)] += 1
                totals_by_repeat[key_rep] += 1

        if not totals_overall:
            return {}

        overall_rows: List[Dict[str, Any]] = []
        for (emotion, intensity, behavior), count in counts_overall.items():
            total = totals_overall[(emotion, intensity)]
            if total:
                overall_rows.append(
                    {
                        "emotion": emotion,
                        "intensity": intensity,
                        "behavior_label": behavior,
                        "ratio": count / total,
                    }
                )

        by_repeat_rows: List[Dict[str, Any]] = []
        for (emotion, intensity, repeat_id, behavior), count in counts_by_repeat.items():
            total = totals_by_repeat[(emotion, intensity, repeat_id)]
            if total:
                by_repeat_rows.append(
                    {
                        "emotion": emotion,
                        "intensity": intensity,
                        "repeat_id": repeat_id,
                        "behavior_label": behavior,
                        "ratio": count / total,
                    }
                )

        return {
            "overall": overall_rows,
            "by_repeat": by_repeat_rows,
        }

    def _fallback_option_via_llm(
        self, response: str, options: Sequence[str]
    ) -> Optional[int]:
        if os.environ.get("DISABLE_LLM_JUDGE") == "1":
            return None
        if not options:
            return None

        client_name = str(self.llm_eval_config.get("client", "openai")).lower()
        formatted_options = ", ".join(
            f"Option {idx + 1}: {text}" for idx, text in enumerate(options)
        )

        # Gemini path: delegate to shared evaluation helper
        if client_name == "gemini":
            system_prompt = (
                "You are helping classify a model's decision. "
                "Given the available options, identify which option best matches the response. "
                "Return a JSON object with an integer field 'option_id' indicating the "
                "1-based index of the chosen option. Use -1 if none apply."
            )
            query = (
                f"Available options: {formatted_options}\n\n"
                f"Response:\n{response}"
            )
            try:
                result = evaluation_utils.llm_evaluate_response(
                    system_prompt=system_prompt,
                    query=query,
                    llm_eval_config=self.llm_eval_config,
                )
            except Exception as exc:  # pragma: no cover - network failure safeguard
                logger.warning("LLM extraction failed (gemini): %s", exc)
                return None

            return self._parse_option_id_from_result(result)

        # OpenAI / Azure path: use existing beta parse helper
        client = self._ensure_llm_client()
        if client is None:
            return None

        prompt = (
            "You are helping classify a model's decision. Given the available options "
            f"({formatted_options}), identify which option best matches the following "
            f"response. Respond with JSON containing an integer field named option_id.\n\n"
            f"If the response is not one of the options, return option_id -1.\n\n"
            f"Response:\n{response}"
        )

        try:
            from neuro_manipulation.utils import oai_response  # type: ignore

            result = oai_response(
                prompt,
                client=client,
                model=self.llm_eval_config.get("model", "gpt-4o-mini"),
                response_format=self._ExtractionSchema,
            )
        except Exception as exc:  # pragma: no cover - network failure safeguard
            logger.warning("LLM extraction failed: %s", exc)
            return None

        return self._parse_option_id_from_result(result)

    @staticmethod
    def _parse_option_id_from_result(result: Any) -> Optional[int]:
        if isinstance(result, BaseModel):
            option_id = getattr(result, "option_id", None)
            if isinstance(option_id, int) and option_id > 0:
                return option_id
            return None

        if isinstance(result, dict):
            option_id = result.get("option_id")
            if isinstance(option_id, int) and option_id > 0:
                return option_id
            if isinstance(option_id, str) and option_id.isdigit():
                return int(option_id)
            return None

        text = str(result)
        try:
            data = json.loads(text)
            if isinstance(data, dict):
                option_id = data.get("option_id")
                if isinstance(option_id, int) and option_id > 0:
                    return option_id
                if isinstance(option_id, str) and option_id.isdigit():
                    return int(option_id)
        except json.JSONDecodeError:
            pass

        match = re.search(r"option_id\s*[:=]\s*([0-9]+)", text, re.IGNORECASE)
        if match:
            return int(match.group(1))
        match = re.search(r"option\s*(\d+)", text, re.IGNORECASE)
        if match:
            return int(match.group(1))
        return None

    def _ensure_llm_client(self):
        if self._llm_client is not None:
            return self._llm_client

        client_name = str(self.llm_eval_config.get("client", "openai")).lower()
        try:
            if client_name == "azure":
                from openai import AzureOpenAI  # type: ignore

                from api_configs import AZURE_OPENAI_CONFIG

                self._llm_client = AzureOpenAI(**AZURE_OPENAI_CONFIG)
            else:
                from openai import OpenAI  # type: ignore

                from api_configs import OAI_CONFIG

                self._llm_client = OpenAI(**OAI_CONFIG)
        except Exception as exc:  # pragma: no cover - optional dependency
            logger.warning("Unable to initialise LLM client: %s", exc)
            self._llm_client = None
        return self._llm_client

    # --------------------------
    # Statistical analysis utils
    # --------------------------
    def _compute_stats(self, records: List[ResultRecord]) -> Dict[str, Any]:
        # Build counts per (emotion, intensity) over option IDs
        counts_by_ei: Dict[Tuple[str, float], Dict[int, int]] = defaultdict(lambda: defaultdict(int))
        category_ids: set[int] = set()

        for r in records:
            if r.score is None:
                continue
            try:
                val = float(r.score)
            except (TypeError, ValueError):
                continue
            if math.isnan(val):
                continue
            oid = int(val)
            counts_by_ei[(r.emotion, float(r.intensity))][oid] += 1
            category_ids.add(oid)

        if not counts_by_ei or not category_ids:
            return {}

        cats = sorted(category_ids)

        # Emotion effect: aggregate counts across intensities per emotion
        counts_by_emotion: Dict[str, Dict[int, int]] = defaultdict(lambda: {c: 0 for c in cats})
        for (emotion, _intensity), counts in counts_by_ei.items():
            for c in cats:
                counts_by_emotion[emotion][c] = counts_by_emotion[emotion].get(c, 0) + counts.get(c, 0)

        emotion_effect = self._contingency_stats(counts_by_emotion, cats)

        # Intensity effect: for each emotion with >=2 intensities
        intensity_effect: Dict[str, Any] = {}
        intensities_by_emotion: Dict[str, set[float]] = defaultdict(set)
        for (emotion, intensity) in counts_by_ei.keys():
            intensities_by_emotion[emotion].add(float(intensity))

        for emotion, intens in intensities_by_emotion.items():
            if len(intens) < 2:
                continue
            counts_by_intensity: Dict[str, Dict[int, int]] = {}
            for i in sorted(intens):
                key = (emotion, float(i))
                counts = counts_by_ei.get(key, {})
                counts_by_intensity[str(i)] = {c: counts.get(c, 0) for c in cats}
            intensity_effect[emotion] = self._contingency_stats(counts_by_intensity, cats)

        payload: Dict[str, Any] = {
            "category_ids": cats,
            "emotion_effect": emotion_effect,
        }
        if intensity_effect:
            payload["intensity_effect"] = intensity_effect
        return payload

    def _contingency_stats(self, counts_by_condition: Dict[str, Dict[int, int]], cats: List[int]) -> Dict[str, Any]:
        # Build table rows in stable condition order
        conditions = sorted(counts_by_condition.keys(), key=lambda x: str(x))
        table = np.array([[counts_by_condition[cond].get(c, 0) for c in cats] for cond in conditions], dtype=float)

        chi2, p_value = self._chi_or_fisher(table)
        result: Dict[str, Any] = {
            "conditions": conditions,
            "chi_square": float(chi2),
            "p_value": float(p_value),
            "significant": bool(p_value < 0.05),
            "pairwise": {},
        }

        # Pairwise comparisons
        for i in range(len(conditions)):
            for j in range(i + 1, len(conditions)):
                a = table[i, :]
                b = table[j, :]
                sub = np.vstack([a, b])
                chi2_pw, p_pw = self._chi_or_fisher(sub)
                key = f"{conditions[i]}_vs_{conditions[j]}"
                result["pairwise"][key] = {
                    "chi_square": float(chi2_pw),
                    "p_value": float(p_pw),
                    "significant": bool(p_pw < 0.05),
                }

        return result

    @staticmethod
    def _chi_or_fisher(table: np.ndarray) -> Tuple[float, float]:
        # Use Fisher's exact test only for 2x2 tables with any small cell (<5)
        if table.shape == (2, 2) and (table < 5).any():
            try:
                _, p = stats.fisher_exact(table)
                # Map to a chi-square-like statistic for reporting (optional)
                # We compute chi2 from p-value and dof=1 with an approximate inverse if desired; keep chi2 as NaN
                return float("nan"), float(p)
            except Exception:
                pass
        try:
            chi2, p, _, _ = stats.chi2_contingency(table)
            return float(chi2), float(p)
        except Exception:
            return 0.0, 1.0


class GameTheoryCompletionOptionIdDataset(GameTheoryDataset):
    """Variant evaluator for base LMs emitting completion-style option ids/text."""

    @staticmethod
    def _match_option(candidate: str, options: Sequence[str]) -> Optional[int]:
        normalized = candidate.lower().strip().strip("\"'").strip()
        if not normalized:
            return None

        for idx, option in enumerate(options, start=1):
            opt_norm = option.lower().strip()
            if normalized == opt_norm:
                return idx

        matches: List[Tuple[int, int]] = []
        for idx, option in enumerate(options, start=1):
            opt_norm = option.lower().strip()
            if normalized in opt_norm:
                matches.append((idx, len(normalized)))
            elif opt_norm in normalized:
                matches.append((idx, len(opt_norm)))

        if not matches:
            return None

        matches.sort(key=lambda x: (-x[1], x[0]))
        best_score = matches[0][1]
        best = [idx for idx, score in matches if score == best_score]
        if len(best) == 1:
            return best[0]
        return None

    @staticmethod
    def _extract_option_from_response(
        response: str, options: Sequence[str]
    ) -> Optional[int]:
        # Some base checkpoints emit chat-style speaker tags despite using
        # completion prompts, e.g. "Human: 1" or "Assistant: Option 2".
        response = re.sub(
            r"^\s*(?:human|user|assistant|system)\s*:\s*",
            "",
            response,
            flags=re.IGNORECASE,
        )
        response = re.sub(r"^\s*(?:answer|final)\s*:\s*", "", response, flags=re.IGNORECASE)

        # Start with the strict JSON decision parsing from the base class.
        choice = GameTheoryDataset._extract_option_from_response(response, options)
        if choice is not None:
            return choice

        # Completion-style variants:
        # 1) Leading numeric id: '2', '2.', '"2"', '2) ...'
        match = re.match(r"\s*\"?\s*(\d+)", response)
        if match:
            option_id = int(match.group(1))
            if 1 <= option_id <= len(options):
                return option_id

        # 2) Leading 'Option N. ...' or 'Option N: ...'
        match = re.match(
            r"\s*option\s*(\d+)(?:\s*[:\.\)\-]|$)", response, re.IGNORECASE
        )
        if match:
            option_id = int(match.group(1))
            if 1 <= option_id <= len(options):
                return option_id

        # 3) Full option line parse (captures trailing text too)
        match = _OPTION_LINE_PATTERN.match(response)
        if match:
            option_id = int(match.group(1))
            if 1 <= option_id <= len(options):
                return option_id
            matched = GameTheoryCompletionOptionIdDataset._match_option(
                match.group(2), options
            )
            if matched is not None:
                return matched

        # 4) Plain option text (possibly truncated)
        return GameTheoryCompletionOptionIdDataset._match_option(response, options)


__all__ = ["GameTheoryDataset", "GameTheoryCompletionOptionIdDataset"]
