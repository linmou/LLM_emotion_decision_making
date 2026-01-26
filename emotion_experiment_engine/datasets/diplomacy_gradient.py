"""Dataset for PD-style Diplomacy decision gradients."""

from __future__ import annotations

import math
import re
from typing import Any, Dict, List, Optional, Sequence

from ..data_models import BenchmarkItem
from .base import BaseBenchmarkDataset


_OPTION_LINE_PATTERN = re.compile(r"\s*Option\s*(\d+)[\.:)]\s*(.+)", re.IGNORECASE)
_OPTION_NUMBER_PATTERN = re.compile(r"option\s*(\d+)", re.IGNORECASE)

BEHAVIOR_OPTION_ORDER = {"withdraw": 1, "escalate": 2}


class DiplomacyGradientDataset(BaseBenchmarkDataset):
    """Simple adapter for Diplomacy gradient choices (1..5 options)."""

    LLM_EVAL_CONFIG = {
        "model": "gpt-4o-mini",
        "temperature": 0.0,
        "client": "openai",
    }

    def _load_and_parse_data(self) -> List[BenchmarkItem]:
        raw = self._load_raw_data()
        items: List[BenchmarkItem] = []

        for idx, record in enumerate(raw):
            scenario = str(record.get("scenario") or record.get("event") or "").strip()
            description = str(record.get("description") or "").strip()
            question_text = description or scenario
            behavior_choices = record.get("behavior_choices") or {}
            if behavior_choices:
                options_source = []
                for label, opt_id in sorted(BEHAVIOR_OPTION_ORDER.items(), key=lambda kv: kv[1]):
                    if label in behavior_choices:
                        options_source.append({"id": opt_id, "text": str(behavior_choices[label])})
                next_id = (max(BEHAVIOR_OPTION_ORDER.values()) if BEHAVIOR_OPTION_ORDER else 0) + 1
                for label, text in behavior_choices.items():
                    if label not in BEHAVIOR_OPTION_ORDER:
                        options_source.append({"id": next_id, "text": str(text)})
                        next_id += 1
            else:
                options_source = (
                    record.get("gradient_options")
                    or record.get("options")
                    or []
                )
            options = options_source
            if not question_text or not options:
                # Skip malformed rows; keep dataset resilient to editing mistakes.
                continue

            normalized_options: List[Dict[str, Any]] = []
            for opt_index, option in enumerate(options):
                if isinstance(option, dict):
                    text = option.get("text") or option.get("value") or str(option)
                    opt_id = option.get("id") or opt_index + 1
                else:
                    text = str(option)
                    opt_id = opt_index + 1
                normalized_options.append({"id": int(opt_id), "text": str(text)})

            header_lines: List[str] = []
            whose_option = record.get("whose_option")
            your_country = record.get("your_country") or whose_option
            game_name = record.get("game") or record.get("game_name")
            target_country = record.get("target_country")
            phase = record.get("phase") or {}
            season = phase.get("season")
            year = phase.get("year")
            subphase = phase.get("subphase")

            if your_country:
                header_lines.append(f"Your Country: {your_country}")
            if game_name:
                header_lines.append(f"Game: {game_name}")
            if whose_option and not your_country:
                header_lines.append(f"Decision Owner: {whose_option}")
            phase_bits = [str(bit) for bit in (season, year, subphase) if bit]
            if phase_bits:
                header_lines.append("Phase: " + " ".join(phase_bits))
            if target_country:
                header_lines.append(f"Target Country: {target_country}")
            if scenario:
                header_lines.append(f"Scenario: {scenario}")

            context_header = "\n".join(header_lines) if header_lines else None

            items.append(
                BenchmarkItem(
                    id=record.get("id", idx),
                    input_text=question_text,
                    context=context_header,
                    ground_truth=None,
                    metadata={
                        "options": normalized_options,
                        "scenario": scenario,
                        "whose_option": whose_option,
                    },
                )
            )

        if not items:
            raise ValueError("No valid Diplomacy PD-style items found")
        return items

    # ------------------------------------------------------------------
    # Evaluation helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _extract_options_from_prompt(prompt: str) -> List[str]:
        extracted: List[str] = []
        for line in prompt.splitlines():
            match = _OPTION_LINE_PATTERN.match(line)
            if match:
                extracted.append(match.group(2).strip())
        return extracted

    @staticmethod
    def _normalize(text: str) -> str:
        return re.sub(r"\s+", " ", text.strip()).lower()

    def _extract_option_from_response(
        self, response: str, options: Sequence[str]
    ) -> Optional[int]:
        if not response:
            return None

        match = _OPTION_NUMBER_PATTERN.search(response)
        if match:
            option_id = int(match.group(1))
            if 1 <= option_id <= len(options):
                return option_id

        normalized_response = self._normalize(response)
        for idx, opt_text in enumerate(options, start=1):
            normalized_option = self._normalize(opt_text)
            if normalized_option in normalized_response or normalized_response in normalized_option:
                return idx
        return None

    def evaluate_response(
        self, response: str, ground_truth: Any, task_name: str, prompt: str
    ) -> float:
        del ground_truth, task_name
        options = self._extract_options_from_prompt(prompt)
        choice = self._extract_option_from_response(response, options)
        return float(choice) if choice is not None else math.nan

    def get_task_metrics(self, task_name: str) -> List[str]:
        del task_name
        return ["option_id"]


__all__ = ["DiplomacyGradientDataset"]
