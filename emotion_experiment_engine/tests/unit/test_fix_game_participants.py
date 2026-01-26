# Tests for scripts.fix_game_participants
"""Validate dataset repair utilities for participant roles."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.fix_game_participants import repair_participant_roles


@pytest.fixture()
def sample_paths(tmp_path: Path) -> tuple[Path, Path]:
    trust_data = [
        {
            "scenario": "Trust sample",
            "description": "",
            "participants": [
                {"name": "Alice", "profile": "Investor"},
                {"name": "Bob", "profile": "Trustee profile"},
            ],
            "trustor_behavior_choices": {
                "trust_none": "A",
                "trust_low": "B",
                "trust_high": "C",
            },
            "trustee_behavior_choices": {
                "return_none": "X",
                "return_medium": "Y",
                "return_high": "Z",
            },
            "previous_actions_length": 1,
            "previous_trust_level": 0,
        }
    ]

    ultimatum_data = [
        {
            "scenario": "Ultimatum sample",
            "description": "",
            "participants": [
                {"name": "Cara", "profile": "Artist", "role": "Space Coordinator"},
                {"name": "Dan", "profile": "Artist"},
            ],
            "proposer_behavior_choices": {
                "offer_low": "low",
                "offer_medium": "medium",
                "offer_high": "high",
            },
            "responder_behavior_choices": {
                "accept": "accept",
                "reject": "reject",
            },
            "previous_actions_length": 0,
        }
    ]

    trust_path = tmp_path / "trust.json"
    ultimatum_path = tmp_path / "ultimatum.json"

    trust_path.write_text(json.dumps(trust_data))
    ultimatum_path.write_text(json.dumps(ultimatum_data))
    return trust_path, ultimatum_path


@pytest.mark.parametrize(
    "prime_valid, expected_trust, expected_ult",
    [
        (False, 1, 1),
        (True, 0, 0),
    ],
)
def test_repair_participant_roles_behavior(
    sample_paths: tuple[Path, Path],
    prime_valid: bool,
    expected_trust: int,
    expected_ult: int,
) -> None:
    trust_path, ultimatum_path = sample_paths

    if prime_valid:
        trust_data = json.loads(trust_path.read_text())
        trust_data[0]["participants"][0]["role"] = "Trustor"
        trust_data[0]["participants"][1]["role"] = "Trustee"
        trust_path.write_text(json.dumps(trust_data))

        ultimatum_data = json.loads(ultimatum_path.read_text())
        ultimatum_data[0]["participants"][0]["role"] = "Proposer"
        ultimatum_data[0]["participants"][1]["role"] = "Responder"
        ultimatum_path.write_text(json.dumps(ultimatum_data))

    report = repair_participant_roles(
        trust_game_path=trust_path,
        ultimatum_responder_path=ultimatum_path,
    )

    assert report.trust_fixed == expected_trust
    assert report.ultimatum_fixed == expected_ult

    trust_records = json.loads(trust_path.read_text())
    ultimatum_records = json.loads(ultimatum_path.read_text())

    trust_roles = {p["role"] for p in trust_records[0]["participants"]}
    assert trust_roles == {"Trustor", "Trustee"}

    ultimatum_roles = {p["role"] for p in ultimatum_records[0]["participants"]}
    assert ultimatum_roles == {"Proposer", "Responder"}
