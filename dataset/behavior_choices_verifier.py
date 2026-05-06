################################################################################
# File: data_creation/scenario_creation/langgraph_creation/behavior_choices_verifier.py
# Purpose: Build and parse LLM-based behavior_choices verification prompts.
#
# This module is intentionally free of LangGraph / LangChain imports so it can be
# unit-tested in isolation. The Diplomacy graph is responsible for providing
# the concrete LLM client and message wrappers.
################################################################################
from __future__ import annotations

import json
from typing import Any, Dict, List, Tuple


def _extract_behavior_choices(state: Dict[str, Any]) -> Dict[str, str]:
    draft = state.get("scenario_draft") or {}
    behaviors = draft.get("behavior_choices") if isinstance(draft, dict) else None
    if not isinstance(behaviors, dict):
        return {}
    # Coerce values to strings for prompting.
    return {k: str(v) for k, v in behaviors.items()}


def build_behavior_verification_prompt(state: Dict[str, Any]) -> Tuple[str, str]:
    """
    Construct system and human prompts to enforce objective, emotion-neutral behaviors.

    The prompt explicitly asks the LLM to:
    - Reject emotional adjectives/adverbs (e.g., aggressively, happily)
    - Reject moral or evaluative labels
    - Reject non-objective coordination words like "coordinate" / "cooperate"
    - Require concrete descriptions of actions instead of feelings or intent.
    """
    behaviors = _extract_behavior_choices(state)
    game_name = state.get("game_name", "Unknown_Game")
    participants = state.get("participants") or []
    description = (state.get("scenario_draft") or {}).get("description", "")

    system_prompt = (
        "You are reviewing candidate behavior options for a game-theoretic scenario.\n"
        "Your job is to check the wording of each behavior option.\n\n"
        "Requirements:\n"
        "1. Behaviors must be phrased as objective actions (what units/agents do), "
        "   not as feelings, intentions, or styles.\n"
        "2. Do NOT allow emotional adjectives or adverbs such as "
        "   'aggressive', 'aggressively', 'angry', 'angrily', 'happy', 'happily', etc.\n"
        "3. Do NOT allow moral or evaluative labels such as "
        "   'generous', 'selfish', 'altruistic', 'cowardly', 'brave', etc.\n"
        "4. Do NOT allow coordination-style verbs like 'coordinate', 'coordinated', "
        "   'cooperated', 'cooperate', 'cooperative'. Instead, describe the concrete "
        "   actions each side takes.\n"
        "5. Behaviors should be specific and grounded in the scenario description, "
        "   e.g., which unit moves where, or which area is defended.\n\n"
        "You must not rewrite the behaviors yourself. Only judge whether each option "
        "satisfies the criteria above."
    )

    human_payload = {
        "game_name": game_name,
        "participants": participants,
        "scenario_description": description,
        "behavior_choices": behaviors,
        "instructions": (
            "For each behavior option key, decide if the text is acceptable.\n"
            "Return JSON with exactly these fields:\n"
            '- "feedback": a list of strings, one or more messages describing any '
            "violations you notice (include the option key in each message). If all "
            "options are acceptable, this list may be empty.\n"
            '- "converged": boolean, true only if all options are acceptable.\n'
            "Be strict: if any option contains emotional, moral, or coordination words, "
            "set converged to false and explain which option failed and why."
        ),
    }
    human_prompt = json.dumps(human_payload, indent=2)
    return system_prompt, human_prompt


def parse_behavior_verification_result(result_content: Any) -> Dict[str, Any]:
    """
    Normalize raw LLM result into {\"feedback\": [...], \"converged\": bool}.
    Accepts either a JSON string or a dict-like object.
    """
    if isinstance(result_content, str):
        try:
            obj = json.loads(result_content)
        except json.JSONDecodeError:
            return {
                "feedback": ["Error parsing behavior verification result"],
                "converged": False,
            }
    elif isinstance(result_content, dict):
        obj = result_content
    else:
        return {
            "feedback": ["Unexpected format for behavior verification result"],
            "converged": False,
        }

    feedback = obj.get("feedback", [])
    converged = obj.get("converged", False)
    if not isinstance(feedback, list):
        feedback = [str(feedback)]
    return {"feedback": feedback, "converged": bool(converged)}


def verify_behavior_choices(state: Dict[str, Any], llm: Any) -> Dict[str, Any]:
    """
    LLM-based verifier entry point.

    This function is intentionally simple: it delegates prompt construction and
    result parsing to helpers, and relies on the caller to supply a configured
    LLM client with an `.invoke(messages, response_format=...)` method.

    The `messages` passed to the LLM are a list of dicts with `role` and `content`
    fields, so callers can wrap or translate them into framework-specific types.
    """
    if llm is None:
        return {
            "behavior_feedback": [
                "No LLM client provided for behavior verification."
            ],
            "behavior_converged": False,
        }

    behaviors = _extract_behavior_choices(state)
    if not behaviors:
        return {
            "behavior_feedback": [
                "Scenario draft is missing a behavior_choices dictionary to validate."
            ],
            "behavior_converged": False,
        }

    system_prompt, human_prompt = build_behavior_verification_prompt(state)
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": human_prompt},
    ]

    try:
        response = llm.invoke(messages, response_format={"type": "json_object"})
        raw = getattr(response, "content", response)
    except Exception as e:
        return {
            "behavior_feedback": [
                f"Error calling LLM for behavior verification: {e}"
            ],
            "behavior_converged": False,
        }

    parsed = parse_behavior_verification_result(raw)
    return {
        "behavior_feedback": parsed["feedback"],
        "behavior_converged": parsed["converged"],
    }

