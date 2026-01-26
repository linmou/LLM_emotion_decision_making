"""
Responsible file: emotion_experiment_engine/benchmark_component_registry.py
Purpose: Ensure registry declares ("pubmed_qa", "pqa_labeled") mapping.

Red phase: This test asserts source text contains the mapping literal without
importing the registry (avoids heavy imports and side effects).
"""

from pathlib import Path


def test_registry_contains_pubmedqa_entry():
    src = Path("emotion_experiment_engine/benchmark_component_registry.py").read_text(
        encoding="utf-8"
    )
    # Assert the exact literal appears in the mapping
    assert "(\"pubmed_qa\", \"pqa_labeled\")" in src, (
        "Registry missing entry for ('pubmed_qa', 'pqa_labeled')"
    )
