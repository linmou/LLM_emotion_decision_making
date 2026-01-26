"""
Red phase tests: ensure BENCHMARK_SPECS contains TrustLLM family entries.

We avoid importing the registry (which pulls heavy deps). Instead we assert that
the source file contains mapping entries for each family key, which protects the
contract without incurring import-time dependencies.
"""

from pathlib import Path


def test_registry_contains_trustllm_family_entries():
    src = Path("emotion_experiment_engine/benchmark_component_registry.py").read_text(
        encoding="utf-8"
    )

    required_keys = [
        "(\"trustllm_ethics\", \"*\")",
        "(\"trustllm_fairness\", \"*\")",
        "(\"trustllm_privacy\", \"*\")",
        "(\"trustllm_robustness\", \"*\")",
        "(\"trustllm_safety\", \"*\")",
        "(\"trustllm_truthfulness\", \"*\")",
    ]

    for key in required_keys:
        assert key in src, f"Registry missing entry for {key}"

