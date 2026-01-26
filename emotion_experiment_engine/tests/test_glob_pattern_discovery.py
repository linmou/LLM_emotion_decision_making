#!/usr/bin/env python3
"""
Test file focused on glob-style pattern discovery in BenchmarkConfig.

This covers the uncovered case where a config supplies shell-style wildcards
like '*gen*' (common in YAML configs) instead of a formal regex. Previously,
this caused the expansion step to fail and the runner fell back to using the
literal pattern as a filename (e.g., fantom_*gen*.jsonl), leading to a
FileNotFoundError at load time. We assert correct expansion now.

Responsible code: emotion_experiment_engine/data_models.py (BenchmarkConfig)
Purpose: Ensure discover_datasets_by_pattern accepts glob-style patterns.
"""

import tempfile
import unittest
from pathlib import Path

from ..data_models import BenchmarkConfig
from ..emotion_experiment_series_runner import MemoryExperimentSeriesRunner


class TestGlobPatternDiscovery(unittest.TestCase):
    def _touch(self, root: Path, names):
        for n in names:
            (root / n).touch()

    def test_benchmarkconfig_discovers_from_glob(self):
        """'*gen*' should expand to matching task types using fnmatch semantics."""
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            # Create a few fantom files covering 'gen' and non-gen
            files = [
                "fantom_short_belief_gen_accessible.jsonl",
                "fantom_short_belief_gen_inaccessible.jsonl",
                "fantom_full_belief_gen_accessible.jsonl",
                "fantom_short_belief_choice_inaccessible.jsonl",  # should NOT match
                "fantom_short_fact.jsonl",  # should NOT match
            ]
            self._touch(root, files)

            cfg = BenchmarkConfig(
                name="fantom",
                task_type="*gen*",  # glob-style pattern from YAML
                data_path=None,
                base_data_dir=str(root),
                sample_limit=None,
                augmentation_config=None,
                enable_auto_truncation=False,
                truncation_strategy="right",
                preserve_ratio=0.8,
                llm_eval_config=None,
            )

            tasks = cfg.discover_datasets_by_pattern()
            # Extracted task names should exclude the 'fantom_' prefix
            expected = sorted(
                [
                    "short_belief_gen_accessible",
                    "short_belief_gen_inaccessible",
                    "full_belief_gen_accessible",
                ]
            )
            self.assertEqual(sorted(tasks), expected)

    def test_series_runner_expands_glob_in_benchmarks(self):
        """MemoryExperimentSeriesRunner.expand_benchmark_configs should expand '*gen*'."""
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            self._touch(
                root,
                [
                    "fantom_short_belief_gen_accessible.jsonl",
                    "fantom_full_belief_gen_inaccessible.jsonl",
                    "fantom_short_belief_choice_inaccessible.jsonl",  # non-match
                ],
            )

            # Minimal YAML-like config dict to pass into the runner (without actual file IO)
            config = {
                "benchmarks": [
                    {
                        "name": "fantom",
                        "task_type": "*gen*",
                        "base_data_dir": str(root),
                    }
                ],
                # models/emotions/intensities are required by runner, but not used in expansion
                "models": ["dummy-model"],
                "emotions": ["neutral"],
                "intensities": [0.0],
                "output_dir": str(root / "out"),
            }

            # Instantiate runner without reading a YAML file by monkeypatching _load_config
            class _R(MemoryExperimentSeriesRunner):
                def _load_config(self):
                    self.base_config = config

            runner = _R(config_path="unused.yaml", dry_run=True)
            expanded = runner.expand_benchmark_configs(config["benchmarks"])
            expanded_task_types = sorted(b["task_type"] for b in expanded)

            self.assertEqual(
                expanded_task_types,
                sorted(["short_belief_gen_accessible", "full_belief_gen_inaccessible"]),
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)

