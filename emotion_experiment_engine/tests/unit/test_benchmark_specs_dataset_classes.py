#!/usr/bin/env python3
"""
Responsible file: emotion_experiment_engine/benchmark_component_registry.py
Purpose: Ensure BENCHMARK_SPECS holds concrete dataset classes (not strings)
         and that dataset_factory derives its name->class map from the specs.
"""

import unittest
from typing import Dict, Tuple

from emotion_experiment_engine.benchmark_component_registry import BENCHMARK_SPECS
from emotion_experiment_engine.datasets.base import BaseBenchmarkDataset


class TestBenchmarkSpecsDatasetClasses(unittest.TestCase):
    def test_specs_hold_concrete_dataset_classes(self):
        for (name, task), spec in BENCHMARK_SPECS.items():
            self.assertTrue(hasattr(spec, "dataset_class"))
            cls = spec.dataset_class
            # Must be a class and a subclass of BaseBenchmarkDataset
            self.assertTrue(isinstance(cls, type), f"dataset_class for {(name, task)} is not a class")
            self.assertTrue(issubclass(cls, BaseBenchmarkDataset), f"dataset_class for {(name, task)} must extend BaseBenchmarkDataset")

    def test_factory_registry_derives_from_specs(self):
        # Build expected mapping by folding over specs
        expected: Dict[str, type] = {}
        for (name, _task), spec in BENCHMARK_SPECS.items():
            lower = name.lower()
            if lower in expected:
                # Must be consistent across tasks of the same benchmark
                self.assertIs(expected[lower], spec.dataset_class, f"Conflicting dataset_class for benchmark {name}")
            else:
                expected[lower] = spec.dataset_class

        # Import here to avoid any circular import concerns in test discovery
        from emotion_experiment_engine.dataset_factory import DATASET_REGISTRY

        self.assertIsInstance(DATASET_REGISTRY, dict)
        self.assertSetEqual(set(expected.keys()), set(DATASET_REGISTRY.keys()))
        for k, v in expected.items():
            self.assertIs(DATASET_REGISTRY[k], v)


if __name__ == "__main__":
    unittest.main()

