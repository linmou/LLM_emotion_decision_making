"""
Responsible file: emotion_experiment_engine/dataset_factory.py
Purpose: Verify dataset_registry derives PubMedQADataset for 'pubmed_qa'.

Red phase: Fails until BENCHMARK_SPECS declares the mapping and the
PubMedQADataset class exists.
"""

import unittest

from emotion_experiment_engine.dataset_factory import get_dataset_class


class TestDatasetRegistryPubMedQA(unittest.TestCase):
    def test_get_dataset_class_pubmedqa(self):
        cls = get_dataset_class("pubmed_qa")
        # Expect the concrete class to be named PubMedQADataset
        # (class existence + registry derivation)
        self.assertIsNotNone(cls, "Dataset class for pubmed_qa should be registered")
        self.assertEqual(getattr(cls, "__name__", None), "PubMedQADataset")


if __name__ == "__main__":
    unittest.main()

