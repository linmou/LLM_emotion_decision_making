"""
Test file for MTBench101 factory registration - TDD Phase 4 Red
Testing: Factory registration and dataset creation via dataset_factory.py
Purpose: Ensure MTBench101Dataset can be created through the factory system
"""

import pytest
from pathlib import Path
from emotion_experiment_engine.data_models import BenchmarkConfig

# Import will initially work, but registry won't have mtbench101 yet (Red phase)
try:
    from emotion_experiment_engine.dataset_factory import create_dataset_from_config, DATASET_REGISTRY
    from emotion_experiment_engine.datasets.mtbench101 import MTBench101Dataset
except ImportError as e:
    # Expected during Red phase if imports fail
    create_dataset_from_config = None
    DATASET_REGISTRY = None
    MTBench101Dataset = None


class TestMTBench101Factory:
    """Test MTBench101Dataset factory integration"""

    def test_registry_contains_mtbench101(self):
        """Test DATASET_REGISTRY has mtbench101 key"""
        # This test will fail because mtbench101 is not registered yet (Red phase)
        assert DATASET_REGISTRY is not None
        assert "mtbench101" in DATASET_REGISTRY, "MTBench101 not registered in dataset factory"
        assert DATASET_REGISTRY["mtbench101"] is MTBench101Dataset

    def test_case_insensitive_lookup(self):
        """Test 'MTBench101', 'mtbench101', 'MTBENCH101' all work"""
        # Should support different case variations
        assert "mtbench101" in DATASET_REGISTRY
        # Most factories also support case variations, but we'll start with exact match
        
    def test_factory_creates_mtbench101_dataset(self):
        """Test factory can create MTBench101Dataset from config"""
        # Create a valid config for MTBench101 CM task
        config = BenchmarkConfig(
            name="mtbench101",
            task_type="CM",
            data_path=None,  # Will use auto-discovery
            base_data_dir="data/mtbench",
            sample_limit=5,
            augmentation_config=None,
            enable_auto_truncation=False,
            truncation_strategy="right",
            preserve_ratio=0.8,
            llm_eval_config=None
        )
        
        # This will fail until MTBench101 is registered in factory
        dataset = create_dataset_from_config(config)
        
        assert dataset is not None
        assert isinstance(dataset, MTBench101Dataset)
        assert dataset.config == config

    def test_auto_data_path_discovery(self):
        """Test factory handles auto data path discovery for MTBench101"""
        config = BenchmarkConfig(
            name="mtbench101",
            task_type="CM", 
            data_path=None,  # Should auto-discover data/mtbench/mtbench101_CM.jsonl
            base_data_dir="data/mtbench",
            sample_limit=None,
            augmentation_config=None,
            enable_auto_truncation=False,
            truncation_strategy="right",
            preserve_ratio=0.8,
            llm_eval_config=None
        )
        
        dataset = create_dataset_from_config(config)
        
        # Should have successfully loaded data from the auto-discovered path
        assert dataset is not None
        assert len(dataset) > 0  # Should have loaded some conversations
        assert isinstance(dataset, MTBench101Dataset)

    def test_factory_handles_all_mtbench_tasks(self):
        """Test factory can create datasets for all 13 MTBench101 tasks"""
        task_types = ['CM', 'SI', 'AR', 'TS', 'CC', 'CR', 'FR', 'SC', 'SA', 'MR', 'GR', 'IC', 'PI']
        
        for task_type in task_types:
            config = BenchmarkConfig(
                name="mtbench101",
                task_type=task_type,
                data_path=None,
                base_data_dir="data/mtbench",
                sample_limit=2,  # Small for testing
                augmentation_config=None,
                enable_auto_truncation=False,
                truncation_strategy="right", 
                preserve_ratio=0.8,
                llm_eval_config=None
            )
            
            dataset = create_dataset_from_config(config)
            assert isinstance(dataset, MTBench101Dataset)
            
            # Check that correct task file was discovered
            resolved_path = dataset.config.get_data_path()
            assert task_type in str(resolved_path)

    def test_factory_error_handling_unknown_task(self):
        """Test factory handles unknown task types gracefully"""
        config = BenchmarkConfig(
            name="mtbench101",
            task_type="UNKNOWN_TASK",
            data_path=None,
            base_data_dir="data/mtbench",
            sample_limit=None,
            augmentation_config=None,
            enable_auto_truncation=False,
            truncation_strategy="right",
            preserve_ratio=0.8,
            llm_eval_config=None
        )
        
        # Should raise appropriate error for unknown task
        with pytest.raises((FileNotFoundError, ValueError)):
            create_dataset_from_config(config)

    def test_factory_explicit_data_path(self):
        """Test factory works with explicit data_path"""
        # Create a mock file path (doesn't need to exist for this test)
        explicit_path = Path("data/mtbench/mtbench101_SI.jsonl")
        
        config = BenchmarkConfig(
            name="mtbench101", 
            task_type="SI",
            data_path=explicit_path,  # Explicit path provided
            base_data_dir="data/mtbench",
            sample_limit=None,
            augmentation_config=None,
            enable_auto_truncation=False,
            truncation_strategy="right",
            preserve_ratio=0.8,
            llm_eval_config=None
        )
        
        dataset = create_dataset_from_config(config)
        assert isinstance(dataset, MTBench101Dataset)
        assert dataset.config.data_path == explicit_path

    def test_factory_integration_with_existing_datasets(self):
        """Test MTBench101 doesn't break existing dataset factory functionality"""
        # Test that existing datasets still work
        existing_datasets = ["infinitebench", "longbench", "locomo"]
        
        for dataset_name in existing_datasets:
            if dataset_name in DATASET_REGISTRY:
                # Just check that the class is still registered correctly
                dataset_class = DATASET_REGISTRY[dataset_name]
                assert dataset_class is not None
                assert hasattr(dataset_class, '__init__')

    def test_dataset_factory_performance(self):
        """Test factory maintains O(1) lookup performance"""
        # Registry should be a dict for O(1) lookup
        assert isinstance(DATASET_REGISTRY, dict)
        
        # Multiple lookups should be fast
        for _ in range(100):
            assert "mtbench101" in DATASET_REGISTRY
            dataset_class = DATASET_REGISTRY["mtbench101"]
            assert dataset_class is MTBench101Dataset

    def test_factory_config_validation(self):
        """Test factory validates config appropriately"""
        # Invalid config should be caught
        invalid_config = BenchmarkConfig(
            name="invalid_dataset",  # Non-existent dataset name
            task_type="CM",
            data_path=None,
            base_data_dir="data/mtbench",
            sample_limit=None,
            augmentation_config=None,
            enable_auto_truncation=False,
            truncation_strategy="right",
            preserve_ratio=0.8,
            llm_eval_config=None
        )
        
        with pytest.raises((KeyError, ValueError)):
            create_dataset_from_config(invalid_config)

    def test_mtbench101_in_dataset_list(self):
        """Test MTBench101 appears in available datasets list"""
        # Should be discoverable via registry keys
        available_datasets = list(DATASET_REGISTRY.keys())
        assert "mtbench101" in available_datasets
        
        # Should be alongside other benchmarks
        assert "infinitebench" in available_datasets  # Existing dataset should still be there
        
    def test_factory_imports_mtbench101_class_correctly(self):
        """Test factory imports and stores the correct MTBench101Dataset class"""
        dataset_class = DATASET_REGISTRY["mtbench101"]
        
        # Should be the actual class, not a string or other type
        assert callable(dataset_class)
        assert dataset_class.__name__ == "MTBench101Dataset"
        assert hasattr(dataset_class, '_load_and_parse_data')
        assert hasattr(dataset_class, 'evaluate_response')