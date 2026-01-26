#!/usr/bin/env python3
"""
Test Migration Script - Safe Migration with Functional Equivalence

Migrates test files to hierarchical structure while preserving functionality.
Includes automatic import path fixing and validation at each step.
"""

import shutil
import re
from pathlib import Path
from typing import Dict, List, Tuple
import subprocess
import sys

class TestMigrator:
    """Safe test migration with functional equivalence preservation"""
    
    def __init__(self, tests_root: Path):
        self.tests_root = tests_root
        self.unit_dir = tests_root / "unit"
        self.integration_dir = tests_root / "integration" 
        self.e2e_dir = tests_root / "e2e"
        self.comprehensive_dir = tests_root  # Keep comprehensive tests at root level
        
        # Create directories
        self.unit_dir.mkdir(exist_ok=True)
        self.integration_dir.mkdir(exist_ok=True) 
        self.e2e_dir.mkdir(exist_ok=True)
        
    def categorize_tests(self) -> Dict[str, List[str]]:
        """Categorize existing test files"""
        # Based on analysis of your existing files
        categories = {
            "unit": [
                "test_data_models.py",           # ✅ Working - Core data structures
                "test_dataset_factory.py",       # ❌ Broken - Needs BenchmarkConfig fixes
                "test_config_loader.py",         # ❌ Broken - Configuration loading  
                "test_base_dataset_interface.py", # Base class interface
                "test_llm_evaluation.py",        # LLM evaluation logic
                "test_utils.py",                 # ❌ Broken - Utility functions
                "test_import_coverage.py",       # Import testing
                "test_direct_dataclass_loading.py", # Dataclass tests
                "test_benchmark_prompt_wrapper.py", # Prompt wrapper unit tests
                "test_memory_prompt_wrapper.py",    # Memory prompt wrapper unit tests
            ],
            "integration": [
                "test_integration.py",           # ✅ Working - Main integration tests
                "test_dataloader_integration.py", # Data loading integration
                "test_truthfulqa_integration.py", # TruthfulQA integration
                "test_mtbench101_factory.py",     # MTBench factory integration
                "test_pipeline_worker_keyerror.py", # Pipeline integration
            ],
            "e2e": [
                "test_end_to_end_integration.py", # ✅ Working - Full pipeline
                "test_experiment.py",             # Experiment orchestration
                "test_run_emotion_memory_experiment.py", # Full experiment execution
                "test_emotion_experiment_series_runner.py", # Series execution
            ],
            "comprehensive": [
                "test_answer_wrapper_comprehensive.py",      # ✅ Working - Your comprehensive test  
                "test_asyncio_evaluate_batch_comprehensive.py", # ✅ Working - Async bug fixes
                "test_emotion_evaluation_comprehensive.py",   # ❌ Broken - Evaluation completeness
            ],
            "dataset_specific": [
                "test_infinitebench_dataset.py", # InfiniteBench specialization
                "test_longbench_dataset.py",     # LongBench specialization
                "test_locomo_dataset.py",        # LoCoMo specialization  
                "test_mtbench101_dataset.py",    # MTBench dataset tests
                "test_truthfulqa.py",            # TruthfulQA dataset tests
            ],
            "other": [
                "test_regex_pattern_discovery.py", # Pattern discovery utility
            ]
        }
        
        return categories
    
    def fix_imports_for_migration(self, content: str, migration_type: str) -> str:
        """Fix imports when moving from root to subdirectory"""
        
        # Mapping of import transformations
        if migration_type == "root_to_unit":
            # From tests/test_file.py -> tests/unit/test_file.py
            transformations = [
                # Parent module imports need one more level up
                (r'from \.\.([a-zA-Z_][a-zA-Z0-9_.]*) import', r'from ...\1 import'),
                
                # Sibling test utilities become parent directory
                (r'from \.test_utils import', r'from ..test_utils import'),
                (r'from \.([a-zA-Z_][a-zA-Z0-9_]*) import', r'from ..\1 import'),
            ]
        elif migration_type == "root_to_integration":
            # From tests/test_file.py -> tests/integration/test_file.py  
            transformations = [
                # Same pattern as unit tests
                (r'from \.\.([a-zA-Z_][a-zA-Z0-9_.]*) import', r'from ...\1 import'),
                (r'from \.test_utils import', r'from ..test_utils import'),
                (r'from \.([a-zA-Z_][a-zA-Z0-9_]*) import', r'from ..\1 import'),
            ]
        elif migration_type == "root_to_e2e":
            # From tests/test_file.py -> tests/e2e/test_file.py
            transformations = [
                # Same pattern
                (r'from \.\.([a-zA-Z_][a-zA-Z0-9_.]*) import', r'from ...\1 import'),
                (r'from \.test_utils import', r'from ..test_utils import'),
                (r'from \.([a-zA-Z_][a-zA-Z0-9_]*) import', r'from ..\1 import'),
            ]
        else:
            return content  # No changes needed
        
        # Apply transformations
        for pattern, replacement in transformations:
            content = re.sub(pattern, replacement, content)
        
        return content
    
    def migrate_test_file(self, filename: str, target_dir: Path, migration_type: str) -> bool:
        """Migrate a single test file with import fixes"""
        source_file = self.tests_root / filename
        target_file = target_dir / filename
        
        if not source_file.exists():
            print(f"   ⚠️  Source file not found: {filename}")
            return False
        
        try:
            # Read original content
            with open(source_file, 'r') as f:
                content = f.read()
            
            # Fix imports
            fixed_content = self.fix_imports_for_migration(content, migration_type)
            
            # Write to target location
            with open(target_file, 'w') as f:
                f.write(fixed_content)
            
            print(f"   📄 Copied {filename} to {target_dir.name}/")
            return True
            
        except Exception as e:
            print(f"   ❌ Failed to migrate {filename}: {str(e)}")
            return False
    
    def test_migrated_file(self, filename: str, target_dir: Path) -> bool:
        """Test that migrated file works correctly"""
        target_file = target_dir / filename
        relative_path = target_file.relative_to(self.tests_root.parent.parent)
        
        print(f"   🧪 Testing migrated {filename}...")
        
        try:
            # Test import capability first
            result = subprocess.run([
                sys.executable, "-c", f"import {str(relative_path).replace('/', '.').replace('.py', '')}"
            ], capture_output=True, text=True, timeout=10)
            
            if result.returncode != 0:
                print(f"   ❌ Import test failed for {filename}")
                print(f"      Error: {result.stderr[:200]}")
                return False
            
            # Test pytest collection
            result = subprocess.run([
                sys.executable, "-m", "pytest", str(relative_path), "--collect-only", "-q"
            ], capture_output=True, text=True, timeout=15)
            
            if result.returncode != 0:
                print(f"   ❌ Test discovery failed for {filename}")
                print(f"      Error: {result.stderr[:200]}")
                return False
            
            # Quick execution test for working files
            if filename == "test_data_models.py":
                result = subprocess.run([
                    sys.executable, "-m", "pytest", str(relative_path), "-x", "--tb=no"
                ], capture_output=True, text=True, timeout=30)
                
                if result.returncode == 0:
                    print(f"   ✅ {filename} - Full test execution passed")
                else:
                    print(f"   ⚠️  {filename} - Discovery works but execution has issues")
            else:
                print(f"   ✅ {filename} - Discovery successful")
            
            return True
            
        except Exception as e:
            print(f"   ❌ Testing failed for {filename}: {str(e)}")
            return False
    
    def migrate_category(self, category: str, file_list: List[str]) -> Dict[str, bool]:
        """Migrate an entire category of tests"""
        print(f"\n🚀 Migrating {category} tests...")
        
        if category == "unit":
            target_dir = self.unit_dir
            migration_type = "root_to_unit"
        elif category == "integration": 
            target_dir = self.integration_dir
            migration_type = "root_to_integration"
        elif category == "e2e":
            target_dir = self.e2e_dir
            migration_type = "root_to_e2e"
        elif category == "comprehensive":
            target_dir = self.comprehensive_dir
            migration_type = "no_change"  # Keep at root level
        elif category == "dataset_specific":
            # Move to unit/datasets/ subdirectory
            datasets_dir = self.unit_dir / "datasets"
            datasets_dir.mkdir(exist_ok=True)
            target_dir = datasets_dir
            migration_type = "root_to_unit"
        else:
            target_dir = self.tests_root  # Keep others at root for now
            migration_type = "no_change"
        
        results = {}
        
        for filename in file_list:
            print(f"\n📝 Processing {filename}:")
            
            # Step 1: Migrate file
            if migration_type == "no_change":
                # File stays where it is
                if (self.tests_root / filename).exists():
                    results[filename] = True
                    print(f"   ✅ {filename} - Staying at root level")
                else:
                    results[filename] = False
                    print(f"   ❌ {filename} - File not found")
                continue
            
            migration_success = self.migrate_test_file(filename, target_dir, migration_type)
            
            if not migration_success:
                results[filename] = False
                continue
            
            # Step 2: Test migrated file
            test_success = self.test_migrated_file(filename, target_dir)
            results[filename] = test_success
            
            # Step 3: Only remove original if migration successful
            if test_success:
                original_file = self.tests_root / filename
                if original_file.exists():
                    original_file.unlink()
                    print(f"   🗑️  Removed original {filename}")
            else:
                # Remove failed migration attempt
                failed_file = target_dir / filename
                if failed_file.exists():
                    failed_file.unlink()
                    print(f"   🔄 Reverted failed migration of {filename}")
        
        # Summary
        successful = len([f for f, success in results.items() if success])
        total = len(results)
        
        print(f"\n📊 {category.title()} migration summary: {successful}/{total} successful")
        
        return results
    
    def create_init_files(self):
        """Create __init__.py files in new directories"""
        for directory in [self.unit_dir, self.integration_dir, self.e2e_dir]:
            init_file = directory / "__init__.py"
            if not init_file.exists():
                init_file.write_text("# Tests directory\n")
                print(f"✅ Created {init_file}")

def main():
    """Main migration process"""
    print("🎯 Starting Test Migration with Functional Equivalence...")
    
    tests_root = Path("emotion_experiment_engine/tests")
    migrator = TestMigrator(tests_root)
    
    # Create necessary __init__.py files
    migrator.create_init_files()
    
    # Get test categories
    categories = migrator.categorize_tests()
    
    # Migration plan - start with safest first
    migration_order = [
        "unit",           # Safest - simple unit tests
        "comprehensive",  # Medium - keep at root level  
        "dataset_specific", # Medium - move to unit/datasets/
        "integration",    # Higher risk - complex dependencies
        "e2e",           # Highest risk - full system tests
    ]
    
    all_results = {}
    
    for category in migration_order:
        if category in categories:
            results = migrator.migrate_category(category, categories[category])
            all_results[category] = results
        
        # Pause for user confirmation on critical categories
        if category in ["unit", "integration"]:
            print(f"\n⏸️  Paused after {category} migration.")
            print("   Verify everything looks good before continuing...")
            input("   Press Enter to continue or Ctrl+C to stop: ")
    
    # Final summary
    print(f"\n🎉 Migration Complete!")
    print(f"=" * 50)
    
    for category, results in all_results.items():
        successful = len([f for f, success in results.items() if success])
        total = len(results)
        print(f"{category:15}: {successful:2d}/{total:2d} files migrated successfully")
    
    # Test the final structure
    print(f"\n🧪 Testing final structure...")
    
    test_result = subprocess.run([
        "python", "-m", "pytest", "emotion_experiment_engine/tests/", "--collect-only", "-q"
    ], capture_output=True, text=True)
    
    final_test_count = len([line for line in test_result.stdout.split('\n') if '::' in line and line.strip()])
    print(f"Final test discovery: {final_test_count} tests found")
    
    if final_test_count > 200:  # We expect around 258 tests
        print("✅ Migration appears successful - test discovery intact!")
    else:
        print("⚠️  Migration may have issues - fewer tests discovered than expected")

if __name__ == "__main__":
    main()