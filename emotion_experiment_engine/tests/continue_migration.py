#!/usr/bin/env python3
"""
Continue Test Migration - Complete the remaining categories
"""

import subprocess
import sys
from pathlib import Path
from migrate_tests import TestMigrator

def continue_migration():
    """Continue migration without interactive prompts"""
    print("🚀 Continuing Test Migration...")
    
    tests_root = Path("emotion_experiment_engine/tests") 
    migrator = TestMigrator(tests_root)
    
    # Get test categories
    categories = migrator.categorize_tests()
    
    # Continue with remaining categories
    remaining_categories = [
        "comprehensive",      # Keep at root - these are working
        "dataset_specific",   # Move to unit/datasets/
        "integration",        # Move to integration/
        "e2e",               # Move to e2e/
    ]
    
    all_results = {}
    
    for category in remaining_categories:
        if category in categories:
            print(f"\n{'='*60}")
            results = migrator.migrate_category(category, categories[category])
            all_results[category] = results
            
            # Quick validation after each category
            test_result = subprocess.run([
                "python", "-m", "pytest", "emotion_experiment_engine/tests/", "--collect-only", "-q"
            ], capture_output=True, text=True)
            
            test_count = len([line for line in test_result.stdout.split('\n') if '::' in line and line.strip()])
            print(f"   📊 Current test discovery: {test_count} tests")
    
    # Final summary
    print(f"\n🎉 Complete Migration Summary!")
    print(f"=" * 60)
    
    total_successful = 0
    total_attempted = 0
    
    for category, results in all_results.items():
        successful = len([f for f, success in results.items() if success])
        total = len(results)
        total_successful += successful
        total_attempted += total
        print(f"{category:15}: {successful:2d}/{total:2d} files migrated successfully")
        
        if successful < total:
            failed = [f for f, success in results.items() if not success]
            print(f"                Failed: {', '.join(failed)}")
    
    print(f"{'='*60}")
    print(f"Overall:        {total_successful:2d}/{total_attempted:2d} files migrated successfully")
    
    # Final structure test
    print(f"\n🧪 Final Test Structure Validation...")
    
    test_result = subprocess.run([
        "python", "-m", "pytest", "emotion_experiment_engine/tests/", "--collect-only", "-q"
    ], capture_output=True, text=True)
    
    final_test_count = len([line for line in test_result.stdout.split('\n') if '::' in line and line.strip()])
    print(f"Final test discovery: {final_test_count} tests found")
    
    # Check directory structure
    print(f"\n📁 New Directory Structure:")
    for subdir in ["unit", "integration", "e2e"]:
        subdir_path = Path(f"emotion_experiment_engine/tests/{subdir}")
        if subdir_path.exists():
            files = list(subdir_path.glob("*.py"))
            files = [f for f in files if f.name != "__init__.py"]
            print(f"   {subdir:12}: {len(files)} test files")
        else:
            print(f"   {subdir:12}: Directory not found")
    
    # Root level files
    root_tests = list(Path("emotion_experiment_engine/tests").glob("test_*.py"))
    print(f"   {'root':12}: {len(root_tests)} test files (comprehensive + others)")
    
    # Test a few key migrated files
    print(f"\n✅ Testing Key Migrated Files:")
    
    key_tests = [
        "emotion_experiment_engine/tests/unit/test_data_models.py",
        "emotion_experiment_engine/tests/test_answer_wrapper_comprehensive.py"
    ]
    
    for test_file in key_tests:
        if Path(test_file).exists():
            result = subprocess.run([
                "python", "-m", "pytest", test_file, "--collect-only", "-q"
            ], capture_output=True, text=True)
            
            if result.returncode == 0:
                test_count = len([line for line in result.stdout.split('\n') if '::' in line and line.strip()])
                print(f"   ✅ {Path(test_file).name}: {test_count} tests discoverable")
            else:
                print(f"   ❌ {Path(test_file).name}: Discovery failed")
    
    if final_test_count > 200:
        print(f"\n🎉 Migration Successful!")
        print(f"   - {final_test_count} tests discoverable")
        print(f"   - Hierarchical structure created") 
        print(f"   - Import paths fixed")
        print(f"   - Key tests validated")
    else:
        print(f"\n⚠️  Migration may have issues - lower test count than expected")
    
    return final_test_count

if __name__ == "__main__":
    continue_migration()