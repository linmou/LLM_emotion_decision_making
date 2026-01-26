#!/usr/bin/env python3
"""
Quick Baseline Capture - Fast test functionality assessment
"""

import subprocess
import json
import time
from pathlib import Path
from typing import Dict, List, Any

def capture_quick_baseline():
    """Capture baseline with efficient methods"""
    print("📊 Quick Baseline Capture...")
    
    # 1. Test Discovery
    print("🔍 Discovering tests...")
    discovery_result = subprocess.run([
        "python", "-m", "pytest", "emotion_experiment_engine/tests/", 
        "--collect-only", "-q"
    ], capture_output=True, text=True)
    
    total_tests = len([line for line in discovery_result.stdout.split('\n') if '::' in line and line.strip()])
    print(f"   Found: {total_tests} tests")
    
    # 2. Fast execution test - just unit tests
    print("🧪 Testing unit test files...")
    unit_test_files = [
        "test_data_models.py",
        "test_dataset_factory.py", 
        "test_config_loader.py",
        "test_utils.py"
    ]
    
    working_files = []
    broken_files = []
    
    for test_file in unit_test_files:
        test_path = f"emotion_experiment_engine/tests/{test_file}"
        if Path(test_path).exists():
            print(f"   Testing {test_file}...")
            result = subprocess.run([
                "python", "-m", "pytest", test_path, "-v", "--tb=no", "-x"
            ], capture_output=True, text=True, timeout=30)
            
            if result.returncode == 0:
                working_files.append(test_file)
                print(f"     ✅ {test_file}")
            else:
                broken_files.append(test_file)
                print(f"     ❌ {test_file}")
    
    # 3. Quick integration test sample
    print("🔗 Testing integration files...")
    integration_files = [
        "test_integration.py",
        "test_end_to_end_integration.py"
    ]
    
    for test_file in integration_files:
        test_path = f"emotion_experiment_engine/tests/{test_file}"
        if Path(test_path).exists():
            print(f"   Testing {test_file}...")
            result = subprocess.run([
                "python", "-m", "pytest", test_path, "--collect-only"
            ], capture_output=True, text=True, timeout=10)
            
            if result.returncode == 0:
                working_files.append(test_file)
                print(f"     ✅ {test_file} (discoverable)")
            else:
                broken_files.append(test_file)
                print(f"     ❌ {test_file} (discovery failed)")
    
    # 4. Comprehensive test files
    comprehensive_files = [
        "test_answer_wrapper_comprehensive.py",
        "test_asyncio_evaluate_batch_comprehensive.py",
        "test_emotion_evaluation_comprehensive.py"
    ]
    
    print("📋 Testing comprehensive files...")
    for test_file in comprehensive_files:
        test_path = f"emotion_experiment_engine/tests/{test_file}"
        if Path(test_path).exists():
            print(f"   Testing {test_file}...")
            result = subprocess.run([
                "python", "-m", "pytest", test_path, "--collect-only"
            ], capture_output=True, text=True, timeout=10)
            
            if result.returncode == 0:
                working_files.append(test_file)
                print(f"     ✅ {test_file} (discoverable)")
            else:
                broken_files.append(test_file)
                print(f"     ❌ {test_file} (discovery failed)")
    
    baseline = {
        "timestamp": time.strftime("%Y-%m-%d_%H-%M-%S"),
        "total_discovered_tests": total_tests,
        "sampled_files": len(unit_test_files + integration_files + comprehensive_files),
        "working_files": working_files,
        "broken_files": broken_files,
        "summary": {
            "unit_tests_working": len([f for f in working_files if f in unit_test_files]),
            "integration_tests_working": len([f for f in working_files if f in integration_files]),
            "comprehensive_tests_working": len([f for f in working_files if f in comprehensive_files])
        }
    }
    
    # Save baseline
    baseline_file = Path("emotion_experiment_engine/tests/migration_baselines/quick_baseline.json")
    baseline_file.parent.mkdir(exist_ok=True)
    
    with open(baseline_file, 'w') as f:
        json.dump(baseline, f, indent=2)
    
    print(f"\n📋 Quick Baseline Summary:")
    print(f"   Total discoverable tests: {baseline['total_discovered_tests']}")
    print(f"   Sampled files: {baseline['sampled_files']}")
    print(f"   Working files: {len(baseline['working_files'])}")
    print(f"   Broken files: {len(baseline['broken_files'])}")
    print(f"   Unit tests working: {baseline['summary']['unit_tests_working']}/{len(unit_test_files)}")
    print(f"   Integration tests working: {baseline['summary']['integration_tests_working']}/{len(integration_files)}")
    print(f"   Comprehensive tests working: {baseline['summary']['comprehensive_tests_working']}/{len(comprehensive_files)}")
    print(f"   Saved to: {baseline_file}")
    
    return baseline

if __name__ == "__main__":
    capture_quick_baseline()