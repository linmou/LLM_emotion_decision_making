#!/usr/bin/env python3
"""
Intelligent Test Suite Orchestration System

Provides smart test execution with dependency awareness, parallel execution,
and adaptive test selection based on code changes.
"""

import argparse
import json
import multiprocessing
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import pytest
import yaml


class TestDependencyGraph:
    """Manages test dependencies and smart execution ordering"""
    
    # Component dependency mapping based on architecture analysis
    TEST_DEPENDENCIES = {
        # Core components (no dependencies)
        "data_models": [],
        "base_dataset": ["data_models"],
        
        # Specialized datasets (depend on base)
        "infinitebench_dataset": ["base_dataset", "evaluation_utils"],
        "longbench_dataset": ["base_dataset", "evaluation_utils"],
        "locomo_dataset": ["base_dataset", "evaluation_utils"],
        "emotion_check_dataset": ["base_dataset", "evaluation_utils"],
        
        # Factory pattern (depends on all datasets)
        "dataset_factory": ["base_dataset", "infinitebench_dataset", "longbench_dataset", 
                           "locomo_dataset", "emotion_check_dataset"],
        
        # High-level orchestration
        "experiment": ["dataset_factory", "config_loader", "memory_prompt_wrapper"],
        "config_loader": ["data_models"],
        "evaluation_utils": [],
        "memory_prompt_wrapper": [],
        
        # Integration tests depend on everything
        "integration": ["experiment", "dataset_factory", "config_loader"],
        "e2e": ["integration"]
    }
    
    # File to component mapping
    COMPONENT_FILES = {
        "data_models": ["data_models.py"],
        "base_dataset": ["datasets/base.py"],
        "infinitebench_dataset": ["datasets/infinitebench.py"],
        "longbench_dataset": ["datasets/longbench.py"],
        "locomo_dataset": ["datasets/locomo.py"],
        "emotion_check_dataset": ["datasets/emotion_check.py"],
        "dataset_factory": ["dataset_factory.py"],
        "experiment": ["experiment.py"],
        "config_loader": ["config_loader.py"],
        "evaluation_utils": ["evaluation_utils.py"],
        "memory_prompt_wrapper": ["memory_prompt_wrapper.py"],
    }
    
    def get_affected_components(self, changed_files: List[str]) -> Set[str]:
        """Determine which components are affected by file changes"""
        affected = set()
        
        for file in changed_files:
            file = file.replace("emotion_experiment_engine/", "")
            
            # Direct mapping
            for component, files in self.COMPONENT_FILES.items():
                if any(file.endswith(f) for f in files):
                    affected.add(component)
        
        return affected
    
    def get_dependent_components(self, components: Set[str]) -> Set[str]:
        """Get all components that depend on the given components"""
        dependents = set(components)
        
        # Keep expanding until no new dependencies found
        while True:
            new_dependents = set()
            for component, deps in self.TEST_DEPENDENCIES.items():
                if any(dep in dependents for dep in deps) and component not in dependents:
                    new_dependents.add(component)
            
            if not new_dependents:
                break
            dependents.update(new_dependents)
        
        return dependents
    
    def get_execution_order(self, components: Set[str]) -> List[str]:
        """Get optimal execution order respecting dependencies"""
        ordered = []
        remaining = set(components)
        
        while remaining:
            # Find components with no unresolved dependencies
            ready = []
            for component in remaining:
                deps = self.TEST_DEPENDENCIES.get(component, [])
                if all(dep in ordered or dep not in components for dep in deps):
                    ready.append(component)
            
            if not ready:
                # Circular dependency or missing component
                ready = list(remaining)  # Just add remaining to avoid infinite loop
            
            # Sort alphabetically for deterministic order
            ready.sort()
            ordered.extend(ready)
            remaining -= set(ready)
        
        return ordered


class SmartTestRunner:
    """AI-powered test selection and execution"""
    
    TEST_SUITES = {
        "smoke": {
            "description": "Essential tests that must pass before any development",
            "tests": [
                "priorities/research_critical.py",
                "unit/test_data_models.py"
            ],
            "max_time": 60,  # 1 minute
            "parallel": False
        },
        
        "comprehensive": {
            "description": "Comprehensive component test suites",
            "tests": [
                "test_answer_wrapper_comprehensive.py"
            ],
            "max_time": 180,  # 3 minutes
            "parallel": False
        },
        
        "unit": {
            "description": "All unit tests",
            "pattern": "unit/",
            "max_time": 300,  # 5 minutes
            "parallel": True
        },
        
        "integration": {
            "description": "Component integration tests",
            "pattern": "integration/",
            "max_time": 600,  # 10 minutes
            "parallel": True
        },
        
        "regression": {
            "description": "Regression prevention tests",
            "pattern": "regression/",
            "max_time": 1800,  # 30 minutes
            "parallel": True
        },
        
        "e2e": {
            "description": "End-to-end workflow tests",
            "pattern": "e2e/",
            "max_time": 3600,  # 1 hour
            "parallel": False
        },
        
        "critical": {
            "description": "Research-critical tests only",
            "markers": ["critical"],
            "max_time": 300,
            "parallel": False
        },
        
        "fast": {
            "description": "Tests that run in under 5 seconds",
            "markers": ["-slow"],
            "max_time": 180,
            "parallel": True
        },
        
        "pre_commit": {
            "description": "Pre-commit validation suite",
            "tests": [
                "priorities/research_critical.py",
                "unit/test_data_models.py",
                "unit/test_dataset_factory.py",
                "integration/test_config_loading.py"
            ],
            "max_time": 300,
            "parallel": True
        },
        
        "nightly": {
            "description": "Comprehensive nightly test suite",
            "patterns": ["regression/", "e2e/", "performance/"],
            "max_time": 7200,  # 2 hours
            "parallel": True
        },
        
        "changed": {
            "description": "Tests for changed components (requires git)",
            "dynamic": True,
            "max_time": 900,  # 15 minutes
            "parallel": True
        }
    }
    
    def __init__(self, test_dir: Path = None):
        self.test_dir = test_dir or Path(__file__).parent.parent
        self.dependency_graph = TestDependencyGraph()
        self.performance_data = self._load_performance_data()
    
    def _load_performance_data(self) -> Dict:
        """Load historical performance data"""
        perf_file = self.test_dir / "performance_data.json"
        if perf_file.exists():
            with open(perf_file) as f:
                return json.load(f)
        return {}
    
    def _save_performance_data(self, data: Dict):
        """Save performance data for future optimization"""
        perf_file = self.test_dir / "performance_data.json"
        with open(perf_file, 'w') as f:
            json.dump(data, f, indent=2)
    
    def get_changed_files(self) -> List[str]:
        """Get changed files from git"""
        try:
            result = subprocess.run(
                ["git", "diff", "--name-only", "HEAD~1"],
                capture_output=True,
                text=True,
                cwd=self.test_dir.parent.parent
            )
            if result.returncode == 0:
                return [f for f in result.stdout.strip().split('\n') if f]
            return []
        except Exception:
            return []
    
    def select_tests_for_changes(self, changed_files: List[str] = None) -> List[str]:
        """Select minimal test set covering changes"""
        if changed_files is None:
            changed_files = self.get_changed_files()
        
        if not changed_files:
            return ["priorities/research_critical.py", "unit/test_data_models.py"]
        
        # Determine affected components
        affected = self.dependency_graph.get_affected_components(changed_files)
        all_affected = self.dependency_graph.get_dependent_components(affected)
        
        # Map components to test files
        test_files = []
        
        # Always include critical tests
        test_files.append("priorities/research_critical.py")
        
        # Add component-specific tests
        for component in all_affected:
            if component in ["integration", "e2e"]:
                test_files.append(f"{component}/")
            else:
                test_files.append(f"unit/test_{component}.py")
        
        # Add integration tests if multiple components affected
        if len(affected) > 1:
            test_files.append("integration/")
        
        # Add regression tests for critical components
        critical_components = {"evaluation_utils", "dataset_factory", "base_dataset", "experiment"}
        if affected & critical_components:
            test_files.append("regression/")
        
        return list(set(test_files))  # Remove duplicates
    
    def build_pytest_args(self, suite_config: Dict, extra_args: List[str] = None) -> List[str]:
        """Build pytest command arguments"""
        args = ["pytest"]
        
        # Add test paths/patterns
        if "tests" in suite_config:
            for test in suite_config["tests"]:
                test_path = self.test_dir / test
                if test_path.exists():
                    args.append(str(test_path))
        
        if "pattern" in suite_config:
            pattern_path = self.test_dir / suite_config["pattern"]
            if pattern_path.exists():
                args.append(str(pattern_path))
        
        if "patterns" in suite_config:
            for pattern in suite_config["patterns"]:
                pattern_path = self.test_dir / pattern
                if pattern_path.exists():
                    args.append(str(pattern_path))
        
        # Add markers
        if "markers" in suite_config:
            for marker in suite_config["markers"]:
                if marker.startswith("-"):
                    args.extend(["-m", f"not {marker[1:]}"])
                else:
                    args.extend(["-m", marker])
        
        # Add performance options
        if suite_config.get("parallel", False):
            cpu_count = multiprocessing.cpu_count()
            args.extend(["-n", str(min(4, cpu_count))])  # Limit parallel workers
        
        # Timeout
        if "max_time" in suite_config:
            args.extend(["--timeout", str(suite_config["max_time"])])
        
        # Extra arguments
        if extra_args:
            args.extend(extra_args)
        
        return args
    
    def run_suite(self, suite_name: str, extra_args: List[str] = None, 
                  verbose: bool = True) -> Tuple[bool, Dict]:
        """Execute test suite with intelligent selection"""
        
        if suite_name not in self.TEST_SUITES:
            available = ", ".join(self.TEST_SUITES.keys())
            raise ValueError(f"Unknown test suite '{suite_name}'. Available: {available}")
        
        suite_config = self.TEST_SUITES[suite_name].copy()
        
        # Handle dynamic test selection
        if suite_config.get("dynamic"):
            if suite_name == "changed":
                test_files = self.select_tests_for_changes()
                suite_config["tests"] = test_files
        
        if verbose:
            print(f"🧪 Running Test Suite: {suite_name}")
            print(f"📝 Description: {suite_config['description']}")
            if "max_time" in suite_config:
                print(f"⏱️  Max Time: {suite_config['max_time']}s")
            print("=" * 60)
        
        # Build pytest command
        pytest_args = self.build_pytest_args(suite_config, extra_args)
        
        if verbose:
            print(f"🔧 Command: {' '.join(pytest_args)}")
            print()
        
        # Execute tests
        start_time = time.time()
        try:
            result = subprocess.run(
                pytest_args,
                cwd=self.test_dir.parent,
                timeout=suite_config.get("max_time")
            )
            
            duration = time.time() - start_time
            success = result.returncode == 0
            
            # Record performance data
            perf_data = {
                "suite": suite_name,
                "duration": duration,
                "success": success,
                "timestamp": time.time()
            }
            
            # Update performance history
            if suite_name not in self.performance_data:
                self.performance_data[suite_name] = []
            self.performance_data[suite_name].append(perf_data)
            
            # Keep only last 50 runs
            self.performance_data[suite_name] = self.performance_data[suite_name][-50:]
            self._save_performance_data(self.performance_data)
            
            if verbose:
                print("\n" + "=" * 60)
                if success:
                    print(f"✅ Suite '{suite_name}' PASSED in {duration:.2f}s")
                else:
                    print(f"❌ Suite '{suite_name}' FAILED in {duration:.2f}s")
                    print(f"Exit code: {result.returncode}")
            
            return success, perf_data
            
        except subprocess.TimeoutExpired:
            duration = time.time() - start_time
            if verbose:
                print(f"\n⏰ Suite '{suite_name}' TIMEOUT after {duration:.2f}s")
            return False, {"suite": suite_name, "duration": duration, "timeout": True}
        
        except Exception as e:
            duration = time.time() - start_time
            if verbose:
                print(f"\n💥 Suite '{suite_name}' ERROR: {str(e)}")
            return False, {"suite": suite_name, "duration": duration, "error": str(e)}
    
    def run_multiple_suites(self, suite_names: List[str], 
                           fail_fast: bool = True) -> Dict[str, Tuple[bool, Dict]]:
        """Run multiple test suites in optimal order"""
        results = {}
        
        # Optimize suite order (critical tests first)
        priority_order = ["smoke", "critical", "unit", "integration", "regression", "e2e"]
        ordered_suites = []
        
        for priority_suite in priority_order:
            if priority_suite in suite_names:
                ordered_suites.append(priority_suite)
                suite_names.remove(priority_suite)
        
        # Add remaining suites
        ordered_suites.extend(sorted(suite_names))
        
        print(f"🎯 Executing {len(ordered_suites)} test suites in order:")
        for i, suite in enumerate(ordered_suites, 1):
            print(f"  {i}. {suite} - {self.TEST_SUITES[suite]['description']}")
        print()
        
        total_start = time.time()
        
        for suite_name in ordered_suites:
            success, perf_data = self.run_suite(suite_name)
            results[suite_name] = (success, perf_data)
            
            if not success and fail_fast:
                print(f"\n🛑 Stopping execution due to failure in '{suite_name}'")
                break
        
        total_duration = time.time() - total_start
        
        # Summary
        print("\n" + "=" * 80)
        print("📊 EXECUTION SUMMARY")
        print("=" * 80)
        
        passed = sum(1 for success, _ in results.values() if success)
        total = len(results)
        
        print(f"Total Suites: {total}")
        print(f"Passed: {passed}")
        print(f"Failed: {total - passed}")
        print(f"Success Rate: {(passed/total)*100:.1f}%")
        print(f"Total Time: {total_duration:.2f}s")
        
        for suite_name, (success, perf_data) in results.items():
            status = "✅ PASS" if success else "❌ FAIL"
            duration = perf_data.get("duration", 0)
            print(f"  {suite_name:20s}: {status} ({duration:.2f}s)")
        
        return results


def main():
    """Command-line interface for test runner"""
    parser = argparse.ArgumentParser(description="Intelligent Test Suite Runner")
    
    parser.add_argument("suite", nargs="?", 
                       choices=list(SmartTestRunner.TEST_SUITES.keys()) + ["all"],
                       help="Test suite to run")
    
    parser.add_argument("--list", action="store_true",
                       help="List available test suites")
    
    parser.add_argument("--multiple", nargs="+", 
                       choices=list(SmartTestRunner.TEST_SUITES.keys()),
                       help="Run multiple test suites")
    
    parser.add_argument("--fail-fast", action="store_true", default=True,
                       help="Stop on first failure")
    
    parser.add_argument("--no-fail-fast", dest="fail_fast", action="store_false",
                       help="Continue on failures")
    
    parser.add_argument("--verbose", "-v", action="store_true", default=True,
                       help="Verbose output")
    
    parser.add_argument("--quiet", "-q", dest="verbose", action="store_false",
                       help="Quiet output")
    
    parser.add_argument("--pytest-args", nargs="*",
                       help="Additional arguments to pass to pytest")
    
    args = parser.parse_args()
    
    runner = SmartTestRunner()
    
    if args.list:
        print("📋 Available Test Suites:")
        print("=" * 50)
        for name, config in runner.TEST_SUITES.items():
            print(f"{name:15s}: {config['description']}")
        return 0
    
    if not args.suite and not args.multiple:
        parser.print_help()
        return 1
    
    try:
        if args.multiple:
            results = runner.run_multiple_suites(args.multiple, args.fail_fast)
            success = all(success for success, _ in results.values())
        elif args.suite == "all":
            all_suites = list(runner.TEST_SUITES.keys())
            results = runner.run_multiple_suites(all_suites, args.fail_fast)
            success = all(success for success, _ in results.values())
        else:
            success, _ = runner.run_suite(args.suite, args.pytest_args, args.verbose)
        
        return 0 if success else 1
        
    except KeyboardInterrupt:
        print("\n🛑 Test execution interrupted by user")
        return 130
    except Exception as e:
        print(f"\n💥 Error: {str(e)}")
        return 1


if __name__ == "__main__":
    sys.exit(main())