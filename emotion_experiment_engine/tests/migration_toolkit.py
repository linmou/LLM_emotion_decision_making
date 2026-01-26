#!/usr/bin/env python3
"""
Test Migration Toolkit - Functional Equivalence Preservation

This toolkit ensures zero functionality loss during test structure migration.
Captures baseline functionality and validates equivalence after each migration step.
"""

import json
import subprocess
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import difflib
import re

@dataclass
class TestResult:
    """Capture complete test execution result"""
    name: str
    status: str  # PASSED, FAILED, SKIPPED, ERROR
    duration: float
    error_message: Optional[str] = None
    output: str = ""

@dataclass
class TestFile:
    """Capture test file metadata and results"""
    file_path: str
    test_count: int
    tests: List[TestResult]
    import_errors: List[str]
    total_duration: float

@dataclass
class BaselineCapture:
    """Complete baseline of test functionality"""
    timestamp: str
    total_files: int
    total_tests: int
    working_files: List[str]
    broken_files: List[str]
    test_files: Dict[str, TestFile]

class MigrationToolkit:
    """Toolkit for safe test migration with functional equivalence"""
    
    def __init__(self, tests_root: Path):
        self.tests_root = tests_root
        self.baseline_dir = tests_root / "migration_baselines"
        self.baseline_dir.mkdir(exist_ok=True)
        
    def discover_test_files(self) -> List[Path]:
        """Discover all test files in the tests directory"""
        test_files = []
        for pattern in ["test_*.py"]:
            test_files.extend(self.tests_root.glob(pattern))
        return sorted(test_files)
    
    def analyze_imports(self, test_file: Path) -> List[str]:
        """Analyze import statements in a test file"""
        imports = []
        try:
            with open(test_file, 'r') as f:
                content = f.read()
            
            # Find all import statements
            import_patterns = [
                r'from\s+\.\.[\w.]+\s+import',  # Relative imports from parent
                r'from\s+\.[\w.]+\s+import',   # Relative imports from current
                r'import\s+[\w.]+',            # Absolute imports
            ]
            
            for pattern in import_patterns:
                matches = re.findall(pattern, content)
                imports.extend(matches)
                
        except Exception as e:
            imports.append(f"ERROR reading file: {str(e)}")
            
        return imports
    
    def run_single_test_file(self, test_file: Path) -> TestFile:
        """Run a single test file and capture all results"""
        print(f"🧪 Testing {test_file.name}...")
        
        start_time = time.time()
        
        # Run pytest on single file
        cmd = [
            sys.executable, "-m", "pytest", 
            str(test_file), 
            "-v", "--tb=short", "--no-header",
            "--disable-warnings"
        ]
        
        try:
            result = subprocess.run(
                cmd, 
                capture_output=True, 
                text=True, 
                timeout=120,  # 2 minute timeout per file
                cwd=self.tests_root.parent.parent
            )
            
            duration = time.time() - start_time
            
            # Parse pytest output
            tests = self.parse_pytest_output(result.stdout, result.stderr)
            import_errors = self.extract_import_errors(result.stderr)
            
            return TestFile(
                file_path=str(test_file.relative_to(self.tests_root)),
                test_count=len(tests),
                tests=tests,
                import_errors=import_errors,
                total_duration=duration
            )
            
        except subprocess.TimeoutExpired:
            return TestFile(
                file_path=str(test_file.relative_to(self.tests_root)),
                test_count=0,
                tests=[],
                import_errors=[f"TIMEOUT: Test file took longer than 120 seconds"],
                total_duration=120.0
            )
        except Exception as e:
            return TestFile(
                file_path=str(test_file.relative_to(self.tests_root)),
                test_count=0,
                tests=[],
                import_errors=[f"EXECUTION ERROR: {str(e)}"],
                total_duration=0.0
            )
    
    def parse_pytest_output(self, stdout: str, stderr: str) -> List[TestResult]:
        """Parse pytest output to extract individual test results"""
        tests = []
        
        # Parse test results from pytest verbose output
        lines = stdout.split('\n')
        
        for line in lines:
            if '::' in line and any(status in line for status in ['PASSED', 'FAILED', 'SKIPPED', 'ERROR']):
                # Example: test_file.py::TestClass::test_method PASSED [8%]
                parts = line.split(' ')
                if len(parts) >= 2:
                    test_name = parts[0].split('::')[-1]  # Get method name
                    status = parts[1]
                    
                    # Extract duration if present
                    duration = 0.0
                    duration_match = re.search(r'(\d+\.?\d*)s', line)
                    if duration_match:
                        duration = float(duration_match.group(1))
                    
                    error_message = None
                    if status in ['FAILED', 'ERROR']:
                        error_message = self.extract_error_from_output(stdout, test_name)
                    
                    tests.append(TestResult(
                        name=test_name,
                        status=status,
                        duration=duration,
                        error_message=error_message,
                        output=line
                    ))
        
        return tests
    
    def extract_import_errors(self, stderr: str) -> List[str]:
        """Extract import errors from stderr"""
        import_errors = []
        
        if "ImportError" in stderr:
            import_errors.append("ImportError detected")
        if "ModuleNotFoundError" in stderr:
            import_errors.append("ModuleNotFoundError detected")
        if "SyntaxError" in stderr:
            import_errors.append("SyntaxError detected")
            
        return import_errors
    
    def extract_error_from_output(self, output: str, test_name: str) -> Optional[str]:
        """Extract error message for a specific test"""
        # Simple error extraction - could be enhanced
        lines = output.split('\n')
        for i, line in enumerate(lines):
            if test_name in line and ('FAILED' in line or 'ERROR' in line):
                # Look for error details in following lines
                for j in range(i+1, min(i+10, len(lines))):
                    if lines[j].strip().startswith('E '):
                        return lines[j].strip()
        return None
    
    def capture_baseline(self) -> BaselineCapture:
        """Capture complete baseline of current test functionality"""
        print("📊 Capturing baseline test functionality...")
        
        test_files = self.discover_test_files()
        print(f"Found {len(test_files)} test files to analyze")
        
        test_file_results = {}
        working_files = []
        broken_files = []
        total_tests = 0
        
        for test_file in test_files:
            try:
                file_result = self.run_single_test_file(test_file)
                test_file_results[file_result.file_path] = file_result
                
                if file_result.import_errors or any(t.status in ['FAILED', 'ERROR'] for t in file_result.tests):
                    broken_files.append(file_result.file_path)
                    print(f"⚠️  {test_file.name}: {len(file_result.import_errors)} import errors, {len([t for t in file_result.tests if t.status in ['FAILED', 'ERROR']])} test failures")
                else:
                    working_files.append(file_result.file_path)
                    print(f"✅ {test_file.name}: {file_result.test_count} tests passed")
                
                total_tests += file_result.test_count
                
            except Exception as e:
                print(f"❌ Failed to analyze {test_file.name}: {str(e)}")
                broken_files.append(str(test_file.relative_to(self.tests_root)))
        
        baseline = BaselineCapture(
            timestamp=time.strftime("%Y-%m-%d_%H-%M-%S"),
            total_files=len(test_files),
            total_tests=total_tests,
            working_files=working_files,
            broken_files=broken_files,
            test_files=test_file_results
        )
        
        # Save baseline
        baseline_file = self.baseline_dir / f"baseline_{baseline.timestamp}.json"
        with open(baseline_file, 'w') as f:
            json.dump(asdict(baseline), f, indent=2)
        
        print(f"\n📋 Baseline Summary:")
        print(f"   Total files: {baseline.total_files}")
        print(f"   Total tests: {baseline.total_tests}")
        print(f"   Working files: {len(baseline.working_files)}")
        print(f"   Broken files: {len(baseline.broken_files)}")
        print(f"   Saved to: {baseline_file}")
        
        return baseline
    
    def compare_baselines(self, baseline1: BaselineCapture, baseline2: BaselineCapture) -> Dict[str, Any]:
        """Compare two baselines for functional equivalence"""
        comparison = {
            "equivalent": True,
            "differences": [],
            "summary": {}
        }
        
        # Compare basic metrics
        if baseline1.total_tests != baseline2.total_tests:
            comparison["equivalent"] = False
            comparison["differences"].append(f"Test count changed: {baseline1.total_tests} -> {baseline2.total_tests}")
        
        if len(baseline1.working_files) != len(baseline2.working_files):
            comparison["equivalent"] = False
            comparison["differences"].append(f"Working files changed: {len(baseline1.working_files)} -> {len(baseline2.working_files)}")
        
        # Compare file-by-file results
        for file_path in baseline1.test_files:
            if file_path not in baseline2.test_files:
                comparison["equivalent"] = False
                comparison["differences"].append(f"File missing in new baseline: {file_path}")
                continue
            
            file1 = baseline1.test_files[file_path]
            file2 = baseline2.test_files[file_path]
            
            # Compare test results
            if file1.test_count != file2.test_count:
                comparison["equivalent"] = False
                comparison["differences"].append(f"{file_path}: Test count changed {file1.test_count} -> {file2.test_count}")
            
            # Compare individual test outcomes
            test1_results = {t.name: t.status for t in file1.tests}
            test2_results = {t.name: t.status for t in file2.tests}
            
            for test_name in test1_results:
                if test_name not in test2_results:
                    comparison["equivalent"] = False
                    comparison["differences"].append(f"{file_path}::{test_name}: Test disappeared")
                elif test1_results[test_name] != test2_results[test_name]:
                    comparison["equivalent"] = False
                    comparison["differences"].append(f"{file_path}::{test_name}: Status changed {test1_results[test_name]} -> {test2_results[test_name]}")
        
        comparison["summary"] = {
            "baseline1": {
                "files": baseline1.total_files,
                "tests": baseline1.total_tests, 
                "working": len(baseline1.working_files)
            },
            "baseline2": {
                "files": baseline2.total_files,
                "tests": baseline2.total_tests,
                "working": len(baseline2.working_files)
            }
        }
        
        return comparison

    def get_latest_baseline(self) -> Optional[BaselineCapture]:
        """Get the most recent baseline"""
        baseline_files = list(self.baseline_dir.glob("baseline_*.json"))
        if not baseline_files:
            return None
        
        latest_file = max(baseline_files, key=lambda f: f.stat().st_mtime)
        
        with open(latest_file, 'r') as f:
            data = json.load(f)
        
        # Convert back to dataclass
        baseline = BaselineCapture(
            timestamp=data['timestamp'],
            total_files=data['total_files'],
            total_tests=data['total_tests'],
            working_files=data['working_files'],
            broken_files=data['broken_files'],
            test_files={
                path: TestFile(
                    file_path=file_data['file_path'],
                    test_count=file_data['test_count'],
                    tests=[
                        TestResult(
                            name=t['name'],
                            status=t['status'],
                            duration=t['duration'],
                            error_message=t.get('error_message'),
                            output=t.get('output', '')
                        ) for t in file_data['tests']
                    ],
                    import_errors=file_data['import_errors'],
                    total_duration=file_data['total_duration']
                )
                for path, file_data in data['test_files'].items()
            }
        )
        
        return baseline

def main():
    """Command line interface for migration toolkit"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Test Migration Toolkit")
    parser.add_argument("command", choices=["capture", "compare"], help="Command to run")
    parser.add_argument("--tests-dir", type=Path, default=Path("emotion_experiment_engine/tests"), help="Tests directory")
    
    args = parser.parse_args()
    
    toolkit = MigrationToolkit(args.tests_dir)
    
    if args.command == "capture":
        baseline = toolkit.capture_baseline()
        print(f"\n✅ Baseline captured successfully!")
        print(f"Working files ({len(baseline.working_files)}):")
        for f in baseline.working_files:
            print(f"  ✅ {f}")
        if baseline.broken_files:
            print(f"\nBroken files ({len(baseline.broken_files)}):")
            for f in baseline.broken_files:
                print(f"  ❌ {f}")
    
    elif args.command == "compare":
        baseline = toolkit.get_latest_baseline()
        if not baseline:
            print("No baseline found. Run 'capture' first.")
            return
        
        print("Capturing new baseline for comparison...")
        new_baseline = toolkit.capture_baseline()
        
        comparison = toolkit.compare_baselines(baseline, new_baseline)
        
        print(f"\n🔍 Functional Equivalence Analysis:")
        print(f"Equivalent: {'✅ YES' if comparison['equivalent'] else '❌ NO'}")
        
        if comparison["differences"]:
            print(f"\nDifferences found:")
            for diff in comparison["differences"]:
                print(f"  - {diff}")
        
        print(f"\nSummary:")
        print(f"  Original: {comparison['summary']['baseline1']['files']} files, {comparison['summary']['baseline1']['tests']} tests, {comparison['summary']['baseline1']['working']} working")
        print(f"  Current:  {comparison['summary']['baseline2']['files']} files, {comparison['summary']['baseline2']['tests']} tests, {comparison['summary']['baseline2']['working']} working")

if __name__ == "__main__":
    main()