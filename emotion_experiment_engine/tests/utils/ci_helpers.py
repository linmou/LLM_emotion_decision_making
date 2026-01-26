#!/usr/bin/env python3
"""
CI/CD Helper Utilities

Provides utilities for continuous integration including intelligent test selection,
failure analysis, and automated reporting.
"""

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Set, Any, Tuple
import argparse
import re

class TestImpactAnalyzer:
    """Analyzes code changes to intelligently select relevant tests"""
    
    # Map file patterns to test categories
    FILE_TO_TEST_MAPPING = {
        # Core dataset files
        r'datasets/.*\.py': ['unit/test_*dataset*.py', 'integration/test_dataset_*.py', 'regression/test_*_equivalence.py'],
        r'dataset_factory\.py': ['regression/test_api_compatibility.py', 'unit/test_dataset_factory.py'],
        
        # Evaluation system
        r'evaluation_utils\.py': ['unit/test_evaluation.py', 'regression/test_behavioral_equivalence.py', 'priorities/research_critical.py'],
        r'data_models\.py': ['unit/test_data_models.py', 'regression/test_api_compatibility.py'],
        
        # Experiment orchestration
        r'experiment\.py': ['integration/test_experiment_*.py', 'e2e/test_full_pipeline.py'],
        
        # Memory and prompt systems
        r'.*memory.*\.py': ['unit/test_memory_*.py', 'integration/test_memory_integration.py'],
        r'.*prompt.*\.py': ['unit/test_prompt_*.py', 'integration/test_prompt_integration.py'],
        
        # Configuration
        r'config_loader\.py': ['unit/test_config.py', 'regression/test_api_compatibility.py'],
        
        # Wrappers (new comprehensive test)
        r'.*wrapper.*\.py': ['unit/test_*wrapper*.py', 'test_answer_wrapper_comprehensive.py'],
        r'answer_wrapper\.py': ['test_answer_wrapper_comprehensive.py'],
    }
    
    def __init__(self, repo_root: Path = None):
        self.repo_root = repo_root or Path.cwd()
        
    def get_changed_files(self, base_ref: str = "main") -> List[str]:
        """Get list of changed files compared to base ref"""
        try:
            result = subprocess.run(
                ["git", "diff", "--name-only", base_ref + "..HEAD"],
                capture_output=True, text=True, cwd=self.repo_root
            )
            if result.returncode == 0:
                return [f.strip() for f in result.stdout.strip().split('\n') if f.strip()]
        except subprocess.SubprocessError:
            pass
        
        # Fallback: get uncommitted changes
        try:
            result = subprocess.run(
                ["git", "diff", "--name-only"],
                capture_output=True, text=True, cwd=self.repo_root
            )
            return [f.strip() for f in result.stdout.strip().split('\n') if f.strip()]
        except subprocess.SubprocessError:
            return []
    
    def map_files_to_tests(self, changed_files: List[str]) -> Set[str]:
        """Map changed files to relevant test files"""
        relevant_tests = set()
        
        for changed_file in changed_files:
            for pattern, test_patterns in self.FILE_TO_TEST_MAPPING.items():
                if re.search(pattern, changed_file):
                    relevant_tests.update(test_patterns)
        
        # Always include critical tests for any change
        relevant_tests.add('priorities/research_critical.py')
        
        return relevant_tests
    
    def generate_test_command(self, changed_files: List[str], test_root: Path) -> List[str]:
        """Generate intelligent pytest command for changed files"""
        relevant_tests = self.map_files_to_tests(changed_files)
        
        # Resolve test patterns to actual files
        actual_test_files = []
        for pattern in relevant_tests:
            # Handle both direct files and glob patterns
            if '*' in pattern:
                # Glob pattern
                from glob import glob
                matches = glob(str(test_root / pattern))
                actual_test_files.extend(matches)
            else:
                # Direct file
                test_file = test_root / pattern
                if test_file.exists():
                    actual_test_files.append(str(test_file))
        
        if not actual_test_files:
            # Fallback: run all unit tests
            actual_test_files = [str(test_root / "unit")]
        
        return [
            "python", "-m", "pytest",
            *actual_test_files,
            "-v", "--tb=short",
            "--maxfail=5"
        ]

class TestFailureAnalyzer:
    """Analyzes test failures and provides intelligent diagnostics"""
    
    def __init__(self):
        self.failure_patterns = {
            # Import errors
            r'ImportError.*cannot import name': 'API_BREAKING_CHANGE',
            r'ModuleNotFoundError.*No module named': 'MISSING_DEPENDENCY',
            
            # Assertion failures in critical tests
            r'AssertionError.*research.*critical': 'RESEARCH_VALIDITY_ISSUE',
            r'AssertionError.*evaluation.*determinism': 'EVALUATION_NON_DETERMINISTIC',
            r'AssertionError.*score.*regression': 'SCORING_REGRESSION',
            
            # Performance regressions
            r'PERFORMANCE REGRESSION': 'PERFORMANCE_DEGRADATION',
            r'duration.*exceeded': 'TIMEOUT_ISSUE',
            r'memory.*exceeded': 'MEMORY_ISSUE',
            
            # API compatibility
            r'signature.*changed': 'API_SIGNATURE_CHANGE',
            r'parameter.*missing': 'API_PARAMETER_REMOVED',
        }
    
    def analyze_failure_log(self, log_content: str) -> Dict[str, Any]:
        """Analyze test failure log and categorize issues"""
        issues = {
            'critical_failures': [],
            'api_changes': [],
            'performance_issues': [],
            'dependency_issues': [],
            'recommendations': []
        }
        
        for pattern, issue_type in self.failure_patterns.items():
            matches = re.findall(pattern, log_content, re.IGNORECASE)
            
            for match in matches:
                issue = {
                    'type': issue_type,
                    'pattern': pattern,
                    'context': match
                }
                
                if issue_type in ['RESEARCH_VALIDITY_ISSUE', 'EVALUATION_NON_DETERMINISTIC']:
                    issues['critical_failures'].append(issue)
                    issues['recommendations'].append(
                        f"🚨 CRITICAL: {issue_type} detected. All development must stop until resolved."
                    )
                elif issue_type in ['API_BREAKING_CHANGE', 'API_SIGNATURE_CHANGE', 'API_PARAMETER_REMOVED']:
                    issues['api_changes'].append(issue)
                    issues['recommendations'].append(
                        f"⚠️ API Change: Update dependent code and documentation for {issue_type}"
                    )
                elif issue_type in ['PERFORMANCE_DEGRADATION', 'TIMEOUT_ISSUE', 'MEMORY_ISSUE']:
                    issues['performance_issues'].append(issue)
                    issues['recommendations'].append(
                        f"📈 Performance: Investigate {issue_type} - check recent changes"
                    )
                elif issue_type == 'MISSING_DEPENDENCY':
                    issues['dependency_issues'].append(issue)
                    issues['recommendations'].append(
                        f"📦 Dependency: Install missing dependency or update requirements"
                    )
        
        return issues

class CoverageAnalyzer:
    """Analyzes test coverage changes and trends"""
    
    def __init__(self, coverage_file: Path = None):
        self.coverage_file = coverage_file or Path("coverage.json")
    
    def compare_coverage(self, old_coverage_file: Path, new_coverage_file: Path) -> Dict[str, Any]:
        """Compare coverage between two runs"""
        try:
            with open(old_coverage_file) as f:
                old_data = json.load(f)
            with open(new_coverage_file) as f:  
                new_data = json.load(f)
            
            old_total = old_data['totals']['percent_covered']
            new_total = new_data['totals']['percent_covered']
            
            analysis = {
                'old_coverage': old_total,
                'new_coverage': new_total,
                'change': new_total - old_total,
                'files_changed': {},
                'recommendations': []
            }
            
            # Analyze per-file changes
            old_files = old_data.get('files', {})
            new_files = new_data.get('files', {})
            
            for filename, new_stats in new_files.items():
                if filename in old_files:
                    old_coverage = old_files[filename]['summary']['percent_covered']
                    new_coverage = new_stats['summary']['percent_covered']
                    change = new_coverage - old_coverage
                    
                    if abs(change) > 5:  # Significant change
                        analysis['files_changed'][filename] = {
                            'old': old_coverage,
                            'new': new_coverage,
                            'change': change
                        }
            
            # Generate recommendations
            if analysis['change'] < -2:
                analysis['recommendations'].append(
                    "⚠️ Significant coverage decrease. Add tests for new/modified code."
                )
            elif analysis['change'] > 5:
                analysis['recommendations'].append(
                    "✅ Coverage improvement detected. Good job!"
                )
            
            if analysis['new_coverage'] < 80:
                analysis['recommendations'].append(
                    "📊 Coverage below 80%. Focus on testing critical paths."
                )
            
            return analysis
            
        except (FileNotFoundError, json.JSONDecodeError) as e:
            return {'error': f"Coverage analysis failed: {str(e)}"}

def main():
    """CLI for CI/CD utilities"""
    parser = argparse.ArgumentParser(description="CI/CD Test Utilities")
    subparsers = parser.add_subparsers(dest='command', help='Commands')
    
    # Smart test selection
    select_parser = subparsers.add_parser('smart-select', help='Intelligently select tests for changed files')
    select_parser.add_argument('--base-ref', default='main', help='Base git reference')
    select_parser.add_argument('--test-root', type=Path, default=Path.cwd(), help='Test root directory')
    
    # Failure analysis
    analyze_parser = subparsers.add_parser('analyze-failures', help='Analyze test failure logs')
    analyze_parser.add_argument('log_file', type=Path, help='Test failure log file')
    
    # Coverage comparison
    coverage_parser = subparsers.add_parser('compare-coverage', help='Compare coverage reports')
    coverage_parser.add_argument('old_coverage', type=Path, help='Old coverage JSON file')
    coverage_parser.add_argument('new_coverage', type=Path, help='New coverage JSON file')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    if args.command == 'smart-select':
        analyzer = TestImpactAnalyzer()
        changed_files = analyzer.get_changed_files(args.base_ref)
        
        if not changed_files:
            print("No changed files detected. Running basic unit tests.")
            cmd = ["python", "-m", "pytest", "unit/", "-v", "--tb=short"]
        else:
            print(f"📝 Changed files: {', '.join(changed_files)}")
            cmd = analyzer.generate_test_command(changed_files, args.test_root)
        
        print(f"🧪 Recommended test command:")
        print(" ".join(cmd))
        
        # Execute the command
        result = subprocess.run(cmd, cwd=args.test_root.parent)
        sys.exit(result.returncode)
    
    elif args.command == 'analyze-failures':
        if not args.log_file.exists():
            print(f"❌ Log file not found: {args.log_file}")
            sys.exit(1)
        
        analyzer = TestFailureAnalyzer()
        with open(args.log_file) as f:
            log_content = f.read()
        
        analysis = analyzer.analyze_failure_log(log_content)
        
        print("🔍 Test Failure Analysis")
        print("=" * 50)
        
        for category, issues in analysis.items():
            if category == 'recommendations':
                continue
            if issues:
                print(f"\n{category.replace('_', ' ').title()}:")
                for issue in issues:
                    print(f"  - {issue['type']}: {issue.get('context', 'N/A')}")
        
        if analysis['recommendations']:
            print(f"\n💡 Recommendations:")
            for rec in analysis['recommendations']:
                print(f"  {rec}")
    
    elif args.command == 'compare-coverage':
        analyzer = CoverageAnalyzer()
        analysis = analyzer.compare_coverage(args.old_coverage, args.new_coverage)
        
        if 'error' in analysis:
            print(f"❌ {analysis['error']}")
            sys.exit(1)
        
        print("📊 Coverage Analysis")
        print("=" * 30)
        print(f"Old Coverage: {analysis['old_coverage']:.2f}%")
        print(f"New Coverage: {analysis['new_coverage']:.2f}%")
        print(f"Change: {analysis['change']:+.2f}%")
        
        if analysis['files_changed']:
            print(f"\n📁 Files with significant coverage changes:")
            for filename, change_data in analysis['files_changed'].items():
                print(f"  {filename}: {change_data['change']:+.2f}% ({change_data['old']:.1f}% → {change_data['new']:.1f}%)")
        
        if analysis['recommendations']:
            print(f"\n💡 Recommendations:")
            for rec in analysis['recommendations']:
                print(f"  {rec}")

if __name__ == "__main__":
    main()