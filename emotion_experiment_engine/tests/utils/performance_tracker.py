#!/usr/bin/env python3
"""
Test Performance Tracking and Optimization System

Monitors test execution performance, detects regressions, and provides
intelligent optimization recommendations for the test suite.
"""

import json
import psutil
import sqlite3
import statistics
import time
import warnings
from collections import defaultdict
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple

import numpy as np


@dataclass
class TestExecutionRecord:
    """Record of a single test execution"""
    test_name: str
    category: str
    duration: float
    memory_usage: int  # Peak memory in bytes
    cpu_percent: float
    timestamp: str
    exit_code: int
    parameters: Dict[str, Any]  # Test parameters/configuration


@dataclass
class PerformanceBenchmark:
    """Performance benchmark for a test category"""
    category: str
    target_duration: float  # Target execution time (seconds)
    max_duration: float     # Maximum acceptable time
    target_memory: int      # Target memory usage (bytes) 
    max_memory: int         # Maximum acceptable memory
    sample_size: int        # Number of samples for statistics


class TestPerformanceTracker:
    """Advanced performance tracking with regression detection"""
    
    # Performance targets by test category
    PERFORMANCE_BENCHMARKS = {
        "unit": PerformanceBenchmark(
            category="unit",
            target_duration=0.1,   # 100ms target
            max_duration=1.0,      # 1s maximum
            target_memory=50_000_000,    # 50MB target
            max_memory=200_000_000,      # 200MB maximum
            sample_size=10
        ),
        "integration": PerformanceBenchmark(
            category="integration", 
            target_duration=2.0,   # 2s target
            max_duration=10.0,     # 10s maximum
            target_memory=200_000_000,   # 200MB target
            max_memory=1_000_000_000,    # 1GB maximum
            sample_size=5
        ),
        "regression": PerformanceBenchmark(
            category="regression",
            target_duration=5.0,   # 5s target
            max_duration=30.0,     # 30s maximum
            target_memory=500_000_000,   # 500MB target
            max_memory=2_000_000_000,    # 2GB maximum
            sample_size=3
        ),
        "e2e": PerformanceBenchmark(
            category="e2e",
            target_duration=15.0,  # 15s target
            max_duration=120.0,    # 2min maximum
            target_memory=1_000_000_000,  # 1GB target
            max_memory=4_000_000_000,     # 4GB maximum
            sample_size=2
        ),
        "critical": PerformanceBenchmark(
            category="critical",
            target_duration=1.0,   # 1s target
            max_duration=5.0,      # 5s maximum
            target_memory=100_000_000,   # 100MB target
            max_memory=500_000_000,      # 500MB maximum
            sample_size=5
        )
    }
    
    def __init__(self, db_path: Path = None):
        self.db_path = db_path or Path(__file__).parent / "performance.db"
        self.current_test = None
        self.start_time = None
        self.start_memory = None
        self.process = psutil.Process()
        
        self._init_database()
    
    def _init_database(self):
        """Initialize SQLite database for performance data"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS test_executions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    test_name TEXT NOT NULL,
                    category TEXT NOT NULL,
                    duration REAL NOT NULL,
                    memory_usage INTEGER NOT NULL,
                    cpu_percent REAL NOT NULL,
                    timestamp TEXT NOT NULL,
                    exit_code INTEGER NOT NULL,
                    parameters TEXT
                )
            """)
            
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_test_name ON test_executions(test_name)
            """)
            
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_category ON test_executions(category)
            """)
            
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_timestamp ON test_executions(timestamp)
            """)
    
    def start_tracking(self, test_name: str, category: str = "unknown", 
                      parameters: Dict[str, Any] = None):
        """Start tracking performance for a test"""
        self.current_test = {
            "name": test_name,
            "category": category,
            "parameters": parameters or {}
        }
        self.start_time = time.time()
        
        # Record initial memory
        try:
            memory_info = self.process.memory_info()
            self.start_memory = memory_info.rss
        except psutil.AccessDenied:
            self.start_memory = 0
    
    def stop_tracking(self, exit_code: int = 0) -> TestExecutionRecord:
        """Stop tracking and record performance data"""
        if not self.current_test or not self.start_time:
            raise RuntimeError("No active tracking session")
        
        end_time = time.time()
        duration = end_time - self.start_time
        
        # Calculate resource usage
        try:
            memory_info = self.process.memory_info()
            peak_memory = memory_info.rss
            cpu_percent = self.process.cpu_percent()
        except psutil.AccessDenied:
            peak_memory = 0
            cpu_percent = 0.0
        
        # Create execution record
        record = TestExecutionRecord(
            test_name=self.current_test["name"],
            category=self.current_test["category"],
            duration=duration,
            memory_usage=peak_memory,
            cpu_percent=cpu_percent,
            timestamp=datetime.now().isoformat(),
            exit_code=exit_code,
            parameters=self.current_test["parameters"]
        )
        
        # Store in database
        self._store_record(record)
        
        # Check for performance issues
        self._check_performance_thresholds(record)
        
        # Reset tracking state
        self.current_test = None
        self.start_time = None
        self.start_memory = None
        
        return record
    
    def _store_record(self, record: TestExecutionRecord):
        """Store performance record in database"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO test_executions 
                (test_name, category, duration, memory_usage, cpu_percent, 
                 timestamp, exit_code, parameters)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                record.test_name,
                record.category,
                record.duration,
                record.memory_usage,
                record.cpu_percent,
                record.timestamp,
                record.exit_code,
                json.dumps(record.parameters)
            ))
    
    def _check_performance_thresholds(self, record: TestExecutionRecord):
        """Check performance against thresholds and issue warnings"""
        benchmark = self.PERFORMANCE_BENCHMARKS.get(record.category)
        if not benchmark:
            return
        
        # Duration check
        if record.duration > benchmark.max_duration:
            warnings.warn(
                f"PERFORMANCE REGRESSION: Test '{record.test_name}' took "
                f"{record.duration:.2f}s (max: {benchmark.max_duration}s)",
                category=UserWarning
            )
        elif record.duration > benchmark.target_duration:
            warnings.warn(
                f"Performance warning: Test '{record.test_name}' took "
                f"{record.duration:.2f}s (target: {benchmark.target_duration}s)",
                category=UserWarning
            )
        
        # Memory check
        if record.memory_usage > benchmark.max_memory:
            warnings.warn(
                f"MEMORY REGRESSION: Test '{record.test_name}' used "
                f"{record.memory_usage / 1_000_000:.1f}MB "
                f"(max: {benchmark.max_memory / 1_000_000:.1f}MB)",
                category=UserWarning
            )
        elif record.memory_usage > benchmark.target_memory:
            warnings.warn(
                f"Memory warning: Test '{record.test_name}' used "
                f"{record.memory_usage / 1_000_000:.1f}MB "
                f"(target: {benchmark.target_memory / 1_000_000:.1f}MB)",
                category=UserWarning
            )
    
    def get_performance_history(self, test_name: str = None, 
                               category: str = None,
                               days: int = 30) -> List[TestExecutionRecord]:
        """Get performance history for analysis"""
        cutoff_date = (datetime.now() - timedelta(days=days)).isoformat()
        
        query = "SELECT * FROM test_executions WHERE timestamp > ?"
        params = [cutoff_date]
        
        if test_name:
            query += " AND test_name = ?"
            params.append(test_name)
        
        if category:
            query += " AND category = ?"
            params.append(category)
        
        query += " ORDER BY timestamp DESC"
        
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(query, params)
            
            records = []
            for row in cursor:
                record = TestExecutionRecord(
                    test_name=row["test_name"],
                    category=row["category"],
                    duration=row["duration"],
                    memory_usage=row["memory_usage"],
                    cpu_percent=row["cpu_percent"],
                    timestamp=row["timestamp"],
                    exit_code=row["exit_code"],
                    parameters=json.loads(row["parameters"]) if row["parameters"] else {}
                )
                records.append(record)
        
        return records
    
    def analyze_performance_trends(self, category: str = None) -> Dict[str, Any]:
        """Analyze performance trends and detect regressions"""
        records = self.get_performance_history(category=category, days=90)
        
        if not records:
            return {"error": "No performance data available"}
        
        # Group by test name
        by_test = defaultdict(list)
        for record in records:
            by_test[record.test_name].append(record)
        
        analysis = {
            "total_tests": len(by_test),
            "total_executions": len(records),
            "analysis_period_days": 90,
            "test_analysis": {},
            "category_summary": {},
            "regressions_detected": []
        }
        
        # Analyze each test individually
        for test_name, test_records in by_test.items():
            if len(test_records) < 3:  # Need at least 3 data points
                continue
            
            # Sort by timestamp
            test_records.sort(key=lambda r: r.timestamp)
            
            durations = [r.duration for r in test_records]
            memory_usages = [r.memory_usage for r in test_records]
            
            # Calculate trends
            recent_durations = durations[-5:]  # Last 5 runs
            historical_durations = durations[:-5] if len(durations) > 5 else durations[:-1]
            
            if historical_durations:
                recent_mean = statistics.mean(recent_durations)
                historical_mean = statistics.mean(historical_durations)
                duration_trend = (recent_mean - historical_mean) / historical_mean * 100
            else:
                duration_trend = 0.0
            
            # Memory trend
            recent_memory = [r.memory_usage for r in test_records[-5:]]
            historical_memory = [r.memory_usage for r in test_records[:-5]] if len(test_records) > 5 else [r.memory_usage for r in test_records[:-1]]
            
            if historical_memory:
                memory_trend = (statistics.mean(recent_memory) - statistics.mean(historical_memory)) / statistics.mean(historical_memory) * 100
            else:
                memory_trend = 0.0
            
            test_analysis = {
                "executions": len(test_records),
                "avg_duration": statistics.mean(durations),
                "duration_std": statistics.stdev(durations) if len(durations) > 1 else 0,
                "duration_trend_percent": duration_trend,
                "avg_memory_mb": statistics.mean(memory_usages) / 1_000_000,
                "memory_trend_percent": memory_trend,
                "success_rate": sum(1 for r in test_records if r.exit_code == 0) / len(test_records) * 100
            }
            
            analysis["test_analysis"][test_name] = test_analysis
            
            # Detect significant regressions (>20% performance degradation)
            if duration_trend > 20:
                analysis["regressions_detected"].append({
                    "test": test_name,
                    "type": "duration",
                    "trend_percent": duration_trend,
                    "severity": "high" if duration_trend > 50 else "medium"
                })
            
            if memory_trend > 30:
                analysis["regressions_detected"].append({
                    "test": test_name,
                    "type": "memory", 
                    "trend_percent": memory_trend,
                    "severity": "high" if memory_trend > 80 else "medium"
                })
        
        # Category-level analysis
        if category:
            category_records = [r for r in records if r.category == category]
            if category_records:
                durations = [r.duration for r in category_records]
                memory_usages = [r.memory_usage for r in category_records]
                
                analysis["category_summary"] = {
                    "category": category,
                    "executions": len(category_records),
                    "avg_duration": statistics.mean(durations),
                    "p95_duration": np.percentile(durations, 95),
                    "avg_memory_mb": statistics.mean(memory_usages) / 1_000_000,
                    "p95_memory_mb": np.percentile(memory_usages, 95) / 1_000_000
                }
        
        return analysis
    
    def get_slowest_tests(self, category: str = None, limit: int = 10) -> List[Tuple[str, float, int]]:
        """Get slowest tests by average execution time"""
        records = self.get_performance_history(category=category, days=30)
        
        # Group by test name and calculate averages
        by_test = defaultdict(list)
        for record in records:
            by_test[record.test_name].append(record.duration)
        
        # Calculate averages and sort
        test_averages = []
        for test_name, durations in by_test.items():
            if len(durations) >= 2:  # Need at least 2 runs
                avg_duration = statistics.mean(durations)
                execution_count = len(durations)
                test_averages.append((test_name, avg_duration, execution_count))
        
        # Sort by average duration (descending)
        test_averages.sort(key=lambda x: x[1], reverse=True)
        
        return test_averages[:limit]
    
    def generate_optimization_recommendations(self, category: str = None) -> List[Dict[str, Any]]:
        """Generate intelligent optimization recommendations"""
        analysis = self.analyze_performance_trends(category)
        slowest_tests = self.get_slowest_tests(category, limit=5)
        
        recommendations = []
        
        # Regression recommendations
        for regression in analysis.get("regressions_detected", []):
            rec = {
                "type": "regression_fix",
                "priority": "high" if regression["severity"] == "high" else "medium",
                "title": f"Performance regression in {regression['test']}",
                "description": f"{regression['type'].title()} usage increased by {regression['trend_percent']:.1f}%",
                "action": f"Investigate recent changes to {regression['test']} that may have caused {regression['type']} regression"
            }
            recommendations.append(rec)
        
        # Slow test recommendations
        benchmark = self.PERFORMANCE_BENCHMARKS.get(category or "unit")
        if benchmark and slowest_tests:
            for test_name, avg_duration, count in slowest_tests[:3]:
                if avg_duration > benchmark.target_duration * 2:  # 2x slower than target
                    rec = {
                        "type": "optimization",
                        "priority": "medium",
                        "title": f"Optimize slow test: {test_name}",
                        "description": f"Average duration: {avg_duration:.2f}s (target: {benchmark.target_duration}s)",
                        "action": "Consider mocking heavy operations, reducing test data size, or splitting into smaller tests"
                    }
                    recommendations.append(rec)
        
        # Category-level recommendations
        if "category_summary" in analysis and analysis["category_summary"]:
            summary = analysis["category_summary"]
            benchmark = self.PERFORMANCE_BENCHMARKS.get(summary["category"])
            
            if benchmark:
                if summary["p95_duration"] > benchmark.max_duration:
                    rec = {
                        "type": "category_optimization",
                        "priority": "high", 
                        "title": f"Category {summary['category']} performance issues",
                        "description": f"95th percentile duration: {summary['p95_duration']:.2f}s exceeds maximum: {benchmark.max_duration}s",
                        "action": f"Review all {summary['category']} tests for optimization opportunities"
                    }
                    recommendations.append(rec)
                
                if summary["p95_memory_mb"] > benchmark.max_memory / 1_000_000:
                    rec = {
                        "type": "memory_optimization",
                        "priority": "medium",
                        "title": f"High memory usage in {summary['category']} tests",
                        "description": f"95th percentile memory: {summary['p95_memory_mb']:.1f}MB exceeds target",
                        "action": "Review memory usage patterns, add cleanup code, or use smaller test datasets"
                    }
                    recommendations.append(rec)
        
        # Sort by priority
        priority_order = {"high": 0, "medium": 1, "low": 2}
        recommendations.sort(key=lambda x: priority_order.get(x["priority"], 2))
        
        return recommendations
    
    def export_performance_report(self, output_file: Path = None, 
                                 category: str = None) -> Path:
        """Export comprehensive performance report"""
        if output_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = Path(f"performance_report_{timestamp}.json")
        
        analysis = self.analyze_performance_trends(category)
        slowest_tests = self.get_slowest_tests(category, limit=20)
        recommendations = self.generate_optimization_recommendations(category)
        
        report = {
            "generated_at": datetime.now().isoformat(),
            "category_filter": category,
            "performance_analysis": analysis,
            "slowest_tests": [
                {"test_name": name, "avg_duration": duration, "executions": count}
                for name, duration, count in slowest_tests
            ],
            "optimization_recommendations": recommendations,
            "performance_benchmarks": {
                name: asdict(benchmark) 
                for name, benchmark in self.PERFORMANCE_BENCHMARKS.items()
            }
        }
        
        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        return output_file


# Context manager for easy test tracking
class PerformanceContext:
    """Context manager for automatic performance tracking"""
    
    def __init__(self, tracker: TestPerformanceTracker, test_name: str, 
                 category: str = "unknown", parameters: Dict[str, Any] = None):
        self.tracker = tracker
        self.test_name = test_name
        self.category = category
        self.parameters = parameters
        self.record = None
    
    def __enter__(self):
        self.tracker.start_tracking(self.test_name, self.category, self.parameters)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        exit_code = 1 if exc_type else 0
        self.record = self.tracker.stop_tracking(exit_code)
        return False  # Don't suppress exceptions


def main():
    """Command-line interface for performance tracking"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Test Performance Tracker")
    subparsers = parser.add_subparsers(dest='command', help='Commands')
    
    # Analyze command
    analyze_parser = subparsers.add_parser('analyze', help='Analyze performance trends')
    analyze_parser.add_argument('--category', help='Filter by test category')
    analyze_parser.add_argument('--days', type=int, default=30, help='Analysis period in days')
    
    # Report command
    report_parser = subparsers.add_parser('report', help='Generate performance report')
    report_parser.add_argument('--category', help='Filter by test category')
    report_parser.add_argument('--output', type=Path, help='Output file path')
    
    # Slowest command
    slowest_parser = subparsers.add_parser('slowest', help='Show slowest tests')
    slowest_parser.add_argument('--category', help='Filter by test category')
    slowest_parser.add_argument('--limit', type=int, default=10, help='Number of tests to show')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    tracker = TestPerformanceTracker()
    
    if args.command == 'analyze':
        analysis = tracker.analyze_performance_trends(args.category)
        
        print("📊 Performance Analysis")
        print("=" * 50)
        print(f"Period: {analysis.get('analysis_period_days', 'N/A')} days")
        print(f"Tests analyzed: {analysis.get('total_tests', 0)}")
        print(f"Total executions: {analysis.get('total_executions', 0)}")
        
        if analysis.get("regressions_detected"):
            print(f"\n❌ Regressions detected: {len(analysis['regressions_detected'])}")
            for regression in analysis["regressions_detected"]:
                print(f"  - {regression['test']}: {regression['type']} +{regression['trend_percent']:.1f}%")
        
        if analysis.get("category_summary"):
            summary = analysis["category_summary"]
            print(f"\n📈 Category: {summary['category']}")
            print(f"  Avg duration: {summary['avg_duration']:.2f}s")
            print(f"  P95 duration: {summary['p95_duration']:.2f}s")
            print(f"  Avg memory: {summary['avg_memory_mb']:.1f}MB")
    
    elif args.command == 'slowest':
        slowest = tracker.get_slowest_tests(args.category, args.limit)
        
        print(f"🐌 Slowest Tests ({args.category or 'all categories'})")
        print("=" * 60)
        
        for i, (test_name, duration, count) in enumerate(slowest, 1):
            print(f"{i:2d}. {test_name[:50]:<50} {duration:6.2f}s ({count} runs)")
    
    elif args.command == 'report':
        output_file = tracker.export_performance_report(args.output, args.category)
        print(f"📄 Performance report generated: {output_file}")
        
        # Show recommendations
        recommendations = tracker.generate_optimization_recommendations(args.category)
        if recommendations:
            print(f"\n💡 Optimization Recommendations ({len(recommendations)}):")
            for i, rec in enumerate(recommendations[:5], 1):
                print(f"{i}. [{rec['priority'].upper()}] {rec['title']}")
                print(f"   {rec['description']}")


if __name__ == "__main__":
    main()