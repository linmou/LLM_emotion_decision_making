# 🧪 Research-Grade Test Framework

A comprehensive, research-critical testing framework for emotion memory experiments that ensures scientific validity and reproducibility.

## 🎯 Framework Overview

This test framework implements sophisticated patterns specifically designed for AI research:

- **Research-Critical Tests**: P0 tests that protect scientific validity
- **Hierarchical Organization**: Unit → Integration → Regression → E2E
- **Intelligent Test Selection**: AI-powered test selection based on code changes
- **Performance Regression Detection**: Automated performance monitoring
- **Golden Dataset Testing**: Versioned test data for consistency
- **Behavioral Equivalence Testing**: Ensures refactored code produces identical results

## 📁 Directory Structure

```
emotion_experiment_engine/tests/
├── conftest.py                    # Pytest configuration & shared fixtures
├── pytest.ini                    # Test configuration & markers
├── README_TEST_FRAMEWORK.md       # This document
│
├── priorities/                    # P0 Research-Critical Tests
│   └── research_critical.py       # Scientific validity protection
│
├── unit/                          # Fast, isolated tests
│   ├── test_data_models.py
│   ├── test_dataset_*.py
│   └── test_evaluation.py
│
├── integration/                   # Component integration tests
│   ├── test_dataset_integration.py
│   └── test_memory_integration.py
│
├── regression/                    # API stability & behavioral equivalence
│   ├── test_api_compatibility.py
│   └── test_behavioral_equivalence.py
│
├── e2e/                          # End-to-end system tests
│   └── test_full_pipeline.py
│
├── test_data/                     # Versioned test datasets
│   ├── version_control.py         # Dataset version management
│   ├── versions/                  # Versioned data storage
│   └── current/                   # Current dataset symlinks
│
├── utils/                         # Test utilities & orchestration
│   ├── test_runners.py            # Intelligent test suite orchestration
│   ├── performance_tracker.py     # Advanced performance monitoring
│   └── ci_helpers.py              # CI/CD automation utilities
│
├── .github/workflows/             # CI/CD configuration
│   └── test-suite.yml             # Research-grade test pipeline
│
└── test_answer_wrapper_comprehensive.py  # Comprehensive component tests
```

## 🏃‍♂️ Quick Start

### Basic Test Execution

```bash
# Run all tests
pytest

# Run by category
pytest -m unit           # Unit tests only
pytest -m critical      # Research-critical tests
pytest -m regression    # Regression tests
pytest -m comprehensive # Comprehensive component tests

# Smart test selection (based on changed files)
python tests/utils/ci_helpers.py smart-select --base-ref main
```

### Advanced Usage

```bash
# Performance analysis
python tests/utils/performance_tracker.py analyze --category unit
python tests/utils/performance_tracker.py report --category regression

# Test data management
python tests/test_data/version_control.py init        # Initialize baseline datasets
python tests/test_data/version_control.py verify      # Verify data integrity

# Suite orchestration
python tests/utils/test_runners.py run smoke         # Essential tests
python tests/utils/test_runners.py run comprehensive # Full component tests
```

## 🎯 Test Categories & Markers

### Core Categories

- **`@pytest.mark.critical`**: Research-critical tests (P0 priority)
  - Must pass before any development
  - Protect scientific validity
  - Zero tolerance for failures

- **`@pytest.mark.unit`**: Fast, isolated unit tests
  - < 1 second per test
  - No external dependencies
  - High coverage target (>90%)

- **`@pytest.mark.integration`**: Component integration tests
  - Test component interactions
  - Moderate execution time (< 10s)
  - Real data simulation

- **`@pytest.mark.regression`**: Backward compatibility tests
  - API stability validation
  - Behavioral equivalence checks
  - No breaking changes allowed

- **`@pytest.mark.e2e`**: End-to-end system tests
  - Full pipeline validation
  - Long execution time acceptable
  - Production-like scenarios

- **`@pytest.mark.comprehensive`**: Complete component coverage
  - All-in-one test suites
  - Multiple testing approaches
  - High confidence validation

### Specialized Markers

- **`@pytest.mark.golden`**: Golden dataset tests
- **`@pytest.mark.performance`**: Performance benchmarking
- **`@pytest.mark.statistical`**: Statistical validation
- **`@pytest.mark.property`**: Property-based testing

## 🔍 Research-Critical Testing

### Scientific Validity Protection

```python
# Example critical test
@pytest.mark.critical
def test_evaluation_determinism():
    """Ensure evaluation results are deterministic for reproducibility"""
    # Test that same input always produces same output
    # MUST NEVER FAIL - blocks all development if broken
```

### Behavioral Equivalence Testing

```python
@pytest.mark.regression
def test_evaluation_behavioral_equivalence():
    """Ensure refactored evaluation produces identical results"""
    # Compare outputs before and after changes
    # Protects scientific validity during refactoring
```

## 📊 Performance Monitoring

### Automated Performance Tracking

The framework automatically tracks:
- **Execution Time**: Per test and category
- **Memory Usage**: Peak memory consumption
- **Regression Detection**: Performance degradation alerts
- **Trend Analysis**: Long-term performance trends

### Performance Benchmarks

```python
PERFORMANCE_BENCHMARKS = {
    "unit": {"target": 0.1, "max": 1.0},       # 100ms target, 1s max
    "integration": {"target": 2.0, "max": 10.0}, # 2s target, 10s max
    "regression": {"target": 5.0, "max": 30.0},  # 5s target, 30s max
}
```

## 🤖 AI-Powered Test Selection

### Intelligent Test Selection

The framework uses AI to select relevant tests based on code changes:

```python
# Automatically maps changed files to relevant tests
FILE_TO_TEST_MAPPING = {
    r'datasets/.*\.py': ['unit/test_*dataset*.py', 'regression/test_*_equivalence.py'],
    r'evaluation_utils\.py': ['unit/test_evaluation.py', 'priorities/research_critical.py'],
}
```

### Smart Test Execution

```bash
# Analyzes git diff and runs only relevant tests
python tests/utils/ci_helpers.py smart-select

# Example output:
# 📝 Changed files: datasets/emotion_check.py, evaluation_utils.py
# 🧪 Recommended test command:
# python -m pytest unit/test_emotion_check_dataset.py priorities/research_critical.py -v
```

## 🗂️ Test Data Management

### Versioned Test Datasets

```python
# Create versioned test data
manager = TestDataVersionManager()
manager.create_version("passkey_test", source_file, "1.0.0", 
                      description="Initial passkey test data")

# Use specific version
test_data_path = manager.get_version("passkey_test", "1.0.0")
```

### Golden Dataset Testing

- **Deterministic Results**: Same data always produces same results
- **Version Control**: Track changes in test data
- **Integrity Verification**: Hash-based corruption detection

## 🚀 CI/CD Integration

### GitHub Actions Pipeline

The framework includes a sophisticated CI/CD pipeline:

1. **Critical Tests** (P0): Must pass before anything else
2. **Unit Tests**: Fast feedback with coverage tracking
3. **Integration Tests**: Component interaction validation
4. **Regression Tests**: API stability and behavioral equivalence
5. **Performance Tests**: Automated performance monitoring
6. **E2E Tests**: Full system validation (scheduled)

### Pipeline Features

- **Fail-Fast**: Critical test failures block all other tests
- **Parallel Execution**: Optimized test execution time
- **Coverage Tracking**: Automated coverage reporting
- **Performance Analysis**: Trend detection and alerting
- **Intelligent Selection**: Changed-file-based test selection

## 🛠️ Development Workflow

### TDD with Research Validity

1. **Red**: Write failing test that defines scientific requirement
2. **Green**: Implement minimum code to pass test
3. **Refactor**: Improve code while maintaining behavioral equivalence
4. **Validate**: Run regression tests to ensure scientific validity

### Example Workflow

```bash
# 1. Run critical tests first
pytest -m critical

# 2. Develop with unit tests
pytest -m unit --watch

# 3. Integration testing
pytest -m integration

# 4. Pre-commit regression validation
pytest -m regression

# 5. Full validation before merge
pytest tests/utils/test_runners.py run comprehensive
```

## 🔧 Configuration

### Key Configuration Files

- **`pytest.ini`**: Pytest settings, markers, coverage requirements
- **`conftest.py`**: Shared fixtures, performance tracking
- **`.github/workflows/test-suite.yml`**: CI/CD pipeline configuration

### Environment Setup

```bash
# Activate conda environment
conda activate llm_behav

# Install test dependencies (if needed)
pip install pytest pytest-cov pytest-mock

# Initialize test data
python tests/test_data/version_control.py init
```

## 🎯 Best Practices

### Test Writing Guidelines

1. **Research-Critical First**: Protect scientific validity above all
2. **Fast Unit Tests**: Keep unit tests under 1 second
3. **Isolated Testing**: No shared state between tests
4. **Golden Data**: Use versioned test data for consistency
5. **Performance Awareness**: Monitor test execution time
6. **Comprehensive Coverage**: Aim for >90% coverage on core code

### Scientific Integrity

- **Deterministic Testing**: Ensure reproducible results
- **Version Control**: Track all changes to test data
- **Behavioral Equivalence**: Validate that refactoring doesn't change results
- **Statistical Validation**: Use proper statistical testing methods

## 🚨 Troubleshooting

### Common Issues

**Critical Tests Failing**
```bash
# Check what's failing
pytest -m critical -v --tb=long

# Critical tests MUST be fixed immediately
# No development should continue until resolved
```

**Performance Regressions**
```bash
# Analyze performance trends
python tests/utils/performance_tracker.py analyze

# Get optimization recommendations
python tests/utils/performance_tracker.py report
```

**Test Data Issues**
```bash
# Verify data integrity
python tests/test_data/version_control.py verify

# Recreate baseline data if corrupted
python tests/test_data/version_control.py init
```

## 📈 Metrics & Reporting

### Key Metrics Tracked

- **Test Coverage**: >85% required, >90% target
- **Execution Time**: Per category performance tracking
- **Failure Rates**: Success/failure trends over time
- **Performance Trends**: Regression detection and alerting

### Automated Reporting

- **Coverage Reports**: HTML and XML output
- **Performance Reports**: JSON format with trends
- **Failure Analysis**: Intelligent issue categorization
- **CI Summary**: GitHub Actions summary with recommendations

## 🎓 Advanced Features

### Property-Based Testing

```python
from hypothesis import given, strategies as st

@given(st.text(min_size=1))
def test_evaluation_handles_any_text(text):
    # Test with randomly generated text inputs
    result = evaluate_response(text, "ground_truth")
    assert 0 <= result <= 1
```

### Statistical Testing

```python
@pytest.mark.statistical
def test_score_distribution():
    """Validate that scores follow expected statistical properties"""
    scores = [evaluate_response(r, gt) for r, gt in test_cases]
    
    # Statistical assertions
    assert 0.3 <= np.mean(scores) <= 0.7
    assert np.std(scores) < 0.5
```

## 🎯 Framework Goals Achieved

✅ **Scientific Validity Protection**: Research-critical tests prevent invalid results
✅ **Intelligent Test Organization**: Hierarchical structure with smart selection
✅ **Performance Regression Prevention**: Automated monitoring and alerting
✅ **Behavioral Equivalence Assurance**: Refactoring safety guarantees
✅ **Comprehensive Coverage**: Multi-level testing approach
✅ **CI/CD Integration**: Production-ready automation pipeline
✅ **Developer Productivity**: Fast feedback and intelligent tooling

This framework transforms basic testing into a research-grade validation system that ensures the scientific integrity and reproducibility of emotion memory experiments.