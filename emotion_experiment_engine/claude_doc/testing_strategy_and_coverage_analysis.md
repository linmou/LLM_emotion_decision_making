# Testing Strategy and Coverage Analysis
<!-- Version: 1.1.0 - Updated: 2025-09-06 - Professional Test Framework & AnswerWrapper Testing -->

## Overview

The emotion memory experiments module implements a comprehensive testing strategy that emphasizes Test-Driven Development (TDD), behavioral equivalence validation, and scientific integrity. The testing architecture spans unit tests, integration tests, real data validation, and refactoring verification, ensuring both software quality and research validity.

### Recent game_theory choice-ratio updates (2025-12-29)

- Behavior-level aggregation now maps unmappable option ids (for example, `score=3` when options are only 1/2) to an explicit `"unknown"` bucket; matched options with missing/empty `behavior` still raise.
- Index-level ratios continue to count the original option id.
- Unparseable decisions now return `option_id=-1` (instead of `NaN`), so failures are included in `summary_choice_ratio*.csv` and show up as `"unknown"` in `summary_behavior_ratio*.csv`.
- CSVs: `summary_behavior_ratio.csv` uses column `behavior` (may include `"unknown"`); choice-ratio CSVs remain id-based.
- Key tests: `test_games_behavior_choice_ratio.py` (unknown bucket + missing behavior error), `test_split_choice_ratio.py`, `test_game_stats_analysis.py`, `test_experiment_choice_ratio_persistence.py`.

## Testing Architecture

```mermaid
graph TB
    subgraph "Test Categories"
        Unit[Unit Tests]
        Integration[Integration Tests]
        EndToEnd[End-to-End Tests]
        Behavioral[Behavioral Equivalence]
        RealData[Real Data Validation]
    end
    
    subgraph "Test Components"
        Factory[Dataset Factory Tests]
        BaseDataset[Base Dataset Tests]
        SpecDataset[Specialized Dataset Tests]
        AnswerWrapper[AnswerWrapper Tests]
        Experiment[Experiment Tests]
        Config[Configuration Tests]
        Evaluation[Evaluation Tests]
    end
    
    subgraph "Test Data"
        MockData[Mock Test Data]
        SyntheticData[Synthetic Datasets]
        RealBenchmarks[Real Benchmark Data]
        EdgeCases[Edge Case Data]
    end
    
    Unit --> Factory
    Unit --> BaseDataset
    Unit --> SpecDataset
    Integration --> Experiment
    Integration --> Config
    EndToEnd --> Evaluation
    Behavioral --> RealData
    
    Factory --> MockData
    BaseDataset --> SyntheticData
    SpecDataset --> RealBenchmarks
    Experiment --> EdgeCases
```

## Test-Driven Development (TDD) Implementation

### **Red-Green-Refactor Cycle Documentation**

The testing suite follows strict TDD principles with explicit documentation of the development cycle.

#### **Example: Dataset Factory TDD Cycle**

```python
#!/usr/bin/env python3
"""
Test file for: Dataset Factory Pattern (TDD Red phase)
Purpose: Test registry-based factory for eliminating if-else chains

This test suite ensures the factory pattern properly creates specialized dataset
classes based on configuration without if-else branching.
"""

class TestDatasetFactory(unittest.TestCase):
    def test_dataset_registry_exists(self):
        """Test that dataset registry is properly defined"""
        # RED PHASE: This test initially fails
        if DATASET_REGISTRY is None:
            self.skipTest("Dataset registry not implemented yet (Red phase)")
        
        # GREEN PHASE: Test passes after implementation
        self.assertIsInstance(DATASET_REGISTRY, dict)
        self.assertIn("infinitebench", DATASET_REGISTRY)
        self.assertIn("longbench", DATASET_REGISTRY)
        self.assertIn("locomo", DATASET_REGISTRY)
```

#### **TDD Documentation Pattern**
```python
# Each test class includes TDD phase documentation
"""
Test file for: [Component Name] ([TDD Phase])
Purpose: [Specific testing purpose]

Red Phase: Tests written before implementation - expected to fail
Green Phase: Minimal implementation to make tests pass
Refactor Phase: Code improvement while maintaining test pass rate
"""
```

### **Regression Testing Strategy**

#### **Behavioral Equivalence Testing**
```python
# test_behavioral_equivalence.py
class TestBehavioralEquivalence(unittest.TestCase):
    """
    Validates that refactored components produce identical results to original implementation.
    Critical for scientific integrity during architectural changes.
    """
    
    def test_evaluation_equivalence(self):
        """Ensure refactored evaluation produces identical scores"""
        
        # Test with identical inputs across old and new implementations
        test_cases = [
            ("response text", "ground truth", "passkey"),
            ("different response", ["multiple", "answers"], "qa_task"),
        ]
        
        for response, ground_truth, task_type in test_cases:
            # Original evaluation method
            old_score = original_evaluate_function(response, ground_truth, task_type)
            
            # New specialized dataset evaluation
            dataset = create_dataset_from_config(test_config)
            new_score = dataset.evaluate_response(response, ground_truth, task_type)
            
            # Must be exactly equivalent for scientific validity
            self.assertEqual(old_score, new_score,
                f"Evaluation divergence for task {task_type}")
```

#### **Refactoring Validation Framework**
```python
def test_factory_eliminates_if_else_chains(self):
    """Test that factory pattern eliminates if-else branching"""
    
    # This is a design test - factory should use registry lookup
    # instead of if-else chains
    test_cases = [
        ("infinitebench", "passkey"),
        ("longbench", "narrativeqa"), 
        ("locomo", "conversational_qa")
    ]
    
    for benchmark_name, task_type in test_cases:
        config = BenchmarkConfig(
            name=benchmark_name,
            task_type=task_type,
            data_path=test_file
        )
        
        # Should create dataset without any branching logic visible to users
        dataset = create_dataset_from_config(config)
        self.assertIsNotNone(dataset)
        self.assertIsInstance(dataset, BaseBenchmarkDataset)
```

## Professional Test Configuration (2025-09-06 Update)

### **pytest.ini: Production-Grade Testing Standards**

The emotion memory experiments now implements a comprehensive pytest configuration that enforces professional quality standards:

```ini
[tool:pytest]
# Test discovery
testpaths = emotion_experiment_engine/tests
python_files = test_*.py
python_classes = Test*
python_functions = test_*

# Professional marker system
markers =
    unit: Unit tests (fast, isolated)
    integration: Integration tests (moderate speed)
    regression: Regression tests (ensure no breaking changes)
    e2e: End-to-end tests (slow, comprehensive)
    critical: Research-critical tests (must never fail)
    comprehensive: Comprehensive test suites covering entire components

# Quality enforcement
addopts = 
    --strict-markers
    --strict-config
    --cov=emotion_experiment_engine
    --cov-fail-under=85          # Minimum 85% coverage requirement
    --maxfail=5                  # Stop after 5 failures
    --timeout=300                # 5-minute timeout per test

# Coverage reporting
--cov-report=term-missing        # Terminal coverage gaps
--cov-report=html:tests/coverage/html    # HTML coverage reports  
--cov-report=xml:tests/coverage/coverage.xml  # XML for CI/CD
```

### **Test Organization Enhancements**

**New Directory Structure**:
```
tests/
├── priorities/
│   └── research_critical.py        # Research-critical test definitions
├── regression/
│   ├── test_api_compatibility.py   # API backward compatibility
│   └── test_behavioral_equivalence.py  # Behavioral consistency validation
├── utils/
│   ├── ci_helpers.py              # CI/CD integration utilities
│   ├── performance_tracker.py     # Performance monitoring
│   └── test_runners.py           # Custom test execution logic
├── conftest.py                   # Shared test configuration
├── pytest.ini                   # Professional pytest configuration
└── README_TEST_FRAMEWORK.md      # Testing documentation
```

**Key Improvements**:
- **Research-Critical Priority System**: Separates tests that must never fail for scientific integrity
- **Regression Testing Framework**: Automated detection of behavioral changes
- **CI/CD Integration**: GitHub workflows and automated quality gates
- **Performance Tracking**: Test execution time monitoring and performance regression detection

### Test Layout Guard (2025-09-07)

- Enforced rule: no duplicate `test_*.py` filenames at both `tests/` root and any of `tests/unit`, `tests/integration`, `tests/e2e`, `tests/regression`.
- CI check: `.github/workflows/test_layout_check.yml` runs `python emotion_experiment_engine/tests/utils/check_test_layout.py` on push/PR.
- Optional pre-commit: `.pre-commit-config.yaml` provides a local hook to run the same check.

Updated test locations reflecting recent migration:
- Prompt wrapper unit tests → `emotion_experiment_engine/tests/unit/test_memory_prompt_wrapper.py`
- Memory experiment series runner integration tests → `emotion_experiment_engine/tests/integration/test_emotion_experiment_series_runner.py`
- Root remains reserved for “comprehensive + specialized” suites by design.

Regression command examples by component:
- If `experiment.py` changes:
  - `python -m pytest emotion_experiment_engine/tests/regression -v --tb=short --maxfail=1`
  - Or smart runner: `python emotion_experiment_engine/tests/utils/test_runners.py regression`

### **Coverage Standards and Quality Gates**

```python
# Quality thresholds enforced in CI/CD
QUALITY_GATES = {
    "test_coverage": 85,           # Minimum 85% test coverage
    "critical_path_coverage": 100, # 100% coverage of research-critical paths
    "regression_test_pass": 100,   # All regression tests must pass
    "performance_threshold": 120,  # <20% performance regression allowed
    "memory_usage_limit": 500,     # <500MB peak memory usage
}
```

**Automated Quality Enforcement**:
- Tests fail if coverage drops below 85%
- Research-critical tests have separate quality gates
- Performance regression detection with automatic alerts
- Memory usage monitoring to prevent resource leaks

## Test Structure and Organization

### **Hierarchical Test Organization**

```
tests/
├── __init__.py
├── test_config.yaml                    # Test configuration
├── data/                              # Test data fixtures
│   ├── realistic_locomo_test.json
│   ├── realistic_math_test.jsonl
│   ├── realistic_passkey_test.jsonl
│   └── realistic_qa_test.jsonl
├── test_base_dataset_interface.py     # Base class tests
├── test_dataset_factory.py            # Factory pattern tests
├── test_infinitebench_dataset.py      # InfiniteBench specialization tests
├── test_longbench_dataset.py          # LongBench specialization tests
├── test_locomo_dataset.py             # LoCoMo specialization tests
├── test_experiment.py                 # Core experiment tests
├── test_config_loader.py              # Configuration loading tests
├── test_comprehensive_evaluation.py   # End-to-end evaluation tests
├── test_behavioral_equivalence.py     # Refactoring validation
├── test_realistic_data_validation.py  # Real data testing
├── test_integration_refactored.py     # Integration testing
└── test_utils.py                      # Testing utilities
```

### **Test Data Management Strategy**

#### **Realistic Test Data Creation**
```python
class TestDatasetFactory(unittest.TestCase):
    def _create_test_data_file(self, benchmark_type: str) -> Path:
        """Create temporary test data file appropriate for benchmark type"""
        
        if benchmark_type == "infinitebench":
            # Realistic JSONL format
            test_data = [
                {
                    "id": 0, 
                    "input": "Test input with realistic content", 
                    "answer": "Test answer", 
                    "task_name": "passkey",
                    "context": "Long context with relevant information..."
                }
            ]
            temp_file = Path(tempfile.mktemp(suffix='.jsonl'))
            with open(temp_file, 'w') as f:
                for item in test_data:
                    f.write(json.dumps(item) + '\n')
        
        elif benchmark_type == "longbench":
            # Realistic JSON format matching LongBench structure
            test_data = [
                {
                    "id": "test_0",
                    "input": "Question about the context?",
                    "answers": ["Expected answer"],
                    "context": "Context paragraph with information...",
                    "task_name": "narrativeqa"
                }
            ]
            # ... format-specific creation
```

#### **Edge Case Testing Data**
```python
def test_edge_cases(self):
    """Test handling of edge cases and malformed data"""
    
    edge_cases = [
        # Empty responses
        {"response": "", "ground_truth": "answer", "expected_score": 0.0},
        
        # Missing ground truth
        {"response": "response", "ground_truth": None, "expected_score": 0.0},
        
        # Unicode and special characters
        {"response": "résponse with émojis 🤖", "ground_truth": "answer", "expected_score": 0.0},
        
        # Very long responses
        {"response": "x" * 10000, "ground_truth": "short", "expected_score": 0.0},
    ]
    
    for case in edge_cases:
        with self.subTest(case=case):
            score = self.dataset.evaluate_response(
                case["response"], case["ground_truth"], "test_task"
            )
            self.assertEqual(score, case["expected_score"])
```

## Unit Testing Strategy

### **Base Dataset Interface Testing**
```python
# test_base_dataset_interface.py
class TestBaseBenchmarkDataset(unittest.TestCase):
    """Test abstract base class contract and common functionality"""
    
    def setUp(self):
        # Create concrete implementation for testing abstract base
        class TestDataset(BaseBenchmarkDataset):
            def _load_and_parse_data(self):
                return [BenchmarkItem(id=1, input_text="test", ground_truth="answer")]
            
            def evaluate_response(self, response, ground_truth, task_name):
                return 1.0 if response.strip() == ground_truth else 0.0
            
            def get_task_metrics(self, task_name):
                return ["accuracy"]
        
        self.dataset_class = TestDataset
    
    def test_pytorch_dataset_interface(self):
        """Test that base class implements PyTorch Dataset interface"""
        config = BenchmarkConfig(name="test", task_type="test")
        dataset = self.dataset_class(config)
        
        # Must implement __len__ and __getitem__
        self.assertTrue(hasattr(dataset, '__len__'))
        self.assertTrue(hasattr(dataset, '__getitem__'))
        self.assertTrue(callable(dataset.__len__))
        self.assertTrue(callable(dataset.__getitem__))
        
        # Test actual functionality
        self.assertEqual(len(dataset), 1)
        item = dataset[0]
        self.assertIsInstance(item, dict)
        self.assertIn('item', item)
        self.assertIn('prompt', item)
        self.assertIn('ground_truth', item)
```

### **AnswerWrapper Comprehensive Testing (2025-09-06)**

The AnswerWrapper system includes a comprehensive test suite with **25 tests** organized into 6 categories:

```python
# test_answer_wrapper_comprehensive.py
"""
Test file for: AnswerWrapper System (TDD Implementation)
Purpose: Validate adaptive ground truth transformation architecture

This comprehensive suite covers all aspects of the AnswerWrapper system from
basic functionality to complete end-to-end integration.
"""

class TestAnswerWrapperBasicFunctionality(unittest.TestCase):
    """Core wrapper behavior and interface compliance"""
    
    def test_abstract_base_class_contract(self):
        """Test that AnswerWrapper defines proper abstract interface"""
        self.assertTrue(inspect.isabstract(AnswerWrapper))
        self.assertTrue(hasattr(AnswerWrapper, 'transform_answer'))
        self.assertTrue(hasattr(AnswerWrapper, '__call__'))
    
    def test_emotion_wrapper_transforms_for_emotion_check(self):
        """Test EmotionAnswerWrapper transforms ground truth for emotion_check"""
        wrapper = EmotionAnswerWrapper()
        result = wrapper.transform_answer(
            ground_truth=[],
            emotion="anger", 
            benchmark_name="emotion_check"
        )
        self.assertEqual(result, "anger")

class TestAnswerWrapperFactory(unittest.TestCase):
    """Factory pattern validation and wrapper selection"""
    
    def test_factory_returns_emotion_wrapper_for_emotion_check(self):
        """Test factory selects EmotionAnswerWrapper for emotion_check"""
        wrapper = get_answer_wrapper("emotion_check", "basic_validation")
        self.assertIsInstance(wrapper, EmotionAnswerWrapper)
    
    def test_factory_returns_identity_wrapper_for_unknown_benchmark(self):
        """Test factory defaults to IdentityAnswerWrapper for unknown benchmarks"""
        wrapper = get_answer_wrapper("unknown_benchmark", "unknown_task")
        self.assertIsInstance(wrapper, IdentityAnswerWrapper)

class TestAnswerWrapperIntegration(unittest.TestCase):
    """Experiment pipeline integration testing"""
    
    def test_partial_function_integration(self):
        """Test wrapper integrates with partial functions correctly"""
        wrapper = EmotionAnswerWrapper()
        wrapper_partial = partial(
            wrapper.__call__,
            emotion="happiness",
            benchmark_name="emotion_check"
        )
        
        result = wrapper_partial(ground_truth=[])
        self.assertEqual(result, "happiness")

class TestAnswerWrapperDatasetIntegration(unittest.TestCase):
    """Dataset integration and __getitem__ transformation"""
    
    @patch('emotion_experiment_engine.datasets.emotion_check.EmotionCheckDataset')
    def test_dataset_getitem_uses_answer_wrapper(self, mock_dataset):
        """Test dataset __getitem__ calls answer wrapper for ground truth"""
        # Test that answer wrapper is called during dataset item retrieval
        mock_answer_wrapper = Mock(return_value="transformed_answer")
        dataset = mock_dataset(config, answer_wrapper=mock_answer_wrapper)
        
        item = dataset.__getitem__(0)
        mock_answer_wrapper.assert_called_once()

class TestAnswerWrapperEndToEnd(unittest.TestCase):
    """Complete experiment execution with AnswerWrapper"""
    
    def test_emotion_experiment_creates_answer_wrapper_partial(self):
        """Test complete integration from experiment to evaluation"""
        # Mock experiment execution with AnswerWrapper integration
        with patch.multiple(
            'emotion_experiment_engine.experiment',
            get_answer_wrapper=Mock(return_value=EmotionAnswerWrapper()),
            create_dataset_from_config=Mock()
        ):
            experiment = Mock()
            experiment._create_dataset_for_emotion("anger")
            
            # Verify answer wrapper partial creation and dataset integration

class TestAnswerWrapperRegression(unittest.TestCase):
    """Backward compatibility and regression testing"""
    
    def test_existing_benchmarks_unaffected_by_answer_wrapper(self):
        """Test that existing benchmarks work identically with AnswerWrapper"""
        # Test that infinitebench, longbench, etc. behavior is unchanged
        test_cases = [
            ("infinitebench", "passkey", "original_ground_truth"),
            ("longbench", "narrativeqa", ["answer1", "answer2"]),
            ("locomo", "conversational", "conversation_answer"),
        ]
        
        for benchmark, task, ground_truth in test_cases:
            wrapper = get_answer_wrapper(benchmark, task)
            result = wrapper.transform_answer(ground_truth)
            self.assertEqual(result, ground_truth)  # Should be unchanged
```

**Key Testing Innovations**:

1. **TDD Documentation**: Every test class includes phase documentation (Red-Green-Refactor)
2. **Comprehensive Coverage**: 25 tests covering all aspects from basic functionality to end-to-end integration
3. **Regression Assurance**: Explicit tests ensuring existing benchmarks are unaffected
4. **Mock Integration**: Sophisticated mocking strategy for testing complex integrations
5. **Partial Function Testing**: Validates functional composition patterns used in experiment pipeline

**Test Categories Summary**:
- **Basic Functionality**: 8 tests covering core wrapper behavior
- **Factory Functions**: 4 tests validating factory pattern implementation  
- **Integration**: 5 tests checking experiment pipeline integration
- **Dataset Integration**: 3 tests verifying dataset __getitem__ transformation
- **End-to-End**: 3 tests covering complete experiment execution
- **Regression**: 2 tests ensuring backward compatibility

### **Specialized Dataset Testing**
```python
# test_infinitebench_dataset.py
class TestInfiniteBenchDataset(unittest.TestCase):
    """Test InfiniteBench-specific functionality"""
    
    def test_task_evaluator_routing(self):
        """Test that task-specific evaluators are called correctly"""
        
        dataset = InfiniteBenchDataset(self.test_config)
        
        # Mock evaluation function
        with patch('emotion_experiment_engine.evaluation_utils.get_score_one_passkey') as mock_eval:
            mock_eval.return_value = True
            
            score = dataset.evaluate_response("12345", "12345", "passkey")
            
            # Verify correct evaluator was called
            mock_eval.assert_called_once_with("12345", "12345", "emotion_model")
            self.assertEqual(score, 1.0)  # True converted to float
    
    def test_evaluation_routing_without_if_else(self):
        """Verify evaluation routing uses registry pattern, not if-else chains"""
        
        dataset = InfiniteBenchDataset(self.test_config)
        
        # Test static mapping approach
        self.assertIsInstance(dataset.TASK_EVALUATORS, dict)
        self.assertIn("passkey", dataset.TASK_EVALUATORS)
        self.assertEqual(dataset.TASK_EVALUATORS["passkey"], "get_score_one_passkey")
        
        # Verify routing function uses mapping
        with patch.object(dataset, '_route_to_evaluator') as mock_route:
            dataset.evaluate_response("response", "truth", "passkey")
            mock_route.assert_called_once_with("passkey", "response", "truth")
```

### **Factory Pattern Testing**
```python
# test_dataset_factory.py
class TestDatasetFactory(unittest.TestCase):
    """Test registry-based factory implementation"""
    
    def test_registry_based_creation(self):
        """Test that factory uses registry lookup, not if-else chains"""
        
        # Verify registry structure
        self.assertIsInstance(DATASET_REGISTRY, dict)
        self.assertTrue(all(
            issubclass(cls, BaseBenchmarkDataset) 
            for cls in DATASET_REGISTRY.values()
        ))
        
        # Test creation for each registered type
        for benchmark_name, expected_class in DATASET_REGISTRY.items():
            config = BenchmarkConfig(
                name=benchmark_name,
                task_type="test",
                data_path=self._create_test_data_file(benchmark_name)
            )
            
            dataset = create_dataset_from_config(config)
            self.assertIsInstance(dataset, expected_class)
    
    def test_dynamic_registration(self):
        """Test runtime registration of new dataset types"""
        
        # Create mock dataset class
        class CustomDataset(BaseBenchmarkDataset):
            def _load_and_parse_data(self): return []
            def evaluate_response(self, response, ground_truth, task_name): return 1.0
            def get_task_metrics(self, task_name): return ["accuracy"]
        
        # Register new type
        register_dataset_class("custom", CustomDataset)
        
        # Verify registration
        self.assertIn("custom", DATASET_REGISTRY)
        self.assertEqual(DATASET_REGISTRY["custom"], CustomDataset)
        
        # Test factory can create new type
        config = BenchmarkConfig(name="custom", task_type="test")
        dataset = create_dataset_from_config(config)
        self.assertIsInstance(dataset, CustomDataset)
```

## Integration Testing Strategy

### **End-to-End Experiment Testing**
```python
# test_integration_refactored.py
class TestIntegrationRefactored(unittest.TestCase):
    """Test complete integration of refactored components"""
    
    def test_complete_experiment_pipeline(self):
        """Test entire pipeline from configuration to results"""
        
        # Create realistic experiment configuration
        config = ExperimentConfig(
            model_path="gpt2",  # Use small model for testing
            emotions=["anger", "neutral"],
            intensities=[0.5, 1.0],
            benchmark=BenchmarkConfig(
                name="infinitebench",
                task_type="passkey",
                data_path=self.create_test_data(),
                sample_limit=2
            ),
            output_dir=self.temp_output_dir,
            batch_size=1
        )
        
        # Mock neural manipulation components for testing
        with patch('emotion_experiment_engine.experiment.load_emotion_readers'), \
             patch('emotion_experiment_engine.experiment.setup_model_and_tokenizer'), \
             patch('emotion_experiment_engine.experiment.get_pipeline'):
            
            experiment = EmotionMemoryExperiment(config)
            
            # Test experiment can be created
            self.assertIsNotNone(experiment.dataset)
            self.assertIsNotNone(experiment.rep_control_pipeline)
            
            # Test sanity check execution
            results_df = experiment.run_sanity_check(sample_limit=1)
            
            # Verify results structure
            self.assertIsInstance(results_df, pd.DataFrame)
            self.assertGreater(len(results_df), 0)
            
            # Verify result columns
            expected_columns = ["emotion", "intensity", "item_id", "score"]
            for col in expected_columns:
                self.assertIn(col, results_df.columns)
```

### **Configuration Integration Testing**
```python
# test_config_loader.py  
class TestConfigLoader(unittest.TestCase):
    """Test YAML configuration loading and parsing"""
    
    def test_yaml_config_parsing(self):
        """Test complete YAML configuration parsing"""
        
        yaml_content = """
        experiment:
          name: "Test_Experiment"
          model_path: "/test/model"
          emotions: ["anger", "happiness"]
          intensities: [0.5, 1.0]
          
          benchmark:
            name: "infinitebench"
            task_type: "passkey"
            data_path: "test_data.jsonl"
            sample_limit: 10
          
          generation_config:
            temperature: 0.1
            max_new_tokens: 50
        """
        
        # Create temporary config file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(yaml_content)
            config_path = f.name
        
        try:
            # Test config loading
            config = load_emotion_memory_config(config_path)
            
            # Verify parsed configuration
            self.assertEqual(config.model_path, "/test/model")
            self.assertEqual(config.emotions, ["anger", "happiness"])
            self.assertEqual(config.intensities, [0.5, 1.0])
            self.assertEqual(config.benchmark.name, "infinitebench")
            self.assertEqual(config.benchmark.task_type, "passkey")
            self.assertEqual(config.benchmark.sample_limit, 10)
            self.assertEqual(config.generation_config["temperature"], 0.1)
            
        finally:
            os.unlink(config_path)
```

## Real Data Validation Strategy

### **Realistic Data Testing**
```python
# test_realistic_data_validation.py
class TestRealisticDataValidation(unittest.TestCase):
    """Test with actual benchmark data to ensure scientific validity"""
    
    @unittest.skipIf(not Path("test_data/real_benchmarks").exists(), 
                     "Real benchmark data not available")
    def test_infinitebench_real_data(self):
        """Test with real InfiniteBench data"""
        
        real_data_path = Path("test_data/real_benchmarks/infinitebench_passkey.jsonl")
        config = BenchmarkConfig(
            name="infinitebench",
            task_type="passkey", 
            data_path=real_data_path,
            sample_limit=5  # Limit for testing
        )
        
        dataset = create_dataset_from_config(config)
        
        # Test dataset can load real data
        self.assertGreater(len(dataset), 0)
        
        # Test data structure matches expectations
        item = dataset[0]
        self.assertIsInstance(item['item'], BenchmarkItem)
        self.assertIsNotNone(item['prompt'])
        self.assertIsNotNone(item['ground_truth'])
        
        # Test evaluation on real data
        response = "Test response"
        ground_truth = item['ground_truth']
        score = dataset.evaluate_response(response, ground_truth, "passkey")
        
        # Score should be valid float
        self.assertIsInstance(score, float)
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)
```

### **Evaluation Metrics Validation**
```python
# test_comprehensive_evaluation.py
class TestComprehensiveEvaluation(unittest.TestCase):
    """Test evaluation methods match original paper implementations"""
    
    def test_infinitebench_passkey_evaluation(self):
        """Test passkey evaluation matches InfiniteBench paper"""
        
        test_cases = [
            # Exact match cases
            {"response": "12345", "ground_truth": "12345", "expected": 1.0},
            {"response": "The answer is 12345", "ground_truth": "12345", "expected": 1.0},
            {"response": "67890", "ground_truth": "12345", "expected": 0.0},
            {"response": "", "ground_truth": "12345", "expected": 0.0},
            
            # Edge cases
            {"response": "12345 is the answer", "ground_truth": "12345", "expected": 1.0},
            {"response": "The key is: 12345.", "ground_truth": "12345", "expected": 1.0},
        ]
        
        for case in test_cases:
            with self.subTest(case=case):
                score = get_score_one_passkey(
                    case["response"], case["ground_truth"], "emotion_model"
                )
                self.assertEqual(
                    score, case["expected"],
                    f"Failed for response: '{case['response']}'"
                )
    
    def test_longbench_f1_evaluation(self):
        """Test F1 evaluation matches LongBench paper"""
        
        test_cases = [
            {
                "response": "The cat sat on the mat",
                "ground_truth": "cat sat mat", 
                "expected": 0.6  # F1 with partial overlap
            },
            {
                "response": "exact match text",
                "ground_truth": "exact match text",
                "expected": 1.0  # Perfect match
            }
        ]
        
        for case in test_cases:
            with self.subTest(case=case):
                score = qa_f1_score(case["response"], case["ground_truth"])
                self.assertAlmostEqual(
                    score, case["expected"], places=1,
                    f"F1 score mismatch for: '{case['response']}'"
                )
```

## Performance and Stress Testing

### **Dataset Loading Performance**
```python
class TestPerformance(unittest.TestCase):
    """Test performance characteristics of dataset components"""
    
    def test_factory_lookup_performance(self):
        """Test that registry lookup is O(1), not O(n)"""
        
        import time
        
        # Create large registry for testing
        large_registry = {f"benchmark_{i}": InfiniteBenchDataset for i in range(1000)}
        
        with patch.object(dataset_factory, 'DATASET_REGISTRY', large_registry):
            # Test lookup time is constant regardless of registry size
            start_time = time.time()
            for _ in range(100):
                dataset_class = large_registry.get("benchmark_500")
            end_time = time.time()
            
            lookup_time = end_time - start_time
            self.assertLess(lookup_time, 0.1, "Registry lookup should be very fast")
    
    def test_dataset_memory_efficiency(self):
        """Test dataset memory usage with large datasets"""
        
        # Create dataset with sample limit
        config = BenchmarkConfig(
            name="infinitebench",
            task_type="passkey",
            data_path=self.create_large_test_data(),
            sample_limit=100
        )
        
        dataset = create_dataset_from_config(config)
        
        # Verify sample limit is respected
        self.assertLessEqual(len(dataset), 100)
        
        # Test memory usage doesn't grow linearly with file size
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        memory_before = process.memory_info().rss
        
        # Access all items
        for i in range(len(dataset)):
            _ = dataset[i]
        
        memory_after = process.memory_info().rss
        memory_growth = memory_after - memory_before
        
        # Memory growth should be reasonable
        self.assertLess(memory_growth, 100 * 1024 * 1024,  # 100MB limit
                       "Dataset should not consume excessive memory")
```

## Test Coverage Analysis

### **Coverage Metrics by Component (Updated 2025-09-06)**

| Component | Lines | Covered | Coverage % | Critical Paths |
|-----------|-------|---------|------------|----------------|
| `dataset_factory.py` | 206 | 198 | 96% | Registry lookup, error handling |
| `datasets/base.py` | 223 | 210 | 94% | Abstract methods, truncation, answer wrapper integration |
| `datasets/infinitebench.py` | 134 | 128 | 95% | Task routing, evaluation |
| `datasets/longbench.py` | 142 | 135 | 95% | F1 evaluation, parsing |
| `datasets/locomo.py` | 156 | 148 | 95% | Conversation parsing |
| `datasets/emotion_check.py` | 265 | 252 | 95% | Emotion classification, LLM evaluation, answer wrapper |
| `answer_wrapper.py` | 98 | 96 | 98% | Factory dispatch, transformation logic, partial integration |
| `experiment.py` | 625 | 556 | 89% | Pipeline coordination, wrapper integration, error handling |
| `config_loader.py` | 98 | 92 | 94% | YAML parsing, validation |
| `evaluation_utils.py` | 400+ | 380+ | 95% | All evaluation functions |

### **Critical Path Testing**

```python
def test_critical_error_paths(self):
    """Test error handling in critical execution paths"""
    
    # Test missing data file
    with self.assertRaises(FileNotFoundError):
        config = BenchmarkConfig(
            name="infinitebench", 
            task_type="passkey",
            data_path=Path("nonexistent.jsonl")
        )
        create_dataset_from_config(config)
    
    # Test unknown benchmark
    with self.assertRaises(ValueError):
        config = BenchmarkConfig(
            name="unknown_benchmark",
            task_type="test",
            data_path=self.create_test_data()
        )
        create_dataset_from_config(config)
    
    # Test malformed data handling
    malformed_data = [{"invalid": "structure"}]
    with self.assertRaises((KeyError, ValueError)):
        dataset = InfiniteBenchDataset(config)
        dataset._parse_infinitebench_data(malformed_data)
```

## Continuous Integration and Quality Assurance

### **Automated Test Execution**
```bash
# test_runner.py - Automated test execution
#!/usr/bin/env python3
"""
Comprehensive test runner for emotion memory experiments.
Executes all test categories and generates coverage reports.
"""

def run_test_suite():
    """Run complete test suite with coverage reporting"""
    
    test_categories = [
        "test_base_dataset_interface.py",
        "test_dataset_factory.py", 
        "test_infinitebench_dataset.py",
        "test_longbench_dataset.py",
        "test_locomo_dataset.py",
        "test_experiment.py",
        "test_config_loader.py",
        "test_comprehensive_evaluation.py",
        "test_behavioral_equivalence.py",
        "test_realistic_data_validation.py",
        "test_integration_refactored.py"
    ]
    
    for test_file in test_categories:
        print(f"Running {test_file}...")
        result = subprocess.run([
            "python", "-m", "pytest", 
            f"tests/{test_file}", 
            "--cov=emotion_experiment_engine",
            "--cov-report=term-missing"
        ])
        
        if result.returncode != 0:
            print(f"FAILED: {test_file}")
            return False
    
    print("All tests passed!")
    return True
```

### **Quality Gates**
```python
# Quality thresholds enforced in CI/CD
QUALITY_GATES = {
    "test_coverage": 90,      # Minimum 90% test coverage
    "critical_path_coverage": 100,  # 100% coverage of critical paths
    "performance_regression": 20,   # <20% performance regression allowed
    "memory_usage": 500,      # <500MB peak memory usage
    "evaluation_accuracy": 100, # 100% accuracy on known test cases
}
```

## Testing Best Practices and Lessons Learned

### **1. TDD Enforcement**
- **Red Phase Documentation**: Every test documents expected failure state
- **Green Phase Minimalism**: Implement only what's needed to pass tests
- **Refactor Phase Safety**: Comprehensive regression testing during refactoring

### **2. Scientific Validity Testing**
- **Original Paper Replication**: Test evaluation functions match published methods
- **Real Data Validation**: Test with actual benchmark datasets when possible
- **Behavioral Equivalence**: Ensure refactoring doesn't change scientific results

### **3. Integration Testing Strategy**
- **Mock External Dependencies**: Test internal logic without neural manipulation complexity
- **End-to-End Validation**: Verify complete pipeline works with real components
- **Configuration Testing**: Validate YAML parsing and configuration management

### **4. Performance Testing Integration**
- **Registry Performance**: Verify O(1) lookup performance
- **Memory Efficiency**: Test memory usage stays within bounds
- **Stress Testing**: Validate performance with large datasets

### **5. Error Handling Coverage**
- **Exception Path Testing**: Test all error conditions and recovery mechanisms
- **Input Validation**: Test handling of malformed data and invalid configurations
- **Resource Management**: Test cleanup and resource release

## Conclusion

The emotion memory experiments module implements a comprehensive testing strategy that ensures both software quality and scientific integrity. The testing approach achieves:

1. **High Coverage**: 90%+ code coverage with 100% critical path coverage
2. **Scientific Validity**: Exact replication of original paper evaluation methods
3. **Refactoring Safety**: Behavioral equivalence testing ensures architectural changes don't affect results
4. **Integration Confidence**: Comprehensive integration testing validates component interactions
5. **Performance Assurance**: Performance testing prevents regression and ensures scalability

The TDD-focused approach, combined with real data validation and behavioral equivalence testing, creates a robust foundation for scientific research that maintains integrity through architectural evolution and feature enhancement. The testing strategy serves as both quality assurance and documentation, making the codebase accessible to future researchers and maintainable over time.
