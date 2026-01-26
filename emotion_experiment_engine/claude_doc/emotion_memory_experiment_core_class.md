# EmotionExperiment - Core Orchestrator Class

## Overview

The `EmotionExperiment` class serves as the central orchestrator for emotion research experiments. Located in `emotion_experiment_engine/experiment.py`, this class follows the established pattern from `emotion_game_experiment.py`

## Class Architecture

### **Inheritance and Composition**
```python
class EmotionExperiment:
    def __init__(self, config: ExperimentConfig)
    
    # Core Components
    self.model: LLM                           # vLLM inference engine
    self.tokenizer: Tokenizer                 # Model-specific tokenizer
    self.emotion_rep_readers: Dict            # Emotion activation readers
    self.rep_control_pipeline: Pipeline      # RepE control pipeline
    self.dataset: BaseBenchmarkDataset       # Benchmark dataset
```

### **Key Responsibilities**
1. **Neural Emotion Activation**: Setup and control of RepE-based emotion manipulation
2. **Experiment Orchestration**: Coordinate multi-emotion, multi-intensity experiments
3. **Batch Processing**: Efficient DataLoader-based inference pipeline
4. **Result Management**: Comprehensive logging, evaluation, and statistical analysis

## Implementation Deep Dive

### **Initialization Process**

#### **1. Configuration and Logging Setup**
```python
def __init__(self, config: ExperimentConfig):
    self.config = config
    self.generation_config = config.generation_config or DEFAULT_GENERATION_CONFIG
    
    # Setup comprehensive logging
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = f"logs/emotion_memory_experiment_{timestamp}.log"
    
    # Configure both file and console logging
    self.logger = logging.getLogger(__name__)
    # ... detailed logging configuration
```

#### **2. Neural Manipulation Pipeline Setup**
```python
# Load RepE configuration
self.repe_config = get_repe_eng_config(
    config.model_path, 
    yaml_config=config.repe_eng_config
)

# Two-stage model loading process:
# 1. HuggingFace model for emotion reader extraction
self.model, self.tokenizer, self.prompt_format, processor = (
    setup_model_and_tokenizer(self.loading_config, from_vllm=False)
)

# Extract emotion readers using HF model
self.emotion_rep_readers = load_emotion_readers(
    self.repe_config, self.model, self.tokenizer, 
    self.hidden_layers, processor, self.enable_thinking
)

# 2. vLLM model for efficient inference
self.model, self.tokenizer, self.prompt_format, _ = (
    setup_model_and_tokenizer(self.loading_config, from_vllm=True)
)
```

> **Neutral-only mode:** When the configuration provides an empty `emotions` list the constructor flags the run as neutral-only, skips `load_emotion_readers`, and still rebuilds the vLLM pipeline. Downstream inference passes `activations=None`, so the experiment executes a pure baseline sweep without loading any emotion vectors.

#### **3. RepE Control Pipeline Initialization**
```python
# Setup representation control pipeline
self.rep_control_pipeline = get_pipeline(
    "rep-control-vllm" if self.is_vllm else "rep-control",
    model=self.model,
    tokenizer=self.tokenizer,
    layers=self.hidden_layers[
        len(self.hidden_layers) // 3 : 2 * len(self.hidden_layers) // 3
    ],  # Use middle third of model layers
    block_name=self.repe_config["block_name"],
    control_method=self.repe_config["control_method"],
)
```

#### **4. Dataset Integration and Context Truncation**
```python
# Assemble benchmark components via registry
prompt_fn, answer_fn, test_dataset = create_benchmark_components(
    benchmark_name=config.benchmark.name,
    task_type=config.benchmark.task_type,
    config=config.benchmark,
    prompt_format=self.prompt_format,
    enable_thinking=self.enable_thinking,
    augmentation_config=config.benchmark.augmentation_config,
)

self.memory_prompt_wrapper_partial = prompt_fn
self.answer_wrapper_partial = answer_fn

# Configure context truncation if enabled
if self.loading_config and self.loading_config.enable_auto_truncation:
    self.max_context_length = calculate_max_context_length(
        self.loading_config.max_model_len,
        self.loading_config.preserve_ratio,
        prompt_overhead=200,
    )
```

### **Core Experiment Execution**

#### **Main Experiment Loop**
```python
def run_experiment(self) -> pd.DataFrame:
    """Run the complete emotion memory experiment"""
    all_results = []
    
    # Test each emotion with each intensity
    for emotion in self.config.emotions:
        rep_reader = self.emotion_rep_readers[emotion]
        self.cur_emotion = emotion
        
        # Fresh DataLoader for each emotion
        data_loader = self.build_dataloader()
        
        for intensity in self.config.intensities:
            self.cur_intensity = intensity
            results = self._infer_with_activation(rep_reader, data_loader)
            all_results.extend(results)
    
    # Add neutral baseline
    neutral_results = self._infer_with_activation(rep_reader, data_loader)
    all_results.extend(neutral_results)
    
    return self._save_results(all_results)
```

#### **DataLoader Construction**
```python
def build_dataloader(self) -> DataLoader:
    """Build DataLoader using specialized dataset factory"""
    
    # Create dataset with truncation parameters
    self.dataset = create_dataset_from_config(
        self.config.benchmark,
        prompt_wrapper=self.memory_prompt_wrapper_partial,
        max_context_length=self.max_context_length,
        tokenizer=self.tokenizer,
        truncation_strategy=self.truncation_strategy
    )
    
    # Create DataLoader with custom collation
    data_loader = DataLoader(
        self.dataset, 
        collate_fn=self.dataset.collate_fn,
        batch_size=self.batch_size,
        shuffle=False
    )
    
    return data_loader
```

### **Neural Activation and Inference**

#### **Activation Vector Setup**
```python
def _infer_with_activation(self, rep_reader, data_loader) -> List[ResultRecord]:
    """Process with activations using DataLoader"""
    
    # Setup emotion activation vectors
    if self.cur_emotion == "neutral" or self.cur_intensity == 0.0:
        activations = {}
    else:
        device = torch.device("cpu") if self.is_vllm else self.model.device
        activations = {
            layer: torch.tensor(
                self.cur_intensity
                * rep_reader.directions[layer] 
                * rep_reader.direction_signs[layer]
            ).to(device).half()
            for layer in self.hidden_layers
        }
    
    return self._forward_dataloader(data_loader, activations)
```

#### **Multi-threaded Pipeline Processing**
```python
def _forward_dataloader(self, data_loader, activations: Dict) -> List[ResultRecord]:
    """Forward pass using DataLoader with pipeline parallelization"""
    
    batch_results = []
    pipeline_queue: Queue = Queue(maxsize=2)  # Memory control
    processed_futures = []
    
    def pipeline_worker():
        for i, batch in enumerate(data_loader):
            # Generate with RepE control
            control_outputs = self.rep_control_pipeline(
                batch["prompt"],
                activations=activations,
                batch_size=self.batch_size,
                **generation_params  # From config
            )
            pipeline_queue.put((i, batch, control_outputs))
        pipeline_queue.put(None)  # Sentinel
    
    # Start inference pipeline in background thread
    worker = Thread(target=pipeline_worker, name="PipelineWorker")
    worker.start()
    
    # Process results with ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=self.batch_size // 2) as executor:
        # Submit post-processing tasks while inference continues
        while True:
            item = pipeline_queue.get()
            if item is None: break
            
            batch_idx, batch, control_outputs = item
            future = executor.submit(
                self._post_process_memory_batch, 
                batch, control_outputs, batch_idx
            )
            processed_futures.append((batch_idx, future))
        
        # Collect results in order
        results_dict = {}
        for batch_idx, future in processed_futures:
            results_dict[batch_idx] = future.result()
        
        # Combine results maintaining order
        for i in sorted(results_dict.keys()):
            batch_results.extend(results_dict[i])
    
    worker.join()
    return batch_results
```

### **Post-processing and Evaluation**

#### **Batch Post-processing**
```python
def _post_process_memory_batch(
    self, batch: Dict[str, Any], control_outputs: List, batch_idx: int
) -> List[ResultRecord]:
    """Post-process memory batch using dataset evaluation methods"""
    
    results = []
    batch_prompts = batch["prompt"]
    batch_items = batch["items"]  # BenchmarkItem objects
    batch_ground_truths = batch["ground_truths"]
    
    for prompt, item, ground_truth, output in zip(
        batch_prompts, batch_items, batch_ground_truths, control_outputs
    ):
        # Extract response text
        if self.is_vllm:
            response = output.outputs[0].text.replace(prompt, "").strip()
        else:
            response = output[0]["generated_text"].replace(prompt, "").strip()
        
        # Use dataset's specialized evaluation method
        try:
            score = self.dataset.evaluate_response(
                response, ground_truth, self.config.benchmark.task_type
            )
        except Exception as e:
            self.logger.warning(f"Evaluation error for item {item.id}: {e}")
            score = 0.0
        
        # Create structured result record
        result = ResultRecord(
            emotion=self.cur_emotion or "unknown",
            intensity=self.cur_intensity or 0.0,
            item_id=item.id,
            task_name=self.config.benchmark.task_type,
            prompt=prompt,
            response=response,
            ground_truth=ground_truth,
            score=score,
            metadata={
                "benchmark": self.config.benchmark.name,
                "item_metadata": item.metadata or {},
            }
        )
        results.append(result)
    
    return results
```

### **Results Management and Analysis**

#### **Comprehensive Result Saving**
```python
def _save_results(self, results: List[ResultRecord]) -> pd.DataFrame:
    """Save experiment results with comprehensive analysis"""
    
    # Save full experiment configuration
    self._save_experiment_config()
    
    # Convert to DataFrame for analysis
    df = pd.DataFrame([result.__dict__ for result in results])
    
    # Save multiple output formats
    csv_filename = self.output_dir / "detailed_results.csv"
    df.to_csv(csv_filename, index=False)
    
    json_filename = self.output_dir / "raw_results.json"
    with open(json_filename, "w") as f:
        json.dump([result.__dict__ for result in results], f, 
                 indent=2, default=str)
    
    # Compute comprehensive summary statistics
    summary = (
        df.groupby(["emotion", "intensity"])
        .agg({"score": ["mean", "std", "count", "min", "max"]})
        .round(4)
    )
    
    summary_filename = self.output_dir / "summary_results.csv"
    summary.to_csv(summary_filename)
    
    return df
```

#### **Configuration Serialization**
```python
def _save_experiment_config(self):
    """Save complete experiment configuration for reproducibility"""
    
    config_dict = {
        "model_path": self.config.model_path,
        "emotions": self.config.emotions,
        "intensities": self.config.intensities,
        "benchmark": {
            "name": self.config.benchmark.name,
            "data_path": str(self.config.benchmark.get_data_path()),
            "task_type": self.config.benchmark.task_type,
            "sample_limit": self.config.benchmark.sample_limit,
        },
        "generation_config": self.generation_config,
        "loading_config": self._serialize_loading_config(),
        "repe_eng_config": self.repe_config,
        "runtime_info": {
            "timestamp": datetime.now().isoformat(),
            "max_context_length": self.max_context_length,
            "truncation_strategy": self.truncation_strategy,
            "hidden_layers": self.hidden_layers,
            "is_vllm": self.is_vllm,
        }
    }
    
    config_filename = self.output_dir / "experiment_config.json"
    with open(config_filename, "w") as f:
        json.dump(config_dict, f, indent=2, default=str)
```

## Key Design Decisions

### **Two-Stage Model Loading**
**Rationale**: Emotion readers require HuggingFace models for layer access, while inference benefits from vLLM's optimization.

**Implementation**:
1. Load HF model → Extract emotion readers → Cleanup memory
2. Load vLLM model → Use for inference

### **Pipeline Parallelization**
**Problem**: Sequential inference and evaluation creates bottlenecks
**Solution**: Multi-threaded pipeline with background inference and parallel post-processing

### **Memory Management**
**Strategies**:
- Queue-based pipeline with configurable size limits
- Context truncation with preserve ratios
- Batch processing with configurable batch sizes
- Automatic cleanup of intermediate results

### **Error Handling and Resilience**
**Features**:
- Comprehensive logging with timestamps and thread identification
- Graceful degradation for evaluation errors
- Experiment configuration serialization for reproducibility
- Statistical analysis even with partial failures

## Integration Patterns

### **With Dataset Factory**
```python
# Uses factory pattern for dataset creation
self.dataset = create_dataset_from_config(
    self.config.benchmark,
    prompt_wrapper=self.memory_prompt_wrapper_partial,
    # ... other parameters
)
```

### **With Neural Manipulation Framework**
```python
# Leverages existing RepE infrastructure
self.emotion_rep_readers = load_emotion_readers(...)
self.rep_control_pipeline = get_pipeline("rep-control-vllm", ...)
```

### **With PyTorch Ecosystem**
```python
# Standard PyTorch DataLoader integration
data_loader = DataLoader(
    self.dataset, 
    collate_fn=self.dataset.collate_fn,
    batch_size=self.batch_size
)
```

## Performance Characteristics

### **Scalability Features**
- **Batch Processing**: Configurable batch sizes for memory/speed tradeoff
- **Multi-threading**: Parallel evaluation reduces total experiment time
- **Memory Control**: Queue limits prevent memory overflow
- **Context Truncation**: Handles arbitrarily long contexts efficiently

### **Monitoring and Observability**
- **Comprehensive Logging**: File and console output with structured messages
- **Progress Tracking**: Batch-level progress reporting
- **Performance Metrics**: Timing information for pipeline stages
- **Error Reporting**: Detailed error context with recovery strategies

## Usage Patterns

### **Standard Experiment**
```python
from emotion_experiment_engine import EmotionMemoryExperiment
from emotion_experiment_engine.config_loader import load_emotion_memory_config

# Load from YAML
config = load_emotion_memory_config("config/emotion_memory_passkey.yaml")
experiment = EmotionMemoryExperiment(config)
results_df = experiment.run_experiment()
```

### **Quick Validation**
```python
# Run sanity check with limited samples
results_df = experiment.run_sanity_check(sample_limit=5)
```

### **Programmatic Configuration**
```python
from emotion_experiment_engine.data_models import *

exp_config = ExperimentConfig(
    model_path="/path/to/model",
    emotions=["anger", "happiness"],
    intensities=[0.5, 1.0, 1.5],
    benchmark=BenchmarkConfig(
        name="infinitebench",
        task_type="passkey",
        sample_limit=100
    ),
    output_dir="results/custom_experiment",
    batch_size=8
)

experiment = EmotionMemoryExperiment(exp_config)
results_df = experiment.run_experiment()
```

## Conclusion

The `EmotionMemoryExperiment` class represents a sophisticated orchestration system that successfully integrates multiple complex components:
- Neural emotion manipulation via RepE
- Efficient dataset processing via factory pattern
- Robust evaluation using original paper metrics
- Comprehensive result tracking and analysis

The class achieves a careful balance between performance optimization (multi-threading, batching, memory management) and scientific rigor (reproducible configurations, comprehensive logging, statistical analysis). Its design follows established patterns from the broader research framework while introducing memory benchmark-specific optimizations.
