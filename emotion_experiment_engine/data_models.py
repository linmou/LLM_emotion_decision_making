"""
Data models for emotion memory experiments.
Defines standard formats for results, configurations, and data structures.
"""

import glob
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

DEFAULT_VLLM_MAX_MODEL_LEN = 8192
DEFAULT_VLLM_MAX_NUM_SEQS_CAP = 32


@dataclass
class ResultRecord:
    """Standard format for individual experiment results"""

    emotion: str
    intensity: float
    item_id: Union[int, str]
    task_name: str
    prompt: str
    response: str
    ground_truth: Any
    score: Optional[float]
    repeat_id: int
    metadata: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


@dataclass
class BenchmarkConfig:
    name: str
    task_type: str  # e.g., 'passkey', 'kv_retrieval', 'longbook_qa_eng'
    data_path: Optional[Path]  # Auto-generated if None
    base_data_dir: Optional[str]
    sample_limit: Optional[int]
    augmentation_config: Optional[
        Dict[str, str]
    ]  # Custom prefix/suffix for context and answer marking
    # Supports: {"prefix": "text", "suffix": "text"} for manual mode
    # or {"method": "adaptive"} for adaptive emotion-based augmentation

    # Context truncation settings (dataset-specific)
    enable_auto_truncation: bool  # Enable automatic context truncation
    truncation_strategy: str  # "right" or "left" (via tokenizer)
    preserve_ratio: float  # Ratio of max_model_len to use for context

    llm_eval_config: Optional[Dict[str, Any]]

    def discover_datasets_by_pattern(
        self, base_data_dir: Optional[str] = None
    ) -> List[str]:
        """
        Discover task types matching the regex pattern in task_type.

        This method scans for files matching the pattern {benchmark_name}_{task_type}.jsonl
        and filters task types using regex pattern matching. Uses regex.search() to allow
        pattern matching anywhere in the task type name, not just at the beginning.

        Args:
            base_data_dir: Base directory for memory benchmark data

        Returns:
            List of task types matching the regex pattern, sorted alphabetically

        Examples:
            - task_type='.*' -> ['narrativeqa', 'qasper'] (matches all tasks)
            - task_type='.*qa.*' -> ['narrativeqa', 'multifieldqa_en'] (contains 'qa' anywhere)
            - task_type='pass.*' -> ['passkey'] (starts with 'pass')
            - task_type='retrieval' -> ['passage_retrieval_en', 'kv_retrieval'] (contains 'retrieval')
            - task_type='.*retrieval.*' -> ['passage_retrieval_en', 'kv_retrieval'] (explicit wildcards)
            - task_type='qa$' -> ['narrativeqa'] (ends with 'qa')

        Notes:
            - Uses regex.search() instead of regex.match() to allow pattern matching
              anywhere in the task type name, not just from the beginning
            - Empty list returned if no files match the pattern
            - Raises ValueError for invalid regex patterns
        """
        if base_data_dir is None:
            assert self.base_data_dir is not None, "base_data_dir is required"
            base_data_dir = self.base_data_dir

        base_path = Path(base_data_dir)
        glob_pattern = str(base_path / f"{self.name}_*.jsonl")

        # Find all files for this benchmark
        all_files = glob.glob(glob_pattern)

        # Extract task types and filter by regex pattern
        task_types = []
        try:
            pattern = self.task_type.strip()

            # Allow shell-style globs in configs (e.g., "*gen*") by translating
            # to a proper regex. Only do this when the pattern looks like a glob
            # and not an explicit regex (heuristic keeps behavior predictable).
            def _looks_like_glob(p: str) -> bool:
                has_glob = ("*" in p) or ("?" in p)
                has_regex_syntax = any(ch in p for ch in [".", "^", "$", "(", ")", "[", "]", "|", "+", "\\"])
                return has_glob and not has_regex_syntax

            if pattern == "*":
                # Fast-path: match everything
                regex_pattern = re.compile(r".*")
            elif _looks_like_glob(pattern):
                import fnmatch

                # fnmatch.translate returns a regex string with \Z anchors; compile directly
                regex_pattern = re.compile(fnmatch.translate(pattern))
            else:
                # Treat as explicit regex or literal token
                regex_pattern = re.compile(pattern)

            for file_path in all_files:
                filename = Path(file_path).stem  # Remove .jsonl extension
                prefix = f"{self.name}_"
                if filename.startswith(prefix):
                    task_type = filename[len(prefix) :]
                    if regex_pattern.search(task_type):
                        task_types.append(task_type)
        except re.error as e:
            raise ValueError(f"Invalid regex pattern '{self.task_type}': {str(e)}")

        return sorted(task_types)

    def get_data_path(self, base_data_dir: Optional[str] = None) -> Path:
        """
        Get the data path for this benchmark. Auto-generates if not set.

        Args:
            base_data_dir: Base directory for memory benchmark data

        Returns:
            Path to the benchmark data file

        Examples:
            - name='longbench', task_type='narrativeqa' -> data/memory_benchmarks/longbench_narrativeqa.jsonl
            - name='infinitebench', task_type='passkey' -> data/memory_benchmarks/infinitebench_passkey.jsonl
        """
        if self.data_path is not None:
            return self.data_path

        if base_data_dir is None:
            assert self.base_data_dir is not None, "base_data_dir is required"
            base_data_dir = self.base_data_dir

        # Auto-generate path based on naming convention
        filename = f"{self.name}_{self.task_type}.jsonl"
        return Path(base_data_dir) / filename


@dataclass
class VLLMLoadingConfig:
    """Flexible vLLM model loading configuration"""

    # All parameters are required - no defaults for safety
    model_path: str  # Model name or path to load
    gpu_memory_utilization: float
    tensor_parallel_size: Optional[int]  # None for auto-detect
    max_model_len: int
    enforce_eager: bool
    quantization: Optional[str]  # 'awq' for AWQ models
    trust_remote_code: bool
    dtype: str  # Model dtype: 'float16', 'bfloat16', 'float32'
    seed: int
    disable_custom_all_reduce: bool
    additional_vllm_kwargs: Dict[str, Any]

    def to_vllm_kwargs(self) -> Dict[str, Any]:
        """Convert to vLLM constructor arguments"""
        base_kwargs = {
            "model": self.model_path,
            "gpu_memory_utilization": self.gpu_memory_utilization,
            "tensor_parallel_size": self.tensor_parallel_size,
            "max_model_len": self.max_model_len,
            "enforce_eager": self.enforce_eager,
            "trust_remote_code": self.trust_remote_code,
            "dtype": self.dtype,
            "seed": self.seed,
            "disable_custom_all_reduce": self.disable_custom_all_reduce,
        }

        # Add quantization if specified
        if self.quantization:
            base_kwargs["quantization"] = self.quantization

        # vLLM v0.11+ blocks function serialization for RPC by default; our
        # rep-control hook uses string RPC methods implemented via this worker
        # extension. Allow override via additional_vllm_kwargs.
        default_worker_extension = (
            "neuro_manipulation.repe.vllm_worker_extension.NMRepControlWorkerExtension"
        )

        additional = dict(self.additional_vllm_kwargs)
        additional.setdefault("worker_extension_cls", default_worker_extension)

        # Merge with additional kwargs, allowing override
        return {**base_kwargs, **additional}


@dataclass
class ExperimentConfig:
    """Configuration for the emotion memory experiment"""

    model_path: str
    emotions: List[str]
    intensities: List[float]
    benchmark: BenchmarkConfig
    output_dir: str
    batch_size: int  # Number of items to process per batch for memory efficiency
    generation_config: Optional[Dict[str, Any]]
    loading_config: Optional[VLLMLoadingConfig]  # vLLM loading configuration

    repe_eng_config: Optional[Dict[str, Any]]

    # Pipeline settings (always enabled with DataLoader)
    max_evaluation_workers: int  # Number of evaluation worker threads
    pipeline_queue_size: int  # Max queued batches (controls memory usage)
    defer_evaluation: bool


@dataclass
class BenchmarkItem:
    """Standardized format for benchmark items after loading"""

    id: Union[int, str]
    input_text: str  # The prompt/question
    context: Optional[str]  # Long context if separate
    ground_truth: Any  # Expected answer
    metadata: Optional[Dict[str, Any]]  # Task-specific data


# Default generation config matching emotion_game_experiment.py
DEFAULT_GENERATION_CONFIG = {
    "temperature": 0.1,
    "max_new_tokens": 100,
    "do_sample": False,
    "top_p": 0.9,
    "repetition_penalty": 1.0,
    "top_k": -1,  # -1 means no top_k filtering
    "min_p": 0.0,
    "presence_penalty": 0.0,
    "frequency_penalty": 0.0,
    "enable_thinking": False,  # Qwen thinking mode support
}
