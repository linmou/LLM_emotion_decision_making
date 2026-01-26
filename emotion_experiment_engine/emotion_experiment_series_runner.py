"""
Memory experiment series runner for running batches of emotion memory experiments.
Adapted from neuro_manipulation/experiment_series_runner.py for EmotionMemoryExperiment.
"""

import copy
import difflib
import gc

# Use dynamic import to avoid relative import issues
import json
import logging
import os
import shutil
import signal
import subprocess
import sys
import threading
import time
import traceback
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml
# Optional: only needed when downloading remote models.
# Importing transformers can transitively import torch/OpenMP, which may SIGABRT
# in sandboxed environments (e.g., missing shared memory). Defer to runtime.
AutoConfig = None  # type: ignore

# Defer heavy imports (torch/vLLM) to runtime when needed

from .data_models import (
    DEFAULT_VLLM_MAX_MODEL_LEN,
    DEFAULT_VLLM_MAX_NUM_SEQS_CAP,
    BenchmarkConfig,
    ExperimentConfig,
    VLLMLoadingConfig,
)


def _ensure_openmp_shm_compat() -> None:
    """
    Avoid hard crashes from Intel OpenMP trying to use SHM in constrained envs.

    This can otherwise abort the process with:
      OMP: Error #179: Function Can't open SHM2 failed
    """
    os.environ.setdefault("KMP_USE_SHM", "0")


class ExperimentStatus:
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class MemoryExperimentReport:
    """Manages and persists the status of memory experiments in a series

    Adapted from ExperimentReport to track memory experiment combinations of
    benchmarks, models, and emotion configurations.
    """

    def __init__(
        self,
        base_dir: Optional[str],
        experiment_series_name: str,
        report_path: Optional[str] = None,
    ):
        """
        Initialize a MemoryExperimentReport.

        If report_path is provided and exists, load it and use it for subsequent
        updates. Otherwise create a new report file under base_dir.
        """
        self.lock = threading.Lock()
        self.series_name = experiment_series_name
        self.series_config: Optional[Dict[str, Any]] = None
        self.experiments: Dict[str, Dict[str, Any]] = {}
        self.series_start_time: datetime
        self.sessions: List[Dict[str, Any]] = []
        self._active_session_id: Optional[str] = None

        if report_path:
            # Resume from existing report
            self.report_file = Path(report_path)
            if not self.report_file.exists():
                raise FileNotFoundError(
                    f"Provided report_path does not exist: {report_path}"
                )
            # Load without creating/overwriting
            self._load_report_internal()
        else:
            # Fresh report
            self.base_dir = base_dir
            self.timestamp = datetime.now().strftime("%Y%m%d_%H")
            self.report_file = Path(
                f"{base_dir}/{experiment_series_name}_{self.timestamp}_memory_experiment_report.json"
            )
            self.report_file.parent.mkdir(parents=True, exist_ok=True)
            self.series_start_time = datetime.now()
            self._save_report()

    def add_experiment(
        self,
        benchmark_name: str,
        model_name: str,
        exp_id: str,
        resolved_model_path: Optional[str] = None,
        status: str = ExperimentStatus.PENDING,
    ) -> None:
        """Add a new memory experiment to the report"""
        with self.lock:
            self.experiments[exp_id] = {
                "benchmark_name": benchmark_name,
                "model_name": model_name,
                "resolved_model_path": resolved_model_path,
                "status": status,
                "start_time": None,
                "end_time": None,
                "time_cost_seconds": None,
                "error": None,
                "output_dir": None,
                "exp_id": exp_id,
            }
            self._save_report()

    def update_experiment(self, exp_id: str, **kwargs) -> None:
        """Update experiment status and details"""
        with self.lock:
            if exp_id in self.experiments:
                self.experiments[exp_id].update(kwargs)

                # Calculate time cost if we have both start and end times
                if (
                    "start_time" in self.experiments[exp_id]
                    and "end_time" in self.experiments[exp_id]
                ):
                    start = self.experiments[exp_id]["start_time"]
                    end = self.experiments[exp_id]["end_time"]
                    if (
                        start
                        and end
                        and not self.experiments[exp_id].get("time_cost_seconds")
                    ):
                        start_dt = datetime.fromisoformat(start)
                        end_dt = datetime.fromisoformat(end)
                        time_cost = (end_dt - start_dt).total_seconds()
                        self.experiments[exp_id]["time_cost_seconds"] = time_cost

                self._save_report()

    def get_incomplete_experiments(self) -> List[Dict[str, Any]]:
        """Get list of pending experiments"""
        with self.lock:
            return [
                exp
                for exp in self.experiments.values()
                if exp["status"] != ExperimentStatus.COMPLETED
            ]

    def get_failed_experiments(self) -> List[Dict[str, Any]]:
        """Get list of failed experiments"""
        with self.lock:
            return [
                exp
                for exp in self.experiments.values()
                if exp["status"] == ExperimentStatus.FAILED
            ]

    def _save_report(self) -> None:
        """Save the report to disk"""
        with open(self.report_file, "w") as f:
            # Calculate series duration so far
            series_duration = (datetime.now() - self.series_start_time).total_seconds()

            json.dump(
                {
                    "last_updated": datetime.now().isoformat(),
                    "series_start_time": self.series_start_time.isoformat(),
                    "series_duration_seconds": series_duration,
                    "series_name": self.series_name,
                    "series_config": self.series_config,
                    "sessions": self.sessions,
                    "experiments": self.experiments,
                },
                f,
                indent=2,
            )

    def _load_report_internal(self) -> bool:
        """Load a report from disk if it exists"""
        if self.report_file.exists():
            with open(self.report_file, "r") as f:
                report_data = json.load(f)
                self.experiments = report_data.get("experiments", {})
                # Preserve original start time if available
                start_time = report_data.get("series_start_time")
                if start_time:
                    try:
                        self.series_start_time = datetime.fromisoformat(start_time)
                    except Exception:
                        self.series_start_time = datetime.now()
                else:
                    self.series_start_time = datetime.now()

                self.series_config = report_data.get("series_config")
                self.series_name = report_data.get("series_name", self.series_name)
                self.sessions = report_data.get("sessions", [])
            return True
        return False

    def load_report(self) -> bool:
        """Public wrapper to load a report from disk if it exists"""
        return self._load_report_internal()

    def attach_series_config(self, config: Dict[str, Any], series_name: Optional[str] = None) -> None:
        """Attach a snapshot of the series configuration to the report and save it."""
        with self.lock:
            if series_name:
                self.series_name = series_name
            # Store a shallow copy to avoid accidental external mutation
            self.series_config = copy.deepcopy(config)
            self._save_report()

    def start_session(
        self,
        resumed_from_report: bool,
        resume_report_path: Optional[str],
        config_source: str,
        config_changed: bool,
    ) -> str:
        """Record the start of a runner session."""
        with self.lock:
            session_id = datetime.now().isoformat(timespec="seconds")
            self._active_session_id = session_id
            self.sessions.append(
                {
                    "session_id": session_id,
                    "start_time": datetime.now().isoformat(),
                    "resumed_from_report": resumed_from_report,
                    "resume_report_path": resume_report_path,
                    "config_source": config_source,  # 'report' or 'config'
                    "config_changed": config_changed,
                    "shutdown_requested_at": None,
                    "end_time": None,
                    "end_reason": None,  # 'completed' | 'shutdown' | 'exception'
                }
            )
            self._save_report()
            return session_id

    def log_shutdown_request(self, reason: str) -> None:
        with self.lock:
            if self._active_session_id and self.sessions:
                self.sessions[-1]["shutdown_requested_at"] = datetime.now().isoformat()
                self.sessions[-1]["shutdown_reason"] = reason
                self._save_report()

    def end_session(self, reason: str) -> None:
        with self.lock:
            if self._active_session_id and self.sessions:
                self.sessions[-1]["end_time"] = datetime.now().isoformat()
                self.sessions[-1]["end_reason"] = reason
                self._active_session_id = None
                self._save_report()

    def get_summary(self) -> Dict[str, Any]:
        """Get summary of experiment statuses"""
        with self.lock:
            # Calculate total and average time costs
            completed_exps = [
                exp
                for exp in self.experiments.values()
                if exp["status"] == ExperimentStatus.COMPLETED
                and exp.get("time_cost_seconds")
            ]

            total_time_cost = (
                sum(exp["time_cost_seconds"] for exp in completed_exps)
                if completed_exps
                else 0
            )
            avg_time_cost = (
                total_time_cost / len(completed_exps) if completed_exps else 0
            )

            # Calculate series duration so far
            series_duration = (datetime.now() - self.series_start_time).total_seconds()

            summary = {
                "total": len(self.experiments),
                "pending": sum(
                    1
                    for exp in self.experiments.values()
                    if exp["status"] == ExperimentStatus.PENDING
                ),
                "running": sum(
                    1
                    for exp in self.experiments.values()
                    if exp["status"] == ExperimentStatus.RUNNING
                ),
                "completed": sum(
                    1
                    for exp in self.experiments.values()
                    if exp["status"] == ExperimentStatus.COMPLETED
                ),
                "failed": sum(
                    1
                    for exp in self.experiments.values()
                    if exp["status"] == ExperimentStatus.FAILED
                ),
                "total_time_cost_seconds": total_time_cost,
                "avg_time_cost_seconds": avg_time_cost,
                "formatted_avg_time": str(timedelta(seconds=int(avg_time_cost))),
                "series_duration_seconds": series_duration,
                "formatted_series_duration": str(
                    timedelta(seconds=int(series_duration))
                ),
            }
            return summary


class MemoryExperimentSeriesRunner:
    """Manages running a series of memory experiments with different benchmark/model combinations

    Adapted from ExperimentSeriesRunner to work with EmotionMemoryExperiment.
    Supports:
    - Running multiple benchmark/model combinations in sequence
    - Graceful shutdown and resumption of experiment series
    - Model download and verification
    - CUDA memory cleanup between experiments
    """

    def __init__(
        self,
        config_path: Optional[str] = None,
        series_name: Optional[str] = None,
        resume: Optional[object] = None,
        dry_run: bool = False,
    ):
        # Setup logging
        self.logger = logging.getLogger(__name__)
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)
            self.logger.propagate = False  # Prevent duplicate logging

        self.config_path = config_path
        self.series_name = series_name or f"memory_experiment_series"
        self.dry_run = dry_run

        # Initialize shutdown flag
        self.shutdown_requested = False
        signal.signal(signal.SIGINT, self._handle_shutdown)

        # Decide initialization mode
        # Support both new style (resume is a report file path) and legacy bool
        resume_report_path: Optional[str] = None
        if isinstance(resume, str) and resume:
            resume_report_path = resume
        elif isinstance(resume, bool) and resume:
            # Legacy truthy resume without path is not supported anymore for CLI,
            # but keep API tolerant: treat as missing and proceed fresh.
            resume_report_path = None

        self._resuming = bool(resume_report_path)

        # Optionally load a new config for comparison if provided
        new_config_if_provided: Optional[Dict[str, Any]] = None
        if self.config_path:
            try:
                with open(self.config_path, "r") as _cf:
                    new_config_if_provided = yaml.safe_load(_cf)
            except Exception as e:
                self.logger.warning(f"Failed to load --config for comparison: {e}")

        if resume_report_path:
            # Load from existing report file and adopt its config snapshot
            self.report = MemoryExperimentReport(
                base_dir=None, experiment_series_name=self.series_name, report_path=resume_report_path
            )

            if not self.report.series_config:
                raise ValueError(
                    "Loaded report does not contain a 'series_config' snapshot; cannot resume from report."
                )
            resume_cfg = copy.deepcopy(self.report.series_config)

            # If a new config is provided and we are interactive, show diff and ask
            if new_config_if_provided is not None and hasattr(sys.stdin, "isatty") and sys.stdin.isatty():
                use_new = False
                if not self._configs_equal(resume_cfg, new_config_if_provided):
                    self._show_config_diff(resume_cfg, new_config_if_provided)
                    try:
                        ans = input("Configs differ. Use new --config to resume? [y/N]: ").strip().lower()
                        use_new = ans in ("y", "yes")
                    except Exception:
                        use_new = False
                if use_new:
                    self.base_config = new_config_if_provided
                    try:
                        self.report.attach_series_config(self.base_config, self.series_name)
                    except Exception as e:
                        self.logger.warning(f"Could not update series_config in report: {e}")
                    self._session_id = self.report.start_session(
                        resumed_from_report=True,
                        resume_report_path=resume_report_path,
                        config_source="config",
                        config_changed=True,
                    )
                else:
                    self.base_config = resume_cfg
                    self._session_id = self.report.start_session(
                        resumed_from_report=True,
                        resume_report_path=resume_report_path,
                        config_source="report",
                        config_changed=False,
                    )
            else:
                # Non-interactive or no new config provided
                self.base_config = resume_cfg
                self._session_id = self.report.start_session(
                    resumed_from_report=True,
                    resume_report_path=resume_report_path,
                    config_source="report" if new_config_if_provided is None else "report(non-interactive)",
                    config_changed=False,
                )
        else:
            # Load and parse config
            if not self.config_path:
                raise ValueError(
                    "config_path is required when resume (report path) is not provided"
                )
            self._load_config()

            # Create a fresh report and persist the series config snapshot
            base_dir = self.base_config.get("output_dir", "results/memory_experiments")
            self.report = MemoryExperimentReport(base_dir, self.series_name)
            # Save the config snapshot into the report for future resumption
            try:
                self.report.attach_series_config(self.base_config, self.series_name)
            except Exception as e:
                self.logger.warning(f"Could not attach series_config to report: {e}")
            # Start a session record
            self._session_id = self.report.start_session(
                resumed_from_report=False,
                resume_report_path=None,
                config_source="config",
                config_changed=False,
            )

    def _load_config(self) -> None:
        """Load configuration from YAML file"""
        assert self.config_path is not None
        with open(self.config_path, "r") as f:
            self.base_config = yaml.safe_load(f)

        # Validate required sections
        if "models" not in self.base_config:
            raise ValueError("Configuration must include 'models' section")
        if "benchmarks" not in self.base_config:
            raise ValueError("Configuration must include 'benchmarks' section")
        if "emotions" not in self.base_config:
            raise ValueError("Configuration must include 'emotions' section")
        if "intensities" not in self.base_config:
            raise ValueError("Configuration must include 'intensities' section")

    def _check_model_existence(self, model_name: str) -> Optional[str]:
        """
        Check if the model exists in either ~/.cache/huggingface/hub/ or ../huggingface.
        If not, download it to ../huggingface.

        Args:
            model_name: The name of the model to check

        Returns:
            Optional[str]: Resolved local path (or repo id for cache hits) if the
            model can be used, otherwise None
        """
        # Fast path for absolute local paths: validate directory existence first
        # KISS: do not attempt autocorrection; fail fast with a clear log.
        if os.path.isabs(model_name):
            resolved = os.path.realpath(model_name)
            if not os.path.isdir(resolved):
                self.logger.error(
                    f"Local model path not found or not a directory: {resolved}")
                return None

            # Minimal HF folder validation: require config.json at root
            cfg = os.path.join(resolved, "config.json")
            if not os.path.isfile(cfg):
                self.logger.error(
                    f"Local model path exists but missing config.json: {resolved}")
                return None

            self.logger.info(f"Using existing local model path: {resolved}")
            return resolved

        # Define paths to check
        home_dir = os.path.expanduser("~")
        cache_path = os.path.join(home_dir, ".cache", "huggingface", "hub")
        parent_dir = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "..")
        )
        alt_path = os.path.join(parent_dir, "huggingface_models")

        # Create model-specific paths based on the model name structure (org/model format)
        if "/" in model_name:
            model_parts = model_name.split("/")
            model_org = model_parts[0]
            model_name_part = "/".join(model_parts[1:])

            cache_model_path = os.path.join(
                cache_path,
                "models--" + model_org + "--" + model_name_part.replace("/", "--"),
            )
            alt_model_path = os.path.join(alt_path, model_org, model_name_part)
        else:
            # For models without organization prefix
            cache_model_path = os.path.join(cache_path, "models--" + model_name)
            alt_model_path = os.path.join(alt_path, model_name)

        self.logger.info(f"Checking if model {model_name} exists...")
        self.logger.info(f"Checking path: {cache_model_path}")
        self.logger.info(f"Checking alternative path: {alt_model_path}")

        # Check if model exists in either location
        if os.path.exists(alt_model_path):
            self.logger.info(f"Model {model_name} found at {alt_model_path}.")
            return alt_model_path

        if os.path.exists(cache_model_path):
            self.logger.info(f"Model {model_name} found in Hugging Face cache.")
            return model_name

        # If model doesn't exist, download it to ../huggingface_models
        self.logger.info(
            f"Model {model_name} not found. Downloading to {alt_model_path}..."
        )
        try:
            # Make sure the target directory exists
            os.makedirs(os.path.dirname(alt_model_path), exist_ok=True)

            # First verify the model exists on HuggingFace (if transformers available)
            try:
                from transformers import AutoConfig as _AutoConfig  # type: ignore

                _AutoConfig.from_pretrained(model_name, trust_remote_code=True)
            except Exception as e:
                self.logger.error(
                    f"Model {model_name} not found on HuggingFace or transformers unavailable: {str(e)}"
                )
                return None

            # Download model using huggingface-cli command
            self.logger.info(
                f"Starting download of model {model_name} to {alt_model_path} using huggingface-cli..."
            )

            # Prepare environment with HF_HUB_ENABLE_HF_TRANSFER=1 for faster downloads
            env = os.environ.copy()
            env["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

            # Run huggingface-cli download command
            cmd = [
                "huggingface-cli",
                "download",
                model_name,
                "--local-dir",
                alt_model_path,
            ]
            self.logger.info(f"Running command: {' '.join(cmd)}")

            # Execute the command and capture output
            process = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env, text=True
            )

            # Stream output in real-time
            while True:
                if process.stdout is not None:
                    output = process.stdout.readline()
                    if output == "" and process.poll() is not None:
                        break
                    if output:
                        self.logger.info(output.strip())
                else:
                    break

            # Get return code and stderr
            return_code = process.poll()
            stderr = process.stderr.read() if process.stderr else ""

            if return_code != 0:
                self.logger.error(
                    f"Download failed with return code {return_code}: {stderr}"
                )
                return None

            self.logger.info(
                f"Model {model_name} successfully downloaded to {alt_model_path}"
            )
            return alt_model_path

        except Exception as e:
            self.logger.error(f"Failed to download model {model_name}: {str(e)}")
            return None

    def _handle_shutdown(self, sig, frame):
        """Handle SIGINT (Ctrl+C)"""
        if not self.shutdown_requested:
            self.logger.info(
                "Shutdown requested. Finishing current experiment and stopping..."
            )
            self.shutdown_requested = True
            try:
                self.report.log_shutdown_request("SIGINT")
            except Exception:
                pass
        else:
            self.logger.warning("Forced shutdown requested. Exiting immediately.")
            sys.exit(1)

    def _format_model_name_for_folder(self, model_name: str) -> str:
        """Format model name for folder name by removing path prefix"""
        # Count the number of forward slashes in the model name
        slash_count = model_name.count("/")

        # Special case for full paths with multiple parts
        if slash_count >= 2 and model_name.startswith("/"):
            # For paths like /data/home/huggingface_models/RWKV/v6-Finch-7B-HF
            # Extract the last two parts
            parts = model_name.split("/")
            if len(parts) >= 3:  # Make sure we have enough parts
                return f"{parts[-2]}/{parts[-1]}"

        # For HuggingFace model paths like meta-llama/Llama-3.1-8B-Instruct
        elif slash_count == 1:
            # Return as is
            return model_name
        elif slash_count == 2:
            # For paths with exactly three parts
            parts = model_name.split("/")
            return f"{parts[1]}/{parts[2]}"

        # For paths with more than three parts
        elif slash_count > 2:
            # Extract the last two parts
            parts = model_name.split("/")
            return f"{parts[-2]}/{parts[-1]}"

        return model_name

    def setup_experiment(self, benchmark_config: Dict, model_name: str):
        """Set up a single memory experiment with the given benchmark and model"""

        # Create BenchmarkConfig directly from dictionary with all required fields
        data_path_raw = benchmark_config.get("data_path")
        if isinstance(data_path_raw, Path):
            data_path = data_path_raw
        elif isinstance(data_path_raw, (str, os.PathLike)):
            data_path = Path(data_path_raw)
        elif data_path_raw is None:
            data_path = None
        else:
            data_path = Path(str(data_path_raw))

        benchmark = BenchmarkConfig(
            name=benchmark_config["name"],
            task_type=benchmark_config["task_type"],
            data_path=data_path,
            base_data_dir=benchmark_config.get("base_data_dir"),
            sample_limit=benchmark_config.get("sample_limit"),
            augmentation_config=benchmark_config.get("augmentation_config"),
            enable_auto_truncation=benchmark_config.get(
                "enable_auto_truncation", False
            ),
            truncation_strategy=benchmark_config.get("truncation_strategy", "right"),
            preserve_ratio=benchmark_config.get("preserve_ratio", 0.8),
            llm_eval_config=benchmark_config.get("llm_eval_config"),
        )

        batch_size = int(self.base_config.get("batch_size", 4))

        # Create VLLMLoadingConfig directly from base config
        loading_cfg = self.base_config["loading_config"]
        additional_vllm_kwargs = dict(loading_cfg.get("additional_vllm_kwargs", {}) or {})
        additional_vllm_kwargs.setdefault(
            "max_num_seqs", min(batch_size, DEFAULT_VLLM_MAX_NUM_SEQS_CAP)
        )
        loading_config = VLLMLoadingConfig(
            model_path=loading_cfg.get("model_path", model_name),
            gpu_memory_utilization=loading_cfg.get("gpu_memory_utilization", 0.90),
            tensor_parallel_size=loading_cfg.get("tensor_parallel_size"),
            max_model_len=loading_cfg.get("max_model_len", DEFAULT_VLLM_MAX_MODEL_LEN),
            enforce_eager=loading_cfg.get("enforce_eager", True),
            quantization=loading_cfg.get("quantization"),
            trust_remote_code=loading_cfg.get("trust_remote_code", True),
            dtype=loading_cfg.get("dtype", "float16"),
            seed=loading_cfg.get("seed", 42),
            disable_custom_all_reduce=loading_cfg.get(
                "disable_custom_all_reduce", False
            ),
            additional_vllm_kwargs=additional_vllm_kwargs,
        )

        defer_eval_flag = bool(self.base_config.get("defer_evaluation", False))

        # Create ExperimentConfig
        experiment_config = ExperimentConfig(
            model_path=model_name,
            emotions=self.base_config["emotions"],
            intensities=self.base_config["intensities"],
            benchmark=benchmark,
            output_dir=self.base_config.get("output_dir", "results/memory_experiments"),
            batch_size=batch_size,
            generation_config=self.base_config.get("generation_config"),
            loading_config=loading_config,
            repe_eng_config=self.base_config.get("repe_eng_config"),
            max_evaluation_workers=self.base_config.get("max_evaluation_workers", 2),
            pipeline_queue_size=self.base_config.get("pipeline_queue_size", 2),
            defer_evaluation=defer_eval_flag,
        )

        # Import and create experiment with dry_run parameter
        from .experiment import EmotionExperiment

        # Allow configuring repeat runs and seed base from config. Support both
        # top-level keys and nested under an optional 'execution' section.
        exec_cfg = self.base_config.get("execution", {})
        repeat_runs = self.base_config.get("repeat_runs", exec_cfg.get("repeat_runs"))
        repeat_seed_base = self.base_config.get(
            "repeat_seed_base", exec_cfg.get("repeat_seed_base")
        )

        experiment = EmotionExperiment(
            experiment_config,
            dry_run=self.dry_run,
            repeat_runs=repeat_runs or 1,
            repeat_seed_base=repeat_seed_base,
        )
        
        if self.dry_run:
            # Validate that datasets were created successfully and have samples
            assert experiment.emotion_datasets is not None
            ok = True
            for emo, ds in experiment.emotion_datasets.items():
                try:
                    n = len(ds)
                    self.logger.info(f"  - Emotion '{emo}': {n} items")
                    if n == 0:
                        ok = False
                        continue
                    sample = ds[0]
                    if "prompt" not in sample or "ground_truth" not in sample:
                        ok = False
                except Exception as e:
                    ok = False
                    self.logger.error(f"Dry-run dataset validation failed for '{emo}': {e}")
            assert ok, "Dry-run dataset construction failed validation"
            self.logger.info(f"✓ Dry-run successful: {len(experiment.emotion_datasets)} emotion datasets validated")
        
        return experiment

    def _clean_cuda_memory(self) -> None:
        """Clean up CUDA memory after an experiment

        This function attempts to free up CUDA memory by:
        1. Running Python's garbage collector
        2. Emptying CUDA cache if PyTorch is available
        3. Running a system command to check CUDA memory usage
        """
        try:
            # Run Python's garbage collector
            gc.collect()

            # Try to empty CUDA cache if PyTorch is available
            try:
                import torch

                if torch.cuda.is_available():
                    self.logger.info("Clearing CUDA cache...")
                    torch.cuda.empty_cache()
                    # Get and log memory stats
                    allocated = torch.cuda.memory_allocated() / (1024**3)
                    max_allocated = torch.cuda.max_memory_allocated() / (1024**3)
                    reserved = torch.cuda.memory_reserved() / (1024**3)
                    max_reserved = torch.cuda.max_memory_reserved() / (1024**3)
                    self.logger.info(
                        f"CUDA memory stats after cleanup: allocated={allocated:.2f}GB, "
                        f"max_allocated={max_allocated:.2f}GB, reserved={reserved:.2f}GB, "
                        f"max_reserved={max_reserved:.2f}GB"
                    )
            except ImportError:
                self.logger.info("PyTorch not available for CUDA memory cleanup")

            # Try to run nvidia-smi to check memory usage
            try:
                self.logger.info("Running nvidia-smi to check GPU memory...")
                result = subprocess.run(
                    [
                        "nvidia-smi",
                        "--query-gpu=memory.used,memory.free",
                        "--format=csv",
                    ],
                    capture_output=True,
                    text=True,
                    check=True,
                )
                self.logger.info(f"NVIDIA-SMI report:\n{result.stdout}")
            except (subprocess.SubprocessError, FileNotFoundError) as e:
                self.logger.info(f"Could not run nvidia-smi: {str(e)}")

        except Exception as e:
            self.logger.warning(f"Error during CUDA memory cleanup: {str(e)}")

    def run_single_experiment(
        self, benchmark_config: Dict[str, Any], model_name: str, exp_id: str
    ) -> bool:
        """Run a single memory experiment with the specified benchmark and model"""
        self.logger.info(
            f"Starting memory experiment with benchmark: {benchmark_config['name']}, model: {model_name}"
        )

        # Start timing
        start_time = datetime.now()

        # Update experiment status
        self.report.update_experiment(
            exp_id, status=ExperimentStatus.RUNNING, start_time=start_time.isoformat()
        )

        experiment = None
        success = False
        try:
            experiment = self.setup_experiment(benchmark_config, model_name)

            # Record output directory for later reference
            output_dir = str(experiment.output_dir)
            self.report.update_experiment(exp_id, output_dir=output_dir)

            # Run the experiment
            if self.base_config.get("run_sanity_check", False) or self.base_config.get("sanity_check", False):
                limit = int(self.base_config.get("sanity_check_limit", 5))
                experiment.run_sanity_check(sample_limit=limit)
            else:
                experiment.run_experiment()

            # End timing
            end_time = datetime.now()
            time_cost = (end_time - start_time).total_seconds()

            # Format time cost for logging
            time_cost_str = str(timedelta(seconds=int(time_cost)))

            # Update experiment status
            self.report.update_experiment(
                exp_id,
                status=ExperimentStatus.COMPLETED,
                end_time=end_time.isoformat(),
                time_cost_seconds=time_cost,
            )

            self.logger.info(
                f"Memory experiment completed: {benchmark_config['name']}, {model_name}, Time cost: {time_cost_str}"
            )

            success = True

        except Exception as e:
            # End timing even if experiment failed
            end_time = datetime.now()
            time_cost = (end_time - start_time).total_seconds()

            error_trace = traceback.format_exc()
            self.logger.error(
                f"Memory experiment failed: {benchmark_config['name']}, {model_name}\nError: {str(e)}\n{error_trace}"
            )

            # Update experiment status
            self.report.update_experiment(
                exp_id,
                status=ExperimentStatus.FAILED,
                end_time=end_time.isoformat(),
                time_cost_seconds=time_cost,
                error=f"{str(e)}\n{error_trace}",
            )

        finally:
            if experiment is not None:
                try:
                    experiment.close()
                except Exception as close_error:
                    self.logger.warning(
                        f"Failed to close experiment resources: {close_error}"
                    )

            if success:
                self.logger.info("Cleaning up CUDA memory...")
            else:
                self.logger.info(
                    "Attempting to clean up CUDA memory after failed experiment..."
                )
            self._clean_cuda_memory()

        return success

    def expand_benchmark_configs(self, benchmarks: List[Dict]) -> List[Dict]:
        """
        Expand benchmark configurations that have task_type='all' into individual configs.

        Args:
            benchmarks: List of benchmark configuration dictionaries

        Returns:
            Expanded list with task_type='all' configs replaced by individual task configs
        """
        expanded_benchmarks = []

        for benchmark_config in benchmarks:
            task_type = benchmark_config.get("task_type", "")

            # Check if task_type is a pattern (contains wildcards or regex characters)
            if self._is_pattern_task_type(task_type):
                # Create a temporary BenchmarkConfig to discover datasets
                # NOTE: This temp config is ONLY used for pattern discovery via discover_datasets_by_pattern().
                # It should NEVER have get_data_path() called on it, as that would generate invalid paths
                # like "infinitebench_.*qa_eng_121k.*.jsonl" instead of concrete file paths.
                temp_benchmark = BenchmarkConfig(
                    name=benchmark_config["name"],
                    task_type=task_type,  # This is a regex pattern, not a literal task name
                    data_path=None,
                    base_data_dir=benchmark_config.get("base_data_dir", None),
                    sample_limit=benchmark_config.get("sample_limit"),
                    augmentation_config=benchmark_config.get("augmentation_config"),
                    enable_auto_truncation=benchmark_config.get(
                        "enable_auto_truncation", False
                    ),
                    truncation_strategy=benchmark_config.get(
                        "truncation_strategy", "right"
                    ),
                    preserve_ratio=benchmark_config.get("preserve_ratio", 0.8),
                    llm_eval_config=benchmark_config.get("llm_eval_config"),
                )

                # Discover task types matching the pattern
                discovery_base = temp_benchmark.base_data_dir
                if discovery_base is None:
                    discovery_base = (
                        benchmark_config.get("base_data_dir")
                        or self.base_config.get("base_data_dir")
                        or self.base_config.get("output_dir")
                    )
                if discovery_base is None and self.config_path:
                    discovery_base = str(Path(self.config_path).parent)
                if discovery_base is None:
                    discovery_base = "."

                discovered_tasks = temp_benchmark.discover_datasets_by_pattern(
                    discovery_base
                )
                if not discovered_tasks:
                    self.logger.warning(
                        f"No datasets found for benchmark '{benchmark_config['name']}' "
                        f"in directory '{temp_benchmark.base_data_dir}'. Skipping."
                    )
                    continue

                self.logger.info(
                    f"Discovered {len(discovered_tasks)} task types for benchmark '{benchmark_config['name']}'"
                )

                # Create individual configs for each discovered task
                # Each expanded config will have a literal task name (not a regex pattern)
                for task_type in discovered_tasks:
                    expanded_config = copy.deepcopy(benchmark_config)
                    expanded_config["task_type"] = (
                        task_type  # Now a concrete task name like "longbook_qa_eng_121k"
                    )
                    expanded_benchmarks.append(expanded_config)
            else:
                # Keep existing non-'all' configs as-is
                expanded_benchmarks.append(benchmark_config)

        return expanded_benchmarks

    def _is_pattern_task_type(self, task_type: str) -> bool:
        """
        Check if task_type is a pattern that needs expansion.

        Args:
            task_type: The task type string to check

        Returns:
            True if task_type contains pattern characters, False otherwise

        Examples:
            - "*" -> True (all files)
            - "narrative*" -> True (starts with narrative)
            - "*qa" -> True (ends with qa)
            - "pass.*" -> True (regex pattern)
            - "narrativeqa" -> False (literal task name)
        """
        if not task_type:
            return False

        # Check for common pattern characters
        pattern_chars = [
            "*",
            "?",
            "[",
            "]",
            "{",
            "}",
            "(",
            ")",
            "^",
            "$",
            "+",
            ".",
            "|",
            "\\",
        ]
        return any(char in task_type for char in pattern_chars)

    def _create_temporary_benchmark_for_discovery(
        self, benchmark_config: Dict[str, Any], pattern: str
    ) -> BenchmarkConfig:
        """Create a temporary BenchmarkConfig used only for pattern discovery."""
        base_data_dir = benchmark_config.get("base_data_dir")
        if base_data_dir is None:
            base_data_dir = self.base_config.get("base_data_dir")
        if base_data_dir is None:
            base_data_dir = self.base_config.get("output_dir")
        if base_data_dir is None and self.config_path:
            base_data_dir = str(Path(self.config_path).parent)
        if base_data_dir is None:
            base_data_dir = "."
        return BenchmarkConfig(
            name=benchmark_config["name"],
            task_type=pattern,
            data_path=None,
            base_data_dir=base_data_dir,
            sample_limit=benchmark_config.get("sample_limit"),
            augmentation_config=benchmark_config.get("augmentation_config"),
            enable_auto_truncation=benchmark_config.get("enable_auto_truncation", False),
            truncation_strategy=benchmark_config.get("truncation_strategy", "right"),
            preserve_ratio=benchmark_config.get("preserve_ratio", 0.8),
            llm_eval_config=benchmark_config.get("llm_eval_config"),
        )

    def _configs_equal(self, a: Dict[str, Any], b: Dict[str, Any]) -> bool:
        try:
            return json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)
        except Exception:
            return a == b

    def _show_config_diff(self, old_cfg: Dict[str, Any], new_cfg: Dict[str, Any]) -> None:
        try:
            old_text = yaml.safe_dump(old_cfg, sort_keys=True)
            new_text = yaml.safe_dump(new_cfg, sort_keys=True)
            diff = difflib.unified_diff(
                old_text.splitlines(),
                new_text.splitlines(),
                fromfile="report(series_config)",
                tofile="new --config",
                lineterm="",
            )
            self.logger.info("Configuration differences detected:")
            for line in diff:
                self.logger.info(line)
        except Exception as e:
            self.logger.warning(f"Could not render config diff: {e}")

    def dry_run_series(self) -> None:
        """Dry run to validate configuration without running experiments"""
        self.logger.info("🚀 Starting DRY RUN - Memory Experiment Series Validation")
        self.logger.info("=" * 60)

        # Get lists of benchmarks and models from config
        original_benchmarks = self.base_config["benchmarks"]
        models = self.base_config["models"]

        # Expand benchmarks with regex patterns
        benchmarks = self.expand_benchmark_configs(original_benchmarks)

        self.logger.info(f"📊 Original benchmarks: {len(original_benchmarks)}")
        self.logger.info(f"📈 Expanded benchmarks: {len(benchmarks)}")
        self.logger.info(f"🤖 Models: {len(models)}")
        self.logger.info(f"😊 Emotions: {len(self.base_config['emotions'])}")
        self.logger.info(f"📈 Intensities: {len(self.base_config['intensities'])}")


        # Calculate experiment combinations
        total_combinations = len(benchmarks) * len(models)
        total_with_emotions = (
            total_combinations
            * len(self.base_config["emotions"])
            * len(self.base_config["intensities"])
        )

        self.logger.info(f"🧮 Total experiment combinations: {total_combinations}")
        self.logger.info(
            f"🎯 Total runs with emotions/intensities: {total_with_emotions}"
        )

        # Test creating experiment configurations
        #
        # NOTE: Dry-run should catch dataset/schema issues, which are benchmark-specific
        # (e.g., required scenario fields like previous_offer_level). Those failures are
        # independent of model choice. For speed and determinism, validate *all* benchmarks
        # against the first model entry rather than sampling the first 3 benchmark×model pairs.
        self.logger.info("\n🔬 Testing experiment configuration creation...")
        if not models:
            raise ValueError("Configuration must include at least one model")
        model_name = models[0]
        self.logger.info(
            f"🔎 Dry-run validating {len(benchmarks)} benchmark(s) using first model: {model_name}"
        )

        errors: List[Tuple[int, str, str]] = []

        for i, benchmark_config in enumerate(benchmarks):
            try:
                experiment = self.setup_experiment(benchmark_config, model_name)
                self.logger.info(
                    f"   ✅ Config {i+1}: {benchmark_config['name']}_{benchmark_config['task_type']} + {model_name}"
                )
                self.logger.info(f"      📁 Output: {experiment.config.output_dir}")
                self.logger.info(
                    f"      🎯 Data path: {experiment.config.benchmark.get_data_path()}"
                )
                
                # Log first dataset item if emotion_datasets exist (dry-run mode)
                if hasattr(experiment, 'emotion_datasets') and experiment.emotion_datasets:
                    first_emotion = list(experiment.emotion_datasets.keys())[0]
                    first_dataset = experiment.emotion_datasets[first_emotion]
                    if len(first_dataset) > 0:
                        first_item = first_dataset[0]
                        self.logger.info(f"      📋 First dataset item from emotion '{first_emotion}':")
                        
                        # Extract and display meaningful dataset content
                        if isinstance(first_item, dict) and 'item' in first_item:
                            benchmark_item = first_item['item']
                            formatted_prompt = first_item.get('prompt', 'N/A')
                            ground_truth = first_item.get('ground_truth', 'N/A')
                            
                            self.logger.info(f"         ID: {getattr(benchmark_item, 'id', 'N/A')}")
                            self.logger.info(f"         Input: {getattr(benchmark_item, 'input_text', 'N/A')}")
                            self.logger.info(f"         Ground truth: {ground_truth}")
                            
                            # Show first 1000 chars of formatted prompt to validate prompt wrapping
                            if isinstance(formatted_prompt, str):
                                if len(formatted_prompt) > 1000:
                                    self.logger.info(f"         Formatted prompt: {formatted_prompt[:1500]}...{formatted_prompt[-200:] if len(formatted_prompt) > 1200 else ''} ")
                                else:
                                    self.logger.info(f"         Formatted prompt: {formatted_prompt}")

                                # Additionally, print a focused window around the first user turn
                                try:
                                    user_tag = "<|im_start|>user"
                                    idx = formatted_prompt.find(user_tag)
                                    if idx != -1:
                                        preview = formatted_prompt[idx: idx + 300]
                                        self.logger.info(f"         User segment preview: {preview}")
                                except Exception:
                                    pass
                        else:
                            # Fallback for unexpected structure
                            self.logger.info(f"         Unexpected item structure: {first_item}")
            except Exception as e:
                bench_id = f"{benchmark_config.get('name')}_{benchmark_config.get('task_type')}"
                self.logger.error(f"   ❌ Config {i+1} failed ({bench_id}): {e}")
                errors.append((i + 1, bench_id, str(e)))

        if errors:
            failure_summary = "; ".join(
                f"Config {idx} failed ({bench_id}): {message}"
                for idx, bench_id, message in errors
            )
            self.logger.error(
                "Dry run aborted with %d configuration error(s)", len(errors)
            )
            raise RuntimeError(failure_summary)

        self.logger.info("\n🎉 DRY RUN COMPLETED SUCCESSFULLY!")
        self.logger.info("✅ Configuration is valid and ready for execution")
        self.logger.info(
            f"✅ Would run {len(benchmarks)} benchmark(s) × {len(models)} model(s)"
        )

    def run_experiment_series(self) -> None:
        """Run the full series of memory experiments with all benchmark/model combinations"""
        # Check if this is a dry run
        if self.dry_run:
            self.dry_run_series()
            try:
                self.report.end_session("completed")
            except Exception:
                pass
            return

        # Record series start time
        series_start_time = datetime.now()

        # Get lists of benchmarks and models from config
        original_benchmarks = self.base_config["benchmarks"]
        models = self.base_config["models"]

        # Expand benchmarks with task_type is regex pattern
        try:
            benchmarks = self.expand_benchmark_configs(original_benchmarks)
        except Exception as e:
            error_trace = traceback.format_exc()
            self.logger.error(
                f"Failed to expand benchmark configurations: {str(e)}\n{error_trace}"
            )
            self.logger.error("Using original benchmarks without expansion")
            benchmarks = original_benchmarks

        self.logger.info(
            f"Starting memory experiment series with {len(benchmarks)} benchmarks "
            f"(expanded from {len(original_benchmarks)} original) and {len(models)} models"
        )

        # Pre-check and download models if needed
        self.logger.info("Checking model availability...")
        resolved_models: Dict[str, Optional[str]] = {}
        for model_name in models:
            resolved_path = self._check_model_existence(model_name)
            resolved_models[model_name] = resolved_path
            if not resolved_path:
                self.logger.warning(
                    f"Model {model_name} could not be verified or downloaded. Experiments with this model may fail."
                )

        # Generate experiment IDs and initialize report
        try:
            for benchmark_config in benchmarks:
                benchmark_name = benchmark_config["name"]
                task_type = benchmark_config["task_type"]
                for model_name in models:
                    try:
                        # For folder names, use the formatted version
                        model_folder_name = self._format_model_name_for_folder(
                            model_name
                        )
                        exp_id = f"{benchmark_name}_{task_type}_{model_folder_name.replace('/', '_')}"

                        # Only add if not resuming or not already in report
                        if not self._resuming or exp_id not in self.report.experiments:
                            self.report.add_experiment(
                                f"{benchmark_name}_{task_type}",
                                model_name,
                                exp_id,
                                resolved_model_path=resolved_models.get(model_name),
                            )
                    except Exception as e:
                        self.logger.error(
                            f"Failed to generate experiment for {benchmark_name}/{task_type} + {model_name}: {e}"
                        )
                        continue
        except Exception as e:
            error_trace = traceback.format_exc()
            self.logger.error(
                f"Failed to generate experiment combinations: {str(e)}\n{error_trace}"
            )
            raise

        # Get pending experiments
        pending_experiments = self.report.get_incomplete_experiments()
        total_experiments = len(pending_experiments)
        self.logger.info(f"Total pending experiments: {total_experiments}")

        # Run each experiment
        for i, exp in enumerate(pending_experiments):
            if self.shutdown_requested:
                self.logger.info("Shutdown requested. Stopping experiment series.")
                break

            resolved_model_path = exp.get("resolved_model_path")
            model_name = exp["model_name"]
            if not resolved_model_path:
                resolved_model_path = resolved_models.get(model_name)
                if not resolved_model_path:
                    resolved_model_path = self._check_model_existence(model_name)
                    if resolved_model_path:
                        self.report.update_experiment(
                            exp["exp_id"], resolved_model_path=resolved_model_path
                        )

            self.logger.info(
                f"Running experiment {i+1}/{total_experiments}: {exp['benchmark_name']}, {model_name}"
            )

            try:
                if not resolved_model_path:
                    self.logger.error(
                        f"Skipping experiment with {exp['benchmark_name']} and {model_name} due to missing model."
                    )
                    self.report.update_experiment(
                        exp["exp_id"],
                        status=ExperimentStatus.FAILED,
                        error=f"Model {model_name} could not be found or downloaded.",
                    )
                    continue

                # Find the benchmark config
                # exp["benchmark_name"] is now in format "benchmark_tasktype"
                found_benchmark_config: Optional[Dict[str, Any]] = None
                for bench in benchmarks:
                    bench_identifier = f"{bench['name']}_{bench['task_type']}"
                    if bench_identifier == exp["benchmark_name"]:
                        found_benchmark_config = bench
                        break

                if not found_benchmark_config:
                    self.logger.error(
                        f"Benchmark config not found for {exp['benchmark_name']}"
                    )
                    self.report.update_experiment(
                        exp["exp_id"],
                        status=ExperimentStatus.FAILED,
                        error=f"Benchmark config not found for {exp['benchmark_name']}",
                    )
                    continue

                # Run the individual experiment (this has its own error handling)
                success = self.run_single_experiment(
                    found_benchmark_config, resolved_model_path, exp["exp_id"]
                )

                if success:
                    self.logger.info(
                        f"✅ Experiment completed successfully: {exp['benchmark_name']}, {model_name}"
                    )
                    # Fallback in case the underlying run_single_experiment mock
                    # did not update the report (common in tests).
                    exp_record = self.report.experiments.get(exp["exp_id"])
                    if not exp_record or exp_record.get("status") != ExperimentStatus.COMPLETED:
                        now = datetime.now().isoformat()
                        update_payload: Dict[str, Any] = {
                            "status": ExperimentStatus.COMPLETED,
                            "end_time": now,
                        }
                        if exp_record and exp_record.get("start_time") and not exp_record.get("time_cost_seconds"):
                            try:
                                start_dt = datetime.fromisoformat(exp_record["start_time"])
                                end_dt = datetime.fromisoformat(now)
                                update_payload["time_cost_seconds"] = (end_dt - start_dt).total_seconds()
                            except Exception:
                                pass
                        self.report.update_experiment(exp["exp_id"], **update_payload)
                else:
                    self.logger.info(
                        f"❌ Experiment failed but series continues: {exp['benchmark_name']}, {exp['model_name']}"
                    )
                    exp_record = self.report.experiments.get(exp["exp_id"])
                    if not exp_record or exp_record.get("status") != ExperimentStatus.FAILED:
                        self.report.update_experiment(
                            exp["exp_id"],
                            status=ExperimentStatus.FAILED,
                            end_time=datetime.now().isoformat(),
                            error="Run marked as failed without explicit report update",
                        )

            except Exception as e:
                # Catch ANY unexpected errors in the experiment series loop
                # This ensures that even if there are issues with config processing,
                # report updating, or other series-level operations, we continue with the next experiment
                error_trace = traceback.format_exc()
                self.logger.error(
                    f"🚨 SERIES-LEVEL ERROR for experiment {exp['benchmark_name']}, {exp['model_name']}: {str(e)}\n{error_trace}"
                )

                try:
                    # Try to update the experiment report with the series-level error
                    self.report.update_experiment(
                        exp["exp_id"],
                        status=ExperimentStatus.FAILED,
                        error=f"Series-level error: {str(e)}\n{error_trace}",
                        end_time=datetime.now().isoformat(),
                    )
                except Exception as report_error:
                    # If even updating the report fails, log it but don't stop the series
                    self.logger.error(
                        f"Failed to update experiment report: {report_error}"
                    )

                # Continue with the next experiment regardless of the error
                self.logger.info(
                    "🔄 Continuing with next experiment despite series-level error"
                )

            finally:
                # Always try to print progress summary, even if there were errors
                try:
                    summary = self.report.get_summary()
                    self.logger.info(f"Memory experiment series progress: {summary}")
                except Exception as summary_error:
                    self.logger.warning(
                        f"Could not generate progress summary: {summary_error}"
                    )

        # Final summary
        summary = self.report.get_summary()

        # Calculate and log total series time
        series_end_time = datetime.now()
        series_time_cost = (series_end_time - series_start_time).total_seconds()
        series_time_str = str(timedelta(seconds=int(series_time_cost)))

        self.logger.info(f"Memory experiment series completed. Final status: {summary}")
        self.logger.info(f"Total series time: {series_time_str}")
        self.logger.info(f"Average experiment time: {summary['formatted_avg_time']}")

        if summary["failed"] > 0:
            self.logger.info("Failed experiments:")
            for exp in self.report.get_failed_experiments():
                # Get time cost for failed experiment if available
                time_cost = exp.get("time_cost_seconds")
                time_info = (
                    f", time: {str(timedelta(seconds=int(time_cost)))}"
                    if time_cost
                    else ""
                )

                self.logger.info(
                    f"  - {exp['benchmark_name']}, {exp['model_name']}{time_info}: {exp.get('error', 'Unknown error')}"
                )

        # End session record
        try:
            reason = "shutdown" if getattr(self, "shutdown_requested", False) else "completed"
            self.report.end_session(reason)
        except Exception:
            pass


def main():
    """Run the memory experiment series from command line"""
    import argparse

    _ensure_openmp_shm_compat()

    parser = argparse.ArgumentParser(
        description="Run a memory experiment series with multiple benchmarks and models"
    )
    parser.add_argument(
        "--config", type=str, required=False, help="Path to experiment config file"
    )
    parser.add_argument(
        "--name", type=str, default=None, help="Custom name for the experiment series"
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to an existing memory_experiment_report.json to resume from",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate configuration without running experiments",
    )

    args = parser.parse_args()

    # Register pipelines only when not dry-run (avoids torch dependency during validation)
    if not args.dry_run:
        from neuro_manipulation.repe.pipelines import repe_pipeline_registry
        repe_pipeline_registry()
    if not args.resume and not args.config:
        parser.error("either --config or --resume must be provided")

    runner = MemoryExperimentSeriesRunner(
        config_path=args.config,
        series_name=args.name,
        resume=args.resume,
        dry_run=args.dry_run,
    )
    runner.run_experiment_series()


if __name__ == "__main__":
    main()
