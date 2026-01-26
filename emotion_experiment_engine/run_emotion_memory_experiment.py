#!/usr/bin/env python3
"""
Runner script for emotion memory experiments.
Loads configuration from YAML files and runs complete emotion memory experiments.
"""

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from .data_models import (
    DEFAULT_VLLM_MAX_MODEL_LEN,
    BenchmarkConfig,
    BenchmarkItem,
    ExperimentConfig,
    VLLMLoadingConfig,
)
from .dataset_factory import create_dataset_from_config


def load_config(config_path: Path) -> Dict[str, Any]:
    """Load experiment configuration from YAML file"""
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    print(f"✅ Loaded configuration from: {config_path}")
    return config


def normalize_config_format(config_dict: Dict[str, Any]) -> Dict[str, Any]:
    """
    Normalize different config formats to a consistent internal format.

    Handles both:
    - New format: {models: [...], benchmarks: [...], emotions: [...], output_dir: "..."}
    - Old format: {model: {model_path: "..."}, benchmarks: {...}, emotions: {target_emotions: [...]}, output: {results_dir: "..."}}

    Returns normalized config with consistent key structure.
    """
    normalized = config_dict.copy()

    # Normalize model format
    if "models" in config_dict and isinstance(config_dict["models"], list):
        if not normalized.get("model"):
            normalized["model"] = {}
        normalized["model"]["model_path"] = (
            config_dict["models"][0] if config_dict["models"] else ""
        )

    # Normalize emotions format
    if isinstance(config_dict.get("emotions"), list):
        normalized["emotions"] = {
            "target_emotions": config_dict["emotions"],
            "intensities": config_dict.get("intensities", [1.0]),
        }

    # Normalize output format
    if "output_dir" in config_dict:
        if not normalized.get("output"):
            normalized["output"] = {}
        normalized["output"]["results_dir"] = config_dict["output_dir"]

    # Normalize execution format
    if "batch_size" in config_dict:
        if not normalized.get("execution"):
            normalized["execution"] = {}
        normalized["execution"]["batch_size"] = config_dict["batch_size"]

    if "max_evaluation_workers" in config_dict:
        if not normalized.get("execution"):
            normalized["execution"] = {}
        normalized["execution"]["max_evaluation_workers"] = config_dict[
            "max_evaluation_workers"
        ]

    return normalized


def create_experiment_config(config_dict: Dict[str, Any]) -> ExperimentConfig:
    """Convert YAML config to ExperimentConfig object"""

    # Normalize config format for consistent processing
    config = normalize_config_format(config_dict)

    # Extract single benchmark configuration (use first benchmark if multiple provided)
    benchmarks = config.get("benchmarks", {})
    if not benchmarks:
        raise ValueError("No benchmarks specified in configuration")

    # Handle both list and dict formats for benchmarks
    if isinstance(benchmarks, list):
        # List format: [{"name": "infinitebench", "task_type": "passkey", ...}]
        if not benchmarks:
            raise ValueError("Empty benchmarks list in configuration")
        benchmark_data = benchmarks[0]
        benchmark_name = benchmark_data.get("name", f"benchmark_0")
    elif isinstance(benchmarks, dict):
        # Dict format: {"infinitebench_passkey": {"name": "infinitebench", ...}}
        benchmark_name, benchmark_data = next(iter(benchmarks.items()))
    else:
        raise ValueError(f"Benchmarks must be list or dict, got {type(benchmarks)}")

    # Extract normalized values
    model_path = config["model"]["model_path"]
    emotions = config["emotions"]["target_emotions"]
    intensities = config["emotions"]["intensities"]
    output_dir = config["output"]["results_dir"]

    # Auto-generate data_path if not provided (like series runner does)
    if "data_path" not in benchmark_data:
        # Create a temporary config to auto-generate the path
        temp_config = BenchmarkConfig(
            name=benchmark_data["name"],
            task_type=benchmark_data["task_type"],
            data_path=None,  # None triggers auto-generation in get_data_path()
            base_data_dir=config.get("base_data_dir", "data/memory_benchmarks"),
            sample_limit=benchmark_data.get("sample_limit"),
            augmentation_config=benchmark_data.get("augmentation_config"),
            enable_auto_truncation=config.get("loading_config", {}).get(
                "enable_auto_truncation", False
            ),
            truncation_strategy=config.get("loading_config", {}).get(
                "truncation_strategy", "right"
            ),
            preserve_ratio=config.get("loading_config", {}).get("preserve_ratio", 0.8),
            llm_eval_config=benchmark_data.get("llm_eval_config"),
        )
        # Get the auto-generated data path
        auto_data_path = temp_config.get_data_path()
        benchmark_data = benchmark_data.copy()
        benchmark_data["data_path"] = str(auto_data_path)

    # Create BenchmarkConfig with all required fields
    benchmark_config = BenchmarkConfig(
        name=benchmark_data["name"],
        task_type=benchmark_data["task_type"],
        data_path=Path(benchmark_data["data_path"]),
        base_data_dir=config.get("base_data_dir", "data/memory_benchmarks"),
        sample_limit=benchmark_data.get("sample_limit"),
        augmentation_config=benchmark_data.get("augmentation_config"),
        enable_auto_truncation=benchmark_data.get("enable_auto_truncation", False),
        truncation_strategy=benchmark_data.get("truncation_strategy", "right"),
        preserve_ratio=benchmark_data.get("preserve_ratio", 0.8),
        llm_eval_config=benchmark_data.get("llm_eval_config"),
    )
    # Create VLLMLoadingConfig directly from YAML
    loading_config = None
    if "loading_config" in config:
        loading_cfg = config["loading_config"]
        loading_config = VLLMLoadingConfig(
            model_path=loading_cfg.get("model_path", model_path),
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
            additional_vllm_kwargs=loading_cfg.get("additional_vllm_kwargs", {}),
        )

    # Create main experiment config
    exp_config = ExperimentConfig(
        model_path=model_path,
        emotions=emotions,
        benchmark=benchmark_config,
        output_dir=output_dir,
        # Optional parameters with defaults
        intensities=intensities,
        batch_size=config.get("execution", {}).get("batch_size", 4),
        generation_config=config.get("generation_config", config.get("generation", {})),
        loading_config=loading_config,
        repe_eng_config=config.get("repe_eng_config", {}),
        max_evaluation_workers=config.get("execution", {}).get(
            "max_evaluation_workers", 4
        ),
        pipeline_queue_size=config.get(
            "pipeline_queue_size", config.get("execution", {}).get("batch_size", 4) * 2
        ),
        defer_evaluation=bool(config.get("defer_evaluation", False)),
    )

    return exp_config


def setup_logging(config: Dict[str, Any]) -> logging.Logger:
    """Setup logging configuration"""
    log_level = config.get("output", {}).get("log_level", "INFO")
    log_file = config.get("output", {}).get("log_file")

    # Configure logging
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    # Add file handler if specified
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_path)
        file_handler.setLevel(getattr(logging, log_level.upper()))
        file_handler.setFormatter(
            logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        )
        logging.getLogger().addHandler(file_handler)
        print(f"📝 Logging to file: {log_path}")

    return logging.getLogger(__name__)


def validate_config(config_dict: Dict[str, Any]) -> bool:
    """Validate configuration before running experiment"""
    print("🔍 Validating configuration...")

    # Normalize config format for consistent validation
    config = normalize_config_format(config_dict)
    errors = []

    # Check required sections
    if not config.get("model", {}).get("model_path"):
        errors.append("Missing model path configuration")
    if not config.get("emotions", {}).get("target_emotions"):
        errors.append("Missing emotions configuration")
    if not config.get("benchmarks"):
        errors.append("Missing benchmarks configuration")

    # Check model path
    model_path_str = config.get("model", {}).get("model_path", "")
    if model_path_str:
        model_path = Path(model_path_str)
        if not model_path.exists() and not str(model_path).startswith(
            ("/mock", "Qwen/")
        ):
            errors.append(f"Model path does not exist: {model_path}")

    # Check benchmark data files (handle both list and dict formats)
    benchmarks = config.get("benchmarks", {})
    if isinstance(benchmarks, list):
        # List format: [{"name": "infinitebench", "data_path": "...", ...}]
        for i, benchmark_config in enumerate(benchmarks):
            benchmark_name = benchmark_config.get("name", f"benchmark_{i}")
            data_path = Path(benchmark_config.get("data_path", ""))
            if data_path and not data_path.exists():
                errors.append(
                    f"Benchmark data file not found: {data_path} (for {benchmark_name})"
                )
    elif isinstance(benchmarks, dict):
        # Dict format: {"infinitebench_passkey": {"data_path": "...", ...}}
        for benchmark_name, benchmark_config in benchmarks.items():
            data_path = Path(benchmark_config.get("data_path", ""))
            if data_path and not data_path.exists():
                errors.append(
                    f"Benchmark data file not found: {data_path} (for {benchmark_name})"
                )

    # Check output directory is writable
    output_dir = Path(config.get("output", {}).get("results_dir", "results"))
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        errors.append(f"Cannot create output directory {output_dir}: {e}")

    if errors:
        print("❌ Configuration validation failed:")
        for error in errors:
            print(f"  - {error}")
        return False

    print("✅ Configuration validation passed")
    return True


def run_experiment(
    config_path: Path, dry_run: bool = False, debug: bool = False,
    repeat_runs: Optional[int] = None,
    repeat_seed_base: Optional[int] = None,
) -> bool:
    """Run emotion memory experiment from configuration file"""

    try:
        # Load and validate configuration
        config_dict = load_config(config_path)

        if not validate_config(config_dict):
            return False

        # Setup logging
        logger = setup_logging(config_dict)

        if debug:
            logging.getLogger().setLevel(logging.DEBUG)
            logger.debug("Debug mode enabled")

        # Create experiment configuration
        exp_config = create_experiment_config(config_dict)

        # Pull execution overrides from config if not provided via args
        exec_cfg = normalize_config_format(config_dict).get("execution", {})
        rr = repeat_runs if repeat_runs is not None else exec_cfg.get("repeat_runs")
        rsb = repeat_seed_base if repeat_seed_base is not None else exec_cfg.get("repeat_seed_base")

        print(f"\n🚀 EMOTION MEMORY EXPERIMENT")
        print(f"Model: {exp_config.model_path}")
        print(f"Emotions: {exp_config.emotions}")
        print(
            f"Benchmark: {exp_config.benchmark.name} ({exp_config.benchmark.task_type})"
        )
        print(f"Output: {exp_config.output_dir}")
        print("=" * 60)

        if dry_run:
            print("🔍 DRY RUN - Configuration validated successfully")
            print("Would run experiment with:")

            # Show what would be tested
            try:
                dataset = create_dataset_from_config(exp_config.benchmark)
                items = len(dataset)
                conditions = (
                    len(exp_config.emotions) * len(exp_config.intensities) * items
                )

                print(
                    f"  📊 {exp_config.benchmark.name}: {items} items × {len(exp_config.emotions)} emotions × {len(exp_config.intensities)} intensities = {conditions} conditions"
                )
                print(f"\n📈 Total experimental conditions: {conditions}")
            except Exception as e:
                print(f"  ❌ {exp_config.benchmark.name}: Error loading - {e}")

            print(f"Dry run complete!")

            return True

        # Import heavy dependencies only when actually running (not for dry-run)
        from neuro_manipulation.repe import repe_pipeline_registry

        from .experiment import EmotionExperiment

        repe_pipeline_registry()
        # Create and run experiment
        experiment = EmotionExperiment(exp_config, repeat_runs=rr or 1, repeat_seed_base=rsb)
        logger.info(f"Starting emotion memory experiment")

        # Run the experiment
        results_df = experiment.run_experiment()

        if results_df is not None and len(results_df) > 0:
            print(f"\n✅ Experiment completed successfully!")
            print(f"📊 Generated {len(results_df)} results")
            print(f"📁 Results saved to: {exp_config.output_dir}")

            # Show summary statistics
            print(f"\n📈 Results Summary:")
            print(f"  Emotions tested: {results_df['emotion'].nunique()}")
            print(f"  Benchmarks tested: {results_df['benchmark'].nunique()}")
            print(f"  Average score: {results_df['score'].mean():.3f}")
            print(
                f"  Score range: {results_df['score'].min():.3f} - {results_df['score'].max():.3f}"
            )

            return True
        else:
            print("❌ Experiment failed - no results generated")
            return False

    except KeyboardInterrupt:
        print("\n⚠️ Experiment interrupted by user")
        return False
    except Exception as e:
        print(f"❌ Experiment failed with error: {e}")
        if debug:
            import traceback

            traceback.print_exc()
        return False


def create_sample_config(output_path: Path):
    """Create a sample configuration file based on the working test config"""

    # Load the working test configuration as template
    template_path = (
        Path(__file__).parent.parent / "config" / "test_downloaded_data.yaml"
    )

    try:
        with open(template_path, "r") as f:
            sample_config = yaml.safe_load(f)
    except FileNotFoundError:
        # Fallback to minimal config if template not found
        sample_config = {
            "experiment_name": "sample_emotion_memory_experiment",
            "description": "Sample configuration for emotion memory experiments",
            "version": "1.0.0",
            "model": {
                "model_path": "/path/to/your/model/Qwen2.5-0.5B-Instruct",
            },
            "emotions": {
                "target_emotions": ["anger", "happiness"],
                "include_neutral": True,
                "intensities": [1.0],
            },
            "benchmarks": {
                "infinitebench_passkey": {
                    "name": "infinitebench",
                    "task_type": "passkey",
                    "data_path": "test_data/real_benchmarks/infinitebench_passkey.jsonl",
                    "evaluation_method": "get_score_one_passkey",
                    "sample_limit": 5,
                }
            },
            "generation": {
                "temperature": 0.1,
                "max_new_tokens": 50,
                "do_sample": False,
                "top_p": 0.9,
                "repetition_penalty": 1.0,
            },
            "execution": {
                "batch_size": 4,
                "max_evaluation_workers": 2,
            },
            "output": {
                "results_dir": "results/emotion_memory",
                "save_intermediate": True,
                "formats": ["csv"],
                "log_level": "INFO",
            },
        }

    # Customize the template for sample config
    sample_config["experiment_name"] = "sample_emotion_memory_experiment"
    sample_config["description"] = "Sample configuration for emotion memory experiments"

    # Update model path to placeholder
    if "model" not in sample_config:
        sample_config["model"] = {}
    sample_config["model"]["model_path"] = "/path/to/your/model/Qwen2.5-0.5B-Instruct"

    # Update output directory
    sample_config["output"]["results_dir"] = "results/emotion_memory"

    with open(output_path, "w") as f:
        yaml.dump(sample_config, f, default_flow_style=False, indent=2)

    print(f"✅ Sample configuration created: {output_path}")
    print("📝 Based on working test configuration")
    print("📝 Edit the model path and data paths before running the experiment.")
    print(
        f"📝 Test with: python scripts/run_emotion_memory_experiment.py {output_path} --dry-run"
    )


def main():
    """Main entry point for the experiment runner"""
    parser = argparse.ArgumentParser(
        description="Run emotion memory experiments from configuration files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run experiment from config
  python scripts/run_emotion_memory_experiment.py config/emotion_memory_test.yaml
  
  # Validate config without running
  python scripts/run_emotion_memory_experiment.py config/test.yaml --dry-run
  
  # Create sample config
  python scripts/run_emotion_memory_experiment.py --create-sample config/sample.yaml
  
  # Run with debug logging
  python scripts/run_emotion_memory_experiment.py config/test.yaml --debug
        """,
    )

    parser.add_argument(
        "config", nargs="?", type=Path, help="Path to YAML configuration file"
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate configuration and show what would be run without executing",
    )

    parser.add_argument("--debug", action="store_true", help="Enable debug logging")

    parser.add_argument(
        "--create-sample",
        type=Path,
        metavar="OUTPUT_PATH",
        help="Create a sample configuration file",
    )

    parser.add_argument(
        "--repeat-runs", type=int, default=None,
        help="Number of independent repeats per condition",
    )
    parser.add_argument(
        "--repeat-seed-base", type=int, default=None,
        help="Base random seed added to repeat index for stochastic decoding",
    )

    args = parser.parse_args()

    # Handle sample config creation
    if args.create_sample:
        create_sample_config(args.create_sample)
        return 0

    # Require config file for other operations
    if not args.config:
        parser.error("Config file required (or use --create-sample)")

    # Run experiment
    success = run_experiment(
        args.config,
        dry_run=args.dry_run,
        debug=args.debug,
        repeat_runs=args.repeat_runs,
        repeat_seed_base=args.repeat_seed_base,
    )
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
