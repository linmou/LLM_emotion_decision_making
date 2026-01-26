"""
This module implements a pipeline for running series of experiments with different combinations
of game names and model names, with support for graceful shutdown and resumption.
"""
import time
import yaml
import signal
import os
import sys
import traceback
import json
import subprocess
import gc
from datetime import datetime, timedelta
from pathlib import Path
import logging
from typing import List, Dict, Any, Optional, Tuple
import threading
import shutil
from transformers import AutoConfig

from constants import GameNames
from neuro_manipulation.repe import repe_pipeline_registry
from neuro_manipulation.configs.experiment_config import get_exp_config, get_repe_eng_config, get_model_config
from neuro_manipulation.experiments.emotion_game_experiment import EmotionGameExperiment
from games.game_configs import get_game_config

class ExperimentStatus:
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"

class ExperimentReport:
    """Manages and persists the status of experiments in a series
    
    The report is saved as a JSON file in a directory named after either:
    1. The experiment name from the config file (preferred)
    2. The series name provided via CLI argument (fallback)
    
    This ensures that experiment reports are stored consistently with the individual experiment outputs.
    """
    
    def __init__(self, base_dir: str, experiment_series_name: str, experiment_name: str = None):
        self.base_dir = base_dir
        self.timestamp = datetime.now().strftime("%Y%m%d_%H")
        # Use experiment_name if provided (from config), otherwise fall back to experiment_series_name
        report_name = experiment_name or experiment_series_name
        self.report_file = Path(f"{base_dir}/{report_name}_{self.timestamp}_experiment_report.json")
        self.report_file.parent.mkdir(parents=True, exist_ok=True)
        self.lock = threading.Lock()
        self.experiments = {}
        self.series_start_time = datetime.now()
        self._save_report()
        
    def add_experiment(
        self,
        game_name: str,
        model_name: str,
        exp_id: str,
        resolved_model_path: Optional[str] = None,
        status: str = ExperimentStatus.PENDING,
    ) -> None:
        """Add a new experiment to the report"""
        with self.lock:
            self.experiments[exp_id] = {
                "game_name": game_name,
                "model_name": model_name,
                "resolved_model_path": resolved_model_path,
                "status": status,
                "start_time": None,
                "end_time": None,
                "time_cost_seconds": None,
                "error": None,
                "output_dir": None,
                "exp_id": exp_id
            }
            self._save_report()
    
    def update_experiment(self, exp_id: str, **kwargs) -> None:
        """Update experiment status and details"""
        with self.lock:
            if exp_id in self.experiments:
                self.experiments[exp_id].update(kwargs)
                
                # Calculate time cost if we have both start and end times
                if "start_time" in self.experiments[exp_id] and "end_time" in self.experiments[exp_id]:
                    start = self.experiments[exp_id]["start_time"]
                    end = self.experiments[exp_id]["end_time"]
                    if start and end and not self.experiments[exp_id].get("time_cost_seconds"):
                        start_dt = datetime.fromisoformat(start)
                        end_dt = datetime.fromisoformat(end)
                        time_cost = (end_dt - start_dt).total_seconds()
                        self.experiments[exp_id]["time_cost_seconds"] = time_cost
                
                self._save_report()
    
    def get_pending_experiments(self) -> List[Dict[str, Any]]:
        """Get list of pending experiments"""
        with self.lock:
            return [exp for exp in self.experiments.values() 
                   if exp["status"] == ExperimentStatus.PENDING]
    
    def get_failed_experiments(self) -> List[Dict[str, Any]]:
        """Get list of failed experiments"""
        with self.lock:
            return [exp for exp in self.experiments.values() 
                   if exp["status"] == ExperimentStatus.FAILED]
    
    def _save_report(self) -> None:
        """Save the report to disk"""
        with open(self.report_file, 'w') as f:
            # Calculate series duration so far
            series_duration = (datetime.now() - self.series_start_time).total_seconds()
            
            json.dump({
                "last_updated": datetime.now().isoformat(),
                "series_start_time": self.series_start_time.isoformat(),
                "series_duration_seconds": series_duration,
                "experiments": self.experiments
            }, f, indent=2)
    
    def load_report(self) -> bool:
        """Load a report from disk if it exists"""
        if self.report_file.exists():
            with open(self.report_file, 'r') as f:
                report_data = json.load(f)
                self.experiments = report_data.get("experiments", {})
            return True
        return False
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary of experiment statuses"""
        with self.lock:
            # Calculate total and average time costs
            completed_exps = [exp for exp in self.experiments.values() 
                             if exp["status"] == ExperimentStatus.COMPLETED and exp.get("time_cost_seconds")]
            
            total_time_cost = sum(exp["time_cost_seconds"] for exp in completed_exps) if completed_exps else 0
            avg_time_cost = total_time_cost / len(completed_exps) if completed_exps else 0
            
            # Calculate series duration so far
            series_duration = (datetime.now() - self.series_start_time).total_seconds()
            
            summary = {
                "total": len(self.experiments),
                "pending": sum(1 for exp in self.experiments.values() if exp["status"] == ExperimentStatus.PENDING),
                "running": sum(1 for exp in self.experiments.values() if exp["status"] == ExperimentStatus.RUNNING),
                "completed": sum(1 for exp in self.experiments.values() if exp["status"] == ExperimentStatus.COMPLETED),
                "failed": sum(1 for exp in self.experiments.values() if exp["status"] == ExperimentStatus.FAILED),
                "total_time_cost_seconds": total_time_cost,
                "avg_time_cost_seconds": avg_time_cost,
                "formatted_avg_time": str(timedelta(seconds=int(avg_time_cost))),
                "series_duration_seconds": series_duration,
                "formatted_series_duration": str(timedelta(seconds=int(series_duration)))
            }
            return summary

class ExperimentSeriesRunner:
    """Manages running a series of experiments with different game/model combinations
    
    This runner supports:
    - Running multiple game/model combinations in sequence
    - Graceful shutdown and resumption of experiment series
    - Model download and verification
    - CUDA memory cleanup between experiments
    """
    
    def __init__(self, config_path: str, series_name: str = None, resume: bool = False):
        # Setup logging
        self.logger = logging.getLogger(__name__)
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)
        
        self.config_path = config_path
        self.exp_config = get_exp_config(config_path)
        self.series_name = series_name or f"experiment_series"
        
        # Initialize shutdown flag
        self.shutdown_requested = False
        signal.signal(signal.SIGINT, self._handle_shutdown)
        
        # Create or load experiment report
        base_dir = self.exp_config['experiment']['output']['base_dir']
        # Get experiment name from config to ensure consistent directory structure
        experiment_name = self.exp_config['experiment'].get('name', None)
        self.report = ExperimentReport(base_dir, self.series_name, experiment_name)
        
        # Check if resuming
        self.resume = resume
        if resume:
            if not self.report.load_report():
                self.logger.warning("No previous experiment report found. Starting fresh.")
            else:
                self.logger.info(f"Resumed experiment series. Status: {self.report.get_summary()}")
    
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
        # Skip for local paths (starting with /)
        if model_name.startswith('/'):
            self.logger.info(f"Skipping model check for local path: {model_name}")
            return model_name
            
        # Define paths to check
        home_dir = os.path.expanduser("~")
        cache_path = os.path.join(home_dir, ".cache", "huggingface", "hub")
        parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
        alt_path = os.path.join(parent_dir, "huggingface_models")
        
        # Create model-specific paths based on the model name structure (org/model format)
        if '/' in model_name:
            model_parts = model_name.split('/')
            model_org = model_parts[0]
            model_name_part = '/'.join(model_parts[1:])
            
            cache_model_path = os.path.join(cache_path, "models--" + model_org + "--" + model_name_part.replace('/', '--'))
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
        self.logger.info(f"Model {model_name} not found. Downloading to {alt_model_path}...")
        try:
            # Make sure the target directory exists
            os.makedirs(os.path.dirname(alt_model_path), exist_ok=True)
            
            # First verify the model exists on HuggingFace
            try:
                AutoConfig.from_pretrained(model_name, trust_remote_code=True)
            except Exception as e:
                self.logger.error(f"Model {model_name} not found on HuggingFace: {str(e)}")
                return None
            
            # Download model using huggingface-cli command
            self.logger.info(f"Starting download of model {model_name} to {alt_model_path} using huggingface-cli...")
            
            # Prepare environment with HF_HUB_ENABLE_HF_TRANSFER=1 for faster downloads
            env = os.environ.copy()
            env["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
            
            # Run huggingface-cli download command
            cmd = ["huggingface-cli", "download", model_name, "--local-dir", alt_model_path]
            self.logger.info(f"Running command: {' '.join(cmd)}")
            
            # Execute the command and capture output
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=env,
                text=True
            )
            
            # Stream output in real-time
            while True:
                output = process.stdout.readline()
                if output == '' and process.poll() is not None:
                    break
                if output:
                    self.logger.info(output.strip())
            
            # Get return code and stderr
            return_code = process.poll()
            stderr = process.stderr.read()
            
            if return_code != 0:
                self.logger.error(f"Download failed with return code {return_code}: {stderr}")
                return None

            self.logger.info(f"Model {model_name} successfully downloaded to {alt_model_path}")
            return alt_model_path

        except Exception as e:
            self.logger.error(f"Failed to download model {model_name}: {str(e)}")
            return None
    
    def _handle_shutdown(self, sig, frame):
        """Handle SIGINT (Ctrl+C)"""
        if not self.shutdown_requested:
            self.logger.info("Shutdown requested. Finishing current experiment and stopping...")
            self.shutdown_requested = True
        else:
            self.logger.warning("Forced shutdown requested. Exiting immediately.")
            sys.exit(1)
    
    def _format_model_name_for_folder(self, model_name: str) -> str:
        """Format model name for folder name by removing path prefix"""
        # Count the number of forward slashes in the model name
        slash_count = model_name.count('/')
        
        # Special case for full paths with multiple parts
        if slash_count >= 2 and model_name.startswith('/'):
            # For paths like /data/home/huggingface_models/RWKV/v6-Finch-7B-HF
            # Extract the last two parts
            parts = model_name.split('/')
            if len(parts) >= 3:  # Make sure we have enough parts
                return f"{parts[-2]}/{parts[-1]}"
        
        # For HuggingFace model paths like meta-llama/Llama-3.1-8B-Instruct
        elif slash_count == 1:
            # Return as is
            return model_name
        elif slash_count == 2:
            # For paths with exactly three parts
            parts = model_name.split('/')
            return f"{parts[1]}/{parts[2]}"
        
        # For paths with more than three parts
        elif slash_count > 2:
            # Extract the last two parts
            parts = model_name.split('/')
            return f"{parts[-2]}/{parts[-1]}"
            
        return model_name
    
    def setup_experiment(self, game_name_str: str, model_name: str) -> EmotionGameExperiment:
        """Set up a single experiment with the given game and model"""
        repe_pipeline_registry()
        
        # Create a copy of the config that we can modify
        exp_config = dict(self.exp_config)
        
        # Update model in config
        exp_config['experiment']['llm']['model_name'] = model_name
        
        # Initialize game section if it doesn't exist (since we removed defaults)
        if 'game' not in exp_config['experiment']:
            exp_config['experiment']['game'] = {}
        
        game_name = GameNames.from_string(game_name_str)
        repe_eng_config = get_repe_eng_config(model_name, yaml_config=exp_config['experiment'])
        game_config = get_game_config(game_name)
        
        # Apply game-specific configurations if available
        games_config = exp_config['experiment'].get('games_config', {})
        if game_name_str in games_config:
            game_specific_config = games_config[game_name_str]
            
            # Apply all game-specific settings to game_config
            for key, value in game_specific_config.items():
                game_config[key] = value
                
            # Also update the experiment config for backward compatibility
            for key, value in game_specific_config.items():
                if key in ['previous_actions_length', 'previous_trust_level', 'previous_offer_level']:
                    exp_config['experiment']['game'][key] = value
        elif game_name.is_sequential():
            # For sequential games without specific config, raise an error to be explicit
            raise ValueError(
                f"Game '{game_name_str}' is a sequential game but has no configuration in 'games_config'. "
                f"Please add configuration for this game with required attributes like 'previous_actions_length'."
            )

        experiment = EmotionGameExperiment(
            repe_eng_config, 
            exp_config, 
            game_config,
            repeat=exp_config['experiment']['repeat'],
            batch_size=exp_config['experiment'].get('batch_size', 300),
            sample_num=exp_config['experiment'].get('sample_num', None)
        )
        
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
                    allocated = torch.cuda.memory_allocated() / (1024 ** 3)
                    max_allocated = torch.cuda.max_memory_allocated() / (1024 ** 3)
                    reserved = torch.cuda.memory_reserved() / (1024 ** 3)
                    max_reserved = torch.cuda.max_memory_reserved() / (1024 ** 3)
                    self.logger.info(f"CUDA memory stats after cleanup: allocated={allocated:.2f}GB, "
                                    f"max_allocated={max_allocated:.2f}GB, reserved={reserved:.2f}GB, "
                                    f"max_reserved={max_reserved:.2f}GB")
            except ImportError:
                self.logger.info("PyTorch not available for CUDA memory cleanup")
            
            # Try to run nvidia-smi to check memory usage
            try:
                self.logger.info("Running nvidia-smi to check GPU memory...")
                result = subprocess.run(['nvidia-smi', '--query-gpu=memory.used,memory.free', '--format=csv'],
                                       capture_output=True, text=True, check=True)
                self.logger.info(f"NVIDIA-SMI report:\n{result.stdout}")
            except (subprocess.SubprocessError, FileNotFoundError) as e:
                self.logger.info(f"Could not run nvidia-smi: {str(e)}")
                
        except Exception as e:
            self.logger.warning(f"Error during CUDA memory cleanup: {str(e)}")
    
    def run_single_experiment(self, game_name: str, model_name: str, exp_id: str) -> bool:
        """Run a single experiment with the specified game and model"""
        self.logger.info(f"Starting experiment with game: {game_name}, model: {model_name}")
        
        # Start timing
        start_time = datetime.now()
        
        # Update experiment status
        self.report.update_experiment(
            exp_id, 
            status=ExperimentStatus.RUNNING,
            start_time=start_time.isoformat()
        )
        
        try:
            experiment = self.setup_experiment(game_name, model_name)
            
            # Record output directory for later reference
            output_dir = experiment.output_dir
            self.report.update_experiment(exp_id, output_dir=output_dir)
            
            # Run the experiment
            if self.exp_config['experiment'].get('run_sanity_check', False):
                experiment.run_sanity_check()
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
                time_cost_seconds=time_cost
            )
            
            self.logger.info(f"Experiment completed: {game_name}, {model_name}, Time cost: {time_cost_str}")
            
            # Clean up CUDA memory after experiment
            self.logger.info("Cleaning up CUDA memory...")
            self._clean_cuda_memory()
            
            return True
            
        except Exception as e:
            # End timing even if experiment failed
            end_time = datetime.now()
            time_cost = (end_time - start_time).total_seconds()
            
            error_trace = traceback.format_exc()
            self.logger.error(f"Experiment failed: {game_name}, {model_name}\nError: {str(e)}\n{error_trace}")
            
            # Update experiment status
            self.report.update_experiment(
                exp_id, 
                status=ExperimentStatus.FAILED,
                end_time=end_time.isoformat(),
                time_cost_seconds=time_cost,
                error=f"{str(e)}\n{error_trace}"
            )
            
            # Try to clean up CUDA memory even after failure
            self.logger.info("Attempting to clean up CUDA memory after failed experiment...")
            self._clean_cuda_memory()
            
            return False
    
    def run_experiment_series(self) -> None:
        """Run the full series of experiments with all game/model combinations"""
        # Record series start time
        series_start_time = datetime.now()
        
        # Get lists of games and models from config
        games = self.exp_config['experiment'].get('games', [])
        models = self.exp_config['experiment'].get('models', [])
        
        # Validate that games and models are specified
        if not games:
            raise ValueError(
                "No games specified in configuration. Please add games to the 'games' list in your config file. "
                "Example: games: ['Escalation_Game', 'Trust_Game_Trustor']"
            )
        
        if not models:
            raise ValueError(
                "No models specified in configuration. Please add models to the 'models' list in your config file. "
                "Example: models: ['/path/to/model1', '/path/to/model2']"
            )
        
        self.logger.info(f"Starting experiment series with {len(games)} games and {len(models)} models")
        
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
        for game_name in games:
            for model_name in models:
                # For folder names, use the formatted version
                model_folder_name = self._format_model_name_for_folder(model_name)
                exp_id = f"{game_name}_{model_folder_name}"
                
                # Only add if not resuming or not already in report
                if not self.resume or exp_id not in self.report.experiments:
                    self.report.add_experiment(
                        game_name,
                        model_name,
                        exp_id,
                        resolved_model_path=resolved_models.get(model_name),
                    )

        # Get pending experiments
        pending_experiments = self.report.get_pending_experiments()
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
                f"Running experiment {i+1}/{total_experiments}: {exp['game_name']}, {model_name}"
            )

            if not resolved_model_path:
                self.logger.error(
                    f"Skipping experiment with {exp['game_name']} and {model_name} due to missing model."
                )
                self.report.update_experiment(
                    exp['exp_id'],
                    status=ExperimentStatus.FAILED,
                    error=f"Model {model_name} could not be found or downloaded."
                )
                continue
                
            self.run_single_experiment(exp['game_name'], resolved_model_path, exp['exp_id'])
            
            # Print summary after each experiment
            summary = self.report.get_summary()
            self.logger.info(f"Experiment series progress: {summary}")
        
        # Final summary
        summary = self.report.get_summary()
        
        # Calculate and log total series time
        series_end_time = datetime.now()
        series_time_cost = (series_end_time - series_start_time).total_seconds()
        series_time_str = str(timedelta(seconds=int(series_time_cost)))
        
        self.logger.info(f"Experiment series completed. Final status: {summary}")
        self.logger.info(f"Total series time: {series_time_str}")
        self.logger.info(f"Average experiment time: {summary['formatted_avg_time']}")
        
        if summary['failed'] > 0:
            self.logger.info("Failed experiments:")
            for exp in self.report.get_failed_experiments():
                # Get time cost for failed experiment if available
                time_cost = exp.get('time_cost_seconds')
                time_info = f", time: {str(timedelta(seconds=int(time_cost)))}" if time_cost else ""
                
                self.logger.info(f"  - {exp['game_name']}, {exp['model_name']}{time_info}: {exp.get('error', 'Unknown error')[:100]}...")

def main():
    """Run the experiment series from command line"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Run an experiment series with multiple games and models')
    parser.add_argument('--config', type=str, required=True, help='Path to experiment config file')
    parser.add_argument('--name', type=str, default=None, help='Custom name for the experiment series')
    parser.add_argument('--resume', action='store_true', help='Resume interrupted experiment series')
    
    args = parser.parse_args()
    
    runner = ExperimentSeriesRunner(args.config, args.name, args.resume)
    runner.run_experiment_series()

if __name__ == "__main__":
    main() 
