#!/usr/bin/env python
"""
Script to run a series of experiments with different combinations of games and models.
"""
import argparse
import time
import logging
from pathlib import Path

from neuro_manipulation.experiment_series_runner import ExperimentSeriesRunner

def setup_logging():
    """Set up logging for the script"""
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # Create file handler
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    log_file = log_dir / f"experiment_series_{time.strftime('%Y%m%d_%H%M%S')}.log"
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    return logger

def main():
    """Main function to run experiment series"""
    logger = setup_logging()
    
    parser = argparse.ArgumentParser(description="Run a series of experiments with multiple games and models")
    parser.add_argument("--config", type=str, default="config/experiment_series_config.yaml",
                        help="Path to the experiment series config file")
    parser.add_argument("--name", type=str, default=None,
                        help="Custom name for the experiment series")
    parser.add_argument("--resume", action="store_true",
                        help="Resume previously interrupted experiment series")
    
    args = parser.parse_args()
    
    logger.info(f"Starting experiment series with config: {args.config}")
    logger.info(f"Series name: {args.name if args.name else 'default'}")
    logger.info(f"Resume mode: {args.resume}")
    
    start_time = time.time()
    
    try:
        # Create and run the experiment series
        runner = ExperimentSeriesRunner(
            config_path=args.config,
            series_name=args.name,
            resume=args.resume
        )
        runner.run_experiment_series()
        
    except Exception as e:
        logger.exception(f"Error running experiment series: {e}")
        return 1
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    
    logger.info(f"Experiment series completed in {elapsed_time:.2f} seconds ({elapsed_time/60:.2f} minutes)")
    return 0

if __name__ == "__main__":
    exit(main()) 