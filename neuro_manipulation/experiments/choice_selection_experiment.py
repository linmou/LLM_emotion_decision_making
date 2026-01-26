#!/usr/bin/env python3
import json
import torch
import pandas as pd
import numpy as np
import logging
import re
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import itertools

from torch.utils.data import DataLoader
from vllm import LLM
from scipy.stats import chi2_contingency

from neuro_manipulation.datasets.context_manipulation_dataset import (
    ContextManipulationDataset,
    collate_context_manipulation_scenarios,
)
from neuro_manipulation.repe.rep_control_vllm_hook import RepControlVLLMHook
from neuro_manipulation.model_utils import (
    setup_model_and_tokenizer,
    load_emotion_readers,
)
from neuro_manipulation.model_layer_detector import ModelLayerDetector
from neuro_manipulation.prompt_wrapper import GameReactPromptWrapper


@dataclass
class ExperimentCondition:
    """Represents a single experimental condition in the 2x2 factorial design."""

    name: str  # e.g., 'context_only', 'activation_only'
    has_context: bool
    has_activation: bool
    activation_emotion: str = "neutral"
    activation_intensity: float = 0.0


class ChoiceSelectionExperiment:
    """
    Experiment class for measuring the selected choice option in a 2x2 factorial design.

    This experiment tests:
    - Context Factor: Present vs. Not Present
    - Activation Factor: Present vs. Not Present

    It uses RepControlVLLMHook to apply representation engineering and generates
    the model's textual choice. The chosen option is then parsed from the output.
    This allows for analyzing how context and activations interact to influence discrete choices.
    """

    def __init__(
        self,
        repe_eng_config: Dict[str, Any],
        exp_config: Dict[str, Any],
        game_config: Dict[str, Any],
        batch_size: int = 4,
        sample_num: Optional[int] = None,
    ):
        # Setup logging
        self.logger = logging.getLogger(__name__)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = f"logs/choice_selection_experiment_{timestamp}.log"

        if not self.logger.handlers:
            Path("logs").mkdir(parents=True, exist_ok=True)
            self.logger.setLevel(logging.INFO)
            # Add file handler
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(
                logging.Formatter(
                    "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
                )
            )
            self.logger.addHandler(file_handler)
            # Add console handler
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(
                logging.Formatter(
                    "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
                )
            )
            self.logger.addHandler(console_handler)
            self.logger.propagate = False

        self.logger.info(
            f"Initializing ChoiceSelectionExperiment with model: {repe_eng_config['model_name_or_path']}"
        )
        self.logger.info(f"Log file created at: {log_file}")

        self.repe_eng_config = repe_eng_config
        self.exp_config = exp_config
        self.game_config = game_config
        self.batch_size = batch_size
        self.sample_num = sample_num
        
        # Extract enable_thinking from config
        self.enable_thinking = self.exp_config["experiment"]["llm"]["generation_config"].get("enable_thinking", False)

        # --- Memory Optimization: Load standard model for setup first, then vLLM model ---

        # 1. Load a temporary standard model to set up RepReaders and get prompt format
        self.logger.info(
            "Loading temporary model for RepReader and prompt format setup..."
        )
        temp_model, temp_tokenizer, self.prompt_format = setup_model_and_tokenizer(
            repe_eng_config, from_vllm=False
        )

        # Load RepReaders for activation
        self.logger.info("Loading RepReaders for emotion activations...")
        num_layers = ModelLayerDetector.num_layers(temp_model)
        self.hidden_layers = list(range(-1, -num_layers - 1, -1))
        self.control_layers = self.hidden_layers[
            len(self.hidden_layers) // 3 : 2 * len(self.hidden_layers) // 3
        ]
        self.emotion_rep_readers = load_emotion_readers(
            self.repe_eng_config, temp_model, temp_tokenizer, self.hidden_layers, enable_thinking=self.enable_thinking
        )
        self.logger.info(
            f"Loaded {len(self.emotion_rep_readers)} emotion readers. Using control layers: {self.control_layers}"
        )

        # 2. Unload the temporary model and clear CUDA memory
        self.logger.info("Unloading temporary model and clearing CUDA cache...")
        del temp_model
        del temp_tokenizer
        torch.cuda.empty_cache()

        # 3. Load the main vLLM model for the experiment
        self.logger.info("Loading main vLLM model for experiment...")
        self.model, self.tokenizer, _ = setup_model_and_tokenizer(
            repe_eng_config, from_vllm=True
        )
        self.is_vllm = isinstance(self.model, LLM)

        if not self.is_vllm:
            raise ValueError(
                "ChoiceSelectionExperiment requires a vLLM model for generation."
            )

        # Get emotion configurations for activation
        self.target_emotion = self.exp_config.get("target_emotion", "anger")
        self.activation_intensity = self.exp_config.get(
            "activation_intensity", 1.5
        )

        # Initialize RepControlVLLMHook
        self.rep_control_hook = RepControlVLLMHook(
            model=self.model,
            tokenizer=self.tokenizer,
            layers=self.control_layers,
            control_method=self.repe_eng_config.get("control_method", "reading_vec"),
            block_name=self.repe_eng_config.get("block_name", "decoder_block"),
            tensor_parallel_size=self.repe_eng_config.get("tensor_parallel_size", 1),
        )

        # Setup prompt wrapper
        self.prompt_wrapper = GameReactPromptWrapper(
            self.prompt_format, response_format=self.game_config["decision_class"]
        )

        # Setup output directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = (
            f"{self.exp_config['output']['base_dir']}/"
            f"choice_selection_{self.exp_config['name']}_"
            f"{self.game_config['game_name']}_{self.repe_eng_config['model_name_or_path'].split('/')[-1]}_{timestamp}"
        )
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)

        # Initialize results storage
        self.results = []

    def run_experiment(self) -> str:
        """
        Run the complete 2x2 factorial experiment.

        Returns:
            Path to the saved results file
        """
        self.logger.info(
            "Starting 2x2 factorial choice selection experiment (Context vs. Activation)"
        )

        # Generate all experimental conditions
        conditions = self._generate_experimental_conditions()
        self.logger.info(
            f"Generated {len(conditions)} experimental conditions: {[c.name for c in conditions]}"
        )

        # Create a single dataset and slice it
        self.logger.info("Creating shared dataset for all conditions...")
        shared_scenarios = self._create_shared_dataset()
        self.logger.info(f"Created {len(shared_scenarios)} scenarios to be used across all conditions")

        # Run experiment for each condition using the same scenarios
        for condition in conditions:
            self.logger.info(f"Running condition: {condition.name}")
            condition_results = self._run_single_condition_with_shared_data(condition, shared_scenarios)
            self.results.extend(condition_results)

        # Save results
        results_file = self._save_results()

        # Run statistical analysis
        self._run_statistical_analysis(results_file)

        self.logger.info(f"Experiment completed. Results saved to: {results_file}")
        return results_file

    def _generate_experimental_conditions(self) -> List[ExperimentCondition]:
        """Generate all conditions for the 2x2 factorial design."""

        conditions = [
            ExperimentCondition(
                name="baseline",
                has_context=False,
                has_activation=False,
            ),
            ExperimentCondition(
                name="context_only",
                has_context=True,
                has_activation=False,
            ),
            ExperimentCondition(
                name="activation_only",
                has_context=False,
                has_activation=True,
                activation_emotion=self.target_emotion,
                activation_intensity=self.activation_intensity,
            ),
            ExperimentCondition(
                name="context_and_activation",
                has_context=True,
                has_activation=True,
                activation_emotion=self.target_emotion,
                activation_intensity=self.activation_intensity,
            ),
        ]
        return conditions

    def _create_shared_dataset(self) -> List[Dict[str, Any]]:
        """Create a shared dataset that will be used across all conditions."""
        # Create a baseline dataset to extract scenarios
        user_message = "You are participating in this scenario."
        
        from functools import partial
        neutral_prompt_wrapper = partial(
            self.prompt_wrapper.__call__,
            user_messages=user_message
        )
        
        # Create dataset with full description to get all scenario information
        dataset = ContextManipulationDataset(
            game_config=self.game_config,
            prompt_wrapper=neutral_prompt_wrapper,
            include_description=True,  # Get full scenario info
            sample_num=self.sample_num
        )
        
        # Extract all scenarios from the dataset
        scenarios = []
        for idx in range(len(dataset)):
            item = dataset[idx]
            scenarios.append({
                'idx': idx,
                'scenario': item['scenario'],
                'description': item['description'],
                'behavior_choices': item['behavior_choices'],
                'options': item['options']
            })
        
        return scenarios
    
    def _run_single_condition_with_shared_data(
        self, 
        condition: ExperimentCondition, 
        shared_scenarios: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Run experiment for a single condition using shared scenario data."""
        condition_results = []
        
        # Process scenarios in batches
        for batch_start in range(0, len(shared_scenarios), self.batch_size):
            batch_end = min(batch_start + self.batch_size, len(shared_scenarios))
            batch_scenarios = shared_scenarios[batch_start:batch_end]
            
            # Prepare batch data with appropriate context manipulation
            batch = self._prepare_batch_for_condition(batch_scenarios, condition)
            
            # Generate text and parse choices for this batch
            batch_results = self._generate_and_parse_choices(batch, condition)
            
            # Add scenario index to results for matching across conditions
            for i, result in enumerate(batch_results):
                result['scenario_idx'] = batch_scenarios[i]['idx']
            
            condition_results.extend(batch_results)
            
        return condition_results
    
    def _prepare_batch_for_condition(
        self, 
        batch_scenarios: List[Dict[str, Any]], 
        condition: ExperimentCondition
    ) -> Dict[str, List]:
        """Prepare batch data with appropriate prompt formatting based on condition."""
        batch = {
            'prompt': [],
            'options': [],
            'scenario': [],
            'behavior_choices': []
        }
        
        user_message = "You are participating in this scenario."
        
        for scenario_data in batch_scenarios:
            # Create prompt based on condition
            if condition.has_context:
                # Include full description
                prompt = self.prompt_wrapper(
                    event=f"Scenario: {scenario_data['scenario']}\nDescription: {scenario_data['description']}\nBehavior Choices: {scenario_data['behavior_choices']}",
                    options=scenario_data['options'],
                    user_messages=user_message
                )
            else:
                # Minimal prompt without description
                prompt = self.prompt_wrapper(
                    event=f"Scenario: {scenario_data['scenario']}\nBehavior Choices: {scenario_data['behavior_choices']}",
                    options=scenario_data['options'],
                    user_messages=user_message
                )
            
            batch['prompt'].append(prompt)
            batch['options'].append(scenario_data['options'])
            batch['scenario'].append(scenario_data['scenario'])
            batch['behavior_choices'].append(str(scenario_data['behavior_choices']))
        
        return batch

    def _get_control_activations(
        self, emotion: str, intensity: float
    ) -> Dict[int, torch.Tensor]:
        """Get control activations for a given emotion and intensity."""
        if emotion not in self.emotion_rep_readers:
            raise ValueError(f"Emotion '{emotion}' not found in loaded RepReaders.")

        rep_reader = self.emotion_rep_readers[emotion]

        activations = {
            layer: torch.tensor(
                intensity
                * rep_reader.directions[layer]
                * rep_reader.direction_signs[layer]
            )
            .cpu()
            .half()
            for layer in self.control_layers
        }
        return activations

    def _run_single_condition(
        self, condition: ExperimentCondition
    ) -> List[Dict[str, Any]]:
        """Run experiment for a single condition."""
        # Create dataset for this condition
        dataset = self._create_dataset_for_condition(condition)

        # Create data loader
        data_loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=collate_context_manipulation_scenarios,
        )

        condition_results = []

        for batch_idx, batch in enumerate(data_loader):
            self.logger.debug(
                f"Processing batch {batch_idx} for condition {condition.name}"
            )

            # Generate text and parse choices for this batch
            batch_results = self._generate_and_parse_choices(batch, condition)
            condition_results.extend(batch_results)

        return condition_results

    def _create_dataset_for_condition(
        self, condition: ExperimentCondition
    ) -> ContextManipulationDataset:
        """Create dataset with appropriate context settings for a condition."""
        from functools import partial

        # The prompt is always neutral, as emotion is handled by activation vectors
        user_message = "You are participating in this scenario."
        neutral_prompt_wrapper = partial(
            self.prompt_wrapper.__call__, user_messages=user_message
        )

        # Create dataset with context manipulation based on the condition
        dataset = ContextManipulationDataset(
            game_config=self.game_config,
            prompt_wrapper=neutral_prompt_wrapper,
            include_description=condition.has_context,
            sample_num=self.sample_num,
        )

        return dataset

    def _generate_and_parse_choices(
        self, batch: Dict[str, List], condition: ExperimentCondition
    ) -> List[Dict[str, Any]]:
        """Generate text responses for a batch and parse the chosen option."""
        prompts = batch["prompt"]
        options_list = batch["options"]

        batch_results = []

        activations = None
        if condition.has_activation:
            activations = self._get_control_activations(
                condition.activation_emotion, condition.activation_intensity
            )

        # Use the hook's __call__ method for generation
        outputs = self.rep_control_hook(
            text_inputs=prompts,
            activations=activations,
            max_new_tokens=self.exp_config.get("max_new_tokens", 50),
            temperature=self.exp_config.get("temperature", 0.0),
        )

        for i, (output, options) in enumerate(zip(outputs, options_list)):
            generated_text = output.outputs[0].text
            chosen_option_id, chosen_option_text = self._parse_choice(
                generated_text, options
            )

            result = {
                "condition_name": condition.name,
                "has_context": condition.has_context,
                "has_activation": condition.has_activation,
                "activation_emotion": condition.activation_emotion,
                "activation_intensity": condition.activation_intensity,
                "prompt": prompts[i],
                "generated_text": generated_text,
                "chosen_option_id": chosen_option_id,
                "chosen_option_text": chosen_option_text,
                "scenario": batch["scenario"][i],
                "behavior_choices": batch["behavior_choices"][i],
                "options": options,
                "batch_idx": i,
            }
            batch_results.append(result)

        return batch_results

    def _parse_choice(
        self, generated_text: str, options: List[str]
    ) -> Tuple[Optional[int], Optional[str]]:
        """Parse the chosen option from the generated text."""
        # Case-insensitive search
        text_lower = generated_text.lower()

        # Try to find "Option X"
        match = re.search(r"option\s+(\d+)", text_lower)
        if match:
            option_id = int(match.group(1))
            if 1 <= option_id <= len(options):
                return option_id, options[option_id - 1]

        # If not found, check if the option text itself is in the response
        for i, option_text in enumerate(options):
            # Extract the core text of the option (after "Option X. ")
            core_text = option_text.split(".", 1)[-1].strip()
            if core_text.lower() in text_lower:
                return i + 1, option_text

        self.logger.warning(
            f"Could not parse a valid choice from text: '{generated_text}'"
        )
        return None, None

    def _save_results(self) -> str:
        """Save experiment results to JSON and CSV files."""
        # Save raw results as JSON
        json_file = f"{self.output_dir}/choice_selection_results.json"
        with open(json_file, "w") as f:
            json.dump(self.results, f, indent=2)

        # Convert to DataFrame for CSV
        df = pd.DataFrame(self.results)
        csv_file = f"{self.output_dir}/choice_selection_results.csv"
        df.to_csv(csv_file, index=False)

        self.logger.info(f"Results saved to {json_file} and {csv_file}")
        return csv_file

    def _run_statistical_analysis(self, csv_file: str):
        """Run statistical analysis on the choice data."""
        self.logger.info(
            "Running statistical analysis for context-activation interaction effects on choice"
        )
        try:
            df = pd.read_csv(csv_file)
            if df.empty:
                self.logger.warning(f"CSV file is empty, skipping analysis: {csv_file}")
                return
        except pd.errors.EmptyDataError:
            self.logger.warning(f"CSV file is empty, skipping analysis: {csv_file}")
            return

        # Ensure chosen_option_id is numeric, coerce errors to NaN and drop them
        df["chosen_option_id"] = pd.to_numeric(df["chosen_option_id"], errors="coerce")
        df.dropna(subset=["chosen_option_id"], inplace=True)
        df["chosen_option_id"] = df["chosen_option_id"].astype(int)

        # Contingency table: chosen_option_id vs. condition_name
        contingency_table = pd.crosstab(
            df["chosen_option_id"], df["condition_name"]
        )

        analysis_results = {
            "choice_counts_by_condition": contingency_table.to_dict(),
            "chi_square_test": {},
        }

        if contingency_table.shape[0] > 1 and contingency_table.shape[1] > 1:
            try:
                chi2, p_value, dof, expected = chi2_contingency(contingency_table)
                analysis_results["chi_square_test"] = {
                    "chi_square_statistic": chi2,
                    "p_value": p_value,
                    "degrees_of_freedom": dof,
                    "is_significant": p_value < 0.05,
                    "message": "A significant p-value (< 0.05) suggests that the choice of option is dependent on the experimental condition.",
                }
            except Exception as e:
                self.logger.error(f"Chi-square test failed: {e}")
                analysis_results["chi_square_test"] = {"error": str(e)}
        else:
            self.logger.warning("Contingency table too small for Chi-square test.")
            analysis_results["chi_square_test"] = {
                "error": "Not enough data or variance in choices/conditions to perform the test."
            }

        # Convert numpy types to native Python types for JSON serialization
        analysis_results_serializable = self._convert_to_json_serializable(
            analysis_results
        )

        # Save analysis results
        analysis_file = Path(self.output_dir) / "statistical_analysis.json"
        with open(analysis_file, "w") as f:
            json.dump(analysis_results_serializable, f, indent=4)

        self.logger.info(f"Statistical analysis saved to {analysis_file}")
        self._generate_summary_report(analysis_results, df)

    def _convert_to_json_serializable(self, obj: Any) -> Any:
        """Recursively convert numpy types to native Python types for JSON serialization."""
        if isinstance(obj, (np.integer, np.int_)):
            return int(obj)
        if isinstance(obj, (np.floating, np.float_)):
            return float(obj)
        if isinstance(obj, np.bool_):
            return bool(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, dict):
            return {
                key: self._convert_to_json_serializable(value)
                for key, value in obj.items()
            }
        if isinstance(obj, list):
            return [self._convert_to_json_serializable(item) for item in obj]
        return obj

    def _generate_summary_report(self, analysis_results: Dict, df: pd.DataFrame):
        """Generate a human-readable summary report."""
        report_file = f"{self.output_dir}/experiment_summary_report.txt"
        with open(report_file, "w") as f:
            f.write("CHOICE SELECTION EXPERIMENT SUMMARY REPORT\n")
            f.write("=" * 50 + "\n\n")

            f.write(f"Experiment: {self.exp_config['name']}\n")
            f.write(f"Game: {self.game_config['game_name']}\n")
            f.write(f"Model: {self.repe_eng_config['model_name_or_path']}\n\n")

            f.write("--- Choice Counts by Condition ---\n")
            f.write(
                str(analysis_results.get("choice_counts_by_condition", "N/A")) + "\n\n"
            )

            f.write("--- Chi-Square Test for Independence ---\n")
            chi2_res = analysis_results.get("chi_square_test", {})
            if "error" in chi2_res:
                f.write(f"Error: {chi2_res['error']}\n")
            else:
                f.write(f"Chi-Square Statistic: {chi2_res.get('chi_square_statistic', 'N/A'):.4f}\n")
                f.write(f"P-value: {chi2_res.get('p_value', 'N/A'):.4f}\n")
                f.write(f"Degrees of Freedom: {chi2_res.get('degrees_of_freedom', 'N/A')}\n")
                f.write(f"Result is significant (p < 0.05): {chi2_res.get('is_significant', 'N/A')}\n")
                f.write(f"Note: {chi2_res.get('message', '')}\n")

        self.logger.info(f"Summary report saved to {report_file}")
 