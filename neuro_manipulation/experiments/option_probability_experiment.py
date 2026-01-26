import json
import torch
import pandas as pd
import numpy as np
import logging
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import itertools

from torch.utils.data import DataLoader
from vllm import LLM, SamplingParams

from neuro_manipulation.datasets.context_manipulation_dataset import (
    ContextManipulationDataset,
    collate_context_manipulation_scenarios
)
from neuro_manipulation.repe.sequence_prob_vllm_hook import CombinedVLLMHook
from neuro_manipulation.model_utils import setup_model_and_tokenizer, load_emotion_readers
from neuro_manipulation.model_layer_detector import ModelLayerDetector
from neuro_manipulation.prompt_wrapper import GameReactPromptWrapper
from neuro_manipulation.repe.pipelines import get_pipeline


@dataclass
class ExperimentCondition:
    """Represents a single experimental condition in the 2x2 factorial design."""
    name: str  # e.g., 'context_only', 'activation_only'
    has_context: bool
    has_activation: bool
    activation_emotion: str = 'neutral'
    activation_intensity: float = 0.0


class OptionProbabilityExperiment:
    """
    Experiment class for measuring choice probabilities across all options in a 2x2 factorial design.
    
    This experiment tests:
    - Context Factor: Present vs. Not Present
    - Activation Factor: Present vs. Not Present
    
    Uses CombinedVLLMHook to measure the probability of each behavioral choice option,
    enabling analysis of how context (prompt engineering) and activation (representation engineering) 
    interact to influence decision probabilities.
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
        log_file = f"logs/option_probability_experiment_{timestamp}.log"
        
        if not self.logger.handlers:
            Path("logs").mkdir(parents=True, exist_ok=True)
            self.logger.setLevel(logging.INFO)
            # Add file handler
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(
                logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
            )
            self.logger.addHandler(file_handler)
            # Add console handler  
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(
                logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
            )
            self.logger.addHandler(console_handler)
            self.logger.propagate = False
            
        self.logger.info(f"Initializing OptionProbabilityExperiment with model: {repe_eng_config['model_name_or_path']}")
        self.logger.info(f"Log file created at: {log_file}")
        
        self.repe_eng_config = repe_eng_config
        self.exp_config = exp_config
        self.game_config = game_config
        self.batch_size = batch_size
        self.sample_num = sample_num
        
        # Get emotion configurations for activation
        self.target_emotion = self.exp_config["experiment"].get("target_emotion", "anger")
        self.activation_intensity = self.exp_config["experiment"].get("activation_intensity", 1.5)
        self.enable_thinking = self.exp_config["experiment"]["llm"]["generation_config"].get("enable_thinking", False)
        
        # Step 1: Load a temporary non-vLLM model to get layer info and load RepReaders
        self.logger.info("Loading temporary model for RepReader initialization...")
        temp_model, temp_tokenizer, self.prompt_format = setup_model_and_tokenizer(repe_eng_config, from_vllm=False)
        
        # Get layer information from the temporary model
        num_layers = ModelLayerDetector.num_layers(temp_model)
        self.hidden_layers = list(range(-1, -num_layers - 1, -1))
        self.control_layers = self.hidden_layers[len(self.hidden_layers) // 3 : 2 * len(self.hidden_layers) // 3]

        # Load RepReaders using the temporary model
        self.logger.info("Loading RepReaders for emotion activations...")
        self.emotion_rep_readers = load_emotion_readers(
            self.repe_eng_config, temp_model, temp_tokenizer, self.hidden_layers, enable_thinking=self.enable_thinking
        )
        self.logger.info(f"Loaded {len(self.emotion_rep_readers)} emotion readers. Using control layers: {self.control_layers}")

        # Step 2: Clean up the temporary model to free up memory
        self.logger.info("Releasing temporary model from memory...")
        del temp_model
        del temp_tokenizer
        torch.cuda.empty_cache()

        # Step 3: Setup the main vLLM model for the experiment
        self.logger.info("Initializing vLLM for the main experiment...")
        self.model, self.tokenizer, _ = setup_model_and_tokenizer(
            repe_eng_config, from_vllm=True
        )
        self.is_vllm = isinstance(self.model, LLM)
        
        if not self.is_vllm:
            raise ValueError("OptionProbabilityExperiment requires a vLLM model for sequence probability measurement")
        
        # Initialize CombinedVLLMHook with both sequence probability and rep control enabled
        self.sequence_prob_hook = CombinedVLLMHook(
            model=self.model,
            tokenizer=self.tokenizer,
            layers=self.control_layers,
            block_name=self.repe_eng_config.get("block_name", "decoder_block"),
            tensor_parallel_size=self.repe_eng_config.get('tensor_parallel_size', 1),
            enable_sequence_prob=True,
            enable_rep_control=True, # Enable representation control
            enable_layer_logit_recording=False
        )
        
        # Setup prompt wrapper (will use a neutral prompt for all conditions)
        self.prompt_wrapper = GameReactPromptWrapper(
            self.prompt_format, 
            response_format=self.game_config["decision_class"]
        )
        
        # Setup output directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = (
            f"{self.exp_config['experiment']['output']['base_dir']}/"
            f"option_probability_{self.exp_config['experiment']['name']}_"
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
        self.logger.info("Starting 2x2 factorial option probability experiment (Context vs. Activation)")
        
        # Generate all experimental conditions
        conditions = self._generate_experimental_conditions()
        self.logger.info(f"Generated {len(conditions)} experimental conditions: {[c.name for c in conditions]}")
        
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
            )
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
            
            # Measure probabilities for all options in this batch
            batch_results = self._measure_option_probabilities(batch, condition)
            
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
        
    def _get_control_activations(self, emotion: str, intensity: float) -> Dict[int, torch.Tensor]:
        """Get control activations for a given emotion and intensity."""
        if emotion not in self.emotion_rep_readers:
            raise ValueError(f"Emotion '{emotion}' not found in loaded RepReaders.")
        
        rep_reader = self.emotion_rep_readers[emotion]
        
        activations = {
            layer: torch.tensor(
                intensity * rep_reader.directions[layer] * rep_reader.direction_signs[layer]
            ).cpu().half()
            for layer in self.control_layers
        }
        return activations

    def _run_single_condition(self, condition: ExperimentCondition) -> List[Dict[str, Any]]:
        """Run experiment for a single condition."""
        # Create dataset for this condition
        dataset = self._create_dataset_for_condition(condition)
        
        # Create data loader
        data_loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=collate_context_manipulation_scenarios
        )
        
        condition_results = []
        
        for batch_idx, batch in enumerate(data_loader):
            self.logger.debug(f"Processing batch {batch_idx} for condition {condition.name}")
            
            # Measure probabilities for all options in this batch
            batch_results = self._measure_option_probabilities(batch, condition)
            condition_results.extend(batch_results)
            
        return condition_results
        
    def _create_dataset_for_condition(self, condition: ExperimentCondition) -> ContextManipulationDataset:
        """Create dataset with appropriate context settings for a condition."""
        # The prompt is always neutral, as emotion is handled by activation vectors
        user_message = "You are participating in this scenario."
            
        from functools import partial
        neutral_prompt_wrapper = partial(
            self.prompt_wrapper.__call__,
            user_messages=user_message
        )
        
        # Create dataset with context manipulation based on the condition
        dataset = ContextManipulationDataset(
            game_config=self.game_config,
            prompt_wrapper=neutral_prompt_wrapper,
            include_description=condition.has_context,
            sample_num=self.sample_num
        )
        
        return dataset
        
    def _measure_option_probabilities(
        self, 
        batch: Dict[str, List], 
        condition: ExperimentCondition
    ) -> List[Dict[str, Any]]:
        """Measure probabilities for all options in a batch, applying activations if needed."""
        prompts = batch['prompt']
        options_list = batch['options']
        
        batch_results = []
        
        activations = None
        if condition.has_activation:
            activations = self._get_control_activations(
                condition.activation_emotion, condition.activation_intensity
            )

        try:
            # Set control activations on the hook if condition requires it
            if activations:
                self.logger.debug(f"Applying '{condition.activation_emotion}' activation for condition '{condition.name}'")
                self.sequence_prob_hook._set_control_activations(activations)

            for i, (prompt, options) in enumerate(zip(prompts, options_list)):
                # Extract option texts for probability measurement
                option_texts = [opt.split('. ', 1)[1] for opt in options]  # Remove "Option 1. " prefix
                
                # Use CombinedVLLMHook to get probabilities
                prob_results = self.sequence_prob_hook.get_log_prob(
                    text_inputs=[prompt],
                    target_sequences=option_texts
                )
                
                if prob_results and len(prob_results) > 0:
                    # Convert log probabilities to regular probabilities
                    log_probs = {}
                    probs = {}
                    
                    for option_text in option_texts:
                        # Find matching result
                        for prob_result in prob_results:
                            if prob_result['sequence'] == option_text:
                                log_probs[option_text] = prob_result['log_prob']
                                probs[option_text] = prob_result['prob']
                                break
                    
                    if not log_probs:
                        self.logger.warning(f"Could not calculate any valid probabilities for prompt {i}")
                        continue

                    # Normalize probabilities to sum to 1
                    total_prob = sum(probs.values())
                    if total_prob > 0:
                        probs = {seq: prob / total_prob for seq, prob in probs.items()}
                    
                    # Store result
                    result = {
                        'condition_name': condition.name,
                        'has_context': condition.has_context,
                        'has_activation': condition.has_activation,
                        'activation_emotion': condition.activation_emotion,
                        'activation_intensity': condition.activation_intensity,
                        'prompt': prompt,
                        'scenario': batch['scenario'][i],
                        'behavior_choices': batch['behavior_choices'][i],
                        'option_probabilities': probs,
                        'log_probabilities': log_probs,
                        'options': options,
                        'batch_idx': i
                    }
                    batch_results.append(result)
                    
                else:
                    self.logger.warning(f"No probability results for prompt {i} in condition {condition.name}")
        
        finally:
            # IMPORTANT: Clear control activations after processing the batch
            if activations:
                self.logger.debug("Clearing activations.")
                self.sequence_prob_hook._clear_control_activations()
                
        return batch_results
        
    def _save_results(self) -> str:
        """Save experiment results to JSON and CSV files."""
        # Save raw results as JSON
        json_file = f"{self.output_dir}/option_probability_results.json"
        with open(json_file, 'w') as f:
            json.dump(self.results, f, indent=2)
            
        # Convert to structured DataFrame for analysis
        structured_data = []
        
        for result in self.results:
            base_row = {
                'condition_name': result['condition_name'],
                'has_context': result['has_context'],
                'has_activation': result['has_activation'],
                'activation_emotion': result['activation_emotion'],
                'activation_intensity': result['activation_intensity'],
                'scenario': result['scenario'],
                'behavior_choices': result['behavior_choices']
            }
            
            # Add probability for each option
            option_probs = result['option_probabilities']
            for i, (option_text, prob) in enumerate(option_probs.items()):
                row = base_row.copy()
                row.update({
                    'option_id': i + 1,
                    'option_text': option_text,
                    'probability': prob,
                    'log_probability': result['log_probabilities'].get(option_text, float('-inf'))
                })
                structured_data.append(row)
                
        # Save as CSV
        df = pd.DataFrame(structured_data)
        csv_file = f"{self.output_dir}/option_probability_results.csv"
        df.to_csv(csv_file, index=False)
        
        self.logger.info(f"Results saved to {json_file} and {csv_file}")
        return csv_file
        
    def _convert_keys_to_str(self, obj):
        """Recursively convert all keys in a dictionary to strings."""
        if isinstance(obj, dict):
            return {str(k): self._convert_keys_to_str(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_keys_to_str(i) for i in obj]
        else:
            return obj

    def _run_statistical_analysis(self, csv_file: str):
        """Run statistical analysis on the probability data."""
        self.logger.info("Running statistical analysis for context-activation interaction effects")

        try:
            df = pd.read_csv(csv_file)
            if df.empty:
                self.logger.warning(f"CSV file is empty, skipping analysis: {csv_file}")
                return
        except pd.errors.EmptyDataError:
            self.logger.warning(f"CSV file is empty, skipping analysis: {csv_file}")
            return
        
        # Analysis results storage
        analysis_results = {
            'summary_statistics': {},
            'interaction_effects': {},
            'pairwise_comparisons': {}
        }
        
        # Summary statistics by condition
        summary_stats = df.groupby(['condition_name', 'has_context', 'has_activation', 'option_id']).agg({
            'probability': ['mean', 'std', 'count']
        }).round(4)
        
        if summary_stats.empty:
            self.logger.warning("No data available for summary statistics. Skipping.")
        else:
            analysis_results['summary_statistics'] = summary_stats.reset_index().to_dict(orient='records')
        
        # Test for interaction effects using probability data
        for option_id in df['option_id'].unique():
            option_data = df[df['option_id'] == option_id]
            
            if option_data.empty:
                continue

            # Create contingency table for this option
            # We'll test if high probability choices (>median) interact with conditions
            median_prob = option_data['probability'].median()
            option_data['high_prob'] = option_data['probability'] > median_prob
            
            contingency = pd.crosstab(
                index=[option_data['has_context'], option_data['has_activation']], 
                columns=option_data['high_prob']
            )
            
            if contingency.shape[0] >= 2 and contingency.shape[1] >= 2:
                try:
                    # Chi-square test for independence
                    chi2, p_value, dof, expected = chi2_contingency(contingency)
                    
                    analysis_results['interaction_effects'][f'option_{option_id}'] = {
                        'chi_square': float(chi2),
                        'p_value': float(p_value),
                        'degrees_of_freedom': int(dof),
                        'contingency_table': [
                            {str(k): v for k, v in row.items()} 
                            for row in contingency.reset_index().to_dict('records')
                        ]
                    }
                    
                except Exception as e:
                    self.logger.warning(f"Could not run chi-square test for option {option_id}: {e}")
        
        # Pairwise comparisons between conditions
        for option_id in df['option_id'].unique():
            option_data = df[df['option_id'] == option_id]

            if option_data.empty:
                continue
            
            pairwise_results = {}
            conditions = option_data.groupby(['condition_name'])
            
            if len(conditions) < 2:
                continue

            for (cond1_name, cond1_data), (cond2_name, cond2_data) in itertools.combinations(conditions, 2):
                try:
                    # Mann-Whitney U test for probability distributions
                    from scipy.stats import mannwhitneyu
                    statistic, p_value = mannwhitneyu(
                        cond1_data['probability'], 
                        cond2_data['probability'], 
                        alternative='two-sided'
                    )
                    
                    comparison_key = f"{cond1_name}_vs_{cond2_name}"
                    pairwise_results[comparison_key] = {
                        'mann_whitney_u': float(statistic),
                        'p_value': float(p_value),
                        'cond1_mean': float(cond1_data['probability'].mean()),
                        'cond2_mean': float(cond2_data['probability'].mean())
                    }
                    
                except Exception as e:
                    self.logger.warning(f"Could not run pairwise comparison for option {option_id}: {e}")
                    
            analysis_results['pairwise_comparisons'][f'option_{option_id}'] = pairwise_results

        # Save analysis results
        analysis_file = Path(self.output_dir) / "statistical_analysis.json"
        
        safe_results = self._convert_keys_to_str(analysis_results)
        with open(analysis_file, 'w') as f:
            json.dump(safe_results, f, indent=2)
            
        self.logger.info(f"Statistical analysis saved to {analysis_file}")
        
        self._generate_summary_report(analysis_results)

    def _generate_summary_report(self, analysis_results: Dict[str, Any]):
        """Generate a human-readable summary report."""
        report_file = f"{self.output_dir}/experiment_summary_report.txt"
        
        with open(report_file, 'w') as f:
            f.write("OPTION PROBABILITY EXPERIMENT SUMMARY REPORT\n")
            f.write("=" * 50 + "\n\n")
            
            f.write(f"Experiment: {self.exp_config['experiment']['name']}\n")
            f.write(f"Game: {self.game_config['game_name']}\n")
            f.write(f"Model: {self.repe_eng_config['model_name_or_path']}\n")
            f.write(f"Total Results: {len(self.results)}\n\n")
            
            f.write("EXPERIMENTAL DESIGN: 2x2 Factorial\n")
            f.write("- Factor 1: Context (Present vs. Not Present)\n")
            f.write(f"- Factor 2: Activation (Present vs. Not Present) using '{self.target_emotion}' emotion\n\n")
            
            f.write("KEY FINDINGS:\n")
            
            f.write("Interaction Effects (p < 0.05):\n")
            significant_effects = []
            if 'interaction_effects' in analysis_results and analysis_results['interaction_effects']:
                for option, results in analysis_results['interaction_effects'].items():
                    if results['p_value'] < 0.05:
                        significant_effects.append(f"  - {option}: χ² = {results['chi_square']:.3f}, p = {results['p_value']:.3f}")
            
            if significant_effects:
                f.write("\n".join(significant_effects) + "\n\n")
            else:
                f.write("  No significant interaction effects found.\n\n")
                
            f.write("For detailed statistics, see statistical_analysis.json\n")
            
        self.logger.info(f"Summary report saved to {report_file}")


if __name__ == "__main__":
    from games.game_configs import get_game_config
    from constants import GameNames
    
    game_config = get_game_config(GameNames.PRISONERS_DILEMMA)
    
    repe_eng_config = {
        'model_name_or_path': 'meta-llama/Llama-3.1-8B-Instruct',
        'tensor_parallel_size': 1,
        'coeffs': [0.0, 1.0],
        'block_name': 'decoder_block',
        'control_method': 'reading_vec'
    }
    
    exp_config = {
        'experiment': {
            'name': 'context_activation_probability_test',
            'target_emotion': 'angry',
            'activation_intensity': 1.0,
            'output': {
                'base_dir': 'experiments/option_probability'
            }
        }
    }
    
    experiment = OptionProbabilityExperiment(
        repe_eng_config=repe_eng_config,
        exp_config=exp_config,
        game_config=game_config,
        batch_size=2,
        sample_num=10
    )
    
    results_file = experiment.run_experiment()
    print(f"Experiment completed. Results: {results_file}")