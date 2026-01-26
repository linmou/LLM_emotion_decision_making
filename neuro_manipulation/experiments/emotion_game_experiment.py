import json
import logging
import re
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from functools import partial
from pathlib import Path
from queue import Queue
from threading import Thread, current_thread

import pandas as pd
import torch
import yaml
from openai import AzureOpenAI, OpenAI
from pydantic import BaseModel
from torch.utils.data import DataLoader
from transformers import pipeline
from vllm import LLM

from api_configs import AZURE_OPENAI_CONFIG, OAI_CONFIG
from neuro_manipulation.datasets.game_scenario_dataset import (
    GameScenarioDataset,
    collate_game_scenarios,
)
from neuro_manipulation.model_layer_detector import ModelLayerDetector
from neuro_manipulation.model_utils import (
    load_emotion_readers,
    setup_model_and_tokenizer,
)
from neuro_manipulation.prompt_wrapper import GameReactPromptWrapper
from neuro_manipulation.repe.pipelines import get_pipeline
from neuro_manipulation.utils import oai_response
from statistical_engine import analyze_emotion_and_intensity_effects


# Define the RPC function at the module level to avoid pickling issues with 'self'
def _get_worker_num_layers_rpc(worker_self):
    """RPC function to get the number of layers from the worker's model."""
    try:
        if hasattr(worker_self, "model_runner") and hasattr(
            worker_self.model_runner, "model"
        ):
            model = worker_self.model_runner.model
            # Get config attribute which should contain num_layers or n_layer
            config = getattr(model, "config", None)
            if config:
                # Try common attribute names for number of layers
                for attr in ["num_hidden_layers", "n_layer", "num_layers"]:
                    if hasattr(config, attr):
                        return getattr(config, attr)
            # If config approach failed, try to count transformer layers directly
            if hasattr(model, "model") and hasattr(model.model, "layers"):
                return len(model.model.layers)
            elif hasattr(model, "layers"):
                return len(model.layers)
        return -1  # Return -1 to indicate failure
    except Exception:
        return -1  # Return -1 on any error


class ExtractedResultWithMethod(BaseModel):
    option_id: int
    rationale: str
    decision: str
    extraction_method: str


class ExtractedResult(BaseModel):
    option_id: int
    rationale: str
    decision: str


class EmotionGameExperiment:
    def __init__(
        self,
        repe_eng_config,
        exp_config,
        game_config,
        batch_size,
        sample_num=None,
        repeat=1,
    ):
        # Setup logging

        # Create a module-specific logger
        self.logger = logging.getLogger(__name__)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = f"logs/emotion_game_experiment_{timestamp}.log"

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
            # Prevent propagation to avoid duplicate logs
            self.logger.propagate = False

        self.logger.info(
            f"Initializing experiment with model: {repe_eng_config['model_name_or_path']}"
        )
        self.logger.info(f"Log file created at: {log_file}")
        self.repe_eng_config = repe_eng_config
        self.exp_config = exp_config
        self.generation_config = exp_config["experiment"]["llm"]["generation_config"]
        self.enable_thinking = self.generation_config.get("enable_thinking", False)
        self.game_config = game_config

        self.repeat = repeat
        self.sample_num = sample_num

        self.batch_size = batch_size

        self.model, self.tokenizer, self.prompt_format, processor = (
            setup_model_and_tokenizer(repe_eng_config, from_vllm=False)
        )  # first load from hf for load_emotion_readers since load_emotion_readers does not support vllm yet TODO: update load_emotion_readers to support vllm
        num_hidden_layers = ModelLayerDetector.num_layers(self.model)
        self.hidden_layers = list(range(-1, -num_hidden_layers - 1, -1))
        self.logger.info(f"Using hidden layers: {self.hidden_layers}")

        self.emotion_rep_readers = load_emotion_readers(
            self.repe_eng_config,
            self.model,
            self.tokenizer,
            self.hidden_layers,
            processor,
            self.enable_thinking,
        )
        del self.model  # to save memory
        self.model, self.tokenizer, self.prompt_format, _ = setup_model_and_tokenizer(
            repe_eng_config, from_vllm=True
        )  # load from vllm for the rest of the experiment

        self.logger.info(
            f"Model: {self.model} loaded from {repe_eng_config['model_name_or_path']}, type: {type(self.model)}"
        )
        self.is_vllm = isinstance(self.model, LLM)

        # Get number of layers
        if self.is_vllm:
            try:
                results = self.model.llm_engine.collective_rpc(
                    _get_worker_num_layers_rpc
                )
                if not results or all(r == -1 for r in results):
                    raise RuntimeError("Failed to get number of layers from any worker")
                num_hidden_layers = next(r for r in results if r > 0)
                self.logger.info(
                    f"Retrieved {num_hidden_layers} layers from vLLM worker"
                )
            except Exception as e:
                self.logger.error(f"Error getting layers from vLLM: {e}")
                raise
        else:
            num_hidden_layers = ModelLayerDetector.num_layers(self.model)

        if num_hidden_layers <= 0:
            raise ValueError(f"Invalid number of layers: {num_hidden_layers}")

        self.intensities = self.exp_config["experiment"].get(
            "intensity", self.repe_eng_config["coeffs"]
        )
        self.rep_control_pipeline = get_pipeline(
            "rep-control-vllm" if self.is_vllm else "rep-control",
            model=self.model,
            tokenizer=self.tokenizer,
            layers=self.hidden_layers[
                len(self.hidden_layers) // 3 : 2 * len(self.hidden_layers) // 3
            ],  # self.repe_eng_config['control_layer_id'], #TODO  set within config
            block_name=self.repe_eng_config["block_name"],
            control_method=self.repe_eng_config["control_method"],
        )

        self.reaction_prompt_wrapper = GameReactPromptWrapper(
            self.prompt_format, response_format=self.game_config["decision_class"]
        )

        self.cur_emotion = None
        self.cur_coeff = None

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = f"{self.exp_config['experiment']['output']['base_dir']}/{self.exp_config['experiment']['name']}_{self.game_config['game_name']}_{self.repe_eng_config['model_name_or_path'].split('/')[-1]}_{timestamp}"
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
        with open(self.output_dir + "/exp_config.yaml", "w") as f:
            yaml.dump(self.exp_config, f)

        client_choice = self.exp_config["experiment"].get("oai_client", "openai")
        if client_choice == "azure":
            self.llm_client = AzureOpenAI(**AZURE_OPENAI_CONFIG)
        elif client_choice == "openai":
            from api_configs import OAI_CONFIG

            self.llm_client = OpenAI(**OAI_CONFIG)
        else:
            raise ValueError(
                f"Invalid LLM client: {client_choice}. Supported: 'azure', 'openai'"
            )

    def build_dataloader(self):
        self.logger.info(
            f"Creating dataset with sample_num={self.sample_num if self.sample_num is not None else 'all'}"
        )
        emo_dataset = GameScenarioDataset(
            self.game_config,
            partial(
                self.reaction_prompt_wrapper.__call__,
                user_messages=self.exp_config["experiment"]["system_message_template"],
                enable_thinking=self.enable_thinking,
            ),
            sample_num=self.sample_num,
        )
        assert (
            len(emo_dataset) > 0
        ), f"DEBUG: Dataset is empty, you are reading {self.game_config['data_path']}"
        data_loader = DataLoader(
            emo_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=collate_game_scenarios,
        )
        return data_loader

    def run_experiment(self):
        self.logger.info("Starting experiment")
        results = []

        for emotion in self.exp_config["experiment"]["emotions"]:
            self.logger.info(f"Processing emotion: {emotion}")
            rep_reader = self.emotion_rep_readers[emotion]
            self.cur_emotion = emotion

            data_loader = self.build_dataloader()

            for coeff in self.intensities:
                self.logger.info(f"Processing coefficient: {coeff}")
                self.cur_coeff = coeff
                results.extend(self._infer_with_activation(rep_reader, data_loader))

        self.cur_emotion = "Neutral"
        self.cur_coeff = 0
        self.logger.info(f"Processing Null Emotion")
        results.extend(self._infer_with_activation(rep_reader, data_loader))

        return self._save_results(results)

    def _infer_with_activation(self, rep_reader, data_loader):
        self.logger.info(f"Setting up activations for coefficient {self.cur_coeff}")

        # For vLLM models, use cuda:0 as the default device since vLLM doesn't expose direct device attribute
        if self.is_vllm:
            device = torch.device("cpu")

            # # Get tensor parallel size for vLLM model if available
            # tp_size = 1
            # if hasattr(self.model.llm_engine, 'parallel_config') and hasattr(self.model.llm_engine.parallel_config, 'tensor_parallel_size'):
            #     tp_size = self.model.llm_engine.parallel_config.tensor_parallel_size

            # self.logger.info(f"Using default cuda:0 device for vLLM model with tensor_parallel_size={tp_size}")
        else:
            device = self.model.device

        activations = {
            layer: torch.tensor(
                self.cur_coeff
                * rep_reader.directions[layer]
                * rep_reader.direction_signs[layer]
            )
            .to(device)
            .half()
            for layer in self.hidden_layers
        }

        return self._forward_dataloader(data_loader, activations)

    def _forward_dataloader(self, data_loader, activations):
        batch_results = []
        pipeline_queue = Queue(maxsize=2)  # Control memory usage
        processed_futures = []  # Keep track of futures

        def pipeline_worker():
            for i, batch in enumerate(data_loader):
                # Repeat each list in the batch dictionary
                repeat_batch = {
                    key: [item for item in value for _ in range(self.repeat)]
                    for key, value in batch.items()
                }

                start_time = time.time()
                control_outputs = self.rep_control_pipeline(
                    repeat_batch["prompt"],
                    activations=activations,
                    batch_size=self.batch_size
                    * self.repeat,  # Ensure batch_size is adjusted for repeats
                    temperature=self.generation_config["temperature"],
                    max_new_tokens=self.generation_config["max_new_tokens"],
                    do_sample=self.generation_config["do_sample"],
                    top_p=self.generation_config["top_p"],
                )
                end_time = time.time()
                pipeline_queue.put((i, repeat_batch, control_outputs))

            pipeline_queue.put(None)  # Sentinel value

        # Start pipeline worker thread
        worker = Thread(target=pipeline_worker, name="PipelineWorker")
        worker.start()

        # Process results while next batch is being generated
        with ThreadPoolExecutor(
            max_workers=self.batch_size // 2, thread_name_prefix="PostProc"
        ) as post_proc_executor:
            active_post_proc_tasks = 0
            while True:
                item = pipeline_queue.get()

                if item is None:
                    break  # Worker finished

                batch_idx, batch, control_outputs = item

                # Submit post-processing to executor
                active_post_proc_tasks += 1
                future = post_proc_executor.submit(
                    self._post_process_batch,
                    batch,
                    control_outputs,
                    batch_idx,  # Pass batch_idx for logging
                )
                processed_futures.append((batch_idx, future))

            # Wait for all submitted tasks to complete and collect results
            results_dict = {}
            for batch_idx, future in processed_futures:
                start_wait = time.time()
                try:
                    result = future.result()  # Blocks here
                    end_wait = time.time()
                    active_post_proc_tasks -= 1
                    results_dict[batch_idx] = result
                except Exception as e:
                    results_dict[batch_idx] = []  # Store empty list on error

            # Combine results in order
            for i in sorted(results_dict.keys()):
                batch_results.extend(results_dict[i])

        worker.join()
        return batch_results

    def _post_process_batch(self, batch, control_outputs, batch_idx):  # Add batch_idx
        start_time = time.time()
        log_prefix = f"{time.time():.2f} [{current_thread().name}]"

        results = []
        output_data = []
        batch_size = int(
            len(batch["prompt"]) / self.repeat
        )  # we dont need self.batch_size here since the last batch might be smaller
        assert (
            len(batch["prompt"])
            == len(batch["options"])
            == len(batch["scenario"])
            == len(batch["description"])
            == len(control_outputs)
            == batch_size * self.repeat
        )

        for i, p in enumerate(control_outputs):
            if self.is_vllm:
                generated_text = p.outputs[0].text.replace(batch["prompt"][i], "")
            else:
                generated_text = p[0]["generated_text"].replace(batch["prompt"][i], "")
            options = batch["options"][i]
            output_data.append((generated_text, options))
            repeat_idx = i // batch_size

        with ThreadPoolExecutor(
            max_workers=min(len(output_data), 64),
            thread_name_prefix=f"Extractor_B{batch_idx}",
        ) as executor:
            extracted_reses = list(
                executor.map(self._post_process_single_output, output_data)
            )

        self.logger.info(
            f"{log_prefix} PostProc: Extracted {len(extracted_reses)} results"
        )

        # Combine results
        for i, (extracted_res, (generated_text, _)) in enumerate(
            zip(extracted_reses, output_data)
        ):
            current_batch_idx = i  # Renamed from batch_idx to avoid confusion with the overall batch index
            original_index_in_batch = i  # Use 'i' directly as it maps 0 to N-1 within this specific post-processing call
            original_scenario_idx = original_index_in_batch % batch_size
            repeat_num = original_index_in_batch // batch_size

            self.logger.debug(
                f"{log_prefix} PostProc B{batch_idx}: Extracted result {i}: {extracted_res}"
            )

            results.append(
                {
                    "emotion": self.cur_emotion,
                    "intensity": self.cur_coeff,
                    "scenario": batch["scenario"][original_scenario_idx],
                    "description": batch["description"][original_scenario_idx],
                    "input": batch["prompt"][
                        original_index_in_batch
                    ],  # Input prompt is already repeated
                    "output": generated_text,
                    "rationale": extracted_res.rationale,
                    "decision": extracted_res.decision,
                    "category": extracted_res.option_id,
                    "repeat_num": repeat_num,
                    "extraction_method": extracted_res.extraction_method,
                }
            )

            self.logger.debug(
                f"{log_prefix} PostProc B{batch_idx}: Added result {i} with scenario {batch['scenario'][original_scenario_idx]} and repeat {repeat_num}"
            )

        end_time = time.time()
        self.logger.info(
            f"{log_prefix} PostProc: Finished for batch {batch_idx} ({end_time - start_time:.2f}s). Returning {len(results)} results."
        )
        return results

    def _post_process_single_output(self, output_data):
        generated_text, options = output_data
        log_prefix = f"{time.time():.2f} [{current_thread().name}]"
        self.logger.debug(f"{log_prefix} Extractor: {generated_text} and {options}")
        try:
            # Primary regex patterns - handle both single and double quotes
            pattern_rational = (
                r'[\'"]rational[\'"]\s*:\s*[\'"](((?:[^\'"]\\.|[^\'"])*))[\'"]'
            )
            pattern_decision = (
                r'[\'"]decision[\'"]\s*:\s*[\'"](((?:[^\'"]\\.|[^\'"])*))[\'"]'
            )

            # Try to extract using primary patterns
            rationale_match = re.search(
                pattern_rational, generated_text, re.MULTILINE | re.DOTALL
            )
            decision_match = re.search(
                pattern_decision, generated_text, re.MULTILINE | re.DOTALL
            )

            # If primary patterns fail, try more flexible patterns
            if not rationale_match or not decision_match:
                alt_pattern_rational = (
                    r'[\'"]?rational[\'"]?\s*:\s*[\'"](((?:[^\'"]\\.|[^\'"])*))[\'"]'
                )
                alt_pattern_decision = (
                    r'[\'"]?decision[\'"]?\s*:\s*[\'"](((?:[^\'"]\\.|[^\'"])*))[\'"]'
                )

                rationale_match = re.search(
                    alt_pattern_rational, generated_text, re.MULTILINE | re.DOTALL
                )
                decision_match = re.search(
                    alt_pattern_decision, generated_text, re.MULTILINE | re.DOTALL
                )

            self.logger.debug(
                f"{log_prefix} Extractor: Regex Rationale: {rationale_match.group(1) if rationale_match else 'Not Found'}"
            )
            self.logger.debug(
                f"{log_prefix} Extractor: Regex Decision: {decision_match.group(1) if decision_match else 'Not Found'}"
            )

            if not rationale_match or not decision_match:
                raise ValueError("Failed to extract rationale or decision via regex")

            # Handle both escaped single and double quotes
            rationale = rationale_match.group(1).replace('\\"', '"').replace("\\'", "'")
            decision = decision_match.group(1).replace('\\"', '"').replace("\\'", "'")

            # Find matching option using case-insensitive comparison
            matching_options = [
                j + 1
                for j, o in enumerate(options)
                if decision.lower().strip() in o.lower().strip()
            ]
            if not matching_options:
                raise ValueError(
                    f"Could not find matching option for decision: {decision}"
                )

            option_id = matching_options[0]
            extracted_res = ExtractedResultWithMethod(
                option_id=option_id,
                rationale=rationale,
                decision=decision,
                extraction_method="regex",
            )

            self.logger.info(
                f"{log_prefix} Extractor: Found option_id {option_id} directly from text"
            )

        except Exception as e:
            self.logger.debug(
                f"{log_prefix} Extractor: Regex extraction failed: {str(e)}. Using LLM to determine option_id."
            )
            extracted_res = self._get_option_id_from_llm(generated_text, options)
            self.logger.info(
                f"{log_prefix} Extractor: LLM extracted res successfully: {extracted_res}"
            )

        return extracted_res

    def _get_option_id_from_llm(self, generated_text, options):
        prompt = f"Given the options {[ 'option ' + str(oid+1) + ' : ' + opt for oid, opt in enumerate(options)]}. "
        prompt += f"Extract the decision, the rationale, and the option id for the following generated text: {generated_text}. "
        prompt += f"If the generated text is not in the options, return -1. Option ranges from 1 to {len(options)}."
        prompt += f"Response in the following format. {{'decision': <decision>, 'rationale': <rationale>, 'option_id': <option_id>}}"

        for _ in range(3):
            try:
                res = oai_response(
                    prompt,
                    model="gpt-4o-mini",
                    client=self.llm_client,
                    response_format=ExtractedResult,
                )
                return ExtractedResultWithMethod(
                    **json.loads(res), extraction_method="llm"
                )
            except Exception as e:
                self.logger.info(f"Error getting post-processed response from LLM: {e}")
                prompt = f"In previous run, there is an error: {e}. Please try again.\n\n{prompt}"

        return ExtractedResult(option_id=-1, rationale=generated_text, decision="")

    def _save_results(self, results):
        self.logger.info("Saving experiment results")

        Path(self.output_dir).mkdir(parents=True, exist_ok=True)

        json_filename = f"{self.output_dir}/exp_results.json"
        with open(json_filename, "w") as f:
            json.dump(results, f, indent=2)
        self.logger.info(f"Results saved to {json_filename}")

        df = pd.DataFrame(results)
        csv_filename = f"{self.output_dir}/exp_results.csv"
        df.to_csv(csv_filename, index=False)
        self.logger.info(f"Results saved to {csv_filename}")

        stats_results = self._run_statistical_analysis(csv_filename)
        stats_filename = f"{self.output_dir}/stats_analysis.json"
        with open(stats_filename, "w") as f:
            json.dump(stats_results, f, indent=2)
        self.logger.info(f"Stats results saved to {stats_filename}")

        return df, stats_results

    def _run_statistical_analysis(self, csv_file_path):
        self.logger.info("Running statistical analysis")

        results = analyze_emotion_and_intensity_effects(csv_file_path)

        # Save analysis results
        return results

    def run_sanity_check(self):
        """Run a sanity check with 10 examples from the dataset.
        This helps validate the experiment setup and model behavior before running the full experiment.

        Returns:
            tuple: (DataFrame with results, statistical analysis results)
        """
        self.logger.info("Starting sanity check with 10 examples")
        original_sample_num = self.sample_num
        original_output_dir = self.output_dir

        # Modify settings for sanity check
        self.sample_num = 10
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = f"{self.exp_config['experiment']['output']['base_dir']}/sanity_check_{self.exp_config['experiment']['name']}_{self.repe_eng_config['model_name_or_path'].split('/')[-1]}_{timestamp}"
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)

        try:
            # Run experiment with reduced sample
            results = []
            test_emotion = self.repe_eng_config["emotions"][
                0
            ]  # Test with first emotion
            self.logger.info(f"Testing with emotion: {test_emotion}")

            rep_reader = self.emotion_rep_readers[test_emotion]
            self.cur_emotion = test_emotion

            data_loader = self.build_dataloader()

            # Test with one intensity value
            test_intensity = self.intensities[0]
            self.logger.info(f"Testing with intensity: {test_intensity}")
            self.cur_coeff = test_intensity
            results.extend(self._infer_with_activation(rep_reader, data_loader))

            # Add neutral condition
            self.cur_emotion = "Neutral"
            self.cur_coeff = 0
            self.logger.info(f"Testing Neutral condition")
            results.extend(self._infer_with_activation(rep_reader, data_loader))

            # Save and analyze results
            df, stats = self._save_results(results)

            self.logger.info("Sanity check completed successfully")
            self.logger.info(f"Results saved in {self.output_dir}")

            return df, stats

        finally:
            # Restore original settings
            self.sample_num = original_sample_num
            self.output_dir = original_output_dir
