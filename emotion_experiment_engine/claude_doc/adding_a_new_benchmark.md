# Adding A New Benchmark To emotion_experiment_engine
Updated: 2025-11-14

This guide explains how to add a new benchmark to the emotion_experiment_engine using the registry-driven architecture. It covers the purpose of each component referenced by `emotion_experiment_engine.benchmark_component_registry.BenchmarkSpec` and provides a minimal checklist.

Key Concepts
- BenchmarkSpec: Declares the three components for a benchmark family
  - `dataset_class`: loads items and computes evaluation scores
  - `prompt_wrapper_class`: formats prompts for the model (must be a subclass of `neuro_manipulation.prompt_wrapper.PromptWrapper`)
  - `answer_wrapper_class`: transforms model output if needed (often `IdentityAnswerWrapper`)
- BENCHMARK_SPECS: A mapping from `(benchmark_name, task_type)` to `BenchmarkSpec`
- create_benchmark_components(): Assembles `(prompt_wrapper_fn, answer_wrapper_fn, dataset)` from the registry

BenchmarkSpec Components
1) dataset_class (required)
   - Subclass of `BaseBenchmarkDataset`
   - Responsibilities:
     - `_load_and_parse_data()` → returns `List[BenchmarkItem]`
       - Each `BenchmarkItem` contains `id`, `input_text` (prompt), `context` (optional), `ground_truth` (optional), `metadata`
     - `evaluate_response(response, ground_truth, task_name, prompt)` → returns a float score (can be an option id)
     - `get_task_metrics(task_name)` → returns list of metric names
   - Tip: Keep evaluation simple. If you need semantic judging, use the provided LLM-eval utilities in other datasets as a template.

2) prompt_wrapper_class (required)
   - Must be a subclass of `neuro_manipulation.prompt_wrapper.PromptWrapper`
   - Responsibilities:
     - Render system prompt (and optionally user messages) from dataset fields
     - Support the registry adapter call signature:
       ```python
       def __call__(self, *, context, question, user_messages, enable_thinking,
                    augmentation_config, answer, emotion, options) -> str
       ```
   - The wrapper should internally invoke `self.prompt_format.build(system_prompt, user_messages, enable_thinking=...)`.
   - Minimal wrappers can ignore `augmentation_config/answer/emotion` and just produce a clean, deterministic prompt.

3) answer_wrapper_class (required)
   - Subclass of `AnswerWrapper`
   - Responsibilities:
     - Transform the raw model output into an evaluation-friendly string (or pass-through)
     - For many multiple-choice-style benchmarks, `IdentityAnswerWrapper` is sufficient

Step-by-Step: Add A Benchmark
1) Create the dataset class
   - Path convention: `emotion_experiment_engine/datasets/<your_dataset>.py`
   - Implement a `BaseBenchmarkDataset` subclass with the three methods above

2) Create the prompt wrapper
   - Path convention: `emotion_experiment_engine/<your_prompt_wrapper>.py`
   - Subclass `PromptWrapper`; implement `system_prompt(event, options)` and an adapter `__call__` as needed

3) Register in BENCHMARK_SPECS
   - Edit `emotion_experiment_engine/benchmark_component_registry.py`
   - Add a mapping. Example:
     ```python
     ("my_benchmark", "*"): BenchmarkSpec(
         dataset_class=MyDataset,
         answer_wrapper_class=IdentityAnswerWrapper,
         prompt_wrapper_class=MyPromptWrapper,
     ),
     ```
   - `task_type` can be an exact string or "*" for all tasks under that name

4) Provide data and a config
   - Data: `data/<family>/<name>_<task_type>.jsonl` (recommended)
   - Series runner config: `config/<family>_series_runner.yaml`
   - Then run a dry run to validate wiring:
     ```bash
     bash -c "source /usr/local/anaconda3/etc/profile.d/conda.sh && \
       conda activate llm_fresh && \
       python -m emotion_experiment_engine.emotion_experiment_series_runner \
       --config config/<family>_series_runner.yaml --dry-run"
     ```

Worked Example: diplomacy_pd
- Dataset: `DiplomacyGradientDataset` (loads 1–5 natural-language options; evaluates selected option id)
- Prompt wrapper: `DiplomacyOptionsPromptWrapper` (subclass of `PromptWrapper`)
- Registry key: `("diplomacy_pd", "*")`
- Data: `data/diplomacy/diplomacy_pd_v1.jsonl` (18 items)
- Config: `config/diplomacy_pd_series_runner.yaml`

Design Rules (KISS)
- Keep wrappers minimal; rely on `PromptWrapper` base contract
- Avoid hidden heuristics in evaluation; prefer explicit extraction or judgment helpers
- Favor small, composable datasets with deterministic prompts and straightforward scoring
