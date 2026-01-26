# Documentation Update Record v2.3.0
Updated: 2025-11-14

Scope: diplomacy_pd benchmark, dataset + prompt wrapper integration

Changes
- Added a novel benchmark family `diplomacy_pd` with PD-style gradient choices (1–5 options).
- New dataset class: `emotion_experiment_engine/datasets/diplomacy_gradient.py`.
- New prompt wrapper: `emotion_experiment_engine/diplomacy_prompts.py` (`DiplomacyOptionsPromptWrapper`),
  subclass of `neuro_manipulation.prompt_wrapper.PromptWrapper` to comply with BenchmarkSpec composition.
- Registry mapping added in `emotion_experiment_engine/benchmark_component_registry.py`:
  `("diplomacy_pd", "*")` → `DiplomacyGradientDataset`, `SimpleOptionsPromptWrapper`, `IdentityAnswerWrapper`.
- Dataset file seeded with 18 items: `data/diplomacy/diplomacy_pd_v1.jsonl`.
- Added runner configs: `config/diplomacy_pd_series_runner.yaml` (series) and `config/diplomacy_pd_series.yaml` (single-runner).
- Updated `emotion_experiment_engine/README.md` with a brief note under Data Format.
- Updated architecture overview doc with a new section describing the benchmark and wrapper contract.
- Added authoring guide: `emotion_experiment_engine/claude_doc/adding_a_new_benchmark.md` describing
  BenchmarkSpec components, PromptWrapper subclass requirements, and the step-by-step process.
- Updated `emotion_experiment_engine/README.md` to replace legacy adapter instructions with the
  registry-based benchmark addition flow and removed references to non-existent adapter files.

Rationale
- Align prompt wrapper types: subclasses of PromptWrapper simplify integration and keep consistent
  expectations across wrappers.
- Provide a compact dataset and dry-run ready config to enable immediate benchmarking without
  adding press/emotion fields.

Compatibility Notes
- `SimpleOptionsPromptWrapper` accepts the registry adapter call signature and maps it to the
  base `PromptWrapper` flow internally.
- No changes to existing game theory datasets or wrappers.
