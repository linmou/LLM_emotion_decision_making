# Documentation Update Record: Version 2.2.5
<!-- Update Record: PubMedQA Benchmark Integration - 2025-10-19 -->

## Update Summary

Update Date: 2025-10-19  
From Version: 2.2.4  
To Version: 2.2.5  
Update Type: Minor Feature Addition

## Changes

- Added PubMedQA benchmark support via registry-based wiring.
  - New dataset: `emotion_experiment_engine/datasets/pubmed_qa.py`
  - Registry entry: (`"pubmed_qa"`, `"pqa_labeled"`) → `PubMedQADataset`, `IdentityAnswerWrapper`, `MemoryPromptWrapper`
  - Prompt wrapper support list updated to include `pubmed_qa`/`pqa_labeled`.
- PubMedQA prompt wrapper now accepts optional chain-of-thought instructions via `augmentation_config` (`{"method": "cot", ...}`) without changing evaluation format.
- Tests (TDD):
  - `tests/unit/test_registry_pubmedqa_entry.py` (source-level registry check)
  - `tests/unit/test_dataset_registry_pubmedqa.py` (factory class mapping)
  - `tests/unit/test_prompt_wrapper_pubmedqa.py` (wrapper routing)
  - `tests/unit/datasets/test_pubmedqa_eval.py` (evaluation logic, no network)
- README updated: PubMedQA usage snippet and header date.

## Notes

- Dataset loading uses `datasets.load_dataset("pubmed_qa", "pqa_labeled")` with default split `test` (overridable).
- Evaluation is exact normalized match over {yes,no,maybe}.
- Full regression run recommended in the `llm_fresh` conda environment (Python ≥3.10). Base interpreter (3.9) triggers union type syntax errors in existing tests and is not supported for the full suite.

## Impact

- Non-breaking additive change.
- Enables classification benchmark coverage relevant to biomedical QA scenarios.
