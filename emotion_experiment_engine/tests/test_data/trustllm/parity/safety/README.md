# TrustLLM Safety Parity Fixtures

Synthetic samples sized for ≥20 items per branch to stabilize RtA measurements.
- `jailbreak_parity.json`: 40 prompts mixing short refusals and explicit compliance scripts. `expected_eval_res` mirrors TrustLLM's longformer output (`LABEL_0`=refuse).
- `misuse_parity.json`: 40 misuse scenarios with balanced refusal/compliance completions using the same label convention.
- `exaggerated_safety_parity.json`: 40 safety guidance prompts split between reasonable help (`LABEL_1`) and overrefusals (`LABEL_0`).

Each record exposes `prompt`, `res`, `label`, and `expected_eval_res` so tests can prime both the TrustLLM evaluator stub and our GPT fallbacks deterministically.
