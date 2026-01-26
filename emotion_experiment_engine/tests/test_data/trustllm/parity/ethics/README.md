# TrustLLM Ethics Parity Fixtures

Synthetic fixtures mirroring TrustLLM ethics heuristics with >=20 samples per branch.
- `explicit_low_parity.json`: 50 low-risk moral choice items mixing letter/digit responses.
- `explicit_high_parity.json`: 40 high-risk items annotated with `expected_eval_res` (`LABEL_0` = refusal).
- `implicit_ethics_parity.json`: 70 items covering wrong/not wrong/not sure with 20 ambiguous cases requiring GPT resolution (`expected_eval_res`).
- `social_norm_parity.json`: 70 items spanning good/neutral/bad contexts.
