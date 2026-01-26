# TrustLLM Fairness Parity Fixtures

These synthetic fixtures exercise TrustLLM fairness heuristics with >=20 samples per branch.
- `stereotype_recognition_parity.json`: 60 items balanced across stereotype / anti-stereotype / unrelated, mixing numeric and phrase responses.
- `stereotype_agreement_parity.json`: 80 items split evenly across yes/no with short tokens and phrases.
- `stereotype_query_parity.json`: 40 items, 20 expected refusals and 20 expected compliances. `expected_eval_res` indicates the label produced by TrustLLM's longformer judge (`LABEL_0`=refuse).
- `preference_parity.json`: 80 items covering plain/force responses with both direct answers and refusals. `expected_eval_res` follows the `LABEL_0` refusal convention.
- `disparagement_parity.json`: 30 items with rotating sex/race groups and A/B/C selections for chi-square parity.
