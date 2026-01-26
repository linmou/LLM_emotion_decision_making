# TrustLLM Robustness Parity Fixtures

Synthetic fixtures covering all robustness subtasks.
- `advinstruction_parity.json`: 40 prompts with expected follow/refuse labels.
- `advglue_parity.json`: Paired original/modified items across QQP, SST2, QNLI, MNLI tasks.
- `ood_detection_parity.json`: 40 prompts annotated with expected refusal/compliance.
- `ood_generalization_parity.json`: 60 prompts per source with correct/incorrect answers for F1 aggregation.
