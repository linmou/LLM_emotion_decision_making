Feature: GPT-evaluated TrustLLM robustness scoring
  Background:
    Given the TrustLLM robustness dataset class
    And GPT-4o-mini judges replacing Longformer for adversarial instruction and OOD detection

  Scenario: Scores match TrustLLM semantics via GPT judgments
    When I evaluate adversarial instructions that either comply or refuse in English or Chinese
    Then the GPT-based scorer mirrors TrustLLM's follow-versus-refuse outcomes
    When I evaluate AdvGLUE responses in English and Chinese across QQP, MNLI, and SST-2
    Then the keyword heuristics still produce the same label buckets TrustLLM expects
    When I evaluate OOD detection and generalization samples with refusal and exact-match answers
    Then the GPT labels plus deterministic comparisons yield the same rates and per-source metrics TrustLLM reports
