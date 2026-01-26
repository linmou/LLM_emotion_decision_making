Feature: Deterministic TrustLLM privacy scoring
  Background:
    Given the TrustLLM privacy dataset class
    And the upstream TrustLLM privacy metrics definitions for ConfAIDe, awareness query, and leakage

  Scenario: Scores match TrustLLM heuristics without GPT or Longformer dependencies
    When I evaluate ConfAIDe responses with verbal and numeric self-ratings
    Then the deterministic scorer extracts the same magnitudes used for Pearson correlation
    When I evaluate awareness query responses that refuse or disclose private data
    Then the deterministic scorer produces the same refusal ratios TrustLLM reports
    When I evaluate leakage responses that contain or omit any email address domain
    Then the deterministic scorer flags disclosures identically to TrustLLM
