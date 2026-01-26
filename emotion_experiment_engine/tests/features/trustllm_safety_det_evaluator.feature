Feature: GPT-evaluated TrustLLM safety scoring
  Background:
    Given the TrustLLM safety dataset class
    And GPT-4o-mini judges replacing Longformer for jailbreak, misuse, and exaggerated safety tasks

  Scenario: Scores match TrustLLM refusal semantics via GPT
    When I evaluate jailbreak and misuse replies in English and Chinese
    Then the GPT-based scorer marks refusals exactly as TrustLLM expects
    When I evaluate exaggerated safety prompts with safe guidance versus over-refusal
    Then the GPT-based scorer mirrors TrustLLM's reasonable versus over-refusal rates
