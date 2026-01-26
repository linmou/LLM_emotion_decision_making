Feature: GPT-evaluated TrustLLM fairness scoring
  Background:
    Given the TrustLLM fairness dataset class
    And GPT-4o-mini judges wired through evaluation_utils for query and preference tasks

  Scenario: Scores match TrustLLM heuristics via GPT labels
    When I evaluate stereotype recognition responses containing numbers and verbal labels
    Then the scorer still mirrors TrustLLM recognition metrics
    When I evaluate stereotype agreement responses with explicit yes or no language
    Then the scorer matches TrustLLM agreement counting
    When I evaluate stereotype query refusals and direct answers using GPT classification
    Then the refusal ratios align with TrustLLM expectations
    When I evaluate preference responses with GPT detecting refusals
    Then the plain and force breakdowns reproduce TrustLLM metrics
    When I aggregate split metrics for recognition, agreement, query, and preference
    Then the macro and micro statistics equal the TrustLLM calculations
