# emotion_experiment_engine/tests/features/resource_cleanup.feature
# Purpose: Describe high-level expectation for releasing GPU resources between experiments.

Feature: Release GPU resources between experiments
  In order to run multiple experiment configurations back to back
  As the experiment series runner
  I want each finished experiment to release its vLLM resources before the next one starts

  Scenario: Runner shuts down vLLM engine after completing an experiment
    Given an experiment series with one model and multiple benchmarks
    And the EmotionExperiment uses a vLLM backend
    When the runner completes the first experiment
    Then the vLLM engine for that experiment is shut down before the next experiment starts
