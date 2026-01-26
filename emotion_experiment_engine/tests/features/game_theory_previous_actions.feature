# Feature file describing expectations for game theory dry run behavior.
Feature: Game theory dry run output quality
  Background:
    Given a configured memory experiment series for the Trust Game trustee benchmark

  Scenario: Previous actions are exposed in constructed prompts
    When the dry run builds the first Trust Game trustee scenario
    Then the formatted scenario text should include previously observed trustor actions

  Scenario: Dry run fails fast when a benchmark configuration raises an error
    When the dry run encounters a benchmark configuration error
    Then the dry run command should exit with a failure instead of reporting success
