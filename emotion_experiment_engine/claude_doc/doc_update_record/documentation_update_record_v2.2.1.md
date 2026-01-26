# Documentation Update Record: Version 2.2.1
<!-- Update Record: From v2.2.0 to v2.2.1 - Test Layout Guard & Path Updates - 2025-09-07 -->

## Update Summary

**Update Date**: September 7, 2025  
**From Version**: 2.2.0  
**To Version**: 2.2.1  
**Update Type**: Documentation consistency + test policy (non-breaking)

## Primary Driver

Ensure consistency between test architecture and documentation by recording the new test layout guard, updated test paths after migration, and canonical regression commands for experiment changes.

## Files Updated

### 1. `testing_strategy_and_coverage_analysis.md`
**Version**: 1.1.0 → **1.1.1**

**Changes Made**:
- Added “Test Layout Guard (2025-09-07)” with CI and pre-commit checks
- Listed updated test locations for prompt wrapper and series runner
- Clarified root policy (“comprehensive + specialized”)
- Added regression command examples for `experiment.py`

**Rationale**: Keep testing docs aligned with the enforced test layout and recent migration work.

**Impact**: Low — documentation-only clarification and consistency fixes

## Related Changes (Non-Doc)
- CI: `.github/workflows/test_layout_check.yml` runs `python emotion_experiment_engine/tests/utils/check_test_layout.py` on push/PR
- Pre-commit: `.pre-commit-config.yaml` adds the same layout check locally
- Test relocations:
  - `tests/unit/test_memory_prompt_wrapper.py` (moved from root)
  - `tests/integration/test_emotion_experiment_series_runner.py` (moved from root)

## Regression Commands
- `python -m pytest emotion_experiment_engine/tests/regression -v --tb=short --maxfail=1`
- `python emotion_experiment_engine/tests/utils/test_runners.py regression`
