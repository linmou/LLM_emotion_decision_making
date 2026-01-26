# Documentation Update Record v2.2.4

Date: 2025-10-13

Scope:
- Registry: Organized HumanEval benchmark tasks as {default, plus, *} in `emotion_experiment_engine/benchmark_component_registry.py`. Added MBPP entries with matching prompt wrapper.
- Tasks Plan: Reorganized `tasks/evalplus_migration_plan.md` to adopt `{benchmark: humaneval|mbpp, tasks: {default, plus, *}}` structure; clarified dataset refactors and test locations.
- Tests: Added HumanEval/MBPP unit + integration suites under `emotion_experiment_engine/tests/...`; legacy benchmark test removed.
- Configs: Added smoke configs (`config/humaneval_smoke.yaml`, `config/humaneval_plus_smoke.yaml`, `config/mbpp_smoke.yaml`, `config/mbpp_plus_smoke.yaml`) with relative dataset paths for dry-run verification.

Rationale:
- Align with dataset factory design (name-only mapping) while supporting multiple task modes per benchmark via a single dataset class branching on `config.task_type` (KISS).
- Prepare for EvalPlus migration with strict oracle evaluation in `plus` modes.

Follow-ups:
- Ensure optional dependency `tree_sitter_python` is installed for MBPP EvalPlus evaluation; current tests skip when absent.
- Run full regression suite once optional deps are available.
