# LLM Evaluation System Update

**Last Updated**: September 25, 2025  
**Key Commits**:  
`9b80f47f64a760b8e015b976051b0a9f32a87356` — build mtbench 101 dataset; update LLM evaluation (introduces the current `llm_eval_config` layering and thread-pooled `evaluate_batch`)  
`b4de93c58d3a6db5c0b2798452bed823dc57` — fix the issue about lacking history (plumbs prompts through batch evaluation)  
`e731e9259f65453a7376a7bf40db53689a3663fd` — update emotion check prompt; enable eval error stored in final results (records per-item evaluation errors)

This revision replaces the August 29, 2025 note. The documentation now reflects how benchmark datasets wire LLM judging parameters and how batch scoring is executed inside the synchronous experiment loop.

## `llm_eval_config` Layering

`BaseBenchmarkDataset` owns a class-level `LLM_EVAL_CONFIG` default. During instantiation it copies that default and merges in any overrides supplied by the benchmark configuration:

```python
class BaseBenchmarkDataset(...):
    LLM_EVAL_CONFIG = {"model": "gpt-4o-mini", "temperature": 0.0}

    def __init__(...):
        if hasattr(self, "LLM_EVAL_CONFIG"):
            self.llm_eval_config = deepcopy(self.LLM_EVAL_CONFIG)
            if self.config.llm_eval_config:
                self.llm_eval_config.update(self.config.llm_eval_config)
        else:
            self.llm_eval_config = self.config.llm_eval_config
```

- Every dataset that exposes LLM judging should set its own `LLM_EVAL_CONFIG`. If the class does not define one, the config provided at runtime becomes the sole source of truth.  
- The merge step happens once per dataset construction, so per-run overrides inside YAML configs are cheap and deterministic.  
- Config values are treated as a shallow update. To change the evaluation model, specify only the keys you intend to override:

```yaml
# config/longbench_custom.yaml
benchmark:
  llm_eval_config:
    model: gpt-4o-mini
    temperature: 0.1
    max_tokens: 16
```

### LongBench Integration

`LongBenchDataset` keeps a local `LLM_EVAL_CONFIG` identical to the base default and forwards the merged configuration to `evaluation_utils.llm_evaluate_response` every time it routes to the GPT judge:

```python
result = evaluation_utils.llm_evaluate_response(
    system_prompt="You are an expert evaluator.",
    query=self.PROMPT_FORMAT.format(response=response, ground_truth=ground_truth),
    llm_eval_config=self.llm_eval_config,
)
score = float(result.get("answer", 0.0))
```

Other TrustLLM families build their own evaluation prompts (injecting the original user prompt and assistant reply) and call `evaluation_utils.llm_evaluate_response(...)` directly after merging dataset-specific overrides with `DEFAULT_LLM_EVAL_CONFIG`. That keeps the prompt format localized to each dataset while preserving the shared configuration defaults.

## Batch Evaluation Execution

`BaseBenchmarkDataset.evaluate_batch` is now a thin synchronous wrapper that drives concurrent calls to `evaluate_response` via a `ThreadPoolExecutor`:

```python
def evaluate_batch(self, responses, ground_truths, task_names, prompts):
    self._last_eval_errors = [None] * len(responses)
    max_workers = getattr(self, "eval_workers", 64)
    max_workers = max(1, min(int(max_workers), len(responses)))

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(self.evaluate_response, r, gt, task, prompt)
            for r, gt, task, prompt in zip(responses, ground_truths, task_names, prompts)
        ]

    scores = []
    for idx, future in enumerate(futures):
        try:
            scores.append(future.result())
        except Exception as exc:
            self._last_eval_errors[idx] = str(exc)
            scores.append(float("nan"))
    return scores
```

- The experiment sets `dataset.eval_workers = max_evaluation_workers`, so concurrency is configurable per run. Defaults to 64 workers but is capped by the batch size.  
- `_last_eval_errors` keeps per-item failure messages so experiments can surface them in result logs. No exception escapes the batch call.  
- Because `evaluate_batch` simply threads over `evaluate_response`, dataset authors can keep their evaluators synchronous—there is no longer an async bridge or fallback string matcher.

## Practical Notes

- **When to override**: Use YAML overrides when you need a different OpenAI endpoint, temperature, or response format. Dataset authors should avoid hardcoding API keys or transport details.  
- **Inspecting failures**: After calling `evaluate_batch`, read `dataset._last_eval_errors` to understand why certain scores are `nan`. This is especially helpful when GPT responses cannot be parsed as JSON.  
- **Extending datasets**: New benchmarks should extend `BaseBenchmarkDataset`, define an `LLM_EVAL_CONFIG`, and ensure every evaluator pulls from `self.llm_eval_config` so global overrides remain consistent.  
- **Concurrency hygiene**: Favor modest `max_evaluation_workers` when the evaluation endpoint is rate-limited. Threading is CPU-light but still subject to API quotas.

This document should help future updates stay aligned with the shared infrastructure. If another revision changes either the configuration merge strategy or the batch evaluation flow, append a new entry under **Key Commits** with the hash and summary.
