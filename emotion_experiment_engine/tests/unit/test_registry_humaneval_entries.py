# Responsible: emotion_experiment_engine/benchmark_component_registry.py
# Purpose: Verify registry wires HumanEval wrapper/dataset correctly.

from emotion_experiment_engine.data_models import BenchmarkConfig
from pathlib import Path
import pytest
from emotion_experiment_engine.benchmark_component_registry import create_benchmark_components


class _PlainPromptFormat:
    def build(self, system_prompt, user_messages, assistant_messages, images=None, enable_thinking=False):
        return "\n".join((system_prompt or "", *(user_messages or [])))


def test_registry_wires_humaneval():
    humaneval_data = Path('/home/jjl7137/human-eval/data/HumanEval.jsonl.gz')
    if not humaneval_data.exists():
        pytest.skip('HumanEval data file missing')
    cfg = BenchmarkConfig(
        name='humaneval',
        task_type='main',
        data_path=humaneval_data,
        base_data_dir=None,
        sample_limit=1,
        augmentation_config=None,
        enable_auto_truncation=False,
        truncation_strategy='right',
        preserve_ratio=1.0,
        llm_eval_config=None,
    )
    pf = _PlainPromptFormat()
    pw, aw, ds = create_benchmark_components('humaneval', '*', cfg, pf, emotion=None)
    assert callable(pw) and callable(aw)
    assert hasattr(ds, '__len__')
