# Responsible: emotion_experiment_engine/datasets/humaneval.py
# Purpose: Verify HumanEval dataset loading and parity (ids/prompts/entry_point).

import gzip
import json
from pathlib import Path
import pytest

from emotion_experiment_engine.data_models import BenchmarkConfig
from emotion_experiment_engine.dataset_factory import create_dataset_from_config

HUMANEVAL_DATA = Path('/home/jjl7137/human-eval/data/HumanEval.jsonl.gz')


@pytest.mark.skipif(not HUMANEVAL_DATA.exists(), reason='HumanEval data file missing')
def test_humaneval_dataset_loads_and_parity_fields():
    rows = []
    with gzip.open(HUMANEVAL_DATA, 'rt') as fp:
        for line in fp:
            if not line.strip():
                continue
            rows.append(json.loads(line))
            if len(rows) >= 5:
                break

    cfg = BenchmarkConfig(
        name='humaneval',
        task_type='main',
        data_path=HUMANEVAL_DATA,
        base_data_dir=None,
        sample_limit=5,
        augmentation_config=None,
        enable_auto_truncation=False,
        truncation_strategy='right',
        preserve_ratio=1.0,
        llm_eval_config=None,
    )
    ds = create_dataset_from_config(cfg, prompt_wrapper=lambda **kw: kw.get('question',''))
    assert len(ds) == 5
    for i, row in enumerate(rows):
        rec = ds[i]
        item = rec['item']
        assert item.id == row['task_id']
        assert rec['prompt'] == row['prompt']
        assert item.metadata and item.metadata.get('entry_point') == row['entry_point']
        assert isinstance(rec['ground_truth'], dict) and rec['ground_truth'].get('task_id') == item.id
