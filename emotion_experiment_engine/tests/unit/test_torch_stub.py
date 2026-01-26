"""
Responsible file: emotion_experiment_engine/tests/conftest.py
Purpose: Ensure the torch fallback stub exposes Dataset and DataLoader interfaces for tests.
"""

import importlib
import sys


def test_torch_stub_exposes_dataloader(monkeypatch):
    # Capture and remove real torch modules so the stub path executes on reload.
    preserved = {}
    for name in ("torch", "torch.utils", "torch.utils.data"):
        if name in sys.modules:
            preserved[name] = sys.modules.pop(name)

    import emotion_experiment_engine.tests.conftest as conftest
    importlib.reload(conftest)

    try:
        from torch.utils.data import DataLoader, Dataset

        class _SimpleDataset(Dataset):
            def __iter__(self):
                yield from (1, 2, 3)

        loader = DataLoader(_SimpleDataset())
        assert list(loader) == [1, 2, 3]
    finally:
        # Remove stubbed modules and restore the real torch stack.
        for name in ("torch.utils.data", "torch.utils", "torch"):
            sys.modules.pop(name, None)
        for name, module in preserved.items():
            sys.modules[name] = module
        importlib.reload(conftest)
