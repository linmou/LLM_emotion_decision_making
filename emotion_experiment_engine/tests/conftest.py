"""Test configuration for emotion_experiment_engine tests."""

import os
import importlib.machinery
import sys
import types
from typing import Optional


def _install_torch_stub() -> None:
    if os.environ.get("EMOTION_EXPERIMENT_ENGINE_USE_REAL_TORCH") == "1":
        try:
            import torch  # type: ignore  # noqa: F401
            return
        except Exception:
            pass

    if "torch" in sys.modules:
        return

    torch_module = types.ModuleType("torch")
    torch_module.__spec__ = importlib.machinery.ModuleSpec("torch", loader=None)
    utils_module = types.ModuleType("torch.utils")
    utils_module.__spec__ = importlib.machinery.ModuleSpec("torch.utils", loader=None)
    data_module = types.ModuleType("torch.utils.data")
    data_module.__spec__ = importlib.machinery.ModuleSpec("torch.utils.data", loader=None)

    class _TorchDataset:  # type: ignore[override]
        def __iter__(self):
            return iter(())

    class _TorchDataLoader:
        def __init__(self, dataset, batch_size: Optional[int] = 1, shuffle: bool = False, collate_fn=None):
            self.dataset = dataset
            self.batch_size = batch_size or 1
            self.shuffle = shuffle
            self.collate_fn = collate_fn

        def __iter__(self):  # pragma: no cover - simple helper
            data_iter = None
            if hasattr(self.dataset, "__iter__"):
                data_iter = iter(self.dataset)
            elif hasattr(self.dataset, "__len__") and hasattr(self.dataset, "__getitem__"):
                data_iter = (self.dataset[i] for i in range(len(self.dataset)))
            else:
                data_iter = iter(())

            batch = []
            for item in data_iter:
                batch.append(item)
                if len(batch) == self.batch_size:
                    yield self._collate(batch)
                    batch = []
            if batch:
                yield self._collate(batch)

        def _collate(self, items):
            if self.collate_fn is not None:
                return self.collate_fn(items)
            if self.batch_size == 1:
                return items[0]
            return list(items)

    data_module.Dataset = _TorchDataset
    data_module.DataLoader = _TorchDataLoader
    utils_module.data = data_module
    torch_module.utils = utils_module

    sys.modules["torch"] = torch_module
    sys.modules["torch.utils"] = utils_module
    sys.modules["torch.utils.data"] = data_module


_install_torch_stub()


def _install_vllm_stub() -> None:
    if "vllm" in sys.modules:
        return

    vllm_module = types.ModuleType("vllm")

    class _StubLLM:  # minimal placeholder for tests
        def __init__(self, *args, **kwargs):
            self.args = args
            self.kwargs = kwargs

    vllm_module.LLM = _StubLLM
    class _StubSamplingParams:
        def __init__(self, *args, **kwargs):
            self.args = args
            self.kwargs = kwargs

    vllm_module.SamplingParams = _StubSamplingParams
    sys.modules["vllm"] = vllm_module


_install_vllm_stub()


def _install_openai_stub() -> None:
    if "openai" in sys.modules:
        return

    openai_module = types.ModuleType("openai")
    openai_module.__spec__ = importlib.machinery.ModuleSpec("openai", loader=None)

    class _StubChatCompletions:
        def create(self, *args, **kwargs):
            return types.SimpleNamespace(choices=[types.SimpleNamespace(message=types.SimpleNamespace(content='{"label": "refuse"}'))])

    class _StubChat:
        def __init__(self):
            self.completions = _StubChatCompletions()

    class _StubOpenAI:
        def __init__(self, *args, **kwargs):
            self.chat = _StubChat()

    openai_module.OpenAI = _StubOpenAI
    sys.modules["openai"] = openai_module


_install_openai_stub()
