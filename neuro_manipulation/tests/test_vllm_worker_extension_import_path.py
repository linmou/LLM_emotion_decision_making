# Tests for neuro_manipulation/utils.py: ensure vLLM workers can import repo modules via PYTHONPATH.

import os
import unittest
from unittest.mock import patch


class _DummyLoadingConfig:
    def to_vllm_kwargs(self):
        return {
            "model": "/mock/model",
            "tensor_parallel_size": 1,
            "max_model_len": 8,
            "trust_remote_code": True,
            "enforce_eager": True,
            "gpu_memory_utilization": 0.5,
            "dtype": "float16",
            "seed": 0,
            "disable_custom_all_reduce": False,
            "worker_extension_cls": "neuro_manipulation.repe.vllm_worker_extension.NMRepControlWorkerExtension",
        }


class TestVllmWorkerExtensionImportPath(unittest.TestCase):
    def test_load_model_only_sets_pythonpath_for_workers(self):
        # Responsible file: neuro_manipulation/utils.py
        # Purpose: vLLM multiproc workers may not run from repo cwd; ensure PYTHONPATH
        # includes repo root so worker_extension_cls import works.
        import neuro_manipulation.utils as nm_utils

        original = os.environ.get("PYTHONPATH")
        try:
            os.environ["PYTHONPATH"] = ""

            with patch.object(nm_utils.utils_module, "LLM", lambda **kwargs: object()):
                _ = nm_utils.load_model_only(  # type: ignore[attr-defined]
                    model_name_or_path="/mock/model",
                    from_vllm=True,
                    loading_config=_DummyLoadingConfig(),
                )

            repo_root = os.path.dirname(os.path.dirname(nm_utils.utils_module.__file__))
            pythonpath = os.environ.get("PYTHONPATH", "")
            paths = [p for p in pythonpath.split(os.pathsep) if p]
            self.assertIn(repo_root, paths)
        finally:
            if original is None:
                os.environ.pop("PYTHONPATH", None)
            else:
                os.environ["PYTHONPATH"] = original
