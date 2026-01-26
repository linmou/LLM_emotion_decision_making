# Tests for neuro_manipulation/utils.py: allow forcing vLLM attention backend via config.

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
            "attention_backend": "TRITON_ATTN",
        }


class TestVllmAttentionBackendEnv(unittest.TestCase):
    def test_load_model_only_applies_attention_backend_env(self):
        # Responsible file: neuro_manipulation/utils.py
        # Purpose: when flash-attn binary is incompatible with torch, we must be able
        # to force TRITON/FLASHINFER via VLLM_ATTENTION_BACKEND.
        import neuro_manipulation.utils as nm_utils

        original = os.environ.get("VLLM_ATTENTION_BACKEND")
        try:
            os.environ.pop("VLLM_ATTENTION_BACKEND", None)

            captured = {}

            def _fake_llm(**kwargs):
                captured.update(kwargs)
                return object()

            with patch.object(nm_utils.utils_module, "LLM", _fake_llm):
                _ = nm_utils.load_model_only(  # type: ignore[attr-defined]
                    model_name_or_path="/mock/model",
                    from_vllm=True,
                    loading_config=_DummyLoadingConfig(),
                )

            self.assertEqual(os.environ.get("VLLM_ATTENTION_BACKEND"), "TRITON_ATTN")
            self.assertNotIn("attention_backend", captured)
        finally:
            if original is None:
                os.environ.pop("VLLM_ATTENTION_BACKEND", None)
            else:
                os.environ["VLLM_ATTENTION_BACKEND"] = original

