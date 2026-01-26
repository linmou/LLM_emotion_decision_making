# Tests for neuro_manipulation/utils.py: ensure vLLM load failures are not silently swallowed.

import unittest
from unittest.mock import MagicMock, patch


class TestLoadModelOnlyVllmFailure(unittest.TestCase):
    def test_from_vllm_true_raises_on_vllm_failure(self):
        from neuro_manipulation.utils import load_model_only
        import neuro_manipulation.utils as nm_utils

        fake_loading_config = MagicMock()
        fake_loading_config.to_vllm_kwargs.return_value = {
            "model": "/does/not/matter",
            "tensor_parallel_size": 1,
        }

        with patch.object(nm_utils.utils_module, "LLM", side_effect=Exception("boom")):
            with self.assertRaises(RuntimeError) as ctx:
                load_model_only(
                    model_name_or_path="/does/not/matter",
                    from_vllm=True,
                    loading_config=fake_loading_config,
                )

        self.assertIn("vLLM loading failed: boom", str(ctx.exception))
