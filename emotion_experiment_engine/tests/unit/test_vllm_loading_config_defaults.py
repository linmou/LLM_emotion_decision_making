# Tests for emotion_experiment_engine/data_models.py: VLLMLoadingConfig should provide safe vLLM worker extension defaults.

import unittest


class TestVllmLoadingConfigDefaults(unittest.TestCase):
    def test_to_vllm_kwargs_sets_worker_extension_cls_by_default(self):
        from emotion_experiment_engine.data_models import VLLMLoadingConfig

        cfg = VLLMLoadingConfig(
            model_path="/tmp/model",
            gpu_memory_utilization=0.9,
            tensor_parallel_size=1,
            max_model_len=1024,
            enforce_eager=True,
            quantization=None,
            trust_remote_code=True,
            dtype="bfloat16",
            seed=0,
            disable_custom_all_reduce=False,
            additional_vllm_kwargs={},
        )
        kwargs = cfg.to_vllm_kwargs()
        self.assertIn("worker_extension_cls", kwargs)
        self.assertEqual(
            kwargs["worker_extension_cls"],
            "neuro_manipulation.repe.vllm_worker_extension.NMRepControlWorkerExtension",
        )

