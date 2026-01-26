# Tests for sitecustomize.py: ensure we don't patch vLLM WorkerBase when using worker_extension_cls.

import importlib
import unittest


class TestSitecustomizeNoWorkerBasePatch(unittest.TestCase):
    def test_sitecustomize_does_not_patch_vllm_workerbase(self):
        # Responsible file: sitecustomize.py
        # Purpose: vLLM v1 rejects worker_extension_cls if worker already has same attrs.
        # sitecustomize must not inject _nm_repcontrol_* methods onto WorkerBase.
        import sitecustomize  # noqa: F401

        # Force import to ensure any patching logic has run.
        importlib.reload(sitecustomize)

        from vllm.v1.worker.worker_base import WorkerBase  # type: ignore

        self.assertFalse(hasattr(WorkerBase, "_nm_repcontrol_register_hook"))
        self.assertFalse(hasattr(WorkerBase, "_nm_repcontrol_set_state"))
        self.assertFalse(hasattr(WorkerBase, "_nm_repcontrol_reset_state"))

