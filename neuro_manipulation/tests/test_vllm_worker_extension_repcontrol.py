# Tests for neuro_manipulation/repe/vllm_worker_extension.py: extension exposes repcontrol RPC methods.

import unittest


class TestVllmWorkerExtensionRepControl(unittest.TestCase):
    def test_extension_has_expected_methods(self):
        from neuro_manipulation.repe.vllm_worker_extension import (
            NMRepControlWorkerExtension,
        )

        self.assertTrue(hasattr(NMRepControlWorkerExtension, "_nm_repcontrol_register_hook"))
        self.assertTrue(hasattr(NMRepControlWorkerExtension, "_nm_repcontrol_set_state"))
        self.assertTrue(hasattr(NMRepControlWorkerExtension, "_nm_repcontrol_reset_state"))

