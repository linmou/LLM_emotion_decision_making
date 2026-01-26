# Tests for neuro_manipulation/repe/rep_control_vllm_hook.py: hook_fn_rep_control must accept list/ndarray controller/mask payloads.

import unittest

import torch


class _DummyModule:
    pass


class TestRepControlVllmHookControllerList(unittest.TestCase):
    def test_hook_accepts_list_controller(self):
        from neuro_manipulation.repe.rep_control_vllm_hook import hook_fn_rep_control

        module = _DummyModule()
        module._rep_control_state = {
            "controller": [0.5, -1.0],  # vLLM RPC may decode tensors/arrays as lists
            "mask": 1.0,
            "token_pos": None,
            "normalize": False,
            "operator_name": "linear_comb",
            "tp_size": 1,
            "kwargs": {},
        }

        output = torch.zeros((1, 1, 2), dtype=torch.float16)
        got = hook_fn_rep_control(module, (), output)
        expected = torch.tensor([[[0.5, -1.0]]], dtype=torch.float16)
        self.assertTrue(torch.allclose(got, expected))

    def test_hook_accepts_list_mask(self):
        from neuro_manipulation.repe.rep_control_vllm_hook import hook_fn_rep_control

        module = _DummyModule()
        module._rep_control_state = {
            "controller": [0.5, -1.0],
            "mask": [0.0, 1.0],  # vLLM RPC may decode tensors/arrays as lists
            "token_pos": None,
            "normalize": False,
            "operator_name": "linear_comb",
            "tp_size": 1,
            "kwargs": {},
        }

        output = torch.zeros((1, 1, 2), dtype=torch.float16)
        got = hook_fn_rep_control(module, (), output)
        expected = torch.tensor([[[0.0, -1.0]]], dtype=torch.float16)
        self.assertTrue(torch.allclose(got, expected))
