# Tests for neuro_manipulation/repe/rep_control_vllm_hook.py: worker-side hook registration should be idempotent; TP slicing should not crash without a usable rank; packed [T,H] tensors should be supported.

import unittest

import torch
import torch.nn as nn


class _CountingLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.hook_registrations = 0

    def register_forward_hook(self, hook, prepend=False, with_kwargs=False):  # type: ignore[override]
        self.hook_registrations += 1
        return super().register_forward_hook(hook, prepend=prepend, with_kwargs=with_kwargs)


class _DummyModel(nn.Module):
    def __init__(self, layers: nn.ModuleList):
        super().__init__()
        self.layers = layers


class _DummyModelRunner:
    def __init__(self, model: nn.Module):
        self.model = model


class _DummyWorker:
    def __init__(self, model: nn.Module, rank: int = 0):
        self.rank = rank
        self.model_runner = _DummyModelRunner(model)


class _DummyModule:
    pass


class TestRepControlVllmHookWorkerSafety(unittest.TestCase):
    def test_worker_register_hook_is_idempotent(self):
        from neuro_manipulation.repe.rep_control_vllm_hook import (
            _register_hook_on_worker_rpc,
            hook_fn_rep_control,
        )

        layer = _CountingLayer()
        worker = _DummyWorker(_DummyModel(nn.ModuleList([layer])))

        self.assertTrue(_register_hook_on_worker_rpc(worker, 0, "decoder_block", hook_fn_rep_control))
        self.assertTrue(_register_hook_on_worker_rpc(worker, 0, "decoder_block", hook_fn_rep_control))
        self.assertEqual(layer.hook_registrations, 1)

    def test_tp_slicing_does_not_crash_without_distributed_rank(self):
        from neuro_manipulation.repe.rep_control_vllm_hook import hook_fn_rep_control

        module = _DummyModule()
        module._rep_control_state = {
            "controller": [0.1, 0.2, 0.3, 0.4],  # full_dim=4
            "mask": 1.0,
            "token_pos": None,
            "normalize": False,
            "operator_name": "linear_comb",
            "tp_size": 2,  # modified_dim==full_dim//tp_size triggers slicing path
            "kwargs": {},
        }

        output = torch.zeros((1, 1, 2), dtype=torch.float16)  # modified_dim=2
        got = hook_fn_rep_control(module, (), output)
        self.assertTrue(torch.allclose(got, output))

    def test_hook_supports_packed_token_tensor(self):
        from neuro_manipulation.repe.rep_control_vllm_hook import hook_fn_rep_control

        module = _DummyModule()
        module._rep_control_state = {
            "controller": [0.5, -1.0],
            "mask": 1.0,
            "token_pos": 1,
            "normalize": False,
            "operator_name": "linear_comb",
            "tp_size": 1,
            "kwargs": {},
        }

        output = torch.zeros((3, 2), dtype=torch.float16)  # [num_tokens, hidden]
        got = hook_fn_rep_control(module, (), output)
        expected = output.clone()
        expected[1] = torch.tensor([0.5, -1.0], dtype=torch.float16)
        self.assertTrue(torch.allclose(got, expected))

