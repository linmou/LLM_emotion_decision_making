# Tests for neuro_manipulation/repe/rep_control_vllm_hook.py: ensure collective_rpc uses serializable method names (no functions).

import unittest


class _FakeParallelConfig:
    tensor_parallel_size = 1


class _FakeEngine:
    def __init__(self):
        self.parallel_config = _FakeParallelConfig()
        self.calls = []

    def collective_rpc(self, method, timeout=None, args=(), kwargs=None):
        self.calls.append((method, timeout, args, kwargs))
        return [True]


class _FakeLLM:
    def __init__(self):
        self.llm_engine = _FakeEngine()

    def generate(self, text_inputs, sampling_params):
        return []


class TestRepControlVllmHookRpcSerializationSafe(unittest.TestCase):
    def test_init_registers_hooks_using_method_names(self):
        from neuro_manipulation.repe.rep_control_vllm_hook import RepControlVLLMHook

        model = _FakeLLM()
        RepControlVLLMHook(
            model=model,
            tokenizer=None,
            layers=[0, 1],
            block_name="decoder_block",
            control_method="reading_vec",
            tensor_parallel_size=1,
        )

        methods = [m for (m, _t, _a, _k) in model.llm_engine.calls]
        self.assertEqual(methods, ["_nm_repcontrol_register_hook", "_nm_repcontrol_register_hook"])
        self.assertTrue(all(isinstance(m, str) for m in methods))

    def test_call_serializes_controller_tensor_for_rpc(self):
        # Responsible file: neuro_manipulation/repe/rep_control_vllm_hook.py
        # Purpose: vLLM v0.11+ RPC does not safely serialize torch.Tensor payloads.
        import torch

        from neuro_manipulation.repe.rep_control_vllm_hook import RepControlVLLMHook

        model = _FakeLLM()
        rep_control = RepControlVLLMHook(
            model=model,
            tokenizer=None,
            layers=[0],
            block_name="decoder_block",
            control_method="reading_vec",
            tensor_parallel_size=1,
        )

        activations = {0: torch.ones(4, dtype=torch.float32)}
        _ = rep_control(["hi"], activations=activations, masks=torch.ones(4, dtype=torch.float32), max_new_tokens=1)

        set_calls = [c for c in model.llm_engine.calls if c[0] == "_nm_repcontrol_set_state"]
        self.assertEqual(len(set_calls), 1)

        _method, _timeout, args, _kwargs = set_calls[0]
        self.assertEqual(len(args), 3)
        state = args[2]
        self.assertIsInstance(state, dict)
        self.assertIsInstance(state.get("controller"), list)
        self.assertIsInstance(state.get("mask"), list)

    def test_call_serializes_kwargs_recursively_for_rpc(self):
        # Responsible file: neuro_manipulation/repe/rep_control_vllm_hook.py
        # Purpose: kwargs passed through state must be RPC-serializable (no tensors/ndarrays).
        import numpy as np
        import torch

        from neuro_manipulation.repe.rep_control_vllm_hook import RepControlVLLMHook

        model = _FakeLLM()
        rep_control = RepControlVLLMHook(
            model=model,
            tokenizer=None,
            layers=[0],
            block_name="decoder_block",
            control_method="reading_vec",
            tensor_parallel_size=1,
        )

        _ = rep_control(
            ["hi"],
            activations={0: torch.ones(4, dtype=torch.float32)},
            masks=torch.ones(4, dtype=torch.float32),
            max_new_tokens=1,
            position_ids=torch.tensor([[0, 1, 2]]),
            nested={"arr": np.zeros((2,), dtype=np.float32)},
        )

        set_calls = [c for c in model.llm_engine.calls if c[0] == "_nm_repcontrol_set_state"]
        self.assertEqual(len(set_calls), 1)
        _method, _timeout, args, _kwargs = set_calls[0]
        state = args[2]
        self.assertIsInstance(state.get("kwargs"), dict)
        self.assertIsInstance(state["kwargs"]["position_ids"], list)
        self.assertIsInstance(state["kwargs"]["nested"]["arr"], list)
