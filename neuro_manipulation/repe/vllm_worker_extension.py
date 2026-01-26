"""
vLLM worker extension for RepControl hooks.

vLLM v0.11+ disallows serializing Python callables for collective RPC by default.
This extension exposes string-addressable worker methods that delegate to the
existing RepControl worker RPC helpers.
"""


class NMRepControlWorkerExtension:
    def _nm_repcontrol_register_hook(self, layer_index: int, block_name: str) -> bool:  # type: ignore[no-untyped-def]
        from neuro_manipulation.repe.rep_control_vllm_hook import (
            _register_hook_on_worker_rpc,
            hook_fn_rep_control,
        )

        return _register_hook_on_worker_rpc(
            self, layer_index, block_name, hook_fn_rep_control
        )

    def _nm_repcontrol_set_state(self, layer_index: int, block_name: str, state) -> bool:  # type: ignore[no-untyped-def]
        from neuro_manipulation.repe.rep_control_vllm_hook import (
            _set_controller_state_on_worker_rpc,
        )

        return _set_controller_state_on_worker_rpc(self, layer_index, block_name, state)

    def _nm_repcontrol_reset_state(self, layer_index: int, block_name: str) -> bool:  # type: ignore[no-untyped-def]
        from neuro_manipulation.repe.rep_control_vllm_hook import (
            _reset_controller_state_on_worker_rpc,
        )

        return _reset_controller_state_on_worker_rpc(self, layer_index, block_name)

