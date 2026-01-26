# Tests for neuro_manipulation/repe/rep_reading_pipeline.py: ensure hidden-state extraction disables KV cache.

import unittest

import torch


class _DummyModelWithLegacyCacheBug(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.device = torch.device("cpu")

    def forward(self, input_ids=None, attention_mask=None, output_hidden_states=False, use_cache=True, **kwargs):
        if use_cache:
            past_key_values = object()  # no get_usable_length
            # Mimic Phi-3.5 custom modeling code calling a missing legacy API.
            past_key_values.get_usable_length(0)  # type: ignore[attr-defined]
        if not output_hidden_states:
            raise AssertionError("test expects output_hidden_states=True")
        hidden = torch.zeros((input_ids.shape[0], input_ids.shape[1], 4), dtype=torch.float32)
        return {"hidden_states": [hidden]}


class TestRepReadingPipelineUseCacheDisabled(unittest.TestCase):
    def test_forward_for_hidden_states_disables_use_cache(self):
        from neuro_manipulation.repe.rep_reading_pipeline import RepReadingPipeline

        pipe = RepReadingPipeline.__new__(RepReadingPipeline)
        pipe.model = _DummyModelWithLegacyCacheBug()
        pipe.tokenizer = None

        model_inputs = {
            "input_ids": torch.tensor([[1, 2, 3]]),
            "attention_mask": torch.tensor([[1, 1, 1]]),
        }

        hidden = RepReadingPipeline._forward(
            pipe,
            model_inputs=model_inputs,
            rep_token=-1,
            hidden_layers=[0],
            rep_reader=None,
            component_index=0,
            which_hidden_states=None,
            pad_token_id=None,
        )

        self.assertIn(0, hidden)

