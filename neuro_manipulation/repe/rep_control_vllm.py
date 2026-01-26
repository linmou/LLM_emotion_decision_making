from types import MethodType
from vllm import LLM, SamplingParams
import torch

from neuro_manipulation.repe.rep_control_reading_vec import WrappedReadingVecModel

class RepControlVLLM:
    def __init__(self, model, tokenizer, layers, block_name, control_method):
        self.model = model
        self.tokenizer = tokenizer
        self.layers = layers
        self.block_name = block_name
        self.control_method = control_method
        
        assert control_method == "reading_vec", f"{control_method} not supported yet"
        assert (block_name == "decoder_block" or 
                "LlamaForCausalLM" in model.config.architectures or
                "Qwen2ForCausalLM" in model.config.architectures or
                any("Qwen" in arch for arch in model.config.architectures)), \
               f"{model.config.architectures} {block_name} not supported yet"
        
        self.raw_llm = self.model.llm_engine.model_executor.driver_worker.model_runner.model #FIXME: hardcoded for now  
        self.wrapped_model = WrappedReadingVecModel(self.model, self.tokenizer, self.raw_llm) # TODO: 
        
        self.wrapped_model.unwrap()
        self.wrapped_model.wrap_block(layers, block_name=block_name)
        
    def __call__(self, text_inputs: list[str], activations=None, **kwargs):
        if activations is not None:
            self.wrapped_model.reset()
            self.wrapped_model.set_controller(self.layers, activations, self.block_name)

        tokens = self.tokenizer(text_inputs)
        prompt_token_ids = tokens['input_ids']
        sampling_params = SamplingParams(
                    max_tokens=kwargs.get('max_new_tokens', 40000),
                    temperature=kwargs.get('temperature', 0.7),
                    repetition_penalty=kwargs.get('repetition_penalty', 1.1),
                    top_p=kwargs.get('top_p', 0.95)
                )
                
        outputs = self.wrapped_model.generate(prompt_token_ids=prompt_token_ids, sampling_params=sampling_params)
        self.wrapped_model.reset()

        return outputs

 
if __name__ == "__main__":
    from transformers import AutoTokenizer
    model = "meta-llama/Llama-3.1-8B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model)
    model = LLM(model=model, tensor_parallel_size=4, max_model_len=40000, gpu_memory_utilization=0.9)
    layers = [0, 1, 2, 3, 4, 5]
    block_name = "decoder_block"
    control_method = "reading_vec"
    rep_control = RepControlVLLM(model, tokenizer, layers, block_name, control_method)
    text_inputs = ["Hello, how are you?", "Hello, how are you?"]
    outputs = rep_control(text_inputs)
    print(outputs)