import torch
from transformers import AutoModel, AutoModelForCausalLM, AutoProcessor
from transformers.pipelines import PIPELINE_REGISTRY
from transformers.pipelines import pipeline

from neuro_manipulation.repe.rep_control_vllm import RepControlVLLM

from neuro_manipulation.repe.rep_control_vllm_hook import RepControlVLLMHook
from neuro_manipulation.repe.rep_reading_vllm import RepReadingVLLM
from .rep_reading_pipeline import RepReadingPipeline
from .rep_control_pipeline import RepControlPipeline
from .rep_reading_prob_calc_pipeline import RepReadingNProbCalcPipeline

from vllm import LLM
from transformers.generation.utils import GenerationMixin

def repe_pipeline_registry():
    PIPELINE_REGISTRY.register_pipeline(
        "rep-reading",
        pipeline_class=RepReadingPipeline,
        pt_model=AutoModel,
    )

    # Multimodal representation reading pipeline
    PIPELINE_REGISTRY.register_pipeline(
        "multimodal-rep-reading",
        pipeline_class=RepReadingPipeline,
        pt_model=(AutoModel, AutoModelForCausalLM),  # Support both for multimodal models
    )

    PIPELINE_REGISTRY.register_pipeline(
        "rep-control",
        pipeline_class=RepControlPipeline,
        pt_model=AutoModelForCausalLM,
    )
    PIPELINE_REGISTRY.register_pipeline(
        "rep-reading&prob-calc",
        pipeline_class=RepReadingNProbCalcPipeline,
        pt_model=AutoModelForCausalLM,
    ) 

vllm_task2pipeline = {
    "rep-control-vllm": RepControlVLLMHook,
    "rep-reading-vllm": RepReadingVLLM,
    "multimodal-rep-reading-vllm": RepReadingVLLM,  # Use same vLLM implementation for multimodal
}

def get_pipeline(task, model, tokenizer, layers, block_name, control_method):
    if 'vllm' in task:
        return vllm_task2pipeline[task](model, tokenizer, layers, block_name, control_method)
    else:
        return pipeline(task, model=model, tokenizer=tokenizer, layers=layers, block_name=block_name, control_method=control_method)

if __name__ == "__main__":
    repe_pipeline_registry()
    from transformers import AutoTokenizer, AutoModelForCausalLM
    model_hf = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.1-8B-Instruct", device_map="auto", torch_dtype=torch.bfloat16)
    model_vllm = LLM(model="meta-llama/Llama-3.1-8B-Instruct", tensor_parallel_size=2, max_model_len=40000, gpu_memory_utilization=0.9)    
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
    print(get_pipeline("rep-control", model_hf, tokenizer, [0], "decoder_block", "reading_vec")(["Hello, how are you?"]))
    print(get_pipeline("rep-control-vllm", model_vllm, tokenizer, [0], "decoder_block", "reading_vec")(["Hello, how are you?"]))