import torch
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    pipeline,
)
from peft import  PeftModel


def pipeline(base_model_path, new_model_path):

    device_map = {"": 0}

    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        low_cpu_mem_usage=True,
        return_dict=True,
        torch_dtype=torch.float16,
        device_map=device_map,
    )
    
    model = PeftModel.from_pretrained(base_model, new_model_path)
    model = model.merge_and_unload()

    # Reload tokenizer to save it
    tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    pipeline_kwargs={
    "temperature": 1.0,
    "max_new_tokens": 250,
    "top_k": 1,
    "repetition_penalty": 1.1
    }

    pipe = pipeline(
        model=model,
        tokenizer=tokenizer,
        return_full_text=False,
        task='text-generation',
        **pipeline_kwargs
    )
    return pipe