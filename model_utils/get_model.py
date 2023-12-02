from transformers import LlamaTokenizer, LlamaForCausalLM
import torch
from peft import (
    prepare_model_for_kbit_training,
)


def get_alpaca_model_and_tokenizer(global_model, device_map='auto'):
    model = LlamaForCausalLM.from_pretrained(
        global_model,
        load_in_8bit=True,     # True
        torch_dtype=torch.float16,
        device_map=device_map,
    )

    tokenizer = LlamaTokenizer.from_pretrained(global_model, use_fast=False, legacy=True)
    # 但是这里0 decode出来是<unk>
    tokenizer.pad_token_id = (0)
    tokenizer.padding_side = "left"

    return model, tokenizer

def get_llama27b_model_and_tokenizer(global_model, device_map='auto'):
    model = LlamaForCausalLM.from_pretrained(
        global_model,
        load_in_8bit=True,     # True
        torch_dtype=torch.bfloat16,
        device_map=device_map,
    )
    tokenizer = LlamaTokenizer.from_pretrained(global_model, use_fast=False, legacy=True)
    model.config.pad_token_id = tokenizer.pad_token_id = 0
    tokenizer.padding_side = "left"
    return model, tokenizer