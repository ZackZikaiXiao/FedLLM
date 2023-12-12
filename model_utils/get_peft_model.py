from peft import (
    LoraConfig,
    PrefixTuningConfig,
    get_peft_model,
)

def get_lora_peft_model(args, model):
    config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=args.lora_target_modules,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, config)
    return model, config

def get_prefix_tuning_peft_model(args, model):
    config = PrefixTuningConfig(
        task_type="CAUSAL_LM",
        # inference_mode=False,
        num_virtual_tokens=args.num_virtual_tokens,
    )
    model = get_peft_model(model, config)
    return model, config

if __name__ == "__main__":
    config = LoraConfig(
        bias="none",
        task_type="CAUSAL_LM",
    )
    print('a')
    