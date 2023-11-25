import os

import fire
import gradio as gr
import torch
import transformers
from parse import parse_eval_args
import random
import json
from tqdm import tqdm

# offline
os.environ['HF_DATASETS_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'

from peft import (
    PeftModel,
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    prepare_model_for_int8_training,
    set_peft_model_state_dict,
)

from transformers import GenerationConfig, LlamaForCausalLM, LlamaTokenizer,AutoTokenizer
from utils.callbacks import Iteratorize, Stream
from utils.prompter import Prompter
if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

try:
    if torch.backends.mps.is_available():
        device = "mps"
except:
    pass


class Evaluator():
    def __init__(self, args):
        self.args = args
        self.prompter = None
        self.tokenizer = None
        self.model = None
        
    def model_init(self):
        args = self.args

        base_model = args.base_model or os.environ.get("BASE_MODEL", "")
        assert (
            base_model
        ), "Please specify a --base_model, e.g. --base_model='huggyllama/llama-7b'"

        self.prompter = Prompter(args.prompt_template)
        self.tokenizer = LlamaTokenizer.from_pretrained(base_model)
        if not args.lora_weights_path.endswith(".bin"):
            if device == "cuda":
                model = LlamaForCausalLM.from_pretrained(
                    base_model,
                    load_in_8bit=args.load_8bit,
                    torch_dtype=torch.float16,
                    device_map="auto",
                )
                model = PeftModel.from_pretrained(
                    model,
                    args.lora_weights_path,
                    torch_dtype=torch.float16,
                )
            elif device == "mps":
                model = LlamaForCausalLM.from_pretrained(
                    base_model,
                    device_map={"": device},
                    torch_dtype=torch.float16,
                )
                model = PeftModel.from_pretrained(
                    model,
                    args.lora_weights_path,
                    device_map={"": device},
                    torch_dtype=torch.float16,
                )
            else:
                model = LlamaForCausalLM.from_pretrained(
                    base_model, device_map={"": device}, low_cpu_mem_usage=True
                )
                model = PeftModel.from_pretrained(
                    model,
                    args.lora_weights_path,
                    device_map={"": device},
                )
        else:
            model = LlamaForCausalLM.from_pretrained(
                base_model,
                load_in_8bit=True,
                torch_dtype=torch.float16,
                device_map="auto",
            )
            model = prepare_model_for_int8_training(model)
            if args.be_trained:         # lora微调过
                config = LoraConfig.from_pretrained(args.lora_config_path)
                lora_weights = torch.load(args.lora_weights_path)
                model = PeftModel(model, config)
                set_peft_model_state_dict(model, lora_weights,"default")
                del lora_weights

        # unwind broken decapoda-research config
        model.config.pad_token_id = self.tokenizer.pad_token_id = 0  # unk
        model.config.bos_token_id = 1
        model.config.eos_token_id = 2

        model.eval()
        self.model = model

        
    def run(self, instruction, input=None, temperature=0.1, top_p=0.75, top_k=40, num_beams=4, max_new_tokens=128, **kwargs):
        prompt = self.prompter.generate_prompt(instruction, input)
        
        inputs = self.tokenizer(prompt, return_tensors="pt")
        input_ids = inputs['input_ids'].to(device)
        # input_mask = inputs['attention_mask'].to(device)
        generation_config = GenerationConfig(
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            num_beams=num_beams,
            **kwargs,
        )
        # if not args.load_8bit:
        #     input_ids = input_ids.half()  # 转换 input_ids 为半精度
            
        with torch.no_grad():
            generation_output = self.model.generate(
                input_ids = input_ids,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                num_beams=num_beams,
                max_new_tokens=2,
            )
        # output = generation_output.sequences[0]
        output = generation_output[0]
        response = self.tokenizer.decode(output, skip_special_tokens=True)
        # response = self.tokenizer.decode(generation_output[0], skip_special_tokens=True)
        print(response)
        return self.prompter.get_response(response)
    
    def load_json_data(self, file_path):
        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
        return data

if __name__ == "__main__":
    args = parse_eval_args()
    evaluator = Evaluator(args)
    evaluator.model_init()
    
    if args.dataset == "sst-2":
        all = 0
        correct = 0
        from data_download.GLUE.instructions import INSTRUCTIONS
        testset_path = './data_download/GLUE/sst-2/SST-2/SST-2_test.json'
        testset = evaluator.load_json_data(testset_path)
        for item in tqdm(testset, desc="Evaluating"):
            # print(f"Instruction: {item['instruction']}")
            # print(f"Context: {item['context']}")
            # print(f"Response: {item['response']}")
            # print(f"Category: {item['category']}\n")
            response = evaluator.run(instruction=item['instruction'], input=item['context'])
            if response.lower() == item['response'].lower():
                correct += 1
            all += 1
            acc = correct / all
            print(f"Accuracy of the {args.dataset} dataset: {acc:.4f} (Correct: {correct}, Total: {all})")

    elif args.dataset == "rte":
        all = 0
        correct = 0
        from data_download.GLUE.instructions import INSTRUCTIONS
        testset_path = './data_download/GLUE/rte/RTE/RTE_test.json'
        testset = evaluator.load_json_data(testset_path)
        for item in tqdm(testset, desc="Evaluating"):
            # print(f"Instruction: {item['instruction']}")
            # print(f"Context: {item['context']}")
            # print(f"Response: {item['response']}")
            # print(f"Category: {item['category']}\n")
            response = evaluator.run(instruction=item['instruction'], input=item['context'])
            if response.lower() == item['response'].lower():
                correct += 1
            all += 1
            acc = correct / all
            print(f"Accuracy of the {args.dataset} dataset: {acc:.4f} (Correct: {correct}, Total: {all})")
    
    elif args.dataset == "cola":
        all = 0
        correct = 0
        from data_download.GLUE.instructions import INSTRUCTIONS
        testset_path = './data_download/GLUE/cola/CoLA/CoLA_test.json'
        testset = evaluator.load_json_data(testset_path)
        for item in tqdm(testset, desc="Evaluating"):
            # print(f"Instruction: {item['instruction']}")
            # print(f"Context: {item['context']}")
            # print(f"Response: {item['response']}")
            # print(f"Category: {item['category']}\n")
            response = evaluator.run(instruction=item['instruction'], input=item['context'])
            if response.lower() == item['response'].lower():
                correct += 1
            all += 1
            acc = correct / all
            print(f"Accuracy of the {args.dataset} dataset: {acc:.4f} (Correct: {correct}, Total: {all})")


    elif args.dataset == "qnli":
        all = 0
        correct = 0
        from data_download.GLUE.instructions import INSTRUCTIONS
        testset_path = './data_download/GLUE/qnli/QNLI/QNLI_test.json'
        testset = evaluator.load_json_data(testset_path)
        for item in tqdm(testset, desc="Evaluating"):
            # print(f"Instruction: {item['instruction']}")
            # print(f"Context: {item['context']}")
            # print(f"Response: {item['response']}")
            # print(f"Category: {item['category']}\n")
            response = evaluator.run(instruction=item['instruction'], input=item['context'])
            if response.lower() == item['response'].lower():
                correct += 1
            all += 1
            acc = correct / all
            print(f"Accuracy of the {args.dataset} dataset: {acc:.4f} (Correct: {correct}, Total: {all})")

    elif args.dataset == "mrpc":
        all = 0
        correct = 0
        from data_download.GLUE.instructions import INSTRUCTIONS
        testset_path = './data_download/GLUE/mrpc/MRPC/MRPC_test.json'
        testset = evaluator.load_json_data(testset_path)
        for item in tqdm(testset, desc="Evaluating"):
            # print(f"Instruction: {item['instruction']}")
            # print(f"Context: {item['context']}")
            # print(f"Response: {item['response']}")
            # print(f"Category: {item['category']}\n")
            response = evaluator.run(instruction=item['instruction'], input=item['context'])
            if response.lower() == item['response'].lower():
                correct += 1
            all += 1
            acc = correct / all
            print(f"Accuracy of the {args.dataset} dataset: {acc:.4f} (Correct: {correct}, Total: {all})")
    

    elif args.dataset == "rte":
        all = 0
        correct = 0
        from data_download.GLUE.instructions import INSTRUCTIONS
        testset_path = './data_download/GLUE/rte/RTE/RTE_test.json'
        testset = evaluator.load_json_data(testset_path)
        for item in tqdm(testset, desc="Evaluating"):
            # print(f"Instruction: {item['instruction']}")
            # print(f"Context: {item['context']}")
            # print(f"Response: {item['response']}")
            # print(f"Category: {item['category']}\n")
            response = evaluator.run(instruction=item['instruction'], input=item['context'])
            if response.lower() == item['response'].lower():
                correct += 1
            all += 1
            acc = correct / all
            print(f"Accuracy of the {args.dataset} dataset: {acc:.4f} (Correct: {correct}, Total: {all})")
    

    elif args.dataset == "sst-2":
        all = 0
        correct = 0
        from data_download.GLUE.instructions import INSTRUCTIONS
        testset_path = './data_download/GLUE/sst-2/SST-2/SST-2_test.json'
        testset = evaluator.load_json_data(testset_path)
        for item in tqdm(testset, desc="Evaluating"):
            # print(f"Instruction: {item['instruction']}")
            # print(f"Context: {item['context']}")
            # print(f"Response: {item['response']}")
            # print(f"Category: {item['category']}\n")
            response = evaluator.run(instruction=item['instruction'], input=item['context'])
            if response.lower() == item['response'].lower():
                correct += 1
            all += 1
            acc = correct / all
            print(f"Accuracy of the {args.dataset} dataset: {acc:.4f} (Correct: {correct}, Total: {all})")


    elif args.dataset == "wnli":
        all = 0
        correct = 0
        from data_download.GLUE.instructions import INSTRUCTIONS
        testset_path = './data_download/GLUE/wnli/WNLI/WNLI_test.json'
        testset = evaluator.load_json_data(testset_path)
        for item in tqdm(testset, desc="Evaluating"):
            # print(f"Instruction: {item['instruction']}")
            # print(f"Context: {item['context']}")
            # print(f"Response: {item['response']}")
            # print(f"Category: {item['category']}\n")
            response = evaluator.run(instruction=item['instruction'], input=item['context'])
            if response.lower() == item['response'].lower():
                correct += 1
            all += 1
            acc = correct / all
            print(f"Accuracy of the {args.dataset} dataset: {acc:.4f} (Correct: {correct}, Total: {all})")