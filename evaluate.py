import os

import fire
import gradio as gr
import torch
import transformers
from parse import parse_eval_args
import random
import json
from tqdm import tqdm
from datasets import load_dataset
from data_tool.data_tokenizer import DataTokenizer
from torch.utils.data import DataLoader
from model_utils.get_model import get_alpaca_model_and_tokenizer, get_llama27b_model_and_tokenizer
# offline
os.environ['HF_DATASETS_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
from peft import (
    PeftModel,
    LoraConfig,
    PrefixTuningConfig,
    get_peft_model,
    get_peft_model_state_dict,
    prepare_model_for_kbit_training,
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

        self.prompter = Prompter(args.prompt_template_name)
        # set legacy=True means use the previous version, to fix the user warning
        if args.model == 'alpaca':
            model, self.tokenizer = get_alpaca_model_and_tokenizer(base_model, 'auto')
        elif args.model == 'Llama2-7B':
            model, self.tokenizer = get_llama27b_model_and_tokenizer(base_model, 'auto')
        model = prepare_model_for_kbit_training(model)
        if args.be_trained:         # lora微调过
            # print(args.lora_config_path)
            if args.peft_method == 'lora':
                config = LoraConfig.from_pretrained(args.peft_config_path)
            elif args.peft_method == 'prefix_tuning':
                config = PrefixTuningConfig.from_pretrained(args.peft_config_path)
            peft_weights = torch.load(args.peft_weights_path)
            model = PeftModel(model, config)
            set_peft_model_state_dict(model, peft_weights, "default")
            del peft_weights
        model.eval()
        self.model = model

    def reset_peft_adapter(self, peft_config_path):
        peft_weights = torch.load(peft_config_path)
        set_peft_model_state_dict(self.model, peft_weights,"default")
        self.model.eval()
        del peft_weights

    def batch_run(self, batch_input):
        tokenized_inputs = self.tokenizer(batch_input['full_prompt'], padding='longest', return_tensors="pt")
        tokenized_inputs = tokenized_inputs.to(device)
        generation_config = GenerationConfig(
            max_new_tokens=10,
        )
        with torch.no_grad():
            generation_output = self.model.generate(
                **tokenized_inputs,
                generation_config = generation_config,
            )
        response = self.tokenizer.batch_decode(generation_output, skip_special_tokens=True)
        list_of_response = [self.prompter.get_response(res) for res in response]
        print(list_of_response)
        return(list_of_response)

    def run(self, instruction, input=None, temperature=0.1, top_p=0.75, top_k=40, num_beams=4, max_new_tokens=128, **kwargs):
        prompt = self.prompter.generate_prompt(instruction, input)
         
        inputs = self.tokenizer(prompt, return_tensors="pt")
        input_ids = inputs['input_ids'].to(device)
        # input_mask = inputs['attention_mask'].to(device)
        generation_config = GenerationConfig(
            max_new_tokens=10,
        )
        # if not args.load_8bit:
        #     input_ids = input_ids.half()  # 转换 input_ids 为半精度
            
        with torch.no_grad():
            generation_output = self.model.generate(
                # 如果我们只需要一个输出结果，这些注释掉的设置不需要
                input_ids = input_ids,
                generation_config = generation_config,
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
    
    def generate_prompt(self, data_point):
        full_prompt = self.prompter.generate_prompt(
            data_point["instruction"],
            data_point["context"],
        )
        # tokenized_prompt = self.tokenizer(full_prompt, padding='longest', return_tensors="pt")
        data_dict = {
            "full_prompt": full_prompt,
            "label": data_point['response']
        }
        return data_dict
def write_to_file(index, result):
    with open("evaluate_result", 'a') as f:
        f.write(str(index) + " " + str(result) + '\n')


if __name__ == "__main__":
    testset_path = {
    "sst-2": "./data_download/GLUE/sst-2/SST-2/SST-2_test.json",
    "rte": "./data_download/GLUE/rte/RTE/RTE_test.json",
    "qnli": "./data_download/GLUE/qnli/QNLI/QNLI_test.json",
    "cola": "./data_download/GLUE/cola/CoLA/CoLA_test.json",
    "mnli": "./data_download/GLUE/mnli/MNLI/MNLI_test.json",
    "mrpc": "./data_download/GLUE/mrpc/MRPC/MRPC_test.json",
    "qqp": "./data_download/GLUE/qqp/QQP/QQP_test.json",
    "sts-b": "./data_download/GLUE/sts-b/STS-B/STS-B_test.json",
    "wnli": "./data_download/GLUE/wnli/WNLI/WNLI_test.json",
    }
    args = parse_eval_args()
    auto_testing = True
    batch_running = True
    if auto_testing and not batch_running:
        num_communication_rounds = 5
        for index in range(num_communication_rounds):
            args.peft_weights_path = os.path.join(args.peft_config_path, str(index), "adapter_model.bin")
            evaluator = Evaluator(args)
            evaluator.model_init()
            all = 0
            correct = 0
            testset = evaluator.load_json_data(testset_path[args.dataset])
            from data_download.GLUE.instructions import INSTRUCTIONS
            if args.dataset == "sts-b":
                pass
            else:
                for item in tqdm(testset, desc="Evaluating"):
                    response = evaluator.run(instruction=item['instruction'], input=item['context'])
                    if response.lower() == item['response'].lower():
                        correct += 1
                    all += 1
                    acc = correct / all
                    print(f"Accuracy of the {args.dataset} dataset: {acc:.4f} (Correct: {correct}, Total: {all})")
            
                write_to_file(index, acc)


    elif auto_testing and batch_running:
        num_communication_rounds = 3
        evaluator = Evaluator(args)
        evaluator.model_init()
        testset = load_dataset("json", data_files=testset_path[args.dataset])
        data_tokenizer = DataTokenizer(args, evaluator.tokenizer)
        cols = ['instruction', 'response', 'context', 'category']
        cleared_testset = testset["train"].shuffle().map(evaluator.generate_prompt, remove_columns=cols)
        cleared_testset.set_format(type="torch", columns=["full_prompt", "label"])
        dataloader = DataLoader(cleared_testset, batch_size=64, drop_last=False)
        
        for index in range(num_communication_rounds):
            # args.lora_weights_path = os.path.join(args.lora_config_path, str(index), "adapter_model.bin")
            # evaluator = Evaluator(args)
            # evaluator.model_init()
            
            # peft_weights_path = os.path.join(args.peft_config_path, str(0), "local_output_2", "pytorch_model.bin")
            peft_weights_path = os.path.join(args.peft_config_path, str(index), "adapter_model.bin")
            evaluator.reset_peft_adapter(peft_weights_path)
            all = 0
            correct = 0
            for batch in tqdm(dataloader, desc="Evaluating"):
                list_of_response = evaluator.batch_run(batch)
                for pred, label in zip(list_of_response, batch['label']):
                    if (pred.lower() == label.lower()):
                        correct += 1
                all += len(batch['label'])
                acc = correct / all
                print(f"Accuracy of the {args.dataset} dataset: {acc:.4f} (Correct: {correct}, Total: {all})")
            write_to_file(index, acc)


                
                    
