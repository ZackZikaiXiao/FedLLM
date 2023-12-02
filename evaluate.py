import os

import fire
import gradio as gr
import torch
import transformers
from parse import parse_eval_args
import random
import json
import pandas as pd
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
        self.testset_path =  {
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
        self.save_path = {
            "sst-2": "./output/GLUE/sst-2/alpaca.xlsx",
            "rte": "./output/GLUE/rte/alpaca.xlsx",
            "qnli": "./output/GLUE/qnli/alpaca.xlsx",
            "cola": "./output/GLUE/cola/alpaca.xlsx",
            "mnli": "./output/GLUE/mnli/alpaca.xlsx",
            "mrpc": "./output/GLUE/mrpc/alpaca.xlsx",
            "qqp": "./output/GLUE/qqp/alpaca.xlsx",
            "sts-b": "./output/GLUE/sts-b/alpaca.xlsx",
            "wnli": "./output/GLUE/wnli/alpaca.xlsx",
        }
        
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
        if args.be_trained:         # peft微调过
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
        # print(list_of_response)
        return batch_input['full_prompt'], response, list_of_response

    def run(self, instruction, input=None, temperature=0.1, top_p=0.75, top_k=40, num_beams=4, max_new_tokens=32, **kwargs):
        full_prompt = self.prompter.generate_prompt(instruction, input)
         
        inputs = self.tokenizer(full_prompt, return_tensors="pt")
        input_ids = inputs['input_ids'].to(device)
        # input_mask = inputs['attention_mask'].to(device)
        generation_config = GenerationConfig(
            temperature=temperature,
            do_sample=True,
            top_p=top_p,
            top_k=top_k,
            num_beams=num_beams,
            max_new_tokens=max_new_tokens,
            **kwargs,
        )
        # if not args.load_8bit:
        #     input_ids = input_ids.half()  # 转换 input_ids 为半精度

        with torch.no_grad():
            generation_output = self.model.generate(
                input_ids=input_ids,
                generation_config = generation_config,
            )
        # output = generation_output.sequences[0]
        output = generation_output[0]
        full_response = self.tokenizer.decode(output, skip_special_tokens=True)
        # response = self.tokenizer.decode(generation_output[0], skip_special_tokens=True)
        split_response = self.prompter.get_response(full_response)
        return full_prompt, full_response, split_response
    
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
    
    def pearson_correlation(self, excel_file_path):
        df = pd.read_excel(excel_file_path)
        df['label'] = pd.to_numeric(df['label'], errors='coerce')
        df['split_response'] = pd.to_numeric(df['split_response'], errors='coerce')

        pearson_correlation = df['split_response'].corr(df['label'])
        return pearson_correlation

def write_to_file(index, result):
    with open("evaluate_result", 'a') as f:
        f.write(str(index) + " " + str(result) + '\n')

def batch_evaluate(num_communication_rounds, args_passed=None):
    args = parse_eval_args()
    if args_passed:
        args.model = args_passed.model
        args.peft_method = args_passed.peft_method
        args.dataset = args_passed.dataset
        args.peft_config_path = args_passed.output_dir
        args.peft_weights_path = args.peft_config_path + '/0/adapter_model.bin'
        args.base_model = args_passed.global_model
    evaluator = Evaluator(args)
    evaluator.model_init()
    testset = load_dataset("json", data_files=evaluator.testset_path[args.dataset])
    cols = ['instruction', 'response', 'context', 'category']
    cleared_testset = testset["train"].shuffle().map(evaluator.generate_prompt, remove_columns=cols)
    cleared_testset.set_format(type="torch", columns=["full_prompt", "label"])
    dataloader = DataLoader(cleared_testset, batch_size=64, drop_last=False)
    
    for index in range(num_communication_rounds):
        peft_weights_path = os.path.join(args.peft_config_path, str(index), "adapter_model.bin")
        evaluator.reset_peft_adapter(peft_weights_path)
        all = 0
        correct = 0
        for batch in tqdm(dataloader, desc="Evaluating"):
            full_prompt_list, full_response_list, list_of_response = evaluator.batch_run(batch)
            for pred, label in zip(list_of_response, batch['label']):
                if (pred.lower() == label.lower()):
                    correct += 1
            all += len(batch['label'])
            acc = correct / all
            print(f"Accuracy of the {args.dataset} dataset: {acc:.4f} (Correct: {correct}, Total: {all})")
        write_to_file(index, acc)

def batch_eva_write_to_excel(num_communication_rounds, args_passed=None):
    args = parse_eval_args()
    if args_passed:
        args.model = args_passed.model
        args.peft_method = args_passed.peft_method
        args.dataset = args_passed.dataset
        args.peft_config_path = args_passed.output_dir
        args.peft_weights_path = args.peft_config_path + '/0/adapter_model.bin'
        args.base_model = args_passed.global_model
    
    evaluator = Evaluator(args)
    evaluator.model_init()
    testset = load_dataset("json", data_files=evaluator.testset_path[args.dataset])
    cols = ['instruction', 'response', 'context', 'category']
    cleared_testset = testset["train"].shuffle().map(evaluator.generate_prompt, remove_columns=cols)
    cleared_testset.set_format(type="torch", columns=["full_prompt", "label"])
    dataloader = DataLoader(cleared_testset, batch_size=64, drop_last=False)
    
    for index in range(num_communication_rounds):
        save_excel = pd.DataFrame(columns=["full_prompt", "full_response", "response", "label", "match", "accuracy"])
        peft_weights_path = os.path.join(args.peft_config_path, str(index), "adapter_model.bin")
        evaluator.reset_peft_adapter(peft_weights_path)
        all = 0
        correct = 0
        match_list = []
        full_prompt_list2 = []
        full_response_list2 = []
        list_of_response2 = []
        labels = []
        for batch in tqdm(dataloader, desc="Evaluating"):
            full_prompt_list, full_response_list, list_of_response = evaluator.batch_run(batch)
            full_prompt_list2.extend(full_prompt_list)
            full_response_list2.extend(full_response_list)
            list_of_response2.extend(list_of_response)
            labels.extend(batch['label'])
            for pred, label in zip(list_of_response, batch['label']):
                if (pred.lower() == label.lower()):
                    correct += 1
                    match_list.extend([1])
                else:
                    match_list.extend([0])
            all += len(batch['label'])
            acc = correct / all
            print(f"Accuracy of the {args.dataset} dataset: {acc:.4f} (Correct: {correct}, Total: {all})")
        save_excel['full_prompt'] = full_prompt_list2
        save_excel['full_response'] = full_response_list2
        save_excel['response'] = list_of_response2
        save_excel['label'] = labels
        save_excel['match'] = match_list
        save_excel['accuracy'] = [acc] * len(match_list)
        directory = os.path.dirname(eavluator.save_path[args.dataset])
        if not os.path.exists(directory):
            os.makedirs(directory)
        file_name = os.path.join(directory, str(index) + '.xlsx')
        save_excel.to_excel(file_name, index=False)

if __name__ == "__main__":
    # batch_evaluate(10)
    batch_eva_write_to_excel(10)

    args = parse_eval_args()
    evaluator = Evaluator(args)
    evaluator.model_init()
    
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
    save_path = {
    "sst-2": "./output/GLUE/sst-2/alpaca.xlsx",
    "rte": "./output/GLUE/rte/alpaca.xlsx",
    "qnli": "./output/GLUE/qnli/alpaca.xlsx",
    "cola": "./output/GLUE/cola/alpaca.xlsx",
    "mnli": "./output/GLUE/mnli/alpaca.xlsx",
    "mrpc": "./output/GLUE/mrpc/alpaca.xlsx",
    "qqp": "./output/GLUE/qqp/alpaca.xlsx",
    "sts-b": "./output/GLUE/sts-b/alpaca.xlsx",
    "wnli": "./output/GLUE/wnli/alpaca.xlsx",
    }

    

    all = 0
    correct = 0
    from data_download.GLUE.instructions import INSTRUCTIONS
    testset = evaluator.load_json_data(testset_path[args.dataset])
    
    if args.dataset == "sts-b":     # 斯皮尔曼系数
        
        directory = os.path.dirname(save_path[args.dataset])
        if not os.path.exists(directory):
            os.makedirs(directory)
            
        save_excel = pd.DataFrame(columns=["instruction", "context", 
                                      "label", "category", 
                                      "full_prompt", "full_response", 
                                      "split_response"
                                      ])
        # 计算保存间隔
        interval = len(testset) // 100
        counter = 0
        for item in tqdm(testset, desc="Evaluating"):
            full_prompt, full_response, split_response = evaluator.run(instruction=item['instruction'], input=item['context'])
            print(f"Output: {split_response}, Label: {item['response']}")
            save_excel.loc[len(save_excel)] = [item['instruction'], item['context'], item['response'], item['category'],
                                        full_prompt, full_response, split_response]

            # 每当达到保存间隔时保存 Excel 文件
            if counter % interval == 0 and counter > 0:
                save_excel.to_excel(save_path[args.dataset], index=False)
                pearson_correlation = evaluator.pearson_correlation()
                print("Pearson Correlation Coefficient:", pearson_correlation)
            counter += 1
        

    elif args.dataset == "cola" or args.dataset == "sst-2" or args.dataset == "rte" or args.dataset == "qnli":
        save_excel = pd.DataFrame(columns=["instruction", "context", 
                                      "label", "category", 
                                      "full_prompt", "full_response", 
                                      "split_response", "match", "accuracy"
                                      ])
        
        for item in tqdm(testset, desc="Evaluating"):
            full_prompt, full_response, split_response = evaluator.run(instruction=item['instruction'], input=item['context'])
            print(full_response)
            print(f"Output: {str(split_response)}, Label: {str(item['response'])}")
            match = str(split_response).lower() == str(item['response']).lower()
            
            save_excel.loc[len(save_excel)] = [item['instruction'], item['context'], item['response'], item['category'],
                                       full_prompt, full_response, split_response, str(int(match)), str(correct)+"/"+str(all)]
            if match:
                correct += 1
            all += 1
            acc = correct / all
            print(f"Accuracy of the {args.dataset} dataset: {acc:.4f} (Correct: {correct}, Total: {all})")
    
    
    directory = os.path.dirname(save_path[args.dataset])
    if not os.path.exists(directory):
        os.makedirs(directory)
    save_excel.to_excel(save_path[args.dataset], index=False)


                
                    
