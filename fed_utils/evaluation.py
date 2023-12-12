import os
import sys
sys.path.append("./")
import gradio as gr
import torch
from parse import parse_eval_args, parse_train_args
import random
import json
import pandas as pd
from tqdm import tqdm
from datasets import load_dataset
from torch.utils.data import DataLoader
from model_utils.get_model import get_alpaca_model_and_tokenizer, get_llama27b_model_and_tokenizer
from sklearn.metrics import matthews_corrcoef, f1_score
from scipy.stats import pearsonr
from output.GLUE.postprocess import cleansed_response
# offline
os.environ['HF_DATASETS_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
from peft import (
    PeftModel,
    LoraConfig,
    PrefixTuningConfig,
    prepare_model_for_kbit_training,
    set_peft_model_state_dict,
    get_peft_model,
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
            "sst-2": "./output/GLUE/sst-2/",
            "rte": "./output/GLUE/rte/",
            "qnli": "./output/GLUE/qnli/",
            "cola": "./output/GLUE/cola/",
            "mnli": "./output/GLUE/mnli/",
            "mrpc": "./output/GLUE/mrpc/",
            "qqp": "./output/GLUE/qqp/",
            "sts-b": "./output/GLUE/sts-b/",
            "wnli": "./output/GLUE/wnli/",
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
            config.inference_mode = True
            model = get_peft_model(model, config)
            # model = PeftModel(model, config)
            set_peft_model_state_dict(model, peft_weights, "default")
            del peft_weights
        model.eval()
        self.model = model

    def reset_peft_adapter(self, peft_config_path):
        if self.args.be_trained:
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

    # def run(self, instruction, input=None, temperature=0.1, top_p=0.75, top_k=40, num_beams=4, max_new_tokens=32, **kwargs):
    #     full_prompt = self.prompter.generate_prompt(instruction, input)
         
    #     inputs = self.tokenizer(full_prompt, return_tensors="pt")
    #     input_ids = inputs['input_ids'].to(device)
    #     # input_mask = inputs['attention_mask'].to(device)
    #     generation_config = GenerationConfig(
    #         temperature=temperature,
    #         do_sample=True,
    #         top_p=top_p,
    #         top_k=top_k,
    #         num_beams=num_beams,
    #         max_new_tokens=max_new_tokens,
    #         **kwargs,
    #     )
    #     # if not args.load_8bit:
    #     #     input_ids = input_ids.half()  # 转换 input_ids 为半精度

    #     with torch.no_grad():
    #         generation_output = self.model.generate(
    #             input_ids=input_ids,
    #             generation_config = generation_config,
    #         )
    #     # output = generation_output.sequences[0]
    #     output = generation_output[0]
    #     full_response = self.tokenizer.decode(output, skip_special_tokens=True)
    #     # response = self.tokenizer.decode(generation_output[0], skip_special_tokens=True)
    #     split_response = self.prompter.get_response(full_response)
    #     return full_prompt, full_response, split_response
    
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
    
    # def pearson_correlation(self, excel_file_path):
    #     df = pd.read_excel(excel_file_path)
    #     df['label'] = pd.to_numeric(df['label'], errors='coerce')
    #     df['split_response'] = pd.to_numeric(df['split_response'], errors='coerce')

    #     pearson_correlation = df['split_response'].corr(df['label'])
    #     return pearson_correlation

def write_to_file(index, result, file_name=None):
    if file_name:
        with open(file_name, 'a') as f:
            f.write(str(index) + " " + str(result) + '\n')
    else:
        with open("evaluate_result", 'a') as f:
            f.write(str(index) + " " + str(result) + '\n')

# def batch_evaluate(num_communication_rounds, args_passed=None, metrics='accuracy', positive_label=None):
#     args = parse_eval_args()
#     if args_passed:
#         args.model = args_passed.model
#         args.peft_method = args_passed.peft_method
#         args.dataset = args_passed.dataset
#         args.peft_config_path = args_passed.output_dir
#         args.peft_weights_path = args.peft_config_path + '/0/adapter_model.bin'
#         args.base_model = args_passed.global_model
#     evaluator = Evaluator(args)
#     evaluator.model_init()
#     testset = load_dataset("json", data_files=evaluator.testset_path[args.dataset])
#     cols = ['instruction', 'response', 'context', 'category']
#     cleared_testset = testset["train"].shuffle().map(evaluator.generate_prompt, remove_columns=cols)
#     cleared_testset.set_format(type="torch", columns=["full_prompt", "label"])
#     dataloader = DataLoader(cleared_testset, batch_size=64, drop_last=False)

#     for index in range(num_communication_rounds):
#         peft_weights_path = os.path.join(args.peft_config_path, str(index), "adapter_model.bin")
#         evaluator.reset_peft_adapter(peft_weights_path)
#         all = 0
#         correct = 0
#         list_of_response2 = []
#         labels = []
#         for batch in tqdm(dataloader, desc="Evaluating"):
#             full_prompt_list, full_response_list, list_of_response = evaluator.batch_run(batch)
#             list_of_response = cleansed_response(list_of_response)
#             for pred, label in zip(list_of_response, batch['label']):
#                 if (pred.lower() == label.lower()):
#                     correct += 1
#             all += len(batch['label'])
#             acc = correct / all
#             list_of_response2.extend(list_of_response)
#             labels.extend(batch['label'])
#             print(f"Accuracy of the {args.dataset} dataset: {acc:.4f} (Correct: {correct}, Total: {all})")
#         result = str(acc)
#         if 'mcc' in metrics:
#             mcc = matthews_corrcoef(y_true=labels, y_pred=list_of_response2)
#             result = result + " " +str(mcc)
#         if 'f1_score' in metrics:
#             f1 = f1_score(y_true=labels, y_pred=list_of_response2, pos_label=positive_label)
#             result = result + " " +str(f1)
#         if 'pearson_correlation' in metrics:
#             labels = [float(item) for item in labels]
#             list_of_response2 = [float(item) for item in list_of_response2]
#             pearson = pearsonr(labels, list_of_response2)
#             result = result + " " +str(pearson)
#         write_to_file(index, result)

def batch_eva_write_to_excel(num_communication_rounds, args_passed=None, write_to_excel=True, metrics='accurcay', positive_label=None):
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
        cleaned_list_of_response2 = []
        labels = []
        for batch in tqdm(dataloader, desc="Evaluating"):
            full_prompt_list, full_response_list, list_of_response = evaluator.batch_run(batch)
            cleaned_list_of_response = cleansed_response(list_of_response)
            cleaned_list_of_response2.extend(cleaned_list_of_response)
            full_prompt_list2.extend(full_prompt_list)
            full_response_list2.extend(full_response_list)
            list_of_response2.extend(list_of_response)
            labels.extend(batch['label'])
            for pred, label in zip(cleaned_list_of_response, batch['label']):
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
        save_excel['cleaned_response'] = cleaned_list_of_response2
        save_excel['response'] = list_of_response2
        save_excel['label'] = labels
        save_excel['match'] = match_list
        save_excel['accuracy'] = [acc] * len(match_list)
        short_result = str(acc)
        if 'mcc' in metrics:
            mcc = matthews_corrcoef(y_true=labels, y_pred=list_of_response2)
            save_excel['mcc'] = [mcc] * len(match_list)
            short_result = short_result + " " +str(mcc)
        if 'f1_score' in metrics:
            f1 = f1_score(y_true=labels, y_pred=list_of_response2, pos_label=positive_label)
            save_excel['f1_score'] = [f1] * len(match_list)
            short_result = short_result + " " +str(f1)
        if 'pearson_correlation' in metrics:
            labels = [float(item) for item in labels]
            list_of_response2 = [float(item) for item in list_of_response2]
            pearson = pearsonr(labels, list_of_response2)
            save_excel['pearson_correlation'] = [pearson] * len(match_list)
            short_result = short_result + " " +str(pearson)
        directory = evaluator.save_path[args.dataset]
        short_result_file_name = os.path.join(directory, "short_result.txt")
        write_to_file(index, short_result, file_name=short_result_file_name)
        if write_to_excel:
            if not os.path.exists(directory):
                os.makedirs(directory)
            file_name = os.path.join(directory, str(index) + '.xlsx')
            save_excel.to_excel(file_name, index=False)

if __name__ == "__main__":
    args = parse_train_args()
    # batch_evaluate(args.num_communication_rounds, args, metrics='accuracy, mcc')
    batch_eva_write_to_excel(args.num_communication_rounds, args, metrics='mcc')

