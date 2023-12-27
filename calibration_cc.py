# 在FedEKit/fed_utils/evaluation.py基础上改写
import os
import sys
sys.path.append("./")
import torch
from parse import parse_eval_args, parse_args
import pandas as pd
from tqdm import tqdm
from datasets import load_dataset
from torch.utils.data import DataLoader
from model_utils.get_model import ModelHelper
from model_utils import PeftHelper
from sklearn.metrics import matthews_corrcoef, f1_score
from scipy.stats import pearsonr
import torch.nn.functional as F
from functools import partial
import re
import numpy as np

# offline
os.environ['HF_DATASETS_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
from peft import (
    prepare_model_for_kbit_training,
    set_peft_model_state_dict,
)

from transformers import GenerationConfig
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



def cleansed_response_for_acceptability(pred):
    pred = [item.lower() for item in pred]
    pred = [item[0:12] for item in pred]
    for index, item in enumerate(pred):
        if item[0:10] == 'acceptable':
            pred[index] = 'acceptable'
    return pred
def cleansed_response_for_quail(pred):
    pred = [item[0] if len(item) > 0 else '4' for item in pred]
    return pred
cleansed_response_methods = {
    'cola': cleansed_response_for_acceptability,
    'quail': cleansed_response_for_quail,
}


class Evaluator():
    def __init__(self, args):
        self.args = args
        self.prompter = None
        self.tokenizer = None
        self.model = None
        self.testset_path =  {
            "quail": "./data_download/quail/dev.json",
            "new-databricks-dolly-15k": './data_download/databricks_dolly_15k/data/10/global_test.json',
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
            "quail": "./output/quail",
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
        self.output_directory = self.save_path[args.dataset]
        if not os.path.exists(self.output_directory):
            os.makedirs(self.output_directory)
        self.output_short_result_file_name = os.path.join(self.output_directory, "short_result.txt")
        
    def model_init(self):
        args = self.args
        base_model = args.global_model
        assert (base_model), "Please specify a --base_model, e.g. --base_model='huggyllama/llama-7b'"

        self.prompter = Prompter(args.prompt_template_name)
        # set legacy=True means use the previous version, to fix the user warning
        model_helper = ModelHelper(global_model_name=args.model, global_model_path=args.global_model, device_map='auto')
        model, self.tokenizer = model_helper.get_model()
        model = prepare_model_for_kbit_training(model)
        if args.be_trained:         # peft微调过
            peft_helper = PeftHelper(model_name=args.model, peft_method=args.peft_method)
            model = peft_helper.get_peft_model_for_inference(model=model, config_path=args.peft_config_path, weight_path=args.peft_weights_path)

        model.eval()
        self.model = model

    def reset_peft_adapter(self, peft_weight_path):
        if self.args.be_trained:
            peft_weights = torch.load(peft_weight_path)
            set_peft_model_state_dict(self.model, peft_weights,"default")
            self.model.eval()
            del peft_weights
    
    def batch_run(self, batch_input):
        tokenized_inputs = self.tokenizer(batch_input['full_prompt'], padding='max_length', max_length=self.args.cutoff_len, return_tensors="pt")
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
    

    def generate_prompt(self, data_point):
        full_prompt = self.prompter.generate_prompt(
            data_point["instruction"],
            data_point["context"],
        )
        data_dict = {
            "full_prompt": full_prompt,
            "label": data_point['response']
        }
        return data_dict
    
    def write_to_file(self, index, result):
        with open(self.output_short_result_file_name, 'a') as f:
            f.write(str(index) + " " + str(result) + '\n')

class Calibrator():
    def __init__(self, args, method, tokenizer, model, prompter, dataloader):
        self.args = args
        self.method = method
        self.tokenizer = tokenizer
        self.model = model
        self.prompter = prompter
        self.dataloader = dataloader

    def run(self):
        if self.method == "cc":     # contextual calibration [Calibrate Before Use: Improving Few-Shot Performance of Language Models]
            prob_mean = self.estimate_bias_cc(self.dataloader)
            return partial(self.batch_run_cc, prob_mean=prob_mean) 
        elif self.method == "dc":
            prob_mean = self.estimate_bias_dc(self.dataloader)
            return partial(self.batch_run_dc, prob_mean=prob_mean)      
    
    def batch_run_cc(self, batch_input, prob_mean):
        tokenized_inputs = self.tokenizer(batch_input['full_prompt'], padding='max_length', max_length=self.args.cutoff_len, return_tensors="pt")
        tokenized_inputs = tokenized_inputs.to(device)
        generation_config = GenerationConfig(
            max_new_tokens=10,
        )
        
        # 设定生成的最大长度，会不会不同数据集不太一样？
        max_length = 2 + self.args.cutoff_len
        # 假设 tokenized_inputs 是您的输入数据
        input_ids = tokenized_inputs['input_ids'].to(device)
        attention_mask = tokenized_inputs['attention_mask'].to(device)
        # 获取批量大小
        batch_size = input_ids.shape[0]
        # 初始化 W 为对角矩阵
        prob_mean_inverse = 1.0 / torch.clamp(prob_mean, min=1e-9)
        num_classes = len(self.tokenizer)  # 词汇表大小
        W = torch.diag(prob_mean_inverse).to(device)
        # 找到 'accept' 和 'un' 的索引
        accept_index = self.tokenizer.convert_tokens_to_ids('accept')
        un_index = self.tokenizer.convert_tokens_to_ids('un')
        # 将 W 的对角线上除了 'accept' 和 'un' 索引外的所有元素设为 0
        for i in range(num_classes):
            if i != accept_index and i != un_index:
                W[i, i] = 0.0
        # 模型生成 token 的循环len(self.tokenizer)
        self.model.eval()
        with torch.no_grad():
            first_step = True
            while True:
                # 预测下一个 token 的分数（logits）
                outputs = self.model(input_ids, attention_mask=attention_mask)
                next_token_logits = outputs.logits[:, -1, :]
                next_token_logits = F.softmax(next_token_logits, dim=-1)
                # 只对首个 logits 应用变换
                if first_step:
                    next_token_logits = torch.matmul(next_token_logits, W)
                    first_step = False 
                # 应用 softmax 获取概率分布
                probs = F.softmax(next_token_logits, dim=-1)
                # 选择概率最高的 token
                next_tokens = torch.argmax(probs, dim=-1)
                # 添加新 token 到输入序列
                input_ids = torch.cat([input_ids, next_tokens.unsqueeze(-1)], dim=-1)
                # 更新 attention mask
                attention_mask = torch.cat([attention_mask.to(device), torch.ones((batch_size, 1), dtype=torch.long, device=device)], dim=1)
                # 检查是否所有样本都生成了结束 token 或达到最大长度
                if torch.all(next_tokens == self.tokenizer.eos_token_id) or input_ids.shape[1] > max_length:
                    break
        # 解码生成的文本
        response = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in input_ids]
        
        list_of_response = [self.prompter.get_response(res) for res in response]
        return batch_input['full_prompt'], response, list_of_response 
    
    
    # percent: 用来估计bias的数据比例
    def estimate_bias_cc(self, dataloader, percent=0.1):
        logits_list = []
        total_batches = len(dataloader)
        batches_to_process = int(percent * total_batches)  # 计算要处理的批次数（默认10%）
        for i, batch_input in enumerate(tqdm(dataloader, desc="Evaluating", total=batches_to_process)):
            if i >= batches_to_process:
                break  
            tokenized_inputs = self.tokenizer(self.replace_input_with_NA_in_batch(batch_input['full_prompt'], content="N/A"), padding='max_length', max_length=self.args.cutoff_len, return_tensors="pt")
            tokenized_inputs = tokenized_inputs.to(device)
            generation_config = GenerationConfig(
                max_new_tokens=10,
            )
            
            # 假设 tokenized_inputs 是输入数据
            input_ids = tokenized_inputs['input_ids'].to(device)
            attention_mask = tokenized_inputs['attention_mask'].to(device)

            # 模型生成 token 的循环len(self.tokenizer)
            self.model.eval()
            with torch.no_grad():
                # 预测下一个 token 的分数（logits）
                outputs = self.model(input_ids, attention_mask=attention_mask)
                next_token_logits = outputs.logits[:, -1, :].to(device)
                next_token_logits = F.softmax(next_token_logits, dim=-1)
                logits_list.append(next_token_logits)
    
        return torch.cat(logits_list, dim=0).mean(dim=0)

    def replace_input_with_NA_in_batch(self, batch_prompts, content):
        modified_prompts = []
        for prompt in batch_prompts:
            modified_prompt = re.sub(r'### Input:\n.*?\n', '### Input:\n' + content + '\n', prompt)
            modified_prompts.append(modified_prompt)
        return modified_prompts
    

    # DC 的标签偏差估计
    def estimate_bias_dc(self, dataloader, percent=0.1):
        """
        Estimate label bias using Domain-context Calibration (DC).
        """
        testset = load_dataset("json", data_files=evaluator.testset_path[args.dataset])
        cols = ['instruction', 'response', 'context', 'category']
        cleared_testset = testset["train"].shuffle().map(evaluator.generate_prompt, remove_columns=cols)
        cleared_testset.set_format(type="torch", columns=["full_prompt", "label"])
        dataloader = DataLoader(cleared_testset, batch_size=args.dataloader_bs, drop_last=False)
        
        np.random.seed(seed)
        content_free_inputs = sample_random_texts(texts=test_sentences, n_sample=num_samples, seed=seed)

        all_p_y = []
        for content_free_input in content_free_inputs:
            prompt = construct_prompt(params, [], [], content_free_input)
            resp = complete(prompt, 1, params['model'], num_log_probs=params['api_num_log_prob'])
            p_y = [0] * len(params['label_dict'])
            for i, answers in params['label_dict'].items():
                prob = 0
                for a in answers:
                    resp = complete(prompt + " " + a, 0, params['model'], echo=True, num_log_probs=1)
                    prob += np.exp(resp['choices'][0]['logprobs']['token_logprobs'][-1])
                p_y[i] = prob
            all_p_y.append(p_y)

        p_y = np.mean(np.array(all_p_y), axis=0)
        p_y = p_y / np.sum(p_y) # normalize
        return p_y

    # 使用 DC 方法的 batch_run
    def batch_run_dc(self, batch_input, prob_mean):
        tokenized_inputs = self.tokenizer(batch_input['full_prompt'], padding='max_length', max_length=self.args.cutoff_len, return_tensors="pt")
        tokenized_inputs = tokenized_inputs.to(device)
        generation_config = GenerationConfig(
            max_new_tokens=10,
        )
        
        # 设定生成的最大长度，会不会不同数据集不太一样？
        max_length = 2 + self.args.cutoff_len
        # 假设 tokenized_inputs 是您的输入数据
        input_ids = tokenized_inputs['input_ids'].to(device)
        attention_mask = tokenized_inputs['attention_mask'].to(device)
        # 获取批量大小
        batch_size = input_ids.shape[0]
        # 初始化 W 为对角矩阵
        prob_mean_inverse = 1.0 / torch.clamp(prob_mean, min=1e-9)
        num_classes = len(self.tokenizer)  # 词汇表大小
        W = torch.diag(prob_mean_inverse).to(device)
        # 找到 'accept' 和 'un' 的索引
        accept_index = self.tokenizer.convert_tokens_to_ids('accept')
        un_index = self.tokenizer.convert_tokens_to_ids('un')
        # 将 W 的对角线上除了 'accept' 和 'un' 索引外的所有元素设为 0
        for i in range(num_classes):
            if i != accept_index and i != un_index:
                W[i, i] = 0.0
        # 模型生成 token 的循环len(self.tokenizer)
        self.model.eval()
        with torch.no_grad():
            first_step = True
            while True:
                # 预测下一个 token 的分数（logits）
                outputs = self.model(input_ids, attention_mask=attention_mask)
                next_token_logits = outputs.logits[:, -1, :]
                next_token_logits = F.softmax(next_token_logits, dim=-1)
                # 只对首个 logits 应用变换
                if first_step:
                    next_token_logits = torch.matmul(next_token_logits, W)
                    first_step = False 
                # 应用 softmax 获取概率分布
                probs = F.softmax(next_token_logits, dim=-1)
                # 选择概率最高的 token
                next_tokens = torch.argmax(probs, dim=-1)
                # 添加新 token 到输入序列
                input_ids = torch.cat([input_ids, next_tokens.unsqueeze(-1)], dim=-1)
                # 更新 attention mask
                attention_mask = torch.cat([attention_mask.to(device), torch.ones((batch_size, 1), dtype=torch.long, device=device)], dim=1)
                # 检查是否所有样本都生成了结束 token 或达到最大长度
                if torch.all(next_tokens == self.tokenizer.eos_token_id) or input_ids.shape[1] > max_length:
                    break
        # 解码生成的文本
        response = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in input_ids]
        
        list_of_response = [self.prompter.get_response(res) for res in response]
        return batch_input['full_prompt'], response, list_of_response 



def batch_eva_write_to_excel(num_communication_rounds, args, write_to_excel=True, metrics='accurcay', positive_label=None, use_trained_model=True):
    args.peft_config_path = args.output_dir
    args.peft_weights_path = args.peft_config_path + '/0/adapter_model.bin'
    args.dataloader_bs = args.local_micro_batch_size
    args.be_trained = use_trained_model
    # if args_passed:
        # args.model = args_passed.model
        # args.peft_method = args_passed.peft_method
        # args.dataset = args_passed.dataset
        # args.peft_config_path = args_passed.output_dir
        # args.peft_weights_path = args.peft_config_path + '/0/adapter_model.bin'
        # args.base_model = args_passed.global_model
        # args.cutoff_len = args_passed.cutoff_len
        # args.dataloader_bs = args_passed.local_micro_batch_size
    
    evaluator = Evaluator(args)
    evaluator.model_init()
    testset = load_dataset("json", data_files=evaluator.testset_path[args.dataset])
    cols = ['instruction', 'response', 'context', 'category']
    cleared_testset = testset["train"].shuffle().map(evaluator.generate_prompt, remove_columns=cols)
    cleared_testset.set_format(type="torch", columns=["full_prompt", "label"])
    dataloader = DataLoader(cleared_testset, batch_size=args.dataloader_bs, drop_last=False)
    
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

        # 做个矫正
        cali_method = "dc"  # 或 "dc"
        calibrator = Calibrator(args, cali_method, evaluator.tokenizer, evaluator.model, evaluator.prompter, dataloader)
        cali_batch_run = calibrator.run()
        
        # batch_id = 0

        for batch in tqdm(dataloader, desc="Evaluating"):
            full_prompt_list, full_response_list, list_of_response = cali_batch_run(batch)      
            # full_prompt_list, full_response_list, list_of_response = evaluator.batch_run(batch)      
            cleaned_list_of_response = cleansed_response_methods[args.dataset](list_of_response)
            cleaned_list_of_response2.extend(cleaned_list_of_response)
            full_prompt_list2.extend(full_prompt_list)
            full_response_list2.extend(full_response_list)
            list_of_response2.extend(list_of_response)
            labels.extend(batch['label'])
            for pred, label in zip(cleaned_list_of_response, batch['label']):
                if(pred.lower() == label.lower()):
                    correct += 1
                    match_list.extend([1])
                else:
                    match_list.extend([0])
            all += len(batch['label'])
            acc = correct / all
            print(f"Accuracy of the {args.dataset} dataset: {acc:.4f} (Correct: {correct}, Total: {all})")
            
            # batch_id += 1
            # if(batch_id == 100):
            #     break

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
        evaluator.write_to_file(index=index, result=short_result)
        if write_to_excel:
            file_name = os.path.join(evaluator.output_directory, str(index) + '.xlsx')
            save_excel.to_excel(file_name, index=False)

if __name__ == "__main__":
    args = parse_args()
    batch_eva_write_to_excel(args.num_communication_rounds, args)