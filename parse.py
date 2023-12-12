import argparse
from typing import List
import os
def parse_train_args():
    global_model_path = {
        'alpaca': './alpaca_native',
        'Llama2-7B': '/home/jianhuiwei/rsch/jianhui/Llama2-7b-chat/',
    }
    data_paths = {
        'cola': './data_download/GLUE/cola/CoLA',
        'mnli': './data_download/GLUE/mnli/MNLI',
        'mrpc': './data_download/GLUE/mrpc/MRPC',
        'qnli': './data_download/GLUE/qnli/QNLI',
        'qqp': './data_download/GLUE/qqp/QQP',
        'rte': './data_download/GLUE/rte/RTE',
        'sst-2':'./data_download/GLUE/sst-2/SST-2',
        'sts-b': './data_download/GLUE/sts-b/STS-B',
        'wnli': './data_download/GLUE/wnli/WNLI',
    }
    output_dirs = {
        'alpaca':{
            'lora': './lora-shepherd-7b/',
            'prefix_tuning': './alpaca-prefix/',
        },
        'Llama2-7B':{
            'lora': './llama2-lora/',
            'prefix_tuning': './llama2-prefix/'
        }
    }
    parser = argparse.ArgumentParser(description="Federated Learning PEFine-Tuning for LLM")
    parser.add_argument('--model', type=str, default='alpaca', help='which pretrained model to use, now support Llama2-7B and alpaca')
    # parser.add_argument('--global_model', type=str, default='./alpaca_native', help='Path to the global model, /home/jianhuiwei/rsch/jianhui/Llama2-7b-chat/ or ./alpaca_native')
    parser.add_argument('--peft_method', type=str, default='prefix_tuning', help='which peft method to use, now support lora and prefix_tuning')
    # parameters for lora adapter
    parser.add_argument('--lora_r', type=int, default=8, help='LoRA r parameter')
    parser.add_argument('--lora_alpha', type=int, default=16, help='LoRA alpha parameter')
    parser.add_argument('--lora_dropout', type=float, default=0.05, help='LoRA dropout rate')
    parser.add_argument('--lora_target_modules', nargs='+', default=["q_proj", "k_proj", "v_proj", "o_proj"], help='LoRA target modules')
    # parameters for prefix_tuning
    parser.add_argument('--num_virtual_tokens', type=int, default=10, help='num of virtual tokens for prefix tuning')
    # if you want to change the dataset to train, please change the arguments here
    parser.add_argument('--dataset', type=str, default='cola', help='Dataset to use')
    # parser.add_argument('--data_path', type=str, default='./data_download/GLUE/cola/CoLA', help='Data path')
    parser.add_argument('--dirichlet_alpha', type=int, default=1, help='dirichlet alpha parameter')
    parser.add_argument('--partition_method', type=str, default="iid", help='The method used to partition the data, choose from [''iid'', ''dirichlet_label_uni'', ''dirichlet_label'', ''dirichlet_quantity'']')

    # parser.add_argument('--output_dir', type=str, default='./lora-shepherd-7b/', help='Output directory, choices: lora-shepherd-7b, alpaca-prefix, llama2-lora, llama2-prefix')
    parser.add_argument('--client_selection_strategy', type=str, default='random', help='Client selection strategy')
    parser.add_argument('--client_selection_frac', type=float, default=0.4, help='Fraction of clients to select')
    parser.add_argument('--num_communication_rounds', type=int, default=20, help='Number of communication rounds')
    parser.add_argument('--num_clients', type=int, default=10, help='Number of clients')

    parser.add_argument('--local_batch_size', type=int, default=64, help='Local batch size')
    parser.add_argument('--local_micro_batch_size', type=int, default=32, help='Local micro batch size')
    parser.add_argument('--local_num_epochs', type=int, default=2, help='Local number of epochs')
    parser.add_argument('--local_learning_rate', type=float, default=3e-4, help='Local learning rate, 3e-3试过了, for alpaca-lora: 3e-4')
    parser.add_argument('--local_val_set_size', type=int, default=0, help='Local validation set size')
    parser.add_argument('--local_save_steps', type=int, default=3, help='Local save steps')

    parser.add_argument('--cutoff_len', type=int, default=512, help='Cutoff length')
    parser.add_argument('--train_on_inputs', type=bool, default=False, help='Train on inputs')
    parser.add_argument('--group_by_length', type=bool, default=False, help='Group by length')
    parser.add_argument('--resume_from_checkpoint', type=str, default=None, help='Resume from checkpoint')
    parser.add_argument('--prompt_template_name', type=str, default="alpaca", help='Prompt template name')
    args = parser.parse_args()
    args.global_model = global_model_path[args.model]
    if args.partition_method == 'iid':
        args.data_path = os.path.join(data_paths[args.dataset], str(args.num_clients) + "-" + args.partition_method)
    else:
        args.data_path = os.path.join(data_paths[args.dataset], str(args.num_clients) + "-" + args.partition_method +"-"+ str(args.dirichlet_alpha))
    # args.data_path = data_paths[args.dataset]
    args.output_dir = output_dirs[args.model][args.peft_method]
    if args.partition_method == 'iid':
        args.output_dir = os.path.join(args.output_dir, args.dataset +"-"+ args.partition_method + "-" + str(args.num_clients))
    else:
        args.output_dir = os.path.join(args.output_dir, args.dataset +"-"+ args.partition_method + "-"  + str(args.dirichlet_alpha) + "-" + str(args.num_clients))
    return args


def parse_eval_args():
    global_model_path = {
        'alpaca': './alpaca_native',
        'Llama2-7B': '/home/jianhuiwei/rsch/jianhui/Llama2-7b-chat/',
    }
    parser = argparse.ArgumentParser(description="FederatedGPT-shepherd")
    parser.add_argument('--model', type=str, default='alpaca', help='which pretrained model to use, now support Llama2-7B and alpaca')
    # parser.add_argument("--base_model", type=str, default="./alpaca_native", help="Base model path")
    parser.add_argument('--peft_method', type=str, default='lora', help='which peft method to use, now support lora and prefix_tuning')
    # if you want to change the evaluation dataset, please modify the arguments here
    parser.add_argument('--dataset', type=str, default='cola', help='Dataset to evaluate')
    parser.add_argument("--peft_weights_path", type=str, default="./lora-shepherd-7b/cola-iid-10/0/adapter_model.bin", help="peft weights path")
    parser.add_argument("--peft_config_path", type=str, default="./lora-shepherd-7b/cola-iid-10", help="peft config path")
    # if you want to change the evaluation dataset, please modify the arguments here
    parser.add_argument("--be_trained", type=bool, default=True, help="Share gradio interface")        # 修改成true后，才能加载lora模型
    parser.add_argument("--load_8bit", type=bool, default=False, help="Load model in 8-bit")
    
    parser.add_argument("--prompt_template_name", type=str, default="alpaca", help="Prompt template")
    parser.add_argument("--server_name", type=str, default="127.0.0.1", help="Server name")
    parser.add_argument("--share_gradio", type=bool, default=False, help="Share gradio interface")
    parser.add_argument('--cutoff_len', type=int, default=512, help='Cutoff length')
    args = parser.parse_args()
    args.base_model = global_model_path[args.model]
    return args


if __name__ == "__main__":
    args = parse_train_args()
    # args.global_model = 'something_else'
    print(args.data_path)