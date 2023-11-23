import argparse
from typing import List

def parse_train_args():
    parser = argparse.ArgumentParser(description="Federated Learning Fine-Tuning for LLM-LoRA")

    parser.add_argument('--dataset', type=str, default='rte', help='Dataset to use')
    parser.add_argument('--global_model', type=str, default='./alpaca_native', help='Path to the global model')
    parser.add_argument('--data_path', type=str, default='./data_download/GLUE/rte/RTE', help='Data path')
    parser.add_argument('--output_dir', type=str, default='./lora-shepherd-7b/', help='Output directory')
    parser.add_argument('--client_selection_strategy', type=str, default='random', help='Client selection strategy')
    parser.add_argument('--client_selection_frac', type=float, default=0.1, help='Fraction of clients to select')
    parser.add_argument('--num_communication_rounds', type=int, default=10, help='Number of communication rounds')
    parser.add_argument('--num_clients', type=int, default=10, help='Number of clients')
    parser.add_argument('--partition_method', type=str, default="dirichlet_quantity", help='The method used to partition the data')
    parser.add_argument('--local_batch_size', type=int, default=64, help='Local batch size')
    parser.add_argument('--local_micro_batch_size', type=int, default=32, help='Local micro batch size')
    parser.add_argument('--local_num_epochs', type=int, default=2, help='Local number of epochs')
    parser.add_argument('--local_learning_rate', type=float, default=3e-4, help='Local learning rate')
    parser.add_argument('--local_val_set_size', type=int, default=0, help='Local validation set size')
    parser.add_argument('--local_save_steps', type=int, default=3, help='Local save steps')
    parser.add_argument('--cutoff_len', type=int, default=512, help='Cutoff length')
    parser.add_argument('--lora_r', type=int, default=8, help='LoRA r parameter')
    parser.add_argument('--dirichlet_alpha', type=int, default=3, help='dirichlet alpha parameter')
    parser.add_argument('--lora_alpha', type=int, default=16, help='LoRA alpha parameter')
    parser.add_argument('--lora_dropout', type=float, default=0.05, help='LoRA dropout rate')
    parser.add_argument('--lora_target_modules', nargs='+', default=["q_proj", "k_proj", "v_proj", "o_proj"], help='LoRA target modules')
    
    parser.add_argument('--train_on_inputs', type=bool, default=False, help='Train on inputs')
    parser.add_argument('--group_by_length', type=bool, default=False, help='Group by length')
    parser.add_argument('--resume_from_checkpoint', type=str, default=None, help='Resume from checkpoint')
    parser.add_argument('--prompt_template_name', type=str, default="alpaca", help='Prompt template name')

    args = parser.parse_args()
    return args


def parse_eval_args():
    parser = argparse.ArgumentParser(description="FederatedGPT-shepherd")
    parser.add_argument('--dataset', type=str, default='sts-b', help='Dataset to evaluate')
    parser.add_argument("--be_trained", type=bool, default=False, help="Share gradio interface")        # 修改成true后，才能加载lora模型
    parser.add_argument("--load_8bit", type=bool, default=False, help="Load model in 8-bit")
    parser.add_argument("--base_model", type=str, default="./alpaca_native", help="Base model path")
    parser.add_argument("--lora_weights_path", type=str, default="./lora-shepherd-7b/10/9/adapter_model.bin", help="LoRA weights path")
    parser.add_argument("--lora_config_path", type=str, default="./lora-shepherd-7b/10", help="LoRA config path")
    parser.add_argument("--prompt_template", type=str, default="", help="Prompt template")
    parser.add_argument("--server_name", type=str, default="127.0.0.1", help="Server name")
    parser.add_argument("--share_gradio", type=bool, default=False, help="Share gradio interface")
    return parser.parse_args()