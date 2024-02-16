import argparse
import os

# hyperparameters selection
# model: alpaca, peft method: lora, dataset: cola
# lr=3e-4, lora_r=8, lora_alpha=16, lora_dropout=0.05, bs=64, micro_bs=32

# 采样的时候client的数据太少
# hyperparameter for num_clients=10:
# local_epoch=2, selection_fraction=0.4, batch_size=64


# hyperparameter for num_clients=100:
# local_epoch=2, selection_fraction=0.1, batch_size=32

def parse_args():
    GLUE_dataset =["sst-2", "rte", "cola", "qnli", "qqp", "sts-b", "wnli", "mrpc", "mnli"]
    global_model_path = {
        'alpaca': '/home/jianhuiwei/rsch/jianhui/alpaca_native',
        'Llama2-7B': '/home/jianhuiwei/rsch/jianhui/Llama2-7b-chat/',
    }
    data_paths = {
        "20news": "./data_download/20news",
        "quail": "./data_download/quail",
        "new-databricks-dolly-15k": './data_download/databricks_dolly_15k/data',
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
        # "/home/jianhuiwei/rsch/jianhui/checkpoints"
        'alpaca':{
            'lora': '/home/jianhuiwei/rsch/jianhui/checkpoints/lora-shepherd-7b/',
            'prefix_tuning': '/home/jianhuiwei/rsch/jianhui/checkpoints/alpaca-prefix/',
        },
        'Llama2-7B':{
            'lora': '/home/jianhuiwei/rsch/jianhui/checkpoints/llama2-lora/',
            'prefix_tuning': '/home/jianhuiwei/rsch/jianhui/checkpoints/llama2-prefix/'
        }
    }
    parser = argparse.ArgumentParser(description="Federated Learning PEFine-Tuning for LLM")
    parser.add_argument('--model', type=str, default='alpaca', help='which pretrained model to use, now support Llama2-7B and alpaca')
    parser.add_argument('--peft_method', type=str, default='lora', help='which peft method to use, now support lora and prefix_tuning')
    # parameters for lora adapter
    parser.add_argument('--lora_r', type=int, default=8, help='LoRA r parameter')
    parser.add_argument('--lora_alpha', type=int, default=16, help='LoRA alpha parameter')
    parser.add_argument('--lora_dropout', type=float, default=0.05, help='LoRA dropout rate')
    parser.add_argument('--lora_target_modules', nargs='+', default=["q_proj", "k_proj", "v_proj", "o_proj"], help='LoRA target modules')
    # parameters for prefix_tuning
    parser.add_argument('--num_virtual_tokens', type=int, default=5, help='num of virtual tokens for prefix tuning')
    # if you want to change the dataset to train, please change the arguments here
<<<<<<< HEAD
    parser.add_argument('--dataset', type=str, default='cola', help='Dataset to use')
    parser.add_argument('--dirichlet_alpha', type=int, default=1, help='dirichlet alpha parameter')
=======
    parser.add_argument('--dataset', type=str, default='rte', help='Dataset to use')
    parser.add_argument('--dirichlet_alpha', type=int, default=0.8, help='dirichlet alpha parameter, 1, 1.5, 2')
>>>>>>> 58431c210f539d5d12b0970e019e656930d30251
    parser.add_argument('--partition_method', type=str, default="dirichlet_label_uni", help='The method used to partition the data, choose from [''iid'', ''dirichlet_label_uni'', ''dirichlet_label'', ''dirichlet_quantity'']')
    parser.add_argument('--client_selection_strategy', type=str, default='random', help='Client selection strategy')
    parser.add_argument('--client_selection_frac', type=float, default=0.4, help='Fraction of clients to select')
    parser.add_argument('--num_communication_rounds', type=int, default=20, help='Number of communication rounds')
    parser.add_argument('--num_clients', type=int, default=10, help='Number of clients')
    # FedProx related arguments
    parser.add_argument('--useFedProx', type=bool, default=False, help='Whether or not add proximal term to the loss function')
    parser.add_argument('--proximal_term_argument', type=float, default=0.01, help='the mu for proximal term')
    # FedNova related arguments
    parser.add_argument('--useFedNova', type=bool, default=False, help='Whether or not use FedNova for aggregation')
    # Scaffold related arguments
    parser.add_argument('--useScaffold', type=bool, default=True, help='Whether or not use Scaffold')
    parser.add_argument('--scaffold_dir', type=str, default='/home/jianhuiwei/rsch/jianhui/scaffold_control_variate', help='the dir to save variate for server and client')

    parser.add_argument('--local_batch_size', type=int, default=64, help='Local batch size')
    parser.add_argument('--local_micro_batch_size', type=int, default=32, help='Local micro batch size, 16 for 20news,quail. 32 for GLUE')
    parser.add_argument('--local_num_epochs', type=int, default=2, help='Local number of epochs')
    parser.add_argument('--local_learning_rate', type=float, default=3e-3, help='Local learning rate, 3e-3试过了, for alpaca-lora: 3e-4')
    parser.add_argument('--local_val_set_size', type=int, default=0, help='Local validation set size')
    parser.add_argument('--local_save_steps', type=int, default=3, help='Local save steps')

<<<<<<< HEAD
    parser.add_argument('--cutoff_len', type=int, default=512, help='Cutoff length, 512 for GLUE, and 1024 for quail')
=======
    parser.add_argument('--cutoff_len', type=int, default=512, help='Cutoff length, 512 for GLUE, and 1024 for quail, 2048 for 20news ')
>>>>>>> 58431c210f539d5d12b0970e019e656930d30251
    parser.add_argument('--train_on_inputs', type=bool, default=False, help='Train on inputs')
    parser.add_argument('--group_by_length', type=bool, default=False, help='Group by length')
    parser.add_argument('--prompt_template_name', type=str, default="alpaca", help='Prompt template name')
    # the arguments below are for resume training from checkpoint
    parser.add_argument('--resume_from_checkpoint', type=bool, default=False, help='Resume from checkpoint')
    parser.add_argument('--parameter_path', type=str, default='/home/jianhuiwei/rsch/jianhui/checkpoints/lora-shepherd-7b/20news-dirichlet_label_uni-0.5-100/aggregated_model_14.bin', help='the parameter path for checkpoint')
    parser.add_argument('--start_round', type=int, default=15, help='the parameter path for checkpoint')
    args = parser.parse_args()
    args.global_model = global_model_path[args.model]
    if args.dataset in GLUE_dataset or args.dataset == "quail" or args.dataset == "20news":
        if args.partition_method == 'iid':
            args.data_path = os.path.join(data_paths[args.dataset], str(args.num_clients) + "-" + args.partition_method)
        else:
            args.data_path = os.path.join(data_paths[args.dataset], str(args.num_clients) + "-" + args.partition_method +"-"+ str(args.dirichlet_alpha))
    elif args.dataset == "new-databricks-dolly-15k":
        args.data_path = os.path.join(data_paths[args.dataset], str(args))
    # args.data_path = data_paths[args.dataset]
    args.output_dir = output_dirs[args.model][args.peft_method]
    if args.useFedProx:
        federated_method='FedProx'
    elif args.useFedNova:
        federated_method='FedNova'
    elif args.useScaffold:
        federated_method='Scaffold'
    else:
        federated_method='FedAvg'
    if args.partition_method == 'iid':
        args.output_dir = os.path.join(args.output_dir, args.dataset +"-"+ args.partition_method + "-" + str(args.num_clients) + "-" + federated_method)
    else:
        args.output_dir = os.path.join(args.output_dir, args.dataset +"-"+ args.partition_method + "-"  + str(args.dirichlet_alpha) + "-" + str(args.num_clients) + "-" + federated_method)
    return args

if __name__ == "__main__":
    args = parse_args()
    print(args.data_path)