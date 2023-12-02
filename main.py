import os
from parse import parse_train_args
from typing import List
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from datasets import load_dataset
from parse import parse_eval_args
from evaluate import write_to_file, Evaluator
from peft import (
    prepare_model_for_kbit_training,
    set_peft_model_state_dict,
    get_peft_model_state_dict,
)
from model_utils.get_peft_model import get_lora_peft_model, get_prefix_tuning_peft_model
import time
import datetime
from fed_utils import FedAvg, client_selection, global_evaluation, GenerateClient
from data_tool.data_partition import DataPartition
from data_tool.data_tokenizer import DataTokenizer
from model_utils.get_model import get_alpaca_model_and_tokenizer, get_llama27b_model_and_tokenizer
# offline
os.environ['HF_DATASETS_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'

def global_lr_scheduler(lr):
    return lr/2

def main(args):

    if int(os.environ.get("LOCAL_RANK", 0)) == 0:
        print(f"Federated Finetuning LLM-LoRA with params:\n")
        for arg in vars(args):
            print(f"{arg}: {getattr(args, arg)}")

    # 确保全局模型已经指定
    assert args.global_model, "Please specify a --global_model, e.g. --global_model='decapoda-research/llama-7b-hf'"

    data_path = os.path.join(args.data_path, str(args.num_clients))
    assert os.path.exists(args.data_path), "Please generate the data files for each client"

    # set up the global model & toknizer
    gradient_accumulation_steps = args.local_batch_size // args.local_micro_batch_size
    device_map = "auto"
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1
    if ddp:
        device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}
        gradient_accumulation_steps = gradient_accumulation_steps // world_size
    if args.model == 'alpaca':
        model, tokenizer = get_alpaca_model_and_tokenizer(global_model=args.global_model, device_map=device_map)
    elif args.model == 'Llama2-7B':
        model, tokenizer = get_llama27b_model_and_tokenizer(global_model=args.global_model, device_map=device_map)
    model = prepare_model_for_kbit_training(model)
    
    data_tokenizer = DataTokenizer(args, tokenizer)
    
    if args.peft_method == 'lora':
        model, config = get_lora_peft_model(args, model)
    elif args.peft_method == 'prefix_tuning':
        model, config = get_prefix_tuning_peft_model(args, model)
    model.print_trainable_parameters()
    if not ddp and torch.cuda.device_count() > 1:
        model.is_parallelizable = True
        model.model_parallel = True
    
    # if you want to resume training from checkpoint
    # set these parameters
    training_from_checkpoint=False
    if(training_from_checkpoint):
        parameter_path = './lora-shepherd-7b/cola iid 0.1 10/4/adapter_model.bin'
        peft_weights = torch.load(parameter_path)
        set_peft_model_state_dict(model, peft_weights,"default")
        

    print("The process of federated instruction-tuning has started..")
    previously_selected_clients_set = set()
    last_client_id = None
    local_dataset_len_dict = dict()
    if args.partition_method == 'iid':
        output_dir = os.path.join(args.output_dir, args.dataset +" "+ args.partition_method + " " + str(args.num_clients))
    else:
        output_dir = os.path.join(args.output_dir, args.dataset +" "+ args.partition_method + " "  + str(args.dirichlet_alpha) + " " + str(args.num_clients))
    
    training_start_time = time.time()
    for epoch in tqdm(range(args.num_communication_rounds)):

        print("\nConducting the client selection")
        selected_clients_set = client_selection(args.num_clients, args.client_selection_frac, args.client_selection_strategy,
                                                other_info=epoch)
        # lr可以在每个num_communication之后减小
        # 在iid的设置下可能效果会更好
        args.local_learning_rate = global_lr_scheduler(args.local_learning_rate)
        for client_id in selected_clients_set:
            client = GenerateClient(args, client_id, model, output_dir)

            print("\nPreparing the local dataset and trainer for Client_{}".format(client_id))
            client.load_raw_load(dataset=args.dataset)
            client.preprare_local_dataset(data_tokenizer.generate_and_tokenize_prompt, args.local_val_set_size)
            client.build_local_trainer(tokenizer,
                                       args.local_micro_batch_size,
                                       gradient_accumulation_steps,
                                       args.local_num_epochs,
                                       args.local_learning_rate,
                                       args.group_by_length,
                                       ddp)

            print("Initiating the local training of Client_{}".format(client_id))
            client.initiate_local_training()

            print("Local training starts ... ")
            client.train()

            print("\nTerminating the local training of Client_{}".format(client_id))
            model, local_dataset_len_dict, previously_selected_clients_set, last_client_id = client.terminate_local_training(
                epoch, local_dataset_len_dict, previously_selected_clients_set)
            del client

        print("Collecting the weights of clients and performing aggregation")
        model = FedAvg(model,
                       selected_clients_set,
                       output_dir,
                       local_dataset_len_dict,
                       epoch,
                       )
        # 修改了一下模型保存地址
        torch.save(get_peft_model_state_dict(model), os.path.join(output_dir, str(epoch), "adapter_model.bin"))
        config.save_pretrained(output_dir)

        # Please design the evaluation method based on your specific requirements in the fed_utils/evaluation.py file.
        global_evaluation()
    training_over_time = time.time()
    training_time = int(round((training_over_time - training_start_time)))
    print("Total training time: " + str(datetime.timedelta(seconds = training_time)))






    # testing phase
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
    args2 = parse_eval_args()
    auto_testing = True
    if auto_testing:
        num_communication_rounds = args.num_communication_rounds
        evaluator = Evaluator(args2)
        evaluator.model_init()
        testset = load_dataset("json", data_files=testset_path[args2.dataset])
        data_tokenizer = DataTokenizer(args2, evaluator.tokenizer)
        cols = ['instruction', 'response', 'context', 'category']
        cleared_testset = testset["train"].shuffle().map(evaluator.generate_prompt, remove_columns=cols)
        cleared_testset.set_format(type="torch", columns=["full_prompt", "label"])
        dataloader = DataLoader(cleared_testset, batch_size=64, drop_last=False)
        
        for index in range(num_communication_rounds):
            peft_weights_path = os.path.join(args2.peft_config_path, str(index), "adapter_model.bin")
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
                print(f"Accuracy of the {args2.dataset} dataset: {acc:.4f} (Correct: {correct}, Total: {all})")
            write_to_file(index, acc)



if __name__ == "__main__":
    args = parse_train_args()
    data_partition = DataPartition(args)
    data_partition.partition()      # 生成
    main(args)
