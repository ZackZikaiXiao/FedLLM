import os
from parse import parse_train_args
from typing import List
from tqdm import tqdm
import torch
from parse import parse_eval_args
from evaluate import write_to_file, Evaluator
from transformers import LlamaTokenizer, LlamaForCausalLM
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_int8_training,
    prepare_model_for_kbit_training,
    set_peft_model_state_dict,
)
import time
import datetime
from fed_utils import FedAvg, client_selection, global_evaluation, GenerateClient
from data_tool.data_partition import DataPartition
from data_tool.data_tokenizer import DataTokenizer

# offline
os.environ['HF_DATASETS_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'


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

    model = LlamaForCausalLM.from_pretrained(
        args.global_model,
        load_in_8bit=True,     # True
        torch_dtype=torch.float16,
        device_map=device_map,
    )

    tokenizer = LlamaTokenizer.from_pretrained(args.global_model, use_fast=False)
    # 但是这里0 decode出来是<unk>
    tokenizer.pad_token_id = (0)
    tokenizer.padding_side = "left"

    
    data_tokenizer = DataTokenizer(args, tokenizer)

    model = prepare_model_for_kbit_training(model)
    # model = prepare_model_for_int8_training(model)
    config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=args.lora_target_modules,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, config)
    if not ddp and torch.cuda.device_count() > 1:
        model.is_parallelizable = True
        model.model_parallel = True
    model.print_trainable_parameters()
    # if you want to resume training from checkpoint
    # set these parameters
    training_from_checkpoint=False
    if(training_from_checkpoint):
        parameter_path = './lora-shepherd-7b/cola iid 0.1 10/4/adapter_model.bin'
        lora_weights = torch.load(parameter_path)
        set_peft_model_state_dict(model, lora_weights,"default")
        

    print("The process of federated instruction-tuning has started..")
    previously_selected_clients_set = set()
    last_client_id = None
    local_dataset_len_dict = dict()
    output_dir = os.path.join(args.output_dir, args.dataset +" "+ args.partition_method + " "  + str(args.dirichlet_alpha) + " " + str(args.num_clients))
    training_start_time = time.time()
    for epoch in tqdm(range(args.num_communication_rounds)):

        print("\nConducting the client selection")
        selected_clients_set = client_selection(args.num_clients, args.client_selection_frac, args.client_selection_strategy,
                                                other_info=epoch)
        # lr可以在每个num_communication之后减小
        # 在iid的设置下可能效果会更好
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
        torch.save(model.state_dict(), os.path.join(output_dir, str(epoch), "adapter_model.bin"))
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
    args = parse_eval_args()
    auto_testing = True
    if auto_testing:
        num_communication_rounds = 20
        for index in range(num_communication_rounds):
            args.lora_weights_path = os.path.join(args.lora_config_path, str(index), "adapter_model.bin")
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
            
                write_to_file(index, acc)



if __name__ == "__main__":
    args = parse_train_args()
    data_partition = DataPartition(args)
    data_partition.partition()      # 生成
    main(args)
