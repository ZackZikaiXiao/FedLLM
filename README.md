
<h4 align="center"><em><span style="font-size:20pt">  Fed-EKit </span></em></h4>
<h4 align="center"><em><span style="font-size:15pt">  Federated Easy-to-Use Large Language Model Kit with Efficient Fine-Tuning </span></em></h4>


# Fed-EKit Overview

Fed-EKit aims to provide researchers with a comprehensive solution for fine-tuning Large Language Models (LLMs) in a Federated Learning environment. This project focuses on combining ease of use with flexibility, supporting a wide range of datasets, models, performance enhancement methods, and evaluation approaches.

## Core Features

### Federated Learning Support
- **Fed-EKit** is specifically designed for federated learning environments, allowing users to collaboratively train and fine-tune large language models while protecting data privacy.

### Ease of Use
- The project offers a set of intuitive tools and interfaces, making it accessible for both beginners and experienced developers. With simplified installation and configuration processes, users can quickly start their projects.

### Large Language Model (LLM) Integration
- Integrates a variety of the latest large language models, offering users a broad selection to suit different application scenarios and requirements.

### Parameter Efficient Fine-Tuning
- Utilizes advanced fine-tuning techniques to optimize model performance while reducing the need for computational resources, making the models more efficient.

### Flexibility and Customization
- Supports various datasets, model structures, and evaluation methods, allowing users to customize and adjust according to their specific needs.

### Community-Driven Open Source
- As an open-source project, **Fed-EKit** encourages community participation, thereby continuously improving and expanding its functionalities.


## Data_Preparation

Prior to commencing the federated fine-tuning, make sure to create a data file for each individual client.
```bash
num_client=10 # The number of clients
diff_quantity=0 # Whether clients have different amounts of data
python client_data_allocation.py $num_client $diff_quantity
```
Running this command will save the data files in the folder `./data/str(num_client)`. The data file `new-databricks-dolly-15k.json` for generating each client's local dataset is the first version of `databricks-dolly-15k` , which is a corpus of more than 15,000 records with 8 categeries generated by thousands of [Databricks Lab](https://www.databricks.com/learn/labs) employees. Please refer to their official repository [dolly](https://github.com/databrickslabs/dolly) for the latest version of data.


### Use your own data

You can simply modify `client_data_allocation.py` to load your own  dataset for federated training.


## Federated_Finetuning

To fully leverage the computational resources of each participating client, our lightweight Federated Learning framework employs the well-established parameter-efficient method, [LoRA](https://github.com/microsoft/LoRA), for conducting local training. The local training process is built upon the implementations of Hugging Face's [PEFT](https://github.com/huggingface/peft), Tim Dettmers' [bitsandbytes](https://github.com/TimDettmers/bitsandbytes), and the [Alpaca-lora](https://github.com/tloen/alpaca-lora), enabling the training to be completed within hours on a single NVIDIA TITAN RTX.

Example usage:
```bash
python main.py --global_model 'chavinlo/alpaca-native'\
      --data_path  "./data" \
      --output_dir  './lora-shepherd-7b/'\
      --num_communication_rounds 10 \
      --num_clients  10 \
      --train_on_inputs \
      --group_by_length
```
Within the `main.py` file, the GeneralClient is a Python class serves as a representation of the local client and encompasses five distinct sections that facilitate local training: "prepare_local_dataset," "build_local_trainer," "initiate_local_training," "train," and "terminate_local_training." Each of these sections is easy to comprehend and can be easily customized by adding your own functions to meet specific requirements.

We can also tweak the hyperparameters:
```bash
python main.py --global_model 'chavinlo/alpaca-native'\
      --data_path  "./data" \
      --output_dir  './lora-shepherd-7b/'\
      --num_communication_rounds 10 \
      --num_clients  10 \
      --client_selection_frac 0.1 \
      --local_num_epochs  2 \
      --local_batch_size  64 \
      --local_micro_batch_size 32 \
      --local_learning_rate 0.0003 \
      --lora_r 8 \
      --lora_target_modules='[q_proj,k_proj,v_proj,o_proj]' \
      --train_on_inputs \
      --group_by_length
```

Our framework supports numerous popular LLMs, such as [LLaMA](https://github.com/facebookresearch/llama), [Alpaca](https://github.com/tatsu-lab/stanford_alpaca), [Vicuna](https://vicuna.lmsys.org/), [Baize](https://github.com/project-baize/baize-chatbot), and others. We welcome any pull requests that adapt our code to support additional models or datasets.


## Inference 

The `GlobalModel_generate.py` file streamlines the inference process for the global model by utilizing a Gradio interface. This file loads the foundation model from the Hugging Face Model Hub and obtains the LoRA weights and configurations from the output directory.

```bash
python GlobalModel_generate.py \
      --load_8bit \
      --base_model 'chavinlo/alpaca-native' \
      --lora_weights_path /output/path/to/lora_weights  \
      --lora_config_path /output/path/to/lora_config   
      
```







