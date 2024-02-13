import sys
sys.path.append("./data_download/GLUE")
sys.path.append("./")
from instructions import INSTRUCTIONS
import os
import pandas as pd
import random
import json
import numpy as np
from data_tool.imbalance_num import get_img_num_per_cls
# print(type(INSTRUCTIONS["sentiment"]))
def generate_train_json_file():
    CoLA_path = "./data_download/GLUE/cola/train.tsv"
    cola_train = pd.read_csv(
        CoLA_path,
        sep='\t',
        header=None,
        names=["source", "label", "other", "sentence"]
    )
    instruction_num = len(INSTRUCTIONS["acceptability"])
    data = []
    # count = 0
    acceptable_index = []
    unacceptable_index = []
    for index, row in cola_train.iterrows():
        # count += 1
        instance = {}
        instruction_index = random.choice(range(instruction_num))
        instruction = INSTRUCTIONS["acceptability"][instruction_index]
        instance["instruction"] = instruction
        instance["context"] = row['sentence']
        if row['label'] == 1:
            instance['response'] = "acceptable"
            acceptable_index.append(index)
        else:
            instance['response'] = 'unacceptable'
            unacceptable_index.append(index)
        instance['category'] = 'acceptability'
        data.append(instance)
        # if count == 5:
        #     print(data)
        #     break
    num_samples_per_cls = get_img_num_per_cls(data, 2, 'exp', 0.4)
    change_train_ratio=True
    if change_train_ratio:
        amount_of_acceptable=num_samples_per_cls[0]
        amount_of_unacceptable=num_samples_per_cls[1]
        selected_acceptable_index = np.random.choice(acceptable_index, amount_of_acceptable, replace=False).tolist()
        if amount_of_unacceptable > len(unacceptable_index):
            selected_unacceptable_index = np.random.choice(unacceptable_index, amount_of_unacceptable, replace=True).tolist()
        else:
            selected_unacceptable_index = np.random.choice(unacceptable_index, amount_of_unacceptable, replace=False).tolist()
        selected_index = selected_acceptable_index + selected_unacceptable_index
        data = [data[index] for index in selected_index]
    with open("./data_download/GLUE/cola/CoLA/CoLA.json", 'w') as write_f:
        write_f.write(json.dumps(data, indent=4))

def generate_test_json_file(ratio=None):
    CoLA_path = "./data_download/GLUE/cola/dev.tsv"
    cola_train = pd.read_csv(
        CoLA_path,
        sep='\t',
        header=None,
        names=["source", "label", "other", "sentence"]
    )
    instruction_num = len(INSTRUCTIONS["acceptability"])
    data = []
    # count = 0
    acceptable_index = []
    unacceptable_index = []
    for index, row in cola_train.iterrows():
        # count += 1
        instance = {}
        instruction_index = random.choice(range(instruction_num))
        instruction = INSTRUCTIONS["acceptability"][instruction_index]
        instance["instruction"] = instruction
        instance["context"] = row['sentence']
        if row['label'] == 1:
            instance['response'] = "acceptable"
            acceptable_index.append(index)
        else:
            instance['response'] = 'unacceptable'
            unacceptable_index.append(index)
        instance['category'] = 'acceptability'
        data.append(instance)
    change_test_ratio=True
    if change_test_ratio:
        amount_of_acceptable=322
        amount_of_unacceptable=322
        selected_acceptable_index = np.random.choice(acceptable_index, amount_of_acceptable, replace=False).tolist()
        selected_unacceptable_index = np.random.choice(unacceptable_index, amount_of_unacceptable, replace=False).tolist()
        selected_index = selected_acceptable_index + selected_unacceptable_index
        data = [data[index] for index in selected_index]
        # if count == 5:
        #     print(data)
        #     break
    with open("./data_download/GLUE/cola/CoLA/CoLA_test.json", 'w') as write_f:
        write_f.write(json.dumps(data, indent=4))

def generate_train_json_file_v2():
    CoLA_path = "./data_download/GLUE/cola/train.tsv"
    cola_train = pd.read_csv(
        CoLA_path,
        sep='\t',
        header=None,
        names=["source", "label", "other", "sentence"]
    )
    instruction_num = len(INSTRUCTIONS["acceptability"])
    data = []
    cola_train = cola_train.sample(frac=1)
    total_number_of_training_samples = len(cola_train)
    number_of_A = int(total_number_of_training_samples * 0.5)
    label_to_be_filled = 0
    labels = ['A', 'B']
    options = ['options: A:unacceptable, B:acceptable.', 'options: A:acceptable, B:unacceptable.']
    for index, row in cola_train.iterrows():
        if label_to_be_filled == 0:
            number_of_A -= 1
        instance = {}
        instance["response"] = labels[label_to_be_filled]
        instruction_index = random.choice(range(instruction_num))
        instruction = INSTRUCTIONS["acceptability"][instruction_index]
        if (row['label'] == 1 and label_to_be_filled == 0) or (row['label'] == 0 and label_to_be_filled == 1):
            instruction += options[1]
        else:
            instruction += options[0]
        instance["instruction"] = instruction
        instance["context"] = row['sentence']
        instance['category'] = 'acceptability'
        data.append(instance)
        if number_of_A == 0:
            label_to_be_filled = 1
    random.shuffle(data)
    with open("./data_download/GLUE/cola/CoLA/CoLA.json", 'w') as write_f:
        write_f.write(json.dumps(data, indent=4))



if __name__ == "__main__":
    # generate_train_json_file()
    generate_test_json_file()
    
