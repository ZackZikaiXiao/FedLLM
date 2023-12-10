import sys
sys.path.append("./data_download/GLUE")
from instructions import INSTRUCTIONS
import os
import pandas as pd
import random
import json
import numpy as np

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
    change_train_ratio=False
    if change_train_ratio:
        amount_of_acceptable=360*8
        amount_of_unacceptable=360*2
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
    change_test_ratio=False
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

if __name__ == "__main__":
    generate_train_json_file()
    generate_test_json_file()
