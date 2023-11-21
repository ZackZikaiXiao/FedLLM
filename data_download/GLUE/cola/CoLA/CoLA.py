import sys
sys.path.append("./data_download/GLUE")
from instructions import INSTRUCTIONS
import os
import pandas as pd
import random
import json

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
    for index, row in cola_train.iterrows():
        # count += 1
        instance = {}
        instruction_index = random.choice(range(instruction_num))
        instruction = INSTRUCTIONS["acceptability"][instruction_index]
        instance["instruction"] = instruction
        instance["context"] = row['sentence']
        if row['label'] == 1:
            instance['response'] = "acceptable"
        else:
            instance['response'] = 'unacceptable'
        instance['category'] = 'acceptability'
        data.append(instance)
        # if count == 5:
        #     print(data)
        #     break
    with open("./data_download/GLUE/cola/CoLA/CoLA.json", 'w') as write_f:
        write_f.write(json.dumps(data, indent=4))

def generate_test_json_file():
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
    for index, row in cola_train.iterrows():
        # count += 1
        instance = {}
        instruction_index = random.choice(range(instruction_num))
        instruction = INSTRUCTIONS["acceptability"][instruction_index]
        instance["instruction"] = instruction
        instance["context"] = row['sentence']
        if row['label'] == 1:
            instance['response'] = "acceptable"
        else:
            instance['response'] = 'unacceptable'
        instance['category'] = 'acceptability'
        data.append(instance)
        # if count == 5:
        #     print(data)
        #     break
    with open("./data_download/GLUE/cola/CoLA/CoLA_test.json", 'w') as write_f:
        write_f.write(json.dumps(data, indent=4))

if __name__ == "__main__":
    generate_train_json_file()
    generate_test_json_file()
