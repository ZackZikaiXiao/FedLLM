import sys
sys.path.append("./data_download/GLUE")
from instructions import INSTRUCTIONS
import os
import pandas as pd
import random
import json
def generate_train_json_file():
    Data_path = "./data_download/GLUE/sst-2/train.tsv"
    data_train = pd.read_csv(
        Data_path,
        sep='\t',
        header=0,
        # names=["source", "label", "other", "sentence"]
    )
    # print(data_train)
    instruction_num = len(INSTRUCTIONS["sentiment"])
    data = []
    # count = 0
    for index, row in data_train.iterrows():
        # count += 1
        instance = {}
        instruction_index = random.choice(range(instruction_num))
        instruction = INSTRUCTIONS["sentiment"][instruction_index]
        instance["instruction"] = instruction
        instance["context"] = row['sentence']
        if row['label'] == 1:
            instance['response'] = "positive"
        else:
            instance['response'] = 'negative'
        instance['category'] = 'sentiment'
        data.append(instance)
        # if count == 5:
        #     print(data)
        #     break
    with open("./data_download/GLUE/sst-2/SST-2/SST2.json", 'w') as write_f:
        write_f.write(json.dumps(data, indent=4))

    # json_str = json.dumps(data)
    # print(type(json_str))

def generate_test_json_file():
    Data_path = "./data_download/GLUE/sst-2/dev.tsv"
    data_train = pd.read_csv(
        Data_path,
        sep='\t',
        header=0,
        # names=["source", "label", "other", "sentence"]
    )
    # print(data_train)
    instruction_num = len(INSTRUCTIONS["sentiment"])
    data = []
    # count = 0
    for index, row in data_train.iterrows():
        # count += 1
        instance = {}
        instruction_index = random.choice(range(instruction_num))
        instruction = INSTRUCTIONS["sentiment"][instruction_index]
        instance["instruction"] = instruction
        instance["context"] = row['sentence']
        if row['label'] == 1:
            instance['response'] = "positive"
        else:
            instance['response'] = 'negative'
        instance['category'] = 'sentiment'
        data.append(instance)
        # if count == 5:
        #     print(data)
        #     break
    with open("./data_download/GLUE/sst-2/SST-2/SST-2_test.json", 'w') as write_f:
        write_f.write(json.dumps(data, indent=4))

    # json_str = json.dumps(data)
    # print(type(json_str))

# print(type(INSTRUCTIONS["sentiment"]))
if __name__ == "__main__":
    generate_train_json_file()
    generate_test_json_file()
