import sys
sys.path.append("./data_download/GLUE")
from instructions import INSTRUCTIONS
import os
import pandas as pd
import random
import json
def generate_train_json_file():
# print(type(INSTRUCTIONS["sentiment"]))
    Data_path = "./data_download/GLUE/mrpc/train.tsv"
    data_train = pd.read_csv(
        Data_path,
        sep='\t',
        on_bad_lines="skip",
        header=0
    )
    instruction_num = len(INSTRUCTIONS["MRPC"])
    data = []
    # count = 0
    for index, row in data_train.iterrows():
        # count += 1
        instance = {}
        instruction_index = random.choice(range(instruction_num))
        instance["instruction"] = INSTRUCTIONS["MRPC"][instruction_index]
        instance["context"] = "Sentence1: " + str(row['#1 String']) + "\n" + "Sentence2: " + str(row['#2 String'])
        if  row['Quality'] == 1:
            instance['response'] = "yes"
        else:
            instance['response'] = 'no'
        instance['category'] = 'paraphrase'
        data.append(instance)
        # if count == 5:
        #     print(data)
        #     break
    with open("./data_download/GLUE/mrpc/MRPC/MRPC.json", 'w') as write_f:
        write_f.write(json.dumps(data, indent=4))

def generate_test_json_file():
# print(type(INSTRUCTIONS["sentiment"]))
    Data_path = "./data_download/GLUE/mrpc/dev.tsv"
    data_train = pd.read_csv(
        Data_path,
        sep='\t',
        on_bad_lines="skip",
        header=0
    )
    instruction_num = len(INSTRUCTIONS["MRPC"])
    data = []
    # count = 0
    for index, row in data_train.iterrows():
        # count += 1
        instance = {}
        instruction_index = random.choice(range(instruction_num))
        instance["instruction"] = INSTRUCTIONS["MRPC"][instruction_index]
        instance["context"] = "Sentence1: " + str(row['#1 String']) + "\n" + "Sentence2: " + str(row['#2 String'])
        if  row['Quality'] == 1:
            instance['response'] = "yes"
        else:
            instance['response'] = 'no'
        instance['category'] = 'paraphrase'
        data.append(instance)
        # if count == 5:
        #     print(data)
        #     break
    with open("./data_download/GLUE/mrpc/MRPC/MRPC_test.json", 'w') as write_f:
        write_f.write(json.dumps(data, indent=4))


if __name__ == "__main__":
    generate_train_json_file()
    generate_test_json_file()