import sys
sys.path.append("./data_download/GLUE")
from instructions import INSTRUCTIONS
import os
import pandas as pd
import random
import json
def generate_train_json_file():
    # print(type(INSTRUCTIONS["sentiment"]))
    Data_path = "./data_download/GLUE/sts-b/train.tsv"
    data_train = pd.read_csv(
        Data_path,
        sep='\t',
        header=0,
        on_bad_lines="skip",
    )
    instruction_num = len(INSTRUCTIONS["sentence_similarity"])
    data = []
    # count = 0
    for index, row in data_train.iterrows():
        # count += 1
        instance = {}
        instruction_index = random.choice(range(instruction_num))
        instruction = INSTRUCTIONS["sentence_similarity"][instruction_index]
        instance["instruction"] = instruction
        instance["context"] = row['sentence1'] + "\n" + row['sentence2']
        instance['response'] = row['score']
        instance['category'] = 'sentence_similarity'
        data.append(instance)
        # if count == 5:
        #     print(data)
        #     break
    with open("./data_download/GLUE/sts-b/STS-B/STS-B.json", 'w') as write_f:
        write_f.write(json.dumps(data, indent=4))

    # json_str = json.dumps(data)
    # print(type(json_str))

def generate_test_json_file():
    # print(type(INSTRUCTIONS["sentiment"]))
    Data_path = "./data_download/GLUE/sts-b/dev.tsv"
    data_train = pd.read_csv(
        Data_path,
        sep='\t',
        header=0,
        on_bad_lines="skip",
    )
    instruction_num = len(INSTRUCTIONS["sentence_similarity"])
    data = []
    # count = 0
    for index, row in data_train.iterrows():
        # count += 1
        instance = {}
        instruction_index = random.choice(range(instruction_num))
        instruction = INSTRUCTIONS["sentence_similarity"][instruction_index]
        instance["instruction"] = instruction
        instance["context"] = row['sentence1'] + "\n" + row['sentence2']
        instance['response'] = row['score']
        instance['category'] = 'sentence_similarity'
        data.append(instance)
        # if count == 5:
        #     print(data)
        #     break
    with open("./data_download/GLUE/sts-b/STS-B/STS-B_test.json", 'w') as write_f:
        write_f.write(json.dumps(data, indent=4))


if __name__ == "__main__":
    generate_train_json_file()
    generate_test_json_file()