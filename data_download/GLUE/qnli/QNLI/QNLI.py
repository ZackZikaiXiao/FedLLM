import sys
sys.path.append("./data_download/GLUE")
from instructions import INSTRUCTIONS
import os
import pandas as pd
import random
import json
def generate_train_json_file():
    # print(type(INSTRUCTIONS["sentiment"]))
    Data_path = "./data_download/GLUE/qnli/train.tsv"
    data_train = pd.read_csv(
        Data_path,
        sep='\t',
        on_bad_lines="skip",
        header=0
    )
    instruction_num = len(INSTRUCTIONS["QNLI"])
    data = []
    # count = 0
    for index, row in data_train.iterrows():
        # count += 1
        instance = {}
        instruction_index = random.choice(range(instruction_num))
        instance["instruction"] = INSTRUCTIONS["QNLI"][instruction_index]
        instance["context"] = "Question: " + str(row['question']) + "\n" + "Sentence: " + str(row['sentence'])
        if  row['label'] == "entailment":
            instance['response'] = "yes"
        else:
            instance['response'] = "no"
        instance['category'] = 'QA'
        data.append(instance)
        # if count == 5:
        #     print(data)
        #     break
    with open("./data_download/GLUE/qnli/QNLI/QNLI.json", 'w') as write_f:
        write_f.write(json.dumps(data, indent=4))

def generate_test_json_file():
    # print(type(INSTRUCTIONS["sentiment"]))
    Data_path = "./data_download/GLUE/qnli/dev.tsv"
    data_train = pd.read_csv(
        Data_path,
        sep='\t',
        on_bad_lines="skip",
        header=0
    )
    instruction_num = len(INSTRUCTIONS["QNLI"])
    data = []
    # count = 0
    for index, row in data_train.iterrows():
        # count += 1
        instance = {}
        instruction_index = random.choice(range(instruction_num))
        instance["instruction"] = INSTRUCTIONS["QNLI"][instruction_index]
        instance["context"] = "Question: " + str(row['question']) + "\n" + "Sentence: " + str(row['sentence'])
        if  row['label'] == "entailment":
            instance['response'] = "yes"
        else:
            instance['response'] = "no"
        instance['category'] = 'QA'
        data.append(instance)
        # if count == 5:
        #     print(data)
        #     break
    with open("./data_download/GLUE/qnli/QNLI/QNLI_test.json", 'w') as write_f:
        write_f.write(json.dumps(data, indent=4))

if __name__ == "__main__":
    generate_train_json_file()
    generate_test_json_file()

