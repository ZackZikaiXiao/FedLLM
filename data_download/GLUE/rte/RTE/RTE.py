import sys
sys.path.append("./data_download/GLUE")
from instructions import INSTRUCTIONS
import os
import pandas as pd
import random
import json

def generate_test_json_file():
    Data_path = "./data_download/GLUE/rte/dev.tsv"
    data_train = pd.read_csv(
        Data_path,
        sep='\t',
        on_bad_lines="skip",
        header=0
    )
    instruction_num = len(INSTRUCTIONS["RTE"])
    data = []
    # count = 0
    for index, row in data_train.iterrows():
        # count += 1
        instance = {}
        instruction_index = random.choice(range(instruction_num))
        instance["instruction"] = INSTRUCTIONS["RTE"][instruction_index]
        instance["context"] = "Premise: " + str(row['sentence1']) + "\n" + "Hypothesis: " + str(row['sentence2'])
        if  row['label'] == "entailment":
            instance['response'] = "yes"
        else:
            instance['response'] = "no"
        instance['category'] = 'NLI'
        data.append(instance)
        # if count == 5:
        #     print(data)
        #     break
    with open("./data_download/GLUE/rte/RTE/RTE_test.json", 'w') as write_f:
        write_f.write(json.dumps(data, indent=4))

    # json_str = json.dumps(data)
    # print(type(json_str))

def generate_train_json_file():
    Data_path = "./data_download/GLUE/rte/train.tsv"
    data_train = pd.read_csv(
        Data_path,
        sep='\t',
        on_bad_lines="skip",
        header=0
    )
    instruction_num = len(INSTRUCTIONS["RTE"])
    data = []
    # count = 0
    for index, row in data_train.iterrows():
        # count += 1
        instance = {}
        instruction_index = random.choice(range(instruction_num))
        instance["instruction"] = INSTRUCTIONS["RTE"][instruction_index]
        instance["context"] = "Premise: " + str(row['sentence1']) + "\n" + "Hypothesis: " + str(row['sentence2'])
        if  row['label'] == "entailment":
            instance['response'] = "yes"
        else:
            instance['response'] = "no"
        instance['category'] = 'NLI'
        data.append(instance)
        # if count == 5:
        #     print(data)
        #     break
    with open("./data_download/GLUE/rte/RTE/RTE.json", 'w') as write_f:
        write_f.write(json.dumps(data, indent=4))

    # json_str = json.dumps(data)
    # print(type(json_str))


if __name__ == "__main__":
    generate_train_json_file()
    generate_test_json_file()
# # print(type(INSTRUCTIONS["sentiment"]))
#     Data_path = "./data_download/GLUE/rte/train.tsv"
#     data_train = pd.read_csv(
#         Data_path,
#         sep='\t',
#         on_bad_lines="skip",
#         header=0
#     )
#     instruction_num = len(INSTRUCTIONS["RTE"])
#     data = []
#     # count = 0
#     for index, row in data_train.iterrows():
#         # count += 1
#         instance = {}
#         instruction_index = random.choice(range(instruction_num))
#         instance["instruction"] = INSTRUCTIONS["RTE"][instruction_index]
#         instance["context"] = "Premise: " + str(row['sentence1']) + "\n" + "Hypothesis: " + str(row['sentence2'])
#         if  row['label'] == "entailment":
#             instance['response'] = "yes"
#         else:
#             instance['response'] = "no"
#         instance['category'] = 'NLI'
#         data.append(instance)
#         # if count == 5:
#         #     print(data)
#         #     break
#     with open("./data_download/GLUE/rte/RTE/RTE2.json", 'w') as write_f:
#         write_f.write(json.dumps(data, indent=4))

#     # json_str = json.dumps(data)
#     # print(type(json_str))
