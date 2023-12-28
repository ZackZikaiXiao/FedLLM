import sys
from sklearn.metrics import confusion_matrix
sys.path.append("./output/GLUE")
sys.path.append("./")
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import random
from fed_utils import cleansed_response_for_acceptability

# def cleansed_response(pred):
#     pred = [item.lower() for item in pred]
#     pred = [item[0:12] for item in pred]
#     for index, item in enumerate(pred):
#         if item[0:10] == 'acceptable':
#             pred[index] = 'acceptable'
#     return pred

def postprocess_response_from_pretraining_CoLA(response_path, change_test_set_ratio):
    result = pd.read_excel(response_path)
    label = result['label']
    pred = cleansed_response_for_acceptability(result['response'])
    if change_test_set_ratio:
        acceptable_index = []
        for index, value in label.items():
            if value == 'acceptable':
                acceptable_index.append(index)
        acceptable_index_to_be_droped = random.sample(acceptable_index, 399)
        label = label.drop(index=acceptable_index_to_be_droped)
        pred = pd.Series(pred)
        pred = pred.drop(acceptable_index_to_be_droped)
        pred = pred.to_list()
    C = confusion_matrix(y_pred=pred, y_true=label.to_list(), labels=['acceptable', 'unacceptable'])
    plt.matshow(C, cmap=plt.cm.Reds)
    for i in range(len(C)):
        for j in range(len(C)):
            plt.annotate(C[j, i], xy=(i, j), horizontalalignment='center', verticalalignment='center')
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.title('0: acceptable, 1:unacceptable')
    save_path = os.path.join(os.path.dirname(response_path), 'confusion_map.png')
    plt.savefig(save_path, bbox_inches='tight')

def draw_acc_curve(short_result_file_name):
    result = pd.read_csv(
        short_result_file_name,
        header=None,
        sep=' ',
    )
    acc_list = result.iloc[:, 1].tolist()
    index = result.iloc[:, 0].tolist()
    plt.plot(index, acc_list)
    plt.title('Accuracy curve')
    plt.xlabel('number of communications')
    plt.ylabel('Accuracy')
    save_path = os.path.join(os.path.dirname(short_result_file_name), 'Accuracy_curve.png')
    plt.savefig(save_path)
if __name__ == "__main__":
    # response_path = './output/GLUE/cola/0.xlsx'
    # postprocess_response_from_pretraining_CoLA(response_path, False)
    short_result_file_name = './output/GLUE/cola-virtual-tokens-5/short_result.txt'
    draw_acc_curve(short_result_file_name)