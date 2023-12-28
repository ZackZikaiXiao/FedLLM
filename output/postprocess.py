import sys
from sklearn.metrics import confusion_matrix
sys.path.append("./output/GLUE")
sys.path.append("./")
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import rcParams
import os
import random
from fed_utils import cleansed_response_for_acceptability

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



def process_result_from_quail(response_path):
    result = pd.read_excel(response_path)
    prediction = result['cleaned_response']
    labels = result['label']
    classes = ['A', 'B', 'C', 'D']
    labels = labels[prediction.isin(classes)]
    prediction = prediction[prediction.isin(classes)]
    save_path = os.path.join(os.path.dirname(response_path), 'confusion_map.png')
    output_confusion_map(prediction=prediction, labels=labels, classes=classes, save_path=save_path)
    print(prediction.value_counts())

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

def output_confusion_map(prediction, labels, classes, save_path):
    cm = confusion_matrix(y_true=labels, y_pred=prediction, labels=classes)
    proportion = []
    length = len(cm)
    for i in cm:
        for j in i:
            temp = j / (np.sum(i))
            proportion.append(temp)
    pshow = []
    for i in proportion:
        pt = "%.2f%%" % (i * 100)
        pshow.append(pt)
    proportion = np.array(proportion).reshape(length, length)  # reshape(列的长度，行的长度)
    pshow = np.array(pshow).reshape(length, length)
    config = {
    "font.family": 'DejaVu Sans',  # 设置字体类型
    }
    rcParams.update(config)
    plt.imshow(proportion, interpolation='nearest', cmap=plt.cm.Blues)  # 按照像素显示出矩阵
    # (改变颜色：'Greys', 'Purples', 'Blues', 'Greens', 'Oranges', 'Reds','YlOrBr', 'YlOrRd',
    # 'OrRd', 'PuRd', 'RdPu', 'BuPu','GnBu', 'PuBu', 'YlGnBu', 'PuBuGn', 'BuGn', 'YlGn')
    # plt.title('confusion_matrix')
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, fontsize=12)
    plt.yticks(tick_marks, classes, fontsize=12)
    iters = np.reshape([[[i, j] for j in range(length)] for i in range(length)], (cm.size, 2))
    for i, j in iters:
        if (i == j):
            plt.text(j, i - 0.12, format(cm[i, j]), va='center', ha='center', fontsize=10, color='white',
                    weight=5)  # 显示对应的数字
            plt.text(j, i + 0.12, pshow[i, j], va='center', ha='center', fontsize=10, color='white')
        else:
            plt.text(j, i - 0.12, format(cm[i, j]), va='center', ha='center', fontsize=10)  # 显示对应的数字
            plt.text(j, i + 0.12, pshow[i, j], va='center', ha='center', fontsize=10)

    plt.ylabel('True label', fontsize=16)
    plt.xlabel('Predict label', fontsize=16)
    plt.tight_layout()
    plt.savefig(save_path)

if __name__ == "__main__":
    # response_path = 'output/quail-dirichlet_label_alpha=1-imbalanced_label-alpaca-lora/19.xlsx'
    # process_result_from_quail(response_path)
    short_result_file_name = 'output/quail-dirichlet_label_alpha=1-imbalanced_label-alpaca-lora/short_result.txt'
    draw_acc_curve(short_result_file_name)