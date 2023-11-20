import json
import os

import numpy as np
from scipy.stats import dirichlet
import pandas as pd


def partition(data_path, num_clients, dirichlet_alpha, partition_method="dirichlet_label"):
    data_path2 = os.path.abspath(os.path.join(data_path, ".."))
    data_path2 = os.path.join(data_path2, str(num_clients))
    if partition_method == "dirichlet_label":
        # 可视化类别分布
        df = pd.read_json(data_path, orient='records')
        dataset_len = len(df)
        num_unique_labels = len(df['response'].unique())
        unique_label_list = np.array(df['response'].unique())
        all_label_list = np.array(df['response'])
        # set alpha = 10 means that the num of samples are nearly uniformly distributed among clients
        dirichlet_samples = dirichlet.rvs([10] * num_clients, size=1)
        num_samples_per_client = (np.floor(dirichlet_samples * dataset_len).astype(int)).squeeze()
        # print(num_samples_per_client)
        cls_priors = np.random.dirichlet(alpha=[dirichlet_alpha] * num_unique_labels, size=num_clients)
        # print(cls_priors)
        prior_cumsum = np.cumsum(cls_priors, axis=1)
        idx_list = [np.where(all_label_list == i)[0] for i in unique_label_list]
        cls_amount = [len(idx_list[i]) for i in list(range(len(idx_list)))]

        sample_idx_per_client = [[] for _ in range(num_clients)]
        while np.sum(num_samples_per_client) != 0:
            curr_clnt = np.random.randint(num_clients)
            if num_samples_per_client[curr_clnt] <= 0:
                continue
            num_samples_per_client[curr_clnt] -= 1
            curr_prior = prior_cumsum[curr_clnt]
            while True:
                cls_label = np.argmax(np.random.uniform() <= curr_prior)
                # Redraw class label if trn_y is out of that class
                if cls_amount[cls_label] <= 0:
                    continue
                cls_amount[cls_label] -= 1
                sample_idx_per_client[curr_clnt].append(idx_list[cls_label][cls_amount[cls_label]])
                break
        os.makedirs(data_path2, exist_ok=True)
        for client_id in range(num_clients):
            print(
                "\n Generating the local training dataset of Client_{}".format(client_id)
            )
            sub_df = df.loc[sample_idx_per_client[client_id]]
            sub_df = sub_df.reset_index().drop('index', axis=1)
            sub_remaining_df_dic = sub_df.to_dict(orient='records')

            with open(os.path.join(data_path2, "local_training_{}.json".format(client_id)), 'w') as outfile:
                json.dump(sub_remaining_df_dic, outfile)
            # print(sub_df)
    elif partition_method == "dirichlet_quantity":
        df = pd.read_json(data_path, orient='records')
        dataset_len = len(df)
        dirichlet_samples = dirichlet.rvs([dirichlet_alpha] * num_clients, size=1)
        client_samples = (np.floor(dirichlet_samples * dataset_len).astype(int)).squeeze()
        for client_id, size in enumerate(client_samples):
            sub_df = df.sample(frac=(size / len(df)))
            df = df.drop(sub_df.index)
            sub_df = sub_df.reset_index().drop('index', axis=1)
            sub_remaining_df_dic = sub_df.to_dict(orient='records')
            with open(os.path.join(data_path2, "local_training_{}.json".format(client_id)), 'w') as outfile:
                json.dump(sub_remaining_df_dic, outfile)
            # print(len(sub_df))


if __name__ == '__main__':
    partition("./data_download/GLUE/sst-2/SST-2/SST-2.json", 10, 3, partition_method="dirichlet_label")
    # df2 = pd.read_json("./SST-2.json", orient='records')
    # dataset_len2 = len(df2)
    # print(dataset_len2)
    # num_unique_labels2 = len(df2['response'].unique())
    # all_label_list2 = np.array(df2['response'])
    # unique_label_list2 = np.array(df2['response'].unique())
    # idx_list2 = [np.where(all_label_list2 == i)[0] for i in unique_label_list2]
    # print(len(idx_list2[0]) / dataset_len2)
    # print(len(idx_list2[1]) / dataset_len2)
    #
    #
    # df = pd.read_json("./local_training_1.json", orient='records')
    # dataset_len = len(df)
    # print(dataset_len)
    # num_unique_labels = len(df['response'].unique())
    # unique_label_list = np.array(df['response'].unique())
    # all_label_list = np.array(df['response'])
    # idx_list = [np.where(all_label_list == i)[0] for i in unique_label_list]
    # print(len(idx_list[0]) / dataset_len)
    # print(len(idx_list[1]) / dataset_len)
