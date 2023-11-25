import os
from data_download.databricks_dolly_15k.client_data_allocation import allocate_data
from data_download.GLUE.partition import partition


class DataPartition:
    def __init__(self, args) -> None:
        self.dataset = args.dataset
        self.num_client = args.num_clients
        self.dirichlet_alpha = args.dirichlet_alpha
        self.partition_method = args.partition_method

    def partition(self):
        if self.dataset == "new-databricks-dolly-15k":
            data_folder = './data_download/databricks_dolly_15k/data'
            if os.path.exists(data_folder):
                print(f"new-databricks-dolly-15k is ready.")
            else:
                num_client = 10  # The number of clients
                diff_quantity = 0  # Whether clients have different amounts of data
                # 直接调用函数
                allocate_data(self.num_client, diff_quantity)
        elif self.dataset == "sst-2":
            data_folder = "./data_download/GLUE/sst-2/SST-2/SST-2.json"
            partition(data_path=data_folder, num_clients=self.num_client, dirichlet_alpha=self.dirichlet_alpha,
                      partition_method=self.partition_method)
        
        elif self.dataset == "rte":
            data_folder = "./data_download/GLUE/rte/RTE/RTE.json"
            partition(data_path=data_folder, num_clients=self.num_client, dirichlet_alpha=self.dirichlet_alpha,
                      partition_method=self.partition_method)
        
        elif self.dataset == "cola":
            data_folder = "./data_download/GLUE/cola/CoLA/CoLA.json"
            partition(data_path=data_folder, num_clients=self.num_client, dirichlet_alpha=self.dirichlet_alpha,
                      partition_method=self.partition_method)
            
        elif self.dataset == "qnli":
            data_folder = "./data_download/GLUE/qnli/QNLI/QNLI.json"
            partition(data_path=data_folder, num_clients=self.num_client, dirichlet_alpha=self.dirichlet_alpha,
                      partition_method=self.partition_method)
            
        elif self.dataset == "qqp":
            data_folder = "./data_download/GLUE/qqp/QQP/QQP.json"
            partition(data_path=data_folder, num_clients=self.num_client, dirichlet_alpha=self.dirichlet_alpha,
                      partition_method=self.partition_method)
            
        elif self.dataset == "sts-b":
            data_folder = "./data_download/GLUE/sts-b/STS-B/STS-B.json"
            # Label为分数如何partition
            partition(data_path=data_folder, num_clients=self.num_client, dirichlet_alpha=self.dirichlet_alpha,
                      partition_method=self.partition_method, num_of_classes_for_stsb=3)

        elif self.dataset == "wnli":
            data_folder = "./data_download/GLUE/wnli/WNLI/WNLI.json"
            partition(data_path=data_folder, num_clients=self.num_client, dirichlet_alpha=self.dirichlet_alpha,
                      partition_method=self.partition_method)
            
        elif self.dataset == "mrpc":
            data_folder = "./data_download/GLUE/mrpc/MRPC/MRPC.json"
            partition(data_path=data_folder, num_clients=self.num_client, dirichlet_alpha=self.dirichlet_alpha,
                      partition_method=self.partition_method)
            
        elif self.dataset == "mnli":
            data_folder = "./data_download/GLUE/mnli/MNLI/MNLI.json"
            partition(data_path=data_folder, num_clients=self.num_client, dirichlet_alpha=self.dirichlet_alpha,
                      partition_method=self.partition_method)
