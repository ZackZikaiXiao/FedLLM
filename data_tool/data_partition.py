import os
from data_download.databricks_dolly_15k.client_data_allocation import allocate_data

class DataPartition:
    def __init__(self, args) -> None:
        self.dataset = args.dataset
        self.num_client = args.num_clients
        
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
        elif self.dataset == "GLUE:sst-2":
            # GLUE:sst-2 相关的逻辑
            pass

    
    
    
