from torch.nn.functional import normalize
import torch

def other_function():
    print('ok')
    # local_batch_size = 64
    # local_dataset_len_dict = {
    #     0 : 200,
    #     1 : 144,
    #     3 : 300,
    #     7 : 200,
    # }
    # selected_clients_set = {0, 1, 3, 7}
    # weights_array = normalize(
    #     torch.tensor([local_dataset_len_dict[client_id] for client_id in selected_clients_set],
    #                  dtype=torch.float32),
    #     p=1, dim=0)
    # local_update_steps = []
    # for client_id in selected_clients_set:
    #     local_update_steps.append(local_dataset_len_dict[client_id] // local_batch_size)
    # dim0 = weights_array.shape[0]
    # sum_of_PiTi = 0
    # for i in range(dim0):
    #     sum_of_PiTi += weights_array[i] * local_update_steps[i]
        
    # Nova_weight_arrary = [sum_of_PiTi / local_update_steps[i] for i in range(dim0)]
    

# if __name__ == "__main__":
#     other_function()



