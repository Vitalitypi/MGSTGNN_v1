import numpy as np
import torch

TOP_K = 3
DATASET = 'PEMS04'
NUM_PERIOD = 10

# get basic flow data
flow_data = np.load('../dataset/{}/{}.npz'.format(DATASET,DATASET))['data']
flow_data = flow_data[...,:1]
print(flow_data.shape)
time_stamps,num_nodes,dim = flow_data.shape

# Get information about the last k periods
def FFT_for_Period(x, k=2):
    # x [B, T, C]
    xf = torch.fft.rfft(x, dim=1)
    # find period by amplitudes
    frequency_list = abs(xf).mean(0).mean(-1)
    frequency_list[0] = 0
    _, top_list = torch.topk(frequency_list, k)
    top_list = top_list.detach().cpu().numpy()
    period = x.shape[1] // top_list
    return period, abs(xf).mean(-1)[:, top_list]

def generate_period():
    data = torch.tensor(flow_data)
    data = torch.unsqueeze(data,0)
    indices = []
    for i in range(num_nodes):
        x,_ = FFT_for_Period(data[:,:,i],NUM_PERIOD)
        indices.append(x)
    matrix = np.array(indices)
    # 获取唯一值和计数
    unique_values, counts = np.unique(matrix, return_counts=True)
    # 对计数进行排序
    sorted_indices = np.argsort(counts)[::-1]  # 降序排序
    # 取前k个唯一值和计数
    top_k_values = unique_values[sorted_indices[:NUM_PERIOD]]
    print("The first {} periods are: ".format(NUM_PERIOD),top_k_values)
    res = np.zeros((time_stamps,num_nodes,TOP_K))
    for t in range(time_stamps):
        for k in range(TOP_K):
            res[t,:,k] = np.ones((num_nodes))*(t%top_k_values[k])
    print("finished!")
    np.savez('../dataset/{}/period_{}.npz'.format(DATASET,TOP_K),data=res)

generate_period()

