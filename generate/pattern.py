import numpy as np
#import time

import torch
from tslearn.clustering import KShape

#from utils.dataloader import split_data_by_ratio
#from utils.norm import StandardScaler
class StandardScaler:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        if type(data) == torch.Tensor and type(self.mean) == np.ndarray:
            self.std = torch.from_numpy(self.std).to(data.device).type(data.dtype)
            self.mean = torch.from_numpy(self.mean).to(data.device).type(data.dtype)
        return (data * self.std) + self.mean

def split_data_by_ratio(data, val_ratio, test_ratio):
    data_len = data.shape[0]
    test_data = data[-int(data_len * test_ratio):]
    val_data = data[-int(data_len * (test_ratio + val_ratio)):-int(data_len * test_ratio)]
    train_data = data[:-int(data_len * (test_ratio + val_ratio))]

    return train_data, val_data, test_data

dataset = "PEMS04"
cand_key_time_steps = 28*288
window = 6
horizon = 6
RATIO = 0.6
val_ratio = 0.2
test_ratio = 0.2
n_cluster = 1
ITER = 5
output_dim = 1
data = np.load('../dataset/{}/{}.npz'.format(dataset,dataset))['data']
num_nodes = data.shape[1]

data = data[...,:output_dim]
# data_len = data.shape[0]
# normalize st data(only normalize the first dimension data)
mean = data[..., 0].mean()
std = data[..., 0].std()
scaler = StandardScaler(mean, std)
data[..., 0] = scaler.transform(data[..., 0])
# spilit dataset by days or by ratio
train_data, _, _ = split_data_by_ratio(data, val_ratio, test_ratio)
# train_data = data[:-int(data_len * (test_ratio + val_ratio))]

total = train_data.shape[0]
samples = []
for t in range(window,total-(window+horizon)+1):
    samples.append(train_data[t:t+window+horizon])
samples = np.array(samples)     # (10180, 12, 307, 1)

samples = samples[:cand_key_time_steps].swapaxes(1, 2)  # (28*288, 307, 12, 1)
print(samples.shape)
res = []
for i in range(num_nodes):
    km = KShape(n_clusters=n_cluster, max_iter=ITER).fit(samples[:,i])
    pattern_key = km.cluster_centers_
    res.append(pattern_key[...,0])
    print("finished {}!".format(i+1))
res = np.concatenate(res,axis=0)
np.savez('../dataset/{}/pattern.npz'.format(dataset), data=res)


