import torch

import numpy as np

data = np.load('./dataset/PEMS04/PEMS04.npz')['data']
print(data.shape)
print(data[285:295,0])
