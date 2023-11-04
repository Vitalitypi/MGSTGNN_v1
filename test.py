import numpy as np
DATASET='PEMS08'
data = np.load('./dataset/{}/{}.npz'.format(DATASET,DATASET))['data']

# data = (data[...,:1]).astype(int)
print(data.dtype)
np.savez('./{}.npz'.format(DATASET),data=data)

# PEMS03 (26208, 358, 1)
# PEMS04 (16992, 307, 1)
# PEMS07 (28224, 883, 1)
# pems08 (17856, 170, 1)


'''
还需要的调参方式：
    1、不同的holiday和weekend组合方式
    2、不同的正则化方式
    3、
相关命令：
    1、git add --ignore-errors .忽略文件
其他：
    1、生成周期数据只是用训练集
'''
