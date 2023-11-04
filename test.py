import numpy as np

x = np.ones((16,12,3))[...,:0]
y = np.ones((16,12,3))
z = np.concatenate([x,y],axis=-1)
print(z.shape,x)
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
