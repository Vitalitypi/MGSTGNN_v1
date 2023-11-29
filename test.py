import numpy as np
import pandas as pd


def get_node_feature(dataset, num_nodes):
    node_feature = np.zeros((num_nodes,4))
    # 读取TXT文件
    data = pd.read_csv('./dataset/{}/feature.txt'.format(dataset), sep='\t', header=None)


    for i in range(num_nodes):
        arr = data[0][i].split(',')
        node_feature[i] = [int(arr[1]),int(arr[2]),int(arr[3]),float(arr[4])]
    return node_feature
get_node_feature('PEMS03',358)
