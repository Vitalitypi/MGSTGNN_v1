import csv
import numpy as np

max_dict = {
    'PEMS03': 10.194,
    'PEMS04': 2712.1,
    'PEMS07': 20.539,
    'PEMS08': 3274.4
}


def open_graph(DATASET, filename, num_of_vertices, direction=False, id_filename=None):
    adj = np.zeros((int(num_of_vertices), int(num_of_vertices)), dtype=np.float32)
    dis = np.zeros((int(num_of_vertices), int(num_of_vertices)), dtype=np.float32)
    if id_filename:
        with open(id_filename, 'r') as f:
            id_dict = {int(i): idx for idx, i in enumerate(f.read().strip().split('\n'))}  # 把节点id（idx）映射成从0开始的索引

        with open(filename, 'r') as f:
            f.readline()  # 略过表头那一行
            reader = csv.reader(f)
            for row in reader:
                if len(row) != 3:
                    continue
                i, j, distance = int(row[0]), int(row[1]), float(row[2])
                adj[id_dict[i], id_dict[j]] = 1
                dis[id_dict[i], id_dict[j]] = max_dict[DATASET] / distance
                if not direction:
                    adj[id_dict[j], id_dict[i]] = 1
                    dis[id_dict[j], id_dict[i]] = max_dict[DATASET] / distance

        return adj, dis  # adj matrix, distance matrix

    with open(filename, 'r') as f:
        f.readline()
        reader = csv.reader(f)
        for row in reader:
            if len(row) != 3:
                continue
            i, j, distance = int(row[0]), int(row[1]), float(row[2])
            adj[i, j] = 1
            dis[i, j] = max_dict[DATASET] / distance
            if not direction:
                adj[j, i] = 1
                dis[j, i] = max_dict[DATASET] / distance
    return adj, dis


def generate_hop(DATASET, HOP_K):
    # get basic flow data
    flow_data = np.load('../dataset/{}/{}.npz'.format(DATASET, DATASET))['data']
    flow_data = flow_data[..., :1]
    print(flow_data.shape)
    time_stamps, num_nodes, dim = flow_data.shape

    csv_name = '../dataset/{}/{}.csv'.format(DATASET, DATASET)
    id_filename = None
    if DATASET.lower() == 'pems03':
        id_filename = "./dataset/PEMS03/PEMS03.txt"
    adj, dis = open_graph(DATASET, csv_name, num_nodes, id_filename=id_filename)
    dis = dis / np.max(dis)
    # 定义想要生成的文件
    hops = np.zeros((time_stamps, num_nodes, HOP_K))
    A = np.copy(adj)
    for k in range(HOP_K):
        # 计算第k跳邻居的平均值
        # 依次计算所有点
        for i in range(num_nodes):
            # 获取该点的k+1阶邻居
            indices = np.nonzero(A[i])
            value = np.sum(flow_data[:, indices[0], 0] * dis[i, indices[0]], axis=1)
            cnt = len(indices[0])
            if A[i][i] > 0:
                value -= flow_data[:, i, 0]
                cnt -= 1
            if cnt == 0:
                hops[:, i, k] = [0 for _ in range(time_stamps)]
            else:
                hops[:, i, k] = value / cnt
        # 正则化
        mean, std = hops[..., k].mean(), hops[..., k].std()
        hops[..., k] = (hops[..., k] - mean) / std
        A = np.dot(A, adj)
    # 保存文件
    np.savez('../dataset/{}/hop.npz'.format(DATASET), data=hops)


def test(DATASET):
    # csv_name = '../dataset/{}/{}.csv'.format(DATASET,DATASET)
    # open_graph(csv_name,num_nodes)
    hops = np.load('../dataset/{}/hop.npz'.format(DATASET))['data']
    print(hops[:10, 0, 0])
    # [-0.37989923 -0.39155981 -0.36349649 -0.31860226 -0.29351812 -0.27729293 -0.23883943 -0.27468301 -0.24360124 -0.25215458]
    # [-0.99063989 -0.99149587 -0.9894358  -0.98614021 -0.98429884 -0.98310778 -0.98028499 -0.98291619 -0.98063454 -0.98126242]

# test('PEMS04')

# generate_hop('PEMS04',1)
