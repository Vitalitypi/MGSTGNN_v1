import csv

import numpy as np



def get_adj_dis_matrix(dataset, num_of_vertices, direction=False, id_filename=None):
    '''
    Parameters
    ----------
    distance_df_filename: str, path of the csv file contains edges information

    num_of_vertices: int, the number of vertices

    Returns
    ----------
    A: np.ndarray, adjacency matrix

    '''
    max_dict = {
        'PEMS03': 10.194,
        'PEMS04': 2712.1,
        'PEMS07': 20.539,
        'PEMS08': 3274.4
    }
    distance_df_filename = '../dataset/{}/{}.csv'.format(dataset,dataset)
    A = np.zeros((int(num_of_vertices), int(num_of_vertices)), dtype=np.float32)
    distaneA = np.zeros((int(num_of_vertices), int(num_of_vertices)), dtype=np.float32)
    # distaneA = np.full((int(num_of_vertices), int(num_of_vertices)),np.inf)
    # if node id in distance_df_file doesn't start from zero,
    # it needs to be remap via id_filename which contains the corresponding id with sorted index.
    if id_filename:
        with open(id_filename, 'r') as f:
            id_dict = {int(i): idx for idx, i in enumerate(f.read().strip().split('\n'))}  # 把节点id（idx）映射成从0开始的索引

        with open(distance_df_filename, 'r') as f:
            f.readline()  # 略过表头那一行
            reader = csv.reader(f)
            for row in reader:
                if len(row) != 3:
                    continue
                i, j, distance = int(row[0]), int(row[1]), float(row[2])
                A[id_dict[i], id_dict[j]] = 1
                distaneA[id_dict[i], id_dict[j]] = distance
                if not direction:
                    A[id_dict[j], id_dict[i]] = 1
                    distaneA[id_dict[j], id_dict[i]] = distance

        return A, distaneA # adj matrix, distance matrix

    else:  # distance_df_file: node id starts from zero
        with open(distance_df_filename, 'r') as f:
            f.readline()
            reader = csv.reader(f)
            for row in reader:
                if len(row) != 3:
                    continue
                i, j, distance = int(row[0]), int(row[1]), float(row[2])
                A[i, j] = 1
                distaneA[i, j] = max_dict[dataset]/distance
                if not direction:
                    A[j, i] = 1
                    distaneA[j, i] = max_dict[dataset]/distance
        return A, distaneA
