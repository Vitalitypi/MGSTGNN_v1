import csv

import numpy as np
import torch

from utils.norm import StandardScaler


# For PEMS03/04/07/08 Datasets
def get_dataloader_pems(dataset, batch_size=64, val_ratio=0.2, test_ratio=0.2, in_steps=12, out_steps=12,
                        flow_dim=1, period_dim=1, weekend_dim=1, holiday_dim=1, hop_dim=1, weather_dim=1):
    # load flow data
    flow = np.load('./dataset/{}/{}.npz'.format(dataset, dataset))['data'][..., :flow_dim]
    # load period data
    period = np.load('./dataset/{}/period.npz'.format(dataset))['data'][..., :period_dim]
    # load weekend data
    weekend = np.load('./dataset/{}/weekend.npz'.format(dataset))['data'][..., :weekend_dim]
    # load holiday data
    holiday = np.load('./dataset/{}/holiday.npz'.format(dataset))['data'][..., :holiday_dim]
    # load hops data
    hop = np.load('./dataset/{}/hop.npz'.format(dataset))['data'][...,:hop_dim]
    # load weather data
    weather = np.load('./dataset/{}/weather.npz'.format(dataset))['data'][...,:weather_dim]
    # concatenate all data
    data = np.concatenate([flow,period,weekend,holiday,hop,weather],axis=-1)
    print(data.shape)
    # normalize data(only normalize the first dimension data)
    mean = data[..., 0].mean()
    std = data[..., 0].std()
    scaler = StandardScaler(mean, std)
    data[..., 0] = scaler.transform(data[..., 0])
    # spilit dataset by days or by ratio
    data_train, data_val, data_test = split_data_by_ratio(data, val_ratio, test_ratio)
    # add time window [B, N, 1]
    x_tra, y_tra = Add_Window_Horizon(data_train, in_steps, out_steps)
    x_val, y_val = Add_Window_Horizon(data_val, in_steps, out_steps)
    x_test, y_test = Add_Window_Horizon(data_test, in_steps, out_steps)

    print('Train: ', x_tra.shape, y_tra.shape)
    print('Val: ', x_val.shape, y_val.shape)
    print('Test: ', x_test.shape, y_test.shape)

    ##############get dataloader######################
    train_dataloader = data_loader(x_tra, y_tra, batch_size, shuffle=True, drop_last=True)
    if len(x_val) == 0:
        val_dataloader = None
    else:
        val_dataloader = data_loader(x_val, y_val, batch_size, shuffle=False, drop_last=True)
    test_dataloader = data_loader(x_test, y_test, batch_size, shuffle=False, drop_last=False)

    return train_dataloader, val_dataloader, test_dataloader, scaler

# For PEMS-Bay and METR-LA Datasets
def get_dataloader_meta_la(args, normalizer='std', tod=False, dow=False, weather=False, single=True):
    data = {}
    for category in ['train', 'val', 'test']:
        cat_data = np.load(os.path.join("./dataset", args.dataset, category + '.npz'))
        data['x_' + category] = cat_data['x'] # [B, T, N, 2]
        data['y_' + category] = np.expand_dims(cat_data['y'][:, :, :, 0], axis=-1) # [B, T, N, 1]

    # data normalization method following DCRNN
    scaler = StandardScaler(mean=data['x_train'][..., 0].mean(), std=data['x_train'][..., 0].std())
    for category in ['train', 'val', 'test']:
        data['x_' + category][:, :, :, 0] = scaler.transform(data['x_' + category][:, :, :, 0])
    if not args.real_value:
        data['y_' + category][:, :, :, 0] = scaler.transform(data['y_' + category][:, :, :, 0])

    x_tra, y_tra = data['x_train'], data['y_train']
    x_val, y_val = data['x_val'], data['y_val']
    x_test, y_test = data['x_test'], data['y_test']

    print('Train: ', x_tra.shape, y_tra.shape)
    print('Val: ', x_val.shape, y_val.shape)
    print('Test: ', x_test.shape, y_test.shape)
    # print(x_tra[:10], x_val[:10], x_test[:10])
    # print(y_tra[:10], y_val[:10], y_test[:10])

    ##############get dataloader######################
    train_dataloader = data_loader(x_tra, y_tra, args.batch_size, shuffle=True, drop_last=True)
    if len(x_val) == 0:
        val_dataloader = None
    else:
        val_dataloader = data_loader(x_val, y_val, args.batch_size, shuffle=False, drop_last=True)
    test_dataloader = data_loader(x_test, y_test, args.batch_size, shuffle=False, drop_last=False)

    return train_dataloader, val_dataloader, test_dataloader, scaler


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
    distance_df_filename = './dataset/{}/{}.csv'.format(dataset, dataset)
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

        return A, distaneA  # adj matrix, distance matrix

    else:  # distance_df_file: node id starts from zero
        with open(distance_df_filename, 'r') as f:
            f.readline()
            reader = csv.reader(f)
            for row in reader:
                if len(row) != 3:
                    continue
                i, j, distance = int(row[0]), int(row[1]), float(row[2])
                A[i, j] = 1
                distaneA[i, j] = max_dict[dataset] / distance
                if not direction:
                    A[j, i] = 1
                    distaneA[j, i] = max_dict[dataset] / distance
        return A, distaneA


def split_data_by_ratio(data, val_ratio, test_ratio):
    data_len = data.shape[0]
    test_data = data[-int(data_len * test_ratio):]
    val_data = data[-int(data_len * (test_ratio + val_ratio)):-int(data_len * test_ratio)]
    train_data = data[:-int(data_len * (test_ratio + val_ratio))]

    return train_data, val_data, test_data


def Add_Window_Horizon(data, window=12, horizon=12):
    '''
    :param data: shape [B, N, D]
    :param window:
    :param horizon:
    :return: X is [B', W, N, D], Y is [B', H, N, D], B' = B - W - H + 1
    '''
    length = len(data)
    total_num = length - horizon - window + 1
    X = []  # windows
    Y = []  # horizon
    index = 0
    while index < total_num:
        X.append(data[index:index + window])
        Y.append(data[index + window:index + window + horizon, :, :1])
        index = index + 1
    X = np.array(X)
    Y = np.array(Y)

    return X, Y


def data_loader(X, Y, batch_size, shuffle=True, drop_last=True):
    cuda = True if torch.cuda.is_available() else False
    TensorFloat = torch.cuda.FloatTensor if cuda else torch.FloatTensor
    X, Y = TensorFloat(X), TensorFloat(Y)

    data = torch.utils.data.TensorDataset(X, Y)
    dataloader = torch.utils.data.DataLoader(data,
                                             batch_size=batch_size,
                                             shuffle=shuffle,
                                             drop_last=drop_last)
    return dataloader


def norm_adj(W):
    '''
    compute  normalized Adj matrix

    Parameters
    ----------
    W: np.ndarray, shape is (N, N), N is the num of vertices

    Returns
    ----------
    normalized Adj matrix: (D^hat)^{-1} A^hat; np.ndarray, shape (N, N)
    '''
    assert W.shape[0] == W.shape[1]

    N = W.shape[0]
    W = W + np.identity(N)  # add self-connection
    D = np.diag(1.0 / np.sum(W, axis=1))
    norm_Adj_matrix = np.dot(D, W)

    return norm_Adj_matrix
