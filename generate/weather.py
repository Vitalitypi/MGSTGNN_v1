import numpy as np
import csv




def generate_weather(DATASET):
    # get basic flow data
    flow_data = np.load('../dataset/{}/{}.npz'.format(DATASET,DATASET))['data']
    flow_data = flow_data[...,:1]
    print(flow_data.shape)
    time_stamps,num_nodes,dim = flow_data.shape
    min_max = get_min_max(DATASET)
    print("The maximums are: ",min_max[1])
    weas = None
    with open('../dataset/{}/weather.csv'.format(DATASET), 'r', encoding='gbk') as f:
            f.readline()
            reader = csv.reader(f)
            for rows in reader:
                temp,wind,prec = float(rows[0]),float(rows[1]),float(rows[2])
                wea = np.empty((288, num_nodes, 3))
                wea[0].fill(temp/min_max[1][0])
                wea[1].fill(wind/min_max[1][1])
                wea[2].fill(prec/min_max[1][2])
                if weas is None:
                    weas = wea
                else:
                    weas = np.concatenate([weas,wea],axis=0)
                if weas.shape[0]==time_stamps:
                    break
    print('Finished! The data shape is: ',weas.shape)
    np.savez('../dataset/{}/weather.npz'.format(DATASET), data=weas)

def test_weather(DATASET):
    info = np.load('../dataset/{}/weather.npz'.format(DATASET))['data']
    for i in range(info.shape[0]//288):
        for j in range(info.shape[1]):
            print(info[i*288,j])
            break

def get_min_max(DATASET):
    with open('../dataset/{}/weather.csv'.format(DATASET), 'r', encoding='gbk') as f:
            f.readline()
            reader = csv.reader(f)
            mx1,mx2,mx3 = -1,-1,-1
            mn1,mn2,mn3 = 10000,10000,10000
            for rows in reader:
                temp,wind,prec = float(rows[0]),float(rows[1]),float(rows[2])
                if temp>mx1:
                    mx1 = temp
                if temp<mn1:
                    mn1 = temp
                if wind>mx2:
                    mx2 = wind
                if wind<mn2:
                    mn2 = wind
                if prec>mx3:
                    mx3 = prec
                if prec<mn3:
                    mn3 = prec
            print("The minimums are: ",mn1,mn2,mn3)
            print('The maximums are: ',mx1,mx2,mx3)
    return [[mn1,mn2,mn3],[mx1,mx2,mx3]]

# generate_weather('PEMS04')
