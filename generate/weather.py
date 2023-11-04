import numpy as np
import csv

HOP_K = 1
DATASET = 'PEMS04'

# get basic flow data
flow_data = np.load('../dataset/{}/{}.npz'.format(DATASET,DATASET))['data']
flow_data = flow_data[...,:1]
print(flow_data.shape)
time_stamps,num_nodes,dim = flow_data.shape

def generate_weather():
    min_max = get_min_max()
    print("The maximums are: ",min_max[1])
    weas = []
    with open('../dataset/{}/weather.csv'.format(DATASET), 'r', encoding='gbk') as f:
            f.readline()
            reader = csv.reader(f)
            for rows in reader:
                temp,wind,prec = float(rows[0]),float(rows[1]),float(rows[2])
                for i in range(288):
                    wea = []
                    for j in range(num_nodes):
                        wea.append([temp/min_max[1][0],wind/min_max[1][1],prec/min_max[1][2]])
                    weas.append(wea)

    weas = np.array(weas)
    print('Finished! The data shape is: ',weas.shape)
    np.savez('../dataset/{}/weather.npz'.format(DATASET), data=weas)

def test_weather():
    info = np.load('../dataset/{}/weather.npz'.format(DATASET))['data']
    for i in range(info.shape[0]//288):
        for j in range(info.shape[1]):
            print(info[i*288,j])
            break

def get_min_max():
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

generate_weather()
