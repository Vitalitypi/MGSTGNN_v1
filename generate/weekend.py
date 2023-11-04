import numpy as np
import datetime as dt
from datetime import datetime

day_dict = {
    'PEMS03':'2018-09-01',
    'PEMS04':'2018-01-01',
    'PEMS07':'2017-05-01',
    'PEMS08':'2016-07-01'
}
def generate_weekend(DATASET):
    # get basic flow data
    flow_data = np.load('../dataset/{}/{}.npz'.format(DATASET,DATASET))['data']
    flow_data = flow_data[...,:1]
    print(flow_data.shape)
    time_stamps,num_nodes,dim = flow_data.shape
    start_date = datetime.strptime(day_dict[DATASET], "%Y-%m-%d")
    print('the start date of dataset: ', start_date)

    res = np.zeros((time_stamps,num_nodes,1))
    current_date = start_date
    for i in range(time_stamps//288):
        date_info = np.zeros((288,num_nodes))
        if current_date.weekday()>=5:
            date_info = np.ones((288,num_nodes))
            print("This is a weekend! ", current_date)

        res[i*288:(i+1)*288,:,0] = date_info

        current_date = current_date + dt.timedelta(days=1)
    # 保存文件
    np.savez('../dataset/{}/weekend.npz'.format(DATASET), data=res)

# generate_weekend('PEMS04')
