import holidays
import numpy as np
import datetime as dt
from datetime import datetime

def is_holiday(date):
    us_holidays = holidays.US()
    return date in us_holidays

def generate_holiday(DATASET):
    # get basic flow data
    flow_data = np.load('../dataset/{}/{}.npz'.format(DATASET,DATASET))['data']
    flow_data = flow_data[...,:1]
    print(flow_data.shape)
    time_stamps,num_nodes,dim = flow_data.shape

    day_dict = {
        'PEMS03':'2018-09-01',
        'PEMS04':'2018-01-01',
        'PEMS07':'2017-05-01',
        'PEMS08':'2016-07-01'
    }
    start_date = datetime.strptime(day_dict[DATASET], "%Y-%m-%d")
    print('the start date of dataset: ', start_date)
    res = np.zeros((time_stamps,num_nodes,1))
    current_date = start_date
    for i in range(time_stamps//288):
        date_info = np.zeros((288,num_nodes))
        if is_holiday(current_date):
            date_info = np.ones((288,num_nodes))
            print("This is a holiday! ", current_date)

        res[i*288:(i+1)*288,:,0] = date_info

        current_date = current_date + dt.timedelta(days=1)
    # 保存文件
    np.savez('../dataset/{}/holiday.npz'.format(DATASET), data=res)

# generate_holiday('PEMS04')
