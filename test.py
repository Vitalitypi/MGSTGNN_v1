import torch
import requests

import numpy as np
#
# data = np.load('../DSTAGNN/DSTAGNN/data/PEMS07/PEMS07.npz')['data']
# print(data.shape)
# print(data[0,0,:],data[-5:,0,0])
# print(data[285:295,0])
# [246. 249. 243. 232. 243.] [333. 305. 358. 309. 278.]
# [246. 249. 243. 232. 243.] [333. 305. 358. 309. 278.]
def down_load_data():
    url = 'https://pems.dot.ca.gov/?download=301759&dnode=Clearinghouse'
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                             "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/54.0.2840.99 Safari/537.36"}
    data = {"redirect": "", "username": "779958365@qq.com",
            "password": "treei+709o", "login": "Login"}
    session = requests.session()
    response = session.post(url, headers=headers, data=data)
    response = session.get(url)
down_load_data()
# https://pems.dot.ca.gov/?download=301759&dnode=Clearinghouse
# https://pems.dot.ca.gov/?download=301736&dnode=Clearinghouse
# https://pems.dot.ca.gov/?download=301760&dnode=Clearinghouse
# https://pems.dot.ca.gov/?download=301739&dnode=Clearinghouse
# https://pems.dot.ca.gov/?download=301752&dnode=Clearinghouse
# https://pems.dot.ca.gov/?download=301756&dnode=Clearinghouse
# https://pems.dot.ca.gov/?download=301731&dnode=Clearinghouse
# https://pems.dot.ca.gov/?download=301750&dnode=Clearinghouse
# https://pems.dot.ca.gov/?download=301758&dnode=Clearinghouse
# https://pems.dot.ca.gov/?download=301741&dnode=Clearinghouse
# https://pems.dot.ca.gov/?download=301749&dnode=Clearinghouse
# https://pems.dot.ca.gov/?download=301737&dnode=Clearinghouse
# https://pems.dot.ca.gov/?download=301747&dnode=Clearinghouse
# https://pems.dot.ca.gov/?download=301748&dnode=Clearinghouse
# https://pems.dot.ca.gov/?download=301757&dnode=Clearinghouse
# https://pems.dot.ca.gov/?download=301732&dnode=Clearinghouse
# https://pems.dot.ca.gov/?download=301734&dnode=Clearinghouse
# https://pems.dot.ca.gov/?download=301755&dnode=Clearinghouse
# https://pems.dot.ca.gov/?download=301746&dnode=Clearinghouse
# https://pems.dot.ca.gov/?download=301745&dnode=Clearinghouse
# https://pems.dot.ca.gov/?download=301754&dnode=Clearinghouse
# https://pems.dot.ca.gov/?download=301742&dnode=Clearinghouse
# https://pems.dot.ca.gov/?download=301744&dnode=Clearinghouse
# https://pems.dot.ca.gov/?download=301738&dnode=Clearinghouse
# https://pems.dot.ca.gov/?download=301753&dnode=Clearinghouse
# https://pems.dot.ca.gov/?download=301740&dnode=Clearinghouse
# https://pems.dot.ca.gov/?download=301751&dnode=Clearinghouse
# https://pems.dot.ca.gov/?download=301761&dnode=Clearinghouse
# https://pems.dot.ca.gov/?download=301733&dnode=Clearinghouse
# https://pems.dot.ca.gov/?download=301743&dnode=Clearinghouse
# https://pems.dot.ca.gov/?download=301735&dnode=Clearinghouse
# https://pems.dot.ca.gov/?download=301928&dnode=Clearinghouse
# https://pems.dot.ca.gov/?download=301951&dnode=Clearinghouse
# https://pems.dot.ca.gov/?download=301929&dnode=Clearinghouse
# https://pems.dot.ca.gov/?download=301937&dnode=Clearinghouse
# https://pems.dot.ca.gov/?download=301948&dnode=Clearinghouse
# https://pems.dot.ca.gov/?download=301956&dnode=Clearinghouse
# https://pems.dot.ca.gov/?download=301930&dnode=Clearinghouse
# https://pems.dot.ca.gov/?download=301940&dnode=Clearinghouse
# https://pems.dot.ca.gov/?download=301944&dnode=Clearinghouse
# https://pems.dot.ca.gov/?download=301933&dnode=Clearinghouse
# https://pems.dot.ca.gov/?download=301942&dnode=Clearinghouse
# https://pems.dot.ca.gov/?download=301947&dnode=Clearinghouse
# https://pems.dot.ca.gov/?download=301954&dnode=Clearinghouse
# https://pems.dot.ca.gov/?download=301952&dnode=Clearinghouse
# https://pems.dot.ca.gov/?download=301950&dnode=Clearinghouse
# https://pems.dot.ca.gov/?download=301953&dnode=Clearinghouse
# https://pems.dot.ca.gov/?download=301939&dnode=Clearinghouse
# https://pems.dot.ca.gov/?download=301938&dnode=Clearinghouse
# https://pems.dot.ca.gov/?download=301931&dnode=Clearinghouse
# https://pems.dot.ca.gov/?download=301946&dnode=Clearinghouse
# https://pems.dot.ca.gov/?download=301926&dnode=Clearinghouse
# https://pems.dot.ca.gov/?download=301936&dnode=Clearinghouse
# https://pems.dot.ca.gov/?download=301927&dnode=Clearinghouse
# https://pems.dot.ca.gov/?download=301941&dnode=Clearinghouse
# https://pems.dot.ca.gov/?download=301935&dnode=Clearinghouse
# https://pems.dot.ca.gov/?download=301932&dnode=Clearinghouse
# https://pems.dot.ca.gov/?download=301945&dnode=Clearinghouse
# https://pems.dot.ca.gov/?download=301949&dnode=Clearinghouse
# https://pems.dot.ca.gov/?download=301955&dnode=Clearinghouse
# https://pems.dot.ca.gov/?download=301934&dnode=Clearinghouse
# https://pems.dot.ca.gov/?download=301943&dnode=Clearinghouse
