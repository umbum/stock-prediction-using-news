#-*- coding: cp949 -*-
#-*- coding: utf-8 -*- 

import numpy as np



data_temp = np.load('./kospi_data.npy', allow_pickle=True)
print(data_temp.shape)
print(data_temp[32,:])


