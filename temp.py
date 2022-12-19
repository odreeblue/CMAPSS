''' 코드 내용은 아래와 같음 '''

## 1. 가우시안 필터 적용 전에 Engine2_Regime1_data.txt 뽑아내기

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
# from ipywidgets import interact
from time import time


train_FD002 = np.loadtxt("./10_Traindata/train_FD002.txt", dtype = 'float', delimiter=" ")

print(np.shape(train_FD002))


x_columns_ = list(['unit','timestep','set1','set2','set3','sensor1','sensor2','sensor3',
                    'sensor4','sensor5','sensor6',
                    'sensor7','sensor8','sensor9',
                    'sensor10','sensor11','sensor12',
                    'sensor13','sensor14','sensor15',
                    'sensor16','sensor17','sensor18',
                    'sensor19','sensor20','sensor21'])


train_FD002_df = pd.DataFrame(train_FD002,columns=x_columns_)
#train_FD002_df_timestep = train_FD002_df['timestep']
print(train_FD002_df['timestep'].max())
