''' 코드 내용은 아래와 같음 '''

## 1. 데이터 표준화하기

'''
논문    My      센서
1       1       [2,3,4,6,7,8,9,11,13,14,15,17,21]
2       3       [2,3,4,7,8,9,11,12,13,14,15,17,20]
3       4       [2,3,4,7,8,9,11,12,13,14,15,17,20,21]
4       5       [2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,20,21]
5       2       [2,3,4,6,7,8,9,11,12,13,14,15,17,20,21]
6       6       [2,3,4,7,8,9,11,12,13,14,15,17,20]
'''

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
# from ipywidgets import interact
from time import time
from sklearn.cluster import KMeans
from sklearn import preprocessing
from sklearn.metrics import silhouette_score
import scipy.stats as stats
import math
import seaborn as sn


print('--------------------------------------')
print('-------------데이터 로드--------------')
FD002 = np.loadtxt('./01_FD002_Reg_data/FD002_Reg_data.txt',delimiter='\t',dtype='float')
x_columns_ = list(['unit','timestep','set1','set2','set3',
                    'sensor1','sensor2','sensor3',
                    'sensor4','sensor5','sensor6',
                    'sensor7','sensor8','sensor9',
                    'sensor10','sensor11','sensor12',
                    'sensor13','sensor14','sensor15',
                    'sensor16','sensor17','sensor18',
                    'sensor19','sensor20','sensor21',
                    'type','regime','target']) # type -> train : 0, test : 1

sensorlist = list(['sensor1','sensor2','sensor3',
                    'sensor4','sensor5','sensor6',
                    'sensor7','sensor8','sensor9',
                    'sensor10','sensor11','sensor12',
                    'sensor13','sensor14','sensor15',
                    'sensor16','sensor17','sensor18',
                    'sensor19','sensor20','sensor21'])

FD002_df = pd.DataFrame(FD002,columns=x_columns_)
print(FD002_df)
FD002_df = FD002_df[FD002_df['type']==0]
print(FD002_df)
# 2) 데이터 Standardization
print('--------------------------------------')
print('-------------데이터 표준화--------------')
x_columns_2 = list(['sensor1','sensor2','sensor3',
                    'sensor4','sensor5','sensor6',
                    'sensor7','sensor8','sensor9',
                    'sensor10','sensor11','sensor12',
                    'sensor13','sensor14','sensor15',
                    'sensor16','sensor17','sensor18',
                    'sensor19','sensor20','sensor21'])

from sklearn.preprocessing import StandardScaler
std_scaler = StandardScaler()

    # 3.0)  Call 'fit' with appropriate arguments before using this estimator.
fitted = std_scaler.fit(FD002_df[x_columns_2])
print(fitted)

    # 3.1) 데이터 표준화 변환
FD002_std_df = std_scaler.transform(FD002_df[x_columns_2],with_mean=False,with_std=False)
    
    # 3.2) numpy to Pandas

FD002_std_df = pd.DataFrame(FD002_std_df,columns=x_columns_2)

print(type(FD002_std_df))

for sensorNum in x_columns_2:
    FD002_df[sensorNum] = FD002_std_df[sensorNum]

import os
from scipy.ndimage import gaussian_filter1d
'''
print('--------------------------------------')
print('-------------04번 폴더에 표준화된 데이터 저장--------------')


directory = '04_FD002_Reg_Std'
if os.path.isdir(directory):
    print(directory+'폴더 있음')
else:
    print(directory+'폴더 없음, 생성함')
    os.makedirs(directory)


filepath = './'+directory+'/FD002_Reg_Std_data.txt'
np.savetxt(filepath,FD002_df,delimiter='\t',newline='\n',fmt='%1.6e')

'''
print(FD002_df)
print(FD002_df['sensor2'].max())
print(FD002_df['sensor2'].min())

print('--------------------------------------')
print('-------------표준화된 데이터에 가우시안 적용해서 05번 폴더의 img 넣기--------------')

sensorlist = [2,3,4,9,11,15,17] # 논문 상 Spearman value가 0.4 이상이 센서 리스트

directory = '05_FD002_std_STD_Gaussian_img'
if os.path.isdir(directory):
    print(directory+'폴더 있음')
else:
    print(directory+'폴더 없음, 생성함')
    os.makedirs(directory)


# x = Engine2_Reg6_std_df['timestep']
x = FD002_df[(FD002_df['unit'] == 2) &(FD002_df['type']==0) & (FD002_df['regime']==0)]['timestep']
for i in sensorlist:
    # Engine2_Reg1_sensornum = Engine2_Reg6_std_df['sensor'+str(i)]
    Engine2_Reg1_sensornum = FD002_df[(FD002_df['unit']==2)&(FD002_df['type']==0) & (FD002_df['regime']==0)]['sensor'+str(i)]
    Engine2_Reg1_sensornum_filter = gaussian_filter1d(Engine2_Reg1_sensornum,1)
    plt.plot(x,Engine2_Reg1_sensornum,'ok',ms = 2,label='original data') # ms = markersize
    plt.plot(x,Engine2_Reg1_sensornum_filter,'-r',label='filtered data, sigma =1')
    plt.title('sensor #'+str(i))
    plt.legend()
    plt.grid()
    plt.savefig('./'+directory+'/Engine2_Reg1_STD_Gaussian_sensor'+str(i)+'_2.png',bbox_inches='tight')
    plt.cla()
