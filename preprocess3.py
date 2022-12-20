''' 코드 내용은 아래와 같음 '''
# 1. 표준화된 데이터를 불러와 가우시안 필터 적용하여 저장하기


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from scipy.ndimage import gaussian_filter1d

# 1) FD002의 표준화된 regime1~6 데이터 불러오기
FD002 = np.loadtxt('./04_FD002_Reg_Std/FD002_Reg_Std_data.txt',delimiter='\t',dtype='double')
x_columns_ = list(['unit','timestep','set1','set2','set3',
                    'sensor1','sensor2','sensor3',
                    'sensor4','sensor5','sensor6',
                    'sensor7','sensor8','sensor9',
                    'sensor10','sensor11','sensor12',
                    'sensor13','sensor14','sensor15',
                    'sensor16','sensor17','sensor18',
                    'sensor19','sensor20','sensor21',
                    'type','regime','target']) # type -> train : 0, test : 1

FD002_df = pd.DataFrame(FD002,columns=x_columns_)

directory = '06_FD002_Reg_Std_Gaussian_data'
if os.path.isdir(directory):
    print(directory+'폴더 있음')
else:
    print(directory+'폴더 없음, 생성함')
    os.makedirs(directory)

for j in range(1,22): # sensor 총 개수 만큼 반복
    FD002_df['sensor'+str(j)] = gaussian_filter1d(FD002_df['sensor'+str(j)],1) # 필터링한 데이터로 교체
    
filepath = './'+directory+'/FD002_Reg_Std_Gaussian_data.txt' # 경로 지정
np.savetxt(filepath,FD002_df,delimiter='\t',newline='\n',fmt='%1.6e') # 저장

