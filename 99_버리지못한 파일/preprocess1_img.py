''' 코드 내용은 아래와 같음 '''

## 1. k-means 군집화 알고리즘 적용
#  1.1 K-means 군집화 알고리즘 적용
#  1.2 논문과 같은 사진(그래프) 도출


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
# from ipywidgets import interact
from time import time
print('--------------------------------------')
print('-------------데이터 로드 및 Pandas 변환---------------')
FD002 = np.loadtxt('./01_FD002_Reg_data/FD002_Reg_data.txt',delimiter='\t',dtype='double')
x_columns_ = list(['unit','timestep','set1','set2','set3',
                    'sensor1','sensor2','sensor3',
                    'sensor4','sensor5','sensor6',
                    'sensor7','sensor8','sensor9',
                    'sensor10','sensor11','sensor12',
                    'sensor13','sensor14','sensor15',
                    'sensor16','sensor17','sensor18',
                    'sensor19','sensor20','sensor21',
                    'type','regime']) # type -> train : 0, test : 1
FD002_df = pd.DataFrame(FD002,columns=x_columns_)

print('--------------------------------------')
print('-------------데이터 크기--------------')
row = np.shape(FD002)[0]
col = np.shape(FD002)[1]
print("row : ",row," col : ", col)



print('--------------------------------------')
print('-------------image 생성을 위한 조건식 부여--------------')
# Engine2_Reg6_std_df[Engine2_Reg6_std_df['unit']==2 ]['sensor'+str(i)]
regime = dict()
regime[1] = FD002_df[(FD002_df['unit']==2) & (FD002_df['regime']==0) * (FD002_df['type']==0)]
regime[2] = FD002_df[(FD002_df['unit']==2) & (FD002_df['regime']==1) * (FD002_df['type']==0)]
regime[3] = FD002_df[(FD002_df['unit']==2) & (FD002_df['regime']==2) * (FD002_df['type']==0)]
regime[4] = FD002_df[(FD002_df['unit']==2) & (FD002_df['regime']==3) * (FD002_df['type']==0)]
regime[5] = FD002_df[(FD002_df['unit']==2) & (FD002_df['regime']==4) * (FD002_df['type']==0)]
regime[6] = FD002_df[(FD002_df['unit']==2) & (FD002_df['regime']==5) * (FD002_df['type']==0)]

# print(regime[1].shape)
print('--------------------------------------')
print('-------------image 저장--------------')
import os.path
directory = '02_FD002_Sensor2_img'
if os.path.isdir(directory):
    print(directory+'폴더 있음')
else:
    print(directory+'폴더 없음, 생성함')
    os.makedirs(directory)

#print(np.shape(regime1))
for i in range(1,6+1):
    plt.cla()
    plt.plot(regime[i]['timestep'],regime[i]['sensor2'])
    plt.title('regime'+str(i))
    plt.savefig('./02_FD002_Sensor2_img/regime'+str(i)+'.png', bbox_inches='tight') #논문엔 1번 센서라고 나왔지만, 2번 센서임