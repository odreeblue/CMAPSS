''' 코드 내용은 아래와 같음 '''
# 1. Regime6(논문:regime1)에 대해 가우시안 필터 적용해서 이미지 도출만 하기
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from scipy.ndimage import gaussian_filter1d

FD002 = np.loadtxt('./04_FD002_Reg_Std/FD002_Reg_Std_data.txt',delimiter='\t',dtype='double')
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

sensorlist = [2,3,4,9,11,15,17] # 논문 상 Spearman value가 0.4 이상이 센서 리스트

directory = '05_FD002_std_STD_Gaussian_img'
if os.path.isdir(directory):
    print(directory+'폴더 있음')
else:
    print(directory+'폴더 없음, 생성함')
    os.makedirs(directory)


# x = Engine2_Reg6_std_df['timestep']
x = FD002_df[(FD002_df['unit'] == 2) &(FD002_df['type']==0) & (FD002_df['regime']==1)]['timestep']
for i in sensorlist:
    # Engine2_Reg1_sensornum = Engine2_Reg6_std_df['sensor'+str(i)]
    Engine2_Reg1_sensornum = FD002_df[(FD002_df['unit']==2)&(FD002_df['type']==0) & (FD002_df['regime']==1)]['sensor'+str(i)]
    Engine2_Reg1_sensornum_filter = gaussian_filter1d(Engine2_Reg1_sensornum,1)
    plt.plot(x,Engine2_Reg1_sensornum,'ok',ms = 2,label='original data') # ms = markersize
    plt.plot(x,Engine2_Reg1_sensornum_filter,'-r',label='filtered data, sigma =1')
    plt.title('sensor #'+str(i))
    plt.legend()
    plt.grid()
    plt.savefig('./'+directory+'/Engine2_Reg1_STD_Gaussian_sensor'+str(i)+'.png',bbox_inches='tight')
    plt.cla()