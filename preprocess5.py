''' 코드 내용은 아래와 같음 '''
# 1. 표준화된 데이터를 불러와 가우시안 필터 적용하여 저장하기

'''
논문    My      센서
1       6       [2,3,4,9,11,15,17]
2       3       [2,3,4,9,11,15,17]
3       4       [2,3,4,6,7,8,9,11,12,13,15,16,17,20,21]
4       5       [2,3,4,7,8,9,11,12,13,15,17,20,21]
5       1       [2,3,4,9,11,15,17]
6       2       [2,3,4,6,7,8,9,11,12,13,15,17,20,21]
'''
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from scipy.ndimage import gaussian_filter1d

# 1) 2번 엔진의 표준화된 regime1~6 데이터 불러오기
Engine2_Reg1_std = np.loadtxt('./05_Engine2_RegimeData_Standardization/Engine2_Reg1_std.txt',delimiter='\t')
Engine2_Reg2_std = np.loadtxt('./05_Engine2_RegimeData_Standardization/Engine2_Reg2_std.txt',delimiter='\t')
Engine2_Reg3_std = np.loadtxt('./05_Engine2_RegimeData_Standardization/Engine2_Reg3_std.txt',delimiter='\t')
Engine2_Reg4_std = np.loadtxt('./05_Engine2_RegimeData_Standardization/Engine2_Reg4_std.txt',delimiter='\t')
Engine2_Reg5_std = np.loadtxt('./05_Engine2_RegimeData_Standardization/Engine2_Reg5_std.txt',delimiter='\t')
Engine2_Reg6_std = np.loadtxt('./05_Engine2_RegimeData_Standardization/Engine2_Reg6_std.txt',delimiter='\t')

# 2) numpy to Pandas 변환하기
x_columns_ = list(['unit','timestep','sensor1','sensor2','sensor3',
                    'sensor4','sensor5','sensor6',
                    'sensor7','sensor8','sensor9',
                    'sensor10','sensor11','sensor12',
                    'sensor13','sensor14','sensor15',
                    'sensor16','sensor17','sensor18',
                    'sensor19','sensor20','sensor21'])
Engine2_Reg_std_df = dict()
Engine2_Reg_std_df[1] = pd.DataFrame(Engine2_Reg1_std,columns=x_columns_)
Engine2_Reg_std_df[2] = pd.DataFrame(Engine2_Reg2_std,columns=x_columns_)
Engine2_Reg_std_df[3] = pd.DataFrame(Engine2_Reg3_std,columns=x_columns_)
Engine2_Reg_std_df[4] = pd.DataFrame(Engine2_Reg4_std,columns=x_columns_)
Engine2_Reg_std_df[5] = pd.DataFrame(Engine2_Reg5_std,columns=x_columns_)
Engine2_Reg_std_df[6] = pd.DataFrame(Engine2_Reg6_std,columns=x_columns_)


# sensorlist = [2,3,4,9,11,15,17] # 논문 상 Spearman value가 0.4 이상이 센서 리스트
sensorlist = dict()
sensorlist[1] = [2,3,4,9,11,15,17]
sensorlist[2] = [2,3,4,9,11,15,17]
sensorlist[3] = [2,3,4,6,7,8,9,11,12,13,15,16,17,20,21]
sensorlist[4] = [2,3,4,7,8,9,11,12,13,15,17,20,21]
sensorlist[5] = [2,3,4,9,11,15,17]
sensorlist[6] = [2,3,4,6,7,8,9,11,12,13,15,17,20,21]
directory = '07_Engine2_RegimeData_STD_Gaussian_Data'
if os.path.isdir(directory):
    print(directory+'폴더 있음')
else:
    print(directory+'폴더 없음, 생성함')
    os.makedirs(directory)

'''
# x = Engine2_Reg6_std_df['timestep']
x = Engine2_Reg6_std_df[Engine2_Reg6_std_df['unit'] == 2]['timestep']
for i in sensorlist:
    # Engine2_Reg1_sensornum = Engine2_Reg6_std_df['sensor'+str(i)]
    Engine2_Reg1_sensornum = Engine2_Reg6_std_df[Engine2_Reg6_std_df['unit']==2]['sensor'+str(i)]
    Engine2_Reg1_sensornum_filter = gaussian_filter1d(Engine2_Reg1_sensornum,1)
    plt.plot(x,Engine2_Reg1_sensornum,'ok',ms = 2,label='original data') # ms = markersize
    plt.plot(x,Engine2_Reg1_sensornum_filter,'-r',label='filtered data, sigma =1')
    plt.title('sensor #'+str(i))
    plt.legend()
    plt.grid()
    plt.savefig('./'+directory+'/Engine2_Reg1_STD_Gaussian_sensor'+str(i)+'.png',bbox_inches='tight')
    plt.cla()
'''

traindata = dict()

for i in range(1,7): # regime 1~6까지 반복
    for j in range(1,22): # sensor 총 개수 만큼 반복
        Engine2_Reg_std_df[i]['sensor'+str(j)] = gaussian_filter1d(Engine2_Reg_std_df[i]['sensor'+str(j)],1) # 필터링한 데이터로 교체
    filepath = './'+directory+'/Engine2_Reg'+str(i)+'_std_gaussian.txt' # 경로 지정
    np.savetxt(filepath,Engine2_Reg_std_df[i],delimiter='\t',newline='\n') # 저장

        
        
