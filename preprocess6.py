''' 코드 내용은 아래와 같음 '''
# 1. FD002 데이터 불러온 뒤 Unit 별 최대 Cycle수(=고장 시점)를 구하기
# 2. 최대 Cycle수를 활용하여 Target 값 구하기 = 최대 Cycle 수 - timestep값

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from scipy.ndimage import gaussian_filter1d



# 1) FD002 데이터 불러오기
train_FD002 = np.loadtxt("./10_Traindata/train_FD002.txt", dtype = 'float', delimiter=" ")

# 2) FD002 데이터의 Unit 열과 Timestep 열을 DataFrame으로 변환
columns = list(['unit','timestep']) 
data = pd.DataFrame(train_FD002[:,0:2],columns=columns)

# 3) 최대 Unit 번호 구하기
maxUnitNum = data['unit'].max() # maybe 260개

# 4) Unit별 최대 사이클수 구하기
maxCycle = dict()
for i in range(1,int(maxUnitNum)+1):
    maxCycle[i] = data[data['unit']==i].max()['timestep']


# 5) 표준화 및 필터링된 데이터 로딩
Engine2_Reg_std_gaussian=dict()
Engine2_Reg_std_gaussian[1] = np.loadtxt('./07_Engine2_RegimeData_STD_Gaussian_Data/Engine2_Reg1_std_gaussian.txt',delimiter='\t')
Engine2_Reg_std_gaussian[2] = np.loadtxt('./07_Engine2_RegimeData_STD_Gaussian_Data/Engine2_Reg2_std_gaussian.txt',delimiter='\t')
Engine2_Reg_std_gaussian[3] = np.loadtxt('./07_Engine2_RegimeData_STD_Gaussian_Data/Engine2_Reg3_std_gaussian.txt',delimiter='\t')
Engine2_Reg_std_gaussian[4] = np.loadtxt('./07_Engine2_RegimeData_STD_Gaussian_Data/Engine2_Reg4_std_gaussian.txt',delimiter='\t')
Engine2_Reg_std_gaussian[5] = np.loadtxt('./07_Engine2_RegimeData_STD_Gaussian_Data/Engine2_Reg5_std_gaussian.txt',delimiter='\t')
Engine2_Reg_std_gaussian[6] = np.loadtxt('./07_Engine2_RegimeData_STD_Gaussian_Data/Engine2_Reg6_std_gaussian.txt',delimiter='\t')

# 6) target값 계산해서 열 추가하기

x_columns = list(['unit','timestep','sensor1','sensor2','sensor3',
                    'sensor4','sensor5','sensor6',
                    'sensor7','sensor8','sensor9',
                    'sensor10','sensor11','sensor12',
                    'sensor13','sensor14','sensor15',
                    'sensor16','sensor17','sensor18',
                    'sensor19','sensor20','sensor21','target'])

print(Engine2_Reg_std_gaussian[1].shape)
for i in range(1,7):
    Engine2_Reg_std_gaussian[i] = np.concatenate((Engine2_Reg_std_gaussian[i],np.zeros((Engine2_Reg_std_gaussian[i].shape[0],1))),axis=1)
    for j in range(Engine2_Reg_std_gaussian[i].shape[0]):
        Engine2_Reg_std_gaussian[i][j,-1] = maxCycle[Engine2_Reg_std_gaussian[i][j,0]] - Engine2_Reg_std_gaussian[i][j,1]


directory = '08_Engine2_RegimeData_STD_Gaussian_Target_Data'
if os.path.isdir(directory):
    print(directory+'폴더 있음')
else:
    print(directory+'폴더 없음, 생성함')
    os.makedirs(directory)

for i in range(1,7): # regime 1~6까지 반복
    filepath = './'+directory+'/Engine2_Reg'+str(i)+'_std_gaussian_target.txt' # 경로 지정
    np.savetxt(filepath,Engine2_Reg_std_gaussian[i],delimiter='\t',newline='\n') # 저장