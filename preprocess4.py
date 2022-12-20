''' 코드 내용은 아래와 같음 '''
# 1. FD002 데이터 불러온 뒤 Unit 별 최대 Cycle수(=고장 시점)를 구하기
# 2. 최대 Cycle수를 활용하여 Target 값 구하기 = 최대 Cycle 수 - timestep값

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from scipy.ndimage import gaussian_filter1d

# 1) FD002의 표준화된 regime1~6 데이터 불러오기
FD002 = np.loadtxt('./06_FD002_Reg_Std_Gaussian_data/FD002_Reg_Std_Gaussian_data.txt',delimiter='\t',dtype='double')

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

#x = FD002_df[(FD002_df['unit'] == 2) &(FD002_df['type']==0) & (FD002_df['regime']==1)]['timestep']
# 2) FD002 데이터의 Unit 열과 Timestep 열을 DataFrame으로 변환

# 3) 최대 Unit 번호 구하기
maxUnitNum_train = FD002_df[(FD002_df['type']==0)]['unit'].max() # maybe 260

print("maxUnitNum_Train :",maxUnitNum_train)
# 4) Unit별 최대 사이클수 구하기
maxCycle_train = dict()
for i in range(1,int(maxUnitNum_train)+1):
    maxCycle_train[i] = FD002_df[(FD002_df['unit']==i)& (FD002_df['type']==0)].max()['timestep']


maxUnitNum_test = FD002_df[(FD002_df['type']==1)]['unit'].max() #maybe 259
print("maxUnitNum_Test :",maxUnitNum_test)

FD002_TestRUL = np.loadtxt('./11_Testdata/RUL_FD002.txt',delimiter='\t',dtype='double')
maxCycle_test = dict()

for i in range(1,int(maxUnitNum_test)+1):
    maxCycle_test[i] = FD002_df[(FD002_df['unit']==i)& (FD002_df['type']==1)].max()['timestep'] + FD002_TestRUL[i-1]

print(maxCycle_test[1])
print(maxCycle_test[2])
print(maxCycle_test[3])


# 6) target값 계산해서 열 추가하기

x_columns = list(['unit','timestep','sensor1','sensor2','sensor3',
                    'sensor4','sensor5','sensor6',
                    'sensor7','sensor8','sensor9',
                    'sensor10','sensor11','sensor12',
                    'sensor13','sensor14','sensor15',
                    'sensor16','sensor17','sensor18',
                    'sensor19','sensor20','sensor21','target'])
print(FD002_df.shape[0])
for i in range(FD002_df.shape[0]):
    # print(i)
    if FD002_df.loc[i,'type']==0: # train
        FD002_df.loc[i,'target'] = 130 if maxCycle_train[FD002_df.loc[i,'unit']]-FD002_df.loc[i,'timestep'] > 130 else maxCycle_train[FD002_df.loc[i,'unit']]-FD002_df.loc[i,'timestep']
    # elif FD002_df.loc[i,'type']==1: # test
    else:
        FD002_df.loc[i,'target'] = 130 if maxCycle_test[FD002_df.loc[i,'unit']]-FD002_df.loc[i,'timestep'] > 130 else maxCycle_test[FD002_df.loc[i,'unit']]-FD002_df.loc[i,'timestep']

directory = '07_FD002_Reg_Std_Gaussian_Target_data'
if os.path.isdir(directory):
    print(directory+'폴더 있음')
else:
    print(directory+'폴더 없음, 생성함')
    os.makedirs(directory)

for i in range(1,7): # regime 1~6까지 반복
    filepath = './'+directory+'/FD002_Reg_Std_Gaussian_Target_data.txt' # 경로 지정
    np.savetxt(filepath,FD002_df,delimiter='\t',newline='\n',fmt='%1.6e') # 저장