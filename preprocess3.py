''' 코드 내용은 아래와 같음 '''

## 1. 가우시안 필터 적용하기

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os


# 1) 2번엔진의 regime1~6 데이터 불러오기
Engine2_Reg1 = np.loadtxt('./03_Engine2_RegimeData/regime1.txt',delimiter='\t')
Engine2_Reg2 = np.loadtxt('./03_Engine2_RegimeData/regime2.txt',delimiter='\t')
Engine2_Reg3 = np.loadtxt('./03_Engine2_RegimeData/regime3.txt',delimiter='\t')
Engine2_Reg4 = np.loadtxt('./03_Engine2_RegimeData/regime4.txt',delimiter='\t')
Engine2_Reg5 = np.loadtxt('./03_Engine2_RegimeData/regime5.txt',delimiter='\t')
Engine2_Reg6 = np.loadtxt('./03_Engine2_RegimeData/regime6.txt',delimiter='\t')


# 2) numpy to Pandas 변환하기

x_columns_ = list(['unit','timestep','sensor1','sensor2','sensor3',
                    'sensor4','sensor5','sensor6',
                    'sensor7','sensor8','sensor9',
                    'sensor10','sensor11','sensor12',
                    'sensor13','sensor14','sensor15',
                    'sensor16','sensor17','sensor18',
                    'sensor19','sensor20','sensor21'])
                    
x_columns_2 = list(['sensor1','sensor2','sensor3',
                    'sensor4','sensor5','sensor6',
                    'sensor7','sensor8','sensor9',
                    'sensor10','sensor11','sensor12',
                    'sensor13','sensor14','sensor15',
                    'sensor16','sensor17','sensor18',
                    'sensor19','sensor20','sensor21'])

Engine2_Reg1_df = pd.DataFrame(Engine2_Reg1,columns=x_columns_)
Engine2_Reg2_df = pd.DataFrame(Engine2_Reg2,columns=x_columns_)
Engine2_Reg3_df = pd.DataFrame(Engine2_Reg3,columns=x_columns_)
Engine2_Reg4_df = pd.DataFrame(Engine2_Reg4,columns=x_columns_)
Engine2_Reg5_df = pd.DataFrame(Engine2_Reg5,columns=x_columns_)
Engine2_Reg6_df = pd.DataFrame(Engine2_Reg6,columns=x_columns_)

# 3) 데이터 Standardization
from sklearn.preprocessing import StandardScaler
std_scaler1 = StandardScaler()
std_scaler2 = StandardScaler()
std_scaler3 = StandardScaler()
std_scaler4 = StandardScaler()
std_scaler5 = StandardScaler()
std_scaler6 = StandardScaler()
    # 3.0)  Call 'fit' with appropriate arguments before using this estimator.
fitted = std_scaler1.fit(Engine2_Reg1_df[x_columns_2])
fitted = std_scaler2.fit(Engine2_Reg2_df[x_columns_2])
fitted = std_scaler3.fit(Engine2_Reg3_df[x_columns_2])
fitted = std_scaler4.fit(Engine2_Reg4_df[x_columns_2])
fitted = std_scaler5.fit(Engine2_Reg5_df[x_columns_2])
fitted = std_scaler6.fit(Engine2_Reg6_df[x_columns_2])

    # 3.1) 데이터 표준화 변환
Engine2_Reg1_std_df = std_scaler1.transform(Engine2_Reg1_df[x_columns_2])
Engine2_Reg2_std_df = std_scaler2.transform(Engine2_Reg2_df[x_columns_2])
Engine2_Reg3_std_df = std_scaler3.transform(Engine2_Reg3_df[x_columns_2])
Engine2_Reg4_std_df = std_scaler4.transform(Engine2_Reg4_df[x_columns_2])
Engine2_Reg5_std_df = std_scaler5.transform(Engine2_Reg5_df[x_columns_2])
Engine2_Reg6_std_df = std_scaler6.transform(Engine2_Reg6_df[x_columns_2])
    # 3.2) 변환된 데이터 데이터프레임에 넣기
Engine2_Reg_std_df = dict()
Engine2_Reg_std_df[1] = pd.DataFrame(Engine2_Reg1_std_df,columns = x_columns_2)
Engine2_Reg_std_df[2] = pd.DataFrame(Engine2_Reg2_std_df,columns = x_columns_2)
Engine2_Reg_std_df[3] = pd.DataFrame(Engine2_Reg3_std_df,columns = x_columns_2)
Engine2_Reg_std_df[4] = pd.DataFrame(Engine2_Reg4_std_df,columns = x_columns_2)
Engine2_Reg_std_df[5] = pd.DataFrame(Engine2_Reg5_std_df,columns = x_columns_2)
Engine2_Reg_std_df[6] = pd.DataFrame(Engine2_Reg6_std_df,columns = x_columns_2)


Engine2_Reg_std_df[1] = pd.concat([Engine2_Reg1_df[["unit","timestep"]],Engine2_Reg_std_df[1]],axis=1)
Engine2_Reg_std_df[2] = pd.concat([Engine2_Reg2_df[["unit","timestep"]],Engine2_Reg_std_df[2]],axis=1)
Engine2_Reg_std_df[3] = pd.concat([Engine2_Reg3_df[["unit","timestep"]],Engine2_Reg_std_df[3]],axis=1)
Engine2_Reg_std_df[4] = pd.concat([Engine2_Reg4_df[["unit","timestep"]],Engine2_Reg_std_df[4]],axis=1)
Engine2_Reg_std_df[5] = pd.concat([Engine2_Reg5_df[["unit","timestep"]],Engine2_Reg_std_df[5]],axis=1)
Engine2_Reg_std_df[6] = pd.concat([Engine2_Reg6_df[["unit","timestep"]],Engine2_Reg_std_df[6]],axis=1)


import os
directory = '05_Engine2_RegimeData_Standardization'
if os.path.isdir(directory):
    print(directory+'폴더 있음')
else:
    print(directory+'폴더 없음, 생성함')
    os.makedirs(directory)

for i in range(1,6+1):
    filepath = './'+directory+'/Engine2_Reg'+str(i)+'_std.txt'
    np.savetxt(filepath,Engine2_Reg_std_df[i],delimiter='\t',newline='\n')
