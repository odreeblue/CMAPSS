''' 코드 내용은 아래와 같음 '''
# 1. Regime6(논문:regime1)에 대해 가우시안 필터 적용해서 이미지 도출만 하기
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

'''
# 1) 2번엔진의 regime1~6 데이터 불러오기
Engine2_Reg1_std = np.loadtxt('./03_Engine2_RegimeData/regime1.txt',delimiter='\t')
Engine2_Reg2_std = np.loadtxt('./03_Engine2_RegimeData/regime2.txt',delimiter='\t')
Engine2_Reg3_std = np.loadtxt('./03_Engine2_RegimeData/regime3.txt',delimiter='\t')
Engine2_Reg4_std = np.loadtxt('./03_Engine2_RegimeData/regime4.txt',delimiter='\t')
Engine2_Reg5_std = np.loadtxt('./03_Engine2_RegimeData/regime5.txt',delimiter='\t')
Engine2_Reg6_std = np.loadtxt('./03_Engine2_RegimeData/regime6.txt',delimiter='\t')
'''

# 2) numpy to Pandas 변환하기
x_columns_ = list(['unit','timestep','sensor1','sensor2','sensor3',
                    'sensor4','sensor5','sensor6',
                    'sensor7','sensor8','sensor9',
                    'sensor10','sensor11','sensor12',
                    'sensor13','sensor14','sensor15',
                    'sensor16','sensor17','sensor18',
                    'sensor19','sensor20','sensor21'])

Engine2_Reg1_std_df = pd.DataFrame(Engine2_Reg1_std,columns=x_columns_)
Engine2_Reg2_std_df = pd.DataFrame(Engine2_Reg2_std,columns=x_columns_)
Engine2_Reg3_std_df = pd.DataFrame(Engine2_Reg3_std,columns=x_columns_)
Engine2_Reg4_std_df = pd.DataFrame(Engine2_Reg4_std,columns=x_columns_)
Engine2_Reg5_std_df = pd.DataFrame(Engine2_Reg5_std,columns=x_columns_)
Engine2_Reg6_std_df = pd.DataFrame(Engine2_Reg6_std,columns=x_columns_)


sensorlist = [2,3,4,9,11,15,17] # 논문 상 Spearman value가 0.4 이상이 센서 리스트

directory = '06_Engine2_RegimeData_STD_Gaussian'
if os.path.isdir(directory):
    print(directory+'폴더 있음')
else:
    print(directory+'폴더 없음, 생성함')
    os.makedirs(directory)


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