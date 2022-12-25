''' 코드 내용은 아래와 같음 '''

## 1. k-means 군집화 알고리즘 적용
#  1.1 K-means 군집화 알고리즘 적용
#  1.2 논문과 같은 사진(그래프) 도출

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
FD002_input = np.loadtxt('./01_FD002_Reg_data/FD002_Reg_data.txt',delimiter='\t',dtype='float')
FD002_target = np.loadtxt('./07_FD002_Reg_Std_Gaussian_Target_data/FD002_Reg_Std_Gaussian_Target_data.txt',delimiter='\t',dtype='float')
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

FD002_input_df = pd.DataFrame(FD002_input,columns=x_columns_)
FD002_target_df = pd.DataFrame(FD002_target,columns=x_columns_)
FD002_input_df['target'] = FD002_target_df['target']

FD002_df = FD002_input_df
regime = dict()
regime[1] = FD002_df[(FD002_df['unit']==2) & (FD002_df['regime']==0) & (FD002_df['type']==0)]
regime[2] = FD002_df[(FD002_df['unit']==2) & (FD002_df['regime']==1) & (FD002_df['type']==0)]
regime[3] = FD002_df[(FD002_df['unit']==2) & (FD002_df['regime']==2) & (FD002_df['type']==0)]
regime[4] = FD002_df[(FD002_df['unit']==2) & (FD002_df['regime']==3) & (FD002_df['type']==0)]
regime[5] = FD002_df[(FD002_df['unit']==2) & (FD002_df['regime']==4) & (FD002_df['type']==0)]
regime[6] = FD002_df[(FD002_df['unit']==2) & (FD002_df['regime']==5) & (FD002_df['type']==0)]


print('--------------------------------------')
print('-------------Spearman Value image 저장--------------')
spearmanValue1 = np.zeros((21,1))
spearmanValue2 = np.zeros((21,1))
spearmanValue3 = np.zeros((21,1))
spearmanValue4 = np.zeros((21,1))
spearmanValue5 = np.zeros((21,1))
spearmanValue6 = np.zeros((21,1))

rho = 0
p_val =0

import math

for regNum in range(1,7):
    for i in range(1,22):
        
        #rho, p_val = stats.spearmanr(regime[regNum]['sensor'+str(i)],regime[regNum]['sensor'+str(j)])
        command1 = 'rho,p_val = stats.spearmanr(regime['+str(regNum)+'][\'sensor'+str(i)+'\'],regime['+str(regNum)+'][\'target\'])'
        exec(command1)
        if math.isnan(rho):
            rho = 0
        #spearmanValue[i-1,j-1] = rho
        command2 = "spearmanValue"+str(regNum)+"["+str(i-1)+",0]=abs(rho)"
        exec(command2)
sensorlist = list(['sensor1','sensor2','sensor3',
                    'sensor4','sensor5','sensor6',
                    'sensor7','sensor8','sensor9',
                    'sensor10','sensor11','sensor12',
                    'sensor13','sensor14','sensor15',
                    'sensor16','sensor17','sensor18',
                    'sensor19','sensor20','sensor21'])
spearmanValue1_df = pd.DataFrame(spearmanValue1, index = [i for i in sensorlist], columns = ['target'])
spearmanValue2_df = pd.DataFrame(spearmanValue2, index = [i for i in sensorlist], columns = ['target'])
spearmanValue3_df = pd.DataFrame(spearmanValue3, index = [i for i in sensorlist], columns = ['target'])
spearmanValue4_df = pd.DataFrame(spearmanValue4, index = [i for i in sensorlist], columns = ['target'])
spearmanValue5_df = pd.DataFrame(spearmanValue5, index = [i for i in sensorlist], columns = ['target'])
spearmanValue6_df = pd.DataFrame(spearmanValue6, index = [i for i in sensorlist], columns = ['target'])

directory = '03_FD002_spearmanValue_img'
import os.path
if os.path.isdir(directory):
    print(directory+'폴더 있음')
else:
    print(directory+'폴더 없음, 생성함')
    os.makedirs(directory)

for i in range(1,7):
    plt.figure(figsize=(15,15)) # 단위 inch
    #sn.heatmap(name, annot=True,cmap="OrRd")
    command1 = 'sn.heatmap(spearmanValue'+str(i)+'_df, vmax = 0.4, vmin=0,annot=True,cmap="OrRd")'
    eval(command1)
    plt.title('regime'+str(i))
    plt.savefig('./'+directory+'/spearmanValue'+str(i)+'2.png', bbox_inches='tight')
    plt.cla()

print('--------------------------------------')
print('-------------Spearman Value image 저장--------------')
spearmanValue1 = np.zeros((21,21))
spearmanValue2 = np.zeros((21,21))
spearmanValue3 = np.zeros((21,21))
spearmanValue4 = np.zeros((21,21))
spearmanValue5 = np.zeros((21,21))
spearmanValue6 = np.zeros((21,21))

rho = 0
p_val =0

import math

for regNum in range(1,7):
    for i in range(1,22):
        for j in range(1,22):
            #rho, p_val = stats.spearmanr(regime[regNum]['sensor'+str(i)],regime[regNum]['sensor'+str(j)])
            command1 = 'rho,p_val = stats.spearmanr(regime['+str(regNum)+'][\'sensor'+str(i)+'\'],regime['+str(regNum)+'][\'sensor'+str(j)+'\'])'
            exec(command1)
            if math.isnan(rho):
                rho = 0
            #spearmanValue[i-1,j-1] = rho
            command2 = "spearmanValue"+str(regNum)+"["+str(i-1)+","+str(j-1)+"]=abs(rho)"
            exec(command2)
sensorlist = list(['sensor1','sensor2','sensor3',
                    'sensor4','sensor5','sensor6',
                    'sensor7','sensor8','sensor9',
                    'sensor10','sensor11','sensor12',
                    'sensor13','sensor14','sensor15',
                    'sensor16','sensor17','sensor18',
                    'sensor19','sensor20','sensor21'])
spearmanValue1_df = pd.DataFrame(spearmanValue1, index = [i for i in sensorlist], columns = [i for i in sensorlist])
spearmanValue2_df = pd.DataFrame(spearmanValue2, index = [i for i in sensorlist], columns = [i for i in sensorlist])
spearmanValue3_df = pd.DataFrame(spearmanValue3, index = [i for i in sensorlist], columns = [i for i in sensorlist])
spearmanValue4_df = pd.DataFrame(spearmanValue4, index = [i for i in sensorlist], columns = [i for i in sensorlist])
spearmanValue5_df = pd.DataFrame(spearmanValue5, index = [i for i in sensorlist], columns = [i for i in sensorlist])
spearmanValue6_df = pd.DataFrame(spearmanValue6, index = [i for i in sensorlist], columns = [i for i in sensorlist])

directory = '03_FD002_spearmanValue_img'
import os.path
if os.path.isdir(directory):
    print(directory+'폴더 있음')
else:
    print(directory+'폴더 없음, 생성함')
    os.makedirs(directory)

for i in range(1,7):
    plt.figure(figsize=(15,15)) # 단위 inch
    #sn.heatmap(name, annot=True,cmap="OrRd")
    command1 = 'sn.heatmap(spearmanValue'+str(i)+'_df, vmax = 0.4, vmin=0,annot=True,cmap="OrRd")'
    eval(command1)
    plt.title('regime'+str(i))
    plt.savefig('./'+directory+'/spearmanValue'+str(i)+'.png', bbox_inches='tight')
    plt.cla()
