''' 코드 내용은 아래와 같음 '''

## 1. Spearman 상관계수 도출
## 2. 논문과 상관 계수 비교



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
# regime 데이터 로드
regime = np.loadtxt("./01_kMeansModelPred/results.txt",dtype='int')
#print(np.shape(regime))
regime=regime.reshape(-1,1)
# train_FD002 데이터 로드
train_FD002 = np.loadtxt("./10_Traindata/train_FD002.txt", dtype = 'float', delimiter=" ")
# 위의 2개 데이터 연접
data = np.concatenate((regime,train_FD002),axis=1)

#


# data_df = pd.DataFrame(data,columns=x_columns_)
#print(data_df)
#print('data_df 크기:', data_df.shape)



# add1 = np.reshape(data[0,2],(1,1))
# add2 = np.reshape(data[0,6:27],(1,-1))

# print(np.shape(add1))
# print(np.shape(add2))

# #print(np.shape(data[0,2].expand_dims(axis=0)))

# regime1 = np.concatenate((add1,add2), axis=1)
# print(np.shape(regime1))
# # print(regime1)

# regime별로 FD002 데이터 분리
row = np.shape(data)[0]
col = np.shape(data)[1]

regime1 = np.zeros((1,23))
regime2 = np.zeros((1,23))
regime3 = np.zeros((1,23))
regime4 = np.zeros((1,23))
regime5 = np.zeros((1,23))
regime6 = np.zeros((1,23))

for i in range(0,row):
    add = np.concatenate((np.reshape(data[i,1:3],(1,-1)),np.reshape(data[i,6:26+1],(1,-1))),axis=1)
    # add = ['unit','timestep','sensor1','sensor2','sensor3','sensor4','sensor5','sensor6','sensor7','sensor8','sensor9','sensor10',
    #           'sensor11','sensor12','sensor13','sensor14','sensor15','sensor16','sensor17','sensor18','sensor19','sensor20','sensor21']
    # if data[i,1] == 2: # 2번 엔진
    if data[i,0] == 0: # 여기서, data[i,0] = regime
        regime1 = np.concatenate((regime1,add),axis=0)
    elif data[i,0] == 1:
        regime2 = np.concatenate((regime2,add),axis=0)
    elif data[i,0] == 2:
        regime3 = np.concatenate((regime3,add),axis=0)
    elif data[i,0] == 3:
        regime4 = np.concatenate((regime4,add),axis=0)
    elif data[i,0] == 4:
        regime5 = np.concatenate((regime5,add),axis=0)
    elif data[i,0] == 5:
        regime6 = np.concatenate((regime6,add),axis=0)

regime = dict()
regime[1] = regime1[1:,:] # zero 값인 첫번째 행 제외
regime[2] = regime2[1:,:]
regime[3] = regime3[1:,:]
regime[4] = regime4[1:,:]
regime[5] = regime5[1:,:]
regime[6] = regime6[1:,:]
print("FD002 행의 수는 53759, 전처리 후 행의 수는 :",
    np.shape(regime[1])[0]+np.shape(regime[2])[0]+
        np.shape(regime[3])[0]+np.shape(regime[4])[0]+
        np.shape(regime[5])[0]+np.shape(regime[6])[0])

import os

directory = '03_Engine2_RegimeData'
if os.path.isdir(directory):
    print(directory+'폴더 있음')
else:
    print(directory+'폴더 없음, 생성함')
    os.makedirs(directory)

for i in range(1,6+1):
    filepath = './'+directory+'/regime'+str(i)+'.txt'
    np.savetxt(filepath,regime[i],delimiter='\t',newline='\n')



#------------------------------------------------------------------------------
# regime shape
# timestep sensor1 sensor2 ..... sensor21


x_columns_ = list(['unit','timestep','sensor1','sensor2','sensor3',
                    'sensor4','sensor5','sensor6',
                    'sensor7','sensor8','sensor9',
                    'sensor10','sensor11','sensor12',
                    'sensor13','sensor14','sensor15',
                    'sensor16','sensor17','sensor18',
                    'sensor19','sensor20','sensor21'])

sensorlist = list(['sensor1','sensor2','sensor3',
                    'sensor4','sensor5','sensor6',
                    'sensor7','sensor8','sensor9',
                    'sensor10','sensor11','sensor12',
                    'sensor13','sensor14','sensor15',
                    'sensor16','sensor17','sensor18',
                    'sensor19','sensor20','sensor21'])






###regime2_df = pd.DataFrame(regime[2],columns=x_columns_)
###spearmanValue = np.zeros((21,21))

regime1_df = pd.DataFrame(regime[1],columns=x_columns_)
regime2_df = pd.DataFrame(regime[2],columns=x_columns_)
regime3_df = pd.DataFrame(regime[3],columns=x_columns_)
regime4_df = pd.DataFrame(regime[4],columns=x_columns_)
regime5_df = pd.DataFrame(regime[5],columns=x_columns_)
regime6_df = pd.DataFrame(regime[6],columns=x_columns_)
spearmanValue1 = np.zeros((21,21))
spearmanValue2 = np.zeros((21,21))
spearmanValue3 = np.zeros((21,21))
spearmanValue4 = np.zeros((21,21))
spearmanValue5 = np.zeros((21,21))
spearmanValue6 = np.zeros((21,21))

# for i in range(1,22):
#     #command1 = "regime"+str(i)+"_df = pd.DataFrame(regime["+str(i)+"],columns=x_columns_)"
#     #eval(command1)
#     command2 = "spearmanValue"+str(i)+"=np.zeros((21,21))"
#     eval(command2)

rho = 0
p_val =0

for regNum in range(1,7):
    for i in range(1,22):
        for j in range(1,22):
            #rho, p_val = stats.spearmanr(regime2_df['sensor'+str(i)],regime2_df['sensor'+str(j)])
            command1 = 'rho,p_val = stats.spearmanr(regime'+str(regNum)+'_df[\'sensor'+str(i)+'\'],regime'+str(regNum)+'_df[\'sensor'+str(j)+'\'])'
            exec(command1)
            if math.isnan(rho):
                rho = 0

            #spearmanValue[i-1,j-1] = rho
            command2 = "spearmanValue"+str(regNum)+"["+str(i-1)+","+str(j-1)+"]=abs(rho)"
            exec(command2)


        
spearmanValue1_df = pd.DataFrame(spearmanValue1, index = [i for i in sensorlist], columns = [i for i in sensorlist])
spearmanValue2_df = pd.DataFrame(spearmanValue2, index = [i for i in sensorlist], columns = [i for i in sensorlist])
spearmanValue3_df = pd.DataFrame(spearmanValue3, index = [i for i in sensorlist], columns = [i for i in sensorlist])
spearmanValue4_df = pd.DataFrame(spearmanValue4, index = [i for i in sensorlist], columns = [i for i in sensorlist])
spearmanValue5_df = pd.DataFrame(spearmanValue5, index = [i for i in sensorlist], columns = [i for i in sensorlist])
spearmanValue6_df = pd.DataFrame(spearmanValue6, index = [i for i in sensorlist], columns = [i for i in sensorlist])

directory = '04_Engine2_spearmanValue'
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





#rho, p_val = stats.spearmanr(regime2_df['sensor2'],regime2_df['sensor2'])
#print('rho : ', rho, ' p_val : ',p_val)
# rho, p_val = stats.spearmanr(regime[2][:,1],regime[2][:,2])
# print('rho : ', rho, ' p_val : ',p_val)



# np.savetxt('regime2.txt',regime[2],delimiter='\t',newline='\n')

