import torch
from torch import nn, optim                           # torch 에서 제공하는 신경망 기술, 손실함수, 최적화를 할 수 있는 함수들을 불러온다.
import torch.nn.functional as F                       # torch 내의 세부적인 기능을 불러옴.
from torch.utils.data import DataLoader, Dataset      # 데이터를 모델에 사용할 수 있게 정리해주는 라이브러리.
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
# 데이터 불러오기
Engine2_Reg_std_gaussian_target = np.loadtxt('./07_FD002_Reg_Std_Gaussian_Target_data/FD002_Reg_std_Gaussian_Target_data.txt',delimiter='\t')

# Pandas로 변환
x_columns = list(['unit','timestep','set1','set2','set3',
                    'sensor1','sensor2','sensor3',
                    'sensor4','sensor5','sensor6',
                    'sensor7','sensor8','sensor9',
                    'sensor10','sensor11','sensor12',
                    'sensor13','sensor14','sensor15',
                    'sensor16','sensor17','sensor18',
                    'sensor19','sensor20','sensor21',
                    'type','regime','target'])
df = pd.DataFrame(Engine2_Reg_std_gaussian_target,columns=x_columns)
print(df)
TrainData = df[(df['type']==0)&(df['regime']==5)] # Train Data & Regime = 6  데이터 추출

Train_X = TrainData.drop(labels=['unit','timestep','set1','set2','set3','type','regime','target'],axis=1).to_numpy()
Train_Y = TrainData['target'].to_numpy().reshape((-1,1))

# TestData = df[(df['type']==1)&(df['regime']==5)&(df['unit']==51)] # Test Data & Regime = 6 & 51번 엔진 데이터 추출
TestData = df[(df['type']==1)&(df['unit']==51)&(df['regime']==5)] # Test Data & Regime = 6 & 51번 엔진 데이터 추출
print(TestData)
Test_X = TestData.drop(labels=['unit','timestep','set1','set2','set3','type','regime','target'],axis=1).to_numpy()
Test_Y = TestData['target'].to_numpy().reshape((-1,1))

Test_Timestep = TestData['timestep'].to_numpy().reshape((-1,1))
print(Test_Timestep)