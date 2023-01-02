import torch
from torch import nn, optim                           # torch 에서 제공하는 신경망 기술, 손실함수, 최적화를 할 수 있는 함수들을 불러온다.
import torch.nn.functional as F                       # torch 내의 세부적인 기능을 불러옴.
from torch.utils.data import DataLoader, Dataset      # 데이터를 모델에 사용할 수 있게 정리해주는 라이브러리.
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error        # regression 문제의 모델 성능 측정을 위해서 MSE를 불러온다.
import sys
import os
# torch의 Dataset 을 상속.
class TensorData(Dataset):
    def __init__(self, x_data, y_data):
        self.x_data = torch.FloatTensor(x_data)
        self.y_data = torch.FloatTensor(y_data)
        self.len = self.y_data.shape[0]
    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index] 
    def __len__(self):
        return self.len
class Regressor(nn.Module):
    def __init__(self,model_num,input_dim,loss_fcn,optimizer,epoch,trainloader):
        super().__init__() # 모델 연산 정의
        self.model_num = model_num
        self.loss_fcn = loss_fcn
        self.optimzer = optimizer
        self.epoch = epoch
        self.trainloader=trainloader
        self.loss_ = []

        self.fc1 = nn.Linear(input_dim, 50, bias=True) # 입력층(13) -> 은닉층1(50)으로 가는 연산
        self.fc2 = nn.Linear(50, 30, bias=True) # 은닉층1(50) -> 은닉층2(30)으로 가는 연산
        self.fc3 = nn.Linear(30, 1, bias=True) # 은닉층2(30) -> 출력층(1)으로 가는 연산
        self.dropout = nn.Dropout(0.2) # 연산이 될 때마다 20%의 비율로 랜덤하게 노드를 없앤다.
        

    def forward(self, x): # 모델 연산의 순서를 정의
        x = F.relu(self.fc1(x)) # Linear 계산 후 활성화 함수 ReLU를 적용한다.  
        x = self.dropout(F.relu(self.fc2(x))) # 은닉층2에서 드랍아웃을 적용한다.(즉, 30개의 20%인 6개의 노드가 계산에서 제외된다.)
        # x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x)) # Linear 계산 후 활성화 함수 ReLU를 적용한다.        
        return x

    def learning(self):
        # Loss Fcn 정의
        Criterion = nn.MSELoss() # default 값
        if self.loss_fcn == "MSE":
            Criterion = nn.MSELoss()
        elif self.loss_fcn == "": 
            Criterion = None # Loss Fcn 정의하면됨..
        else:
            Criterion = None
        # Optimzer 정의
        Optimizer = optim.Adam(self.parameters(), lr=0.001, weight_decay=1e-7)
        if self.optimzer == "Adam":
            Optimizer = optim.Adam(self.parameters(), lr=0.001, weight_decay=1e-7)
        else:
            Optimizer = None
        n = len(self.trainloader)
        self.train()
        for Epoch in range(self.epoch):
            running_loss = 0.0 # 한 에폭이 돌 때 그안에서 배치마다 loss가 나온다. 즉 한번 학습할 때 그렇게 쪼개지면서 loss가 다 나오니 MSE를 구하기 위해서 사용한다.
            print("regime"+str(self.model_num)+" 번째 현재 학습 진행 "+str(Epoch)+"/"+str(self.epoch))
            for i, data in enumerate(self.trainloader, 0): # 무작위로 섞인 32개의 데이터가 담긴 배치가 하나씩 들어온다.

                inputs, values = data # data에는 X, Y가 들어있다.

                Optimizer.zero_grad() # 최적화 초기화.

                outputs = self(inputs) # 모델에 입력값을 넣어 예측값을 산출한다.
                loss = Criterion(outputs, values) # 손실함수를 계산. error 계산.
                loss.backward() # 손실 함수를 기준으로 역전파를 설정한다.
                Optimizer.step() # 역전파를 진행하고 가중치를 업데이트한다.

                running_loss += loss.item() # epoch 마다 평균 loss를 계산하기 위해 배치 loss를 더한다.

            self.loss_.append(running_loss/n) # MSE(Mean Squared Error) 계산
    
    def saveModel(self,modelNumber,directory):
        if os.path.isdir(directory):
            print(directory+'폴더 있음')
        else:
            print(directory+'폴더 없음, 생성함')
            os.makedirs(directory)
        torch.save(self,directory+'model'+str(modelNumber)+'.pt')

    def saveLossGraph(self,modelNumber,directory):
        if os.path.isdir(directory):
            print(directory+'폴더 있음')
        else:
            print(directory+'폴더 없음, 생성함')
            os.makedirs(directory)
        plt.plot(self.loss_)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.grid()
        plt.title('Lossgraph_'+str(modelNumber))
        plt.savefig(directory+'Lossgraph_'+str(modelNumber)+'.png')
        plt.clf()