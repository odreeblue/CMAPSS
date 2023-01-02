''' 코드 내용은 아래와 같음 '''
# 1. Target 값까지 포함된 regime1데이터를 불러와서 학습하기

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
    def __init__(self,input_dim,hidden1_dim,hidden2_dim):
        super().__init__() # 모델 연산 정의
        self.fc1 = nn.Linear(input_dim, hidden1_dim, bias=True) # 입력층(13) -> 은닉층1(50)으로 가는 연산
        self.fc2 = nn.Linear(hidden1_dim, hidden2_dim, bias=True) # 은닉층1(50) -> 은닉층2(30)으로 가는 연산
        self.fc3 = nn.Linear(hidden2_dim, 1, bias=True) # 은닉층2(30) -> 출력층(1)으로 가는 연산
        self.dropout = nn.Dropout(0.2) # 연산이 될 때마다 20%의 비율로 랜덤하게 노드를 없앤다.

    def forward(self, x): # 모델 연산의 순서를 정의
        x = F.relu(self.fc1(x)) # Linear 계산 후 활성화 함수 ReLU를 적용한다.  
        x = self.dropout(F.relu(self.fc2(x))) # 은닉층2에서 드랍아웃을 적용한다.(즉, 30개의 20%인 6개의 노드가 계산에서 제외된다.)
        # x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x)) # Linear 계산 후 활성화 함수 ReLU를 적용한다.        
        return x
sensor = dict()
sensor[1]=[2,3,4,9,11,15,17]

sensor[2]=[2,3,4,9,11,15,17]

#sensor[3]=[2,3,4,6,7,8,9,11,12,13,15,16,17,20,21]
sensor[3]=[2,3,4,9,11,15,17]

#sensor[4]=[2,3,4,7,8,9,11,12,13,15,17,20,21]
sensor[4]=[2,3,4,6,7,8,9,11,12,13,15,16,17,20,21]

#sensor[5]=[2,3,4,9,11,15,17]
sensor[5]=[2,3,4,7,8,9,11,12,13,15,17,20,21]

sensor[6]=[2,3,4,6,7,8,9,11,12,13,15,17,20,21]

# sensor[1]=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21]
# sensor[2]=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21]
# sensor[3]=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21]
# sensor[4]=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21]
# sensor[5]=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21]
# sensor[6]=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21]


SENSORNUMBER=dict()
for i in range(1,7):
    SENSORNUMBER[i] = list(['sensor'+str(number) for number in sensor[i]])
    print(SENSORNUMBER[i])


# 주의 사항
# 드랍아웃은 과적합(overfitting)을 방지하기 위해 노드의 일부를 배제하고 계산하는 방식이기 때문에 절대로 출력층에 사용해서는 안 된다.
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

TrainData = dict()
TrainData[1] = df[(df['type']==0)&(df['regime']==0)]
TrainData[2] = df[(df['type']==0)&(df['regime']==1)]
TrainData[3] = df[(df['type']==0)&(df['regime']==2)]
TrainData[4] = df[(df['type']==0)&(df['regime']==3)]
TrainData[5] = df[(df['type']==0)&(df['regime']==4)]
TrainData[6] = df[(df['type']==0)&(df['regime']==5)]

Train_X = dict()
# Train_X[1] = TrainData[1].drop(labels=['unit','timestep','set1','set2','set3','type','regime','target'],axis=1).to_numpy()
# Train_X[2] = TrainData[2].drop(labels=['unit','timestep','set1','set2','set3','type','regime','target'],axis=1).to_numpy()
# Train_X[3] = TrainData[3].drop(labels=['unit','timestep','set1','set2','set3','type','regime','target'],axis=1).to_numpy()
# Train_X[4] = TrainData[4].drop(labels=['unit','timestep','set1','set2','set3','type','regime','target'],axis=1).to_numpy()
# Train_X[5] = TrainData[5].drop(labels=['unit','timestep','set1','set2','set3','type','regime','target'],axis=1).to_numpy()
# Train_X[6] = TrainData[6].drop(labels=['unit','timestep','set1','set2','set3','type','regime','target'],axis=1).to_numpy()
Train_X[1] = TrainData[1][SENSORNUMBER[1]].to_numpy()
Train_X[2] = TrainData[2][SENSORNUMBER[2]].to_numpy()
Train_X[3] = TrainData[3][SENSORNUMBER[3]].to_numpy()
Train_X[4] = TrainData[4][SENSORNUMBER[4]].to_numpy()
Train_X[5] = TrainData[5][SENSORNUMBER[5]].to_numpy()
Train_X[6] = TrainData[6][SENSORNUMBER[6]].to_numpy()


Train_Y = dict()
Train_Y[1] = TrainData[1]['target'].to_numpy().reshape((-1,1))
Train_Y[2] = TrainData[2]['target'].to_numpy().reshape((-1,1))
Train_Y[3] = TrainData[3]['target'].to_numpy().reshape((-1,1))
Train_Y[4] = TrainData[4]['target'].to_numpy().reshape((-1,1))
Train_Y[5] = TrainData[5]['target'].to_numpy().reshape((-1,1))
Train_Y[6] = TrainData[6]['target'].to_numpy().reshape((-1,1))

#TestData = df[(df['type']==0)&(df['regime']==5)&(df['unit']==50)] # Test Data & Regime = 6  데이터 추출
#Test_X = TestData.drop(labels=['unit','timestep','set1','set2','set3','type','regime','target'],axis=1).to_numpy()
#Test_Y = TestData['target'].to_numpy().reshape((-1,1))
# 전체 데이터를 학습 데이터와 평가 데이터로 나눈다.
# 기준으로 잡은 논문이 전체 데이터를 50%, 50%로 나눴기 때문에 test size를 0.5로 설정한다.


#X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1)

# 학습 데이터, 시험 데이터 배치 형태로 구축하기
trainsets = dict()
trainsets[1] = TensorData(Train_X[1], Train_Y[1])
trainsets[2] = TensorData(Train_X[2], Train_Y[2])
trainsets[3] = TensorData(Train_X[3], Train_Y[3])
trainsets[4] = TensorData(Train_X[4], Train_Y[4])
trainsets[5] = TensorData(Train_X[5], Train_Y[5])
trainsets[6] = TensorData(Train_X[6], Train_Y[6])

trainloader=dict()
trainloader[1] = torch.utils.data.DataLoader(trainsets[1], batch_size=256, shuffle=True)
trainloader[2] = torch.utils.data.DataLoader(trainsets[2], batch_size=256, shuffle=True)
trainloader[3] = torch.utils.data.DataLoader(trainsets[3], batch_size=256, shuffle=True)
trainloader[4] = torch.utils.data.DataLoader(trainsets[4], batch_size=256, shuffle=True)
trainloader[5] = torch.utils.data.DataLoader(trainsets[5], batch_size=256, shuffle=True)
trainloader[6] = torch.utils.data.DataLoader(trainsets[6], batch_size=256, shuffle=True)

model = dict()
model[1] = Regressor(len(sensor[1]),50,30)
model[2] = Regressor(len(sensor[2]),50,30)
model[3] = Regressor(len(sensor[3]),50,30)
model[4] = Regressor(len(sensor[4]),50,30)
model[5] = Regressor(len(sensor[5]),50,30)
model[6] = Regressor(len(sensor[6]),50,30)

criterion = dict()
criterion[1] = nn.MSELoss()
criterion[2] = nn.MSELoss()
criterion[3] = nn.MSELoss()
criterion[4] = nn.MSELoss()
criterion[5] = nn.MSELoss()
criterion[6] = nn.MSELoss()

optimizer = dict()
optimizer[1] = optim.Adam(model[1].parameters(), lr=0.001, weight_decay=1e-7)
optimizer[2] = optim.Adam(model[2].parameters(), lr=0.001, weight_decay=1e-7)
optimizer[3] = optim.Adam(model[3].parameters(), lr=0.001, weight_decay=1e-7)
optimizer[4] = optim.Adam(model[4].parameters(), lr=0.001, weight_decay=1e-7)
optimizer[5] = optim.Adam(model[5].parameters(), lr=0.001, weight_decay=1e-7)
optimizer[6] = optim.Adam(model[6].parameters(), lr=0.001, weight_decay=1e-7)
# optimizer[1] = optim.LBFGS(model[1].parameters(), history_size=10, max_iter=4)
# optimizer[2] = optim.LBFGS(model[2].parameters(), history_size=10, max_iter=4)
# optimizer[3] = optim.LBFGS(model[3].parameters(), history_size=10, max_iter=4)
# optimizer[4] = optim.LBFGS(model[4].parameters(), history_size=10, max_iter=4)
# optimizer[5] = optim.LBFGS(model[5].parameters(), history_size=10, max_iter=4)
# optimizer[6] = optim.LBFGS(model[6].parameters(), history_size=10, max_iter=4)


loss_ = dict()
loss_[1]=[]
loss_[2]=[]
loss_[3]=[]
loss_[4]=[]
loss_[5]=[]
loss_[6]=[]

n = dict()
n[1] = len(trainloader[1])
n[2] = len(trainloader[2])
n[3] = len(trainloader[3])
n[4] = len(trainloader[4])
n[5] = len(trainloader[5])
n[6] = len(trainloader[6])

EPOCH = dict()
EPOCH[1]=200
EPOCH[2]=200
EPOCH[3]=200
EPOCH[4]=200
EPOCH[5]=200
EPOCH[6]=400

import os
directory = '12_ANN_Model'
if os.path.isdir(directory):
    print(directory+'폴더 있음')
else:
    print(directory+'폴더 없음, 생성함')
    os.makedirs(directory)
model[1].eval()
model[2].eval()
model[3].eval()
model[4].eval()
model[5].eval()
model[6].eval()


for regime in range(1,7):
    model[regime].train()
    for epoch in range(EPOCH[regime]):
        running_loss = 0.0 # 한 에폭이 돌 때 그안에서 배치마다 loss가 나온다. 즉 한번 학습할 때 그렇게 쪼개지면서 loss가 다 나오니 MSE를 구하기 위해서 사용한다.
        print("regime"+str(regime)+" 번째 현재 학습 진행 "+str(epoch)+"/"+str(EPOCH[regime]))
        for i, data in enumerate(trainloader[regime], 0): # 무작위로 섞인 32개의 데이터가 담긴 배치가 하나씩 들어온다.

            inputs, values = data # data에는 X, Y가 들어있다.

            optimizer[regime].zero_grad() # 최적화 초기화.

            outputs = model[regime](inputs) # 모델에 입력값을 넣어 예측값을 산출한다.
            loss = criterion[regime](outputs, values) # 손실함수를 계산. error 계산.
            loss.backward() # 손실 함수를 기준으로 역전파를 설정한다.
            optimizer[regime].step() # 역전파를 진행하고 가중치를 업데이트한다.

            running_loss += loss.item() # epoch 마다 평균 loss를 계산하기 위해 배치 loss를 더한다.
    
        loss_[regime].append(running_loss/n[regime]) # MSE(Mean Squared Error) 계산
    model[regime].eval()
    torch.save(model[regime],'./12_ANN_Model/test'+str(regime)+'.pt')

# plt.plot(loss_)
# plt.title('Loss')
# plt.xlabel('epoch')
# plt.show()






