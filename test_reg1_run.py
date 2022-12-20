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
    def __init__(self):
        super().__init__() # 모델 연산 정의
        self.fc1 = nn.Linear(21, 50, bias=True) # 입력층(13) -> 은닉층1(50)으로 가는 연산
        self.fc2 = nn.Linear(50, 30, bias=True) # 은닉층1(50) -> 은닉층2(30)으로 가는 연산
        self.fc3 = nn.Linear(30, 1, bias=True) # 은닉층2(30) -> 출력층(1)으로 가는 연산
        self.dropout = nn.Dropout(0.2) # 연산이 될 때마다 20%의 비율로 랜덤하게 노드를 없앤다.

    def forward(self, x): # 모델 연산의 순서를 정의
        x = F.relu(self.fc1(x)) # Linear 계산 후 활성화 함수 ReLU를 적용한다.  
        x = self.dropout(F.relu(self.fc2(x))) # 은닉층2에서 드랍아웃을 적용한다.(즉, 30개의 20%인 6개의 노드가 계산에서 제외된다.)
        x = F.relu(self.fc3(x)) # Linear 계산 후 활성화 함수 ReLU를 적용한다.  
      
        return x
    
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
TrainData = df[(df['type']==0)&(df['regime']==5)] # Train Data & Regime = 6  데이터 추출

Train_X = TrainData.drop(labels=['unit','timestep','set1','set2','set3','type','regime','target'],axis=1).to_numpy()
Train_Y = TrainData['target'].to_numpy().reshape((-1,1))

TestData = df[(df['type']==0)&(df['regime']==5)&(df['unit']==50)] # Test Data & Regime = 6  데이터 추출
Test_X = TestData.drop(labels=['unit','timestep','set1','set2','set3','type','regime','target'],axis=1).to_numpy()
Test_Y = TestData['target'].to_numpy().reshape((-1,1))
# 전체 데이터를 학습 데이터와 평가 데이터로 나눈다.
# 기준으로 잡은 논문이 전체 데이터를 50%, 50%로 나눴기 때문에 test size를 0.5로 설정한다.


#X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1)

# 학습 데이터, 시험 데이터 배치 형태로 구축하기
trainsets = TensorData(Train_X, Train_Y)
trainloader = torch.utils.data.DataLoader(trainsets, batch_size=256, shuffle=True)

#testsets = TensorData(X_test, Y_test)
#testloader = torch.utils.data.DataLoader(testsets, batch_size=32, shuffle=False)



model = Regressor()
criterion = nn.MSELoss()

optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-7)

loss_ = [] # loss를 저장할 리스트.
n = len(trainloader)

EPOCH = 200


for epoch in range(EPOCH):
    running_loss = 0.0 # 한 에폭이 돌 때 그안에서 배치마다 loss가 나온다. 즉 한번 학습할 때 그렇게 쪼개지면서 loss가 다 나오니 MSE를 구하기 위해서 사용한다.
    print("현재 학습 진행 "+str(epoch)+"/"+str(EPOCH))
    for i, data in enumerate(trainloader, 0): # 무작위로 섞인 32개의 데이터가 담긴 배치가 하나씩 들어온다.
        
        inputs, values = data # data에는 X, Y가 들어있다.

        optimizer.zero_grad() # 최적화 초기화.
    
        outputs = model(inputs) # 모델에 입력값을 넣어 예측값을 산출한다.
        loss = criterion(outputs, values) # 손실함수를 계산. error 계산.
        loss.backward() # 손실 함수를 기준으로 역전파를 설정한다.
        optimizer.step() # 역전파를 진행하고 가중치를 업데이트한다.
    
        running_loss += loss.item() # epoch 마다 평균 loss를 계산하기 위해 배치 loss를 더한다.
  
    loss_.append(running_loss/n) # MSE(Mean Squared Error) 계산

torch.save(model,'test1.pt')

plt.plot(loss_)
plt.title('Loss')
plt.xlabel('epoch')
plt.show()


