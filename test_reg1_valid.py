
import torch
from torch import nn, optim                           # torch 에서 제공하는 신경망 기술, 손실함수, 최적화를 할 수 있는 함수들을 불러온다.
import torch.nn.functional as F                       # torch 내의 세부적인 기능을 불러옴.
from torch.utils.data import DataLoader, Dataset      # 데이터를 모델에 사용할 수 있게 정리해주는 라이브러리.
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
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
TestData = df[(df['type']==0)&(df['unit']==50)&(df['regime']==1)] # Test Data & Regime = 6 & 51번 엔진 데이터 추출
print(TestData)
Test_X = TestData.drop(labels=['unit','timestep','set1','set2','set3','type','regime','target'],axis=1).to_numpy()
Test_Y = TestData['target'].to_numpy().reshape((-1,1))

Test_Timestep = TestData['timestep'].to_numpy().reshape((-1,1))
# 전체 데이터를 학습 데이터와 평가 데이터로 나눈다.
# 기준으로 잡은 논문이 전체 데이터를 50%, 50%로 나눴기 때문에 test size를 0.5로 설정한다.


#X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1)

# 학습 데이터, 시험 데이터 배치 형태로 구축하기
trainsets = TensorData(Train_X, Train_Y)
trainloader = torch.utils.data.DataLoader(trainsets, batch_size=256, shuffle=False)

testsets = TensorData(Test_X, Test_Y)
testloader = torch.utils.data.DataLoader(testsets, batch_size=256, shuffle=False)


model = torch.load('test1.pt')
model.eval()


def evaluation(dataloader):

    predictions = torch.tensor([], dtype=torch.float) # 예측값을 저장하는 텐서.
    actual = torch.tensor([], dtype=torch.float) # 실제값을 저장하는 텐서.

    with torch.no_grad():
        model.eval() # 평가를 할 땐 반드시 eval()을 사용해야 한다.

        for data in dataloader:
            inputs, values = data
            outputs = model(inputs)

            predictions = torch.cat((predictions, outputs), 0) # cat함수를 통해 예측값을 누적.
            actual = torch.cat((actual, values), 0) # cat함수를 통해 실제값을 누적.

    predictions = predictions.numpy() # 넘파이 배열로 변경.
    actual = actual.numpy() # 넘파이 배열로 변경.
    rmse = np.sqrt(mean_squared_error(predictions, actual)) # sklearn을 이용해 RMSE를 계산.
    
    return rmse,actual,predictions

# train_rmse, train_actualValue, train_predValue = evaluation(trainloader) # 원래는 이렇게 하면 안되지만, 비교를 위해서 train을 넣어서 본다. 
test_rmse, test_actualValue, test_predValue = evaluation(testloader)
# plt.plot(train_predValue,train_actualValue,'.')
# plt.show()
# plt.cla()

# print(f'train rmse:{train_rmse}')


plt.plot(test_predValue,test_actualValue,'.')
plt.show()
plt.cla()

print(f'train rmse:{test_rmse}')

print(Test_Timestep.shape)
print(test_actualValue.shape)
print(test_predValue.shape)

plt.plot(test_actualValue,'b')
plt.plot(test_predValue,'r')
plt.xlim((0,40))
plt.ylim((0,150))
plt.show()
