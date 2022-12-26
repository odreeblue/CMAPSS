
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
    def __init__(self,input_dim,hidden1_dim):
        super().__init__() # 모델 연산 정의
        self.fc1 = nn.Linear(input_dim, hidden1_dim, bias=True) # 입력층(13) -> 은닉층1(50)으로 가는 연산
        self.fc2 = nn.Linear(hidden1_dim, hidden1_dim, bias=True) # 은닉층1(50) -> 은닉층2(30)으로 가는 연산
        self.fc3 = nn.Linear(hidden1_dim, 1, bias=True) # 은닉층2(30) -> 출력층(1)으로 가는 연산
        #self.dropout = nn.Dropout(0.2) # 연산이 될 때마다 20%의 비율로 랜덤하게 노드를 없앤다.

    def forward(self, x): # 모델 연산의 순서를 정의
        x = F.relu(self.fc1(x)) # Linear 계산 후 활성화 함수 ReLU를 적용한다.  
        # x = self.dropout(F.relu(self.fc2(x))) # 은닉층2에서 드랍아웃을 적용한다.(즉, 30개의 20%인 6개의 노드가 계산에서 제외된다.)
        x = F.relu(self.fc2(x))
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
TestData = dict()
TestData[1] = df[(df['type']==0)&(df['regime']==0)]
TestData[2] = df[(df['type']==0)&(df['regime']==1)]
TestData[3] = df[(df['type']==0)&(df['regime']==2)]
TestData[4] = df[(df['type']==0)&(df['regime']==3)]
TestData[5] = df[(df['type']==0)&(df['regime']==4)]
TestData[6] = df[(df['type']==0)&(df['regime']==5)]

Test_X = dict()
Test_X[1] = TestData[1][SENSORNUMBER[1]].to_numpy()
Test_X[2] = TestData[2][SENSORNUMBER[2]].to_numpy()
Test_X[3] = TestData[3][SENSORNUMBER[3]].to_numpy()
Test_X[4] = TestData[4][SENSORNUMBER[4]].to_numpy()
Test_X[5] = TestData[5][SENSORNUMBER[5]].to_numpy()
Test_X[6] = TestData[6][SENSORNUMBER[6]].to_numpy()

Test_Y = dict()
Test_Y[1] = TestData[1]['target'].to_numpy().reshape((-1,1))
Test_Y[2] = TestData[2]['target'].to_numpy().reshape((-1,1))
Test_Y[3] = TestData[3]['target'].to_numpy().reshape((-1,1))
Test_Y[4] = TestData[4]['target'].to_numpy().reshape((-1,1))
Test_Y[5] = TestData[5]['target'].to_numpy().reshape((-1,1))
Test_Y[6] = TestData[6]['target'].to_numpy().reshape((-1,1))

# TestData = df[(df['type']==1)&(df['regime']==5)&(df['unit']==51)] # Test Data & Regime = 6 & 51번 엔진 데이터 추출
#TestData = df[(df['type']==0)&(df['unit']==50)&(df['regime']==1)] # Test Data & Regime = 6 & 51번 엔진 데이터 추출
# print(TestData)
# Test_X = TestData.drop(labels=['unit','timestep','set1','set2','set3','type','regime','target'],axis=1).to_numpy()
# Test_Y = TestData['target'].to_numpy().reshape((-1,1))

# Test_Timestep = TestData['timestep'].to_numpy().reshape((-1,1))
# 전체 데이터를 학습 데이터와 평가 데이터로 나눈다.
# 기준으로 잡은 논문이 전체 데이터를 50%, 50%로 나눴기 때문에 test size를 0.5로 설정한다.


#X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1)

# 학습 데이터, 시험 데이터 배치 형태로 구축하기
testsets = dict()
testsets[1] = TensorData(Test_X[1], Test_Y[1])
testsets[2] = TensorData(Test_X[2], Test_Y[2])
testsets[3] = TensorData(Test_X[3], Test_Y[3])
testsets[4] = TensorData(Test_X[4], Test_Y[4])
testsets[5] = TensorData(Test_X[5], Test_Y[5])
testsets[6] = TensorData(Test_X[6], Test_Y[6])

testloader=dict()
testloader[1] = torch.utils.data.DataLoader(testsets[1], batch_size=256, shuffle=False)
testloader[2] = torch.utils.data.DataLoader(testsets[2], batch_size=256, shuffle=False)
testloader[3] = torch.utils.data.DataLoader(testsets[3], batch_size=256, shuffle=False)
testloader[4] = torch.utils.data.DataLoader(testsets[4], batch_size=256, shuffle=False)
testloader[5] = torch.utils.data.DataLoader(testsets[5], batch_size=256, shuffle=False)
testloader[6] = torch.utils.data.DataLoader(testsets[6], batch_size=256, shuffle=False)

model = dict()
model[1] = torch.load('./12_ANN_Model/test1.pt')
model[2] = torch.load('./12_ANN_Model/test2.pt')
model[3] = torch.load('./12_ANN_Model/test3.pt')
model[4] = torch.load('./12_ANN_Model/test4.pt')
model[5] = torch.load('./12_ANN_Model/test5.pt')
model[6] = torch.load('./12_ANN_Model/test6.pt')

def evaluation(dataloader,model):

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
test_rmse= dict()
test_actualValue =dict()
test_predValue = dict()
for i in range(1,7):
    test_rmse[i], test_actualValue[i], test_predValue[i] = evaluation(testloader[i],model[i])
    print(str(i)+' 번째')
    print('test_rmse: '+str(test_rmse[i])+', test_pred의 shape: '+str(test_predValue[i].shape))
print('전:',TestData[1].shape)
#TestData[1] = pd.concat([TestData[1],pd.DataFrame(data=test_predValue[1],columns=['predValue'])],axis=1)
TestData[1].insert(29,'predValue',test_predValue[1])
print('후:',TestData[1].shape)
TestData[2].insert(29,'predValue',test_predValue[2])
TestData[3].insert(29,'predValue',test_predValue[3])
TestData[4].insert(29,'predValue',test_predValue[4])
TestData[5].insert(29,'predValue',test_predValue[5])
TestData[6].insert(29,'predValue',test_predValue[6])

print('--------------------------------------')
print('-------------13번 폴더에 테스트 데이터 예측값 저장--------------')
import os

directory = '13_Testdata_predicted'
if os.path.isdir(directory):
    print(directory+'폴더 있음')
else:
    print(directory+'폴더 없음, 생성함')
    os.makedirs(directory)

for i in range(1,7):
    filepath = './'+directory+'/TestPredictedValue_regime'+str(i)+'.txt'
    np.savetxt(filepath,TestData[i],delimiter='\t',newline='\n',fmt='%1.6e')

directory = '13_50,100,200,260Engine_RUL'
if os.path.isdir(directory):
    print(directory+'폴더 있음')
else:
    print(directory+'폴더 없음, 생성함')
    os.makedirs(directory)

enginelist=[50,100,200,260]
for i in range(1,7):
    for j in enginelist:
        plt.plot(TestData[i][TestData[i]['unit']==j]['timestep'],TestData[i][TestData[i]['unit']==j]['target'],'b')
        plt.plot(TestData[i][TestData[i]['unit']==j]['timestep'],TestData[i][TestData[i]['unit']==j]['predValue'],'r')
        plt.title('regime:'+str(i)+', engine:'+str(j))
        plt.savefig('./'+directory+'/regime'+str(i)+'_engine'+str(j)+'.png')
        plt.cla()
# plt.plot(test_predValue,test_actualValue,'.')
# plt.show()
# plt.cla()

# print(f'train rmse:{test_rmse}')

# print(Test_Timestep.shape)
# print(test_actualValue.shape)
# print(test_predValue.shape)

# plt.plot(test_actualValue,'b')
# plt.plot(test_predValue,'r')
# plt.xlim((0,40))
# plt.ylim((0,150))
# plt.show()
