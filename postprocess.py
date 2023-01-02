from ANN import TensorData, Regressor
import torch
from torch import nn, optim                           # torch 에서 제공하는 신경망 기술, 손실함수, 최적화를 할 수 있는 함수들을 불러온다.
import torch.nn.functional as F                       # torch 내의 세부적인 기능을 불러옴.
from torch.utils.data import DataLoader, Dataset      # 데이터를 모델에 사용할 수 있게 정리해주는 라이브러리.
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
class postprocess:
    def __init__(self,data,sensornumber,modelpath):

        self.data = data.copy()
        self.SENSORNUMBER = sensornumber
        self.PATH = modelpath
        self.TrainData = dict()
        self.Train_X = dict()
        self.Train_Y = dict()
        self.TrainSets = dict()
        self.TrainLoaders = dict()

        self.TestData = dict()
        self.Test_X = dict()
        self.Test_Y = dict()
        self.TestSets = dict()
        self.TestLoaders = dict()

        model = dict()

        for i in range(1,7):
            self.TrainData[i] = self.data[(self.data['type']=='train') & (self.data['regime']==i-1)]
            self.Train_X[i] = self.TrainData[i][self.SENSORNUMBER[i]].to_numpy()
            self.Train_Y[1] = self.TrainData[i]['target'].to_numpy().reshape((-1,1))
            self.TrainSets[i] = TensorData(self.Train_X[i], self.Train_Y[i])
            self.TrainLoaders[i] = torch.utils.data.DataLoader(self.TrainSets[i], batch_size=256, shuffle=False)

            self.TestData[i] = self.data[(self.data['type']=='test') & (self.data['regime']==i-1)]
            self.Test_X[i] = self.TestData[i][self.SENSORNUMBER[i]].to_numpy()
            self.Test_Y[1] = self.TestData[i]['target'].to_numpy().reshape((-1,1))
            self.TestSets[i] = TensorData(self.Test_X[i], self.Test_Y[i])
            self.TestLoaders[i] = torch.utils.data.DataLoader(self.TestSets[i], batch_size=256, shuffle=False)

            model[i] = torch.load(self.PATH+'model'+str(i)+'.pt')
    def evaluation(self,modelnumber):

        predictions = torch.tensor([], dtype=torch.float) # 예측값을 저장하는 텐서.
        actual = torch.tensor([], dtype=torch.float) # 실제값을 저장하는 텐서.

        with torch.no_grad():
            self.model[modelnumber].eval() # 평가를 할 땐 반드시 eval()을 사용해야 한다.

            for data in self.dataloader[modelnumber]:
                inputs, values = data
                outputs = self.model[modelnumber](inputs)

                predictions = torch.cat((predictions, outputs), 0) # cat함수를 통해 예측값을 누적.
                actual = torch.cat((actual, values), 0) # cat함수를 통해 실제값을 누적.

        predictions = predictions.numpy() # 넘파이 배열로 변경.
        actual = actual.numpy() # 넘파이 배열로 변경.
        rmse = np.sqrt(mean_squared_error(predictions, actual)) # sklearn을 이용해 RMSE를 계산.
        return rmse,actual,predictions

    def COD(Y,yhat):#coefficient of determination, 결정 계수
        #Y = Y.detach().numpy()
        #yhat = yhat.detach().numpy()
        prediction_mean = np.mean(np.ravel(Y))
        #SSR = 회귀제곱합
        SSR = np.sum((np.ravel(yhat)-prediction_mean)**2)
        #SSE = 오차제곱합
        SSE = np.sum((np.ravel(Y)-np.ravel(yhat))**2)
        #SST = SSR + SSE 
        SST = SSR + SSE
        R_squared = SSR/SST
        return R_squared
    
    def RGraph(model_number,train_yhat,y_train,R,directory):#Regression Graph
        #train_yhat = train_yhat.detach().numpy()
        #y_train = y_train.detach().numpy()
        R_squared = "R = %.3f"%R
        plt.scatter(y_train,train_yhat,s=2,c='black')
        #plt.axis([0,0.5,0,0.5])
        plt.xlabel('Actual value')
        plt.ylabel('Predicted value')
        plt.title(R_squared)
        plt_name = './'+directory+'/regime_'+str(model_number)+'.png'
        plt.savefig(plt_name,dpi=500)
        plt.clf()

