from preprocess import preprocess
from postprocess import postprocess
from make_image import make_image
from ANN import TensorData,Regressor 
import numpy as np
import pandas as pd
import torch

## 0.데이터 준비 ##
train_FD002 = np.loadtxt("./[00]_Traindata/train_FD002.txt", dtype = 'float', delimiter=" ")
train_FD002 = np.concatenate((train_FD002,np.zeros((train_FD002.shape[0],1)),
                                    np.zeros((train_FD002.shape[0],1)),
                                    np.zeros((train_FD002.shape[0],1)),
                                    np.zeros((train_FD002.shape[0],1))),axis=1)
test_FD002 = np.loadtxt("./[01]_Testdata/test_FD002.txt", dtype = 'float', delimiter=" ")
test_FD002 = np.concatenate((test_FD002,np.ones((test_FD002.shape[0],1)),
                                    np.zeros((test_FD002.shape[0],1)),
                                    np.zeros((test_FD002.shape[0],1)),
                                    np.zeros((test_FD002.shape[0],1))),axis=1)
FD002 = np.concatenate((train_FD002,test_FD002),axis=0)
columns = list(['unit','timestep','set1','set2','set3',
                    'sensor1','sensor2','sensor3',
                    'sensor4','sensor5','sensor6',
                    'sensor7','sensor8','sensor9',
                    'sensor10','sensor11','sensor12',
                    'sensor13','sensor14','sensor15',
                    'sensor16','sensor17','sensor18',
                    'sensor19','sensor20','sensor21',
                    'type','regime_orig','regime_new','target'])
FD002_df = pd.DataFrame(FD002,columns=columns)
for i in range(FD002.shape[0]):
    set1 = FD002_df.loc[i,'set1']
    set2 = FD002_df.loc[i,'set2']
    set3 = FD002_df.loc[i,'set3']
    if (set1>=40 and set1<=43) and (set2>=0.83 and set2 <=0.85) and (set3>=99 and set3 <=101):
        FD002_df.loc[i,'regime_orig'] = 0
    elif (set1>=9 and set1<=11) and (set2>=0.24 and set2 <=0.26) and (set3>=99 and set3 <=101):
        FD002_df.loc[i,'regime_orig'] = 4
    elif (set1>=24 and set1<=26) and (set2>=0.61 and set2 <=0.63) and (set3>=59 and set3 <=61):
        FD002_df.loc[i,'regime_orig'] = 1
    elif (set1>=0 and set1<=1) and (set2>=0 and set2 <= 0.1) and (set3>=99 and set3 <=101):
        FD002_df.loc[i,'regime_orig'] = 2
    elif (set1>=19 and set1<=21) and (set2>=0.69 and set2 <= 0.71) and (set3>=99 and set3 <=101):
        FD002_df.loc[i,'regime_orig'] = 3
    elif (set1>=33 and set1<=36) and (set2>=0.83 and set2 <= 0.85) and (set3>=99 and set3 <=101):
        FD002_df.loc[i,'regime_orig'] = 5
    else:
        FD002_df.loc[i,'regime_orig'] = 99
        print('99 error 발생!!')

# print(FD002_df)
## 1.preprocess 인스턴스 생성 ##
preprocessor = preprocess(FD002_df)

    # 1.1 군집화
preprocessor.Clustering_Kmeans()
    # 1.2 표준화
preprocessor.Standard_Scaler()
    # 1.3 필터링
preprocessor.Filtering_Gaussian()
    # 1.4 타겟 열 만들기
directory = '[01]_Testdata'
preprocessor.Make_target(directory)
    # 1.5 전처리된 데이터 저장
directory = '[02]_Preprocessed_data'
preprocessor.Save_data(directory)


## 2.make_image 인스턴스 생성 하여 image 생성##

makeImage = make_image(preprocessor.data)

    # 2.1 Train 데이터의 Unit2번->Sensor2번 "데이터 1~6번 그룹" image 출력
directory = '[03]_image/[01]_Unit2_Sensor2'
makeImage.Image_Unit2_Sensor2(preprocessor.U2S2_Image_data,directory)
    # 2.2 Train, Test 데이터의 SpearmanValue image 도출
directory = '[03]_image/[02]_SpearmanValue'
makeImage.Image_SpearmanValue(preprocessor.Spearman_Image_data,directory,0)
makeImage.Image_SpearmanValue(preprocessor.Spearman_Image_data,directory,1)
    # 2.3 2번 unit의 1번 그룹(regime)의 SpearmanValue가 0.4이상인 센서들에 대해 원본 데이터와 Filtering 데이터 그래프 비교
directory = '[03]_image/[03]_Compare_Orig_vs_Filter'
makeImage.Image_Compare_Orig_vs_Filter(preprocessor.Spearman_Image_data,directory)


## 3.ANN 학습 ##
    # 3.1 논문 상의 SpearmanValue 0.4 이상의 센서들
sensor = dict()
sensor[1]=[2,3,4,9,11,15,17]
sensor[2]=[2,3,4,9,11,15,17]
sensor[3]=[2,3,4,6,7,8,9,11,12,13,15,16,17,20,21]
sensor[4]=[2,3,4,7,8,9,11,12,13,15,17,20,21]
sensor[5]=[2,3,4,9,11,15,17]
sensor[6]=[2,3,4,6,7,8,9,11,12,13,15,17,20,21]
SENSORNUMBER=dict()
for i in range(1,7):
    SENSORNUMBER[i] = list(['sensor'+str(number) for number in sensor[i]])

    # 3.2 DataLoader를 사용하기 위해 Train Data 준비
TrainData = dict()
Train_X = dict()
Train_Y = dict()
Trainsets = dict()
Trainloader = dict()

for i in range(1,7):
    TrainData[i] = preprocessor.data[(preprocessor.data['type']==0)&(preprocessor.data['regime_orig']==i-1)]
    Train_X[i] = TrainData[i][SENSORNUMBER[i]].to_numpy()
    print("Train_X[i].shape: ",Train_X[i].shape)
    Train_Y[i] = TrainData[i]['target'].to_numpy().reshape((-1,1))
    print("Train_Y[i].shape: ",Train_Y[i].shape)
    Trainsets[i] = TensorData(Train_X[i], Train_Y[i])
    Trainloader[i] = torch.utils.data.DataLoader(Trainsets[i], batch_size=256, shuffle=True)
    # 3.3 Model input 파라미터를 리스트화
model_parameters = [[1,len(sensor[1]),'MSE','Adam',50,Trainloader[1]],  # model별 input 파라미터
                    [2,len(sensor[2]),'MSE','Adam',50,Trainloader[2]],  # input_dim, LossFunction, Optimizer, Epoch, trainloader
                    [3,len(sensor[3]),'MSE','Adam',50,Trainloader[3]],  # 1            2           3           4       5   
                    [4,len(sensor[4]),'MSE','Adam',50,Trainloader[4]],     
                    [5,len(sensor[5]),'MSE','Adam',50,Trainloader[5]],     
                    [6,len(sensor[6]),'MSE','Adam',50,Trainloader[6]]] 
    # 3.4 설정한 파라미터를 모델에 입력하여 인스턴스 생성(그룹별 한개씩, 총 6개 모델)
model = dict()
for parameters in model_parameters:
    model[parameters[0]] = Regressor(parameters[0],parameters[1],parameters[2],parameters[3],parameters[4],parameters[5])

    # 3.5 모델 학습
for i in range(1,7):
    model[i].learning()

    # 3.6 학습된 모델 저장
directory = './[04]_ANN/Model/'
for i in range(1,7):
    model[i].saveModel(i,directory)

    # 3.7 Loss image 저장
directory = './[04]_ANN/Loss_graph/'
for i in range(1,7):
    model[i].saveLossGraph(i,directory)

# 4. Postprocess
directory = './[04]_ANN/Model/'
postprocessor = postprocess(preprocessor.data,SENSORNUMBER,directory)

for i in range(1,7):
    rmse,actual,predictions = postprocessor.evaluation(i)
    postprocessor.[1].insert(29,'predValue',test_predValue[1])