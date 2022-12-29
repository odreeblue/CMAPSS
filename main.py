from preprocess import preprocess, make_image
import numpy as np
import pandas as pd

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

print(FD002_df)
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


## 2.make_image 인스턴스 생성 ##
makeImage = make_image(preprocessor.data)

    # 2.1 Train 데이터의 Unit2번->Sensor2번 "데이터 1~6번 그룹" image 출력
directory = '[03]_image/[01]_Unit2_Sensor2'
makeImage.Image_Unit2_Sensor2(preprocessor.U2S2_Image_data,directory)
    # 2.2 Train, Test 데이터의 SpearmanValue image 도출
directory = '[03]_image/[02]_SpearmanValue'
makeImage.Image_SpearmanValue(preprocessor.Spearman_Image_data,directory,0)
makeImage.Image_SpearmanValue(preprocessor.Spearman_Image_data,directory,1)