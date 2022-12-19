''' 코드 내용은 아래와 같음 '''

## 1. k-means 군집화 알고리즘 적용
#  1.1 K-means 군집화 알고리즘 적용
#  1.2 논문과 같은 사진(그래프) 도출



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
print('--------------------------------------')
print('-------------TRAIN 모양---------------')
train_FD002 = np.loadtxt("./10_Traindata/train_FD002.txt", dtype = 'float', delimiter=" ")
print(train_FD002.shape)
train_FD002 = np.concatenate((train_FD002,np.zeros((train_FD002.shape[0],1)),np.zeros((train_FD002.shape[0],1)),np.zeros((train_FD002.shape[0],1))),axis=1)
print(train_FD002.shape)

print('--------------------------------------')
print('--------------TEST 모양---------------')

test_FD002 = np.loadtxt("./11_Testdata/test_FD002.txt", dtype = 'float', delimiter=" ")
print(test_FD002.shape)
test_FD002 = np.concatenate((test_FD002,np.ones((test_FD002.shape[0],1)),np.zeros((test_FD002.shape[0],1)),np.zeros((test_FD002.shape[0],1))),axis=1)
print(test_FD002.shape)


print('--------------------------------------')
print('--------TRAIN+TEST 모양---------------')
FD002 = np.concatenate((train_FD002,test_FD002),axis=0)
print(np.shape(FD002))

x_columns_ = list(['unit','timestep','set1','set2','set3',
                    'sensor1','sensor2','sensor3',
                    'sensor4','sensor5','sensor6',
                    'sensor7','sensor8','sensor9',
                    'sensor10','sensor11','sensor12',
                    'sensor13','sensor14','sensor15',
                    'sensor16','sensor17','sensor18',
                    'sensor19','sensor20','sensor21',
                    'type','regime','target']) # type -> train : 0, test : 1
print('--------------------------------------')
print('--------FD002(numpy)->FD002Pandas ---------------')
FD002_df = pd.DataFrame(FD002,columns=x_columns_)



print('--------------------------------------')
print('--------K-Means 군집화 알고리즘 적용---------------')
## 3. k-means 군집화 알고리즘 적용
# K-Means 클러스터링 알고리즘 학습
# 출처 : https://blog.naver.com/PostView.nhn?blogId=wideeyed&logNo=221534602937&categoryNo=49&parentCategoryNo=0&viewDate=&currentPage=1&postListTopCurrentPage=1&from=postView
from sklearn.cluster import KMeans
from sklearn import preprocessing
from sklearn.metrics import silhouette_score


# 사용할 X 컬럼들과 Y컬럼을 지정한다.
#x_columns = ['var3','var4','var5'] # 사용됨 -> Operation Setting 변수임
x_columns = ['sensor1','sensor2','sensor3',
                'sensor4','sensor5','sensor6',
                'sensor7','sensor8','sensor9',
                'sensor10','sensor11','sensor12',
                'sensor13','sensor14','sensor15',
                'sensor16','sensor17','sensor18',
                'sensor19','sensor20','sensor21']

kMeansModels =dict() #k값 별 모델 저장할 딕셔너리
kMeansModelPreds = dict() # k값별 모델 예측 결과 저장할 딕셔너리
kMeansModelLabelEncoder = dict() # k 값별 라벨인코더 저장할 딕셔너리
sumSquaredDistancesList = list() # 샘플과 클러스터 센터간 거리 제곱의 합 리스트
silhouetteScoreList = list() # Silhouette Coefficient 평균 리스트

#ks = [2,3,4,5,6,7,8,9] # k 값으로부터 2~9까지 테스트한다

ks = [6]
for k in ks:
    start = time()
    model = KMeans(n_clusters=k, random_state=0, n_init=100) # n_init: 초기 중심 위치 시도 횟수
                                                             # random_state : 시드값
    cluster_labels = model.fit_predict(FD002_df[x_columns]) # X 컬럼으로 지정된 필드갑으로 피팅
    
    kMeansModels[k] = model
    kMeansModelPreds[k] = cluster_labels

    sumSquaredDistancesList.append(model.inertia_)    # # 샘플과 클러스터 센터간 거리 제곱의 합 저장
    silhouetteScoreList.append(silhouette_score(FD002_df[x_columns].values, cluster_labels)) # Silhouette Score저장

    end = time()
    print(k,'time elapsed',end-start)

print('--------------------------------------')
print('--------군집화 결과 저장 및 01번 폴더에 데이터 저장---------------')

FD002_df['regime'] = kMeansModelPreds[6]

#print(type(kMeansModelPreds[6])) # type :numpy.ndarray
import os.path
directory = '01_FD002_Reg_data'
if os.path.isdir(directory):
    print(directory+'폴더 있음')
else:
    print(directory+'폴더 없음, 생성함')
    os.makedirs(directory)
np.savetxt('./01_FD002_Reg_data/FD002_Reg_data.txt',FD002_df,delimiter='\t',newline='\n',fmt='%1.6e')




print('--------------------------------------')
print('-------------논문에 있는 image 생성을 위한 조건식 부여--------------')
# Engine2_Reg6_std_df[Engine2_Reg6_std_df['unit']==2 ]['sensor'+str(i)]
regime = dict()
regime[1] = FD002_df[(FD002_df['unit']==2) & (FD002_df['regime']==0) * (FD002_df['type']==0)]
regime[2] = FD002_df[(FD002_df['unit']==2) & (FD002_df['regime']==1) * (FD002_df['type']==0)]
regime[3] = FD002_df[(FD002_df['unit']==2) & (FD002_df['regime']==2) * (FD002_df['type']==0)]
regime[4] = FD002_df[(FD002_df['unit']==2) & (FD002_df['regime']==3) * (FD002_df['type']==0)]
regime[5] = FD002_df[(FD002_df['unit']==2) & (FD002_df['regime']==4) * (FD002_df['type']==0)]
regime[6] = FD002_df[(FD002_df['unit']==2) & (FD002_df['regime']==5) * (FD002_df['type']==0)]

# print(regime[1].shape)
print('--------------------------------------')
print('-------------2번 센서의 Regime별 image 저장--------------')
import os.path
directory = '02_FD002_Sensor2_img'
if os.path.isdir(directory):
    print(directory+'폴더 있음')
else:
    print(directory+'폴더 없음, 생성함')
    os.makedirs(directory)

#print(np.shape(regime1))
for i in range(1,6+1):
    plt.cla()
    plt.plot(regime[i]['timestep'],regime[i]['sensor2'])
    plt.title('regime'+str(i))
    plt.savefig('./02_FD002_Sensor2_img/regime'+str(i)+'.png', bbox_inches='tight') #논문엔 1번 센서라고 나왔지만, 2번 센서임

print('--------------------------------------')
print('-------------Spearman Value image 저장--------------')
spearmanValue1 = np.zeros((21,21))
spearmanValue2 = np.zeros((21,21))
spearmanValue3 = np.zeros((21,21))
spearmanValue4 = np.zeros((21,21))
spearmanValue5 = np.zeros((21,21))
spearmanValue6 = np.zeros((21,21))

rho = 0
p_val =0

import math

for regNum in range(1,7):
    for i in range(1,22):
        for j in range(1,22):
            #rho, p_val = stats.spearmanr(regime[regNum]['sensor'+str(i)],regime[regNum]['sensor'+str(j)])
            command1 = 'rho,p_val = stats.spearmanr(regime['+str(regNum)+'][\'sensor'+str(i)+'\'],regime['+str(regNum)+'][\'sensor'+str(j)+'\'])'
            exec(command1)
            if math.isnan(rho):
                rho = 0
            #spearmanValue[i-1,j-1] = rho
            command2 = "spearmanValue"+str(regNum)+"["+str(i-1)+","+str(j-1)+"]=abs(rho)"
            exec(command2)
sensorlist = list(['sensor1','sensor2','sensor3',
                    'sensor4','sensor5','sensor6',
                    'sensor7','sensor8','sensor9',
                    'sensor10','sensor11','sensor12',
                    'sensor13','sensor14','sensor15',
                    'sensor16','sensor17','sensor18',
                    'sensor19','sensor20','sensor21'])
spearmanValue1_df = pd.DataFrame(spearmanValue1, index = [i for i in sensorlist], columns = [i for i in sensorlist])
spearmanValue2_df = pd.DataFrame(spearmanValue2, index = [i for i in sensorlist], columns = [i for i in sensorlist])
spearmanValue3_df = pd.DataFrame(spearmanValue3, index = [i for i in sensorlist], columns = [i for i in sensorlist])
spearmanValue4_df = pd.DataFrame(spearmanValue4, index = [i for i in sensorlist], columns = [i for i in sensorlist])
spearmanValue5_df = pd.DataFrame(spearmanValue5, index = [i for i in sensorlist], columns = [i for i in sensorlist])
spearmanValue6_df = pd.DataFrame(spearmanValue6, index = [i for i in sensorlist], columns = [i for i in sensorlist])

directory = '03_FD002_spearmanValue_img'
import os.path
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

