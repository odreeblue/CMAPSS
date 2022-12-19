''' 코드 내용은 아래와 같음 '''

## 1. k-means 군집화 알고리즘 적용
#  1.1 K-means 군집화 알고리즘 적용
#  1.2 논문과 같은 사진(그래프) 도출


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
# from ipywidgets import interact
from time import time

train_FD001 = np.loadtxt("./10_Traindata/train_FD001.txt", dtype = 'float', delimiter=" ") # 사용안함
train_FD002 = np.loadtxt("./10_Traindata/train_FD002.txt", dtype = 'float', delimiter=" ")
train_FD003 = np.loadtxt("./10_Traindata/train_FD003.txt", dtype = 'float', delimiter=" ") # 사용안함
train_FD004 = np.loadtxt("./10_Traindata/train_FD004.txt", dtype = 'float', delimiter=" ") # 사용안함

print(np.shape(train_FD001))
print(np.shape(train_FD002))
print(np.shape(train_FD003))
print(np.shape(train_FD004))
x_columns_ = list(['var1','var2','var3',
                    'var4','var5','var6',
                    'var7','var8','var9',
                    'var10','var11','var12',
                    'var13','var14','var15',
                    'var16','var17','var18',
                    'var19','var20','var21',
                    'var22','var23','var24',
                    'var25','var26'])
y_columns_ = np.array(['regime 1','regime 2',#--> 사용안함, 레퍼런스 따라해본 것임
                    'regime 2','regime 3',
                    'regime 4','regime 2',
                    'regime 2','regime 5',
                    'regime 6','regime 2'])

train_FD002_df = pd.DataFrame(train_FD002,columns=x_columns_)

## 3. k-means 군집화 알고리즘 적용
# K-Means 클러스터링 알고리즘 학습
# 출처 : https://blog.naver.com/PostView.nhn?blogId=wideeyed&logNo=221534602937&categoryNo=49&parentCategoryNo=0&viewDate=&currentPage=1&postListTopCurrentPage=1&from=postView
from sklearn.cluster import KMeans
from sklearn import preprocessing
from sklearn.metrics import silhouette_score


# 사용할 X 컬럼들과 Y컬럼을 지정한다.
#x_columns = ['var3','var4','var5'] # 사용됨 -> Operation Setting 변수임
x_columns = ['var6','var7','var8','var9','var10','var11','var12',
            'var13','var14','var15','var16','var17','var18',
            'var19','var20','var21','var22','var23','var24','var25','var26']
y_columns = 'regime' # 사용 안함

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
    cluster_labels = model.fit_predict(train_FD002_df[x_columns]) # X 컬럼으로 지정된 필드갑으로 피팅
    
    kMeansModels[k] = model
    kMeansModelPreds[k] = cluster_labels

    sumSquaredDistancesList.append(model.inertia_)    # # 샘플과 클러스터 센터간 거리 제곱의 합 저장
    silhouetteScoreList.append(silhouette_score(train_FD002_df[x_columns].values, cluster_labels)) # Silhouette Score저장

    end = time()
    print(k,'time elapsed',end-start)

#plt.plot(ks, sumSquaredDistancesList)
#min_value_index = sumSquaredDistancesList.index(min(sumSquaredDistancesList))
#plt.plot(ks[min_value_index],sumSquaredDistancesList[min_value_index],'rx')
#plt.xlabel('Number of cluster')
#plt.ylabel('Sum of squared distance')
#plt.show()

#print(type(kMeansModelPreds[6])) # type :numpy.ndarray
import os.path
directory = '01_kMeansModelPred'
if os.path.isdir(directory):
    print(directory+'폴더 있음')
else:
    print(directory+'폴더 없음, 생성함')
    os.makedirs(directory)
np.savetxt('./01_kMeansModelPred/results.txt',kMeansModelPreds[6],delimiter='\t',newline='\n')



row = np.shape(train_FD002)[0]
col = np.shape(train_FD002)[1]
print("row : ",row," col : ", col)
regime1 = np.zeros((1,2))
print("regime shape is " , np.shape(regime1))
regime2 = np.zeros((1,2))
regime3 = np.zeros((1,2))
regime4 = np.zeros((1,2))
regime5 = np.zeros((1,2))
regime6 = np.zeros((1,2))
print("regime 생성 진입")


for i in range(0,row):
    if train_FD002[i][0] == 2: # 2번 엔진
        #print('-------------------------------------------------------------------------------------------------')
        #print('kMeansModelPreds[6][i] : ',kMeansModelPreds[6][i], 'train_FD002[i][0] : ',train_FD002[i][0], 'train_FD002[i][1]: ',train_FD002[i][1], 'train_FD002[i][5]: ',train_FD002[i][6])
        #print('kMeansModelPreds[6][i] : ',kMeansModelPreds[6][i], 'train_FD002[i,0] : ',train_FD002[i,0], 'train_FD002[i,1]: ',train_FD002[i,1], 'train_FD002[i,5]: ',train_FD002[i,6])
        #print('-------------------------------------------------------------------------------------------------')
        add = [train_FD002[i][1],train_FD002[i][6]] # timestep, 2번 센서
        add = np.array([add])

        #add = np.concatenate((np.reshape(train_FD002[i,1],(1,1)),np.reshape(train_FD002[i,5:25+1],(1,-1))),axis=1)
        #add = np.concatenate((np.reshape(train_FD002[i,1],(1,1)),np.reshape(train_FD002[i,5:25+1],(1,-1))),axis=1)
    # if i==0:
    #     print("add shape is " , np.shape(add))
    if kMeansModelPreds[6][i] == 0 and train_FD002[i][0] == 2:
        regime1 = np.concatenate((regime1,add),axis=0)
    elif kMeansModelPreds[6][i] == 1 and train_FD002[i][0] == 2:
        regime2 = np.concatenate((regime2,add),axis=0)
    elif kMeansModelPreds[6][i] == 2 and train_FD002[i][0] == 2:
        regime3 = np.concatenate((regime3,add),axis=0)
    elif kMeansModelPreds[6][i] == 3 and train_FD002[i][0] == 2:
        regime4 = np.concatenate((regime4,add),axis=0)
    elif kMeansModelPreds[6][i] == 4 and train_FD002[i][0] == 2:
        regime5 = np.concatenate((regime5,add),axis=0)
    elif kMeansModelPreds[6][i] == 5 and train_FD002[i][0] == 2:
        regime6 = np.concatenate((regime6,add),axis=0)

regime = dict()
regime[1] = regime1[1:,:] # zero 값인 첫번째 행 제외
regime[2] = regime2[1:,:]
regime[3] = regime3[1:,:]
regime[4] = regime4[1:,:]
regime[5] = regime5[1:,:]
regime[6] = regime6[1:,:]

# np.savetxt('Engine2_Regime1_data.txt',regime[1],delimiter='\t',newline='\n')
# np.savetxt('Engine2_Regime2_data.txt',regime[2],delimiter='\t',newline='\n')
# np.savetxt('Engine2_Regime3_data.txt',regime[3],delimiter='\t',newline='\n')
# np.savetxt('Engine2_Regime4_data.txt',regime[4],delimiter='\t',newline='\n')
# np.savetxt('Engine2_Regime5_data.txt',regime[5],delimiter='\t',newline='\n')
# np.savetxt('Engine2_Regime6_data.txt',regime[6],delimiter='\t',newline='\n')

directory = '02_Engine2_Sensor2'
if os.path.isdir(directory):
    print(directory+'폴더 있음')
else:
    print(directory+'폴더 없음, 생성함')
    os.makedirs(directory)

#print(np.shape(regime1))
for i in range(1,6+1):
    plt.cla()
    plt.plot(regime[i][:,0],regime[i][:,1])
    plt.title('regime'+str(i))
    plt.savefig('./02_Engine2_Sensor2/regime'+str(i)+'.png', bbox_inches='tight') #논문엔 1번 센서라고 나왔지만, 2번 센서임
