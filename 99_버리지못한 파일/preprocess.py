
''' 코드 내용은 아래와 같음 '''
## 1. 논문의 6개의 작동 조건(Operationg Condition) 그래프 표시
## 2. 논문의 Fig.6의 엔진 2번의 센서 1번에 대한(K-means 알고리즘 없이) 그래프
## 3. k-means 군집화 알고리즘 적용(연습까지 수행)


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
# from ipywidgets import interact
from time import time

train_FD001 = np.loadtxt("../10_Traindata/train_FD001.txt", dtype = 'float', delimiter=" ")
train_FD002 = np.loadtxt("../10_Traindata/train_FD002.txt", dtype = 'float', delimiter=" ")
train_FD003 = np.loadtxt("../10_Traindata/train_FD003.txt", dtype = 'float', delimiter=" ")
train_FD004 = np.loadtxt("../10_Traindata/train_FD004.txt", dtype = 'float', delimiter=" ")

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
y_columns_ = np.array(['regime 1','regime 2', 
                    'regime 2','regime 3',
                    'regime 4','regime 2',
                    'regime 2','regime 5',
                    'regime 6','regime 2'])

train_FD002_df = pd.DataFrame(train_FD002,columns=x_columns_)


## 1. 논문의 6개의 작동 조건(Operationg Condition) 그래프 표시
# fig = plt.figure()
# ax = fig.add_subplot(projection='3d')

# ax.scatter(train_FD002[:,2],train_FD002[:,3],train_FD002[:,4],marker='o')
# ax.set_xlabel('VAR 3')
# ax.set_ylabel('VAR 4')
# ax.set_zlabel('VAR 5')
# plt.show()

## 2. 논문의 Fig.6의 엔진 2번의 센서 1번에 대한(K-means 알고리즘 없이) 그래프

# row_min = np.where(train_FD002[:,0]>1)
# row_min = np.min(row_min)
# print(row_min)

# row_max = np.where(train_FD002[:,0]<3)
# row_max = np.max(row_max)
# print(row_max)

# # fig = plt.figure()
# plt.plot(train_FD002[row_min:row_max+1,5])
# plt.show()

## 3. k-means 군집화 알고리즘 적용
# K-Means 클러스터링 알고리즘 학습
# 출처 : https://blog.naver.com/PostView.nhn?blogId=wideeyed&logNo=221534602937&categoryNo=49&parentCategoryNo=0&viewDate=&currentPage=1&postListTopCurrentPage=1&from=postView
from sklearn.cluster import KMeans
from sklearn import preprocessing
from sklearn.metrics import silhouette_score


# 사용할 X 컬럼들과 Y컬럼을 지정한다.
x_columns = ['var3','var4','var5']
y_columns = 'regime'

kMeansModels =dict() #k값 별 모델 저장할 딕셔너리
kMeansModelPreds = dict() # k값별 모델 예측 결과 저장할 딕셔너리
kMeansModelLabelEncoder = dict() # k 값별 라벨인코더 저장할 딕셔너리

sumSquaredDistancesList = list() # 샘플과 클러스터 센터간 거리 제곱의 합 리스트
silhouetteScoreList = list() # Silhouette Coefficient 평균 리스트

#ks = [2,3,4,5,6,7,8,9] # k 값으로부터 2~9까지 테스트한다
ks = [6]
for k in ks:
    start = time()
    model = KMeans(n_clusters=k, random_state=0, n_init=10) # n_init: 초기 중심 위치 시도 횟수
                                                             # random_state : 시드값
    cluster_labels = model.fit_predict(train_FD002_df[x_columns]) # X 컬럼으로 지정된 필드갑으로 피팅
    
    kMeansModels[k] = model
    kMeansModelPreds[k] = cluster_labels

    sumSquaredDistancesList.append(model.inertia_)    # # 샘플과 클러스터 센터간 거리 제곱의 합 저장
    silhouetteScoreList.append(silhouette_score(train_FD002_df[x_columns].values, cluster_labels)) # Silhouette Score저장

    end = time()
    print(k,'time elapsed',end-start)

plt.plot(ks, sumSquaredDistancesList)
min_value_index = sumSquaredDistancesList.index(min(sumSquaredDistancesList))
plt.plot(ks[min_value_index],sumSquaredDistancesList[min_value_index],'rx')
plt.xlabel('Number of cluster')
plt.ylabel('Sum of squared distance')
plt.show()

print(kMeansModelPreds[6])
np.savetxt('results.txt',kMeansModelPreds[6],delimiter='\t',newline='\n')
