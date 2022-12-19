''' 코드 내용은 아래와 같음 '''

## 1. 가우시안 필터 적용 전에 Engine2_Regime1_data.txt 뽑아내기

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
# from ipywidgets import interact
from time import time


train_FD002 = np.loadtxt("train_FD002.txt", dtype = 'float', delimiter=" ")

print(np.shape(train_FD002))

x_columns_ = list(['var1','var2','var3',
                    'var4','var5','var6',
                    'var7','var8','var9',
                    'var10','var11','var12',
                    'var13','var14','var15',
                    'var16','var17','var18',
                    'var19','var20','var21',
                    'var22','var23','var24',
                    'var25','var26'])


train_FD002_df = pd.DataFrame(train_FD002,columns=x_columns_)

## 3. k-means 군집화 알고리즘 적용
# K-Means 클러스터링 알고리즘 학습
# 출처 : https://blog.naver.com/PostView.nhn?blogId=wideeyed&logNo=221534602937&categoryNo=49&parentCategoryNo=0&viewDate=&currentPage=1&postListTopCurrentPage=1&from=postView
from sklearn.cluster import KMeans
from sklearn import preprocessing
from sklearn.metrics import silhouette_score


# 사용할 X 컬럼들과 Y컬럼을 지정한다.
x_columns = ['var3','var4','var5'] # 사용됨 -> Operation Setting 변수임
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
    model = KMeans(n_clusters=k, random_state=0, n_init=10) # n_init: 초기 중심 위치 시도 횟수
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
#np.savetxt('results.txt',kMeansModelPreds[6],delimiter='\t',newline='\n')



row = np.shape(train_FD002)[0]
col = np.shape(train_FD002)[1]
print("row : ",row," col : ", col)
regime1 = np.zeros((1,22))
regime2 = np.zeros((1,22))
regime3 = np.zeros((1,22))
regime4 = np.zeros((1,22))
regime5 = np.zeros((1,22))
regime6 = np.zeros((1,22))
print("regime 생성 진입")


for i in range(0,row):
    if train_FD002[i][0] == 2: # 2번 엔진
        add = np.concatenate((np.reshape(train_FD002[i,1],(1,1)),np.reshape(train_FD002[i,5:25+1],(1,-1))),axis=1)
        if kMeansModelPreds[6][i] == 0:
            regime1 = np.concatenate((regime1,add),axis=0)
        elif kMeansModelPreds[6][i] == 1:
            regime2 = np.concatenate((regime2,add),axis=0)
        elif kMeansModelPreds[6][i] == 2:
            regime3 = np.concatenate((regime3,add),axis=0)
        elif kMeansModelPreds[6][i] == 3:
            regime4 = np.concatenate((regime4,add),axis=0)
        elif kMeansModelPreds[6][i] == 4:
            regime5 = np.concatenate((regime5,add),axis=0)
        elif kMeansModelPreds[6][i] == 5:
            regime6 = np.concatenate((regime6,add),axis=0)

regime = dict()
regime[1] = regime1[1:,:] # zero 값인 첫번째 행 제외
regime[2] = regime2[1:,:]
regime[3] = regime3[1:,:]
regime[4] = regime4[1:,:]
regime[5] = regime5[1:,:]
regime[6] = regime6[1:,:]

np.savetxt('Engine2_Regime1_data.txt',regime[1],delimiter='\t',newline='\n')
np.savetxt('Engine2_Regime2_data.txt',regime[2],delimiter='\t',newline='\n')
np.savetxt('Engine2_Regime3_data.txt',regime[3],delimiter='\t',newline='\n')
np.savetxt('Engine2_Regime4_data.txt',regime[4],delimiter='\t',newline='\n')
np.savetxt('Engine2_Regime5_data.txt',regime[5],delimiter='\t',newline='\n')
np.savetxt('Engine2_Regime6_data.txt',regime[6],delimiter='\t',newline='\n')



# #print(np.shape(regime1))
# for i in range(1,6+1):
#     plt.cla()
#     plt.plot(regime[i][:,0],regime[i][:,1])
#     plt.title('regime'+str(i))
#     plt.savefig('regime'+str(i)+'.png', bbox_inches='tight')

