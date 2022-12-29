import numpy as np
from sklearn.cluster import KMeans
from sklearn import preprocessing
from sklearn.metrics import silhouette_score
from time import time
import pandas as pd
from sklearn.preprocessing import StandardScaler
from scipy.ndimage import gaussian_filter1d
import os
import matplotlib.pyplot as plt
import math
import seaborn as sn
import scipy.stats as stats
class preprocess:
    def __init__(self,data):
        self.columns = ['sensor1','sensor2','sensor3',
                'sensor4','sensor5','sensor6',
                'sensor7','sensor8','sensor9',
                'sensor10','sensor11','sensor12',
                'sensor13','sensor14','sensor15',
                'sensor16','sensor17','sensor18',
                'sensor19','sensor20','sensor21']
        self.data = data.copy()
        self.U2S2_Image_data = data.copy()
        self.Spearman_Image_data = data.copy()
    def Clustering_Kmeans(self):
        start = time()
        model = KMeans(n_clusters=6, random_state=0, n_init=100) # n_init: 초기 중심 위치 시도 횟수
                                                                 # random_state : 시드값
        cluster_labels = model.fit_predict(self.data[self.columns]) # X 컬럼으로 지정된 필드갑으로 피팅

        #kMeansModels[k] = model
        #kMeansModelPreds[k] = cluster_labels

        #sumSquaredDistancesList.append(model.inertia_)    # # 샘플과 클러스터 센터간 거리 제곱의 합 저장
        #silhouetteScoreList.append(silhouette_score(FD002_df[x_columns].values, cluster_labels)) # Silhouette Score저장
        end = time()
        print('Clustering time elapsed',end-start)
        self.data['regime_new']=cluster_labels
        self.U2S2_Image_data['regime_new']=cluster_labels
        self.Spearman_Image_data['regime_new'] = cluster_labels
        # return cluster_labels

    def Standard_Scaler(self):
        std_scaler = StandardScaler()
        fitted = std_scaler.fit(self.data[self.columns])
        new_data = std_scaler.transform(self.data[self.columns])
        new_data = pd.DataFrame(new_data,columns=self.columns)
        for sensorNum in self.columns:
            self.data[sensorNum] = new_data[sensorNum]
    
    def Filtering_Gaussian(self):
        for sensorNum in self.columns: # sensor 총 개수 만큼 반복
            self.data[sensorNum] = gaussian_filter1d(self.data[sensorNum],1) # 필터링한 데이터로 교체

    def Make_target(self,directory):
        maxUnitNum_train = self.data[(self.data['type']==0)]['unit'].max() # Train : maybe 260
        
        # Unit별 최대 사이클수 구하기
        maxCycle_train = dict()
        for i in range(1,int(maxUnitNum_train)+1):
            maxCycle_train[i] = self.data[(self.data['unit']==i)& (self.data['type']==0)].max()['timestep']

        maxUnitNum_test = self.data[(self.data['type']==1)]['unit'].max() # Test : maybe 259
        
        # Unit별 최대 사이클수 구하기
        FD002_TestRUL = np.loadtxt('./'+directory+'/RUL_FD002.txt',delimiter='\t',dtype='double')
        maxCycle_test = dict()
        for i in range(1,int(maxUnitNum_test)+1):
            maxCycle_test[i] = self.data[(self.data['unit']==i)& (self.data['type']==1)].max()['timestep'] + FD002_TestRUL[i-1]
        
        for i in range(self.data.shape[0]):
            if self.data.loc[i,'type']==0: # train
                self.data.loc[i,'target'] = 130 if maxCycle_train[self.data.loc[i,'unit']]-self.data.loc[i,'timestep'] > 130 else maxCycle_train[self.data.loc[i,'unit']]-self.data.loc[i,'timestep']
            else:
                self.data.loc[i,'target'] = 130 if maxCycle_test[self.data.loc[i,'unit']]-self.data.loc[i,'timestep'] > 130 else maxCycle_test[self.data.loc[i,'unit']]-self.data.loc[i,'timestep']
        
        self.Spearman_Image_data['target'] = self.data['target']


    def Save_data(self,directory):
        Directory = directory
        if os.path.isdir(Directory):
            print(Directory+'폴더 있음')
        else:
            print(Directory+'폴더 없음, 생성함')
            os.makedirs(Directory)

        for i in range(1,7): # regime 1~6까지 반복
            filepath = './'+Directory+'/preprocessed_data.txt' # 경로 지정
            np.savetxt(filepath,self.data,delimiter='\t',newline='\n',fmt='%1.6e') # 저장

class make_image:
    def __init__(self,data):
        self.data = data
        self.sensorlist = list(['sensor1','sensor2','sensor3',
                    'sensor4','sensor5','sensor6',
                    'sensor7','sensor8','sensor9',
                    'sensor10','sensor11','sensor12',
                    'sensor13','sensor14','sensor15',
                    'sensor16','sensor17','sensor18',
                    'sensor19','sensor20','sensor21'])
    def Image_Unit2_Sensor2(self,U2S2_Image_data,directory):
        regime = dict()
        for i in range(1,7):
            regime[i] = U2S2_Image_data[(U2S2_Image_data['unit']==2) & (U2S2_Image_data['regime_orig']==i-1) & (U2S2_Image_data['type']==0)]
            # print(regime[i].shape)
        
        if os.path.isdir(directory):
            print(directory+'폴더 있음')
        else:
            print(directory+'폴더 없음, 생성함')
            os.makedirs(directory)

        for i in range(1,6+1):
            plt.plot(regime[i]['timestep'],regime[i]['sensor2'])
            plt.title('regime'+str(i))
            plt.savefig('./'+directory+'/regime'+str(i)+'.png', bbox_inches='tight') #논문엔 1번 센서라고 나왔지만, 2번 센서임
            plt.cla()

    def Image_SpearmanValue(self,Spearman_Image_data,directory,type):
        regime = dict()
        for i in range(1,7):
            regime[i] = Spearman_Image_data[(Spearman_Image_data['regime_orig']==i-1)]
            print(regime[i].shape)

        if type==0:
            spearmanValue = dict()
            for i in range(1,7):
                spearmanValue[i] = np.zeros((21,21))
            rho = 0
            p_val =0
            for regNum in range(1,7):
                for i in self.sensorlist:
                    for j in self.sensorlist:
                        #rho, p_val = stats.spearmanr(regime[regNum]['sensor'+str(i)],regime[regNum]['sensor'+str(j)])
                        # command1 = 'rho,p_val = stats.spearmanr(regime['+str(regNum)+'][\'sensor'+str(i)+'\'],regime['+str(regNum)+'][\'sensor'+str(j)+'\'])'
                        rho, p_val = stats.spearmanr(regime[regNum][i],regime[regNum][j])
                        # exec(command1)
                        print('regime: ',str(regNum),' ',i,'--',j,' 의 값은 ->',str(rho))
                        if math.isnan(rho):
                            rho = 0
                        #spearmanValue[i-1,j-1] = rho
                        # command2 = "spearmanValue["+str(regNum)+"]["+str(i-1)+","+str(j-1)+"]=abs(rho)"
                        spearmanValue[regNum][int(i[6:])-1,int(j[6:])-1]=abs(rho)
                        # exec(command2)

            spearmanValue_df = dict()
            for i in range(1,7):
                spearmanValue_df[i]= pd.DataFrame(spearmanValue[i], index = [i for i in self.sensorlist], columns = [i for i in self.sensorlist])   
                # if i == 1:
                #     sn.heatmap(spearmanValue_df[i], vmax = 0.4, vmin=0,annot=True,cmap="OrRd")
                #     plt.show()
        elif type==1:
            spearmanValue = dict()
            for i in range(1,7):
                spearmanValue[i] = np.zeros((21,1))
            rho = 0
            p_val =0
            for regNum in range(1,7):
                for i in self.sensorlist:
                    
                    #rho, p_val = stats.spearmanr(regime[regNum]['sensor'+str(i)],regime[regNum]['sensor'+str(j)])
                    # command1 = 'rho,p_val = stats.spearmanr(regime['+str(regNum)+'][\'sensor'+str(i)+'\'],regime['+str(regNum)+'][\'target\'])'
                    rho,p_val = stats.spearmanr(regime[regNum][i],regime[regNum]['target'])
                    # exec(command1)
                    if math.isnan(rho):
                        rho = 0
                    #spearmanValue[i-1,j-1] = rho
                    # command2 = "spearmanValue["+str(regNum)+"]["+str(i-1)+",0]=abs(rho)"
                    spearmanValue[regNum][int(i[6:])-1,0]=abs(rho)
                    # exec(command2)

            spearmanValue_df = dict()
            for i in range(1,7):
                spearmanValue_df[i]= pd.DataFrame(spearmanValue[i], index = [i for i in self.sensorlist], columns = ['target'])   


        if os.path.isdir(directory):
            print(directory+'폴더 있음')
        else:
            print(directory+'폴더 없음, 생성함')
            os.makedirs(directory)


        for i in range(1,7):
            plt.figure(figsize=(15,15)) # 단위 inch
            #sn.heatmap(name, annot=True,cmap="OrRd")
            command1 = 'sn.heatmap(spearmanValue_df['+str(i)+'], vmax = 0.4, vmin=0,annot=True,cmap="OrRd")'
            eval(command1)
            plt.title('regime'+str(i))
            plt.savefig('./'+directory+'/spearmanValue'+str(i)+'_'+str(type)+'.png', bbox_inches='tight')
            plt.cla()