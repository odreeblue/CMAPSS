import numpy as np
import os
import matplotlib.pyplot as plt
import math
import seaborn as sn
import scipy.stats as stats
import pandas as pd
from scipy.ndimage import gaussian_filter1d
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
            plt.clf()
        
    def Image_SpearmanValue(self,Spearman_Image_data,directory,type):
        regime = dict()
        for i in range(1,7):
            regime[i] = Spearman_Image_data[(Spearman_Image_data['regime_orig']==i-1)]
            # print(regime[i].shape)

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
                        
                        # print('regime: ',str(regNum),' ',i,'--',j,' 의 값은 ->',str(rho))
                        if math.isnan(rho):
                            rho = 0
                        
                        spearmanValue[regNum][int(i[6:])-1,int(j[6:])-1]=abs(rho)
                        

            spearmanValue_df = dict()
            for i in range(1,7):
                spearmanValue_df[i]= pd.DataFrame(spearmanValue[i], index = [i for i in self.sensorlist], columns = [i for i in self.sensorlist])   
        elif type==1:
            spearmanValue = dict()
            for i in range(1,7):
                spearmanValue[i] = np.zeros((21,1))
            rho = 0
            p_val =0
            for regNum in range(1,7):
                for i in self.sensorlist:
                    rho,p_val = stats.spearmanr(regime[regNum][i],regime[regNum]['target'])

                    if math.isnan(rho):
                        rho = 0
                    spearmanValue[regNum][int(i[6:])-1,0]=abs(rho)

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
            command1 = 'sn.heatmap(spearmanValue_df['+str(i)+'], vmax = 0.4, vmin=0,annot=True,cmap="OrRd")'
            eval(command1)
            plt.title('regime'+str(i))
            plt.savefig('./'+directory+'/spearmanValue'+str(i)+'_'+str(type)+'.png', bbox_inches='tight')
            plt.clf()
        
    def Image_Compare_Orig_vs_Filter(self,compare_data,directory):
        Sensorlist = [2,3,4,9,11,15,17] # 논문상 그룹1번의 Spearman value가 0.4 이상이 센서 리스트
        Directory = directory
        if os.path.isdir(Directory):
            print(Directory+'폴더 있음')
        else:
            print(Directory+'폴더 없음, 생성함')
            os.makedirs(Directory)

        x = compare_data[(compare_data['unit'] == 2) &(compare_data['type']==0) & (compare_data['regime_orig']==0)]['timestep']
        for i in Sensorlist:
            Engine2_Reg1_sensornum = compare_data[(compare_data['unit']==2)&(compare_data['type']==0) & (compare_data['regime_orig']==0)]['sensor'+str(i)]
            Engine2_Reg1_sensornum_filter = gaussian_filter1d(Engine2_Reg1_sensornum,1)
            plt.plot(x,Engine2_Reg1_sensornum,'ok',ms = 2,label='original data') # ms = markersize
            plt.plot(x,Engine2_Reg1_sensornum_filter,'-r',label='filtered data, sigma =1')
            plt.title('sensor #'+str(i))
            plt.legend()
            plt.grid()
            plt.savefig('./'+directory+'/Engine2_Reg1_STD_Gaussian_sensor'+str(i)+'.png',bbox_inches='tight')
            plt.clf()
        