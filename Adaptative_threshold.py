import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from sklearn.model_selection import train_test_split
import math
import os
from scipy import stats as st
import scipy.io as sio
from sklearn.model_selection import StratifiedKFold
import eegBayesianEvaluate
import eegBayesianUtils

def adaptative_threshold(subject, model, ):
    if  subject[0] == 'A':
        nb_classes=4
        channels=22
        dropoutRate=0.8
        folds=6       
                
    else:
        nb_classes=2
        channels=3
        dropoutRate=0.5
        folds=5  
    droputStr = "%0.2f" % dropoutRate 
    
    datalist1, labelslist1 = eegBayesianUtils.load_eeg(dataDirectory + subject+'/Evaluating/', ['Left','Right','Foot','Tongue'])
    ind=0
    for n in range(1,17):   
       tensor_test=eegBayesianEvaluate.Evaluation(subject, cropDistance, cropSize, model, datalist1, weightsDirectory, seed=n) 
       labelslist1=np.array(labelslist1)
       label=np.repeat(labelslist1,int(math.ceil((1125-cropSize)/cropDistance)))
       label=label.reshape(len(datalist1),int(math.ceil((1125-cropSize)/cropDistance)))   # label is a matrix of len(datalist1)x63
       mean=np.mean(tensor_test, axis=0)  
       y=np.argmax(mean, axis=-1)
       true=1*(y==label)
       test=np.zeros((len(datalist1),int(math.ceil((1125-cropSize)/cropDistance))))
       ciertos_good=0
       ciertos_bad=0
       inciertos_good=0
       inciertos_bad=0
       dif=np.zeros(50) 
       a=np.zeros(50)
       desv=np.zeros((len(datalist),int(math.ceil((1125-cropSize)/cropDistance))))
       mean_dif=np.zeros((len(datalist),int(math.ceil((1125-cropSize)/cropDistance))))
       ztest_99=np.zeros((len(datalist),int(math.ceil((1125-cropSize)/cropDistance))))
       ztest_95=np.zeros((len(datalist),int(math.ceil((1125-cropSize)/cropDistance))))
       value_ztest_95=np.zeros((len(datalist),int(math.ceil((1125-cropSize)/cropDistance))))
       value_ztest_99=np.zeros((len(datalist),int(math.ceil((1125-cropSize)/cropDistance))))
       for i in range(len(datalist)):
            for j in range(int(math.ceil((1125-cropSize)/cropDistance))):
                mean_max=np.argmax(mean[i,j,:])       #maximum index from the average of 50 forward pass per class
                for k in range(50):
                    a[k]=np.argmax(tensor_test[k,i,j])
                    indexs=np.argsort(-1*tensor_test[k,i,j,:])    #index of the outputs sorted from highest to lowest value
                    if indexs[0]==mean_max:
                        dif[k]=tensor_test[k,i,j,mean_max]-tensor_test[k,i,j,indexs[1]]
                    else:
                        dif[k]=tensor_test[k,i,j,mean_max]-tensor_test[k,i,j,indexs[0]]
                                     
                desv[i,j]=np.std(dif)
                mean_dif[i,j]=np.mean(dif)
                
           # New_test with 99% of confidence level (2.326) using margen of confidence as metric of uncertainty
                if mean_dif[i,j]> desv[i,j]*2.326/math.sqrt(50):
                   ztest_99[i,j]=True
                   value_ztest_99[i,j]=mean_dif[i,j]-desv[i,j]*2.326/math.sqrt(50)
               
                else:
                   ztest_99[i,j]=False
                   value_ztest_99[i,j]=mean_dif[i,j]-desv[i,j]*2.326/math.sqrt(50)
               
           # New_test with 95% of confidence level (1.645) using margen of confidence as metric of uncertainty
                if mean_dif[i,j]> desv[i,j]*1.645/math.sqrt(50):
                   ztest_95[i,j]=True
                   value_ztest_95[i,j]=mean_dif[i,j]-desv[i,j]*1.645/math.sqrt(50)
                else:
                   ztest_95[i,j]=False
                   value_ztest_95[i,j]=mean_dif[i,j]-desv[i,j]*1.645/math.sqrt(50)
               
       value_ztest_95_list=value_ztest_95.reshape(len(datalist)*63)
       ztest_95_list=ztest_95.reshape(len(datalist)*63)
       value_ztest_99_list=value_ztest_99.reshape(len(datalist)*63)
       ztest_99_list=ztest_99.reshape(len(datalist)*63)
        
       for i in range(len(list_true)):
            if list_true[i]== True:
                if ztest_95_list[i]== True :
                    ciertos_good_95=ciertos_good_95 + 1
                else:
                    inciertos_good_95=inciertos_good_95 + 1
             
                if ztest_99_list[i]== True :
                    ciertos_good_99=ciertos_good_99 + 1
                else:
                    inciertos_good_99=inciertos_good_99 + 1   
             
            else:
                if ztest_95_list[i]== True :
                    ciertos_bad_95= ciertos_bad_95 + 1            
             
                else:
                    inciertos_bad_95 = inciertos_bad_95 + 1
             
                if ztest_99_list[i]== True :
                    ciertos_bad_99= ciertos_bad_99 + 1            
             
                else:
                    inciertos_bad_99 = inciertos_bad_99 + 1     
        
       resultados_95[ind,0]=ciertos_good_95
       resultados_95[ind,1]=inciertos_good_95
       resultados_95[ind,2]=ciertos_bad_95
       resultados_95[ind,3]=inciertos_bad_95
   
       resultados_99[ind,0]=ciertos_good_99
       resultados_99[ind,1]=inciertos_good_99
       resultados_99[ind,2]=ciertos_bad_99
       resultados_99[ind,3]=inciertos_bad_99
       ind=ind+1
   rc=(np.sum(resultados_95[:,0])+np.sum(resultados_95[:,2]))/(np.sum(resultados_95[:,0])+np.sum(resultados_95[:,1])+np.sum(resultados_95[:,2])+np.sum(resultados_95[:,3]))*100  
   rcc=np.sum(resultados_95[:,0])/(np.sum(resultados_95[:,0])+np.sum(resultados_95[:,2]))*100  
   rcu 
   ua=(np.sum(resultados_95[:,0])+np.sum(resultados_95[:,3]))/(np.sum(resultados_95[:,0])+np.sum(resultados_95[:,1])+np.sum(resultados_95[:,2])+np.sum(resultados_95[:,3]))*100   
   
   return ua_result           
