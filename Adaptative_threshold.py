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
    result_95=np.zeros([16,4])
    result_99=np.zeros([16,4])
    for n in range(1,17):   
       tensor_test=eegBayesianEvaluate.Evaluation(subject, cropDistance, cropSize, model, datalist1, weightsDirectory, seed=n) 
       labelslist1=np.array(labelslist1)
       label=np.repeat(labelslist1,int(math.ceil((1125-cropSize)/cropDistance)))
       label=label.reshape(len(datalist1),int(math.ceil((1125-cropSize)/cropDistance)))   # label is a matrix of len(datalist1)x63
       mean=np.mean(tensor_test, axis=0)  
       y=np.argmax(mean, axis=-1)
       true=1*(y==label)
       test=np.zeros((len(datalist1),int(math.ceil((1125-cropSize)/cropDistance))))
       certain_good=0
       certain_bad=0
       uncertain_good=0
       uncertain_bad=0
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
                    certain_good_95=certain_good_95 + 1
                else:
                    uncertain_good_95=uncertain_good_95 + 1
             
                if ztest_99_list[i]== True :
                    certain_good_99=certain_good_99 + 1
                else:
                    uncertain_good_99=uncertain_good_99 + 1   
             
            else:
                if ztest_95_list[i]== True :
                    certain_bad_95= certain_bad_95 + 1            
             
                else:
                    uncertain_bad_95 = uncertain_bad_95 + 1
             
                if ztest_99_list[i]== True :
                    certain_bad_99= certain_bad_99 + 1            
             
                else:
                    uncertain_bad_99 = uncertain_bad_99 + 1     
        
       result_95[ind,0]=certain_good_95
       result_95[ind,1]=uncertain_good_95
       result_95[ind,2]=certain_bad_95
       result_95[ind,3]=uncertain_bad_95
   
       result_99[ind,0]=certain_good_99
       result_99[ind,1]=uncertain_good_99
       result_99[ind,2]=certain_bad_99
       result_99[ind,3]=uncertain_bad_99
       ind=ind+1
    rc_95=(np.sum(result_95[:,0])+np.sum(result_95[:,2]))/(np.sum(result_95[:,0])+np.sum(result_95[:,1])+np.sum(result_95[:,2])+np.sum(result_95[:,3]))*100  
    rcc_95=np.sum(result_95[:,0])/(np.sum(result_95[:,0])+np.sum(result_95[:,2]))*100  
    rcu_95=np.sum(result_95[:,1])/(np.sum(result_95[:,1])+np.sum(result_95[:,3]))*100     
    ua_95=(np.sum(result_95[:,0])+np.sum(result_95[:,3]))/(np.sum(result_95[:,0])+np.sum(result_95[:,1])+np.sum(result_95[:,2])+np.sum(result_95[:,3]))*100  
    
    rc_99=(np.sum(result_99[:,0])+np.sum(result_99[:,2]))/(np.sum(result_99[:,0])+np.sum(result_99[:,1])+np.sum(result_99[:,2])+np.sum(result_99[:,3]))*100  
    rcc_99=np.sum(result_99[:,0])/(np.sum(result_99[:,0])+np.sum(result_99[:,2]))*100  
    rcu_99=np.sum(result_99[:,1])/(np.sum(result_99[:,1])+np.sum(result_99[:,3]))*100     
    ua_99=(np.sum(result_99[:,0])+np.sum(result_99[:,3]))/(np.sum(result_99[:,0])+np.sum(result_99[:,1])+np.sum(result_99[:,2])+np.sum(result_99[:,3]))*100 
    return rc_95, rcc_95, rcu_95, ua_95           
