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

def mobiny method(subject, ):
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
    
    datalist, labelslist = eegBayesianUtils.load_eeg(dataDirectory + subject+'/Training/', ['Left','Right','Foot','Tongue'])
    datalist1, labelslist1 = eegBayesianUtils.load_eeg(dataDirectory + subject+'/Evaluating/', ['Left','Right','Foot','Tongue'])
    threshold=np.arange(0.05,1.05,0.05)
    ind1=0
    ua_mobiny=np.zeros(len(threshold))
    for n in range(1,17):   
       cv = StratifiedKFold(n_splits = folds, random_state=n, shuffle=True)
       pseudoTrialList = range(len(datalist))
       pseudolabelList = np.array(labelslist)
       for train_indices, test_indices in cv.split(pseudoTrialList, pseudolabelList): 
         break  
       labelslist=np.array(labelslist)
       text_label= labelslist[test_indices]
       label=np.repeat(text_label,int(math.ceil((1125-cropSize)/cropDistance)))
       label=label.reshape(len(text_label),int(math.ceil((1125-cropSize)/cropDistance)))   
       tensor_val=eegBayesianEvaluate.Validation(subject, cropDistance, cropSize, model, datadirectory, weightsDirectory, seed=n)
       mean=np.mean(tensor_val, axis=0)  
       y=np.argmax(mean, axis=-1)
       true=1*(y==label)    
       list_true=true.reshape(len(test_indices)*int(math.ceil((1125-cropSize)/cropDistance)))
        
       goods_list=np.where(list_true==True)
       bads_list=np.where(list_true==False)
       goods=np.sum(list_true==True)
       bads=np.sum(list_true==False)
        
       dif=np.zeros(50) 
       desv=np.zeros((len(test_indices),int(math.ceil((1125-cropSize)/cropDistance))))
       mean_dif=np.zeros((len(test_indices),int(math.ceil((1125-cropSize)/cropDistance))))
       test=np.zeros((len(test_indices),int(math.ceil((1125-cropSize)/cropDistance))))
       entropy=st.entropy(mean,base=2,axis=-1)/np.log2(nb_classes)
       for m in threshold:
          ciertos_good=0
          ciertos_bad=0
          inciertos_good=0
          inciertos_bad=0
          for i in range(len(test_indices)):
            for j in range(int(math.ceil((1125-cropSize)/cropDistance))): 
                if entropy[i,j]<m:
                     test[i,j]=True
                     if true[i,j]==True:
                         ciertos_good=ciertos_good + 1
                     else:
                         ciertos_bad=ciertos_bad+1
                else:
                       test[i,j]=False
                       if true[i,j]==True:
                         inciertos_good=inciertos_good + 1
                       else:
                         inciertos_bad=inciertos_bad+1
                   
          ua_mobiny[ind1]=(ciertos_good+inciertos_bad)/(ciertos_good+ciertos_bad+inciertos_good+inciertos_bad)
          ind1=ind1+1
       max=np.argmax(ua_mobiny, axis=-1)

            # Now, we determinate the UA from optimal threshold selected over validation set
       tensor_test=eegBayesianEvaluate.Evaluation(subject, cropDistance, cropSize, model, datadirectory, weightsDirectory, seed=n) 
       labelslist1=np.array(labelslist1)
       label=np.repeat(labelslist1,int(math.ceil((1125-cropSize)/cropDistance)))
       label=label.reshape(len(datalist1),int(math.ceil((1125-cropSize)/cropDistance)))   # label is a matrix 288x63
       mean=np.mean(tensor_test, axis=0)  
       y=np.argmax(mean, axis=-1)
       true=1*(y==label)
       test=np.zeros((len(datalist1),int(math.ceil((1125-cropSize)/cropDistance))))
       entropy=st.entropy(mean,base=2,axis=-1)/np.log2(nb_classes)
       ciertos_good=0
       ciertos_bad=0
       inciertos_good=0
       inciertos_bad=0
       for i in range(len(datalist1)):
          for j in range(int(math.ceil((1125-cropSize)/cropDistance))): 
              if entropy[i,j] < max/len(threshold)+0.05:
                 test[i,j]=True
                 if true[i,j]==True:
                     ciertos_good=ciertos_good + 1
                 else:
                      ciertos_bad=ciertos_bad+1
              else:
                 test[i,j]=False
                 if true[i,j]==True:
                     inciertos_good=inciertos_good + 1
                 else:
                     inciertos_bad=inciertos_bad+1
       
       mobiny_result[ind2,0]=ciertos_good
       mobiny_result[ind2,1]=inciertos_good
       mobiny_result[ind2,2]=ciertos_bad
       mobiny_result[ind2,3]=inciertos_bad
       ind2==ind2+1    
               
   
    
    
