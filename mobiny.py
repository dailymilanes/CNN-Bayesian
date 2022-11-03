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

def mobiny_method(subject, cropDistance=2, cropSize=1000, model='MCD', type_training='SE'):
    if  subject[0] == 'A':
        nb_classes=4
        channels=22
        dropoutRate=0.8
        folds=6  
        strLabels=['Left','Right', 'Foot', 'Tongue']
        dropoutRate=0.8  
                
    else:
        nb_classes=2
        channels=3
        dropoutRate=0.5
        folds=5  
        strLabels=['Left','Right']
        dropoutRate=0.5
    droputStr = "%0.2f" % dropoutRate 
    
    datalist, labelslist = eegBayesianUtils.load_eeg(dataDirectory + subject+'/Training/', strLabels)
    datalist1, labelslist1 = eegBayesianUtils.load_eeg(dataDirectory + subject+'/Evaluating/', strLabels)
    threshold=np.arange(0.05,1.05,0.05)
    ind1=0
    ind2=0
    ua_mobiny=np.zeros(len(threshold))
    mobiny_result=np.zeros([16,4])   # 16 amount of seeds, and 4 groups, cc, ci, ic, ii
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
       tensor_val=eegBayesianEvaluate.Validation(subject, datalist, labelslist, nb_classes, folds=5, cropDistance=2, cropSize=1000, seed=n, model='MCD', type_training='SE')
       mean=np.mean(tensor_val, axis=0)  
       y=np.argmax(mean, axis=-1)
       true=1*(y==label)    
       list_true=true.reshape(len(test_indices)*int(math.ceil((1125-cropSize)/cropDistance)))
        
       test=np.zeros((len(test_indices),int(math.ceil((1125-cropSize)/cropDistance))))
       entropy=st.entropy(mean,base=2,axis=-1)/np.log2(nb_classes)
       for m in threshold:
          certain_good=0
          certain_bad=0
          uncertain_good=0
          uncertain_bad=0
          for i in range(len(test_indices)):
            for j in range(int(math.ceil((1125-cropSize)/cropDistance))): 
                if entropy[i,j]<m:
                     test[i,j]=True
                     if true[i,j]==True:
                         certain_good=certain_good + 1
                     else:
                         certain_bad=certain_bad+1
                else:
                       test[i,j]=False
                       if true[i,j]==True:
                         uncertain_good=uncertain_good + 1
                       else:
                         uncertain_bad=uncertain_bad+1
                   
          ua_mobiny[ind1]=(certain_good+uncertain_bad)/(certain_good+certain_bad+uncertain_good+uncertain_bad)*100
          ind1=ind1+1
       max=np.argmax(ua_mobiny, axis=-1)

            # Now, we determinate the UA from optimal threshold selected over validation set
       tensor_test=eegBayesianEvaluate.evaluation(subject, datalist1, labelslist1, nb_classes, folds=5, cropDistance=2, cropSize=1000, seed=n, model='MCD', type_training='SE') 
       labelslist1=np.array(labelslist1)
       label=np.repeat(labelslist1,int(math.ceil((1125-cropSize)/cropDistance)))
       label=label.reshape(len(datalist1),int(math.ceil((1125-cropSize)/cropDistance)))   # label is a matrix of len(datalist1)x63
       mean=np.mean(tensor_test, axis=0)  
       y=np.argmax(mean, axis=-1)
       true=1*(y==label)
       test=np.zeros((len(datalist1),int(math.ceil((1125-cropSize)/cropDistance))))
       entropy_test=st.entropy(mean,base=2,axis=-1)/np.log2(nb_classes)
       certain_good_test=0
       certain_bad_test=0
       uncertain_good_test=0
       uncertain_bad_test=0
       for i in range(len(datalist1)):
          for j in range(int(math.ceil((1125-cropSize)/cropDistance))): 
              if entropy_test[i,j] < max/len(threshold)+0.05:
                 test[i,j]=True
                 if true[i,j]==True:
                     certain_good_test=certain_good_test + 1
                 else:
                      certain_bad_test=certain_bad_test+1
              else:
                 test[i,j]=False
                 if true[i,j]==True:
                     uncertain_good_test=uncertain_good_test + 1
                 else:
                     uncertain_bad_test=uncertain_bad_test+1
       
       mobiny_result[ind2,0]=certain_good_test
       mobiny_result[ind2,1]=uncertain_good_test
       mobiny_result[ind2,2]=certain_bad_test
       mobiny_result[ind2,3]=uncertain_bad_test
       ind2==ind2+1    
    ua_result=(np.sum(mobiny_result[:,0])+np.sum(mobiny_result[:,3]))/(np.sum(mobiny_result[:,0])+np.sum(mobiny_result[:,1])+np.sum(mobiny_result[:,2])+np.sum(mobiny_result[:,3]))*100   
    return ua_result           
   
    
    
