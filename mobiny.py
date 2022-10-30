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



def load_eeg(dir, strLabels):
    data = []
    labels = []    
    contenido = sorted(os.listdir(dir))
    for fichero in contenido:
        if strLabels[1]=='Foot':
            nombre1='data'
        else:
            nombre1 = fichero[:3] + "trial"
        file_load = sio.loadmat(os.path.join(dir, fichero))
        draw = file_load[nombre1]     
        x = np.array(draw, dtype = np.float32)
        for i in range(0, len(strLabels)):
            strLabel = strLabels[i]
            if strLabel in fichero:
                data.append(np.copy(x))
                labels.append(i)
                break
    return data, labels

   
if __name__=='__main__':
 for i in range(1,10):
    subject='A0'+str(i)
    cropSize=1000
    cropDistance=2
    seeds=[11,13,15,16,20,21,29,30,40,48,65,66,70,72,83,84]
    var=0.1
    threshold=np.arange(0.05,1.05,0.05)
    
    
    if  subject[0] == 'A':
        nb_classes=4
        channels=22
        dropoutRate=0.8
        folds=6       
        dataDirectory = '../../DataBase/'  
        weightsDirectory=dataDirectory +'SubjectSpecific/New_model/Bayesian/Con_prior/'
        
    else:
        nb_classes=2
        channels=3
        dropoutRate=0.5
        folds=5  
        dataDirectory='../../DataBase/Data2b/'
        weightsDirectory=dataDirectory + 'SubjectSpecific/New_model/Bayesian/Con_prior/'  
    droputStr = "%0.2f" % dropoutRate 
    
    
    datalist, labelslist = load_eeg(dataDirectory + subject+'/Training/', ['Left','Right','Foot','Tongue'])
    ind1=0
    ua=np.zeros([len(seeds),len(threshold)])
   
    for i in seeds:  
      cv = StratifiedKFold(n_splits = folds, random_state=i, shuffle=True)
      pseudoTrialList = range(len(datalist))
      pseudolabelList = np.array(labelslist)
              
      for train_indices, test_indices in cv.split(pseudoTrialList, pseudolabelList): 
         break 
      
      labelslist=np.array(labelslist)
      text_label= labelslist[test_indices]
      label=np.repeat(text_label,63)
      label=label.reshape(len(text_label),63)   
      
      
      ind2=0  
      tensor=np.load(weightsDirectory+ 'tensor_SE_validation_set_Bayesian'+subject+'_seed_'+str(i)+'_con_prior_var_0.1_sin_drop_after.npy')        
      mean=np.mean(tensor, axis=0)  
      y=np.argmax(mean, axis=-1)
      true=1*(y==label)
     
      list_true=true.reshape(len(test_indices)*int(math.ceil((1125-cropSize)/cropDistance)))
      goods_list=np.where(list_true==True)
      bads_list=np.where(list_true==False)
      goods=np.sum(list_true==True)
      bads=np.sum(list_true==False)
    
      dif=np.zeros(50) 
        
      desv=np.zeros((len(test_indices),63))
      mean_dif=np.zeros((len(test_indices),63))
           
      test=np.zeros((len(test_indices),63))
      for i in range(len(test_indices)):
            for j in range(int(math.ceil((1125-cropSize)/cropDistance))):
                mean_max=np.argmax(mean[i,j,:])       #indice del maximo de las medias x clase
                mean_entropy=st.entropy(mean[i,j,:],base=2,axis=-1)  # la entropia de la media
                for k in range(50):
                    indexs=np.argsort(-1*tensor[k,i,j,:])    #indices de las salidas ordenadas de mayor a menor valor
                    if indexs[0]==mean_max:
                        dif[k]=tensor[k,i,j,mean_max]-tensor[k,i,j,indexs[1]]
                    else:
                        dif[k]=tensor[k,i,j,mean_max]-tensor[k,i,j,indexs[0]]
                    
                desv[i,j]=np.std(dif)
                mean_dif[i,j]=np.mean(dif)
      
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
                   
        ua[ind1,ind2]=(ciertos_good+inciertos_bad)/(ciertos_good+ciertos_bad+inciertos_good+inciertos_bad)    
        ind2=ind2+1
      ind1=ind1+1  
    
    ua_validation=pd.DataFrame(ua)
    m=np.argmax(ua_validation, axis=-1)    
    for n in seed:
      tensor=np.load(weightsDirectory+ 'tensor_SE_'+subject+'_seed_'+str(n)+'_mc_drop_after.npy')        
      mean=np.mean(tensor, axis=0)  
      y=np.argmax(mean, axis=-1)
      true=1*(y==label)
      test=np.zeros((len(datalist),63))
      entropy=st.entropy(mean,base=2,axis=-1)/np.log2(nb_classes)
            
      ciertos_good=0
      ciertos_bad=0
      inciertos_good=0
      inciertos_bad=0
        
      for i in range(len(datalist)):
            for j in range(int(math.ceil((1125-cropSize)/cropDistance))): 
                if entropy[i,j] < m[a]/20+0.05:
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
                   
      moviny_result[a,0]=ciertos_good
      moviny_result[a,1]=inciertos_good
      moviny_result[a,2]=ciertos_bad
      moviny_result[a,3]=inciertos_bad
      a=a+1    
               
    Tabla=pd.DataFrame(moviny_result, columns=['ciertos_good', 'inciertos_good','ciertos_bad','inciertos_bad'])
    file=weightsDirectory+ subject+'moviny_result.xlsx'
    Tabla.to_excel(file, index=False)
    
    
