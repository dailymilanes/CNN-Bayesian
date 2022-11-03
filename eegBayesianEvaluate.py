import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt
import numpy as np
import tensorflow_probability as tfp
import pandas as pd
import seaborn as sns
import time
from sklearn.model_selection import train_test_split
import math
import os
from tensorflow.keras import backend as K
import tensorflow.keras.utils
import scipy.io as sio
import random as random
from tensorflow.keras.layers import Conv2D, Input, Flatten, Dense, BatchNormalization, Dropout, SeparableConv2D
from tensorflow.keras.constraints import max_norm
from tensorflow.keras.models import Model
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
from scipy import stats as st
import scipy.io as sio
from sklearn.model_selection import StratifiedKFold
import types
# Dependency imports
import numpy as np
import tensorflow.compat.v1 as tf1
import tensorflow.compat.v2 as tf
from tensorflow_probability.python import util as tfp_util
from tensorflow_probability.python.distributions import deterministic as deterministic_lib
from tensorflow_probability.python.distributions import independent as independent_lib
from tensorflow_probability.python.distributions import normal as normal_lib
import eegBayesianUtils
import modelBayesian

def accuracy(subject, cropDistance=2, cropSize=1000, model='MCD', type_training='SE'):
        
    if  subject[0] == 'A':
        nb_classes=4
        channels=22
        dropoutRate=0.8
        folds=6
        strLabels=['Left','Right', 'Foot', 'Tongue']
    else:
        nb_classes=2
        channels=3
        dropoutRate=0.5
        folds=5
        strLabels=['Left','Right']
        
    droputStr = "%0.2f" % dropoutRate 
    acc=np.zeros(16)   
    cont=0
    datalist, labelslist = eegBayesianUtils.load_eeg(dataDirectory + subject+'/Evaluating/', strLabels)
             
    for i in range(1,17):
       tensor= Evaluate(subject, datalist, labelslist, nb_classes, folds, cropDistance=cropDistance, cropSize=cropSize, seed=i, model=model, type_training=type_training) 
       mean=np.mean(tensor, axis=0)     # tensor of len(datalist) x number of crops x nb_classes
       y=np.argmax(mean, axis=-1)
       true=(y==label)
       list_true=true.reshape(len(datalist)*int(math.ceil((1125-cropSize)/cropDistance)))
       goods=np.sum(list_true==True)
       bads=np.sum(list_true==False)
       acc[cont]=goods/(goods+bads)
       cont=cont+1  
    accuracy=no.mean(acc)
    return accuracy    
      
def evaluation(subject, datalist, labelslist, nb_classes, folds=5, cropDistance=2, cropSize=1000, seed=1, model='MCD', type_training='SE'):
        
    tensor_after=np.zeros((50,len(datalist),int(math.ceil((1125-cropSize)/cropDistance)),nb_classes))
    cv = StratifiedKFold(n_splits = folds, random_state=seed, shuffle=True)
    pseudoTrialList = range(len(datalist))
    pseudolabelList = np.array(labelslist)
       
    for train_indices, test_indices in cv.split(pseudoTrialList, pseudolabelList): 
       test_data, test_labels = eegBayesianUtils.makeNumpys1(datalist, labelslist, cropDistance, cropSize, nb_classes, test_indices)
       if model=='MCD':
            classifier=modelBayesian.createModel(nb_classes = nb_classes,Chans = channels,Samples = cropSize,dropoutRate=dropoutRate, cropDistance=cropDistance)  
            # Load deterministic weights from the pretained model: please review the weight file name, por example: 'A01_d_0.80_c_2_seed_1_weights.hdf5' 
            baseFileName= weightsDirectory + subject +  type_training +  '_d_' + droputStr + '_c_'+str(cropDistance)+'_seed'+str(i) 
            weightFileName=baseFileName + '_weights.hdf5'
            classifier.load_weights(weightFileName)
            prediction_after =[classifier(test_data, training=True) for _ in range(50)]
                  
       elif model=='MOPED':
            classifier=modelBayesian.SCNBayesianTL(nb_classes = nb_classes,Chans = channels,Samples = cropSize,dropoutRate=dropoutRate, cropDistance=cropDistance)  
            # Load weights from the 'MOPED' model: please review the weight file name, por example: 'A01_Bayesian_MOPED_SE_d_0.80_c_2_seed_1_weights.hdf5'  
            # If non-subject-specific, subject='All'      
            baseFileName= weightsDirectory + subject + '_Bayesian_MOPED_' + type_training + '_d_' + droputStr + '_c_'+str(cropDistance)+'_seed'+str(i) 
            weightFileName=baseFileName + '_weights.hdf5' 
            classifier.load_weights(weightsFileName)
            prediction_after =[classifier.predict(test_data, batch_size=32) for _ in range(50)]
                  
       else:
            classifier=modelBayesian.SCNBayesian(nb_classes = nb_classes,Chans = channels,Samples = cropSize,dropoutRate=dropoutRate, cropDistance=cropDistance)  
             # Load weights from the 'NORMAL' model: please review the weight file name, por example: 'A01_Bayesian_NORMAL_SE_d_0.80_c_2_seed_1_weights.hdf5' 
             # If non-subject-specific, subject='All'
            baseFileName= weightsDirectory + subject + '_Bayesian_NORMAL_' + type_training + '_d_' + droputStr + '_c_'+str(cropDistance)+'_seed'+str(i) 
            weightFileName=baseFileName + '_weights.hdf5' 
            classifier.load_weights(weightsFileName)
            prediction_after =[classifier.predict(test_data, batch_size=32) for _ in range(50)]
            
       prediction_after = np.array(prediction_after)
       prediction_after=prediction_after.reshape((50,len(test_indices),int(math.ceil((1125-cropSize)/cropDistance)),nb_classes))

       for k in range(len(test_indices)):
           j=test_indices[k]
           tensor_after[:,j,:,:]=prediction_after[:,k,:,:]

    np.save(weightsDirectory+'tensor_'+subject+'_Bayesian_'+ model+'_'+ type_training+ '_seed_'+str(i)+'_sin_drop_after.npy', tensor_after, allow_pickle=False)  
    return tensor_after
        
def validation(subject, datalist, labelslist, nb_classes, folds=5, cropDistance=2, cropSize=1000, seed=1, model='MCD', type_training='SE'):
         
    tensor_val_after=np.zeros((50,len(datalist)/folds,int(math.ceil((1125-cropSize)/cropDistance)),nb_classes))
    for i in range(1,17):  
      cv = StratifiedKFold(n_splits = folds, random_state=i, shuffle=True)
      pseudoTrialList = range(len(datalist))
      pseudolabelList = np.array(labelslist)
              
      for train_indices, test_indices in cv.split(pseudoTrialList, pseudolabelList): 
         test_data, test_labels = eegBayesianUtils.makeNumpys1(datalist, labelslist, cropDistance, cropSize, nb_classes, test_indices)
         if model=='MCD':
            classifier=modelBayesian.createModel(nb_classes = nb_classes,Chans = channels,Samples = cropSize,dropoutRate=dropoutRate, cropDistance=cropDistance)  
            # Load deterministic weights from the pretained model: please review the weight file name, por example: 'A01_d_0.80_c_2_seed_1_weights.hdf5' 
            baseFileName= weightsDirectory + subject + '_d_' + droputStr + '_c_'+str(cropDistance)+'_seed'+str(i) 
            weightFileName=baseFileName + '_weights.hdf5'
            classifier.load_weights(weightFileName)
            tensor_mc_after =[classifier(test_data, training=True) for _ in range(50)]
                  
         elif model=='MOPED':
            classifier=modelBayesian.SCNBayesianTL(nb_classes = nb_classes,Chans = channels,Samples = cropSize,dropoutRate=dropoutRate, cropDistance=cropDistance)  
            # Load weights from the 'MOPED' model: please review the weight file name, por example: 'A01_Bayesian_MOPED_SE_d_0.80_c_2_seed_1_weights.hdf5'  
            # If non-subject-specific, subject='All'      
            baseFileName= weightsDirectory + subject + '_Bayesian_MOPED_' + type_training + '_d_' + droputStr + '_c_'+str(cropDistance)+'_seed'+str(i) 
            weightFileName=baseFileName + '_weights.hdf5' 
            classifier.load_weights(weightsFileName)
            tensor_mc_after =[classifier.predict(test_data, batch_size=32) for _ in range(50)]
                  
         else:
            classifier=modelBayesian.SCNBayesian(nb_classes = nb_classes,Chans = channels,Samples = cropSize,dropoutRate=dropoutRate, cropDistance=cropDistance)  
             # Load weights from the 'NORMAL' model: please review the weight file name, por example: 'A01_Bayesian_NORMAL_SE_d_0.80_c_2_seed_1_weights.hdf5' 
             # If non-subject-specific, subject='All'
            baseFileName= weightsDirectory + subject + '_Bayesian_NORMAL_' + type_training + '_d_' + droputStr + '_c_'+str(cropDistance)+'_seed'+str(i) 
            weightFileName=baseFileName + '_weights.hdf5' 
            classifier.load_weights(weightsFileName)
            tensor_mc_after =[classifier.predict(test_data, batch_size=32) for _ in range(50)]
            
         tensor_val_after = np.array(tensor_mc_after)
         tensor_val_after=tensor_val_after.reshape((50,len(test_indices),int(math.ceil((1125-cropSize)/cropDistance)),nb_classes))
         break
         
      np.save(weightsDirectory+'tensor_validation_set_'+subject+'_Bayesian_'+ model+'_'+ type_training+ '_seed_'+str(i)+'_sin_drop_after.npy', tensor_mc_after, allow_pickle=False)  
      return tensor_val_after     
        
        
