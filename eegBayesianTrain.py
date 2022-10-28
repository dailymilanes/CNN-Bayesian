# -*- coding: utf-8 -*-
"""
Created on Thu Feb 11 16:59:54 2021

@author: Daily Milanés Hermosilla
"""


import keras.utils
import random as random
import keras.metrics
import pandas as pd
import eegUtils
import model
import scipy.io as sio
from sklearn.model_selection import StratifiedKFold, RepeatedStratifiedKFold
import math
import numpy as np

# Global Variable that contains the path where the dataset is preprocessed
dataDirectory = ''

# Global Variable where the pretrained model weights will be saved 
weightsDirectory = ''

# All functions to train the differents experiments.
# n= Parameter that indicates the repetition of each experiment, 
         # in order to differentiate the files of each weights and results of each repetition in the training process.
# We use Adam as optimizer, and loss=categorical crossentropy

# This is a funcion to realize the training for experiments #2,#3 and #4
# subject: subject identifier (intra-subject training, Experiment #2), if "All" correspond to Experiment #3 or #4
# seed:Seed to initialize the python random number generator to ensure repeatability of the experiment. 
# For experiments #2,#3 and #4 please, use the seed shows in main.py, and this experiments are used only on dataset 2 and 2b
# Divide randomly each datalist, one to train and other to validate the model. This datalist correspond to training session 
# The portion selected to train and to validate depends of the dataset
# If dataset 2a fraction=5/6, if dataset 2b fraction=4/5 
#channels: Number of channels of EEG, for dataset 2a are 22,for dataset 2b are 3
#nb_classes:Number of classes, for dataset 2a are 4, for dataset 2b are 2.
#exclude: If experiment #4, exclude different of 0, for the other experiments exclude must be 0.

 
def oneSubjectTrain(datalist,labelslist, subject, seed, repeat, 
              exclude = 0, cropDistance = 2, cropSize = 1000, dropoutRate = 0.5, 
              fraction = 5/6, channels = 22, nb_classes = 4):
              
   
    droputStr = "%0.2f" % dropoutRate 
    
    cv = StratifiedKFold(n_splits = fraction, random_state=seed)
    pseudoTrialList = range(len(datalist))
    pseudolabelList = np.array(labelslist)
    
    for train_indices, val_indices in cv.split(pseudoTrialList, pseudolabelList):
        
        baseFileName= weightsDirectory+subject+ '_d_' + droputStr + '_c_'+str(cropDistance)+'_seed'+str(seed)+'_exp_'+str(repeat)+'_exclude_'+str(exclude)
        weightFileName=baseFileName +'_weights.hdf5'
                    
        classifier = model.createModel(Samples = cropSize, dropoutRate = dropoutRate,
                                       nb_classes = nb_classes, Chans = channels)
        classifier.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
        callback1 = keras.callbacks.ModelCheckpoint(monitor='val_loss', filepath = weightFileName,
                                                   save_best_only=True,
                                                   save_weights_only=True)
        callback2 = keras.callbacks.EarlyStopping(monitor='val_loss', patience=12)    
        callback3 = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.33, patience=5, verbose=1, min_delta=1e-6) 
        classifier.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
        gen1 = eegUtils.Generator(datalist, labelslist, nb_classes, train_indices, channels, cropDistance, cropSize, int(math.ceil((1125-cropSize) / cropDistance)))
        gen2 = eegUtils.Generator(datalist, labelslist, nb_classes, val_indices, channels, cropDistance, cropSize, int(math.ceil((1125-cropSize) / cropDistance)))
        pasosxepocaT=int((len(train_indices)*int(math.ceil((1125-cropSize)/cropDistance)))/32)
        pasosxepocaV= int((len(val_indices)*int(math.ceil((1125-cropSize)/cropDistance)))/32)
        
        history = classifier.fit(gen1, steps_per_epoch=pasosxepocaT, epochs = 100, verbose = 1, 
                                       validation_data = gen2, validation_steps=pasosxepocaV, callbacks=[callback1, callback2, callback3])
        
        hist_df = pd.DataFrame(history.history) 

    
        file=baseFileName+'.json'
        with open(file,mode='w') as f:
          hist_df.to_json(f)
        f.close()
        break
    
# This function prepares a intra subject training for Experiment #2. The number of repetitions is now 16, each one with a different seed
   
def intraSubjectTrain(subject, dropoutRate=0.5, cropDistance = 50, cropSize = 1000):
     
    
    if subject[0] == 'A':
       channels=22
       fraction=6
       nb_classes=4
       strLabels=['Left','Right', 'Foot', 'Tongue']
    elif subject[0] == 'B':
       channels=3
       fraction=5
       nb_classes=2
       strLabels=['Left','Right']
      
    trainDirectory = dataDirectory + subject + '/Training/'
    datalist, labelslist = eegUtils.load_eeg(trainDirectory,strLabels)
    
    seed=1 
    for j in range(1,17):
       oneSubjectTrain(datalist, labelslist, subject, seed, j,
                   cropDistance = cropDistance, cropSize = cropSize, 
                   dropoutRate = dropoutRate, fraction = fraction, 
                   channels = channels, nb_classes = nb_classes)
       seed=seed+1



# This function prepares a inter-subject training for Experiments #3 and #4. The number of repetitions is 16 
# If the experiment is #3 exclude=0, and all data training for all subjects are load
# If experiment #4, exclude different of 0 and all data training and evaluating for all subjects except exclude subject are load
  
def interSubjectTrain(dropoutRate=0.5, cropDistance = 50, cropSize = 1000,
                      nb_classes = 4,exclude = 0):
      
     if nb_classes==4:
       data='A'
       channels=22
       fraction=6
       strLabels=['Left','Right', 'Foot', 'Tongue']
     elif nb_classes==2:
       data='B'
       channels=3
       fraction=5
       strLabels=['Left','Right']
     start = 1  
     if exclude == 1:
        start = 2
        
     datalist, labelslist = eegUtils.load_eeg(dataDirectory + data+'0'+str(start)+'/Training/', strLabels )
   
     if exclude!=0:
        datalistT, labelslistT = eegUtils.load_eeg(dataDirectory + data+'0'+str(start)+'/Evaluating/', strLabels)
        datalist=datalist + datalistT
        labelslist=labelslist + labelslistT  
        
     for i in range(start + 1, 10):
        if (i == exclude):
            continue
        datalistT, labelslistT = eegUtils.load_eeg(dataDirectory + data+'0'+str(i)+'/Training/', strLabels)
        datalist=datalist + datalistT
        labelslist=labelslist + labelslistT 
        if exclude!=0:
            datalistT, labelslistT = eegUtils.load_eeg(dataDirectory + data+'0'+str(i)+'/Evaluating/', strLabels)
            datalist=datalist + datalistT
            labelslist=labelslist + labelslistT 
    
     seed=1              
     for j in range(1,17):
        oneSubjectTrain(datalist, labelslist, 'All', seed, j, exclude=exclude,
                   cropDistance = cropDistance, cropSize = cropSize, 
                   dropoutRate = dropoutRate, fraction = fraction, 
                   channels = channels, nb_classes = nb_classes)
        seed=seed+1

# This function implemented a KFold with 10 repetitions, dataset 2a and dataset IVa are 9x10 KFold and in dataset 2b is 10x10 Kfold

def eegKFold(trialList, labelList, folds, subject,
             cropDistance = 2, cropSize = 1000, 
             dropoutRate = 0.5, nb_classes = 4, channels=22):
   

    result = []
    n=1
    seed=5
    droputStr = "%0.2f" % dropoutRate 
    if subject[0]=='a':
        len_trial=875
    else:
        len_trial=1000
    cv = RepeatedStratifiedKFold(n_splits = folds, n_repeats = 10, random_state=seed)
    pseudoTrialList = range(len(trialList))
    pseudolabelList = np.array(labelList)
    
    for train_indices, test_indices in cv.split(pseudoTrialList, pseudolabelList):
        classifier = model.createModel(Samples = cropSize, dropoutRate = dropoutRate,
                                      nb_classes = nb_classes, Chans = channels)
        
        baseFileName= weightsDirectory+subject+ '_KFold_d_' + droputStr + '_c_'+str(cropDistance)+'_seed'+str(seed)+'_exp_'+str(n)
        weightFileName=baseFileName +'_weights.hdf5'
        callback1 = keras.callbacks.ModelCheckpoint(monitor='val_loss', filepath = weightsDirectory,
                                                   save_best_only=True,
                                                   save_weights_only=True)
        callback2 = keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=12)    
        callback3 = keras.callbacks.ReduceLROnPlateau(monitor='val_accuracy', factor=0.33, patience=5, verbose=1, min_delta=1e-6) 
        classifier.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
        gen2 = eegUtils.Generator(trialList, labelList, nb_classes, test_indices, channels, cropDistance, cropSize, int(math.ceil((len_trial-cropSize)/cropDistance)))
        gen1 = eegUtils.Generator(trialList, labelList, nb_classes, train_indices, channels, cropDistance, cropSize, int(math.ceil((len_trial-cropSize)/cropDistance)))
        
        pasosxepocaT=int((len(train_indices)*int(math.ceil((len_trial-cropSize)/cropDistance)))/32)
        pasosxepocaV= int((len(test_indices)*int(math.ceil((len_trial-cropSize)/cropDistance)))/32)

        history = classifier.fit_generator(gen1, steps_per_epoch=pasosxepocaT, epochs = 100, verbose = 1, 
                                 validation_data = gen2,validation_steps=pasosxepocaV, callbacks=[callback1, callback2, callback3])
        
        hist_df = pd.DataFrame(history.history) 
        result.append(str(hist_df))
        file = baseFileName+'.json'
        with open(file,mode='w') as f:
           hist_df.to_json(f)
        f.close() 
        n=n+1
    return result    


# This function implemented a NestedKFold on dataset IVa over all dataset (280 trials)
# A 10x9 Nested Kfold was used, 10 in outer loop and 9 in inner loop
 
def eegNestedKFold(trialList, labelList, inner_folds, outer_folds, subject,cropDistance = 2, cropSize = 750, 
                   dropoutRate = 0.5, nb_classes = 2, channels=118):
    n=1
    seed=5
    droputStr = "%0.2f" % dropoutRate 
    len_trial=875
          
    cv = StratifiedKFold(n_splits = 10, random_state=seed)
    cv1= StratifiedKFold(n_splits = 9, random_state=seed)
    pseudoTrialList = range(len(trialList))
    pseudolabelList = np.array(labelList)
    for train_indices, test_indices in cv.split(pseudoTrialList, pseudolabelList):
       for train_indices1,val_indices in cv1.split(range(len(train_indices)), pseudolabelList[train_indices]):
           
           realTrain = []
           for i in train_indices1:
                realTrain.append(train_indices[i])
           realVals = []
           for j in val_indices:
                realVals.append(train_indices[j])
               
           baseFileName= weightsDirectory+subject+ '_Nested_KFold_d_' + droputStr + '_c_'+str(cropDistance)+'_seed'+str(seed)+'_exp_'+str(n)
           weightFileName=baseFileName +'_weights.hdf5'
                    
           classifier = model.createModel(Samples = cropSize, dropoutRate = dropoutRate,
                                       nb_classes = nb_classes, Chans = channels)
           callback1 = keras.callbacks.ModelCheckpoint(monitor='val_accuracy', filepath = weightFileName,
                                                    save_best_only=True,
                                                    save_weights_only=True)
           callback2 = keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=12)    
           callback3 = keras.callbacks.ReduceLROnPlateau(monitor='val_accuracy', factor=0.33, patience=5, verbose=1, min_delta=1e-6) 
           classifier.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
           gen1 = eegUtils.Generator(trialList, labelList, nb_classes, realTrain, channels, cropDistance, cropSize, int(math.ceil((len_trial-cropSize) / cropDistance)))

           gen2 = eegUtils.Generator(trialList, labelList, nb_classes, realVals, channels, cropDistance, cropSize, int(math.ceil((len_trial-cropSize) / cropDistance)))
           gen3 = eegUtils.Generator(trialList, labelList, nb_classes, test_indices, channels, cropDistance, cropSize, int(math.ceil((len_trial-cropSize) / cropDistance)))

           pasosxepocaT=int((len(realTrain)*int(math.ceil((len_trial-cropSize)/cropDistance)))/32)
           pasosxepocaV= int((len(realVals)*int(math.ceil((len_trial-cropSize)/cropDistance)))/32)
           pasosxepocaE= int((len(test_indices)*int(math.ceil((len_trial-cropSize)/cropDistance)))/32)

           history = classifier.fit(gen1, steps_per_epoch=pasosxepocaT, epochs = 100, verbose = 1, 
                                        validation_data = gen2, validation_steps=pasosxepocaV, callbacks=[callback1, callback2, callback3])
        
           hist_df = pd.DataFrame(history.history) 
                
           #evaluate
           classifier.load_weights(weightFileName)
           classifier.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
           result=classifier.evaluate_generator(gen3,pasosxepocaE)     
           result_df=pd.DataFrame(result)
                   
           file = baseFileName+'.json'
           with open(file,mode='w') as f:
                 hist_df.to_json(f)
                 result_df.to_json(f)
                 f.close() 
           n=n+1
    return result 

# This Function prepares the conditions to realize a cross validation (KFlod) for a specific subject
# The function that realizes a KFold is eegKfold and it is calling here
# If dataset 2a or IVa number of Folds= 9, if dataset 2b number of Folds=10
  


def trainKFold(subject, dropoutRate=0.5, optim='adam', cropDistance = 50, cropSize = 1000):
        
    if subject[0] == 'A':
        strLabels=['Left','Right', 'Foot', 'Tongue']
        nb_classes=4
        channels=22
        folds = 9
    elif subject[0]=='B':
        strLabels=['Left','Right']
        nb_classes=2
        channels=3
        folds = 10
    elif subject[0]=='a':
        strLabels=['Right','Foot']
        nb_classes=2
        channels=118
        folds = 9       
    datalist, labelslist = eegUtils.load_eeg(dataDirectory + subject+'/Training/', strLabels)
    return eegKFold(datalist, labelslist, folds,  subject=subject, cropDistance = cropDistance, 
                    cropSize = cropSize, dropoutRate = dropoutRate, nb_classes=nb_classes, channels=channels)   
   

# This Function prepares the conditions to realize a Nested cross-validation (NestedKFlod) for a specific subject in dataset IVa
# The function that realizes a NestedKFold is eegNestedKfold and it is calling here
# In the outer loop, the number of folds are 10, and in the inner loop the number of folds are 9

def trainNestedKFold(subject, dropoutRate=0.9, optim='adam', cropDistance = 50, cropSize = 1000):
        
    nb_classes=2
    channels=118
 
    datalist, labelslist = eegUtils.load_eeg(dataDirectory + subject+'/Training/', ['Right', 'Foot'])
    datalist1, labelslist1=eegUtils.load_eeg(dataDirectory + subject+'/Evaluating/', ['Right', 'Foot'])
    datalist=datalist + datalist1
    labelslist=labelslist + labelslist1
    outer_folds = 10
    inner_folds = 9
    return eegNestedKFold(datalist, labelslist, inner_folds, outer_folds, cropDistance = cropDistance, subject=subject,
                          cropSize = cropSize, nb_classes=nb_classes, channels=channels, dropoutRate = dropoutRate)         


#This function is to perform the experiment #4
#exclude variable is a number that represents the subject whose data will not be used in the training process.

def trainUnkownSubject(dropoutRate=0.5, cropDistance = 50, cropSize = 1000,
                      nb_classes = 4, exclude = 1):
    interSubjectTrain(dropoutRate=dropoutRate, cropDistance = cropDistance, 
                      cropSize = cropSize, nb_classes = nb_classes, exclude = exclude)
    
         