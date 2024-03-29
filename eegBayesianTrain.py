# -*- coding: utf-8 -*-
"""
Created on Thu Feb 11 16:59:54 2021

@author: Daily Milanés Hermosilla
"""


import keras.utils
import random as random
import keras.metrics
import pandas as pd
import eegBayesianUtils
import modelBayesian
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

# This is a funcion to realize the training for experiments #2 and #3 
# subject: subject identifier (intra-subject training, Experiment #2), if "All" correspond to Experiment #3
# seed:Seed to initialize the python random number generator to ensure repeatability of the experiment. 
# For experiments #2 and #3 please, use the seeds show in main.py, and this experiments are used only on dataset 2 and 2b
# Divide randomly each datalist, one to train and other to validate the model. This datalist correspond to training session 
# The portion selected to train and to validate depends of the dataset
# If dataset 2a fraction=5/6, if dataset 2b fraction=4/5 
#channels: Number of channels of EEG, for dataset 2a are 22,for dataset 2b are 3
#nb_classes:Number of classes, for dataset 2a are 4, for dataset 2b are 2.


def trainBayesian(datalist,labelslist, subject, seed, cropDistance = 2, cropSize = 1000, 
                  dropoutRate = 0.8, fraction = 6, channels = 22, nb_classes = 4, model='MOPED', type_training='SE'):
    
        
    droputStr = "%0.2f" % dropoutRate    
    cv = StratifiedKFold(n_splits = fraction, random_state=seed, shuffle=True)
    pseudoTrialList = range(len(datalist))
    pseudolabelList = np.array(labelslist)
    
    for train_indices, val_indices in cv.split(pseudoTrialList, pseudolabelList):
        
        baseFileName= weightsDirectory + subject + '_Bayesian_' + model+'_' + type_training + '_d_' + droputStr + '_c_'+str(cropDistance)+'_seed'+str(i) 
        weightFileName=baseFileName + '_weights.hdf5' 
        count_trial= len(train_indices)
        if model='MOPED':  
          obtainWeights(subject,cropSize=cropSize, dropoutRate=dropoutRate,channels = channels,nb_classes=nb_classes, seed=seed)  
          classifier = modelBayesian.SCNBayesianTL(nb_classes = nb_classes, Chans = channels,Samples = cropSize, 
                                        dropoutRate = dropoutRate,cropDistance=cropDistance, count_trial=count_trial)
          file = baseFileName+'_with_prior_var_0.1_no_drop.json'
        elif model='NORMAL': 
          classifier = modelBayesian.SCNBayesian(nb_classes = nb_classes, Chans = channels,Samples = cropSize, 
                                        dropoutRate = dropoutRate,cropDistance=cropDistance, count_trial=count_trial)
          file = baseFileName+'_no_prior_no_drop.json'
         
        classifier.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
        callback1 = tf.keras.callbacks.ModelCheckpoint(monitor='val_loss', filepath = weightFileName,
                                                   save_best_only=True,
                                                   save_weights_only=True)
        callback2 = tf.keras.callbacks.EarlyStopping(monitor='val_loss',patience=100)
        callback3 = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.33,  mode="auto", verbose=1,
                              patience=25, min_lr=0.000001) 
        
        gen1 = eegBayesianUtils.Generator(datalist, labelslist, nb_classes, train_indices, channels, cropDistance, cropSize, int(math.ceil((1125-cropSize)/cropDistance)))
        gen2 = eegBayesianUtils.Generator(datalist, labelslist, nb_classes, val_indices, channels, cropDistance, cropSize, int(math.ceil((1125-cropSize)/cropDistance)))
        
        stepxepochT=int((len(train_indices)*int(math.ceil((1125-cropSize)/cropDistance)))/32)
        stepxepochV= int((len(val_indices)*int(math.ceil((1125-cropSize)/cropDistance)))/32)
       
        history = classifier.fit(gen1, steps_per_epoch=stepxepochT, epochs = 1000, verbose = 1, 
                                       validation_data = gen2, validation_steps=stepxepochV, callbacks=[callback1, callback2, callback3])
        hist_df = pd.DataFrame(history.history)
        with open(file,mode='w') as f:
             hist_df.to_json(f)
        f.close()
        break 

 
    
# This function prepares a intra subject training for Experiment #2. The number of repetitions is now 16, each one with a different seed
   
def intraSubjectTrain(subject, cropDistance = 50, cropSize = 1000, model='MOPED'):
         
    if subject[0] == 'A':
       channels=22
       fraction=6
       nb_classes=4
       strLabels=['Left','Right', 'Foot', 'Tongue']
       dropoutRate=0.8  
    elif subject[0] == 'B':
       channels=3
       fraction=5
       nb_classes=2
       strLabels=['Left','Right']
       dropoutRate=0.5
      
    trainDirectory = dataDirectory + subject + '/Training/'
    datalist, labelslist = eegBayesianUtils.load_eeg(trainDirectory,strLabels)
    
    for j in range(1,17):
       trainBayesian(datalist, labelslist, subject, seed=j,
                   cropDistance = cropDistance, cropSize = cropSize, 
                   dropoutRate = dropoutRate, fraction = fraction, 
                   channels = channels, nb_classes = nb_classes,model=model, type_training='SE')
       
# This function prepares a inter-subject training for Experiments #3 and #4. The number of repetitions is 16 
# If the experiment is #3 exclude=0, and all data training for all subjects are load
# If experiment #4, exclude different of 0 and all data training and evaluating for all subjects except exclude subject are load
  
def interSubjectTrain(cropDistance = 50, cropSize = 1000, nb_classes = 4, model='MOPED'):
      
     if nb_classes==4:
       data='A'
       channels=22
       fraction=6
       strLabels=['Left','Right', 'Foot', 'Tongue']
       dropoutRate=0.8
     elif nb_classes==2:
       data='B'
       channels=3
       fraction=5
       strLabels=['Left','Right']
       dropoutRate=0.5
     start = 1  
            
     datalist, labelslist = eegBayesianUtils.load_eeg(dataDirectory + data+'0'+str(start)+'/Training/', strLabels)
          
     for i in range(start + 1, 10):
        datalistT, labelslistT = eegBayesianUtils.load_eeg(dataDirectory + data+'0'+str(i)+'/Training/', strLabels)
        datalist=datalist + datalistT
        labelslist=labelslist + labelslistT 
            
     for j in range(1,17):
        trainBayesian(datalist, labelslist, subject='All', seed=j, 
                   cropDistance = cropDistance, cropSize = cropSize, 
                   dropoutRate = dropoutRate, fraction = fraction, 
                   channels = channels, nb_classes = nb_classes, model=model, type_training='NSE')
        

# function to inicializate the prior from deterministic weights
def obtainWeights(subject,cropSize, dropoutRate,channels,nb_classes, seed, variance=0.1):
    global weights_layer1
    global var_layer1
    global weights_layer2
    global var_layer2
    global weights_layer33
        
    dropoutStr = "%0.2f" % dropoutRate

    classifier=modelBayesian.createModel(nb_classes = nb_classes,Chans = channels,Samples = cropSize,dropoutRate=dropoutRate)  
    # Load deterministic weights from the pretained model: please review the weight file name, por example: 'A01_d_0.80_c_2_seed_1_weights.hdf5'  
    classifier.load_weights(weightsDirectory+subject+ '_d_' + droputStr + '_c_'+str(cropDistance)+'_seed'+str(seed)+'_weights.hdf5')
    layer=classifier.get_layer(name='TimeConv')
    weights_layer11=layer.get_weights()
    weights_layer1 = tf.convert_to_tensor(weights_layer11,tf.float32)
    z=np.abs(weights_layer11)
    var_layer1=variance*z
    print(weights_layer1.shape)
     
    layer=classifier.get_layer(name='ChannelConv')
    weights_layer22=layer.get_weights()
    weights_layer2 = tf.convert_to_tensor(weights_layer22,tf.float32)
    z=np.abs(weights_layer22)
    var_layer2=variance*z
   
    layer=classifier.get_layer(name='Dense')
    weights_layer33=layer.get_weights()
