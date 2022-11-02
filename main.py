# -*- coding: utf-8 -*-
"""
Created on Thu Feb 11 20:09:07 2021

@author: Daily Milanés Hermosilla
"""

# The authors recomend dropoutRate 0.8 to dataset 2a, dropoutRate 0.5 to dataset 2b and dropout 0.9 to dataset IVa


# To ensure repeatability of the experiment #2, #3 and #4, please use seed=1 up to 16

# To run any experiment, select appropietly subject, seed (1 to 16), dropoutRate, cropDistance=2, cropSize=1000 

# dropoutRate=0.8, nb_classes=4, channels=22, fraction=6 to dataset 2a
# dropoutRate=0.5, nb_classes=2, channels=3, fraction=5 to dataset 2b

import eegBayesianTrain
import eegBayesianEvaluate

global dataDirectory
eegBayesianTrain.dataDirectory = '../Data/'     # Place where there are the datasets

global weightsDirectory 
eegBayesianTrain.weightsDirectory = '../Weights/'   # Place where the weights will be saved


## To train Bayesian Neural Network models using subject-specific (SE) strategy, specify subject, dropout level, distance between crops, size of crops, model 'MOPED' or 'NORMAL'
# eegBayesianTrain.intraSubjectTrain('B01', dropoutRate=0.5, cropDistance = 2, cropSize = 1000, model='MOPED')    # for example to train subject B01 of dataset 2b with Moped method

## To train Bayesian Neural Network models using non-subject-specific (NSE) strategy, dropout level, distance between crops, size of crops, model 'MOPED' or 'NORMAL'
# eegBayesianTrain.interSubjectTrain(dropoutRate=0.5, cropDistance = 2, cropSize = 1000, nb_classes = 2, moped=True)  # for example to train with all subjects of dataset 2b, if dataset 2a please replace dropoutRate=0.8 and nb_classes=4

## Function to evaluate the different models: MCD, MOPED or NORMAL using 50 forward pass over testing set, and to determinate the accuracy.  
# Specified the weightsFileName
#weightsFileName='../Weights/B01_Seed_19_R_1_d_0.50_c_2_x_0_weights.hdf5'
#eegEvaluate.eegEvaluate('B01', cropDistance=2, cropSize=1000, weightsFileName=weightsFileName,
     #                   dropoutRate = 0.5,channels = 3, nb_classes = 2)

