# -*- coding: utf-8 -*-
"""
Created on Thu Feb 11 20:09:07 2021

@author: Daily Milanés Hermosilla
"""

# The authors recomend dropoutRate 0.8 to dataset 2a, dropoutRate 0.5 to dataset 2b and dropout 0.9 to dataset IVa


# To ensure repeatability of the experiment #2, #3 and #4, please use seed=1 up to 16

# To run any experiment, select appropietly subject, seed, dropoutRate, cropDistance=2
# cropSize=1000 to datasets 2a and 2b, cropSize=750 to datset IVa

# nb_classes=4, channels=22, fraction=6 to dataset 2a
# nb_classes=2, channels=3, fraction=5 to dataset 2b
# nb_classes=2, channels=118 to dataset IVa

# Tu run experiment #4, please exclude parameter must be different of 0, and specify unknown subject



import eegBayesianTrain
import eegBayesianEvaluate

global dataDirectory
eegBayesianTrain.dataDirectory = '../Data/'    

global weightsDirectory 
eegBayesianTrain.weightsDirectory = '../Weights/'


## To train Bayesian Neural Network models using subject-specific (SE) strategy, specify subject, dropout level, distance between crops, size of crops, model 'MOPED' or 'NORMAL'

# eegBayesianTrain.intraSubjectTrain('B01', dropoutRate=0.5, cropDistance = 2, cropSize = 1000, model='MOPED')    # for example to train subject B01 of dataset 2b with Moped method

## To train Bayesian Neural Network models using non-subject-specific (NSE) strategy
# eegBayesianTrain.interSubjectTrain(dropoutRate=0.5, cropDistance = 2, cropSize = 1000,
#                       nb_classes = 2,exclude = 0, moped=True)                                        # for example to train with all subjects of dataset 2b, if dataset 2a please replace nd_classes=4

# Evaluate function
# Specified the weightsFileName
#weightsFileName='../Weights/B01_Seed_19_R_1_d_0.50_c_2_x_0_weights.hdf5'
#eegEvaluate.eegEvaluate('B01', cropDistance=2, cropSize=1000, weightsFileName=weightsFileName,
     #                   dropoutRate = 0.5,channels = 3, nb_classes = 2)

